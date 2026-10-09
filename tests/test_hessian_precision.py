"""Mixed precision preserves Hessian state and bounds concurrent LAPACK work."""

from concurrent.futures import ThreadPoolExecutor
from threading import BoundedSemaphore, Lock
from time import sleep
from types import SimpleNamespace

import numpy as np
import pytest
from ase.build import molecule
from ase.calculators.emt import EMT

from sella import Sella, _gpu, _numba, configure_compute
from sella.linalg import ApproximateHessian
from sella import linalg


@pytest.fixture
def configured_backend(monkeypatch):
    for name in (
        "_has_torch",
        "_hessian_eigh_dtype",
        "_hessian_eigh_slots",
        "_hessian_eigh_min_dim",
    ):
        monkeypatch.setattr(_gpu, name, getattr(_gpu, name))
    configure_compute(
        use_gpu=False,
        hessian_eigh_dtype="float32",
        hessian_eigh_max_concurrent=6,
        hessian_eigh_min_dim=200,
    )


@pytest.mark.parametrize("configure", [configure_compute, _gpu.configure_linalg])
def test_partial_configuration_preserves_unspecified_settings(
    configure, configured_backend
):
    slots = _gpu._hessian_eigh_slots
    configure(hessian_eigh_min_dim=250)
    assert _gpu._hessian_eigh_dtype == np.dtype("float32")
    assert _gpu._hessian_eigh_slots is slots
    configure(use_gpu=False)
    assert _gpu._hessian_eigh_dtype == np.dtype("float32")
    assert _gpu._hessian_eigh_min_dim == 250
    assert _gpu._hessian_eigh_slots is slots
    configure(hessian_eigh_dtype="float64")
    assert _gpu._hessian_eigh_min_dim == 250
    assert _gpu._hessian_eigh_slots is slots
    configure(hessian_eigh_max_concurrent=2)
    assert _gpu._hessian_eigh_dtype == np.dtype("float64")
    assert _gpu._hessian_eigh_min_dim == 250
    assert _gpu._hessian_eigh_slots is not slots
    configure(hessian_eigh_max_concurrent=None)
    assert _gpu._hessian_eigh_slots is None
    assert _gpu._hessian_eigh_dtype == np.dtype("float64")
    assert _gpu._hessian_eigh_min_dim == 250


@pytest.mark.parametrize(
    "settings",
    [
        {"use_gpu": True},
        {"hessian_eigh_dtype": "int32"},
        {"hessian_eigh_max_concurrent": 0},
        {"hessian_eigh_min_dim": -1},
    ],
)
def test_invalid_partial_configuration_leaves_policy_unchanged(
    settings, configured_backend
):
    slots = _gpu._hessian_eigh_slots
    with pytest.raises(ValueError):
        configure_compute(**settings)
    assert _gpu._hessian_eigh_dtype == np.dtype("float32")
    assert _gpu._hessian_eigh_slots is slots
    assert _gpu._hessian_eigh_min_dim == 200
    assert not _gpu._has_torch


class TrackingBudget:
    """Fail on nested acquisition rather than hanging a fallback regression."""

    def __init__(self, capacity):
        self.slots = BoundedSemaphore(capacity)
        self.lock = Lock()
        self.acquisitions = 0
        self.active = 0

    def __enter__(self):
        assert self.slots.acquire(timeout=2), "Nested or leaked Hessian budget"
        with self.lock:
            self.acquisitions += 1
            self.active += 1
        return self

    def __exit__(self, *exc):
        with self.lock:
            self.active -= 1
        self.slots.release()


class FakeTensor:
    """Model GPU outputs without requiring Torch or a CUDA device."""

    def __init__(self, array):
        self.array = array
        self.shape = array.shape

    def cpu(self):
        return self

    def numpy(self):
        return self.array


@pytest.mark.parametrize("dimension", [199, 200])
@pytest.mark.parametrize("gpu_failure", [False, True])
def test_approximate_hessian_gpu_and_fallback_share_one_budget(
    dimension, gpu_failure, configured_backend, monkeypatch
):
    monkeypatch.setattr(
        _gpu, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
    )
    configure_compute(
        use_gpu=True, hessian_eigh_dtype="float64", hessian_eigh_max_concurrent=2
    )
    budget = TrackingBudget(2)
    monkeypatch.setattr(_gpu, "_hessian_eigh_slots", budget)
    monkeypatch.setattr(
        ApproximateHessian, "_get_B_gpu", lambda self: FakeTensor(self.asarray())
    )
    active = maximum = cpu_calls = 0
    lock = Lock()

    def solve(matrix):
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        sleep(0.02)
        with lock:
            active -= 1
        return np.ones(len(matrix)), np.eye(len(matrix))

    def gpu_solve(tensor):
        values, vectors = solve(tensor.array)
        if gpu_failure:
            return None, None
        return FakeTensor(values), FakeTensor(vectors)

    def cpu_solve(matrix, A_gpu=None):
        nonlocal cpu_calls
        with lock:
            cpu_calls += 1
        return solve(matrix)

    monkeypatch.setattr(linalg, "gpu_eigh_t", gpu_solve)
    monkeypatch.setattr(_gpu, "gpu_eigh", cpu_solve)
    hessians = [
        ApproximateHessian(dimension, dimension, np.eye(dimension)) for _ in range(8)
    ]
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda hessian: hessian.evals, hessians))
    assert len(results) == 8
    assert cpu_calls == (8 if gpu_failure else 0)
    assert maximum <= 2 if dimension >= 200 else maximum > 2
    assert budget.acquisitions == (8 if dimension >= 200 else 0)
    assert budget.active == 0
    for hessian in hessians:
        np.testing.assert_array_equal(hessian.evecs, np.eye(dimension))
        assert (hessian._evecs_gpu is None) == gpu_failure
    # Reading the cached eigenpairs does not reacquire or rerun the solver.
    assert budget.acquisitions == (8 if dimension >= 200 else 0)


@pytest.mark.parametrize("failure_stage", ["gpu", "cpu"])
def test_failed_gpu_dispatch_releases_shared_budget(
    failure_stage, configured_backend, monkeypatch
):
    configure_compute(hessian_eigh_dtype="float64", hessian_eigh_max_concurrent=1)
    budget = TrackingBudget(1)
    monkeypatch.setattr(_gpu, "_hessian_eigh_slots", budget)
    monkeypatch.setattr(
        ApproximateHessian, "_get_B_gpu", lambda self: FakeTensor(self.asarray())
    )

    def gpu_solve(tensor):
        if failure_stage == "gpu":
            raise ValueError("Unexpected GPU dispatch failure")
        return None, None

    def cpu_solve(matrix, A_gpu=None):
        raise np.linalg.LinAlgError("CPU fallback failed")

    monkeypatch.setattr(linalg, "gpu_eigh_t", gpu_solve)
    monkeypatch.setattr(_gpu, "gpu_eigh", cpu_solve)
    for _ in range(3):
        hessian = ApproximateHessian(200, 200, np.eye(200))
        with pytest.raises(
            ValueError if failure_stage == "gpu" else np.linalg.LinAlgError
        ):
            hessian.evals
    assert budget.acquisitions == 3
    assert budget.active == 0
    assert budget.slots.acquire(blocking=False)
    budget.slots.release()


def test_optimizer_initialization_preserves_shared_policy(configured_backend):
    slots = _gpu._hessian_eigh_slots
    atoms = molecule("H2O")
    atoms.calc = EMT()
    Sella(atoms, order=0, internal=True, logfile=None)
    assert _gpu._hessian_eigh_dtype == np.dtype("float32")
    assert _gpu._hessian_eigh_slots is slots


@pytest.mark.parametrize("dimension", [0, 7, 201])
def test_approximate_hessian_retains_float64_state(dimension, configured_backend):
    rng = np.random.default_rng(19)
    matrix = rng.normal(size=(dimension, dimension))
    matrix += matrix.T.copy()
    hessian = ApproximateHessian(dimension, dimension, matrix)
    assert hessian.asarray().dtype == np.float64
    assert hessian.evals.dtype == hessian.evecs.dtype == np.float64
    np.testing.assert_allclose(
        (hessian.evecs * hessian.evals) @ hessian.evecs.T, matrix, atol=2e-5, rtol=2e-5
    )


@pytest.mark.parametrize("failure", ["lapack", "nonfinite", "disabled"])
def test_failed_float32_solve_retries_float64(failure, configured_backend, monkeypatch):
    original = _numba.eigh
    calls = []

    def eigh(matrix):
        calls.append(matrix.dtype)
        if matrix.dtype == np.float32:
            if failure == "lapack":
                raise np.linalg.LinAlgError("LAPACK failed")
            return np.full(len(matrix), np.nan), np.eye(len(matrix), dtype=np.float32)
        return original(matrix)

    monkeypatch.setattr(_numba, "eigh", eigh)
    if failure == "disabled":
        monkeypatch.setattr(_numba, "ENABLED", False)
    matrix = np.array([[1.0, 0.3], [0.3, 2.0]])
    values, vectors = _gpu.hessian_eigh(matrix)
    np.testing.assert_allclose((vectors * values) @ vectors.T, matrix, atol=1e-12)
    assert values.dtype == vectors.dtype == np.float64
    assert calls == (
        [] if failure == "disabled" else [np.dtype("float32"), np.dtype("float64")]
    )


@pytest.mark.parametrize("dimension", [199, 200])
def test_shared_budget_caps_large_solves_only(
    dimension, configured_backend, monkeypatch
):
    active = maximum = 0
    lock = Lock()

    def eigh(matrix):
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        sleep(0.02)
        with lock:
            active -= 1
        return np.ones(len(matrix), dtype=matrix.dtype), np.eye(
            len(matrix), dtype=matrix.dtype
        )

    monkeypatch.setattr(_numba, "eigh", eigh)
    with ThreadPoolExecutor(max_workers=32) as pool:
        results = list(pool.map(_gpu.hessian_eigh, [np.eye(dimension)] * 32))
    assert len(results) == 32
    assert maximum <= 6 if dimension >= 200 else maximum > 6


def test_failed_solver_releases_budget(configured_backend, monkeypatch):
    def fail(matrix):
        raise np.linalg.LinAlgError("Both precisions failed")

    monkeypatch.setattr(_numba, "eigh", fail)
    for _ in range(8):
        with pytest.raises(np.linalg.LinAlgError):
            _gpu.hessian_eigh(np.eye(200))
    assert _gpu._hessian_eigh_slots.acquire(blocking=False)
    _gpu._hessian_eigh_slots.release()

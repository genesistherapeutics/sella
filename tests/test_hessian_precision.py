"""Mixed precision preserves Hessian state and bounds concurrent LAPACK work."""

from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from time import sleep

import numpy as np
import pytest
from ase.build import molecule
from ase.calculators.emt import EMT

from sella import Sella, _gpu, _numba, configure_compute
from sella.linalg import ApproximateHessian


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
        use_gpu=False, hessian_eigh_dtype="float32", hessian_eigh_max_concurrent=6
    )


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

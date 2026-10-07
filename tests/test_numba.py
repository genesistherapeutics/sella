"""Compare accelerated internals against the retained Sella/JAX implementation."""

import numpy as np
import pytest
from ase import Atoms
from ase.build import molecule

from sella import Internals, _numba
from sella.internal import _BATCHED_COORD_FAMILIES


@pytest.mark.parametrize("name", ["H2O", "CH4", "C2H6", "C6H6", "CO2"])
@pytest.mark.parametrize("inactive", [False, True])
def test_internals_match_reference(
    name: str, inactive: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    atoms = molecule(name)
    rng = np.random.default_rng(42)
    atoms.positions += rng.normal(scale=0.01, size=atoms.positions.shape)
    internals = Internals(atoms)
    internals.find_all_bonds()
    internals.find_all_angles()
    internals.find_all_dihedrals()
    if inactive:
        for key in internals._names:
            internals._active[key][::2] = [False] * len(internals._active[key][::2])
    tangent = rng.normal(size=internals.ndof)
    matrix = rng.normal(size=(internals.ndof, 2))

    def evaluate() -> list[np.ndarray]:
        internals._cache = {}
        hvp = internals.hessian_rdot(tangent)
        return [
            internals.calc(),
            internals.jacobian(),
            internals.guess_hessian(),
            internals.hessian_rdot_mat(tangent, matrix),
            internals.hessian_rdot_mat(tangent, matrix[:, 0]),
            hvp.toarray() if hasattr(hvp, "toarray") else hvp,
        ]

    monkeypatch.setattr(_numba, "ENABLED", False)
    expected = evaluate()
    monkeypatch.setattr(_numba, "ENABLED", True)
    actual = evaluate()
    for result, reference in zip(actual, expected, strict=True):
        np.testing.assert_allclose(result, reference, atol=2e-9, rtol=2e-9)


@pytest.mark.parametrize("periodic", [False, True])
def test_fragment_hessian_matches_reference(
    periodic: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    atoms = molecule("H2O") + molecule("H2O")
    atoms.positions[3:] += [5, 0, 0]
    atoms.cell = [12, 12, 12]
    atoms.pbc = periodic
    internals = Internals(atoms, allow_fragments=True)
    internals.find_all_bonds()
    internals.find_all_angles()
    internals.find_all_dihedrals()
    monkeypatch.setattr(_numba, "ENABLED", False)
    expected = internals.guess_hessian()
    monkeypatch.setattr(_numba, "ENABLED", True)
    np.testing.assert_allclose(
        internals.guess_hessian(), expected, atol=1e-10, rtol=1e-10
    )


@pytest.mark.parametrize("width", [2, 3, 4])
def test_exact_coordinate_derivatives(width: int) -> None:
    rng = np.random.default_rng(7)
    positions = rng.normal(size=(9, width, 3))
    translations = rng.normal(size=(9, width - 1, 3))
    tangent = rng.normal(size=positions.shape)
    family = _BATCHED_COORD_FAMILIES[width - 2]
    np.testing.assert_allclose(
        _numba.gradients(positions, translations),
        family.grad_fn(positions, translations),
        atol=1e-10,
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        _numba.specialized_hvps(positions, translations, tangent),
        family.hvp_fn(positions, translations, tangent),
        atol=1e-9,
        rtol=1e-9,
    )
    assert _numba.gradients.nopython_signatures
    assert _numba.specialized_hvps.nopython_signatures


def test_collinear_derivatives_fall_back() -> None:
    positions = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]])
    assert not _numba.regular(positions, np.zeros((1, 2, 3)))


def test_zero_length_angle_preserves_nan() -> None:
    positions = np.array([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]])
    translations = np.zeros((1, 2, 3))
    assert not _numba.regular(positions, translations)
    assert np.isnan(_numba.values(positions, translations)[0])


@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize(
    "degrees",
    [
        0.0,
        1.0,
        15.0 - 1e-9,
        15.0,
        15.0 + 1e-9,
        90.0,
        165.0 - 1e-9,
        165.0,
        165.0 + 1e-9,
        179.0,
        180.0,
        None,
    ],
)
def test_bad_angle_checks_match_reference(
    degrees: float | None, periodic: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    angle = np.deg2rad(90.0 if degrees is None else degrees)
    atoms = Atoms(
        "H3",
        positions=[
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [np.cos(angle), np.sin(angle), 0.0],
        ],
    )
    if degrees is None:
        atoms.positions[0] = atoms.positions[1]  # A zero-length bond gives NaN.
    atoms.cell = [5.0, 5.0, 5.0]
    atoms.pbc = periodic
    internals = Internals(atoms)
    internals.add_angle((0, 1, 2))
    # Inactive coordinates must still be checked, as in the reference.
    internals._active["angles"][0] = False
    monkeypatch.setattr(_numba, "ENABLED", False)
    expected = internals._bad_angles()
    monkeypatch.setattr(_numba, "ENABLED", True)
    assert internals._bad_angles() == expected


@pytest.mark.parametrize("shape", [(0, 0), (5, 5), (9, 4), (4, 9)])
def test_cpu_linalg_reconstruction(
    shape: tuple[int, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    from sella import _gpu  # noqa: PLC0415 — isolate runtime GPU state

    monkeypatch.setattr(_gpu, "_has_torch", False)
    rng = np.random.default_rng(14)
    matrix = rng.normal(size=shape)
    orthogonal, triangular = _gpu.gpu_qr(matrix)
    np.testing.assert_allclose(orthogonal @ triangular, matrix, atol=1e-12)
    if shape[0] == shape[1]:
        symmetric = matrix + matrix.T
        values, vectors = _gpu.gpu_eigh(symmetric)
        np.testing.assert_allclose(
            (vectors * values) @ vectors.T, symmetric, atol=1e-12
        )

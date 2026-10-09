"""Compare accelerated internals against the retained Sella/JAX implementation."""

import numpy as np
import pytest
from ase import Atoms
from ase.build import molecule

from sella import Internals, _numba
from sella.internal import Angle, _BATCHED_COORD_FAMILIES


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


def test_contracted_hvp_compiles_on_supported_numba(monkeypatch) -> None:
    monkeypatch.setattr(_numba, "ENABLED", True)
    internals = Internals(molecule("H2O"))
    internals.find_all_bonds()
    internals.find_all_angles()
    internals.find_all_dihedrals()
    tangent = np.arange(internals.ndof, dtype=float)
    matrix = np.arange(internals.ndof * 2, dtype=float).reshape(-1, 2)
    expected = internals.hessian_rdot(tangent) @ matrix
    actual = internals.hessian_rdot_mat(tangent, matrix)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
    assert _numba.specialized_contracted_family.nopython_signatures


@pytest.mark.parametrize("width", [2, 3, 4])
@pytest.mark.parametrize("empty", [False, True])
def test_contracted_hvp_gather_matches_reference(width, empty) -> None:
    rng = np.random.default_rng(81)
    positions = rng.normal(size=(7, 3))
    # The two coordinates share atoms; the final matrix has a strided layout.
    indices = np.array([np.arange(width), np.arange(1, width + 1)], dtype=np.int32)
    if empty:
        indices = indices[:0]
    translations = rng.normal(size=(len(indices), width - 1, 3))
    tangent = rng.normal(size=(7, 6))[:, ::2]
    matrix = rng.normal(size=(7, 3, 6))[:, :, ::2]
    family = _BATCHED_COORD_FAMILIES[width - 2]
    local_hvp = family.hvp_fn(positions[indices], translations, tangent[indices])
    expected = np.einsum("ijk,ijkc->ic", local_hvp, matrix[indices])
    actual = _numba.specialized_contracted_family(
        positions, indices, translations, tangent, matrix
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


def test_warmup_compiles_on_supported_numba(monkeypatch) -> None:
    monkeypatch.setattr(_numba, "ENABLED", True)
    _numba.warmup()


@pytest.mark.parametrize("moved", ["real", "dummy"])
def test_hessian_guess_refreshes_dummy_position_cache(moved, monkeypatch) -> None:
    monkeypatch.setattr(_numba, "ENABLED", True)
    internals = Internals(molecule("CO2"))
    internals.find_all_bonds()
    internals.find_all_angles()
    internals.find_all_dihedrals()
    assert internals.ndummies == 1
    original = internals.guess_hessian()
    if moved == "real":
        internals.atoms.positions[1, 2] += 0.1
    else:
        internals.dummies.positions[0, 2] += 0.1
    actual = internals.guess_hessian()
    monkeypatch.setattr(_numba, "ENABLED", False)
    expected = internals.guess_hessian()
    assert not np.allclose(original, expected)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize(
    "positions",
    [
        [
            [-0.0005409026870546452, 0.9996479184944448, 0.026528220332456297],
            [0.0, 0.0, 0.0],
            [-0.25253284869713705, 0.9670139783034056, -0.033333558086855786],
        ],
        [
            [-0.7790332960177346, 0.007900674480262227, -0.6269327739387228],
            [0.0, 0.0, 0.0],
            [0.7945056431773062, -0.25694631551668373, 0.5502175696048027],
        ],
    ],
    ids=["lower-cutoff", "upper-cutoff"],
)
def test_new_angle_topology_passes_its_own_singularity_check(positions, monkeypatch):
    signatures = []
    for enabled in (False, True):
        monkeypatch.setattr(_numba, "ENABLED", enabled)
        internals = Internals(Atoms("H3", positions=positions))
        internals.add_bond((0, 1))
        internals.add_bond((1, 2))
        internals.find_all_angles()
        assert internals.check_for_bad_internals() is None
        signatures.append(
            tuple(
                (
                    name,
                    tuple(
                        internals._internal_key(coord)
                        for coord in internals.internals[name]
                    ),
                )
                for name in internals._names
            )
        )
    assert signatures[0] == signatures[1]


@pytest.mark.parametrize("tolerance", [15.0, 30.0])
@pytest.mark.parametrize("upper", [False, True])
def test_dummy_angle_insertion_replays_ambiguous_cutoffs(tolerance, upper, monkeypatch):
    degrees = 180.0 - tolerance if upper else tolerance
    positions = [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [np.cos(np.deg2rad(degrees)), np.sin(np.deg2rad(degrees)), 0.0],
    ]
    atoms = Atoms("H2X", positions=positions)
    internals = Internals(atoms[:2], atol=tolerance)
    angle = Angle((0, 1, 2))
    # Force the fast value across the strict boundary; the dummy insertion
    # predicate must still use the JAX value, as does ordinary construction.
    translations = np.zeros((2, 3))
    reference = float(angle._eval0(atoms.positions, translations))
    boundary = np.pi - internals.atol if upper else internals.atol
    shifted = np.nextafter(boundary, -np.inf if upper else np.inf)
    monkeypatch.setattr(_numba, "ENABLED", True)
    monkeypatch.setattr(angle.__class__, "calc", lambda self, atoms: shifted)
    assert internals._angle_in_range(angle, atoms) == (
        internals.atol < reference < np.pi - internals.atol
    )


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

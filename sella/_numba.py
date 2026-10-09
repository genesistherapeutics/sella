"""FP64 coordinate kernels with compact exact directional derivatives; no fastmath."""

import os
import numpy as np
from numba import njit
from . import _generated_hvp as generated_hvp
from . import _generated_gradients


ANGLE_BOUNDARY_EPS = 1e-12


@njit(nogil=True, cache=True, error_model="numpy")
def values(positions: np.ndarray, translations: np.ndarray) -> np.ndarray:
    out = np.empty(len(positions))
    for i in range(len(out)):
        ax = positions[i, 1, 0] - positions[i, 0, 0] + translations[i, 0, 0]
        ay = positions[i, 1, 1] - positions[i, 0, 1] + translations[i, 0, 1]
        az = positions[i, 1, 2] - positions[i, 0, 2] + translations[i, 0, 2]
        aa = ax * ax + ay * ay + az * az
        if positions.shape[1] == 2:
            out[i] = np.sqrt(aa)
            continue
        bx = positions[i, 2, 0] - positions[i, 1, 0] + translations[i, 1, 0]
        by = positions[i, 2, 1] - positions[i, 1, 1] + translations[i, 1, 1]
        bz = positions[i, 2, 2] - positions[i, 1, 2] + translations[i, 1, 2]
        bb = bx * bx + by * by + bz * bz
        if positions.shape[1] == 3:
            cosine = -(ax * bx + ay * by + az * bz) / np.sqrt(aa * bb)
            out[i] = np.arccos(np.minimum(1.0, np.maximum(-1.0, cosine)))
            continue
        cx = positions[i, 3, 0] - positions[i, 2, 0] + translations[i, 2, 0]
        cy = positions[i, 3, 1] - positions[i, 2, 1] + translations[i, 2, 1]
        cz = positions[i, 3, 2] - positions[i, 2, 2] + translations[i, 2, 2]
        nx, ny, nz = ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx
        mx, my, mz = by * cz - bz * cy, bz * cx - bx * cz, bx * cy - by * cx
        numer = (
            bx * (ny * mz - nz * my)
            + by * (nz * mx - nx * mz)
            + bz * (nx * my - ny * mx)
        )
        denom = np.sqrt(bb) * (nx * mx + ny * my + nz * mz)
        out[i] = np.arctan2(numer, denom)
    return out


@njit(nogil=True, cache=True, error_model="numpy")
def bad_angle_mask(
    positions: np.ndarray, translations: np.ndarray, tolerance: float
) -> tuple[np.ndarray, bool]:
    """Check angle limits, flagging ambiguous strict boundaries for JAX replay."""
    angles = values(positions, translations)
    mask = np.empty(len(angles), dtype=np.bool_)
    ambiguous = False
    upper = np.pi - tolerance
    for i in range(len(angles)):
        angle = angles[i]
        mask[i] = not (tolerance < angle and angle < upper)
        if (
            abs(angle - tolerance) < ANGLE_BOUNDARY_EPS
            or abs(angle - upper) < ANGLE_BOUNDARY_EPS
        ):
            ambiguous = True
    return mask, ambiguous


@njit(nogil=True, cache=True, error_model="numpy")
def guess_hessian_diagonal(
    positions: np.ndarray,
    numbers: np.ndarray,
    bonds: np.ndarray,
    angles: np.ndarray,
    dihedrals: np.ndarray,
    radii: np.ndarray,
    bohr: float,
    hartree: float,
    natoms: int,
) -> np.ndarray:
    counts = np.zeros(len(positions), dtype=np.int64)
    for bond in bonds:
        counts[bond[0]] += 1
        counts[bond[1]] += 1
    out = np.empty(len(bonds) + len(angles) + len(dihedrals))
    offset = 0
    for bond in bonds:
        distance = np.linalg.norm(positions[bond[1]] - positions[bond[0]])
        rcov = radii[numbers[bond[0]]] + radii[numbers[bond[1]]]
        out[offset] = (
            0.3601 * np.exp(-1.944 * (distance - rcov) / bohr) * hartree / bohr**2
        )
        offset += 1
    for angle in angles:
        rab = np.linalg.norm(positions[angle[1]] - positions[angle[0]])
        rbc = np.linalg.norm(positions[angle[2]] - positions[angle[1]])
        rcovab = radii[numbers[angle[0]]] + radii[numbers[angle[1]]]
        rcovbc = radii[numbers[angle[1]]] + radii[numbers[angle[2]]]
        out[offset] = (
            0.089
            + 0.11
            * np.exp(-0.44 * (rab + rbc - rcovab - rcovbc) / bohr)
            / (rcovab * rcovbc / bohr**2) ** (-0.42)
        ) * hartree
        offset += 1
    for dihedral in dihedrals:
        if np.any(dihedral >= natoms):
            out[offset] = 0.5 * hartree
            offset += 1
            continue
        left, right = dihedral[1], dihedral[2]
        distance = np.linalg.norm(positions[right] - positions[left])
        rcov = radii[numbers[left]] + radii[numbers[right]]
        count = counts[left] + counts[right] - 2
        out[offset] = (
            0.0015
            + 14.0
            * count**0.57
            * np.exp(-2.85 * (distance - rcov) / bohr)
            / (distance * rcov / bohr**2) ** 4
        ) * hartree
        offset += 1
    return out


@njit(nogil=True, cache=True, error_model="numpy")
def eigh(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.linalg.eigh(matrix)


@njit(nogil=True, cache=True, error_model="numpy")
def qr(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.linalg.qr(matrix)


# Generated formulas specialize the scalar arithmetic and remove dual-array allocations.


@njit(nogil=True, cache=True, error_model="numpy")
def specialized_hvps(
    positions: np.ndarray, translations: np.ndarray, tangent: np.ndarray
) -> np.ndarray:
    if positions.shape[1] == 2:
        return generated_hvp.bond_batch(positions, translations, tangent)
    if positions.shape[1] == 3:
        return generated_hvp.angle_batch(positions, translations, tangent)
    return generated_hvp.dihedral_batch(positions, translations, tangent)


@njit(nogil=True, cache=True, error_model="numpy")
def specialized_contracted_family(
    positions: np.ndarray,
    indices: np.ndarray,
    translations: np.ndarray,
    tangent: np.ndarray,
    matrix: np.ndarray,
) -> np.ndarray:
    # Numba 0.61 cannot gather with a two-dimensional integer index array.
    # Scalar indexing preserves the same inputs on the supported version floor.
    local_positions = np.empty(
        (len(indices), indices.shape[1], 3), dtype=positions.dtype
    )
    local_tangent = np.empty((len(indices), indices.shape[1], 3), dtype=tangent.dtype)
    for i in range(len(indices)):
        for atom in range(indices.shape[1]):
            for axis in range(3):
                local_positions[i, atom, axis] = positions[indices[i, atom], axis]
                local_tangent[i, atom, axis] = tangent[indices[i, atom], axis]
    local_hvp = specialized_hvps(local_positions, translations, local_tangent)
    out = np.zeros((len(indices), matrix.shape[2]))
    for i in range(len(indices)):
        for atom in range(indices.shape[1]):
            for axis in range(3):
                for column in range(matrix.shape[2]):
                    out[i, column] += (
                        local_hvp[i, atom, axis]
                        * matrix[indices[i, atom], axis, column]
                    )
    return out


ENABLED = os.environ.get("SELLA_DISABLE_NUMBA", "").lower() not in ("1", "true", "yes")


@njit(nogil=True, cache=True, error_model="numpy")
def regular(positions: np.ndarray, translations: np.ndarray) -> bool:
    """Reject zero-length and collinear derivatives without temporary arrays."""
    for i in range(len(positions)):
        last_x, last_y, last_z, last_norm = 0.0, 0.0, 0.0, 0.0
        for j in range(positions.shape[1] - 1):
            dx = positions[i, j + 1, 0] - positions[i, j, 0] + translations[i, j, 0]
            dy = positions[i, j + 1, 1] - positions[i, j, 1] + translations[i, j, 1]
            dz = positions[i, j + 1, 2] - positions[i, j, 2] + translations[i, j, 2]
            norm = dx * dx + dy * dy + dz * dz
            if norm < 1e-24:
                return False
            if j > 0:
                cx, cy, cz = (
                    last_y * dz - last_z * dy,
                    last_z * dx - last_x * dz,
                    last_x * dy - last_y * dx,
                )
                if cx * cx + cy * cy + cz * cz <= 1e-14 * last_norm * norm:
                    return False
            last_x, last_y, last_z, last_norm = dx, dy, dz, norm
    return True


@njit(nogil=True, cache=True, error_model="numpy")
def gradients(positions: np.ndarray, translations: np.ndarray) -> np.ndarray:
    if positions.shape[1] == 2:
        return _generated_gradients.bond_batch(positions, translations)
    if positions.shape[1] == 3:
        return _generated_gradients.angle_batch(positions, translations)
    return _generated_gradients.dihedral_batch(positions, translations)


@njit(nogil=True, cache=True)
def scatter_jacobian(
    output: np.ndarray,
    row: int,
    indices: np.ndarray,
    derivatives: np.ndarray,
    active: np.ndarray,
) -> int:
    for i in range(len(indices)):
        if active[i]:
            for j in range(indices.shape[1]):
                for axis in range(3):
                    output[row, indices[i, j], axis] = derivatives[i, j, axis]
            row += 1
    return row


def warmup() -> None:
    """Compile molecular coordinate and CPU linear algebra signatures eagerly."""
    if not ENABLED:
        return
    positions = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.1, 0.0], [1.5, 1.0, 0.2], [2.0, 1.0, 1.0]]
    )
    tangent = np.ones_like(positions)
    matrix = np.ones((4, 3, 2))
    families = []
    for width in (2, 3, 4):
        indices = np.arange(width, dtype=np.int32).reshape(1, width)
        families.append(indices)
        translations = np.zeros((1, width - 1, 3))
        local = positions[indices]
        values(local, translations)
        if width == 3:
            bad_angle_mask(local, translations, 15.0 * np.pi / 180.0)
        regular(local, translations)
        gradients(local, translations)
        specialized_hvps(local, translations, tangent[indices])
        specialized_contracted_family(positions, indices, translations, tangent, matrix)
        scatter_jacobian(
            np.zeros((1, 4, 3)),
            0,
            indices,
            gradients(local, translations),
            np.ones(1, dtype=np.bool_),
        )
    guess_hessian(
        positions, np.ones(4, dtype=np.int64), *families, np.ones(2), 1.0, 1.0, 4
    )
    eigh(np.eye(2))
    eigh(np.eye(2, dtype=np.float32))
    qr(np.eye(2))


@njit(nogil=True, cache=True, error_model="numpy")
def guess_hessian(
    positions: np.ndarray,
    numbers: np.ndarray,
    bonds: np.ndarray,
    angles: np.ndarray,
    dihedrals: np.ndarray,
    radii: np.ndarray,
    bohr: float,
    hartree: float,
    natoms: int,
) -> np.ndarray:
    return np.diag(
        np.abs(
            guess_hessian_diagonal(
                positions,
                numbers,
                bonds,
                angles,
                dihedrals,
                radii,
                bohr,
                hartree,
                natoms,
            )
        )
    )

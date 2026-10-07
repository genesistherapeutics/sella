"""Generate exact scalar directional HVPs with SymPy common subexpressions."""

from pathlib import Path
import sympy as sp

source = [
    '"""Generated exact coordinate HVPs; see generate_hvp.py for derivation."""',
    "import math",
    "import numpy as np",
    "from numba import njit",
]
for kind, natoms in [("bond", 2), ("angle", 3), ("dihedral", 4)]:
    coordinates = sp.symbols(f"x0:{3 * natoms}", real=True)
    tangents = sp.symbols(f"v0:{3 * natoms}", real=True)
    translations = sp.symbols(f"t0:{3 * (natoms - 1)}", real=True)
    vectors = [
        sp.Matrix(
            [
                coordinates[3 * (i + 1) + axis]
                - coordinates[3 * i + axis]
                + translations[3 * i + axis]
                for axis in range(3)
            ]
        )
        for i in range(natoms - 1)
    ]
    if kind == "bond":
        value = sp.sqrt(vectors[0].dot(vectors[0]))
    elif kind == "angle":
        left, right = vectors
        value = sp.acos(-left.dot(right) / sp.sqrt(left.dot(left) * right.dot(right)))
    else:
        left, middle, right = vectors
        first, second = left.cross(middle), middle.cross(right)
        value = sp.atan2(
            middle.dot(first.cross(second)),
            sp.sqrt(middle.dot(middle)) * first.dot(second),
        )
    gradient = [sp.diff(value, coordinate) for coordinate in coordinates]
    # Differentiate the compact gradient computation graph, keeping its common
    # subexpressions instead of expanding the Hessian before CSE. This is exact
    # directional forward differentiation and avoids hundreds of redundant terms.
    replacements, gradient_reduced = sp.cse(
        gradient, symbols=sp.numbered_symbols("tmp")
    )
    derivatives = dict(zip(coordinates, tangents))
    operations = []
    for symbol, expression in replacements:
        directional = sum(
            sp.diff(expression, var) * derivatives[var]
            for var in expression.free_symbols
            if var in derivatives
        )
        derivative_symbol = sp.Symbol(f"d_{symbol}")
        operations.append((symbol, expression))
        operations.append((derivative_symbol, directional))
        derivatives[symbol] = derivative_symbol
    reduced = [
        sum(
            sp.diff(expression, var) * derivatives[var]
            for var in expression.free_symbols
            if var in derivatives
        )
        for expression in gradient_reduced
    ]
    replacements = operations
    params = ",".join(map(str, (*coordinates, *tangents, *translations)))
    source.extend(
        [
            "",
            '@njit(nogil=True, cache=True, error_model="numpy")',
            f"def {kind}_single({params}):",
        ]
    )
    source.extend(
        f"    {symbol} = {sp.pycode(expression)}" for symbol, expression in replacements
    )
    source.append(
        "    return ("
        + ",".join(sp.pycode(expression) for expression in reduced)
        + ",)"
    )
    source.extend(
        [
            "",
            '@njit(nogil=True, cache=True, error_model="numpy")',
            f"def {kind}_batch(positions, translations, tangents):",
            "    out = np.empty_like(positions)",
            "    for i in range(len(positions)):",
        ]
    )
    args = ",".join(
        [f"positions[i,{atom},{axis}]" for atom in range(natoms) for axis in range(3)]
        + [f"tangents[i,{atom},{axis}]" for atom in range(natoms) for axis in range(3)]
        + [
            f"translations[i,{atom},{axis}]"
            for atom in range(natoms - 1)
            for axis in range(3)
        ]
    )
    source.append(f"        result = {kind}_single({args})")
    source.extend(
        f"        out[i,{atom},{axis}] = result[{3 * atom + axis}]"
        for atom in range(natoms)
        for axis in range(3)
    )
    source.append("    return out")
    print(kind, "CSE temporaries", len(replacements), flush=True)
(Path(__file__).resolve().parents[1] / "sella").joinpath(
    "_generated_hvp.py"
).write_text("\n".join(source) + "\n")

# Numba caches imported functions in callers without tracking dependency edits.
# Invalidate the wrapper cache whenever generated expressions change.
(Path(__file__).resolve().parents[1] / "sella").joinpath("_numba.py").touch()

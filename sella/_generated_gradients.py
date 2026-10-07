"""Generated exact coordinate gradients; see generate_gradients.py for derivation."""

import math
import numpy as np
from numba import njit


@njit(nogil=True, cache=True, error_model="numpy")
def bond_single(x0, x1, x2, x3, x4, x5, t0, t1, t2):
    tmp0 = t0 - x0 + x3
    tmp1 = t1 - x1 + x4
    tmp2 = t2 - x2 + x5
    tmp3 = 1 / math.sqrt(tmp0**2 + tmp1**2 + tmp2**2)
    return (
        -tmp0 * tmp3,
        -tmp1 * tmp3,
        -tmp2 * tmp3,
        tmp0 * tmp3,
        tmp1 * tmp3,
        tmp2 * tmp3,
    )


@njit(nogil=True, cache=True, error_model="numpy")
def bond_batch(positions, translations):
    out = np.empty_like(positions)
    for i in range(len(positions)):
        result = bond_single(
            positions[i, 0, 0],
            positions[i, 0, 1],
            positions[i, 0, 2],
            positions[i, 1, 0],
            positions[i, 1, 1],
            positions[i, 1, 2],
            translations[i, 0, 0],
            translations[i, 0, 1],
            translations[i, 0, 2],
        )
        out[i, 0, 0] = result[0]
        out[i, 0, 1] = result[1]
        out[i, 0, 2] = result[2]
        out[i, 1, 0] = result[3]
        out[i, 1, 1] = result[4]
        out[i, 1, 2] = result[5]
    return out


@njit(nogil=True, cache=True, error_model="numpy")
def angle_single(x0, x1, x2, x3, x4, x5, x6, x7, x8, t0, t1, t2, t3, t4, t5):
    tmp0 = t0 - x0 + x3
    tmp1 = t1 - x1 + x4
    tmp2 = t2 - x2 + x5
    tmp3 = tmp0**2 + tmp1**2 + tmp2**2
    tmp4 = t3 + x6
    tmp5 = tmp4 - x3
    tmp6 = t4 + x7
    tmp7 = tmp6 - x4
    tmp8 = t5 + x8
    tmp9 = tmp8 - x5
    tmp10 = tmp5**2 + tmp7**2 + tmp9**2
    tmp11 = -tmp0 * tmp5 - tmp1 * tmp7 - tmp2 * tmp9
    tmp12 = 1 / math.sqrt(1 - tmp11**2 / (tmp10 * tmp3))
    tmp13 = 1 / math.sqrt(tmp3)
    tmp14 = 1 / math.sqrt(tmp10)
    tmp15 = tmp13 * tmp14
    tmp16 = tmp11 * tmp14 / tmp3 ** (3 / 2)
    tmp17 = tmp11 * tmp13 / tmp10 ** (3 / 2)
    tmp18 = -tmp0
    tmp19 = -tmp1
    tmp20 = -tmp2
    return (
        -tmp12 * (tmp0 * tmp16 + tmp15 * tmp5),
        -tmp12 * (tmp1 * tmp16 + tmp15 * tmp7),
        -tmp12 * (tmp15 * tmp9 + tmp16 * tmp2),
        -tmp12 * (tmp15 * (t0 - tmp4 - x0 + 2 * x3) + tmp16 * tmp18 + tmp17 * tmp5),
        -tmp12 * (tmp15 * (t1 - tmp6 - x1 + 2 * x4) + tmp16 * tmp19 + tmp17 * tmp7),
        -tmp12 * (tmp15 * (t2 - tmp8 - x2 + 2 * x5) + tmp16 * tmp20 + tmp17 * tmp9),
        -tmp12 * (tmp15 * tmp18 - tmp17 * tmp5),
        -tmp12 * (tmp15 * tmp19 - tmp17 * tmp7),
        -tmp12 * (tmp15 * tmp20 - tmp17 * tmp9),
    )


@njit(nogil=True, cache=True, error_model="numpy")
def angle_batch(positions, translations):
    out = np.empty_like(positions)
    for i in range(len(positions)):
        result = angle_single(
            positions[i, 0, 0],
            positions[i, 0, 1],
            positions[i, 0, 2],
            positions[i, 1, 0],
            positions[i, 1, 1],
            positions[i, 1, 2],
            positions[i, 2, 0],
            positions[i, 2, 1],
            positions[i, 2, 2],
            translations[i, 0, 0],
            translations[i, 0, 1],
            translations[i, 0, 2],
            translations[i, 1, 0],
            translations[i, 1, 1],
            translations[i, 1, 2],
        )
        out[i, 0, 0] = result[0]
        out[i, 0, 1] = result[1]
        out[i, 0, 2] = result[2]
        out[i, 1, 0] = result[3]
        out[i, 1, 1] = result[4]
        out[i, 1, 2] = result[5]
        out[i, 2, 0] = result[6]
        out[i, 2, 1] = result[7]
        out[i, 2, 2] = result[8]
    return out


@njit(nogil=True, cache=True, error_model="numpy")
def dihedral_single(
    x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, t0, t1, t2, t3, t4, t5, t6, t7, t8
):
    tmp0 = -x5
    tmp1 = t5 + x8
    tmp2 = tmp0 + tmp1
    tmp3 = tmp2**2
    tmp4 = -x4
    tmp5 = t4 + x7
    tmp6 = tmp4 + tmp5
    tmp7 = t8 + x11
    tmp8 = tmp7 - x8
    tmp9 = t7 + x10
    tmp10 = tmp9 - x7
    tmp11 = -tmp10 * tmp2 + tmp6 * tmp8
    tmp12 = -tmp6
    tmp13 = tmp11 * tmp6
    tmp14 = -x3
    tmp15 = t3 + x6
    tmp16 = tmp14 + tmp15
    tmp17 = t6 + x9
    tmp18 = tmp17 - x6
    tmp19 = tmp10 * tmp16 - tmp18 * tmp6
    tmp20 = -tmp16 * tmp8 + tmp18 * tmp2
    tmp21 = tmp16**2
    tmp22 = tmp6**2
    tmp23 = tmp21 + tmp22 + tmp3
    tmp24 = math.sqrt(tmp23)
    tmp25 = t0 - x0
    tmp26 = tmp25 + x3
    tmp27 = t1 - x1
    tmp28 = tmp27 + x4
    tmp29 = -tmp16 * tmp28 + tmp26 * tmp6
    tmp30 = t2 - x2
    tmp31 = tmp30 + x5
    tmp32 = tmp16 * tmp31 - tmp2 * tmp26
    tmp33 = tmp2 * tmp28 - tmp31 * tmp6
    tmp34 = tmp11 * tmp33 + tmp19 * tmp29 + tmp20 * tmp32
    tmp35 = tmp20 * tmp29
    tmp36 = tmp19 * tmp32
    tmp37 = tmp35 - tmp36
    tmp38 = tmp11 * tmp29
    tmp39 = tmp19 * tmp33
    tmp40 = tmp38 - tmp39
    tmp41 = tmp11 * tmp32
    tmp42 = tmp20 * tmp33
    tmp43 = tmp41 - tmp42
    tmp44 = -tmp16 * tmp37 - tmp2 * tmp43 + tmp40 * tmp6
    tmp45 = 1 / (tmp23 * tmp34**2 + tmp44**2)
    tmp46 = tmp24 * tmp34 * tmp45
    tmp47 = tmp2 * tmp20
    tmp48 = -tmp44 * tmp45
    tmp49 = tmp24 * tmp48
    tmp50 = -tmp2
    tmp51 = tmp16 * tmp19
    tmp52 = -tmp16
    tmp53 = tmp1 + tmp30
    tmp54 = -tmp53
    tmp55 = -tmp10
    tmp56 = tmp27 + tmp5
    tmp57 = tmp34 / tmp24
    tmp58 = -tmp8
    tmp59 = tmp15 + tmp25
    tmp60 = -tmp59
    tmp61 = -tmp56
    tmp62 = -tmp18
    tmp63 = t4 + tmp4 + tmp9
    tmp64 = -tmp28
    tmp65 = t5 + tmp0 + tmp7
    tmp66 = -tmp65
    tmp67 = t3 + tmp14 + tmp17
    tmp68 = -tmp67
    tmp69 = -tmp31
    tmp70 = -tmp26
    tmp71 = -tmp63
    tmp72 = tmp33 * tmp6
    tmp73 = tmp2 * tmp32
    tmp74 = tmp16 * tmp29
    return (
        tmp46
        * (-tmp11 * tmp3 + tmp12 * tmp13 + tmp16 * (-tmp12 * tmp20 + tmp19 * tmp2))
        + tmp49 * (tmp12 * tmp19 + tmp47),
        tmp46
        * (-tmp20 * tmp21 + tmp47 * tmp50 + tmp6 * (tmp11 * tmp16 - tmp19 * tmp50))
        + tmp49 * (tmp11 * tmp50 + tmp51),
        tmp46
        * (-tmp19 * tmp22 + tmp2 * (-tmp11 * tmp52 + tmp20 * tmp6) + tmp51 * tmp52)
        + tmp49 * (tmp13 + tmp20 * tmp52),
        tmp46
        * (
            tmp16 * (tmp19 * tmp54 - tmp20 * tmp56 - tmp29 * tmp8 + tmp32 * tmp55)
            + tmp2 * (-tmp11 * tmp54 + tmp33 * tmp8)
            + tmp37
            + tmp6 * (tmp11 * tmp56 - tmp33 * tmp55)
        )
        + tmp48
        * (
            tmp24 * (tmp19 * tmp56 + tmp20 * tmp54 + tmp29 * tmp55 + tmp32 * tmp8)
            + tmp52 * tmp57
        ),
        tmp46
        * (
            tmp16 * (tmp18 * tmp32 - tmp20 * tmp60)
            + tmp2 * (tmp20 * tmp53 - tmp32 * tmp58)
            - tmp38
            + tmp39
            + tmp6 * (tmp11 * tmp60 - tmp18 * tmp33 - tmp19 * tmp53 + tmp29 * tmp58)
        )
        + tmp48
        * (
            tmp12 * tmp57
            + tmp24 * (tmp11 * tmp53 + tmp18 * tmp29 + tmp19 * tmp60 + tmp33 * tmp58)
        ),
        tmp46
        * (
            tmp16 * (tmp19 * tmp59 - tmp29 * tmp62)
            + tmp2 * (-tmp10 * tmp32 - tmp11 * tmp59 + tmp20 * tmp61 + tmp33 * tmp62)
            + tmp43
            + tmp6 * (tmp10 * tmp29 - tmp19 * tmp61)
        )
        + tmp48
        * (
            tmp24 * (tmp10 * tmp33 + tmp11 * tmp61 + tmp20 * tmp59 + tmp32 * tmp62)
            + tmp50 * tmp57
        ),
        tmp46
        * (
            tmp16 * (tmp19 * tmp31 - tmp20 * tmp64 - tmp29 * tmp66 + tmp32 * tmp63)
            + tmp2 * (-tmp11 * tmp31 + tmp33 * tmp66)
            - tmp35
            + tmp36
            + tmp6 * (tmp11 * tmp64 - tmp33 * tmp63)
        )
        + tmp48
        * (
            tmp16 * tmp57
            + tmp24 * (tmp19 * tmp64 + tmp20 * tmp31 + tmp29 * tmp63 + tmp32 * tmp66)
        ),
        tmp46
        * (
            tmp16 * (-tmp20 * tmp26 + tmp32 * tmp68)
            + tmp2 * (tmp20 * tmp69 - tmp32 * tmp65)
            + tmp40
            + tmp6 * (tmp11 * tmp26 - tmp19 * tmp69 + tmp29 * tmp65 - tmp33 * tmp68)
        )
        + tmp48
        * (
            tmp24 * (tmp11 * tmp69 + tmp19 * tmp26 + tmp29 * tmp68 + tmp33 * tmp65)
            + tmp57 * tmp6
        ),
        tmp46
        * (
            tmp16 * (tmp19 * tmp70 - tmp29 * tmp67)
            + tmp2 * (-tmp11 * tmp70 + tmp20 * tmp28 - tmp32 * tmp71 + tmp33 * tmp67)
            - tmp41
            + tmp42
            + tmp6 * (-tmp19 * tmp28 + tmp29 * tmp71)
        )
        + tmp48
        * (
            tmp2 * tmp57
            + tmp24 * (tmp11 * tmp28 + tmp20 * tmp70 + tmp32 * tmp67 + tmp33 * tmp71)
        ),
        tmp46 * (-tmp12 * tmp72 + tmp16 * (tmp12 * tmp32 - tmp2 * tmp29) + tmp3 * tmp33)
        + tmp49 * (tmp12 * tmp29 + tmp73),
        tmp46
        * (tmp21 * tmp32 - tmp50 * tmp73 + tmp6 * (-tmp16 * tmp33 + tmp29 * tmp50))
        + tmp49 * (tmp33 * tmp50 + tmp74),
        tmp46 * (tmp2 * (-tmp32 * tmp6 + tmp33 * tmp52) + tmp22 * tmp29 - tmp52 * tmp74)
        + tmp49 * (tmp32 * tmp52 + tmp72),
    )


@njit(nogil=True, cache=True, error_model="numpy")
def dihedral_batch(positions, translations):
    out = np.empty_like(positions)
    for i in range(len(positions)):
        result = dihedral_single(
            positions[i, 0, 0],
            positions[i, 0, 1],
            positions[i, 0, 2],
            positions[i, 1, 0],
            positions[i, 1, 1],
            positions[i, 1, 2],
            positions[i, 2, 0],
            positions[i, 2, 1],
            positions[i, 2, 2],
            positions[i, 3, 0],
            positions[i, 3, 1],
            positions[i, 3, 2],
            translations[i, 0, 0],
            translations[i, 0, 1],
            translations[i, 0, 2],
            translations[i, 1, 0],
            translations[i, 1, 1],
            translations[i, 1, 2],
            translations[i, 2, 0],
            translations[i, 2, 1],
            translations[i, 2, 2],
        )
        out[i, 0, 0] = result[0]
        out[i, 0, 1] = result[1]
        out[i, 0, 2] = result[2]
        out[i, 1, 0] = result[3]
        out[i, 1, 1] = result[4]
        out[i, 1, 2] = result[5]
        out[i, 2, 0] = result[6]
        out[i, 2, 1] = result[7]
        out[i, 2, 2] = result[8]
        out[i, 3, 0] = result[9]
        out[i, 3, 1] = result[10]
        out[i, 3, 2] = result[11]
    return out

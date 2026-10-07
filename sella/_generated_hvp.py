"""Generated exact coordinate HVPs; see generate_hvp.py for derivation."""

import math
import numpy as np
from numba import njit


@njit(nogil=True, cache=True, error_model="numpy")
def bond_single(x0, x1, x2, x3, x4, x5, v0, v1, v2, v3, v4, v5, t0, t1, t2):
    tmp0 = t0 - x0 + x3
    d_tmp0 = -v0 + v3
    tmp1 = t1 - x1 + x4
    d_tmp1 = -v1 + v4
    tmp2 = t2 - x2 + x5
    d_tmp2 = -v2 + v5
    tmp3 = 1 / math.sqrt(tmp0**2 + tmp1**2 + tmp2**2)
    d_tmp3 = (
        -d_tmp0 * tmp0 / (tmp0**2 + tmp1**2 + tmp2**2) ** (3 / 2)
        - d_tmp1 * tmp1 / (tmp0**2 + tmp1**2 + tmp2**2) ** (3 / 2)
        - d_tmp2 * tmp2 / (tmp0**2 + tmp1**2 + tmp2**2) ** (3 / 2)
    )
    return (
        -d_tmp0 * tmp3 - d_tmp3 * tmp0,
        -d_tmp1 * tmp3 - d_tmp3 * tmp1,
        -d_tmp2 * tmp3 - d_tmp3 * tmp2,
        d_tmp0 * tmp3 + d_tmp3 * tmp0,
        d_tmp1 * tmp3 + d_tmp3 * tmp1,
        d_tmp2 * tmp3 + d_tmp3 * tmp2,
    )


@njit(nogil=True, cache=True, error_model="numpy")
def bond_batch(positions, translations, tangents):
    out = np.empty_like(positions)
    for i in range(len(positions)):
        result = bond_single(
            positions[i, 0, 0],
            positions[i, 0, 1],
            positions[i, 0, 2],
            positions[i, 1, 0],
            positions[i, 1, 1],
            positions[i, 1, 2],
            tangents[i, 0, 0],
            tangents[i, 0, 1],
            tangents[i, 0, 2],
            tangents[i, 1, 0],
            tangents[i, 1, 1],
            tangents[i, 1, 2],
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
def angle_single(
    x0,
    x1,
    x2,
    x3,
    x4,
    x5,
    x6,
    x7,
    x8,
    v0,
    v1,
    v2,
    v3,
    v4,
    v5,
    v6,
    v7,
    v8,
    t0,
    t1,
    t2,
    t3,
    t4,
    t5,
):
    tmp0 = t0 - x0 + x3
    d_tmp0 = -v0 + v3
    tmp1 = t1 - x1 + x4
    d_tmp1 = -v1 + v4
    tmp2 = t2 - x2 + x5
    d_tmp2 = -v2 + v5
    tmp3 = tmp0**2 + tmp1**2 + tmp2**2
    d_tmp3 = 2 * d_tmp0 * tmp0 + 2 * d_tmp1 * tmp1 + 2 * d_tmp2 * tmp2
    tmp4 = t3 + x6
    d_tmp4 = v6
    tmp5 = tmp4 - x3
    d_tmp5 = d_tmp4 - v3
    tmp6 = t4 + x7
    d_tmp6 = v7
    tmp7 = tmp6 - x4
    d_tmp7 = d_tmp6 - v4
    tmp8 = t5 + x8
    d_tmp8 = v8
    tmp9 = tmp8 - x5
    d_tmp9 = d_tmp8 - v5
    tmp10 = tmp5**2 + tmp7**2 + tmp9**2
    d_tmp10 = 2 * d_tmp5 * tmp5 + 2 * d_tmp7 * tmp7 + 2 * d_tmp9 * tmp9
    tmp11 = -tmp0 * tmp5 - tmp1 * tmp7 - tmp2 * tmp9
    d_tmp11 = (
        -d_tmp0 * tmp5
        - d_tmp1 * tmp7
        - d_tmp2 * tmp9
        - d_tmp5 * tmp0
        - d_tmp7 * tmp1
        - d_tmp9 * tmp2
    )
    tmp12 = 1 / math.sqrt(1 - tmp11**2 / (tmp10 * tmp3))
    d_tmp12 = (
        -1
        / 2
        * d_tmp10
        * tmp11**2
        / (tmp10**2 * tmp3 * (1 - tmp11**2 / (tmp10 * tmp3)) ** (3 / 2))
        + d_tmp11 * tmp11 / (tmp10 * tmp3 * (1 - tmp11**2 / (tmp10 * tmp3)) ** (3 / 2))
        - 1
        / 2
        * d_tmp3
        * tmp11**2
        / (tmp10 * tmp3**2 * (1 - tmp11**2 / (tmp10 * tmp3)) ** (3 / 2))
    )
    tmp13 = 1 / math.sqrt(tmp3)
    d_tmp13 = -1 / 2 * d_tmp3 / tmp3 ** (3 / 2)
    tmp14 = 1 / math.sqrt(tmp10)
    d_tmp14 = -1 / 2 * d_tmp10 / tmp10 ** (3 / 2)
    tmp15 = tmp13 * tmp14
    d_tmp15 = d_tmp13 * tmp14 + d_tmp14 * tmp13
    tmp16 = tmp11 * tmp14 / tmp3 ** (3 / 2)
    d_tmp16 = (
        d_tmp11 * tmp14 / tmp3 ** (3 / 2)
        + d_tmp14 * tmp11 / tmp3 ** (3 / 2)
        - 3 / 2 * d_tmp3 * tmp11 * tmp14 / tmp3 ** (5 / 2)
    )
    tmp17 = tmp11 * tmp13 / tmp10 ** (3 / 2)
    d_tmp17 = (
        -3 / 2 * d_tmp10 * tmp11 * tmp13 / tmp10 ** (5 / 2)
        + d_tmp11 * tmp13 / tmp10 ** (3 / 2)
        + d_tmp13 * tmp11 / tmp10 ** (3 / 2)
    )
    tmp18 = -tmp0
    d_tmp18 = -d_tmp0
    tmp19 = -tmp1
    d_tmp19 = -d_tmp1
    tmp20 = -tmp2
    d_tmp20 = -d_tmp2
    return (
        -d_tmp0 * tmp12 * tmp16
        + d_tmp12 * (-tmp0 * tmp16 - tmp15 * tmp5)
        - d_tmp15 * tmp12 * tmp5
        - d_tmp16 * tmp0 * tmp12
        - d_tmp5 * tmp12 * tmp15,
        -d_tmp1 * tmp12 * tmp16
        + d_tmp12 * (-tmp1 * tmp16 - tmp15 * tmp7)
        - d_tmp15 * tmp12 * tmp7
        - d_tmp16 * tmp1 * tmp12
        - d_tmp7 * tmp12 * tmp15,
        d_tmp12 * (-tmp15 * tmp9 - tmp16 * tmp2)
        - d_tmp15 * tmp12 * tmp9
        - d_tmp16 * tmp12 * tmp2
        - d_tmp2 * tmp12 * tmp16
        - d_tmp9 * tmp12 * tmp15,
        d_tmp12 * (-tmp15 * (t0 - tmp4 - x0 + 2 * x3) - tmp16 * tmp18 - tmp17 * tmp5)
        - d_tmp15 * tmp12 * (t0 - tmp4 - x0 + 2 * x3)
        - d_tmp16 * tmp12 * tmp18
        - d_tmp17 * tmp12 * tmp5
        - d_tmp18 * tmp12 * tmp16
        + d_tmp4 * tmp12 * tmp15
        - d_tmp5 * tmp12 * tmp17
        + tmp12 * tmp15 * v0
        - 2 * tmp12 * tmp15 * v3,
        d_tmp12 * (-tmp15 * (t1 - tmp6 - x1 + 2 * x4) - tmp16 * tmp19 - tmp17 * tmp7)
        - d_tmp15 * tmp12 * (t1 - tmp6 - x1 + 2 * x4)
        - d_tmp16 * tmp12 * tmp19
        - d_tmp17 * tmp12 * tmp7
        - d_tmp19 * tmp12 * tmp16
        + d_tmp6 * tmp12 * tmp15
        - d_tmp7 * tmp12 * tmp17
        + tmp12 * tmp15 * v1
        - 2 * tmp12 * tmp15 * v4,
        d_tmp12 * (-tmp15 * (t2 - tmp8 - x2 + 2 * x5) - tmp16 * tmp20 - tmp17 * tmp9)
        - d_tmp15 * tmp12 * (t2 - tmp8 - x2 + 2 * x5)
        - d_tmp16 * tmp12 * tmp20
        - d_tmp17 * tmp12 * tmp9
        - d_tmp20 * tmp12 * tmp16
        + d_tmp8 * tmp12 * tmp15
        - d_tmp9 * tmp12 * tmp17
        + tmp12 * tmp15 * v2
        - 2 * tmp12 * tmp15 * v5,
        d_tmp12 * (-tmp15 * tmp18 + tmp17 * tmp5)
        - d_tmp15 * tmp12 * tmp18
        + d_tmp17 * tmp12 * tmp5
        - d_tmp18 * tmp12 * tmp15
        + d_tmp5 * tmp12 * tmp17,
        d_tmp12 * (-tmp15 * tmp19 + tmp17 * tmp7)
        - d_tmp15 * tmp12 * tmp19
        + d_tmp17 * tmp12 * tmp7
        - d_tmp19 * tmp12 * tmp15
        + d_tmp7 * tmp12 * tmp17,
        d_tmp12 * (-tmp15 * tmp20 + tmp17 * tmp9)
        - d_tmp15 * tmp12 * tmp20
        + d_tmp17 * tmp12 * tmp9
        - d_tmp20 * tmp12 * tmp15
        + d_tmp9 * tmp12 * tmp17,
    )


@njit(nogil=True, cache=True, error_model="numpy")
def angle_batch(positions, translations, tangents):
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
            tangents[i, 0, 0],
            tangents[i, 0, 1],
            tangents[i, 0, 2],
            tangents[i, 1, 0],
            tangents[i, 1, 1],
            tangents[i, 1, 2],
            tangents[i, 2, 0],
            tangents[i, 2, 1],
            tangents[i, 2, 2],
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
    x0,
    x1,
    x2,
    x3,
    x4,
    x5,
    x6,
    x7,
    x8,
    x9,
    x10,
    x11,
    v0,
    v1,
    v2,
    v3,
    v4,
    v5,
    v6,
    v7,
    v8,
    v9,
    v10,
    v11,
    t0,
    t1,
    t2,
    t3,
    t4,
    t5,
    t6,
    t7,
    t8,
):
    tmp0 = -x5
    d_tmp0 = -v5
    tmp1 = t5 + x8
    d_tmp1 = v8
    tmp2 = tmp0 + tmp1
    d_tmp2 = d_tmp0 + d_tmp1
    tmp3 = tmp2**2
    d_tmp3 = 2 * d_tmp2 * tmp2
    tmp4 = -x4
    d_tmp4 = -v4
    tmp5 = t4 + x7
    d_tmp5 = v7
    tmp6 = tmp4 + tmp5
    d_tmp6 = d_tmp4 + d_tmp5
    tmp7 = t8 + x11
    d_tmp7 = v11
    tmp8 = tmp7 - x8
    d_tmp8 = d_tmp7 - v8
    tmp9 = t7 + x10
    d_tmp9 = v10
    tmp10 = tmp9 - x7
    d_tmp10 = d_tmp9 - v7
    tmp11 = -tmp10 * tmp2 + tmp6 * tmp8
    d_tmp11 = -d_tmp10 * tmp2 - d_tmp2 * tmp10 + d_tmp6 * tmp8 + d_tmp8 * tmp6
    tmp12 = -tmp6
    d_tmp12 = -d_tmp6
    tmp13 = tmp11 * tmp6
    d_tmp13 = d_tmp11 * tmp6 + d_tmp6 * tmp11
    tmp14 = -x3
    d_tmp14 = -v3
    tmp15 = t3 + x6
    d_tmp15 = v6
    tmp16 = tmp14 + tmp15
    d_tmp16 = d_tmp14 + d_tmp15
    tmp17 = t6 + x9
    d_tmp17 = v9
    tmp18 = tmp17 - x6
    d_tmp18 = d_tmp17 - v6
    tmp19 = tmp10 * tmp16 - tmp18 * tmp6
    d_tmp19 = d_tmp10 * tmp16 + d_tmp16 * tmp10 - d_tmp18 * tmp6 - d_tmp6 * tmp18
    tmp20 = -tmp16 * tmp8 + tmp18 * tmp2
    d_tmp20 = -d_tmp16 * tmp8 + d_tmp18 * tmp2 + d_tmp2 * tmp18 - d_tmp8 * tmp16
    tmp21 = tmp16**2
    d_tmp21 = 2 * d_tmp16 * tmp16
    tmp22 = tmp6**2
    d_tmp22 = 2 * d_tmp6 * tmp6
    tmp23 = tmp21 + tmp22 + tmp3
    d_tmp23 = d_tmp21 + d_tmp22 + d_tmp3
    tmp24 = math.sqrt(tmp23)
    d_tmp24 = (1 / 2) * d_tmp23 / math.sqrt(tmp23)
    tmp25 = t0 - x0
    d_tmp25 = -v0
    tmp26 = tmp25 + x3
    d_tmp26 = d_tmp25 + v3
    tmp27 = t1 - x1
    d_tmp27 = -v1
    tmp28 = tmp27 + x4
    d_tmp28 = d_tmp27 + v4
    tmp29 = -tmp16 * tmp28 + tmp26 * tmp6
    d_tmp29 = -d_tmp16 * tmp28 + d_tmp26 * tmp6 - d_tmp28 * tmp16 + d_tmp6 * tmp26
    tmp30 = t2 - x2
    d_tmp30 = -v2
    tmp31 = tmp30 + x5
    d_tmp31 = d_tmp30 + v5
    tmp32 = tmp16 * tmp31 - tmp2 * tmp26
    d_tmp32 = d_tmp16 * tmp31 - d_tmp2 * tmp26 - d_tmp26 * tmp2 + d_tmp31 * tmp16
    tmp33 = tmp2 * tmp28 - tmp31 * tmp6
    d_tmp33 = d_tmp2 * tmp28 + d_tmp28 * tmp2 - d_tmp31 * tmp6 - d_tmp6 * tmp31
    tmp34 = tmp11 * tmp33 + tmp19 * tmp29 + tmp20 * tmp32
    d_tmp34 = (
        d_tmp11 * tmp33
        + d_tmp19 * tmp29
        + d_tmp20 * tmp32
        + d_tmp29 * tmp19
        + d_tmp32 * tmp20
        + d_tmp33 * tmp11
    )
    tmp35 = tmp20 * tmp29
    d_tmp35 = d_tmp20 * tmp29 + d_tmp29 * tmp20
    tmp36 = tmp19 * tmp32
    d_tmp36 = d_tmp19 * tmp32 + d_tmp32 * tmp19
    tmp37 = tmp35 - tmp36
    d_tmp37 = d_tmp35 - d_tmp36
    tmp38 = tmp11 * tmp29
    d_tmp38 = d_tmp11 * tmp29 + d_tmp29 * tmp11
    tmp39 = tmp19 * tmp33
    d_tmp39 = d_tmp19 * tmp33 + d_tmp33 * tmp19
    tmp40 = tmp38 - tmp39
    d_tmp40 = d_tmp38 - d_tmp39
    tmp41 = tmp11 * tmp32
    d_tmp41 = d_tmp11 * tmp32 + d_tmp32 * tmp11
    tmp42 = tmp20 * tmp33
    d_tmp42 = d_tmp20 * tmp33 + d_tmp33 * tmp20
    tmp43 = tmp41 - tmp42
    d_tmp43 = d_tmp41 - d_tmp42
    tmp44 = -tmp16 * tmp37 - tmp2 * tmp43 + tmp40 * tmp6
    d_tmp44 = (
        -d_tmp16 * tmp37
        - d_tmp2 * tmp43
        - d_tmp37 * tmp16
        + d_tmp40 * tmp6
        - d_tmp43 * tmp2
        + d_tmp6 * tmp40
    )
    tmp45 = 1 / (tmp23 * tmp34**2 + tmp44**2)
    d_tmp45 = (
        -d_tmp23 * tmp34**2 / (tmp23 * tmp34**2 + tmp44**2) ** 2
        - 2 * d_tmp34 * tmp23 * tmp34 / (tmp23 * tmp34**2 + tmp44**2) ** 2
        - 2 * d_tmp44 * tmp44 / (tmp23 * tmp34**2 + tmp44**2) ** 2
    )
    tmp46 = tmp24 * tmp34 * tmp45
    d_tmp46 = (
        d_tmp24 * tmp34 * tmp45 + d_tmp34 * tmp24 * tmp45 + d_tmp45 * tmp24 * tmp34
    )
    tmp47 = tmp2 * tmp20
    d_tmp47 = d_tmp2 * tmp20 + d_tmp20 * tmp2
    tmp48 = -tmp44 * tmp45
    d_tmp48 = -d_tmp44 * tmp45 - d_tmp45 * tmp44
    tmp49 = tmp24 * tmp48
    d_tmp49 = d_tmp24 * tmp48 + d_tmp48 * tmp24
    tmp50 = -tmp2
    d_tmp50 = -d_tmp2
    tmp51 = tmp16 * tmp19
    d_tmp51 = d_tmp16 * tmp19 + d_tmp19 * tmp16
    tmp52 = -tmp16
    d_tmp52 = -d_tmp16
    tmp53 = tmp1 + tmp30
    d_tmp53 = d_tmp1 + d_tmp30
    tmp54 = -tmp53
    d_tmp54 = -d_tmp53
    tmp55 = -tmp10
    d_tmp55 = -d_tmp10
    tmp56 = tmp27 + tmp5
    d_tmp56 = d_tmp27 + d_tmp5
    tmp57 = tmp34 / tmp24
    d_tmp57 = -d_tmp24 * tmp34 / tmp24**2 + d_tmp34 / tmp24
    tmp58 = -tmp8
    d_tmp58 = -d_tmp8
    tmp59 = tmp15 + tmp25
    d_tmp59 = d_tmp15 + d_tmp25
    tmp60 = -tmp59
    d_tmp60 = -d_tmp59
    tmp61 = -tmp56
    d_tmp61 = -d_tmp56
    tmp62 = -tmp18
    d_tmp62 = -d_tmp18
    tmp63 = t4 + tmp4 + tmp9
    d_tmp63 = d_tmp4 + d_tmp9
    tmp64 = -tmp28
    d_tmp64 = -d_tmp28
    tmp65 = t5 + tmp0 + tmp7
    d_tmp65 = d_tmp0 + d_tmp7
    tmp66 = -tmp65
    d_tmp66 = -d_tmp65
    tmp67 = t3 + tmp14 + tmp17
    d_tmp67 = d_tmp14 + d_tmp17
    tmp68 = -tmp67
    d_tmp68 = -d_tmp67
    tmp69 = -tmp31
    d_tmp69 = -d_tmp31
    tmp70 = -tmp26
    d_tmp70 = -d_tmp26
    tmp71 = -tmp63
    d_tmp71 = -d_tmp63
    tmp72 = tmp33 * tmp6
    d_tmp72 = d_tmp33 * tmp6 + d_tmp6 * tmp33
    tmp73 = tmp2 * tmp32
    d_tmp73 = d_tmp2 * tmp32 + d_tmp32 * tmp2
    tmp74 = tmp16 * tmp29
    d_tmp74 = d_tmp16 * tmp29 + d_tmp29 * tmp16
    return (
        -d_tmp11 * tmp3 * tmp46
        + d_tmp12 * (tmp19 * tmp49 + tmp46 * (tmp13 - tmp16 * tmp20))
        + d_tmp13 * tmp12 * tmp46
        + d_tmp16 * tmp46 * (-tmp12 * tmp20 + tmp19 * tmp2)
        + d_tmp19 * (tmp12 * tmp49 + tmp16 * tmp2 * tmp46)
        + d_tmp2 * tmp16 * tmp19 * tmp46
        - d_tmp20 * tmp12 * tmp16 * tmp46
        - d_tmp3 * tmp11 * tmp46
        + d_tmp46
        * (-tmp11 * tmp3 + tmp12 * tmp13 + tmp16 * (-tmp12 * tmp20 + tmp19 * tmp2))
        + d_tmp47 * tmp49
        + d_tmp49 * (tmp12 * tmp19 + tmp47),
        d_tmp11 * (tmp16 * tmp46 * tmp6 + tmp49 * tmp50)
        + d_tmp16 * tmp11 * tmp46 * tmp6
        - d_tmp19 * tmp46 * tmp50 * tmp6
        - d_tmp20 * tmp21 * tmp46
        - d_tmp21 * tmp20 * tmp46
        + d_tmp46
        * (-tmp20 * tmp21 + tmp47 * tmp50 + tmp6 * (tmp11 * tmp16 - tmp19 * tmp50))
        + d_tmp47 * tmp46 * tmp50
        + d_tmp49 * (tmp11 * tmp50 + tmp51)
        + d_tmp50 * (tmp11 * tmp49 + tmp46 * (-tmp19 * tmp6 + tmp47))
        + d_tmp51 * tmp49
        + d_tmp6 * tmp46 * (tmp11 * tmp16 - tmp19 * tmp50),
        -d_tmp11 * tmp2 * tmp46 * tmp52
        + d_tmp13 * tmp49
        - d_tmp19 * tmp22 * tmp46
        + d_tmp2 * tmp46 * (-tmp11 * tmp52 + tmp20 * tmp6)
        + d_tmp20 * (tmp2 * tmp46 * tmp6 + tmp49 * tmp52)
        - d_tmp22 * tmp19 * tmp46
        + d_tmp46
        * (-tmp19 * tmp22 + tmp2 * (-tmp11 * tmp52 + tmp20 * tmp6) + tmp51 * tmp52)
        + d_tmp49 * (tmp13 + tmp20 * tmp52)
        + d_tmp51 * tmp46 * tmp52
        + d_tmp52 * (tmp20 * tmp49 + tmp46 * (-tmp11 * tmp2 + tmp51))
        + d_tmp6 * tmp2 * tmp20 * tmp46,
        d_tmp11 * tmp46 * (-tmp2 * tmp54 + tmp56 * tmp6)
        + d_tmp16
        * tmp46
        * (tmp19 * tmp54 - tmp20 * tmp56 - tmp29 * tmp8 + tmp32 * tmp55)
        + d_tmp19 * (tmp16 * tmp46 * tmp54 + tmp24 * tmp48 * tmp56)
        + d_tmp2 * tmp46 * (-tmp11 * tmp54 + tmp33 * tmp8)
        + d_tmp20 * (-tmp16 * tmp46 * tmp56 + tmp24 * tmp48 * tmp54)
        + d_tmp24
        * tmp48
        * (tmp19 * tmp56 + tmp20 * tmp54 + tmp29 * tmp55 + tmp32 * tmp8)
        + d_tmp29 * (-tmp16 * tmp46 * tmp8 + tmp24 * tmp48 * tmp55)
        + d_tmp32 * (tmp16 * tmp46 * tmp55 + tmp24 * tmp48 * tmp8)
        + d_tmp33 * tmp46 * (tmp2 * tmp8 - tmp55 * tmp6)
        + d_tmp37 * tmp46
        + d_tmp46
        * (
            tmp16 * (tmp19 * tmp54 - tmp20 * tmp56 - tmp29 * tmp8 + tmp32 * tmp55)
            + tmp2 * (-tmp11 * tmp54 + tmp33 * tmp8)
            + tmp37
            + tmp6 * (tmp11 * tmp56 - tmp33 * tmp55)
        )
        + d_tmp48
        * (
            tmp24 * (tmp19 * tmp56 + tmp20 * tmp54 + tmp29 * tmp55 + tmp32 * tmp8)
            + tmp52 * tmp57
        )
        + d_tmp52 * tmp48 * tmp57
        + d_tmp54 * (tmp20 * tmp24 * tmp48 + tmp46 * (-tmp11 * tmp2 + tmp16 * tmp19))
        + d_tmp55 * (tmp24 * tmp29 * tmp48 + tmp46 * (tmp16 * tmp32 - tmp33 * tmp6))
        + d_tmp56 * (tmp19 * tmp24 * tmp48 + tmp46 * (tmp11 * tmp6 - tmp16 * tmp20))
        + d_tmp57 * tmp48 * tmp52
        + d_tmp6 * tmp46 * (tmp11 * tmp56 - tmp33 * tmp55)
        + d_tmp8 * (tmp24 * tmp32 * tmp48 + tmp46 * (-tmp16 * tmp29 + tmp2 * tmp33)),
        d_tmp11 * (tmp24 * tmp48 * tmp53 + tmp46 * tmp6 * tmp60)
        + d_tmp12 * tmp48 * tmp57
        + d_tmp16 * tmp46 * (tmp18 * tmp32 - tmp20 * tmp60)
        + d_tmp18 * (tmp24 * tmp29 * tmp48 + tmp46 * (tmp16 * tmp32 - tmp33 * tmp6))
        + d_tmp19 * (tmp24 * tmp48 * tmp60 - tmp46 * tmp53 * tmp6)
        + d_tmp2 * tmp46 * (tmp20 * tmp53 - tmp32 * tmp58)
        + d_tmp20 * tmp46 * (-tmp16 * tmp60 + tmp2 * tmp53)
        + d_tmp24
        * tmp48
        * (tmp11 * tmp53 + tmp18 * tmp29 + tmp19 * tmp60 + tmp33 * tmp58)
        + d_tmp29 * (tmp18 * tmp24 * tmp48 + tmp46 * tmp58 * tmp6)
        + d_tmp32 * tmp46 * (tmp16 * tmp18 - tmp2 * tmp58)
        + d_tmp33 * (-tmp18 * tmp46 * tmp6 + tmp24 * tmp48 * tmp58)
        - d_tmp38 * tmp46
        + d_tmp39 * tmp46
        + d_tmp46
        * (
            tmp16 * (tmp18 * tmp32 - tmp20 * tmp60)
            + tmp2 * (tmp20 * tmp53 - tmp32 * tmp58)
            - tmp38
            + tmp39
            + tmp6 * (tmp11 * tmp60 - tmp18 * tmp33 - tmp19 * tmp53 + tmp29 * tmp58)
        )
        + d_tmp48
        * (
            tmp12 * tmp57
            + tmp24 * (tmp11 * tmp53 + tmp18 * tmp29 + tmp19 * tmp60 + tmp33 * tmp58)
        )
        + d_tmp53 * (tmp11 * tmp24 * tmp48 + tmp46 * (-tmp19 * tmp6 + tmp2 * tmp20))
        + d_tmp57 * tmp12 * tmp48
        + d_tmp58 * (tmp24 * tmp33 * tmp48 + tmp46 * (-tmp2 * tmp32 + tmp29 * tmp6))
        + d_tmp6
        * tmp46
        * (tmp11 * tmp60 - tmp18 * tmp33 - tmp19 * tmp53 + tmp29 * tmp58)
        + d_tmp60 * (tmp19 * tmp24 * tmp48 + tmp46 * (tmp11 * tmp6 - tmp16 * tmp20)),
        d_tmp10 * (tmp24 * tmp33 * tmp48 + tmp46 * (-tmp2 * tmp32 + tmp29 * tmp6))
        + d_tmp11 * (-tmp2 * tmp46 * tmp59 + tmp24 * tmp48 * tmp61)
        + d_tmp16 * tmp46 * (tmp19 * tmp59 - tmp29 * tmp62)
        + d_tmp19 * tmp46 * (tmp16 * tmp59 - tmp6 * tmp61)
        + d_tmp2
        * tmp46
        * (-tmp10 * tmp32 - tmp11 * tmp59 + tmp20 * tmp61 + tmp33 * tmp62)
        + d_tmp20 * (tmp2 * tmp46 * tmp61 + tmp24 * tmp48 * tmp59)
        + d_tmp24
        * tmp48
        * (tmp10 * tmp33 + tmp11 * tmp61 + tmp20 * tmp59 + tmp32 * tmp62)
        + d_tmp29 * tmp46 * (tmp10 * tmp6 - tmp16 * tmp62)
        + d_tmp32 * (-tmp10 * tmp2 * tmp46 + tmp24 * tmp48 * tmp62)
        + d_tmp33 * (tmp10 * tmp24 * tmp48 + tmp2 * tmp46 * tmp62)
        + d_tmp43 * tmp46
        + d_tmp46
        * (
            tmp16 * (tmp19 * tmp59 - tmp29 * tmp62)
            + tmp2 * (-tmp10 * tmp32 - tmp11 * tmp59 + tmp20 * tmp61 + tmp33 * tmp62)
            + tmp43
            + tmp6 * (tmp10 * tmp29 - tmp19 * tmp61)
        )
        + d_tmp48
        * (
            tmp24 * (tmp10 * tmp33 + tmp11 * tmp61 + tmp20 * tmp59 + tmp32 * tmp62)
            + tmp50 * tmp57
        )
        + d_tmp50 * tmp48 * tmp57
        + d_tmp57 * tmp48 * tmp50
        + d_tmp59 * (tmp20 * tmp24 * tmp48 + tmp46 * (-tmp11 * tmp2 + tmp16 * tmp19))
        + d_tmp6 * tmp46 * (tmp10 * tmp29 - tmp19 * tmp61)
        + d_tmp61 * (tmp11 * tmp24 * tmp48 + tmp46 * (-tmp19 * tmp6 + tmp2 * tmp20))
        + d_tmp62 * (tmp24 * tmp32 * tmp48 + tmp46 * (-tmp16 * tmp29 + tmp2 * tmp33)),
        d_tmp11 * tmp46 * (-tmp2 * tmp31 + tmp6 * tmp64)
        + d_tmp16
        * (
            tmp46 * (tmp19 * tmp31 - tmp20 * tmp64 - tmp29 * tmp66 + tmp32 * tmp63)
            + tmp48 * tmp57
        )
        + d_tmp19 * (tmp16 * tmp31 * tmp46 + tmp24 * tmp48 * tmp64)
        + d_tmp2 * tmp46 * (-tmp11 * tmp31 + tmp33 * tmp66)
        + d_tmp20 * (-tmp16 * tmp46 * tmp64 + tmp24 * tmp31 * tmp48)
        + d_tmp24
        * tmp48
        * (tmp19 * tmp64 + tmp20 * tmp31 + tmp29 * tmp63 + tmp32 * tmp66)
        + d_tmp29 * (-tmp16 * tmp46 * tmp66 + tmp24 * tmp48 * tmp63)
        + d_tmp31 * (tmp20 * tmp24 * tmp48 + tmp46 * (-tmp11 * tmp2 + tmp16 * tmp19))
        + d_tmp32 * (tmp16 * tmp46 * tmp63 + tmp24 * tmp48 * tmp66)
        + d_tmp33 * tmp46 * (tmp2 * tmp66 - tmp6 * tmp63)
        - d_tmp35 * tmp46
        + d_tmp36 * tmp46
        + d_tmp46
        * (
            tmp16 * (tmp19 * tmp31 - tmp20 * tmp64 - tmp29 * tmp66 + tmp32 * tmp63)
            + tmp2 * (-tmp11 * tmp31 + tmp33 * tmp66)
            - tmp35
            + tmp36
            + tmp6 * (tmp11 * tmp64 - tmp33 * tmp63)
        )
        + d_tmp48
        * (
            tmp16 * tmp57
            + tmp24 * (tmp19 * tmp64 + tmp20 * tmp31 + tmp29 * tmp63 + tmp32 * tmp66)
        )
        + d_tmp57 * tmp16 * tmp48
        + d_tmp6 * tmp46 * (tmp11 * tmp64 - tmp33 * tmp63)
        + d_tmp63 * (tmp24 * tmp29 * tmp48 + tmp46 * (tmp16 * tmp32 - tmp33 * tmp6))
        + d_tmp64 * (tmp19 * tmp24 * tmp48 + tmp46 * (tmp11 * tmp6 - tmp16 * tmp20))
        + d_tmp66 * (tmp24 * tmp32 * tmp48 + tmp46 * (-tmp16 * tmp29 + tmp2 * tmp33)),
        d_tmp11 * (tmp24 * tmp48 * tmp69 + tmp26 * tmp46 * tmp6)
        + d_tmp16 * tmp46 * (-tmp20 * tmp26 + tmp32 * tmp68)
        + d_tmp19 * (tmp24 * tmp26 * tmp48 - tmp46 * tmp6 * tmp69)
        + d_tmp2 * tmp46 * (tmp20 * tmp69 - tmp32 * tmp65)
        + d_tmp20 * tmp46 * (-tmp16 * tmp26 + tmp2 * tmp69)
        + d_tmp24
        * tmp48
        * (tmp11 * tmp69 + tmp19 * tmp26 + tmp29 * tmp68 + tmp33 * tmp65)
        + d_tmp26 * (tmp19 * tmp24 * tmp48 + tmp46 * (tmp11 * tmp6 - tmp16 * tmp20))
        + d_tmp29 * (tmp24 * tmp48 * tmp68 + tmp46 * tmp6 * tmp65)
        + d_tmp32 * tmp46 * (tmp16 * tmp68 - tmp2 * tmp65)
        + d_tmp33 * (tmp24 * tmp48 * tmp65 - tmp46 * tmp6 * tmp68)
        + d_tmp40 * tmp46
        + d_tmp46
        * (
            tmp16 * (-tmp20 * tmp26 + tmp32 * tmp68)
            + tmp2 * (tmp20 * tmp69 - tmp32 * tmp65)
            + tmp40
            + tmp6 * (tmp11 * tmp26 - tmp19 * tmp69 + tmp29 * tmp65 - tmp33 * tmp68)
        )
        + d_tmp48
        * (
            tmp24 * (tmp11 * tmp69 + tmp19 * tmp26 + tmp29 * tmp68 + tmp33 * tmp65)
            + tmp57 * tmp6
        )
        + d_tmp57 * tmp48 * tmp6
        + d_tmp6
        * (
            tmp46 * (tmp11 * tmp26 - tmp19 * tmp69 + tmp29 * tmp65 - tmp33 * tmp68)
            + tmp48 * tmp57
        )
        + d_tmp65 * (tmp24 * tmp33 * tmp48 + tmp46 * (-tmp2 * tmp32 + tmp29 * tmp6))
        + d_tmp68 * (tmp24 * tmp29 * tmp48 + tmp46 * (tmp16 * tmp32 - tmp33 * tmp6))
        + d_tmp69 * (tmp11 * tmp24 * tmp48 + tmp46 * (-tmp19 * tmp6 + tmp2 * tmp20)),
        d_tmp11 * (-tmp2 * tmp46 * tmp70 + tmp24 * tmp28 * tmp48)
        + d_tmp16 * tmp46 * (tmp19 * tmp70 - tmp29 * tmp67)
        + d_tmp19 * tmp46 * (tmp16 * tmp70 - tmp28 * tmp6)
        + d_tmp2
        * (
            tmp46 * (-tmp11 * tmp70 + tmp20 * tmp28 - tmp32 * tmp71 + tmp33 * tmp67)
            + tmp48 * tmp57
        )
        + d_tmp20 * (tmp2 * tmp28 * tmp46 + tmp24 * tmp48 * tmp70)
        + d_tmp24
        * tmp48
        * (tmp11 * tmp28 + tmp20 * tmp70 + tmp32 * tmp67 + tmp33 * tmp71)
        + d_tmp28 * (tmp11 * tmp24 * tmp48 + tmp46 * (-tmp19 * tmp6 + tmp2 * tmp20))
        + d_tmp29 * tmp46 * (-tmp16 * tmp67 + tmp6 * tmp71)
        + d_tmp32 * (-tmp2 * tmp46 * tmp71 + tmp24 * tmp48 * tmp67)
        + d_tmp33 * (tmp2 * tmp46 * tmp67 + tmp24 * tmp48 * tmp71)
        - d_tmp41 * tmp46
        + d_tmp42 * tmp46
        + d_tmp46
        * (
            tmp16 * (tmp19 * tmp70 - tmp29 * tmp67)
            + tmp2 * (-tmp11 * tmp70 + tmp20 * tmp28 - tmp32 * tmp71 + tmp33 * tmp67)
            - tmp41
            + tmp42
            + tmp6 * (-tmp19 * tmp28 + tmp29 * tmp71)
        )
        + d_tmp48
        * (
            tmp2 * tmp57
            + tmp24 * (tmp11 * tmp28 + tmp20 * tmp70 + tmp32 * tmp67 + tmp33 * tmp71)
        )
        + d_tmp57 * tmp2 * tmp48
        + d_tmp6 * tmp46 * (-tmp19 * tmp28 + tmp29 * tmp71)
        + d_tmp67 * (tmp24 * tmp32 * tmp48 + tmp46 * (-tmp16 * tmp29 + tmp2 * tmp33))
        + d_tmp70 * (tmp20 * tmp24 * tmp48 + tmp46 * (-tmp11 * tmp2 + tmp16 * tmp19))
        + d_tmp71 * (tmp24 * tmp33 * tmp48 + tmp46 * (-tmp2 * tmp32 + tmp29 * tmp6)),
        d_tmp12 * (tmp29 * tmp49 + tmp46 * (tmp16 * tmp32 - tmp72))
        + d_tmp16 * tmp46 * (tmp12 * tmp32 - tmp2 * tmp29)
        - d_tmp2 * tmp16 * tmp29 * tmp46
        + d_tmp29 * (tmp12 * tmp49 - tmp16 * tmp2 * tmp46)
        + d_tmp3 * tmp33 * tmp46
        + d_tmp32 * tmp12 * tmp16 * tmp46
        + d_tmp33 * tmp3 * tmp46
        + d_tmp46
        * (-tmp12 * tmp72 + tmp16 * (tmp12 * tmp32 - tmp2 * tmp29) + tmp3 * tmp33)
        + d_tmp49 * (tmp12 * tmp29 + tmp73)
        - d_tmp72 * tmp12 * tmp46
        + d_tmp73 * tmp49,
        -d_tmp16 * tmp33 * tmp46 * tmp6
        + d_tmp21 * tmp32 * tmp46
        + d_tmp29 * tmp46 * tmp50 * tmp6
        + d_tmp32 * tmp21 * tmp46
        + d_tmp33 * (-tmp16 * tmp46 * tmp6 + tmp49 * tmp50)
        + d_tmp46
        * (tmp21 * tmp32 - tmp50 * tmp73 + tmp6 * (-tmp16 * tmp33 + tmp29 * tmp50))
        + d_tmp49 * (tmp33 * tmp50 + tmp74)
        + d_tmp50 * (tmp33 * tmp49 + tmp46 * (tmp29 * tmp6 - tmp73))
        + d_tmp6 * tmp46 * (-tmp16 * tmp33 + tmp29 * tmp50)
        - d_tmp73 * tmp46 * tmp50
        + d_tmp74 * tmp49,
        d_tmp2 * tmp46 * (-tmp32 * tmp6 + tmp33 * tmp52)
        + d_tmp22 * tmp29 * tmp46
        + d_tmp29 * tmp22 * tmp46
        + d_tmp32 * (-tmp2 * tmp46 * tmp6 + tmp49 * tmp52)
        + d_tmp33 * tmp2 * tmp46 * tmp52
        + d_tmp46
        * (tmp2 * (-tmp32 * tmp6 + tmp33 * tmp52) + tmp22 * tmp29 - tmp52 * tmp74)
        + d_tmp49 * (tmp32 * tmp52 + tmp72)
        + d_tmp52 * (tmp32 * tmp49 + tmp46 * (tmp2 * tmp33 - tmp74))
        - d_tmp6 * tmp2 * tmp32 * tmp46
        + d_tmp72 * tmp49
        - d_tmp74 * tmp46 * tmp52,
    )


@njit(nogil=True, cache=True, error_model="numpy")
def dihedral_batch(positions, translations, tangents):
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
            tangents[i, 0, 0],
            tangents[i, 0, 1],
            tangents[i, 0, 2],
            tangents[i, 1, 0],
            tangents[i, 1, 1],
            tangents[i, 1, 2],
            tangents[i, 2, 0],
            tangents[i, 2, 1],
            tangents[i, 2, 2],
            tangents[i, 3, 0],
            tangents[i, 3, 1],
            tangents[i, 3, 2],
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

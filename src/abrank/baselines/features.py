"""Handcrafted biophysical, compositional, and k-mer sequence baselines.

Follows strict train-only preprocessing (`StandardScaler` fit strictly on training
features before transforming validation or test splits).
"""

from __future__ import annotations

import numpy as np

AA20 = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA20)}

# Kyte-Doolittle hydropathy scale
KD_HYDROPATHY = {
    "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8,
    "G": -0.4, "H": -3.2, "I": 4.5, "K": -3.9, "L": 3.8,
    "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
    "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3,
}

# Approximate side-chain net charge at pH 7.4
CHARGE_PH74 = {
    "D": -1.0, "E": -1.0, "K": 1.0, "R": 1.0, "H": 0.1,
}

# Zimmerman side-chain bulkiness / volume proxy
VOLUME_PROXY = {
    "G": 60.1, "A": 88.6, "S": 89.0, "C": 108.5, "D": 111.1,
    "P": 112.7, "N": 114.1, "T": 116.1, "E": 138.4, "V": 140.0,
    "Q": 143.8, "H": 153.2, "M": 162.9, "I": 166.7, "L": 166.7,
    "K": 168.6, "R": 173.4, "F": 189.9, "Y": 193.6, "W": 227.8,
}


def _segment_biophys(seq: str) -> np.ndarray:
    """Compute 10 core biophysical descriptors + 20 AA composition = 30 dims."""
    n = max(len(seq), 1)
    counts = np.zeros(20, dtype=np.float32)
    hydro = 0.0
    charge = 0.0
    abs_charge = 0.0
    vol = 0.0
    for ch in seq:
        idx = AA_TO_IDX.get(ch)
        if idx is not None:
            counts[idx] += 1.0
        hydro += KD_HYDROPATHY.get(ch, 0.0)
        c = CHARGE_PH74.get(ch, 0.0)
        charge += c
        abs_charge += abs(c)
        vol += VOLUME_PROXY.get(ch, 130.0)

    aac = counts / n
    arom = float(aac[AA_TO_IDX["F"]] + aac[AA_TO_IDX["W"]] + aac[AA_TO_IDX["Y"]])
    tyr = float(aac[AA_TO_IDX["Y"]])
    trp = float(aac[AA_TO_IDX["W"]])
    polar = float(sum(aac[AA_TO_IDX[x]] for x in "STNQYC"))
    aliph = float(
        aac[AA_TO_IDX["A"]]
        + 2.9 * aac[AA_TO_IDX["V"]]
        + 3.9 * (aac[AA_TO_IDX["I"]] + aac[AA_TO_IDX["L"]])
    )
    gly_pro = float(aac[AA_TO_IDX["G"]] + aac[AA_TO_IDX["P"]])

    stats = np.array(
        [
            np.log1p(len(seq)),
            hydro / n,
            charge / n,
            charge,
            abs_charge / n,
            vol / (n * 200.0),
            arom,
            tyr + trp,
            polar,
            aliph + gly_pro,
        ],
        dtype=np.float32,
    )
    return np.concatenate([stats, aac])


def chain_biophys_features(seq: str, is_antibody_var: bool = False) -> np.ndarray:
    """Biophysical + AAC features for a single chain (plus CDR3-proxy window for VH/VL)."""
    full = _segment_biophys(seq)
    if not is_antibody_var:
        return full
    # Approximate CDR3 window in variable domains: ~residues [-32:-10] before J-motif WGQG / FGGG
    cdr3_win = seq[max(0, len(seq) - 34) : max(1, len(seq) - 10)] if len(seq) >= 25 else seq
    cdr12_win = seq[22:68] if len(seq) >= 70 else seq
    return np.concatenate([full, _segment_biophys(cdr3_win), _segment_biophys(cdr12_win)])


def chain_dipeptide_features(seq: str) -> np.ndarray:
    """Normalized 400-dim 2-mer composition vector."""
    vec = np.zeros(400, dtype=np.float32)
    if len(seq) < 2:
        return vec
    valid = 0
    for i in range(len(seq) - 1):
        a = AA_TO_IDX.get(seq[i])
        b = AA_TO_IDX.get(seq[i + 1])
        if a is not None and b is not None:
            vec[a * 20 + b] += 1.0
            valid += 1
    if valid > 0:
        vec /= valid
    return vec


def build_complex_handcrafted_table(
    abag_df,
    *,
    include_dipeptides: bool = False,
) -> dict[str, np.ndarray]:
    """Map `AbAgID -> feature vector` for all complexes in `abag_df`."""
    h_cache: dict[str, np.ndarray] = {}
    l_cache: dict[str, np.ndarray] = {}
    ag_cache: dict[str, np.ndarray] = {}

    for h in abag_df["Ab_heavy_chain_seq"].unique():
        v = chain_biophys_features(h, is_antibody_var=True)
        if include_dipeptides:
            v = np.concatenate([v, chain_dipeptide_features(h)])
        h_cache[h] = v

    for l in abag_df["Ab_light_chain_seq"].fillna("").astype(str).unique():
        v = chain_biophys_features(l, is_antibody_var=True) if l else np.zeros(90, dtype=np.float32)
        if include_dipeptides:
            v = np.concatenate([v, chain_dipeptide_features(l)])
        l_cache[l] = v

    for ag in abag_df["Ag_seq"].unique():
        v = chain_biophys_features(ag, is_antibody_var=False)
        if include_dipeptides:
            v = np.concatenate([v, chain_dipeptide_features(ag)])
        ag_cache[ag] = v

    out: dict[str, np.ndarray] = {}
    for row in abag_df[["AbAgID", "Ab_heavy_chain_seq", "Ab_light_chain_seq", "Ag_seq"]].itertuples(index=False):
        l_key = "" if row.Ab_light_chain_seq is None or (isinstance(row.Ab_light_chain_seq, float) and np.isnan(row.Ab_light_chain_seq)) else str(row.Ab_light_chain_seq)
        vh = h_cache[row.Ab_heavy_chain_seq]
        vl = l_cache[l_key]
        vag = ag_cache[row.Ag_seq]
        # Include electrostatic & hydropathy complementarity product terms (charge_Ab * charge_Ag, hydro_Ab * hydro_Ag)
        ab_charge = vh[2] + vl[2]
        ag_charge = vag[2]
        ab_hydro = 0.5 * (vh[1] + vl[1]) if l_key else vh[1]
        ag_hydro = vag[1]
        ab_arom = 0.5 * (vh[6] + vl[6]) if l_key else vh[6]
        cross = np.array(
            [
                -ab_charge * ag_charge,  # opposite-charge electrostatic attraction
                ab_hydro * ag_hydro,     # hydrophobic burial product
                ab_arom * ag_hydro,      # aromatic-hydrophobic interface proxy
            ],
            dtype=np.float32,
        )
        out[row.AbAgID] = np.concatenate([vh, vl, vag, cross])
    return out


AROMATIC_SET = frozenset({"F", "W", "Y"})


def _nglyc_count(s: str) -> int:
    cnt = 0
    for i in range(len(s) - 2):
        if s[i] == "N" and s[i + 1] != "P" and s[i + 2] in ("S", "T"):
            cnt += 1
    return cnt


def extract_mutation_biophys_odd(
    vh1_l: list[str],
    vl1_l: list[str],
    ag1_l: list[str],
    vh2_l: list[str],
    vl2_l: list[str],
    ag2_l: list[str],
) -> np.ndarray:
    """Compute 18 strictly antisymmetric (`f(1,2) = -f(2,1)`) site-mutation biophysical features."""
    n = len(vh1_l)
    out = np.zeros((n, 18), dtype=np.float32)

    def _chain_mut_delta(s1: str, s2: str) -> tuple[float, ...]:
        if s1 == s2:
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        if len(s1) == len(s2):
            diffs = [i for i in range(len(s1)) if s1[i] != s2[i]]
            if len(diffs) <= 25:
                d_ala = sum((1.0 if s1[i] == "A" else 0.0) - (1.0 if s2[i] == "A" else 0.0) for i in diffs)
                d_arom = sum((1.0 if s1[i] in AROMATIC_SET else 0.0) - (1.0 if s2[i] in AROMATIC_SET else 0.0) for i in diffs)
                d_vol = sum((VOLUME_PROXY.get(s1[i], 130.0) - VOLUME_PROXY.get(s2[i], 130.0)) / 100.0 for i in diffs)
                d_hyd = sum((KD_HYDROPATHY.get(s1[i], 0.0) - KD_HYDROPATHY.get(s2[i], 0.0)) / 4.5 for i in diffs)
                d_chg = sum(CHARGE_PH74.get(s1[i], 0.0) - CHARGE_PH74.get(s2[i], 0.0) for i in diffs)
                d_abs_chg = sum(abs(CHARGE_PH74.get(s1[i], 0.0)) - abs(CHARGE_PH74.get(s2[i], 0.0)) for i in diffs)
                d_gly_pro = sum((1.0 if s1[i] in ("G", "P") else 0.0) - (1.0 if s2[i] in ("G", "P") else 0.0) for i in diffs)
                return (d_ala, d_arom, d_vol, d_hyd, d_chg, d_abs_chg, d_gly_pro)
        denom = max(len(s1), len(s2), 1)
        d_ala = (s1.count("A") - s2.count("A")) / denom * 10.0
        d_arom = (sum(s1.count(c) for c in AROMATIC_SET) - sum(s2.count(c) for c in AROMATIC_SET)) / denom * 10.0
        d_vol = (sum(VOLUME_PROXY.get(c, 130.0) for c in s1) / max(len(s1), 1) - sum(VOLUME_PROXY.get(c, 130.0) for c in s2) / max(len(s2), 1)) / 50.0
        d_hyd = (sum(KD_HYDROPATHY.get(c, 0.0) for c in s1) / max(len(s1), 1) - sum(KD_HYDROPATHY.get(c, 0.0) for c in s2) / max(len(s2), 1))
        d_chg = (sum(CHARGE_PH74.get(c, 0.0) for c in s1) - sum(CHARGE_PH74.get(c, 0.0) for c in s2)) / 5.0
        d_abs_chg = (sum(abs(CHARGE_PH74.get(c, 0.0)) for c in s1) - sum(abs(CHARGE_PH74.get(c, 0.0)) for c in s2)) / 5.0
        d_gly_pro = ((s1.count("G") + s1.count("P")) - (s2.count("G") + s2.count("P"))) / 5.0
        return (d_ala, d_arom, d_vol, d_hyd, d_chg, d_abs_chg, d_gly_pro)

    for i in range(n):
        dh = _chain_mut_delta(vh1_l[i], vh2_l[i])
        dl = _chain_mut_delta(vl1_l[i], vl2_l[i])
        dg = _chain_mut_delta(ag1_l[i], ag2_l[i])
        ab_d = [dh[k] + dl[k] for k in range(7)]

        q_ab1 = (sum(CHARGE_PH74.get(c, 0.0) for c in vh1_l[i]) + sum(CHARGE_PH74.get(c, 0.0) for c in vl1_l[i])) / 10.0
        q_ab2 = (sum(CHARGE_PH74.get(c, 0.0) for c in vh2_l[i]) + sum(CHARGE_PH74.get(c, 0.0) for c in vl2_l[i])) / 10.0
        q_ag1 = sum(CHARGE_PH74.get(c, 0.0) for c in ag1_l[i]) / 10.0
        q_ag2 = sum(CHARGE_PH74.get(c, 0.0) for c in ag2_l[i]) / 10.0
        d_elec_comp = -(q_ab1 * q_ag1 - q_ab2 * q_ag2)

        d_nglyc_ag = float(_nglyc_count(ag1_l[i]) - _nglyc_count(ag2_l[i]))
        d_nglyc_ab = float((_nglyc_count(vh1_l[i]) + _nglyc_count(vl1_l[i])) - (_nglyc_count(vh2_l[i]) + _nglyc_count(vl2_l[i])))
        d_len_ab = float((len(vh1_l[i]) + len(vl1_l[i])) - (len(vh2_l[i]) + len(vl2_l[i]))) / 10.0

        out[i] = [*ab_d, *dg, d_elec_comp, d_nglyc_ag, d_nglyc_ab, d_len_ab]
    return out


def extract_mutation_impact_even(
    vh1_l: list[str],
    vl1_l: list[str],
    ag1_l: list[str],
    vh2_l: list[str],
    vl2_l: list[str],
    ag2_l: list[str],
) -> np.ndarray:
    """Compute 21 strictly even-symmetric (`f(1,2) == +f(2,1)`) hot-spot / impact descriptors.

    Captures CDR vs. framework mutation location, edit multiplicity, and non-canceling
    per-residue physicochemical change magnitudes (`sum_k |prop(aa1_k) - prop(aa2_k)|`).
    """
    n = len(vh1_l)
    out = np.zeros((n, 21), dtype=np.float32)

    def _chain_abs_change(s1: str, s2: str, is_ab: bool) -> tuple[int, bool, bool, tuple[float, ...]]:
        if s1 == s2:
            return 0, False, False, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        L = min(len(s1), len(s2))
        if len(s1) == len(s2):
            diffs = [k for k in range(L) if s1[k] != s2[k]]
            n_ed = len(diffs)
            if n_ed <= 25:
                cdr3_start = max(0, L - 34)
                cdr3_end = max(1, L - 10)
                in_cdr3 = is_ab and any(cdr3_start <= k < cdr3_end for k in diffs)
                in_cdr12 = is_ab and any(22 <= k < 68 for k in diffs)
                a_ala = sum(abs((1.0 if s1[k] == "A" else 0.0) - (1.0 if s2[k] == "A" else 0.0)) for k in diffs)
                a_arom = sum(abs((1.0 if s1[k] in AROMATIC_SET else 0.0) - (1.0 if s2[k] in AROMATIC_SET else 0.0)) for k in diffs)
                a_vol = sum(abs(VOLUME_PROXY.get(s1[k], 130.0) - VOLUME_PROXY.get(s2[k], 130.0)) / 100.0 for k in diffs)
                a_hyd = sum(abs(KD_HYDROPATHY.get(s1[k], 0.0) - KD_HYDROPATHY.get(s2[k], 0.0)) / 4.5 for k in diffs)
                a_chg = sum(abs(CHARGE_PH74.get(s1[k], 0.0) - CHARGE_PH74.get(s2[k], 0.0)) for k in diffs)
                a_gp = sum(abs((1.0 if s1[k] in ("G", "P") else 0.0) - (1.0 if s2[k] in ("G", "P") else 0.0)) for k in diffs)
                return n_ed, in_cdr3, in_cdr12, (a_ala, a_arom, a_vol, a_hyd, a_chg, a_gp)

        # Length-differing / multi-edit fallback (strictly symmetric in s1, s2)
        n_ed = abs(len(s1) - len(s2)) + sum(1 for k in range(L) if s1[k] != s2[k])
        in_cdr3 = is_ab and (s1[max(0, len(s1) - 34) : max(1, len(s1) - 10)] != s2[max(0, len(s2) - 34) : max(1, len(s2) - 10)])
        in_cdr12 = is_ab and (s1[22:68] != s2[22:68])
        denom = max(len(s1), len(s2), 1)
        a_ala = abs(s1.count("A") - s2.count("A")) / denom * 10.0
        a_arom = abs(sum(s1.count(c) for c in AROMATIC_SET) - sum(s2.count(c) for c in AROMATIC_SET)) / denom * 10.0
        a_vol = abs(sum(VOLUME_PROXY.get(c, 130.0) for c in s1) / max(len(s1), 1) - sum(VOLUME_PROXY.get(c, 130.0) for c in s2) / max(len(s2), 1)) / 50.0
        a_hyd = abs(sum(KD_HYDROPATHY.get(c, 0.0) for c in s1) / max(len(s1), 1) - sum(KD_HYDROPATHY.get(c, 0.0) for c in s2) / max(len(s2), 1))
        a_chg = abs(sum(CHARGE_PH74.get(c, 0.0) for c in s1) - sum(CHARGE_PH74.get(c, 0.0) for c in s2)) / 5.0
        a_gp = abs((s1.count("G") + s1.count("P")) - (s2.count("G") + s2.count("P"))) / 5.0
        return n_ed, in_cdr3, in_cdr12, (a_ala, a_arom, a_vol, a_hyd, a_chg, a_gp)

    for i in range(n):
        nh, c3h, c12h, mh = _chain_abs_change(vh1_l[i], vh2_l[i], True)
        nl, c3l, c12l, ml = _chain_abs_change(vl1_l[i], vl2_l[i], True)
        ng, _, _, mg = _chain_abs_change(ag1_l[i], ag2_l[i], False)

        ab_mut = (nh + nl) > 0
        fw_only = ab_mut and not (c3h or c12h or c3l or c12l)
        ab_mag = [mh[k] + ml[k] for k in range(6)]

        q_ab1 = (sum(CHARGE_PH74.get(c, 0.0) for c in vh1_l[i]) + sum(CHARGE_PH74.get(c, 0.0) for c in vl1_l[i])) / 10.0
        q_ab2 = (sum(CHARGE_PH74.get(c, 0.0) for c in vh2_l[i]) + sum(CHARGE_PH74.get(c, 0.0) for c in vl2_l[i])) / 10.0
        q_ag1 = sum(CHARGE_PH74.get(c, 0.0) for c in ag1_l[i]) / 10.0
        q_ag2 = sum(CHARGE_PH74.get(c, 0.0) for c in ag2_l[i]) / 10.0
        abs_elec = abs(q_ab1 * q_ag1 - q_ab2 * q_ag2)
        abs_nglyc_ag = float(abs(_nglyc_count(ag1_l[i]) - _nglyc_count(ag2_l[i])))

        out[i] = [
            float(c3h),                 # 0: CDRH3 mutated
            float(c12h),                # 1: CDRH1/2 mutated
            float(c3l),                 # 2: CDRL3 mutated
            float(c12l),                # 3: CDRL1/2 mutated
            float(fw_only),             # 4: Framework-only antibody mutation
            float(np.log1p(nh + nl)),   # 5: Antibody edit count (log1p)
            float(np.log1p(ng)),        # 6: Antigen edit count (log1p)
            *ab_mag,                    # 7..12: Ab |Ala|, |Arom|, |Vol|, |Hydro|, |Charge|, |GlyPro|
            *mg,                        # 13..18: Ag |Ala|, |Arom|, |Vol|, |Hydro|, |Charge|, |GlyPro|
            abs_elec,                   # 19: |Δ Electrostatic complementarity|
            abs_nglyc_ag,               # 20: |Δ Antigen N-glycan sequons|
        ]
    return out

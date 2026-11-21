"""Schema validation, sequence hygiene, tag trimming, and sign standardization.

Sign convention across ALL AbRank v2 tables:
- Pointwise `binding_score`: HIGHER ALWAYS MEANS STRONGER BINDING.
    * KD [nM]         -> pKd = 9.0 - log10(Kd_nM)
    * EC50 [nM]       -> pEC50 = 9.0 - log10(EC50_nM)
    * IC50 [ug/mL]    -> neg_log10_IC50 = -log10(IC50_ug_mL)
    * ELISA OD ratio  -> +log10(OD_mut / OD_wt)
    * DDG [kcal/mol]  -> -ddG / (R * T * ln(10))
    * Enrichment      -> +log_enrichment
    * DMS             -> +dms_binding_score (verified higher = stronger binding)
    * Escape [0..1]   -> +log10(max(1 - escape, 1e-2))
- Pairwise splits carry 3-state noise-buffered labels:
    * `binding_diff` = binding_score_1 - binding_score_2  (>0 => Complex 1 binds STRONGER)
    * `logKD_diff`   = -binding_diff                      (>0 => Complex 1 binds WEAKER / higher Kd)
    * `label`        in {-1, 0, +1}:
        - `+1` : Impactful improvement (`binding_diff >= +m`)
        - `0`  : Neutral / negligible functional change (`|binding_diff| <= m_neutral`)
        - `-1` : Impactful depletion (`binding_diff <= -m`)
"""

from __future__ import annotations

import re
import numpy as np
import pandas as pd

STANDARD_AA = frozenset("ACDEFGHIKLMNPQRSTVWY")

# Gas constant in kcal/(mol*K) at 298.15 K
R_KCAL = 0.001987204
TEMP_K = 298.15
RT_KCAL = R_KCAL * TEMP_K
RT_LN10 = RT_KCAL * np.log(10.0)

# N-terminal expression tags commonly present on recombinant antigens
_NTERM_TAGS = (
    "MGSSHHHHHHSSGLVPRGSHM",  # pET28a His6-thrombin leader (e.g. IgDesign TSLP)
    "GSHHHHHHGSG",            # N-terminal His6-GS linker (e.g. SAbDab 5jq6)
    "MHHHHHHSSGVDLGTENLYFQS", # His6-TEV N-terminal leader
    "HHHHHHGSG",
    "HHHHHHSSG",
)

# C-terminal purification tags and Fc fusions commonly present on recombinant antigens
_CTERM_TAGS = (
    "GGSGSDKTHTCPPCP",  # IgG1 Fc fusion (hinge), e.g. Kothiwal IL23R
    "EPKSCDKTHTCPPCP",  # IgG1 Fc fusion, alternative hinge
    "DKTHTCPPCPAPE",    # IgG1 Fc fusion, hinge-CH2
    "IEGRMD",           # Factor Xa + Fc fusion linker
    "LNDIFEAQKIEWHE",   # Avi-tag (BirA biotinylation site)
    "SAWSHPQFEK",       # Strep-tag II
    "YPYDVPDYA",        # HA-tag
    "DYKDDDDK",         # FLAG-tag
    "ENLYFQG",          # TEV protease cleavage site (e.g. IgDesign TNFRSF9)
    "ENLYFQS",          # TEV protease cleavage site
    "LEVLFQGP",         # PreScission / HRV-3C cleavage site
    "HHHHHHHHHH",       # His10
    "HHHHHHHH",         # His8
    "HHHHHH",           # His6
)

PAIR_COLUMNS = [
    "pair_id",
    "AbAgID_1",
    "AbAgID_2",
    "binding_diff",
    "logKD_diff",
    "label",
    "Category",
    "Lev_distance",
    "AbRank_group",
    "Ag_group_1",
    "Ag_group_2",
    "Ag_type",
    "Assay_modality",
    "Dataset_1",
    "Dataset_2",
]


def clean_seq(seq: object) -> str:
    """Normalize whitespace and case on an amino-acid sequence string."""
    if seq is None or (isinstance(seq, float) and np.isnan(seq)):
        return ""
    return str(seq).strip().upper().replace(" ", "").replace("\n", "").replace("\r", "")


def is_valid_aa_seq(seq: object, min_len: int = 5) -> bool:
    """Return True iff `seq` is at least `min_len` standard 20 amino acids."""
    s = clean_seq(seq)
    if len(s) < min_len:
        return False
    return set(s) <= STANDARD_AA


def trim_antigen_tags(seq: object) -> str:
    """Strip common N- and C-terminal recombinant expression/purification tags."""
    s = clean_seq(seq)
    if not s:
        return s

    for ntag in _NTERM_TAGS:
        if s.startswith(ntag) and len(s) - len(ntag) >= 12:
            s = s[len(ntag) :]
            break

    # Cut C-terminal tags only if they appear in the C-terminal region or as an Fc fusion
    hits = []
    for ctag in _CTERM_TAGS:
        idx = s.find(ctag)
        if idx >= 12:
            # Short tags (<=10 AA like ENLYFQG or HHHHHH) should be near the C-terminus
            # (within last 45 residues) so internal native motifs are never clipped.
            if len(ctag) >= 12 or (len(s) - idx) <= 45:
                hits.append(idx)
    if hits:
        s = s[: min(hits)]
        s = re.sub(r"(?:GGGS|GGGGS|GSG|GGS|GS|G|SP){1,3}$", "", s)
    return s


def trim_fab_to_fv(heavy_seq: str, light_seq: str) -> tuple[str, str]:
    """Trim full Fab sequences (VH+CH1 / VL+CL, e.g. AbBiBench 5a12) down to variable Fv.

    AbBiBench `5a12_vegf` carries an 8-residue 'XXXXXXXX' mask inside the CH1 constant
    domain (positions 133-140) while all library mutations occur strictly inside VH/VL.
    Trimming at the canonical J-segment -> CH1 ('ASTKGPSVFPLAP') and J-segment -> CL
    ('RTVAAPSVFIFPP' / 'GQPKAAPSVTLFP') junction recovers clean 20-AA variable domains.
    """
    h = clean_seq(heavy_seq)
    l = clean_seq(light_seq)

    ch1_idx = h.find("ASTKGPSVFPLAP")
    if ch1_idx >= 95:
        h = h[:ch1_idx]
    else:
        ch1_short = h.find("ASTKGPSVF", 95)
        if ch1_short >= 95:
            h = h[:ch1_short]

    for cl_motif in ("RTVAAPSVFIFPP", "GQPKAAPSVTLFP", "QPKAAPSVTLFPP", "RTVAAPSVF", "GQPKAAPSV", "QPKANPTVT"):
        cl_idx = l.find(cl_motif, 85)
        if cl_idx >= 85:
            l = l[:cl_idx]
            break

    return h, l


def parse_numeric_measurement(val: object) -> tuple[float, str]:
    """Parse an experimental readout into `(numeric_value, operator)`.

    Operators:
      - `'='`       : exact finite positive measurement
      - `'<'`, `'>'`: censored measurement (`<0.1`, `>1000`, `>=2000`, etc.)
      - `'range'`   : interval string such as `'10-200'` (cannot be used as point estimate)
      - `'invalid'` : missing, non-numeric, zero, or negative
    """
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return (np.nan, "invalid")
    if isinstance(val, (int, float, np.floating, np.integer)):
        fval = float(val)
        if np.isfinite(fval) and fval > 0:
            return (fval, "=")
        return (np.nan, "invalid")

    s = str(val).strip()
    if not s or s.upper() in {"NA", "N/A", "NAN", "NONE", "ND", "N.D.", "NB", "N.B.", "-"}:
        return (np.nan, "invalid")

    # Strip trailing units if present
    s_clean = re.sub(r"\s*(?:nM|uM|µM|pM|M|ug/mL|µg/mL|ng/mL)\s*$", "", s, flags=re.IGNORECASE)

    op = "="
    if s_clean.startswith((">=", "≥", ">")):
        op = ">"
        s_clean = s_clean.lstrip(">≥= ").strip()
    elif s_clean.startswith(("<=", "≤", "<")):
        op = "<"
        s_clean = s_clean.lstrip("<≤= ").strip()
    elif s_clean.startswith("="):
        s_clean = s_clean.lstrip("= ").strip()

    # Detect ranges like '10-200' (distinguished from scientific notation '1e-3')
    if re.match(r"^\d+(?:\.\d+)?\s*[-–]\s*\d+(?:\.\d+)?$", s_clean):
        return (np.nan, "range")

    try:
        fval = float(s_clean)
    except ValueError:
        return (np.nan, "invalid")

    if not np.isfinite(fval) or fval <= 0:
        return (np.nan, "invalid")
    return (fval, op)


def kd_nm_to_pkd(kd_nm: pd.Series | np.ndarray | float) -> pd.Series | np.ndarray | float:
    """Convert Kd in nM to pKd = -log10(Kd [M]) = 9 - log10(Kd [nM]). Higher = stronger."""
    return 9.0 - np.log10(kd_nm)


def validate_assay_dataframe(
    df: pd.DataFrame,
    *,
    name: str,
    require_light_chain: bool = True,
    min_ag_len: int = 10,
) -> pd.DataFrame:
    """Filter and strictly validate a curated assay DataFrame."""
    out = df.copy()
    out["Ab_heavy_chain_seq"] = out["Ab_heavy_chain_seq"].map(clean_seq)
    if require_light_chain:
        out["Ab_light_chain_seq"] = out["Ab_light_chain_seq"].map(clean_seq)
    else:
        out["Ab_light_chain_seq"] = out.get("Ab_light_chain_seq", pd.Series("", index=out.index)).fillna("").map(clean_seq)
    out["Ag_seq"] = out["Ag_seq"].map(clean_seq)

    valid_h = out["Ab_heavy_chain_seq"].map(lambda s: is_valid_aa_seq(s, min_len=70))
    valid_l = (
        out["Ab_light_chain_seq"].map(lambda s: is_valid_aa_seq(s, min_len=70))
        if require_light_chain
        else pd.Series(True, index=out.index)
    )
    valid_ag = out["Ag_seq"].map(lambda s: is_valid_aa_seq(s, min_len=min_ag_len))
    valid_score = pd.to_numeric(out["binding_score"], errors="coerce").notna() & np.isfinite(
        pd.to_numeric(out["binding_score"], errors="coerce")
    )

    keep = valid_h & valid_l & valid_ag & valid_score
    dropped = int((~keep).sum())
    if dropped > 0:
        print(
            f"[{name}] Filtered {dropped} invalid row(s) "
            f"(bad_H={int((~valid_h).sum())}, bad_L={int((~valid_l).sum())}, "
            f"bad_Ag={int((~valid_ag).sum())}, bad_score={int((~valid_score).sum())}); "
            f"kept {int(keep.sum())}."
        )
    out = out.loc[keep].reset_index(drop=True)
    out["binding_score"] = out["binding_score"].astype(float)

    # Final hard assertions
    assert out["binding_score"].isna().sum() == 0, f"{name}: NaN binding_score survived!"
    assert set(out["Ag_type"].unique()) <= {"Viral", "Non-Viral"}, (
        f"{name}: unexpected Ag_type values {out['Ag_type'].unique()}"
    )
    return out

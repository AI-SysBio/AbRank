"""Canonical antigen taxonomy, synonym unification, and Viral/Non-Viral classification.

Resolves cross-dataset naming collisions and historical SAbDab misclassifications so that:
1. Homologous constructs of the same biological target share a single `Ag_group`
   (preventing fake cross-antigen Global pairs and preventing Unseen-Ag leakage).
2. Short linear peptide targets (e.g. `SARSCoV2_HR2`, 14 AA) remain isolated in their own
   `Ag_group` so they stay restricted to within-antigen L1-L3 local pairs and never enter
   cross-antigen Global pairs.
"""

from __future__ import annotations

import re

# Canonical antigen group aliases across all sources
ANTIGEN_SYNONYM_TO_GROUP: dict[str, str] = {
    # SARS-CoV-2 folded Spike / domain constructs -> unified SARSCoV2_Spike family
    # (Specific domain/construct is preserved in `Ag_name`, while `Ag_group` unifies them!)
    "SARSCoV2_RBD": "SARSCoV2_Spike",
    "SARSCoV2_S1": "SARSCoV2_Spike",
    "SARSCoV2_S2": "SARSCoV2_Spike",
    "SARSCoV2_NTD": "SARSCoV2_Spike",
    "SARSCoV2_None": "SARSCoV2_Spike",
    "SARSCoV2_nonRBD": "SARSCoV2_Spike",
    "SARS-CoV-2": "SARSCoV2_Spike",
    "SarsCov2": "SARSCoV2_Spike",
    # 14-AA HR2 linear peptide isolated in its own Ag_group (restricted to L1-L3 local pairs)
    "SARSCoV2_HR2": "SARSCoV2_HR2",
    # SARS-CoV-1
    "SARSCoV_RBD": "SARSCoV1_Spike",
    "SARSCoV_S2": "SARSCoV1_Spike",
    "SARSCoV_None": "SARSCoV1_Spike",
    "SARS-CoV": "SARSCoV1_Spike",
    # Bat sarbecoviruses
    "WIV1CoV_RBD": "WIV1CoV_Spike",
    "WIV1CoV_None": "WIV1CoV_Spike",
    "SHC014CoV_RBD": "SHC014CoV_Spike",
    "SHC014CoV_None": "SHC014CoV_Spike",
    # MERS-CoV
    "MERSCoV_RBD": "MERSCoV_Spike",
    "MersCov_RBD": "MERSCoV_Spike",
    "MersCoV_RBD": "MERSCoV_Spike",
    "MERSCoV_NTD": "MERSCoV_Spike",
    "MersCoV_NTD": "MERSCoV_Spike",
    "MERSCoV_S2": "MERSCoV_Spike",
    "MersCoV_S2": "MERSCoV_Spike",
    "MERSCoV_nonRBD": "MERSCoV_Spike",
    # HIV-1 Envelope (gp120 / gp41 / gp160)
    "HIV_gp120": "HIV1_Env",
    "HIV_gp160": "HIV1_Env",
    "HIV_gp41": "HIV1_Env",
    "transmembrane_protein_gp41": "HIV1_Env",
    "germline-targeting_hiv-1_gp120_engineered_outer_domain": "HIV1_Env",
    "germline-targeting_hiv-1_gp120_engineered_outer_domain_eod-gt8_": "HIV1_Env",
    "epitope_scaffold_2bodx43": "HIV1_Env",
    "t117": "HIV1_Env",
    "capsid_protein_p24": "HIV1_p24",
    # Influenza HA & NA
    "Influenza_AH1": "Influenza_H1_HA",
    "Influenza_AH3": "Influenza_H3_HA",
    "Influenza_AH7": "Influenza_H7_HA",
    "Influenza_AH9": "Influenza_H9_HA",
    "Influenza_B_HA": "Influenza_B_HA",
    "hemagglutinin": "Influenza_H1_HA",
    "Influenza_Neuraminidase": "Influenza_NA",
    # Other Viruses
    "HCMV-gB": "HCMV_gB",
    "Vaccinia_L1": "Vaccinia_L1",
    "protein_l1": "Vaccinia_L1",
    "a33r": "Vaccinia_A33R",
    "imv_membrane_protein": "Vaccinia_D8",
    "domain_iii_of_dengue_virus_2": "Dengue_EDIII",
    "Dengue_EDIII": "Dengue_EDIII",
    "zika_envelope_diii": "Zika_EDIII",
    "Zika_EDIII": "Zika_EDIII",
    "glycoprotein_g": "Hendra_G",
    "capsid_protein_vp1": "Parvovirus_VP1",
    "polyprotein": "HCV_E2",
    "HCV_E2": "HCV_E2",
    "SFTSV_Gn": "SFTSV_Gn",
    # Non-Viral Human / Bacterial / Model Targets (unifying SAbDab, SKEMPI, Kothiwal, IgDesign, AbDesign)
    "HER2": "HER2_ECD",
    "HER2_ECD": "HER2_ECD",
    "HER2_DomainIV": "HER2_ECD",
    "PDL1": "PDL1",
    "PD-L1": "PDL1",
    "programmed_cell_death_1_ligand_1": "PDL1",
    "PDL2": "PDL2",
    "PD-L2": "PDL2",
    "PD1": "PD1",
    "PD-1": "PD1",
    "ROBO1": "ROBO1",
    "roundabout_homolog_1": "ROBO1",
    "ROBO2N": "ROBO2N",
    "hROBO2N": "ROBO2N",
    "DKK 1.00": "DKK1",
    "DKK1": "DKK1",
    "Tissue_Factor": "Tissue_Factor",
    "tissue_factor": "Tissue_Factor",
    "HPr": "HPr",
    "histidine-containing_protein": "HPr",
    "SIRPa": "SIRPa",
    "tyrosine-protein_phosphatase_non-receptor_type_substrate_1": "SIRPa",
    "Staphylococcal_Enterotoxin_B": "Staph_Enterotoxin_B",
    "enterotoxin_type_b": "Staph_Enterotoxin_B",
    "25_kda_ookinete_surface_antigen": "Plasmodium_Pfs25",
    "pfs25": "Plasmodium_Pfs25",
    "mhc_class_i_chain-related_protein_a": "MICA",
    "mhc_class_i_polypeptide-related_sequence_a": "MICA",
    "hepatocyte_growth_factor_activator": "HGFA",
    "hepatocyte_growth_factor_activator_long_chain": "HGFA",
    "induced_myeloid_leukemia_cell_differentiation_protein_mcl-1homolog": "MCL1",
    "tumor_necrosis_factor_receptor_superfamily_member_10b": "TNFRSF10B",
    "tumor_necrosis_factor_receptor_superfamily_member_10bprecursor_": "TNFRSF10B",
    "COL6A3": "COL6A3_C5",
    "COL6A3_C5": "COL6A3_C5",
    "lysozyme": "HEWL",
    "Ang2": "ANG2",
    "COVID-19": "SARSCoV2_Spike",
    "HIV": "HIV1_Env",
    "Lassa": "Lassa_GPC",
    "Nipah": "Nipah_G",
    "zika": "Zika_EDIII",
    "LAMP-1": "LAMP1",
    "IL-7RA": "IL7RA",
    "Insulin_Receptor": "InsulinR",
    # SAbDab-nano / VHH target unification
    "lysozyme_c": "HEWL",
    "ribonuclease_pancreatic": "RNase_A",
    "alpha-amylase": "Alpha_Amylase",
    "tumor_necrosis_factor": "TNFa",
    "tumor_necrosis_factor_receptor_superfamily_member_5": "CD40",
    "vascular_endothelial_growth_factor": "VEGF",
    "vascular_endothelial_growth_factor_a": "VEGF",
    "epidermal_growth_factor_receptor": "EGFR",
    "receptor_tyrosine-protein_kinase_erbb-2": "HER2_ECD",
    "programmed_cell_death_1_ligand_1": "PDL1",
    "fusion_glycoprotein_f0": "RSV_F",
    "capsid_protein_p24": "HIV1_p24",
    "capsid_protein": "Norovirus_VP1",
    "vp1": "Norovirus_VP1",
    "cyclin-g-associated_kinase": "GAK",
    "beta-1_adrenergic_receptor": "ADRB1",
    "beta-2_adrenergic_receptor": "ADRB2",
    "dihydrofolate_reductase": "DHFR",
    "urokinase-type_plasminogen_activator": "uPA",
    "carboxypeptidase_b2": "CPB2",
    "protein_vsig4": "VSIG4",
    "v-set_and_immunoglobulin_domain-containing_protein_4": "VSIG4",
    "k88_fimbrial_protein_ac": "K88_Fimbrial_Adhesin",
    "k88_fimbrial_protein_ad": "K88_Fimbrial_Adhesin",
    "f18_fimbrial_adhesin_ac": "F18_Fimbrial_Adhesin",
    "shiga-like_toxin_2_subunit_b": "Stx2B",
    "major_structural_protein_1": "Phage_Tuc2009_MSP1",
    "green_fluorescent_protein": "GFP",
    "ricin": "Ricin",
    "carbonic_anhydrase": "Carbonic_Anhydrase",
    "immunoglobulin_heavy_constant_epsilon": "IgE_Fc",
    "beta-2-microglobulin": "B2M",
    "neuronal_migration_protein_doublecortin": "DCX",
    "pc4_and_sfrs1-interacting_protein": "LEDGF",
    "fructose-bisphosphate_aldolase": "Trypanosoma_Aldolase",
    "mdm4_protein": "MDM4",
    "lmo-2": "LMO2",
    "complement_c3": "Complement_C3",
    "beta-secretase_2": "BACE2",
    "interleukin-12_subunit_beta": "IL12B",
    "Galectin-3": "Galectin3",
    "LGALS3": "Galectin3",
}

# Explicit PDB-level target mappings for SAbDab viral complexes with generic upstream names
# on generic names ("spike glycoprotein", "envelope protein", "transmembrane glycoprotein", "hemagglutinin")
SABDAB_PDB_OVERRIDES: dict[str, tuple[str, str]] = {
    "3MAC": ("HIV_gp41", "Viral"),
    "2CMR": ("HIV_gp41", "Viral"),
    "3MA9": ("HIV_gp41", "Viral"),
    "3RU8": ("HIV_gp41", "Viral"),
    "4ALA": ("Dengue_EDIII", "Viral"),
    "4AL8": ("Dengue_EDIII", "Viral"),
    "5JHL": ("Zika_EDIII", "Viral"),
    "4ZPT": ("MERSCoV_RBD", "Viral"),
    "5ZXV": ("MERSCoV_RBD", "Viral"),
    "2DD8": ("SARSCoV_RBD", "Viral"),
    "5Y11": ("SFTSV_Gn", "Viral"),
    "2VIS": ("Influenza_AH3", "Viral"),
    "2VIR": ("Influenza_AH3", "Viral"),
    "1QFU": ("Influenza_AH3", "Viral"),
    "1EO8": ("Influenza_AH3", "Viral"),
    "4GMS": ("Influenza_AH3", "Viral"),
    "5W42": ("Influenza_AH3", "Viral"),
    "4FQJ": ("Influenza_B_HA", "Viral"),
    "5VAG": ("Influenza_AH7", "Viral"),
    "6BKC": ("HCV_E2", "Viral"),
    "6BKD": ("HCV_E2", "Viral"),
    "4U6H": ("Vaccinia_L1", "Viral"),
    "4LU5": ("Vaccinia_A33R", "Viral"),
    "4LQF": ("Vaccinia_A33R", "Viral"),
    "5USL": ("Vaccinia_D8", "Viral"),
    "6CMG": ("Hendra_G", "Viral"),
    "5KW9": ("Parvovirus_VP1", "Viral"),
    "1E6J": ("HIV1_p24", "Viral"),
}

VIRAL_GROUPS = frozenset({
    "SARSCoV2_Spike",
    "SARSCoV2_HR2",
    "SARSCoV1_Spike",
    "WIV1CoV_Spike",
    "SHC014CoV_Spike",
    "MERSCoV_Spike",
    "HIV1_Env",
    "HIV1_p24",
    "Influenza_H1_HA",
    "Influenza_H3_HA",
    "Influenza_H7_HA",
    "Influenza_H9_HA",
    "Influenza_B_HA",
    "Influenza_NA",
    "HCMV_gB",
    "Vaccinia_L1",
    "Vaccinia_A33R",
    "Vaccinia_D8",
    "Dengue_EDIII",
    "Zika_EDIII",
    "Hendra_G",
    "Parvovirus_VP1",
    "HCV_E2",
    "SFTSV_Gn",
    "EBV_BHRF1",
    "RSV_F",
    "Nipah_G",
    "Lassa_GPC",
    "HMPV_F",
    "Norovirus_VP1",
    "Phage_Tuc2009_MSP1",
})


def canonical_ag_group(ag_name: str) -> str:
    """Map any raw or dataset-level `Ag_name` to its canonical `Ag_group`."""
    name = str(ag_name).strip().replace("\xa0", " ")
    if name in ANTIGEN_SYNONYM_TO_GROUP:
        return ANTIGEN_SYNONYM_TO_GROUP[name]
    if name.startswith("HIV-") or name.startswith("HIV_"):
        return "HIV1_Env"
    if name.startswith("SARS-CoV-2_") or name.startswith("SARSCoV2_"):
        if "HR2" in name:
            return "SARSCoV2_HR2"
        return "SARSCoV2_Spike"
    return re.sub(r"\s+", "_", name)


def canonical_ag_type(ag_group: str, hint: str | None = None, species: str | None = None) -> str:
    """Return strictly `'Viral'` or `'Non-Viral'`.

    Note: Syncytin-2 (`ERVFRD-1`) is an endogenous human placental protein and is
    classified as `'Non-Viral'`. Bacterial proteins (e.g. HPr, Staph Enterotoxin B,
    Botulinum Neurotoxin A) and protozoan proteins (Pfs25) are `'Non-Viral'`.
    """
    if ag_group == "Syncytin2":
        return "Non-Viral"
    if ag_group in VIRAL_GROUPS:
        return "Viral"
    low_grp = ag_group.lower()
    if any(k in low_grp for k in ("cov", "hiv", "influenza", "dengue", "zika", "vaccinia", "hcmv", "ebv", "rsv")):
        return "Viral"
    if species:
        sp = str(species).lower()
        if ("virus" in sp or "viral" in sp) and "endogenous" not in sp:
            return "Viral"
    if hint:
        h = str(hint).strip().lower()
        if h == "viral":
            return "Viral"
    return "Non-Viral"

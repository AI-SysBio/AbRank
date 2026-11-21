"""Core paired-chain KD curation for foundational datasets:
AbBiBench (KD panels), SKEMPIv2, SAbDab, Origin-1, IgDesign, HER2, OVA, and AbCoV.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from abrank.config import RepoPaths
from abrank.curate.common import (
    finalize_source_table,
    load_antigen_catalog,
    load_pdb_mapping,
)
from abrank.schema import (
    kd_nm_to_pkd,
    parse_numeric_measurement,
    trim_antigen_tags,
    trim_fab_to_fv,
)
from abrank.taxonomy import SABDAB_PDB_OVERRIDES

# Metadata for the 11 AbBiBench KD panels (from Raw/*_benchmarking_data.csv)
# Format: complex_id -> (WT_PDB, Ag_name, Ag_type, WT_Abname, min_raw_binding_score_exclusive)
ABBIBENCH_KD_PANELS: dict[str, tuple[str, str, str, str, float | None]] = {
    "1mhp_LC": ("1MHP", "ITGA1", "Non-Viral", "AQC2", None),
    "1n8z": ("1N8Z", "HER2_ECD", "Non-Viral", "Trastuzumab", None),
    # Tite-seq panels: filter exact censored top-concentration floor (7.0 = 100 nM; 6.0 = 1000 nM)
    "3gbn_h1": ("3GBN_H1", "Influenza_AH1", "Viral", "CR6261", 7.0001),
    "3gbn_h9": ("3GBN_H9", "Influenza_AH9", "Viral", "CR6261", 7.0001),
    "4fqi_h1": ("4FQI_H1", "Influenza_AH1", "Viral", "CR9114", 7.0001),
    "4fqi_h3": ("4FQI_H3", "Influenza_AH3", "Viral", "CR9114", 6.0001),
    # MIT AlphaSeq panels against 14-AA SARSCoV2_HR2 peptide (restricted to L1-L3 local pairs)
    "aayl49": ("NA", "SARSCoV2_HR2", "Viral", "AAYL49", None),
    "aayl49_ML": ("NA", "SARSCoV2_HR2", "Viral", "AAYL49", None),
    "aayl50_LC": ("NA", "SARSCoV2_HR2", "Viral", "AAYL50", None),
    "aayl51": ("NA", "SARSCoV2_HR2", "Viral", "AAYL51", None),
    "aayl52_LC": ("NA", "SARSCoV2_HR2", "Viral", "AAYL52", None),
}

_SABDAB_TERM_MAP: dict[str, str] = {
    **{t: "HIV_gp120" for t in (
        "envelope glycoprotein gp120",
        "exterior membrane glycoprotein(gp120)",
        "envelope glycoprotein gp120 of hiv-1 clade c",
        "clade a/e 93th057 hiv-1 gp120 core",
        "426c.tm1deltav1-3 gp120",
        "93th057 hiv-1 gp120 core",
        "outer domain of hiv-1 gp120 (ker2018 od4.2.2)",
        "hiv-1 hxbc2 gp120 core",
        "glycoprotein 120",
        "hiv-1 clade g strain x2088 gp120",
        "hiv-1 gp120 core",
        "gp120",
        "germline-targeting hiv-1 gp120 engineered outer domain eod-gt8",
        "germline-targeting hiv-1 gp120 engineered outer domain,eod-gt6",
    )},
    "envelope glycoprotein gp160": "HIV_gp160",
    "gp41 helix": "HIV_gp41",
    "10e8 epitope scaffold t117v2": "HIV_gp41",
    "4e10 epitope scaffold t117": "HIV_gp41",
    "spike glycoprotein": "SARSCoV2_RBD",
    "mers-cov rbd": "MERSCoV_RBD",
    "hen egg white lysozyme": "HEWL",
    "lysozyme": "HEWL",
    "lysozyme c": "HEWL",
    "receptor tyrosine-protein kinase erbb-2": "HER2_ECD",
    "envelope glycoprotein b": "HCMV-gB",
    "integrin alpha-l": "ITGAL",
    "interferon alpha-2": "IFNA2",
    "hemagglutinin": "Influenza_AH1",
    "hemagglutinin ha1 chain": "Influenza_AH1",
    "hemagglutinin ha1": "Influenza_AH1",
    "hemagglutinin (ha1 chain)": "Influenza_AH1",
    "protein (hemagglutinin (ha1 chain))": "Influenza_AH1",
    "neuraminidase": "Influenza_Neuraminidase",
    "n9 neuraminidase": "Influenza_Neuraminidase",
    "influenza a subtype n9 neuraminidase": "Influenza_Neuraminidase",
    "angiopoietin-2": "ANG2",
    "botulinum neurotoxin type a": "Botulinum_Neurotoxin_A",
    "c-c motif chemokine 2": "CCL2",
    "small inducible cytokine a2": "CCL2",
    "epidermal growth factor receptor": "EGFR",
    "interferon-gamma receptor alpha chain": "IFNGR1",
    "interleukin-13": "IL13",
    "programmed cell death protein 1": "PD1",
    "vascular endothelial growth factor": "VEGF",
    "vascular endothelial growth factor a": "VEGF",
    "vascular endothelial growth factor b": "VEGF",
    "interleukin-17a": "IL17A",
    "activin receptor type-2b": "ACVR2B",
    "complement c5 beta chain": "C5_MG7",
    "coagulation factor xi": "FXI",
    "thymic stromal lymphopoietin": "TSLP",
}

_ABCOV_SPIKE_SEQS: dict[str, tuple[str, str]] = {
    "SARS-CoV-2": (
        "SARSCoV2",
        "MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAIHVSGTNGTKRFDNPVLPFNDGVYFASTEKSNIIRGWIFGTTLDSKTQSLLIVNNATNVVIKVCEFQFCNDPFLGVYYHKNNKSWMESEFRVYSSANNCTFEYVSQPFLMDLEGKQGNFKNLREFVFKNIDGYFKIYSKHTPINLVRDLPQGFSALEPLVDLPIGINITRFQTLLALHRSYLTPGDSSSGWTAGAAAYYVGYLQPRTFLLKYNENGTITDAVDCALDPLSETKCTLKSFTVEKGIYQTSNFRVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNFNFNGLTGTGVLTESNKKFLPFQQFGRDIADTTDAVRDPQTLEILDITPCSFGGVSVITPGTNTSNQVAVLYQDVNCTEVPVAIHADQLTPTWRVYSTGSNVFQTRAGCLIGAEHVNNSYECDIPIGAGICASYQTQTNSPRRARSVASQSIIAYTMSLGAENSVAYSNNSIAIPTNFTISVTTEILPVSMTKTSVDCTMYICGDSTECSNLLLQYGSFCTQLNRALTGIAVEQDKNTQEVFAQVKQIYKTPPIKDFGGFNFSQILPDPSKPSKRSFIEDLLFNKVTLADAGFIKQYGDCLGDIAARDLICAQKFNGLTVLPPLLTDEMIAQYTSALLAGTITSGWTFGAGAALQIPFAMQMAYRFNGIGVTQNVLYENQKLIANQFNSAIGKIQDSLSSTASALGKLQDVVNQNAQALNTLVKQLSSNFGAISSVLNDILSRLDKVEAEVQIDRLITGRLQSLQTYVTQQLIRAAEIRASANLAATKMSECVLGQSKRVDFCGKGYHLMSFPQSAPHGVVFLHVTYVPAQEKNFTTAPAICHDGKAHFPREGVFVSNGTHWFVTQRNFYEPQIITTDNTFVSGNCDVVIGIVNNTVYDPLQPELDSFKEELDKYFKNHTSPDVDLGDISGINASVVNIQKEIDRLNEVAKNLNESLIDLQELGKYEQYIKWPWYIWLGFIAGLIAIVMVTIMLCCMTSCCSCLKGCCSCGSCCKFDEDDSEPVLKGVKLHYT",
    ),
    "SARS-CoV": (
        "SARSCoV",
        "MFIFLLFLTLTSGSDLDRCTTFDDVQAPNYTQHTSSMRGVYYPDEIFRSDTLYLTQDLFLPFYSNVTGFHTINHTFGNPVIPFKDGIYFAATEKSNVVRGWVFGSTMNNKSQSVIIINNSTNVVIRACNFELCDNPFFAVSKPMGTQTHTMIFDNAFNCTFEYISDAFSLDVSEKSGNFKHLREFVFKNKDGFLYVYKGYQPIDVVRDLPSGFNTLKPIFKLPLGINITNFRAILTAFSPAQDIWGTSAAAYFVGYLKPTTFMLKYDENGTITDAVDCSQNPLAELKCSVKSFEIDKGIYQTSNFRVVPSGDVVRFPNITNLCPFGEVFNATKFPSVYAWERKKISNCVADYSVLYNSTFFSTFKCYGVSATKLNDLCFSNVYADSFVVKGDDVRQIAPGQTGVIADYNYKLPDDFMGCVLAWNTRNIDATSTGNYNYKYRYLRHGKLRPFERDISNVPFSPDGKPCTPPALNCYWPLNDYGFYTTTGIGYQPYRVVVLSFELLNAPATVCGPKLSTDLIKNQCVNFNFNGLTGTGVLTPSSKRFQPFQQFGRDVSDFTDSVRDPKTSEILDISPCSFGGVSVITPGTNASSEVAVLYQDVNCTDVSTAIHADQLTPAWRIYSTGNNVFQTQAGCLIGAEHVDTSYECDIPIGAGICASYHTVSLLRSTSQKSIVAYTMSLGADSSIAYSNNTIAIPTNFSISITTEVMPVSMAKTSVDCNMYICGDSTECANLLLQYGSFCTQLNRALSGIAAEQDRNTREVFAQVKQMYKTPTLKYFGGFNFSQILPDPLKPTKRSFIEDLLFNKVTLADAGFMKQYGECLGDINARDLICAQKFNGLTVLPPLLTDDMIAAYTAALVSGTATAGWTFGAGAALQIPFAMQMAYRFNGIGVTQNVLYENQKQIANQFNKAISQIQESLTTTSTALGKLQDVVNQNAQALNTLVKQLSSNFGAISSVLNDILSRLDKVEAEVQIDRLITGRLQSLQTYVTQQLIRAAEIRASANLAATKMSECVLGQSKRVDFCGKGYHLMSFPQAAPHGVVFLHVTYVPSQERNFTTAPAICHEGKAYFPREGVFVFNGTSWFITQRNFFSPQIITTDNTFVSGNCDVVIGIINNTVYDPLQPELDSFKEELDKYFKNHTSPDVDLGDISGINASVVNIQKEIDRLNEVAKNLNESLIDLQELGKYEQYIKWPWYVWLGFIAGLIAIVMVTILLCCMTSCCSCLKGACSCGSCCKFDEDDSEPVLKGVKLHYT",
    ),
    "MERS": (
        "MERSCoV",
        "MIHSVFLLMFLLTPTESYVDVGPDSVKSACIEVDIQQTFFDKTWPRPIDVSKADGIIYPQGRTYSNITITYQGLFPYQGDHGDMYVYSAGHATGTTPQKLFVANYSQDVKQFANGFVVRIGAAANSTGTVIISPSTSATIRKIYPAFMLGSSVGNFSDGKMGRFFNHTLVLLPDGCGTLLRAFYCILEPRSGNHCPAGNSYTSFATYHTPATDCSDGNYNRNASLNSFKEYFNLRNCTFMYTYNITEDEILEWFGITQTAQGVHLFSSRYVDLYGGNMFQFATLPVYDTIKYYSIIPHSIRSIQSDRKAWAAFYVYKLQPLTFLLDFSVDGYIRRAIDCGFNDLSQLHCSYESFDVESGVYSVSSFEAKPSGSVVEQAEGVECDFSPLLSGTPPQVYNFKRLVFTNCNYNLTKLLSLFSVNDFTCSQISPAAIASNCYSSLILDYFSYPLSMKSDLSVSSAGPISQFNYKQSFSNPTCLILATVPHNLTTITKPLKYSYINKCSRFLSDDRTEVPQLVNANQYSPCVSIVPSTVWEDGDYYRKQLSPLEGGGWLVASGSTVAMTEQLQMGFGITVQYGTDTNSVCPKLEFANDTKIASQLGNCVEYSLYGVSGRGVFQNCTAVGVRQQRFVYDAYQNLVGYYSDDGNYYCLRACVSVPVSVIYDKETKTHATLFGSVACEHISSTMSQYSRSTRSMLKRRDSTYGPLQTPVGCVLGLVNSSLFVEDCKLPLGQSLCALPDTPSTLTPRSVRSVPGEMRLASIAFNHPIQVDQLNSSYFKLSIPTNFSFGVTQEYIQTTIQKVTVDCKQYVCNGFQKCEQLLREYGQFCSKINQALHGANLRQDDSVRNLFASVKSSQSSPIIPGFGGDFNLTLLEPVSISTGSRSARSAIEDLLFDKVTIADPGYMQGYDDCMQQGPASARDLICAQYVAGYKVLPPLMDVNMEAAYTSSLLGSIAGVGWTAGLSSFAAIPFAQSIFYRLNGVGITQQVLSENQKLIANKFNQALGAMQTGFTTTNEAFHKVQDAVNNNAQALSKLASELSNTFGAISASIGDIIQRLDVLEQDAQIDRLINGRLTTLNAFVAQQLVRSESAALSAQLAKDKVNECVKAQSKRSGFCGQGTHIVSFVVNAPNGLYFMHVGYYPSNHIEVVSAYGLCDAANPTNCIAPVNGYFIKTNNTRIVDEWSYTGSSFYAPEPITSLNTKYVAPQVTYQNISTNLPPPLLGNSTGIDFQDELDEFFKNVSTSIPNFGSLTQINTTLLDLTYEMLSLQQVVKALNESYIDLKELGNYTYYNKWPWYIWLGFIAGLVALALCVFFILCCTGCGTNCMGKLKCNRCCDRYEEYDLEPHKVHVH",
    ),
    "WIV1": (
        "WIV1CoV",
        "MKLLVLVFATLVSSYTIEKCLDFDDRTPPANTQFLSSHRGVYYPDDIFRSNVLHLVQDHFLPFDSNVTRFITFGLNFDNPIIPFKDGIYFAATEKSNVIRGWVFGSTMNNKSQSVIIMNNSTNLVIRACNFELCDNPFFVVLKSNNTQIPSYIFNNAFNCTFEYVSKDFNLDLGEKPGNFKDLREFVFRNKDGFLHVYSGYQPISAASGLPTGFNALKPIFKLPLGINITNFRTLLTAFPPRPDYWGTSAAAYFVGYLKPTTFMLKYDENGTITDAVDCSQNPLAELKCSVKSFEIDKGIYQTSNFRVAPSKEVVRFPNITNLCPFGEVFNATTFPSVYAWERKRISNCVADYSVLYNSTSFSTFKCYGVSATKLNDLCFSNVYADSFVVKGDDVRQIAPGQTGVIADYNYKLPDDFTGCVLAWNTRNIDATQTGNYNYKYRSLRHGKLRPFERDISNVPFSPDGKPCTPPAFNCYWPLNDYGFYITNGIGYQPYRVVVLSFELLNAPATVCGPKLSTDLIKNQCVNFNFNGLTGTGVLTPSSKRFQPFQQFGRDVSDFTDSVRDPKTSEILDISPCSFGGVSVITPGTNTSSEVAVLYQDVNCTDVPVAIHADQLTPSWRVHSTGNNVFQTQAGCLIGAEHVDTSYECDIPIGAGICASYHTVSSLRSTSQKSIVAYTMSLGADSSIAYSNNTIAIPTNFSISITTEVMPVSMAKTSVDCNMYICGDSTECANLLLQYGSFCTQLNRALSGIAVEQDRNTREVFAQVKQMYKTPTLKDFGGFNFSQILPDPLKPTKRSFIEDLLFNKVTLADAGFMKQYGECLGDINARDLICAQKFNGLTVLPPLLTDDMIAAYTAALVSGTATAGWTFGAGAALQIPFAMQMAYRFNGIGVTQNVLYENQKQIANQFNKAISQIQESLTTTSTALGKLQDVVNQNAQALNTLVKQLSSNFGAISSVLNDILSRLDKVEAEVQIDRLITGRLQSLQTYVTQQLIRAAEIRASANLAATKMSECVLGQSKRVDFCGKGYHLMSFPQAAPHGVVFLHVTYVPSQERNFTTAPAICHEGKAYFPREGVFVFNGTSWFITQRNFFSPQIITTDNTFVSGSCDVVIGIINNTVYDPLQPELDSFKEELDKYFKNHTSPDVDLGDISGINASVVNIQKEIDRLNEVAKNLNESLIDLQELGKYEQYIKWPWYVWLGFIAGLIAIVMVTILLCCMTSCCSCLKGACSCGSCCKFDEDDSEPVLKGVKLHYT",
    ),
    "SHC014": (
        "SHC014CoV",
        "MKLLVLVFATLVSSYTIEKCLDFDDRTPPANTQFLSSHRGVYYPDDIFRSNVLHLVQDHFLPFDSNVTRFITFGLNFDNPIIPFRDGIYFAATEKSNVIRGWVFGSTMNNKSQSVIIMNNSTNLVIRACNFELCDNPFFVVLKSNNTQIPSYIFNNAFNCTFEYVSKDFNLDLGEKPGNFKDLREFVFRNKDGFLHVYSGYQPISAASGLPTGFNALKPIFKLPLGINITNFRTLLTAFPPRPDYWGTSAAAYFVGYLKPTTFMLKYDENGTITDAVDCSQNPLAELKCSVKSFEIDKGIYQTSNFRVAPSKEVVRFPNITNLCPFGEVFNATTFPSVYAWERKRISNCVADYSVLYNSTSFSTFKCYGVSATKLNDLCFSNVYADSFVVKGDDVRQIAPGQTGVIADYNYKLPDDFLGCVLAWNTNSKDSSTSGNYNYLYRWVRRSKLNPYERDLSNDIYSPGGQSCSAVGPNCYNPLRPYGFFTTAGVGHQPYRVVVLSFELLNAPATVCGPKLSTDLIKNQCVNFNFNGLTGTGVLTPSSKRFQPFQQFGRDVSDFTDSVRDPKTSEILDISPCSFGGVSVITPGTNTSSEVAVLYQDVNCTDVPVAIHADQLTPSWRVYSTGNNVFQTQAGCLIGAEHVDTSYECDIPIGAGICASYHTVSSLRSTSQKSIVAYTMSLGADSSIAYSNNTIAIPTNFSISITTEVMPVSMAKTSVDCNMYICGDSTECANLLLQYGSFCTQLNRALSGIAVEQDRNTREVFAQVKQMYKTPTLKDFGGFNFSQILPDPLKPTKRSFIEDLLFNKVTLADAGFMKQYGECLGDINARDLICAQKFNGLTVLPPLLTDDMIAAYTAALVSGTATAGWTFGAGAALQIPFAMQMAYRFNGIGVTQNVLYENQKQIANQFNKAISQIQESLTTTSTALGKLQDVVNQNAQALNTLVKQLSSNFGAISSVLNDILSRLDKVEAEVQIDRLITGRLQSLQTYVTQQLIRAAEIRASANLAATKMSECVLGQSKRVDFCGKGYHLMSFPQAAPHGVVFLHVTYVPSQERNFTTAPAICHEGKAYFPREGVFVFNGTSWFITQRNFFSPQIITTDNTFVSGSCDVVIGIINNTVYDPLQPELDSFKEELDKYFKNHTSPDVDLGDISGINASVVNIQKEIDRLNEVAKNLNESLIDLQELGKYEQYIKWPWYVWLGFIAGLIAIVMVTILLCCMTSCCSCLKGACSCGSCCKFDEDDSEPVLKGVKLHYT",
    ),
}


def curate_abbibench_kd(paths: RepoPaths) -> pd.DataFrame:
    """Curate the 11 AbBiBench KD panels directly from `Raw/*_benchmarking_data.csv`.

    Fixes two major upstream AbBiBench bugs:
    1. Double-log10 bug on `aayl49..52` (`SARSCoV2_HR2`): `Pred_affinity` in AlphaSeq is ALREADY
       `log10(Kd [nM])`, so `raw_binding_score = 9.0 - log10(log10_Kd_nM)`. We invert via
       `log10_Kd_nM = 10 ** (9.0 - raw_binding_score)` and clamp censored `log10_Kd_nM < 5.0`.
    2. Unfiltered censored Tite-seq floor (`raw_binding_score == 7.0` on `3gbn_h1`, `3gbn_h9`,
       `4fqi_h1` and `6.0` on `4fqi_h3`).
    """
    catalog = load_antigen_catalog(paths.metadata)
    raw_dir = paths.raw / "AbBibench" / "Raw"
    frames = []

    for cid, (pdb_id, ag_name, ag_type, wt_ab, floor_val) in ABBIBENCH_KD_PANELS.items():
        df = pd.read_csv(raw_dir / f"{cid}_benchmarking_data.csv")
        bs = pd.to_numeric(df["binding_score"], errors="coerce")
        df = df[bs.notna()].copy()
        bs = bs.loc[df.index]

        if floor_val is not None:
            keep = bs > floor_val
            df = df.loc[keep].copy()
            bs = bs.loc[keep]

        if cid.startswith("aayl"):
            # Invert AbBiBench double-log10 transformation: raw_bs = 9 - log10(log10(Kd_nM))
            log10_kd_nm = 10.0 ** (9.0 - bs.to_numpy(dtype=float))
            # Filter out weak/undetected AlphaSeq floor (>= 100,000 nM = 100 uM)
            valid = np.isfinite(log10_kd_nm) & (log10_kd_nm < 5.0)
            df = df.iloc[valid].copy()
            log10_kd_nm = log10_kd_nm[valid]
            kd_nm = 10.0 ** log10_kd_nm
            pkd = 9.0 - log10_kd_nm
        else:
            pkd = bs.to_numpy(dtype=float)
            kd_nm = 10.0 ** (9.0 - pkd)

        _, ag_seq = catalog[ag_name]
        hs = [trim_fab_to_fv(h, l)[0] for h, l in zip(df["heavy_chain_seq"], df["light_chain_seq"])]
        ls = [trim_fab_to_fv(h, l)[1] for h, l in zip(df["heavy_chain_seq"], df["light_chain_seq"])]

        part = pd.DataFrame(
            {
                "Ab_name": [f"{wt_ab}_{cid}_{i}" for i in range(len(df))],
                "WT_Abname": wt_ab,
                "WT_PDB": pdb_id,
                "Ag_name": ag_name,
                "Ag_type": ag_type,
                "raw_value": kd_nm,
                "binding_score": pkd,
                "Ab_heavy_chain_seq": hs,
                "Ab_light_chain_seq": ls,
                "Ag_seq": ag_seq,
                "Ag_epitope_restrictions": "HR2_peptide" if ag_name == "SARSCoV2_HR2" else "None",
            }
        )
        frames.append(part)

    merged = pd.concat(frames, ignore_index=True)
    return finalize_source_table(
        merged,
        dataset_name="AbBibench",
        modality="KD",
        raw_unit="nM",
        min_ag_len=10,
    )


def curate_skempiv2(paths: RepoPaths) -> pd.DataFrame:
    ag_map, ty_map, ab_map = load_pdb_mapping(paths.metadata)
    df = pd.read_csv(paths.raw / "SKEMPIv2" / "SKEMPI_preprocessed.csv")
    df["WT_PDB"] = df["WT_PDB"].astype(str).str.strip().str.upper()
    kd_nm = pd.to_numeric(df["Affinity_Kd [nM]"], errors="coerce")
    df = df[kd_nm.notna() & (kd_nm > 0)].copy()
    kd_nm = kd_nm.loc[df.index].to_numpy(dtype=float)

    wt_ab = df["WT_PDB"].map(ab_map).fillna(df["WT_PDB"])
    ag_name = df["WT_PDB"].map(ag_map).fillna(df["WT_PDB"])
    ag_type = df["WT_PDB"].map(ty_map).fillna("Non-Viral")

    hs = [trim_fab_to_fv(h, l)[0] for h, l in zip(df["Ab_heavy_chain_seq"], df["Ab_light_chain_seq"])]
    ls = [trim_fab_to_fv(h, l)[1] for h, l in zip(df["Ab_heavy_chain_seq"], df["Ab_light_chain_seq"])]

    out = pd.DataFrame(
        {
            "Ab_name": [f"{ab}-{i + 1:03d}" for i, ab in enumerate(wt_ab)],
            "WT_Abname": wt_ab.values,
            "WT_PDB": df["WT_PDB"].values,
            "Ag_name": ag_name.values,
            "Ag_type": ag_type.values,
            "raw_value": kd_nm,
            "binding_score": kd_nm_to_pkd(kd_nm),
            "Ab_heavy_chain_seq": hs,
            "Ab_light_chain_seq": ls,
            "Ag_seq": [trim_antigen_tags(s) for s in df["Ag_seq"]],
            "Ag_epitope_restrictions": "None",
        }
    )
    return finalize_source_table(out, dataset_name="SKEMPIv2", modality="KD", raw_unit="nM")


def curate_sabdab(paths: RepoPaths) -> pd.DataFrame:
    df = pd.read_csv(paths.raw / "SabDab" / "SabDab_preprocessed.csv", sep="\t")
    ssum = pd.read_csv(paths.raw / "SabDab" / "SabDab_aff_summary.tsv", sep="\t")
    pdb_to_species = (
        ssum.dropna(subset=["pdb", "antigen_species"])
        .assign(pdb_up=lambda d: d["pdb"].astype(str).str.strip().str.upper())
        .drop_duplicates("pdb_up")
        .set_index("pdb_up")["antigen_species"]
        .to_dict()
    )

    parsed = [parse_numeric_measurement(v) for v in df["Affinity_Kd [nM]"]]
    vals = np.array([p[0] for p in parsed], dtype=float)
    ops = np.array([p[1] for p in parsed], dtype=object)
    keep = (ops == "=") & np.isfinite(vals) & (vals > 0)
    df = df.loc[keep].copy().reset_index(drop=True)
    kd_nm = vals[keep]

    pdb_up = df["bound_AbAg_PDB_ID"].astype(str).str.strip().str.upper()
    ag_names: list[str] = []
    ag_types: list[str] = []
    species_list: list[str | None] = []

    for pdb_id, det in zip(pdb_up, df["Ag_name_details"]):
        sp = pdb_to_species.get(pdb_id)
        species_list.append(sp)
        if pdb_id in SABDAB_PDB_OVERRIDES:
            ov_name, ov_type = SABDAB_PDB_OVERRIDES[pdb_id]
            ag_names.append(ov_name)
            ag_types.append(ov_type)
            continue
        det_clean = str(det).strip().lower()
        mapped = _SABDAB_TERM_MAP.get(det_clean)
        if mapped is None:
            mapped = str(det).strip().replace(" ", "_").split(",")[0]
        ag_names.append(mapped)
        ag_types.append("Non-Viral")

    hs = [trim_fab_to_fv(h, l)[0] for h, l in zip(df["Ab_heavy_chain_seq"], df["Ab_light_chain_seq"])]
    ls = [trim_fab_to_fv(h, l)[1] for h, l in zip(df["Ab_heavy_chain_seq"], df["Ab_light_chain_seq"])]

    out = pd.DataFrame(
        {
            "Ab_name": df["Ab_name"].astype(str).values,
            "WT_Abname": [str(a).split("_")[0] for a in df["Ab_name"]],
            "WT_PDB": pdb_up.values,
            "Ag_name": ag_names,
            "Ag_type": ag_types,
            "antigen_species": species_list,
            "raw_value": kd_nm,
            "binding_score": kd_nm_to_pkd(kd_nm),
            "Ab_heavy_chain_seq": hs,
            "Ab_light_chain_seq": ls,
            "Ag_seq": [trim_antigen_tags(s) for s in df["Ag_seq"]],
            "Ag_epitope_restrictions": df["Ag_epitope_restrictions"].fillna("None").astype(str).values,
        }
    )
    return finalize_source_table(out, dataset_name="SabDab", modality="KD", raw_unit="nM")


def curate_origin1(paths: RepoPaths) -> pd.DataFrame:
    catalog = load_antigen_catalog(paths.metadata)
    df = pd.read_csv(paths.raw / "Origin1" / "lead_optimization_spr_data.csv")
    kd_nm = pd.to_numeric(df["KD (nM)"], errors="coerce")
    df = df[kd_nm.notna() & (kd_nm > 0)].copy().reset_index(drop=True)
    kd_nm = kd_nm.loc[kd_nm.notna() & (kd_nm > 0)].to_numpy(dtype=float)

    raw_targets = [str(t).split("-")[0] for t in df["Target"]]
    ag_names = ["COL6A3_C5" if t == "COL6A3" else t for t in raw_targets]
    wt_abs = df["Target"].astype(str) + "_AbSci"
    ab_names = [
        wt if str(m) == "WT" else f"{wt}-{i}"
        for i, (wt, m) in enumerate(zip(wt_abs, df["Mutation"]))
    ]

    out = pd.DataFrame(
        {
            "Ab_name": ab_names,
            "WT_Abname": wt_abs.values,
            "WT_PDB": "NA",
            "Ag_name": ag_names,
            "Ag_type": "Non-Viral",
            "raw_value": kd_nm,
            "binding_score": kd_nm_to_pkd(kd_nm),
            "Ab_heavy_chain_seq": df["Heavy Chain"].values,
            "Ab_light_chain_seq": df["Light Chain"].values,
            "Ag_seq": [catalog[a][1] for a in ag_names],
            "Ag_epitope_restrictions": "None",
        }
    )
    return finalize_source_table(out, dataset_name="Origin1", modality="KD", raw_unit="nM")


def curate_igdesign(paths: RepoPaths) -> pd.DataFrame:
    df = pd.read_csv(paths.raw / "IgDesign" / "IgDesign.csv", sep="\t")
    kd_nm = pd.to_numeric(df["Affinity_Kd [nM]"], errors="coerce")
    df = df[kd_nm.notna() & (kd_nm > 0)].copy().reset_index(drop=True)
    kd_nm = pd.to_numeric(df["Affinity_Kd [nM]"], errors="coerce").to_numpy(dtype=float)

    out = pd.DataFrame(
        {
            "Ab_name": df["Ab_name"].astype(str).values,
            "WT_Abname": df["WT_Abname"].astype(str).values,
            "WT_PDB": "NA",
            "Ag_name": df["Ag_name"].astype(str).values,
            "Ag_type": df["Ag_type"].astype(str).values,
            "raw_value": kd_nm,
            "binding_score": kd_nm_to_pkd(kd_nm),
            "Ab_heavy_chain_seq": df["Ab_heavy_chain_seq"].values,
            "Ab_light_chain_seq": df["Ab_light_chain_seq"].values,
            "Ag_seq": [trim_antigen_tags(s) for s in df["Ag_seq"]],
            "Ag_epitope_restrictions": "None",
        }
    )
    return finalize_source_table(out, dataset_name="IgDesign", modality="KD", raw_unit="nM")


def curate_her2(paths: RepoPaths) -> pd.DataFrame:
    catalog = load_antigen_catalog(paths.metadata)
    _, ag_seq = catalog["HER2_ECD"]
    lseq_wt = "DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSRSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIK"
    frh1, cdrh1 = "EVQLVESGGGLVQPGGSLRLSCAAS", "GFNIKDTY"
    frh2, cdrh2 = "IHWVRQAPGKGLEWVAR", "IYPTNGYT"
    frh3, frh4 = "RYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYC", "WGQGTLVTVSS"

    df1 = pd.read_csv(paths.raw / "HER2" / "HER2_binders_control.csv")
    df1 = df1[pd.to_numeric(df1["KD (nM)"], errors="coerce").notna()]
    df2 = pd.read_csv(paths.raw / "HER2" / "HER2_binders_zeroshot.csv")
    df2 = df2[pd.to_numeric(df2["KD (nM)"], errors="coerce").notna()]

    h_seqs, kds = [], []
    for _, r in df1.iterrows():
        h_seqs.append(frh1 + str(r["HCDR1"]) + frh2 + str(r["HCDR2"]) + frh3 + str(r["HCDR3"]) + frh4)
        kds.append(float(r["KD (nM)"]))
    for _, r in df2.iterrows():
        h_seqs.append(frh1 + cdrh1 + frh2 + cdrh2 + frh3 + str(r["HCDR3"]) + frh4)
        kds.append(float(r["KD (nM)"]))

    kd_arr = np.asarray(kds, dtype=float)
    out = pd.DataFrame(
        {
            "Ab_name": [f"AbHER-{i + 1:04d}" for i in range(len(h_seqs))],
            "WT_Abname": "Trastuzumab",
            "WT_PDB": "1N8Z",
            "Ag_name": "HER2_ECD",
            "Ag_type": "Non-Viral",
            "raw_value": kd_arr,
            "binding_score": kd_nm_to_pkd(kd_arr),
            "Ab_heavy_chain_seq": h_seqs,
            "Ab_light_chain_seq": lseq_wt,
            "Ag_seq": ag_seq,
            "Ag_epitope_restrictions": "None",
        }
    )
    return finalize_source_table(out, dataset_name="HER2", modality="KD", raw_unit="nM")


def curate_ova(paths: RepoPaths) -> pd.DataFrame:
    catalog = load_antigen_catalog(paths.metadata)
    _, ag_seq = catalog["OVA"]
    df = pd.read_csv(paths.raw / "OVA" / "OVA_processed.csv", sep="\t")
    kd_m = pd.to_numeric(df["KD (M)"], errors="coerce")
    df = df[kd_m.notna() & (kd_m > 0)].copy().reset_index(drop=True)
    kd_nm = (kd_m.loc[kd_m.notna() & (kd_m > 0)] * 1e9).to_numpy(dtype=float)

    epis = [
        "None" if pd.isna(e) else f"{c}-{e}"
        for c, e in zip(df["Conformation"], df["Epitope"])
    ]
    out = pd.DataFrame(
        {
            "Ab_name": [f"AbOVA-{i + 1:03d}" for i in range(len(df))],
            "WT_Abname": [f"AbOVA-{i + 1:03d}" for i in range(len(df))],
            "WT_PDB": "NA",
            "Ag_name": "OVA",
            "Ag_type": "Non-Viral",
            "raw_value": kd_nm,
            "binding_score": kd_nm_to_pkd(kd_nm),
            "Ab_heavy_chain_seq": df["Hseq"].values,
            "Ab_light_chain_seq": df["Lseq"].values,
            "Ag_seq": ag_seq,
            "Ag_epitope_restrictions": epis,
        }
    )
    return finalize_source_table(out, dataset_name="OVA", modality="KD", raw_unit="nM")


def curate_abcov_kd(paths: RepoPaths) -> pd.DataFrame:
    """Curate exact uncensored paired-chain KD measurements from AbCoV."""
    abcov_df = pd.read_csv(paths.raw / "AbCoV" / "AbCoV_AbAg_affinity.csv")
    details = pd.read_csv(paths.raw / "AbCoV" / "AbCoV_antibody_details.csv")

    h_map, l_map, pdb_map = {}, {}, {}
    for _, r in details.iterrows():
        nm = str(r["Name"]).replace(" ", "")
        h_map[nm] = r["VH or VHH"]
        l_map[nm] = r["VL"]
        pdb_map[nm] = str(r["Structures"]).replace("https://www.rcsb.org/structure/", "").strip()

    rows = []
    for _, r in abcov_df.iterrows():
        ab_nm = str(r["Antibody name"]).replace(" ", "")
        if ab_nm not in h_map:
            continue
        fval, op = parse_numeric_measurement(r["Binding Affinity (KD)"])
        if op != "=" or not np.isfinite(fval):
            continue

        neut = str(r["Neutralizes"])
        ag_prefix, ag_seq = None, None
        for key, (pref, seq) in _ABCOV_SPIKE_SEQS.items():
            if key in neut:
                ag_prefix, ag_seq = pref, seq
                break
        if ag_prefix is None:
            continue

        epi_raw = str(r["Viral protein:epitope"])
        epi = "None"
        for tag in ("non-RBD", "RBD", "S2", "NTD", "S1"):
            if tag in epi_raw:
                epi = tag.replace("-", "")
                break

        hs, ls = trim_fab_to_fv(h_map[ab_nm], l_map[ab_nm])
        rows.append(
            {
                "Ab_name": f"AbCoV-{ab_nm}",
                "WT_Abname": f"AbCoV-{ab_nm}",
                "WT_PDB": pdb_map.get(ab_nm, "NA"),
                "Ag_name": f"{ag_prefix}_{epi}",
                "Ag_type": "Viral",
                "raw_value": float(fval),
                "binding_score": float(kd_nm_to_pkd(fval)),
                "Ab_heavy_chain_seq": hs,
                "Ab_light_chain_seq": ls,
                "Ag_seq": ag_seq,
                "Ag_epitope_restrictions": epi,
            }
        )

    out = pd.DataFrame(rows)
    return finalize_source_table(out, dataset_name="AbCoV", modality="KD", raw_unit="nM")

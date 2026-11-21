# AbRank v2.0.0 — Curated Dataset & 3-State Split Statistics (`m = 1.0`, `m_neutral = 0.3`)

## 1. Core Paired-Chain `KD` Assay Inventory
- **Total deduplicated `(VH, VL, Ag)` $K_d$ assays**: **120,721**
- **Unique paired antibodies (`AbID`)**: **111,743** across **262** clonal lineages (`<80%` Fv identity)
- **Unique antigen constructs (`AgID`)**: **543** across **148** canonical antigen groups (**24** Viral, **124** Non-Viral)

| Source Dataset | Unique Assays | Unique Antibodies | Canonical Antigen Groups | Target Types |
| :--- | ---: | ---: | ---: | :--- |
| `AbBibench` | 115,540 | 106,976 | 6 | Non-Viral, Viral |
| `SKEMPIv2` | 913 | 550 | 19 | Non-Viral, Viral |
| `HER2` | 758 | 758 | 1 | Non-Viral |
| `Kirby` | 611 | 611 | 1 | Viral |
| `AIntibody` | 550 | 550 | 1 | Viral |
| `Peterson` | 544 | 544 | 1 | Viral |
| `Kothiwal` | 364 | 364 | 10 | Non-Viral |
| `Origin1` | 354 | 354 | 4 | Non-Viral |
| `IgDesign` | 319 | 319 | 7 | Non-Viral |
| `AbCoV` | 304 | 296 | 5 | Viral |
| `SabDab` | 238 | 215 | 123 | Non-Viral, Viral |
| `OVA` | 89 | 89 | 1 | Non-Viral |
| `Shanker` | 62 | 57 | 1 | Viral |
| `Abbate` | 42 | 42 | 1 | Non-Viral |
| `Porebski` | 26 | 26 | 1 | Non-Viral |
| `Germinal` | 7 | 7 | 2 | Non-Viral |

## 2. Cross-Modal, Transfer & VHH Assay Tracks

| Track | Dataset (Modality) | Validated Assays | Antigens / Groups |
| :--- | :--- | ---: | ---: |
| Cross-Modal | `AbAgym (DMS)` | 26,883 | 11 |
| Cross-Modal | `AbBibench (Enrichment)` | 38,754 | 4 |
| Cross-Modal | `AbDesign (DDG)` | 490 | 13 |
| Cross-Modal | `AbDesign (ELISA)` | 672 | 7 |
| Cross-Modal | `Kothiwal (EC50)` | 345 | 10 |
| Cross-Modal | `NbBench (Enrichment)` | 12,735 | 1 |
| Transfer | `AbCoV (IC50)` | 1,294 | 5 |
| Transfer | `CATNAP (IC50)` | 39,779 | 2,283 |
| Transfer | `RBD_Escape_Bloom (Escape)` | 188,523 | 2,172 |
| Transfer | `Shanker (IC50)` | 142 | 3 |
| VHH (Single-Domain) | `AbCoV_VHH (KD)` | 146 | 2 |
| VHH (Single-Domain) | `Adaptyv_EGFR_VHH (KD)` | 5 | 1 |
| VHH (Single-Domain) | `Bennett_VHH (KD)` | 1 | 1 |
| VHH (Single-Domain) | `BoltzGen_VHH (KD)` | 37 | 10 |
| VHH (Single-Domain) | `Germinal_VHH (KD)` | 20 | 4 |
| VHH (Single-Domain) | `Porebski_VHH (KD)` | 13 | 1 |
| VHH (Single-Domain) | `SKEMPIv2_VHH (KD)` | 6 | 1 |
| VHH (Single-Domain) | `SabDab_VHH (KD)` | 70 | 34 |

## 3. Zero-Leakage 3-State Pairwise Splits (`{-1: Depleted, 0: Neutral, +1: Improved}`) & Unfiltered Panels

| Split Tier / Track | Partition | Total Pairs | Impactful (`±1`) | Neutral (`0`) | `Global` | `L1` | `L2` | `L3` | `Candidate` | Groups | `+1` Frac (`±1`) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `unseen_ag` | `train` | 21,836 | 16,448 | 5,388 | 11,756 | 2,845 | 3,999 | 3,236 | 0 | 15 | 0.500 |
| `unseen_ag` | `val` | 1,906 | 925 | 981 | 275 | 393 | 704 | 534 | 0 | 7 | 0.498 |
| `unseen_ag` | `test` | 1,010 | 538 | 472 | 454 | 133 | 214 | 209 | 0 | 8 | 0.500 |
| `unseen_ag` | `test_candidates` | 583 | 178 | 405 | 0 | 0 | 0 | 0 | 583 | 5 | 0.500 |
| `cross_modal` | `test` | 29,879 | 12,487 | 17,392 | 0 | 12,282 | 15,089 | 2,358 | 150 | 39 | 0.500 |
| `transfer_ic50` | `test` | 2,967 | 1,345 | 1,622 | 0 | 988 | 1,453 | 526 | 0 | 7 | 0.500 |
| `transfer_escape` | `test` | 8,336 | 4,275 | 4,061 | 0 | 621 | 7,715 | 0 | 0 | 4 | 0.500 |
| `vhh_ranking` | `test` | 813 | 534 | 279 | 293 | 4 | 9 | 12 | 495 | 19 | 0.498 |

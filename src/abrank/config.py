"""Configuration and repository path helpers for AbRank v2."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import yaml


def get_repo_root() -> Path:
    """Return the root directory of the AbRank repository."""
    return Path(__file__).resolve().parents[2]


@dataclass
class BenchmarkConfig:
    benchmark_version: str = "2.0.0"
    seed: int = 42
    margin_m: float = 1.0
    neutral_margin: float = 0.30
    neutral_pair_ratio: float = 1.0
    modality_margins: dict[str, float] = field(
        default_factory=lambda: {
            "KD": 1.0,
            "EC50": 1.0,
            "IC50": 1.0,
            "ELISA": 0.5,
            "DDG": 0.5,
            "Enrichment": 0.5,
            "DMS": 0.45,
            "Escape": 0.45,
        }
    )
    modality_neutral_margins: dict[str, float] = field(
        default_factory=lambda: {
            "KD": 0.30,
            "EC50": 0.30,
            "IC50": 0.30,
            "ELISA": 0.15,
            "DDG": 0.15,
            "Enrichment": 0.15,
            "DMS": 0.08,
            "Escape": 0.02,
        }
    )
    global_exclude_antigens: list[str] = field(default_factory=lambda: ["SARSCoV2_HR2"])
    global_num_pairs: int = 25000
    ab_occurrence_cap: int = 3
    local_ag_pair_cap: int = 500
    candidate_ag_pair_cap: int = 500
    transfer_ab_pair_cap: int = 15
    local_exhaustive_seq_limit: int = 2200
    local_sampling_budget: int = 1_500_000
    local_max_l1_per_study: int = 3_000
    split_fractions: dict[str, float] = field(
        default_factory=lambda: {"train": 0.75, "val": 0.10, "test": 0.15}
    )
    clustering: dict[str, float] = field(
        default_factory=lambda: {
            "unseen_ab_fv_identity": 0.80,
            "unseen_ag_identity": 0.50,
        }
    )

    @classmethod
    def load(cls, path: Path | str | None = None) -> "BenchmarkConfig":
        if path is None:
            path = get_repo_root() / "configs" / "benchmark.yaml"
        path = Path(path)
        if not path.exists():
            return cls()
        with open(path, "r", encoding="utf-8") as f:
            raw: dict[str, Any] = yaml.safe_load(f) or {}
        return cls(**{k: v for k, v in raw.items() if k in cls.__dataclass_fields__})


@dataclass(frozen=True)
class RepoPaths:
    root: Path
    configs: Path
    metadata: Path
    raw: Path
    processed: Path
    benchmark: Path
    sequences: Path
    splits: Path

    @classmethod
    def from_root(cls, root: Path | None = None) -> "RepoPaths":
        r = root or get_repo_root()
        bench = r / "data" / "benchmark"
        return cls(
            root=r,
            configs=r / "configs",
            metadata=r / "metadata",
            raw=r / "data" / "raw",
            processed=r / "data" / "processed",
            benchmark=bench,
            sequences=bench / "sequences",
            splits=bench / "splits",
        )

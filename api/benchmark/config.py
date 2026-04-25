from __future__ import annotations

import argparse
from dataclasses import dataclass, fields
from pathlib import Path


@dataclass
class BenchmarkConfig:
    min_ratings: int = 15
    max_users: int = 2000
    visible_ratio: float = 0.3
    profile_seed: int = 123
    group_size: int = 4
    group_seed: int = 42
    num_groups: int = 50
    ndcg_k: int = 10
    llm_model: str = "claude-haiku-4-5-20251001"
    llm_cache_dir: str = "../benchmark/cache/llm"
    results_dir: str = "../benchmark/results"
    groupfit_lambda: float = 0.3
    groupfit_beta: float = 0.5
    groupfit_alpha: float = 0.5
    use_msmarco: bool = False

    @classmethod
    def from_yaml(cls, path: Path | str | None = None) -> BenchmarkConfig:
        data: dict = {}
        if path is not None:
            p = Path(path)
            if p.exists():
                import yaml
                with p.open() as f:
                    data = yaml.safe_load(f) or {}
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    def apply_args(self, args: argparse.Namespace) -> None:
        known = {f.name for f in fields(self)}
        for key, val in vars(args).items():
            normalized = key.replace("-", "_")
            if normalized in known and val is not None:
                setattr(self, normalized, val)

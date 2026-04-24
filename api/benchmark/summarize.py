"""
Read benchmark result JSONs and print comparison tables.

Run: cd api && python -m benchmark.summarize --results-dir ../benchmark/results/
     cd api && python -m benchmark.summarize --ablate visible_ratio --method groupmatch_clustered
     cd api && python -m benchmark.summarize --compare-methods --visible-ratio 0.3
"""
import argparse
import json
from pathlib import Path


def load_results(results_dir: Path) -> list[dict]:
    results = []
    for path in sorted(results_dir.glob("*.json")):
        try:
            with path.open() as f:
                results.append(json.load(f))
        except (json.JSONDecodeError, KeyError):
            print(f"  Skipping malformed file: {path.name}")
    return results


def _row(label: str, mean: float, std: float, worst: float) -> str:
    return f"  {label:<32} | {mean:>10.4f} | {std:>7.4f} | {worst:>16.4f}"


def print_ablation_table(results: list[dict], ablate_key: str, method_filter: str | None) -> None:
    filtered = [
        r for r in results
        if method_filter is None
        or r.get("algorithm", r.get("method")) == method_filter
    ]
    if not filtered:
        print(f"No results for method={method_filter}")
        return

    groups: dict = {}
    for r in filtered:
        val = r["config"].get(ablate_key)
        groups.setdefault(val, []).append(r)

    label = method_filter or "all methods"
    print(f"\n# Ablation: {ablate_key} | method: {label}")
    header = f"  {'  ' + ablate_key:<30} | {'mean_NDCG':>10} | {'std':>7} | {'mean_worst_NDCG':>16}"
    print(header)
    print("-" * len(header))
    for key_val in sorted(groups.keys(), key=lambda x: (x is None, x)):
        entries = groups[key_val]
        avg_mean = sum(e["summary"]["mean_ndcg"] for e in entries) / len(entries)
        avg_std = sum(e["summary"]["std_ndcg"] for e in entries) / len(entries)
        avg_worst = sum(e["summary"]["mean_worst_user_ndcg"] for e in entries) / len(entries)
        print(_row(str(key_val), avg_mean, avg_std, avg_worst))


def print_method_comparison(results: list[dict], fixed_config: dict) -> None:
    filtered = [
        r for r in results
        if all(r["config"].get(k) == v for k, v in fixed_config.items())
    ]

    by_algorithm: dict[str, list[dict]] = {}
    for r in filtered:
        key = r.get("algorithm") or r.get("method", "unknown")
        by_algorithm.setdefault(key, []).append(r)

    config_str = ", ".join(f"{k}={v}" for k, v in fixed_config.items()) if fixed_config else "all configs"
    print(f"\n# Algorithm comparison | {config_str}")
    header = f"  {'algorithm':<32} | {'mean_NDCG':>10} | {'std':>7} | {'mean_worst_NDCG':>16}"
    print(header)
    print("-" * len(header))
    for method in sorted(by_algorithm.keys()):
        entries = by_algorithm[method]
        avg_mean = sum(e["summary"]["mean_ndcg"] for e in entries) / len(entries)
        avg_std = sum(e["summary"]["std_ndcg"] for e in entries) / len(entries)
        avg_worst = sum(e["summary"]["mean_worst_user_ndcg"] for e in entries) / len(entries)
        print(_row(method, avg_mean, avg_std, avg_worst))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="../benchmark/results")
    parser.add_argument("--ablate", help="Config key to ablate (e.g. visible_ratio, group_size)")
    parser.add_argument("--method", help="Filter to a single method")
    parser.add_argument("--compare-methods", action="store_true")
    parser.add_argument("--visible-ratio", type=float, default=None)
    parser.add_argument("--group-size", type=int, default=None)
    parser.add_argument("--ndcg-k", type=int, default=None)
    args = parser.parse_args()

    results_dir = Path(args.results_dir).resolve()
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return

    results = load_results(results_dir)
    if not results:
        print("No result files found.")
        return

    print(f"Loaded {len(results)} result file(s) from {results_dir}")

    if args.ablate:
        print_ablation_table(results, args.ablate, args.method)

    if args.compare_methods:
        fixed: dict = {}
        if args.visible_ratio is not None:
            fixed["visible_ratio"] = args.visible_ratio
        if args.group_size is not None:
            fixed["group_size"] = args.group_size
        if args.ndcg_k is not None:
            fixed["ndcg_k"] = args.ndcg_k
        print_method_comparison(results, fixed)

    if not args.ablate and not args.compare_methods:
        print_method_comparison(results, {})


if __name__ == "__main__":
    main()

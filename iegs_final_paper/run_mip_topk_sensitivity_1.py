
from __future__ import annotations

import copy
import csv
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

# Keep imports consistent with the original runners when launched from the
# project root.
sys.path.insert(0, os.getcwd())

import algorithm.mip_nd_master as mip_nd_master
import main as project_main
import model_distributed_nd_simulation


K_VALUES = (32, 64, 128, 256)
POOL_K = max(K_VALUES)
RESULT_ROOT = Path("results_mip_topk_sensitivity")
FULL_RUN_ROOT = RESULT_ROOT / "K_256_full"
POOL_CSV = RESULT_ROOT / "mip_topk_candidate_pool.csv"
SAMPLES_CSV = RESULT_ROOT / "mip_topk_resampling_samples.csv"
SUMMARY_CSV = RESULT_ROOT / "mip_topk_sensitivity_summary.csv"
METADATA_JSON = RESULT_ROOT / "mip_topk_sensitivity_metadata.json"


N_RESAMPLES = max(1, int(os.environ.get("MIP_TOPK_RESAMPLES", "1000")))
RESAMPLE_SEED = int(os.environ.get("MIP_TOPK_SAMPLE_SEED", "20260831"))
NEAR_OPTIMAL_REL_TOL = max(0.0, float(os.environ.get("MIP_TOPK_NEAR_TOL", "0.01")))

# Numerical equality tolerance used only for the exact-best hit indicator.
EXACT_SCORE_ATOL = 1e-9
EXACT_SCORE_RTOL = 1e-10

# Runtime collectors populated by monkey patches below.  Existing project files
# are never edited.
_COMMON_POOL: list[dict[str, Any]] = []
_RAW_RECORDS: dict[int, dict[str, Any]] = {}

_ORIGINAL_GENERATE = mip_nd_master.MIPNDOptimizer._generate_candidates
_ORIGINAL_SCORE_BATCH = mip_nd_master.MIPNDOptimizer._score_batch
_ORIGINAL_REFINE_BATCH_BEST = mip_nd_master.MIPNDOptimizer._refine_batch_best


def patch_lower_solver_to_model_distributed_nd() -> None:
    """Use exactly the same lower-level DND solver as run_distributed_main.py."""
    sim = model_distributed_nd_simulation.simulate_strategies_model_distributed_nd_batched
    project_main.simulate_strategies_batched = sim
    mip_nd_master.simulate_strategies_batched = sim


def _candidate_scale(candidate: dict[str, Any]) -> float:
    meta = candidate.get("meta", {}) or {}
    if "fdia_scale" in meta:
        return float(meta["fdia_scale"])
    # DoS-only is not used by this experiment, but keep the helper robust.
    return 1.0


def _patched_generate_candidates(self) -> list[dict]:
    """Generate the original K=512 pool once and retain an exact copy."""
    # 【修改标记 3】核心修改：所有 K 共用同一个 512 候选池。
    candidates = _ORIGINAL_GENERATE(self)

    if self.mode == "coordinated":
        if len(candidates) < POOL_K:
            raise RuntimeError(
                f"The common pool contains only {len(candidates)} candidates; "
                f"at least {POOL_K} are required for this sensitivity experiment."
            )

        # The optimizer should already return exactly mip_nd_top_k candidates.
        # Keep exactly the first POOL_K in the unlikely event that a custom
        # generator returns more.
        del _COMMON_POOL[:]
        _COMMON_POOL.extend(copy.deepcopy(candidates[:POOL_K]))
        return candidates[:POOL_K]

    return candidates


def _patched_refine_batch_best(self, *args, **kwargs):
    """Mark local-refinement scoring so it is not mistaken for raw pool data."""
    old_flag = bool(getattr(self, "_topk_sensitivity_in_refinement", False))
    self._topk_sensitivity_in_refinement = True
    try:
        return _ORIGINAL_REFINE_BATCH_BEST(self, *args, **kwargs)
    finally:
        self._topk_sensitivity_in_refinement = old_flag


def _patched_score_batch(self, result, offset: int):
    """Collect raw candidate responses while leaving original scoring unchanged."""
    scores = _ORIGINAL_SCORE_BATCH(self, result, offset)

    if (
        self.mode == "coordinated"
        and not bool(getattr(self, "_topk_sensitivity_in_refinement", False))
        and _COMMON_POOL
    ):
        for score, global_idx, info in scores:
            idx = int(global_idx)
            if idx < 0 or idx >= len(_COMMON_POOL):
                continue

            candidate = _COMMON_POOL[idx]
            _RAW_RECORDS[idx] = {
                "candidate_index": idx + 1,  # one-based for CSV/readability
                "pool_index_zero_based": idx,
                "fdia_scale": _candidate_scale(candidate),
                "T_fdia_h": float(candidate.get("T_fdia", 99.0)),
                "T_dos_h": float(candidate.get("T_dos", 99.0)),
                "optimizer_score": float(score),
                "load_shedding": float(info.get("damage", float("nan"))),
                "min_linepack_pct": float(info.get("min_lp_pct", float("nan"))),
                "linepack_at_DoS_pct": float(info.get("dos_lp_at_trigger", float("nan"))),
                "post_DoS_drop_pct": float(info.get("post_dos_drop_pct", float("nan"))),
                "min_linepack_after_DoS_pct": float(
                    info.get("min_lp_after_dos_pct", float("nan"))
                ),
                "window_pass": bool(info.get("window_pass", False)),
                "candidate_name": str(candidate.get("name", "")),
            }

    return scores


def install_collection_patches() -> None:
    mip_nd_master.MIPNDOptimizer._generate_candidates = _patched_generate_candidates
    mip_nd_master.MIPNDOptimizer._score_batch = _patched_score_batch
    mip_nd_master.MIPNDOptimizer._refine_batch_best = _patched_refine_batch_best


def restore_collection_patches() -> None:
    mip_nd_master.MIPNDOptimizer._generate_candidates = _ORIGINAL_GENERATE
    mip_nd_master.MIPNDOptimizer._score_batch = _ORIGINAL_SCORE_BATCH
    mip_nd_master.MIPNDOptimizer._refine_batch_best = _ORIGINAL_REFINE_BATCH_BEST


def _full_run_summary_path(args: Any) -> Path:
    return Path(args.out_dir) / args.system / args.upper_method / "run_summary.json"


def _read_full_run_final(args: Any) -> dict[str, Any]:
    path = _full_run_summary_path(args)
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    best = summary.get("best", {}).get("coordinated", {})
    return {
        "final_optimizer_fitness_after_local_refinement": best.get("optimizer_fitness"),
        "final_load_shedding_after_local_refinement": best.get("final_total_shed"),
        "final_min_linepack_pct_after_local_refinement": best.get("final_min_lp_pct"),
        "final_T_fdia_h_after_local_refinement": best.get("T_fdia"),
        "final_T_dos_h_after_local_refinement": best.get("T_dos"),
        "final_linepack_at_DoS_pct_after_local_refinement": best.get(
            "final_dos_lp_at_trigger_pct"
        ),
    }


def _validate_collected_pool() -> list[dict[str, Any]]:
    missing = [i for i in range(POOL_K) if i not in _RAW_RECORDS]
    if missing:
        preview = ", ".join(str(i + 1) for i in missing[:10])
        raise RuntimeError(
            f"Raw responses were not collected for {len(missing)} pool candidates. "
            f"First missing one-based indices: {preview}"
        )

    records = [_RAW_RECORDS[i] for i in range(POOL_K)]

    groups: dict[float, int] = Counter(round(float(r["fdia_scale"]), 6) for r in records)
    if len(groups) != 4:
        print(
            f"[Sensitivity warning] Expected four FDIA scales but found {len(groups)}: {dict(groups)}"
        )

    for k in K_VALUES:
        if k == POOL_K:
            continue
        if len(groups) > 0 and k % len(groups) != 0:
            raise RuntimeError(
                f"K={k} is not divisible by the number of FDIA-scale groups ({len(groups)})."
            )

    return records


def _write_candidate_pool(records: list[dict[str, Any]]) -> None:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "candidate_index",
        "fdia_scale",
        "T_fdia_h",
        "T_dos_h",
        "optimizer_score",
        "load_shedding",
        "min_linepack_pct",
        "linepack_at_DoS_pct",
        "post_DoS_drop_pct",
        "min_linepack_after_DoS_pct",
        "window_pass",
        "candidate_name",
    ]
    with POOL_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _score_is_exact_best(score: float, reference: float) -> bool:
    return bool(np.isclose(score, reference, rtol=EXACT_SCORE_RTOL, atol=EXACT_SCORE_ATOL))


def _score_is_near_best(score: float, reference: float) -> bool:
    # Relative tolerance remains well defined even if a user changes model
    # weights and the best score is close to zero or negative.
    scale = max(abs(float(reference)), 1.0)
    return float(score) >= float(reference) - NEAR_OPTIMAL_REL_TOL * scale


def _build_stratified_nested_samples(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Repeated nested sampling from the already evaluated common pool."""
    # 【修改标记 4】分层 + 嵌套抽样：S32 ⊂ S64 ⊂ S128 ⊂ S256 ⊂ S512。
    by_scale: dict[float, list[int]] = defaultdict(list)
    for idx, row in enumerate(records):
        by_scale[round(float(row["fdia_scale"]), 6)].append(idx)

    scales = sorted(by_scale)
    if not scales:
        raise RuntimeError("No FDIA-scale groups were found in the common pool.")

    full_best_idx = max(
        range(len(records)), key=lambda i: float(records[i]["optimizer_score"])
    )
    full_best = records[full_best_idx]
    full_best_score = float(full_best["optimizer_score"])

    samples: list[dict[str, Any]] = []

    for rep in range(1, N_RESAMPLES + 1):
        rng = np.random.default_rng(RESAMPLE_SEED + rep - 1)

        # One independent permutation per scale.  Prefixes are then nested:
        # the 8 candidates/scale used by K=32 are retained by K=64, etc.
        shuffled: dict[float, list[int]] = {}
        for scale in scales:
            arr = np.asarray(by_scale[scale], dtype=int)
            shuffled[scale] = rng.permutation(arr).astype(int).tolist()

        for k in K_VALUES:
            if k == POOL_K:
                selected = list(range(len(records)))
            else:
                per_scale = k // len(scales)
                selected = []
                for scale in scales:
                    if per_scale > len(shuffled[scale]):
                        raise RuntimeError(
                            f"Scale {scale} contains only {len(shuffled[scale])} candidates, "
                            f"but K={k} requires {per_scale}."
                        )
                    selected.extend(shuffled[scale][:per_scale])

            best_idx = max(selected, key=lambda i: float(records[i]["optimizer_score"]))
            best = records[best_idx]
            score = float(best["optimizer_score"])

            samples.append(
                {
                    "repeat": rep,
                    "candidate_number_K": k,
                    "selected_candidate_index": int(best["candidate_index"]),
                    "selected_fdia_scale": float(best["fdia_scale"]),
                    "best_optimizer_score": score,
                    "best_evaluated_load_shedding": float(best["load_shedding"]),
                    "min_linepack_pct": float(best["min_linepack_pct"]),
                    "FDIA_time_h": float(best["T_fdia_h"]),
                    "DoS_time_h": float(best["T_dos_h"]),
                    "linepack_at_DoS_pct": float(best["linepack_at_DoS_pct"]),
                    "post_DoS_drop_pct": float(best["post_DoS_drop_pct"]),
                    "exact_best_hit": int(_score_is_exact_best(score, full_best_score)),
                    "near_optimal_hit": int(_score_is_near_best(score, full_best_score)),
                }
            )

    return samples


def _write_samples(samples: list[dict[str, Any]]) -> None:
    fieldnames = [
        "repeat",
        "candidate_number_K",
        "selected_candidate_index",
        "selected_fdia_scale",
        "best_optimizer_score",
        "best_evaluated_load_shedding",
        "min_linepack_pct",
        "FDIA_time_h",
        "DoS_time_h",
        "linepack_at_DoS_pct",
        "post_DoS_drop_pct",
        "exact_best_hit",
        "near_optimal_hit",
    ]
    with SAMPLES_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(samples)


def _mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    return mean, std


def _mode_discrete(values: list[float]) -> float:
    rounded = [round(float(v), 10) for v in values]
    counts = Counter(rounded)
    max_count = max(counts.values())
    # Deterministic tie break: earliest attack time.
    return float(min(v for v, count in counts.items() if count == max_count))


def _aggregate(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # 【修改标记 5】汇总均值、标准差、Exact-best 命中率和 1% near-optimal 命中率。
    rows: list[dict[str, Any]] = []

    for k in K_VALUES:
        subset = [r for r in samples if int(r["candidate_number_K"]) == k]
        if not subset:
            continue

        score_mean, score_std = _mean_std([float(r["best_optimizer_score"]) for r in subset])
        shed_mean, shed_std = _mean_std(
            [float(r["best_evaluated_load_shedding"]) for r in subset]
        )
        lp_mean, lp_std = _mean_std([float(r["min_linepack_pct"]) for r in subset])
        dos_lp_mean, dos_lp_std = _mean_std(
            [float(r["linepack_at_DoS_pct"]) for r in subset]
        )

        rows.append(
            {
                "candidate_number_K": k,
                "resamples": len(subset),
                "mean_best_optimizer_score": score_mean,
                "std_best_optimizer_score": score_std,
                "mean_best_evaluated_load_shedding": shed_mean,
                "std_best_evaluated_load_shedding": shed_std,
                "mean_min_linepack_pct": lp_mean,
                "std_min_linepack_pct": lp_std,
                "mode_FDIA_time_h": _mode_discrete([float(r["FDIA_time_h"]) for r in subset]),
                "mode_DoS_time_h": _mode_discrete([float(r["DoS_time_h"]) for r in subset]),
                "mean_linepack_at_DoS_pct": dos_lp_mean,
                "std_linepack_at_DoS_pct": dos_lp_std,
                "exact_best_hit_rate_pct": 100.0
                * float(np.mean([int(r["exact_best_hit"]) for r in subset])),
                "near_optimal_hit_rate_pct": 100.0
                * float(np.mean([int(r["near_optimal_hit"]) for r in subset])),
            }
        )

    return rows


def _write_summary(rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "candidate_number_K",
        "resamples",
        "mean_best_optimizer_score",
        "std_best_optimizer_score",
        "mean_best_evaluated_load_shedding",
        "std_best_evaluated_load_shedding",
        "mean_min_linepack_pct",
        "std_min_linepack_pct",
        "mode_FDIA_time_h",
        "mode_DoS_time_h",
        "mean_linepack_at_DoS_pct",
        "std_linepack_at_DoS_pct",
        "exact_best_hit_rate_pct",
        "near_optimal_hit_rate_pct",
    ]
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _print_summary(rows: list[dict[str, Any]]) -> None:
    print("\n" + "=" * 132)
    print("Fixed-pool MIP candidate-set-size sensitivity summary")
    print("=" * 132)
    print(
        f"{'K':>6} | {'Best shed mean±std':>25} | {'Min LP mean±std (%)':>23} | "
        f"{'FDIA mode':>9} | {'DoS mode':>8} | {'Exact hit':>10} | {'1% near hit':>12}"
    )
    print("-" * 132)

    for row in rows:
        shed = (
            f"{float(row['mean_best_evaluated_load_shedding']):.2f}"
            f"±{float(row['std_best_evaluated_load_shedding']):.2f}"
        )
        lp = (
            f"{float(row['mean_min_linepack_pct']):.3f}"
            f"±{float(row['std_min_linepack_pct']):.3f}"
        )
        print(
            f"{int(row['candidate_number_K']):>6d} | {shed:>25} | {lp:>23} | "
            f"{float(row['mode_FDIA_time_h']):>8.2f}h | "
            f"{float(row['mode_DoS_time_h']):>7.2f}h | "
            f"{float(row['exact_best_hit_rate_pct']):>9.1f}% | "
            f"{float(row['near_optimal_hit_rate_pct']):>11.1f}%"
        )

    print("=" * 132)
    print(f"Candidate pool CSV : {POOL_CSV}")
    print(f"Resampling CSV     : {SAMPLES_CSV}")
    print(f"Summary CSV        : {SUMMARY_CSV}")


def _write_metadata(
    records: list[dict[str, Any]],
    elapsed_seconds: float,
    args: Any,
    final_run: dict[str, Any],
) -> None:
    best_idx = max(range(len(records)), key=lambda i: float(records[i]["optimizer_score"]))
    raw_best = records[best_idx]
    scale_counts = Counter(round(float(r["fdia_scale"]), 6) for r in records)

    payload = {
        "design": "one fixed K=512 pool + one DND evaluation + stratified nested post-hoc resampling",
        "K_values": list(K_VALUES),
        "pool_size": len(records),
        "resamples": N_RESAMPLES,
        "resample_seed": RESAMPLE_SEED,
        "near_optimal_relative_score_tolerance": NEAR_OPTIMAL_REL_TOL,
        "fdia_scale_counts": {str(k): int(v) for k, v in sorted(scale_counts.items())},
        "full_pool_raw_best": {
            key: raw_best[key]
            for key in [
                "candidate_index",
                "fdia_scale",
                "T_fdia_h",
                "T_dos_h",
                "optimizer_score",
                "load_shedding",
                "min_linepack_pct",
                "linepack_at_DoS_pct",
            ]
        },
        "full_K512_run_after_local_refinement": final_run,
        "full_run_elapsed_seconds": elapsed_seconds,
        "system": getattr(args, "system", None),
        "upper_method": getattr(args, "upper_method", None),
        "lower_solver": "model-distributed neurodynamics",
    }
    with METADATA_JSON.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main() -> None:

    patch_lower_solver_to_model_distributed_nd()
    install_collection_patches()

    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    _COMMON_POOL.clear()
    _RAW_RECORDS.clear()

    args = project_main.parse_args()
    args.upper_method = "mip-nd"
    args.mip_top_k = POOL_K
    args.out_dir = str(FULL_RUN_ROOT)

    # Output-only settings: these do not alter optimization or physical model
    # parameters.  The upper search curve is still saved as CSV by main.py.
    args.make_method_figures = False
    args.record_lower_trace = False

    print("=" * 96)
    print("MIP-DND candidate-set-size sensitivity experiment")
    print(f"Common candidate pool: K_max = {POOL_K}")
    print(f"Reported K values     : {K_VALUES}")
    print(f"Post-hoc resamples    : {N_RESAMPLES}")
    print(f"Resampling seed       : {RESAMPLE_SEED}")
    print(f"Near-optimal threshold: within {100.0 * NEAR_OPTIMAL_REL_TOL:.2f}% of full-pool raw best score")
    print("Only ONE full K=512 DND candidate evaluation is performed.")
    print("Smaller K values are evaluated by stratified nested resampling of the same responses.")
    print("=" * 96)

    start = time.time()
    try:
        project_main.run_experiment(args)
    finally:
        # Always restore class methods for safety when the script is imported or
        # run in an interactive Python process.
        restore_collection_patches()

    elapsed = time.time() - start

    records = _validate_collected_pool()
    _write_candidate_pool(records)

    samples = _build_stratified_nested_samples(records)
    _write_samples(samples)

    summary_rows = _aggregate(samples)
    _write_summary(summary_rows)

    final_run = _read_full_run_final(args)
    _write_metadata(records, elapsed, args, final_run)
    _print_summary(summary_rows)

    print("\nInterpretation reminder:")
    print(
        "  The code does not force K=128 to be optimal.  It reports whether the "
        "probability of retaining a near-optimal candidate naturally stabilizes "
        "around K=128 under the fixed common-pool experiment."
    )


if __name__ == "__main__":
    main()

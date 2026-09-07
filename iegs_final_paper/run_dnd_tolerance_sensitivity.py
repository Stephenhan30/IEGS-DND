"""
DND stopping-tolerance sensitivity experiment (summary-only version).

This script runs the proposed MIP-DND method with identical base-case settings
and changes only the two DND relative stopping tolerances:

    eps = 1e-2, 1e-3, 1e-4

Key controls
------------
1. The paper/base-case DND iteration cap is kept at 2500 for all cases.
2. The convergence patience is fixed at five consecutive checks.
3. No publication/result figures are saved.
4. Each completed case is immediately written to one combined CSV, so earlier
   results are preserved even if a later case is interrupted.
5. Temporary per-case result folders are deleted after their summary values
   have been extracted.

Final output
------------
    results_tolerance_sensitivity/tolerance_sensitivity_summary.csv

Usage
-----
    python run_dnd_tolerance_sensitivity_summary_only.py

Background/nohup:
    nohup python -u run_dnd_tolerance_sensitivity_summary_only.py > tolerance_sensitivity.log 2>&1 &

Optional: rerun only one tolerance while keeping existing CSV rows:
    python run_dnd_tolerance_sensitivity_summary_only.py --only 1e-4

Any additional arguments not controlled by this script are forwarded to the
main experiment, e.g.:
    python run_dnd_tolerance_sensitivity_summary_only.py --mip-allow-fallback
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


TOLERANCES = (1e-2, 1e-3, 1e-4)
MAX_ODE_STEPS = 2500
CONVERGENCE_PATIENCE = 5
CHECK_EFFECTIVE_INTERVAL = 5

PENALTY_PARAMETERS = {
    "lambda1": 5.0e4,
    "lambda2": 5.0e4,
    "lambda3": 10.0,
}


_CHILD_RUNNER = r"""
import os
import sys

sys.path.insert(0, os.getcwd())

import main as project_main
import model_distributed_nd_simulation
import algorithm.mip_nd_master as mip_nd_master

sim = model_distributed_nd_simulation.simulate_strategies_model_distributed_nd_batched
project_main.simulate_strategies_batched = sim
mip_nd_master.simulate_strategies_batched = sim

# Suppress all figure-file output.
project_main.save_publication_figure = lambda *args, **kwargs: None

args = project_main.parse_args()
args.make_method_figures = False

# Keep one convergence probe for R_S/R_C summary only.
args.record_lower_trace = True

project_main.run_experiment(args)
"""


def _tol_text(value: float) -> str:
    return f"{value:.0e}".replace("e-0", "e-").replace("e+0", "e+")


def _to_float(row: dict[str, str], *keys: str) -> float:
    for key in keys:
        value = row.get(key, "")
        if value not in (None, ""):
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    return float("nan")


def _to_int(row: dict[str, str], *keys: str) -> int:
    value = _to_float(row, *keys)
    return int(round(value)) if math.isfinite(value) else -1


def _read_last_csv_row(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else {}


def _read_existing_summary(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}

    out: dict[str, dict] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            try:
                key = _tol_text(float(row["tolerance"]))
            except Exception:
                continue
            out[key] = dict(row)
    return out


def _write_summary(rows: list[dict], path: Path) -> None:
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    tmp.replace(path)


def _residual_ok(rs: float, rc: float, eps: float) -> bool:
    return (
        math.isfinite(rs)
        and math.isfinite(rc)
        and rs <= eps
        and rc <= eps
    )


def _clean_case_folder(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


def _run_one(
    project_dir: Path,
    work_root: Path,
    tolerance: float,
    extra_args: list[str],
) -> dict:
    tol_text = _tol_text(tolerance)
    run_root = work_root / f"tol_{tol_text}"
    _clean_case_folder(run_root)

    cmd = [
        sys.executable,
        "-u",
        "-c",
        _CHILD_RUNNER,

        "--system", "118-135",
        "--upper-method", "mip-nd",
        "--dynamics-mode", "full",
        "--max-ode-steps", str(MAX_ODE_STEPS),

        "--mp-time-step", "1.0",
        "--mp-chunk-size", "8",

        "--mip-top-k", "128",
        "--mip-fdia-scales", "0.6,0.75,0.9,1.0",
        "--mip-time-limit", "10",
        "--mip-gap", "0.0",

        "--mdnd-state-rel-tolerance", tol_text,
        "--mdnd-consensus-rel-tolerance", tol_text,
        "--mdnd-convergence-patience", str(CONVERGENCE_PATIENCE),
        "--nd-check-effective-interval", str(CHECK_EFFECTIVE_INTERVAL),

        "--seed", "42",
        "--out-dir", str(run_root),
    ]
    cmd.extend(extra_args)

    print("\n" + "=" * 100, flush=True)
    print(f"DND tolerance sensitivity: epsilon = {tol_text}", flush=True)
    print(f"max_ode_steps             : {MAX_ODE_STEPS}", flush=True)
    print(f"patience                  : {CONVERGENCE_PATIENCE}", flush=True)
    print("=" * 100, flush=True)

    start = time.perf_counter()

    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("PYTHONUNBUFFERED", "1")

    try:
        subprocess.run(
            cmd,
            cwd=project_dir,
            env=env,
            check=True,
        )

        runtime_sec = time.perf_counter() - start

        result_dir = run_root / "118-135" / "mip-nd"
        summary_path = result_dir / "run_summary.json"
        trace_path = result_dir / "lower_nd_convergence_118-135.csv"

        if not summary_path.exists():
            raise FileNotFoundError(f"Missing expected summary: {summary_path}")

        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)

        best = summary["best"]["coordinated"]
        conv = summary.get("mdnd_convergence", {})
        perf = summary.get("mdnd_performance_stats", {})
        last = _read_last_csv_row(trace_path)

        probe_iterations = _to_int(last, "raw_step", "effective_step")
        probe_rs = _to_float(
            last,
            "state_relative_residual",
            "projected_gradient_residual",
        )
        probe_rc = _to_float(
            last,
            "gt_proposal_relative_consensus_residual",
            "gt_relative_consensus_residual",
        )
        probe_stop_reason = str(last.get("stop_reason", "") or "").strip()

        stop_reasons = perf.get("stop_reasons", {})
        if isinstance(stop_reasons, dict):
            stop_reasons_text = "; ".join(
                f"{k}:{v}" for k, v in sorted(stop_reasons.items())
            )
        else:
            stop_reasons_text = str(stop_reasons)

        return {
            "tolerance": tolerance,
            "max_ode_steps": MAX_ODE_STEPS,
            "patience": int(conv.get("patience", CONVERGENCE_PATIENCE)),

            "total_load_shedding_MWh": float(best["final_total_shed"]),
            "min_linepack_pct": float(best["final_min_lp_pct"]),
            "dos_linepack_pct": float(best["final_dos_lp_at_trigger_pct"]),
            "T_FDIA_h": float(best["T_fdia"]),
            "T_DoS_h": float(best["T_dos"]),

            "probe_iterations": probe_iterations,
            "probe_final_R_S": probe_rs,
            "probe_final_R_C": probe_rc,
            "probe_residuals_within_tolerance": _residual_ok(
                probe_rs, probe_rc, tolerance
            ),
            "probe_stop_reason": probe_stop_reason,

            "candidate_weighted_steps": int(
                perf.get("candidate_weighted_steps", 0) or 0
            ),
            "max_steps_hits": int(perf.get("max_steps_hits", 0) or 0),
            "all_stop_reasons": stop_reasons_text,

            "runtime_sec": runtime_sec,

            "lambda1": PENALTY_PARAMETERS["lambda1"],
            "lambda2": PENALTY_PARAMETERS["lambda2"],
            "lambda3": PENALTY_PARAMETERS["lambda3"],

            "status": "ok",
        }

    except subprocess.CalledProcessError as exc:
        return {
            "tolerance": tolerance,
            "max_ode_steps": MAX_ODE_STEPS,
            "patience": CONVERGENCE_PATIENCE,
            "runtime_sec": time.perf_counter() - start,
            "lambda1": PENALTY_PARAMETERS["lambda1"],
            "lambda2": PENALTY_PARAMETERS["lambda2"],
            "lambda3": PENALTY_PARAMETERS["lambda3"],
            "status": f"failed(returncode={exc.returncode})",
        }

    except Exception as exc:
        return {
            "tolerance": tolerance,
            "max_ode_steps": MAX_ODE_STEPS,
            "patience": CONVERGENCE_PATIENCE,
            "runtime_sec": time.perf_counter() - start,
            "lambda1": PENALTY_PARAMETERS["lambda1"],
            "lambda2": PENALTY_PARAMETERS["lambda2"],
            "lambda3": PENALTY_PARAMETERS["lambda3"],
            "status": f"failed({type(exc).__name__}: {exc})",
        }

    finally:
        # Keep only the final combined summary CSV.
        _clean_case_folder(run_root)


def _fmt_float(value, fmt: str = ".3e") -> str:
    try:
        x = float(value)
        if math.isfinite(x):
            return format(x, fmt)
    except Exception:
        pass
    return "-"


def _print_summary(rows: list[dict]) -> None:
    print("\n" + "=" * 136)
    print("DND stopping-tolerance sensitivity summary")
    print("=" * 136)

    header = (
        f"{'eps':>8} | "
        f"{'shed(MWh)':>11} | "
        f"{'minLP(%)':>8} | "
        f"{'DoS LP(%)':>9} | "
        f"{'probe it.':>9} | "
        f"{'R_S':>10} | "
        f"{'R_C':>10} | "
        f"{'OK':>4} | "
        f"{'max hits':>8} | "
        f"{'runtime(s)':>10} | status"
    )
    print(header)
    print("-" * len(header))

    for row in sorted(rows, key=lambda x: float(x["tolerance"]), reverse=True):
        status = str(row.get("status", ""))
        if status != "ok":
            print(
                f"{_tol_text(float(row['tolerance'])):>8} | "
                f"{'-':>11} | {'-':>8} | {'-':>9} | {'-':>9} | "
                f"{'-':>10} | {'-':>10} | {'-':>4} | {'-':>8} | "
                f"{_fmt_float(row.get('runtime_sec'), '.2f'):>10} | {status}"
            )
            continue

        ok = str(row.get("probe_residuals_within_tolerance", False)).lower() in (
            "true", "1", "yes"
        )

        print(
            f"{_tol_text(float(row['tolerance'])):>8} | "
            f"{float(row['total_load_shedding_MWh']):11.2f} | "
            f"{float(row['min_linepack_pct']):8.2f} | "
            f"{float(row['dos_linepack_pct']):9.2f} | "
            f"{int(float(row['probe_iterations'])):9d} | "
            f"{_fmt_float(row['probe_final_R_S']):>10} | "
            f"{_fmt_float(row['probe_final_R_C']):>10} | "
            f"{('YES' if ok else 'NO'):>4} | "
            f"{int(float(row['max_steps_hits'])):8d} | "
            f"{float(row['runtime_sec']):10.2f} | ok"
        )

    print("\nPenalty parameters:")
    print(
        f"  lambda1={PENALTY_PARAMETERS['lambda1']:.0f}, "
        f"lambda2={PENALTY_PARAMETERS['lambda2']:.0f}, "
        f"lambda3={PENALTY_PARAMETERS['lambda3']:.0f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run DND stopping-tolerance sensitivity at 1e-2, 1e-3, and 1e-4 "
            "and keep only one combined summary CSV."
        )
    )
    parser.add_argument(
        "--output-root",
        default="results_tolerance_sensitivity",
        help="Directory containing only the final combined CSV.",
    )
    parser.add_argument(
        "--only",
        choices=["1e-2", "1e-3", "1e-4"],
        default=None,
        help="Rerun only one tolerance and merge it into an existing summary CSV.",
    )

    args, extra_args = parser.parse_known_args()

    forbidden = {
        "--system",
        "--upper-method",
        "--dynamics-mode",
        "--max-ode-steps",
        "--mp-time-step",
        "--mp-chunk-size",
        "--mip-top-k",
        "--mip-fdia-scales",
        "--mip-time-limit",
        "--mip-gap",
        "--mdnd-state-rel-tolerance",
        "--mdnd-consensus-rel-tolerance",
        "--mdnd-convergence-patience",
        "--nd-check-effective-interval",
        "--seed",
        "--out-dir",
    }

    for token in extra_args:
        option = token.split("=", 1)[0]
        if option in forbidden:
            parser.error(
                f"Do not pass {option}; this sensitivity script controls it."
            )

    project_dir = Path(__file__).resolve().parent

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = project_dir / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    summary_csv = output_root / "tolerance_sensitivity_summary.csv"
    work_root = output_root / "_working"

    existing = _read_existing_summary(summary_csv)

    if args.only is None:
        selected = list(TOLERANCES)
        rows_by_key: dict[str, dict] = {}
    else:
        selected = [float(args.only)]
        rows_by_key = dict(existing)

    for tol in selected:
        row = _run_one(
            project_dir=project_dir,
            work_root=work_root,
            tolerance=tol,
            extra_args=extra_args,
        )

        rows_by_key[_tol_text(tol)] = row

        rows = sorted(
            rows_by_key.values(),
            key=lambda x: float(x["tolerance"]),
            reverse=True,
        )
        _write_summary(rows, summary_csv)

        print(
            f"\nSaved completed case {_tol_text(tol)} -> {summary_csv}",
            flush=True,
        )

    if work_root.exists():
        shutil.rmtree(work_root, ignore_errors=True)

    final_rows = sorted(
        rows_by_key.values(),
        key=lambda x: float(x["tolerance"]),
        reverse=True,
    )
    _print_summary(final_rows)

    print(f"\nFinal summary CSV: {summary_csv}", flush=True)


if __name__ == "__main__":
    main()

"""Run CND, DND, ADMM, and ALADIN, then generate comparison figures."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


METHODS = [
    ("Centralized ND", "run_centralized_baseline.py"),
    ("Distributed ND", "run_distributed_main.py"),
    ("ADMM", "run_admm_baseline.py"),
    ("ALADIN", "run_aladin_baseline.py"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Run the four methods but do not generate the comparison bar chart.",
    )
    args, common_args = parser.parse_known_args()

    for label, script in METHODS:
        cmd = [sys.executable, "-u", script, *common_args]
        print("\n" + "=" * 88)
        print(f"Running {label}: {' '.join(cmd)}")
        print("=" * 88, flush=True)
        subprocess.run(cmd, check=True)

    if args.no_plot:
        return

    summaries = [
        Path("results_distributed/118-135/mip-nd/run_summary.json"),
        Path("results_centralized/118-135/mip-nd/run_summary.json"),
        Path("results_admm/118-135/mip-nd/run_summary.json"),
        Path("results_aladin/118-135/mip-nd/run_summary.json"),
    ]
    missing = [str(path) for path in summaries if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing method summaries: " + ", ".join(missing))

    plot_cmd = [
        sys.executable,
        "-u",
        "plot_four_method_bar.py",
        "--out-dir",
        "results_method_compare/118-135",
    ]
    print("\nGenerating four-method comparison figure ...", flush=True)
    subprocess.run(plot_cmd, check=True)


if __name__ == "__main__":
    main()

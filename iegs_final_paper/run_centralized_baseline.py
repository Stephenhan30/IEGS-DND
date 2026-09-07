"""Run the MIP–CND comparison method."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.getcwd())

import main as project_main


def main() -> None:
    args = project_main.parse_args()
    args.make_method_figures = False
    args.record_lower_trace = False
    if args.out_dir == "results":
        args.out_dir = "results_centralized"
    print("=" * 88)
    print("Comparison method: MIP + centralized neurodynamic lower solver")
    print("=" * 88)
    project_main.run_experiment(args)


if __name__ == "__main__":
    main()

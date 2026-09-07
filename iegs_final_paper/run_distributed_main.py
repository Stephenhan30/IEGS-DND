"""Run the proposed MIP–DND method."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.getcwd())

import main as project_main
import model_distributed_nd_simulation
import algorithm.mip_nd_master as mip_nd_master


def patch_lower_solver_to_model_distributed_nd() -> None:
    sim = model_distributed_nd_simulation.simulate_strategies_model_distributed_nd_batched
    project_main.simulate_strategies_batched = sim
    mip_nd_master.simulate_strategies_batched = sim


def main() -> None:
    patch_lower_solver_to_model_distributed_nd()
    args = project_main.parse_args()
    args.make_method_figures = True
    args.record_lower_trace = True
    if args.out_dir == "results":
        args.out_dir = "results_distributed"
    print("=" * 88)
    print("Main method: MIP + model-distributed neurodynamic lower solver")
    print("Lower model : distributed electric/gas local submodels + stabilized GFPP consensus coordinator")
    print("=" * 88)
    project_main.run_experiment(args)


if __name__ == "__main__":
    main()

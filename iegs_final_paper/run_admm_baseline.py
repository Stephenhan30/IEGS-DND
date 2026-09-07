"""Run the MIP–ADMM comparison method."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.getcwd())

import main as project_main
import admm_distributed_simulation
import algorithm.mip_nd_master as mip_nd_master
from comparison_trace_export import save_admm_convergence_results


def patch_admm_workflow() -> None:
    batched_sim = (
        admm_distributed_simulation.simulate_strategies_admm_distributed_batched
    )

    # The standard ADMM solver is fully batch-separable, so the same exact
    # local-QP/consensus iteration is used for candidate screening, final replay,
    # and timing-sensitivity scenarios.  No batch-dependent approximation is used.
    mip_nd_master.simulate_strategies_batched = batched_sim
    project_main.simulate_strategies_batched = batched_sim

    original_mip_runner = project_main._run_mip_nd_optimizer

    def run_mip_admm_optimizer(env, mode, initial_state, args):
        env.mip_nd_display_label = "MIP-ADMM"
        env.mip_nd_hard_window = True
        env.mip_nd_single_validation_top_k = 8

        # Physics-aware ADMM defaults.  The full-state residual is included in
        # the stopping test; no neurodynamic or Adam inner loop is used.
        env.admm_rho = float(getattr(args, "admm_rho", 10.0))
        env.admm_outer_iters = int(getattr(args, "admm_outer_iters", 200))
        env.admm_abs_tolerance = float(
            getattr(args, "admm_abs_tolerance", 1e-4)
        )
        env.admm_rel_tolerance = float(
            getattr(args, "admm_rel_tolerance", 1e-3)
        )
        env.admm_consensus_rel_tolerance = env.admm_rel_tolerance
        env.admm_dual_rel_tolerance = env.admm_rel_tolerance
        env.admm_min_outer_iters = int(
            getattr(args, "admm_min_outer_iters", 10)
        )
        env.admm_convergence_patience = int(
            getattr(args, "admm_convergence_patience", 5)
        )
        env.admm_state_rel_tolerance = float(
            getattr(args, "admm_state_rel_tolerance", 1e-3)
        )
        env.admm_relaxation = float(
            getattr(args, "admm_relaxation", 0.65)
        )
        env.admm_electric_weight = float(
            getattr(args, "admm_electric_weight", 1.0)
        )
        env.admm_gas_weight = float(
            getattr(args, "admm_gas_weight", 1.0)
        )
        env.admm_linepack_withdrawal_fraction = float(
            getattr(args, "admm_linepack_withdrawal_fraction", 0.12)
        )
        env.admm_gas_security_floor_ratio = float(
            getattr(args, "admm_gas_security_floor_ratio", 0.005)
        )
        env.admm_gas_budget_margin = float(
            getattr(args, "admm_gas_budget_margin", 0.0)
        )

        return original_mip_runner(env, mode, initial_state, args)

    project_main._run_mip_nd_optimizer = run_mip_admm_optimizer

    def admm_method_name(args) -> str:
        if getattr(args, "upper_method", "") == "mip-nd":
            return "MIP-ADMM"
        return str(args.upper_method)

    project_main._upper_method_name = admm_method_name
    project_main.plot_lower_nd_convergence = save_admm_convergence_results


def main() -> None:
    patch_admm_workflow()
    args = project_main.parse_args()
    args.lower_solver_kind = "standard_admm"
    args.make_method_figures = False
    args.record_lower_trace = True
    if args.out_dir == "results":
        args.out_dir = "results_admm"

    print("=" * 88)
    print("Comparison method: MIP-ADMM")
    print("Lower evaluator  : physics-aware two-block consensus ADMM")
    print("Electric response: projected GFPP QP + complete 118-bus dispatch per iteration")
    print("Gas response     : gas-budget/pressure projection + complete 135-node replay per iteration")
    print("Stopping test    : primal + dual + GFPP consensus + full-state residual")
    print("Excluded         : neurodynamics, Adam, inner ND solve")
    print("=" * 88)

    project_main.run_experiment(args)


if __name__ == "__main__":
    main()

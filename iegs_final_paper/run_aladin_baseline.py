"""Run the MIP–ALADIN comparison method."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.getcwd())

import main as project_main
import aladin_distributed_simulation
import algorithm.mip_nd_master as mip_nd_master
from comparison_trace_export import save_aladin_convergence_results


def _parse_args():
    """Parse ALADIN-specific options before delegating common options to main."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--aladin-rho", type=float, default=5.0)
    parser.add_argument("--aladin-mu", type=float, default=20.0)
    parser.add_argument("--aladin-outer-iters", type=int, default=60)
    parser.add_argument("--aladin-alpha", type=float, default=0.45)
    parser.add_argument("--aladin-electric-weight", type=float, default=1.0)
    parser.add_argument("--aladin-gas-weight", type=float, default=1.0)
    parser.add_argument("--aladin-gas-loss-coefficient", type=float, default=1e-4)
    parser.add_argument("--aladin-hessian-regularization", type=float, default=1e-6)
    parser.add_argument("--aladin-active-tolerance", type=float, default=1e-6)
    parser.add_argument("--aladin-trust-region-fraction", type=float, default=0.50)
    parser.add_argument("--aladin-local-bisection-iters", type=int, default=48)
    parser.add_argument("--aladin-consensus-rel-tolerance", type=float, default=1e-3)
    parser.add_argument("--aladin-step-rel-tolerance", type=float, default=1e-3)
    parser.add_argument("--aladin-kkt-rel-tolerance", type=float, default=1e-3)
    parser.add_argument("--aladin-min-outer-iters", type=int, default=10)
    parser.add_argument("--aladin-convergence-patience", type=int, default=5)
    parser.add_argument("--aladin-state-rel-tolerance", type=float, default=1e-3)
    parser.add_argument("--aladin-linepack-withdrawal-fraction", type=float, default=0.12)
    parser.add_argument("--aladin-gas-security-floor-ratio", type=float, default=0.005)
    parser.add_argument("--aladin-gas-budget-margin", type=float, default=0.0)

    specific, remaining = parser.parse_known_args(sys.argv[1:])
    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0], *remaining]
        common = project_main.parse_args()
    finally:
        sys.argv = old_argv

    for key, value in vars(specific).items():
        setattr(common, key.replace("-", "_"), value)
    return common


def patch_aladin_workflow(args) -> None:
    batched_sim = aladin_distributed_simulation.simulate_strategies_aladin_distributed_batched
    mip_nd_master.simulate_strategies_batched = batched_sim
    project_main.simulate_strategies_batched = batched_sim

    original_mip_runner = project_main._run_mip_nd_optimizer

    def run_mip_aladin_optimizer(env, mode, initial_state, common_args):
        env.mip_nd_display_label = "MIP-ALADIN"
        env.mip_nd_hard_window = True
        env.mip_nd_single_validation_top_k = 8

        env.aladin_rho = float(args.aladin_rho)
        env.aladin_mu = float(args.aladin_mu)
        env.aladin_outer_iters = int(args.aladin_outer_iters)
        env.aladin_alpha = float(args.aladin_alpha)
        env.aladin_electric_weight = float(args.aladin_electric_weight)
        env.aladin_gas_weight = float(args.aladin_gas_weight)
        env.aladin_gas_loss_coefficient = float(args.aladin_gas_loss_coefficient)
        env.aladin_hessian_regularization = float(args.aladin_hessian_regularization)
        env.aladin_active_tolerance = float(args.aladin_active_tolerance)
        env.aladin_trust_region_fraction = float(args.aladin_trust_region_fraction)
        env.aladin_local_bisection_iters = int(args.aladin_local_bisection_iters)
        env.aladin_consensus_rel_tolerance = float(args.aladin_consensus_rel_tolerance)
        env.aladin_step_rel_tolerance = float(args.aladin_step_rel_tolerance)
        env.aladin_kkt_rel_tolerance = float(args.aladin_kkt_rel_tolerance)
        env.aladin_min_outer_iters = int(args.aladin_min_outer_iters)
        env.aladin_convergence_patience = int(args.aladin_convergence_patience)
        env.aladin_state_rel_tolerance = float(args.aladin_state_rel_tolerance)
        env.aladin_linepack_withdrawal_fraction = float(args.aladin_linepack_withdrawal_fraction)
        env.aladin_gas_security_floor_ratio = float(args.aladin_gas_security_floor_ratio)
        env.aladin_gas_budget_margin = float(args.aladin_gas_budget_margin)
        return original_mip_runner(env, mode, initial_state, common_args)

    project_main._run_mip_nd_optimizer = run_mip_aladin_optimizer

    def aladin_method_name(common_args) -> str:
        if getattr(common_args, "upper_method", "") == "mip-nd":
            return "MIP-ALADIN"
        return str(common_args.upper_method)

    project_main._upper_method_name = aladin_method_name
    project_main.plot_lower_nd_convergence = save_aladin_convergence_results


def _append_aladin_summary(args) -> None:
    summary_path = Path(args.out_dir) / args.system / args.upper_method / "run_summary.json"
    if not summary_path.exists():
        return
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    data["lower_solver"] = "physics_aware_distributed_aladin"
    # Remove ND-only diagnostics inherited from the shared main workflow so the
    # ALADIN result file cannot be misread as a neurodynamic run.
    data.pop("mdnd_convergence", None)
    data.pop("mdnd_performance_stats", None)
    data["aladin_config"] = {
        "rho": args.aladin_rho,
        "mu": args.aladin_mu,
        "outer_iters": args.aladin_outer_iters,
        "alpha": args.aladin_alpha,
        "electric_weight": args.aladin_electric_weight,
        "gas_weight": args.aladin_gas_weight,
        "gas_loss_coefficient": args.aladin_gas_loss_coefficient,
        "hessian_regularization": args.aladin_hessian_regularization,
        "active_tolerance": args.aladin_active_tolerance,
        "trust_region_fraction": args.aladin_trust_region_fraction,
        "local_bisection_iters": args.aladin_local_bisection_iters,
        "consensus_rel_tolerance": args.aladin_consensus_rel_tolerance,
        "step_rel_tolerance": args.aladin_step_rel_tolerance,
        "kkt_rel_tolerance": args.aladin_kkt_rel_tolerance,
        "min_outer_iters": args.aladin_min_outer_iters,
        "convergence_patience": args.aladin_convergence_patience,
        "state_relative_tolerance": args.aladin_state_rel_tolerance,
        "linepack_withdrawal_fraction": args.aladin_linepack_withdrawal_fraction,
        "gas_security_floor_ratio": args.aladin_gas_security_floor_ratio,
        "gas_budget_margin": args.aladin_gas_budget_margin,
    }
    data["aladin_model_note"] = (
        "Physics-aware ALADIN: each outer iteration solves the GFPP boundary local "
        "problems, recomputes the complete 118-bus and 135-node operating states, "
        "and applies a second-order coordination QP. The method is not an ALADIN+ND hybrid."
    )
    summary_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    if args.out_dir == "results":
        args.out_dir = "results_aladin"
    args.lower_solver_kind = "aladin"
    args.make_method_figures = False
    args.record_lower_trace = True
    patch_aladin_workflow(args)

    print("=" * 88)
    print("Comparison method : MIP-ALADIN")
    print("Lower evaluator   : physics-aware distributed ALADIN")
    print("Electric response : local GFPP QP + complete 118-bus dispatch per iteration")
    print("Gas response      : nonlinear gas deliverability + complete 135-node replay per iteration")
    print("Coordination      : second-order coordination QP with full-state stopping test")
    print("Excludes          : neurodynamics, Adam, ADMM updates")
    print("=" * 88)

    project_main.run_experiment(args)
    _append_aladin_summary(args)


if __name__ == "__main__":
    main()

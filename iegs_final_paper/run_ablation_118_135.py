"""Fixed-strategy FDIA/DoS ablation on IEEE 118 / GasLib-135.

Only this runner is added; the paper's original 118-135 model and algorithms are
left unchanged.

Ablation principle
------------------
The coordinated attack is optimized ONCE using the paper's original MIP-DND
procedure.  The resulting best evaluated coordinated strategy is then frozen
and used to construct all ablation cases:

Baseline:
    FDIA disabled, DoS disabled.

FDIA-only:
    Keep exactly the coordinated strategy's FDIA vector, FDIA scale, and
    T_FDIA; disable DoS.

DoS-only:
    Keep exactly the coordinated strategy's T_DoS and DoS target/action;
    disable FDIA.

Coordinated:
    Replay the optimized coordinated strategy unchanged.

This is a component-removal ablation rather than an independently optimized
attack-mode comparison.  Therefore any difference between FDIA-only,
DoS-only, and coordinated cases is caused by removing an attack stage while
the active attack parameters are inherited from the same coordinated strategy.

Important interpretation
------------------------
The vulnerable-linepack window is a feasibility condition of the coordinated
attack mechanism.  In the DoS-only ablation, the coordinated T_DoS is retained
even if the gas network no longer lies inside that vulnerable window after
FDIA is removed.  This is intentional: the purpose is to isolate the
preconditioning effect of FDIA-induced linepack depletion.

The original paper model keeps an activated FDIA/DoS action active through the
end of the 24-h horizon.  This runner preserves that physical convention.

Outputs
-------
results_ablation/118-135/mip-nd/
    attack_ablation_summary.csv
    attack_ablation_summary.json
    attack_ablation_118-135.png
    attack_ablation_trajectories.csv
    search_curve_coordinated.csv

Usage
-----
    python run_ablation_118_135_fixed.py

If CPLEX is unavailable and a local smoke test is needed:
    python run_ablation_118_135_fixed.py --mip-allow-fallback
"""
from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())

import main as project_main
import model_distributed_nd_simulation
import algorithm.mip_nd_master as mip_nd_master
from algorithm.mip_nd_master import MIPNDOptimizer
from attacker.fdia_generator import FDIAGenerator


SYSTEM_NAME = "118-135"
DISABLED_TIME = 99.0
HORIZON_END_H = 24.0
BUDGET_TOL = 5.0e-5


def _fdia_limit_base(env) -> np.ndarray:
    """Return the denominator used by FDIAGenerator for bus-wise FDIA bounds."""
    base_load = env.power.base_load.detach().cpu().numpy().astype(float)
    positive = base_load[base_load > 0.0]
    mean_load = float(np.mean(positive)) if positive.size else 1.0
    return np.maximum(base_load, 0.05 * mean_load)


def fdia_budget_metrics(
    env, za: np.ndarray, budget_ratio: float | None = None
) -> dict:
    """Measure FDIA usage in the same normalization as FDIAGenerator."""
    za = np.asarray(za, dtype=float).reshape(-1)
    limit_base = _fdia_limit_base(env)

    if za.size != limit_base.size:
        raise ValueError(
            f"FDIA vector length {za.size} does not match "
            f"power buses {limit_base.size}."
        )

    ratios = np.abs(za) / (limit_base + 1.0e-12)
    max_ratio = float(np.max(ratios)) if ratios.size else 0.0

    if budget_ratio is None:
        budget_ratio = float(getattr(env, "fdia_max_ratio", 0.20))
    budget_ratio = float(budget_ratio)

    return {
        "FDIA_L1_MW": float(np.sum(np.abs(za))),
        "FDIA_max_abs_MW": float(np.max(np.abs(za)) if za.size else 0.0),
        "FDIA_max_ratio_pct": 100.0 * max_ratio,
        "FDIA_budget_limit_pct": 100.0 * budget_ratio,
        "FDIA_budget_utilization_pct": (
            100.0 * max_ratio / budget_ratio
            if budget_ratio > 1.0e-12
            else 0.0
        ),
        "FDIA_zero_sum_MW": float(np.sum(za)),
        "FDIA_budget_ok": bool(max_ratio <= budget_ratio + BUDGET_TOL),
    }


class FixedAblationMIPNDOptimizer(MIPNDOptimizer):
    """Original coordinated MIP-DND with scale-specific FDIA budget repair."""

    def _repair_candidate_to_its_scale(self, candidate: dict) -> dict:
        """Enforce each candidate's own FDIA scale after stealth projection."""
        out = copy.deepcopy(candidate)

        meta = dict(out.get("meta", {}))
        scale = float(meta.get("fdia_scale", 1.0))
        scale = float(np.clip(scale, 1.0e-6, 1.0))

        full_budget = float(getattr(self.env, "fdia_max_ratio", 0.20))
        scaled_budget = full_budget * scale

        raw = np.asarray(out.get("za", out.get("FDIA")), dtype=np.float32)
        fdia_gen = FDIAGenerator(self.env.power)
        legal, proj_meta = fdia_gen.check_and_project_fdia(raw, scaled_budget)
        legal = np.asarray(legal, dtype=np.float32)

        out["FDIA"] = legal
        out["za"] = legal

        meta.update(proj_meta)
        meta.update(
            {
                "fdia_scale": scale,
                "fdia_scaled_budget_ratio": scaled_budget,
                "fdia_scale_budget_repaired": True,
            }
        )
        out["meta"] = meta
        return out

    def _generate_candidates(self) -> list[dict]:
        candidates = super()._generate_candidates()
        repaired = [self._repair_candidate_to_its_scale(c) for c in candidates]
        return self._deduplicate_candidates(repaired)

    def _attach_final_meta(self, best: dict) -> dict:
        """Preserve the optimizer-selected scale in the final strategy."""
        out = self._repair_candidate_to_its_scale(best)
        meta = dict(out.get("meta", {}))
        meta["fdia_final_checked"] = True

        scale = float(meta.get("fdia_scale", 1.0))
        selected_budget = (
            float(getattr(self.env, "fdia_max_ratio", 0.20)) * scale
        )
        metrics = fdia_budget_metrics(self.env, out["za"], selected_budget)
        meta.update({f"final_{k}": v for k, v in metrics.items()})

        if not metrics["FDIA_budget_ok"]:
            raise RuntimeError(
                "Selected coordinated FDIA violates its scale-specific budget "
                "after final projection: "
                f"used={metrics['FDIA_max_ratio_pct']:.6f}% > "
                f"limit={metrics['FDIA_budget_limit_pct']:.6f}%"
            )

        out["meta"] = meta
        return out


def patch_dnd_lower_solver() -> None:
    """Route MIP-DND candidate evaluation through model-distributed DND."""
    sim = (
        model_distributed_nd_simulation
        .simulate_strategies_model_distributed_nd_batched
    )
    project_main.simulate_strategies_batched = sim
    mip_nd_master.simulate_strategies_batched = sim
    project_main.MIPNDOptimizer = FixedAblationMIPNDOptimizer


def apply_cli_overrides(env, args) -> None:
    """Apply the same command-line overrides used by the normal DND runner."""
    if getattr(args, "mip_top_k", None) is not None:
        env.mip_nd_top_k = int(args.mip_top_k)
    if getattr(args, "mip_time_limit", None) is not None:
        env.mip_nd_time_limit = float(args.mip_time_limit)
    if getattr(args, "mip_gap", None) is not None:
        env.mip_nd_mip_gap = float(args.mip_gap)
    if getattr(args, "mip_cplex_log", False):
        env.mip_nd_cplex_log = True
    if getattr(args, "mip_allow_fallback", False):
        env.mip_nd_allow_fallback = True
    if getattr(args, "mip_fdia_scales", None):
        env.mip_nd_fdia_scales = [
            float(x.strip())
            for x in str(args.mip_fdia_scales).split(",")
            if x.strip()
        ]

    env.mdnd_parallel_models = not bool(
        getattr(args, "disable_mdnd_parallel", False)
    )
    env.mdnd_parallel_backend = str(
        getattr(args, "mdnd_parallel_backend", "auto")
    )
    env.mdnd_consensus_rel_tolerance = float(
        getattr(args, "mdnd_consensus_rel_tolerance", 1e-3)
    )
    env.mdnd_state_rel_tolerance = float(
        getattr(args, "mdnd_state_rel_tolerance", 1e-3)
    )
    env.mdnd_convergence_patience = max(
        1, int(getattr(args, "mdnd_convergence_patience", 5))
    )
    env.mdnd_lr_decay = max(
        0.0, float(getattr(args, "mdnd_lr_decay", 0.08))
    )
    env.mdnd_consensus_relaxation = min(
        1.0,
        max(
            0.0,
            float(getattr(args, "mdnd_consensus_relaxation", 1.0)),
        ),
    )
    env.nd_trace_effective_interval = float(
        getattr(args, "nd_check_effective_interval", 5.0)
    )
    env.mip_nd_refinement_cache_enabled = not bool(
        getattr(args, "disable_refinement_cache", False)
    )
    env.fast_dynamics = args.dynamics_mode == "fast"


def strategy_fdia_vector(strategy: dict, n_bus: int) -> np.ndarray:
    raw = strategy.get(
        "za",
        strategy.get("FDIA", np.zeros(n_bus, dtype=np.float32)),
    )
    if isinstance(raw, (list, tuple)):
        raw = raw[-1]
    return np.asarray(raw, dtype=np.float32)


def selected_fdia_scale(strategy: dict) -> float:
    return float(dict(strategy.get("meta", {})).get("fdia_scale", 0.0))


def _active_duration(start_h: float) -> float | None:
    if float(start_h) >= 90.0:
        return None
    return max(0.0, HORIZON_END_H - float(start_h) + 1.0)


def validate_strategy_budget(env, strategy: dict, scenario: str) -> dict:
    """Validate inherited FDIA resource usage in each ablation case."""
    za = strategy_fdia_vector(strategy, env.power.num_buses)

    if scenario in {"DoS-only", "Baseline"}:
        selected_budget = 0.0
    else:
        scale = selected_fdia_scale(strategy)
        if scale <= 0.0:
            scale = 1.0
        selected_budget = float(env.fdia_max_ratio) * scale

    metrics = fdia_budget_metrics(env, za, selected_budget)

    if (
        scenario not in {"DoS-only", "Baseline"}
        and not metrics["FDIA_budget_ok"]
    ):
        raise RuntimeError(
            f"{scenario} FDIA budget violation: "
            f"used={metrics['FDIA_max_ratio_pct']:.6f}% > "
            f"selected limit={metrics['FDIA_budget_limit_pct']:.6f}%"
        )

    return metrics


def save_search_curve(metric_curve: list[dict], path: Path) -> None:
    pd.DataFrame(metric_curve).to_csv(path, index=False)


def save_trajectory_table(result, labels: list[str], path: Path) -> None:
    rows = []
    for label, trajectory in zip(labels, result.trajectories):
        for row in trajectory:
            item = {"scenario": label}
            item.update(row)
            rows.append(item)
    pd.DataFrame(rows).to_csv(path, index=False)


def make_ablation_plot(summary_df: pd.DataFrame, out_path: Path) -> None:
    """Compact bar chart of realized load shedding."""
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.bar(summary_df["scenario"], summary_df["total_load_shedding"])
    ax.set_ylabel("Total load shedding")
    ax.set_xlabel("")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def build_fixed_ablation_strategies(
    env, coordinated: dict
) -> list[dict]:
    """Create component-removal cases from one frozen coordinated strategy."""
    n_bus = int(env.power.num_buses)
    zero = np.zeros(n_bus, dtype=np.float32)

    za_coord = strategy_fdia_vector(coordinated, n_bus).copy()
    t_fdia_coord = float(coordinated["T_fdia"])
    t_dos_coord = float(coordinated["T_dos"])
    meta_coord = copy.deepcopy(coordinated.get("meta", {}))

    common_ablation_meta = {
        "ablation_type": "fixed_coordinated_strategy_component_removal",
        "source_T_FDIA_h": t_fdia_coord,
        "source_T_DoS_h": t_dos_coord,
        "source_FDIA_scale": selected_fdia_scale(coordinated),
    }

    baseline_meta = copy.deepcopy(common_ablation_meta)
    baseline_meta["active_component"] = "none"

    fdia_meta = copy.deepcopy(meta_coord)
    fdia_meta.update(common_ablation_meta)
    fdia_meta["active_component"] = "FDIA"

    dos_meta = copy.deepcopy(common_ablation_meta)
    dos_meta["active_component"] = "DoS"

    coord_meta = copy.deepcopy(meta_coord)
    coord_meta.update(common_ablation_meta)
    coord_meta["active_component"] = "FDIA+DoS"

    return [
        {
            "name": "Baseline",
            "FDIA": zero.copy(),
            "za": zero.copy(),
            "T_fdia": DISABLED_TIME,
            "T_dos": DISABLED_TIME,
            "meta": baseline_meta,
        },
        {
            "name": "FDIA-only",
            "FDIA": za_coord.copy(),
            "za": za_coord.copy(),
            "T_fdia": t_fdia_coord,
            "T_dos": DISABLED_TIME,
            "meta": fdia_meta,
        },
        {
            "name": "DoS-only",
            "FDIA": zero.copy(),
            "za": zero.copy(),
            "T_fdia": DISABLED_TIME,
            "T_dos": t_dos_coord,
            "meta": dos_meta,
        },
        {
            "name": "Coordinated",
            "FDIA": za_coord.copy(),
            "za": za_coord.copy(),
            "T_fdia": t_fdia_coord,
            "T_dos": t_dos_coord,
            "meta": coord_meta,
        },
    ]


def main() -> None:
    patch_dnd_lower_solver()

    args = project_main.parse_args()
    args.system = SYSTEM_NAME
    if args.out_dir == "results":
        args.out_dir = "results_ablation"

    project_main.set_seed(args.seed)

    print("=" * 96)
    print("118-135 fixed-strategy DND ablation")
    print(
        "Optimize the coordinated attack once, then remove FDIA or DoS "
        "without re-optimizing the remaining stage."
    )
    print("=" * 96)

    system_data = project_main.load_system_data(
        target_system=SYSTEM_NAME,
        data_dir=args.data_dir,
    )
    env = project_main.IEGSSystem(system_data, dt_hours=1.0)
    project_main.configure_env(env, SYSTEM_NAME)
    apply_cli_overrides(env, args)

    initial_state = project_main.create_initial_physical_state(
        env, SYSTEM_NAME
    )
    env_snapshot = project_main.make_env_snapshot(env)
    state_snapshot = project_main.clone_state_dict(initial_state)

    print(
        f"FDIA maximum bus-wise budget : "
        f"{100.0 * float(env.fdia_max_ratio):.2f}%"
    )
    print(
        f"FDIA candidate scales        : "
        f"{list(env.mip_nd_fdia_scales)}"
    )
    print(
        f"FDIA time range              : "
        f"[{env.t_fdia_min}, {env.t_fdia_max}] h"
    )
    print(
        f"DoS time range               : "
        f"[{env.t_dos_min}, {env.t_dos_max}] h"
    )
    print(
        "DoS action                   : "
        "same source/compressor SCADA lock as coordinated"
    )
    print(
        "Ablation rule                : "
        "freeze coordinated Za, T_FDIA and T_DoS; remove components only"
    )
    print(
        "DND lower solver             : "
        "model-distributed neurodynamics"
    )

    # ------------------------------------------------------------------
    # 1) Optimize the coordinated attack ONCE.
    # ------------------------------------------------------------------
    print("\n" + "-" * 96)
    print("Optimizing the coordinated attack once ...")

    project_main.restore_env_snapshot(env, env_snapshot)
    optimization_initial = project_main.clone_state_dict(state_snapshot)

    coordinated, fitness, convergence, metric_curve = (
        project_main._run_upper_optimizer(
            env,
            "coordinated",
            optimization_initial,
            args,
        )
    )

    coordinated_budget = validate_strategy_budget(
        env, coordinated, "Coordinated"
    )

    print(
        "Selected coordinated strategy: "
        f"T_FDIA={float(coordinated['T_fdia']):.2f} h | "
        f"T_DoS={float(coordinated['T_dos']):.2f} h | "
        f"FDIA scale={selected_fdia_scale(coordinated):.4f} | "
        f"FDIA max ratio="
        f"{coordinated_budget['FDIA_max_ratio_pct']:.4f}% | "
        f"selected budget="
        f"{coordinated_budget['FDIA_budget_limit_pct']:.4f}%"
    )

    # ------------------------------------------------------------------
    # 2) Freeze that strategy and remove attack components.
    # ------------------------------------------------------------------
    final_strategies = build_fixed_ablation_strategies(
        env, coordinated
    )

    for strategy in final_strategies:
        validate_strategy_budget(env, strategy, strategy["name"])

    print("\nFixed ablation cases")
    for s in final_strategies:
        t_fdia = float(s["T_fdia"])
        t_dos = float(s["T_dos"])
        scale = selected_fdia_scale(s)
        print(
            f"{s['name']:12s} | "
            f"T_FDIA={'-' if t_fdia >= 90.0 else f'{t_fdia:.2f}'} | "
            f"T_DoS={'-' if t_dos >= 90.0 else f'{t_dos:.2f}'} | "
            f"FDIA scale="
            f"{'-' if s['name'] in {'Baseline', 'DoS-only'} else f'{scale:.4f}'}"
        )

    # ------------------------------------------------------------------
    # 3) Clean-state replay of all four cases with the same lower solver.
    # ------------------------------------------------------------------
    print("\n" + "-" * 96)
    print("Final clean-state replay of the four fixed ablation cases ...")

    project_main.restore_env_snapshot(env, env_snapshot)
    final_initial = project_main.clone_state_dict(state_snapshot)

    result = (
        model_distributed_nd_simulation
        .simulate_strategies_model_distributed_nd_batched(
            env,
            final_initial,
            final_strategies,
            max_ode_steps=args.max_ode_steps,
            tolerance=args.tolerance,
            return_trajectories=True,
        )
    )

    out_dir = Path(args.out_dir) / SYSTEM_NAME / args.upper_method
    out_dir.mkdir(parents=True, exist_ok=True)

    scenario_labels = [s["name"] for s in final_strategies]

    rows = []
    for i, strategy in enumerate(final_strategies):
        budget = validate_strategy_budget(
            env, strategy, strategy["name"]
        )
        t_fdia = float(strategy["T_fdia"])
        t_dos = float(strategy["T_dos"])
        scale = selected_fdia_scale(strategy)

        dos_lp_raw = float(result.metrics["dos_lp_at_trigger"][i])
        min_after_dos_raw = float(
            result.metrics["min_lp_after_dos_pct"][i]
        )
        post_drop_raw = float(
            result.metrics["post_dos_drop_pct"][i]
        )

        row = {
            "scenario": strategy["name"],
            "ablation_type": (
                "fixed coordinated strategy; component removal only"
            ),
            "T_FDIA_h": None if t_fdia >= 90.0 else t_fdia,
            "T_DoS_h": None if t_dos >= 90.0 else t_dos,
            "FDIA_duration_h": _active_duration(t_fdia),
            "DoS_duration_h": _active_duration(t_dos),
            "FDIA_selected_scale": (
                None
                if strategy["name"] in {"Baseline", "DoS-only"}
                else scale
            ),
            **budget,
            "total_load_shedding": float(result.damages[i]),
            "min_linepack_pct": float(
                result.metrics["min_lp_pct"][i]
            ),
            "dos_linepack_at_trigger_pct": (
                None if dos_lp_raw < 0.0 else dos_lp_raw
            ),
            "min_linepack_after_dos_pct": (
                None
                if min_after_dos_raw < 0.0
                else min_after_dos_raw
            ),
            "post_dos_drop_pct": (
                None if post_drop_raw < 0.0 else post_drop_raw
            ),
        }
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(
        out_dir / "attack_ablation_summary.csv",
        index=False,
    )

    save_trajectory_table(
        result,
        scenario_labels,
        out_dir / "attack_ablation_trajectories.csv",
    )

    save_search_curve(
        metric_curve,
        out_dir / "search_curve_coordinated.csv",
    )

    json_payload = {
        "system": SYSTEM_NAME,
        "lower_solver": "model-distributed DND",
        "ablation_design": {
            "type": "fixed coordinated strategy component-removal ablation",
            "coordinated_strategy_optimized_once": True,
            "single_attack_cases_reoptimized": False,
            "fdia_only_rule": (
                "inherit coordinated FDIA vector, FDIA scale, and T_FDIA; "
                "disable DoS"
            ),
            "dos_only_rule": (
                "inherit coordinated T_DoS and DoS target/action; "
                "disable FDIA"
            ),
            "baseline_rule": "disable both FDIA and DoS",
            "vulnerable_window_note": (
                "The vulnerable window is a coordinated-attack feasibility "
                "condition. DoS-only retains the coordinated T_DoS even if "
                "removing FDIA causes the trigger-time linepack to lie outside "
                "that window."
            ),
            "dos_fixed_duration": False,
            "dos_duration_note": (
                "Original paper model: DoS remains active from T_DoS "
                "through hour 24."
            ),
        },
        "coordinated_optimization": {
            "optimizer_fitness": float(fitness),
            "convergence": convergence,
            "T_FDIA_h": float(coordinated["T_fdia"]),
            "T_DoS_h": float(coordinated["T_dos"]),
            "FDIA_selected_scale": selected_fdia_scale(coordinated),
            **coordinated_budget,
            "FDIA_vector": strategy_fdia_vector(
                coordinated, env.power.num_buses
            ).astype(float).tolist(),
            "meta": coordinated.get("meta", {}),
        },
        "final_replay": rows,
    }

    with open(
        out_dir / "attack_ablation_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            json_payload,
            f,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    make_ablation_plot(
        summary_df.iloc[1:].copy(),
        out_dir / "attack_ablation_118-135.png",
    )

    display_cols = [
        "scenario",
        "T_FDIA_h",
        "T_DoS_h",
        "FDIA_selected_scale",
        "FDIA_max_ratio_pct",
        "FDIA_budget_limit_pct",
        "FDIA_budget_ok",
        "total_load_shedding",
        "min_linepack_pct",
        "dos_linepack_at_trigger_pct",
        "min_linepack_after_dos_pct",
    ]

    print("\nFixed-strategy ablation result")
    print(summary_df[display_cols].to_string(index=False))

    # Helpful effect-size summary, without changing any paper metric.
    try:
        values = summary_df.set_index("scenario")
        coord_ls = float(
            values.loc["Coordinated", "total_load_shedding"]
        )
        dos_ls = float(
            values.loc["DoS-only", "total_load_shedding"]
        )
        fdia_ls = float(
            values.loc["FDIA-only", "total_load_shedding"]
        )

        print("\nAblation interpretation")
        print(
            f"  FDIA-only load shedding : {fdia_ls:.6f}"
        )
        print(
            f"  DoS-only load shedding  : {dos_ls:.6f}"
        )
        print(
            f"  Coordinated shedding    : {coord_ls:.6f}"
        )

        if dos_ls > 1.0e-12:
            amp = 100.0 * (coord_ls - dos_ls) / dos_ls
            print(
                "  Coordinated vs DoS-only : "
                f"{amp:+.2f}% load-shedding change"
            )
    except Exception:
        pass

    print(f"\nSaved to: {out_dir}")
    print("  attack_ablation_summary.csv")
    print("  attack_ablation_summary.json")
    print("  attack_ablation_118-135.png")
    print("  attack_ablation_trajectories.csv")
    print("  search_curve_coordinated.csv")


if __name__ == "__main__":
    main()

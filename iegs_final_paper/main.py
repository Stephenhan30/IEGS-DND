"""Main experiment workflow for FDIA–DoS attacks in the IEEE 118-bus/GasLib-135 IEGS."""
from __future__ import annotations

import argparse
import json
import os
import random
import time
import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from attacker.fdia_generator import FDIAGenerator
from env_physics.iegs_coupled import IEGSSystem
from simulation import simulate_strategies_batched
from initial_state import create_initial_physical_state
from utils import load_system_data
from plot_utils import (
    DOUBLE_COLUMN_FIGSIZE,
    DOUBLE_COLUMN_TWO_PANEL_FIGSIZE,
    save_publication_figure,
)
try:
    from algorithm.mip_nd_master import MIPNDOptimizer
except Exception:  # Keep other modes runnable even if MIP-ND module is not installed yet.
    MIPNDOptimizer = None


def set_seed(seed: int) -> None:
    if not torch.cuda.is_available():
        try:
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def clone_state_dict(state: dict) -> dict:
    """Deep clone physical state dictionary."""
    out = {}

    for k, v in state.items():
        if torch.is_tensor(v):
            out[k] = v.clone()
        else:
            out[k] = copy.deepcopy(v)

    return out


def make_env_snapshot(env) -> dict:
    """
    Save all mutable env parameters that may be modified during simulation or initialization.

    Baseline should not depend on upper-level search settings.  This snapshot prevents
    optimization from contaminating the final scenario evaluation.
    """
    snap = {}

    # Power-side mutable tensors
    snap["power_base_load"] = env.power.base_load.clone()
    snap["gen_pmax"] = env.power.gen_pmax.clone()
    snap["gen_pmin"] = env.power.gen_pmin.clone()
    snap["gen_ramp_max"] = env.power.gen_ramp_max.clone()
    snap["line_capacity_max"] = env.power.line_capacity_max.clone()

    # Gas-side mutable tensors
    snap["gas_base_load"] = env.gas.base_gas_load.clone()
    snap["linepack_K_p"] = env.gas.linepack_K_p.clone()
    snap["weymouth_C_p"] = env.gas.weymouth_C_p.clone()
    snap["pi_min"] = env.gas.pi_min.clone()
    snap["pi_max"] = env.gas.pi_max.clone()
    snap["pi_warn"] = env.gas.pi_warn.clone()
    snap["pi_trip"] = env.gas.pi_trip.clone()
    snap["S_min"] = env.gas.S_min.clone()
    snap["S_max"] = env.gas.S_max.clone()

    if hasattr(env.gas, "comp_ratio_min"):
        snap["comp_ratio_min"] = env.gas.comp_ratio_min.clone()

    if hasattr(env.gas, "comp_ratio_max"):
        snap["comp_ratio_max"] = env.gas.comp_ratio_max.clone()

    # Coupling / cost tensors
    snap["psi_g"] = env.psi_g.clone()
    snap["c_g"] = env.c_g.clone()
    snap["c_w"] = env.c_w.clone()

    # Scalar parameters that may be changed by experiments or optimizers.
    # Saving them prevents optimization runs from contaminating final scenario evaluation.
    scalar_names = [
        "dispatch_attack_gain",
        "fdia_max_ratio",
        "linepack_balance_scale",
        "normal_inventory_restore_gain",
        "replenish_inventory_restore_gain",
        "normal_restore_flow_cap",
        "replenish_restore_flow_cap",
        "replenish_recovery_rate",
        "normal_source_multiplier_min",
        "normal_source_multiplier_max",
        "replenish_multiplier_max",
        "normal_source_tracking_rate",
        "normal_source_ramp_fraction",
        "replenish_source_tracking_rate",
        "replenish_source_ramp_fraction",
        "deadlock_drain_rate",
        "gfpp_pressure_derate_floor",
        "initial_pressure_ratio",
        "initial_source_reserve",
        "post_dos_drop_weight",
        "weak_post_dos_drop_penalty",
        "ode_reference_steps",
        "ode_min_steps",
        "ode_convergence_patience",
        "ode_rel_tolerance",
        "ode_grad_tolerance",
        "ode_energy_tolerance",
        "ode_energy_convergence_patience",
        "record_nd_convergence",
        "record_nd_hour",
        "mp_nd_max_fdia_vectors",
        "mip_nd_top_k",
        "mip_nd_time_limit",
        "mip_nd_mip_gap",
        "mip_nd_cplex_log",
        "mip_nd_allow_fallback",
        "mip_nd_target_gap",
        "mip_nd_dos_center",
        "mip_nd_fdia_center",
        "mip_nd_fdia_weight",
        "mip_nd_timing_weight",
        "mip_nd_fdia_scales",
        "t_fdia_min",
        "t_fdia_max",
        "t_dos_min",
        "t_dos_max",
        "min_attack_gap",
        "dos_window_target_margin",
        "dos_window_width",
        "single_attack_target_lp",
        "coordinated_damage_weight",
        "coordinated_min_lp_weight",
        "normal_lp_ratio",
        "lp_replenish_ratio",
        "lp_trip_ratio",
        "lp_warn_ratio",
        "lp_shutdown_ratio",
        "fast_inner_steps_cap",
        "mdnd_parallel_models",
        "mdnd_parallel_backend",
        "mdnd_consensus_rel_tolerance",
        "mdnd_state_rel_tolerance",
        "mdnd_convergence_patience",
        "mdnd_lr_decay",
        "mdnd_consensus_relaxation",
        "mdnd_dual_enabled",
        "mdnd_dual_damping",
        "mdnd_dual_clip",
        "nd_trace_effective_interval",
        "admm_rho",
        "admm_outer_iters",
        "admm_abs_tolerance",
        "admm_rel_tolerance",
        "admm_consensus_rel_tolerance",
        "admm_dual_rel_tolerance",
        "admm_min_outer_iters",
        "admm_convergence_patience",
        "admm_electric_weight",
        "admm_gas_weight",
        "admm_linepack_withdrawal_fraction",
        "admm_gas_security_floor_ratio",
        "admm_gas_budget_margin",
        "mip_nd_refinement_cache_enabled",
    ]
    snap["scalar_params"] = {
        name: copy.deepcopy(getattr(env, name))
        for name in scalar_names
        if hasattr(env, name)
    }

    return snap


def restore_env_snapshot(env, snap: dict) -> None:
    """Restore env mutable tensors from snapshot."""
    env.power.base_load = snap["power_base_load"].clone()
    env.power.gen_pmax = snap["gen_pmax"].clone()
    env.power.gen_pmin = snap["gen_pmin"].clone()
    env.power.gen_ramp_max = snap["gen_ramp_max"].clone()
    env.power.line_capacity_max = snap["line_capacity_max"].clone()

    env.gas.base_gas_load = snap["gas_base_load"].clone()
    env.gas.linepack_K_p = snap["linepack_K_p"].clone()
    env.gas.weymouth_C_p = snap["weymouth_C_p"].clone()
    env.gas.pi_min = snap["pi_min"].clone()
    env.gas.pi_max = snap["pi_max"].clone()
    env.gas.pi_warn = snap["pi_warn"].clone()
    env.gas.pi_trip = snap["pi_trip"].clone()
    env.gas.S_min = snap["S_min"].clone()
    env.gas.S_max = snap["S_max"].clone()

    if "comp_ratio_min" in snap:
        env.gas.comp_ratio_min = snap["comp_ratio_min"].clone()

    if "comp_ratio_max" in snap:
        env.gas.comp_ratio_max = snap["comp_ratio_max"].clone()

    env.psi_g = snap["psi_g"].clone()
    env.c_g = snap["c_g"].clone()
    env.c_w = snap["c_w"].clone()

    for name, value in snap.get("scalar_params", {}).items():
        setattr(env, name, copy.deepcopy(value))


def _apply_linepack_thresholds(
    env: IEGSSystem,
    normal_ratio: float = 0.90,
    replenish_ratio: float = 0.60,
    trip_ratio: float = 0.30,
) -> None:
    env.normal_lp_ratio = float(normal_ratio)
    env.lp_replenish_ratio = float(replenish_ratio)
    env.lp_trip_ratio = float(trip_ratio)
    env.lp_warn_ratio = env.lp_replenish_ratio
    env.lp_shutdown_ratio = env.lp_trip_ratio


def configure_env(env: IEGSSystem, target_system: str) -> None:
    """Experiment-level parameters.

    These values define physical/control parameters, not trajectories.  If a
    given dataset needs calibration, tune these parameters and the gas scaling
    factors in initial_state.py.
    """
    if target_system != "118-135":
        raise ValueError("This cleaned package only keeps the 118-135 system.")
    env.lambda1 = 50000.0
    env.lambda2 = 50000.0
    env.fdia_max_ratio = 0.20

    # MIP-ND timing search bounds.
    env.t_fdia_min = 1.0
    env.t_fdia_max = 18.0
    env.t_dos_min = 1.0
    env.t_dos_max = 20.0
    env.min_attack_gap = 1.0
    env.mp_nd_max_fdia_vectors = 24

    # MIP-ND upper candidate-generation settings.
    # These are used only when --upper-method mip-nd is selected;
    # kept for compatibility with earlier parameter naming.
    env.mip_nd_top_k = 128
    env.mip_nd_time_limit = 10.0
    env.mip_nd_mip_gap = 0.0
    env.mip_nd_cplex_log = False
    env.mip_nd_allow_fallback = False
    env.mip_nd_fdia_scales = [0.60, 0.75, 0.90, 1.00]
    env.mip_nd_target_gap = 5.0
    env.mip_nd_dos_center = 10.0
    env.mip_nd_fdia_center = max(env.t_fdia_min, env.mip_nd_dos_center - env.mip_nd_target_gap)
    env.mip_nd_fdia_weight = 1.0
    env.mip_nd_timing_weight = 1.0

    # Coordinated attack should prefer DoS near the replenishment window.
    env.dos_window_target_margin = 12.0
    env.dos_window_width = 8.0
    env.single_attack_target_lp = 45.0
    env.coordinated_damage_weight = 50.0
    env.coordinated_min_lp_weight = 0.1

    # Lower-level neurodynamic convergence criteria used for plotting and
    # paper-style stopping checks: gradient criterion and energy-stability criterion.
    env.ode_grad_tolerance = 1e-4
    env.ode_energy_tolerance = 1e-4
    env.ode_energy_convergence_patience = 8
    env.record_nd_convergence = False
    env.record_nd_hour = None

    # Model-distributed ND execution.  The electric and gas local models use
    # the same iteration-k snapshot and exchange only GFPP coupling proposals.
    # CUDA uses parallel streams.  On CPU, auto selects deterministic Jacobi
    # execution because two PyTorch worker threads cause severe oversubscription
    # on this 118-135 case; cpu_threads remains available explicitly.
    env.mdnd_parallel_models = True
    env.mdnd_parallel_backend = "auto"

    # Stabilized distributed evolution.  Diminishing pseudo-time steps suppress
    # the terminal Adam oscillation, while the coordinator projects both local
    # GFPP proposals onto a common feasible consensus point.  Convergence is
    # judged from the pre-projection proposal mismatch and projected-state
    # change for several consecutive checks.
    env.mdnd_lr_decay = 0.08
    env.mdnd_consensus_relaxation = 1.0
    env.mdnd_dual_enabled = False
    env.mdnd_dual_damping = 0.90
    env.mdnd_dual_clip = 1.0e4
    env.mdnd_consensus_rel_tolerance = 1e-3
    env.mdnd_state_rel_tolerance = 1e-3
    env.mdnd_convergence_patience = 5
    env.nd_trace_effective_interval = 5.0

    # Physics-aware ADMM comparison defaults.  Each coordination iteration
    # solves the GFPP boundary subproblems and then recomputes the complete
    # electric and gas operating states.
    env.admm_rho = 10.0
    env.admm_outer_iters = 200
    env.admm_abs_tolerance = 1e-4
    env.admm_rel_tolerance = 1e-3
    env.admm_consensus_rel_tolerance = 1e-3
    env.admm_dual_rel_tolerance = 1e-3
    env.admm_min_outer_iters = 10
    env.admm_convergence_patience = 5
    env.admm_state_rel_tolerance = 1e-3
    env.admm_relaxation = 0.65
    env.admm_electric_weight = 1.0
    env.admm_gas_weight = 1.0
    env.admm_linepack_withdrawal_fraction = 0.12
    env.admm_gas_security_floor_ratio = 0.005
    env.admm_gas_budget_margin = 0.0

    # Reuse exact duplicate attack responses produced by overlapping local
    # timing-refinement neighborhoods.
    env.mip_nd_refinement_cache_enabled = True

    env.c_g[:] = 0.02
    if len(env.gfpp_indices) > 0:
        env.c_g[env.gfpp_indices] = 0.15
    env.c_w[:] = 0.02

    env.dispatch_attack_gain = 5.00

    _apply_linepack_thresholds(env, normal_ratio=0.90, replenish_ratio=0.60, trip_ratio=0.30)

    env.normal_inventory_restore_gain = 0.55
    env.replenish_inventory_restore_gain = 1.00
    env.normal_restore_flow_cap = 0.12
    env.replenish_restore_flow_cap = 0.30
    env.replenish_recovery_rate = 0.04
    env.normal_source_multiplier_min = 0.94
    env.normal_source_multiplier_max = 1.10
    env.replenish_multiplier_max = 1.18

    env.normal_source_tracking_rate = 0.48
    env.normal_source_ramp_fraction = 0.07
    env.replenish_source_tracking_rate = 0.75
    env.replenish_source_ramp_fraction = 0.16

    env.deadlock_drain_rate = 0.0
    env.gfpp_pressure_derate_floor = 1.0
    env.linepack_balance_scale = 0.14

    env.fast_inner_steps_cap = 60
    env.initial_pressure_ratio = 0.70
    env.initial_source_reserve = 1.02
    env.post_dos_drop_weight = 1500.0
    env.weak_post_dos_drop_penalty = 1.0e5
    env.ode_reference_steps = 300
    env.ode_min_steps = 30
    env.ode_convergence_patience = 8
    env.ode_rel_tolerance = 1e-4


def make_fixed_fdia(env: IEGSSystem, r_d_max: float = 0.20) -> np.ndarray:
    fdia_gen = FDIAGenerator(env.power)
    base_load = env.power.base_load.detach().cpu().numpy()
    mean_load = np.mean(base_load[base_load > 0]) if np.any(base_load > 0) else 1.0

    za = np.zeros(env.power.num_buses, dtype=np.float32)
    target_buses = list(getattr(env, "attack_target_buses", []))

    if not target_buses:
        target_buses = [
            env.power.bus_id_to_idx[int(env.power.gen_df.iloc[g]["Node"])]
            for g in env.gfpp_indices
        ]

    for b in target_buses:
        za[b] = r_d_max * max(base_load[b], 0.05 * mean_load)

    non_targets = [i for i in range(env.power.num_buses) if i not in target_buses]
    if non_targets:
        weights = base_load[non_targets].astype(float)
        if np.sum(weights) <= 1e-9:
            weights = np.ones(len(non_targets), dtype=float)
        weights = weights / np.sum(weights)
        total_pos = float(np.sum(za[target_buses]))
        for idx, b in enumerate(non_targets):
            za[b] = -total_pos * weights[idx]

    legal, _ = fdia_gen.check_and_project_fdia(za, r_d_max)
    return legal.astype(np.float32)


def _clamp_hour(x: float) -> float:
    return float(np.clip(x, 1.0, 24.0))


def _snap_to_time_grid(
    t: float,
    step: float = 1.0,
    t_min: float = 1.0,
    t_max: float = 24.0,
) -> float:
    """
    Snap an attack time to the discrete upper-level scheduling grid.

    The paper model treats attack timing as discrete binary decisions.
    Therefore, final evaluated times should lie on the same time grid.

    Example with step=1h:
        4.0 -> 4.0
        4.2 -> 5.0
        4.9 -> 5.0
    """
    if t >= 90.0:
        return float(t)

    step = max(float(step), 1e-9)
    t = float(t)
    t_min = float(t_min)
    t_max = float(t_max)

    k = np.ceil((t - t_min) / step - 1e-9)
    snapped = t_min + float(k) * step
    return float(np.clip(snapped, t_min, t_max))


def make_fixed_timing_strategies(coord_za: np.ndarray, t_fdia: float, t_dos: float) -> list[dict]:
    return [
        {"name": "FDIA -1h, DoS same", "FDIA": coord_za, "T_fdia": _clamp_hour(t_fdia - 1.0), "T_dos": _clamp_hour(t_dos)},
        {"name": "FDIA same, DoS -1h", "FDIA": coord_za, "T_fdia": _clamp_hour(t_fdia), "T_dos": _clamp_hour(t_dos - 1.0)},
        {"name": "FDIA +1h, DoS same", "FDIA": coord_za, "T_fdia": _clamp_hour(t_fdia + 1.0), "T_dos": _clamp_hour(t_dos)},
        {"name": "FDIA same, DoS +1h", "FDIA": coord_za, "T_fdia": _clamp_hour(t_fdia), "T_dos": _clamp_hour(t_dos + 1.0)},
    ]
def refine_coordinated_timing(
    env: IEGSSystem,
    env_snapshot: dict,
    initial_state_snapshot: dict,
    coord_za: np.ndarray,
    coord_strat: dict,
    args,
) -> dict:
    """
    Local timing refinement around the MIP-ND coordinated solution.

    It evaluates nearby FDIA/DoS timings and selects the candidate with:
        1. valid replenishment window: 30% < DoS LP <= 60%;
        2. maximum load shedding;
        3. lower min LP as tie-breaker.

    FDIA attack vector is kept unchanged.
    """

    base_t_fdia = float(coord_strat["T_fdia"])
    base_t_dos = float(coord_strat["T_dos"])

    t_fdia_min = float(getattr(env, "t_fdia_min", 1.0))
    t_fdia_max = float(getattr(env, "t_fdia_max", 18.0))
    t_dos_min = float(getattr(env, "t_dos_min", 1.0))
    t_dos_max = float(getattr(env, "t_dos_max", 20.0))
    min_gap = float(getattr(env, "min_attack_gap", 1.0))

    # Local neighborhood for final timing refinement.
    # The upper-level attack timing is modeled as discrete scheduling periods.
    # Therefore refinement also uses the same discrete time grid instead of
    # fractional offsets such as 0.25h or 0.75h.
    refine_step = float(getattr(args, "time_grid_step", 1.0))
    refine_step = max(refine_step, 1e-9)
    refine_radius = int(getattr(args, "refine_radius", 1))
    offsets = [i * refine_step for i in range(-refine_radius, refine_radius + 1)]

    candidates = []
    seen = set()

    for df in offsets:
        for dd in offsets:
            t_fdia = _snap_to_time_grid(
                base_t_fdia + df,
                step=refine_step,
                t_min=t_fdia_min,
                t_max=t_fdia_max,
            )
            t_dos = _snap_to_time_grid(
                base_t_dos + dd,
                step=refine_step,
                t_min=t_dos_min,
                t_max=t_dos_max,
            )

            # Keep FDIA before DoS.
            if t_dos < t_fdia + min_gap:
                continue

            key = (round(t_fdia, 4), round(t_dos, 4))
            if key in seen:
                continue
            seen.add(key)

            candidates.append(
                {
                    "name": f"refine_fdia{df:+.1f}_dos{dd:+.1f}",
                    "FDIA": coord_za,
                    "T_fdia": t_fdia,
                    "T_dos": t_dos,
                }
            )

    if not candidates:
        return coord_strat

    restore_env_snapshot(env, env_snapshot)
    initial_state_for_refine = clone_state_dict(initial_state_snapshot)

    refine_result = simulate_strategies_batched(
        env,
        initial_state_for_refine,
        candidates,
        max_ode_steps=args.max_ode_steps,
        tolerance=args.tolerance,
        return_trajectories=False,
    )

    lp_rep = float(getattr(env, "lp_replenish_ratio", 0.60)) * 100.0
    lp_trip = float(getattr(env, "lp_trip_ratio", 0.30)) * 100.0

    best_idx = None
    best_key = None

    for i, cand in enumerate(candidates):
        shed = float(refine_result.damages[i])
        min_lp = float(refine_result.metrics["min_lp_pct"][i])
        dos_lp = float(refine_result.metrics["dos_lp_at_trigger"][i])

        window_pass = (dos_lp > lp_trip) and (dos_lp <= lp_rep)

        if not window_pass:
            continue

        # Primary objective: maximum shedding.
        # Tie-breaker: lower min LP.
        score_key = (shed, -min_lp)

        if best_key is None or score_key > best_key:
            best_key = score_key
            best_idx = i

    if best_idx is None:
        print("\n[Local refinement] No valid window candidate found. Keep original MIP-ND timing.")
        return coord_strat

    best_cand = candidates[best_idx]
    best_shed = float(refine_result.damages[best_idx])
    best_min_lp = float(refine_result.metrics["min_lp_pct"][best_idx])
    best_dos_lp = float(refine_result.metrics["dos_lp_at_trigger"][best_idx])

    old_t_fdia = float(coord_strat["T_fdia"])
    old_t_dos = float(coord_strat["T_dos"])

    refined = dict(coord_strat)
    refined["T_fdia"] = float(best_cand["T_fdia"])
    refined["T_dos"] = float(best_cand["T_dos"])
    refined["za"] = coord_za
    refined["refinement"] = {
        "old_T_fdia": old_t_fdia,
        "old_T_dos": old_t_dos,
        "new_T_fdia": float(best_cand["T_fdia"]),
        "new_T_dos": float(best_cand["T_dos"]),
        "shed": best_shed,
        "min_lp_pct": best_min_lp,
        "dos_lp_pct": best_dos_lp,
        "window_pass": True,
    }

    print("\n[Local refinement]")
    print(f"  old timing: FDIA={old_t_fdia:.2f}h, DoS={old_t_dos:.2f}h")
    print(f"  new timing: FDIA={refined['T_fdia']:.2f}h, DoS={refined['T_dos']:.2f}h")
    print(f"  refined shed={best_shed:.2f}, min LP={best_min_lp:.2f}%, DoS LP={best_dos_lp:.2f}%")

    return refined

def save_trajectory_csv(result, labels: list[str], out_dir: str, prefix: str = "") -> None:
    rows = []
    for label, traj in zip(labels, result.trajectories):
        for item in traj:
            rows.append({"scenario": label, **item})

    base = f"{prefix}_" if prefix else ""
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, f"{base}trajectories.csv"), index=False, encoding="utf-8-sig")

    metrics = pd.DataFrame({"scenario": labels, "total_shed": result.damages})
    for key, values in result.metrics.items():
        metrics[key] = values
    metrics.to_csv(os.path.join(out_dir, f"{base}summary_metrics.csv"), index=False, encoding="utf-8-sig")


def save_upper_search_curve_csv(
    convergence: list[float],
    metric_curve: list[dict] | None,
    out_dir: str,
    target_system: str,
) -> str:
    """Save the upper-level incumbent history needed by comparison figures."""
    os.makedirs(out_dir, exist_ok=True)

    if metric_curve:
        curve_df = pd.DataFrame(metric_curve)
        if "best_feasible_load_shedding" in curve_df.columns:
            values = pd.to_numeric(
                curve_df["best_feasible_load_shedding"], errors="coerce"
            ).to_numpy(dtype=float)
            values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
            curve_df["incumbent_best_feasible_load_shedding"] = np.maximum.accumulate(values)
        elif "best_score" in curve_df.columns:
            values = pd.to_numeric(curve_df["best_score"], errors="coerce").to_numpy(dtype=float)
            values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
            curve_df["incumbent_best_score"] = np.maximum.accumulate(values)
    else:
        values = np.asarray(convergence, dtype=float)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        curve_df = pd.DataFrame({
            "evaluated": np.arange(1, len(values) + 1, dtype=int),
            "best_score": values,
            "incumbent_best_score": np.maximum.accumulate(values) if len(values) else values,
        })

    path = os.path.join(out_dir, f"upper_search_curve_{target_system}.csv")
    curve_df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def _get_hours(result) -> list[int]:
    return [x["time"] for x in result.trajectories[0]]


def _linepack_at_period_start(item: dict) -> float:
    """Linepack at the beginning of a scheduling period.

    DoS is activated before the period-t equilibrium update, so this is the
    value used by ``dos_lp_at_trigger`` and by the vulnerability-window test.
    Older trajectory files fall back to the former post-update ``lp_pct``.
    """
    return float(item.get("lp_pct_pre", item["lp_pct"]))


def _effective_time(t_raw: float) -> float:
    """
    Convert continuous attack time to the effective discrete simulation time.

    Example:
        T = 4.2h, dt = 1h  -> effective time = 5h
        T = 4.0h           -> effective time = 4h
    """
    if t_raw >= 90.0:
        return t_raw
    return float(np.ceil(t_raw))


def _plot_attack_times(coord_strat: dict) -> None:
    """Mark the optimized coordinated FDIA and DoS activation times."""
    coord_fdia_raw = float(coord_strat.get("T_fdia", 99.0))
    coord_dos_raw = float(coord_strat.get("T_dos", 99.0))

    coord_fdia_eff = _effective_time(coord_fdia_raw)
    coord_dos_eff = _effective_time(coord_dos_raw)

    if coord_fdia_raw < 90.0:
        plt.axvline(
            coord_fdia_eff,
            linestyle=":",
            linewidth=1.8,
            alpha=0.85,
            label=f"FDIA eff t={coord_fdia_eff:.0f}h",
        )

    if coord_dos_raw < 90.0:
        plt.axvline(
            coord_dos_eff,
            # Use an explicit dash-dot cycle so the line style is identical
            # in the plot and in the legend.
            linestyle=(0, (7.0, 3.0, 1.5, 3.0)),
            linewidth=1.8,
            alpha=0.85,
            dash_capstyle="butt",
            label=f"DoS eff t={coord_dos_eff:.0f}h",
        )


def plot_main_results(
    result,
    labels: list[str],
    convergence: list[float],
    coord_strat: dict,
    out_dir: str,
    target_system: str,
    env: IEGSSystem,
    metric_curve: list[dict] | None = None,
) -> None:
    """Plot baseline/coordinated physical responses and upper search convergence.

    The former FDIA-only and DoS-only optimization/ablation experiments are no
    longer part of the experiment workflow.
    """
    os.makedirs(out_dir, exist_ok=True)
    label_to_idx = {label: i for i, label in enumerate(labels)}
    hours = _get_hours(result)

    coord_label = next(
        (x for x in labels if x.startswith("Coordinated")),
        "Coordinated (MIP-ND)",
    )
    compare_order = ["Baseline", coord_label]
    style_map = {
        "Baseline": ("--^", 2.1),
        "Coordinated (MIP-ND)": ("-o", 2.4),
        "Coordinated (MIP-ADMM)": ("-o", 2.4),
    }

    # Linepack plot.
    plt.figure(figsize=DOUBLE_COLUMN_FIGSIZE)
    for name in compare_order:
        idx = label_to_idx[name]
        marker, lw = style_map.get(name, ("-o", 2.4))
        plt.plot(
            hours,
            [_linepack_at_period_start(x) for x in result.trajectories[idx]],
            marker,
            linewidth=lw,
            label=name,
        )

    plt.axhline(
        env.lp_replenish_ratio * 100.0,
        linestyle="--",
        linewidth=2.0,
        alpha=0.8,
        label="Replenishment Threshold",
    )
    plt.axhline(
        env.lp_trip_ratio * 100.0,
        linestyle="-.",
        linewidth=2.0,
        alpha=0.8,
        label="Trip Threshold",
    )
    _plot_attack_times(coord_strat)

    # Mark the exact pre-DoS linepack used by the vulnerability-window test.
    coord_idx = label_to_idx[coord_label]
    coord_dos_lp = float(result.metrics["dos_lp_at_trigger"][coord_idx])
    coord_dos_eff = _effective_time(float(coord_strat.get("T_dos", 99.0)))
    if coord_dos_lp >= 0.0 and coord_dos_eff < 90.0:
        plt.scatter(
            [coord_dos_eff],
            [coord_dos_lp],
            marker="o",
            s=46,
            facecolors="white",
            edgecolors="black",
            linewidths=1.2,
            zorder=8,
            label=f"DoS trigger LP={coord_dos_lp:.1f}%",
        )
        plt.annotate(
            f"{coord_dos_lp:.1f}%",
            xy=(coord_dos_eff, coord_dos_lp),
            xytext=(5, 7),
            textcoords="offset points",
            fontsize=7,
        )

    plt.xlabel("Scheduling period (hour)", fontweight="bold")
    plt.ylabel("Linepack at period start (%)", fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.25),
        ncol=3,
        fontsize=7.0,
        handlelength=4.5,
        columnspacing=1.2,
        handletextpad=0.6,
    )
    plt.tight_layout()
    save_publication_figure(plt.gcf(), os.path.join(out_dir, f"linepack_comparison_{target_system}"))
    plt.close()

    # Load shedding plot.
    plt.figure(figsize=DOUBLE_COLUMN_FIGSIZE)
    for name in compare_order:
        idx = label_to_idx[name]
        marker, lw = style_map.get(name, ("-o", 2.4))
        plt.plot(
            hours,
            [x["step_shed"] for x in result.trajectories[idx]],
            marker,
            linewidth=lw,
            label=name,
        )

    _plot_attack_times(coord_strat)
    plt.xlabel("Time (hour)", fontweight="bold")
    plt.ylabel("Electric Load Shedding", fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.25),
        ncol=3,
        fontsize=7.0,
        handlelength=4.5,
        columnspacing=1.2,
        handletextpad=0.6,
    )
    plt.tight_layout()
    save_publication_figure(plt.gcf(), os.path.join(out_dir, f"shedding_comparison_{target_system}"))
    plt.close()

    # Upper-level search convergence.
    # The incumbent-best sequence is piecewise constant, so a step curve is
    # more faithful than a point-by-point line.  Markers are drawn only where
    # the incumbent value actually improves.
    method_name = "MIP-ND"
    plt.figure(figsize=DOUBLE_COLUMN_FIGSIZE)

    if metric_curve:
        curve_df = pd.DataFrame(metric_curve)
        x = curve_df["evaluated"].astype(float).to_numpy()

        if "best_feasible_load_shedding" in curve_df.columns:
            y_raw = curve_df["best_feasible_load_shedding"].astype(float).to_numpy()
            y_raw = np.nan_to_num(y_raw, nan=0.0, posinf=0.0, neginf=0.0)
            y = np.maximum.accumulate(y_raw)
            curve_df["incumbent_best_feasible_load_shedding"] = y
            ylabel = "Best Feasible Load Shedding (MWh)"
        else:
            y_raw = curve_df["best_score"].astype(float).to_numpy()
            y_raw = np.nan_to_num(y_raw, nan=0.0, posinf=0.0, neginf=0.0)
            y = np.maximum.accumulate(y_raw)
            curve_df["incumbent_best_score"] = y
            ylabel = f"Best {method_name} Score"

        curve_df.to_csv(
            os.path.join(out_dir, f"upper_search_curve_{target_system}.csv"),
            index=False,
            encoding="utf-8-sig",
        )

        if len(y) > 0:
            step_line = plt.step(
                x,
                y,
                where="post",
                linewidth=2.2,
            )[0]

            # Mark only genuine incumbent updates.  The initial zero value is
            # not marked unless the first recorded value is already positive.
            change_tol = 1e-9 * max(1.0, float(np.max(np.abs(y))))
            change_mask = np.zeros(len(y), dtype=bool)

            change_mask[0] = not np.isclose(
                y[0],
                0.0,
                atol=change_tol,
                rtol=0.0,
            )

            if len(y) > 1:
                change_mask[1:] = ~np.isclose(
                    y[1:],
                    y[:-1],
                    atol=change_tol,
                    rtol=1e-9,
                )

            if np.any(change_mask):
                plt.plot(
                    x[change_mask],
                    y[change_mask],
                    linestyle="None",
                    marker="o",
                    markersize=5,
                    color=step_line.get_color(),
                )

        plt.xlabel("Evaluated Candidate Strategies", fontweight="bold")
        plt.ylabel(ylabel, fontweight="bold")

    else:
        x = np.arange(1, len(convergence) + 1, dtype=float)
        y_raw = np.asarray(convergence, dtype=float)
        y_raw = np.nan_to_num(y_raw, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.maximum.accumulate(y_raw) if len(y_raw) > 0 else y_raw

        if len(y) > 0:
            step_line = plt.step(
                x,
                y,
                where="post",
                linewidth=2.2,
            )[0]

            change_tol = 1e-9 * max(1.0, float(np.max(np.abs(y))))
            change_mask = np.zeros(len(y), dtype=bool)

            change_mask[0] = not np.isclose(
                y[0],
                0.0,
                atol=change_tol,
                rtol=0.0,
            )

            if len(y) > 1:
                change_mask[1:] = ~np.isclose(
                    y[1:],
                    y[:-1],
                    atol=change_tol,
                    rtol=1e-9,
                )

            if np.any(change_mask):
                plt.plot(
                    x[change_mask],
                    y[change_mask],
                    linestyle="None",
                    marker="o",
                    markersize=5,
                    color=step_line.get_color(),
                )

        plt.xlabel("Iteration", fontweight="bold")
        plt.ylabel("Best Fitness", fontweight="bold")

    plt.grid(True, alpha=0.3)
    plt.margins(x=0.02, y=0.08)
    plt.tight_layout()
    save_publication_figure(
        plt.gcf(),
        os.path.join(out_dir, f"upper_convergence_{target_system}"),
    )
    plt.close()

def plot_lower_nd_convergence(
    env: IEGSSystem,
    env_snapshot: dict,
    initial_state_snapshot: dict,
    coord_strat: dict,
    coord_za: np.ndarray,
    out_dir: str,
    target_system: str,
    args,
) -> None:
    """
    Plot lower-level neurodynamic convergence diagnostics using the two
    Chapter-3 criteria in their constrained numerical form:
        1) gradient criterion: projected-gradient residual;
        2) energy-stability criterion: relative |E(x_{tau+dt}) - E(x_tau)|.

    This function runs one additional single-strategy simulation only for
    visualization.  It does not change the optimized attack strategy or any
    summary metric.
    """
    os.makedirs(out_dir, exist_ok=True)

    if getattr(args, "nd_record_hour", None) is not None:
        record_hour = int(args.nd_record_hour)
    else:
        # Default: diagnose the lower solver at the pre-DoS scheduling period.
        # This avoids the discontinuity introduced exactly at the DoS lock point.
        record_hour = max(1, int(np.ceil(float(coord_strat.get("T_dos", 1.0)))) - 1)

    restore_env_snapshot(env, env_snapshot)
    env.record_nd_convergence = True
    env.record_nd_hour = record_hour
    env.nd_convergence_trace = []

    probe_strategy = [
        {
            "name": "Coordinated convergence probe",
            "FDIA": coord_za,
            "T_fdia": float(coord_strat["T_fdia"]),
            "T_dos": float(coord_strat["T_dos"]),
        }
    ]

    initial_state_for_probe = clone_state_dict(initial_state_snapshot)

    _ = simulate_strategies_batched(
        env,
        initial_state_for_probe,
        probe_strategy,
        max_ode_steps=args.max_ode_steps,
        tolerance=args.tolerance,
        return_trajectories=False,
    )

    trace = list(getattr(env, "nd_convergence_trace", []))
    env.record_nd_convergence = False
    env.record_nd_hour = None

    if len(trace) == 0:
        print("[Warning] No lower-level ND convergence trace was recorded.")
        return

    df = pd.DataFrame(trace)
    csv_path = os.path.join(out_dir, f"lower_nd_convergence_{target_system}.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 2, figsize=DOUBLE_COLUMN_TWO_PANEL_FIGSIZE)

    # Projected-state convergence criterion.  Plot the actual dimensionless
    # residual rather than normalizing it by the first sample; this lets the
    # tolerance crossing be read directly from the figure.
    x = df["effective_step"].astype(float).to_numpy()
    if "projected_gradient_residual" in df.columns:
        grad_y = df["projected_gradient_residual"].astype(float).to_numpy()
        grad_label = "Projected-state residual"
        grad_ylabel = "Relative Projected-State Residual"
    else:
        grad_y = df["gradient_norm"].astype(float).to_numpy()
        grad_label = "Gradient RMS norm"
        grad_ylabel = "Gradient RMS Norm"

    grad_y_plot = np.maximum(np.asarray(grad_y, dtype=float), 1e-16)
    axes[0].semilogy(x, grad_y_plot, "-o", linewidth=2.2, markersize=3, label=grad_label)
    tol_col = "state_relative_tolerance" if "state_relative_tolerance" in df.columns else "projected_gradient_tolerance"
    if tol_col in df.columns:
        tol_series = df[tol_col].dropna()
        if len(tol_series) > 0:
            axes[0].axhline(
                float(tol_series.iloc[-1]),
                linestyle="--",
                linewidth=1.8,
                label="Convergence tolerance",
            )
    axes[0].set_xlabel("Neurodynamic Iteration (effective step)", fontweight="bold")
    axes[0].set_ylabel(grad_ylabel, fontweight="bold")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend()

    # Energy-stability criterion plot.  Prefer the true relative energy-change
    # column.  Older CSVs used energy_abs_change for this relative quantity, so
    # keep a fallback for compatibility.
    energy_col = "energy_rel_change" if "energy_rel_change" in df.columns else "energy_abs_change"
    energy_df = df.dropna(subset=[energy_col]).copy()
    if len(energy_df) > 0:
        ex = energy_df["effective_step"].astype(float).to_numpy()
        ey = np.maximum(energy_df[energy_col].astype(float).to_numpy(), 1e-16)
        axes[1].semilogy(ex, ey, "-o", linewidth=2.2, markersize=3, label="Relative energy change")
        if "energy_tolerance" in energy_df.columns:
            tol_series = energy_df["energy_tolerance"].dropna()
            if len(tol_series) > 0:
                axes[1].axhline(
                    float(tol_series.iloc[-1]),
                    linestyle="--",
                    linewidth=1.8,
                    label="Stability tolerance",
                )
    else:
        axes[1].plot([], [], "-o", label="Relative energy change")

    axes[1].set_xlabel("Neurodynamic Iteration (effective step)", fontweight="bold")
    axes[1].set_ylabel("Relative Energy Change", fontweight="bold")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    stop_reason = str(df["stop_reason"].iloc[-1]) if "stop_reason" in df.columns else "unknown"
    fig.tight_layout()
    save_publication_figure(fig, os.path.join(out_dir, f"lower_nd_convergence_{target_system}"))
    plt.close(fig)

    # Electric/gas local energy-function convergence plot.
    if {"electric_energy_value", "gas_energy_value"}.issubset(df.columns):
        energy_df = df.dropna(subset=["electric_energy_value", "gas_energy_value"]).copy()
        if len(energy_df) > 0:
            ex = energy_df["effective_step"].astype(float).to_numpy()
            e_energy = energy_df["electric_energy_value"].astype(float).to_numpy()
            g_energy = energy_df["gas_energy_value"].astype(float).to_numpy()

            # Normalize for display only, so electric and gas values with
            # different units/scales can be shown in one figure.
            e_base = max(abs(float(e_energy[0])), 1e-12)
            g_base = max(abs(float(g_energy[0])), 1e-12)
            e_energy_norm = e_energy / e_base
            g_energy_norm = g_energy / g_base

            plt.figure(figsize=DOUBLE_COLUMN_FIGSIZE)
            plt.plot(ex, e_energy_norm, "-o", linewidth=2.2, markersize=3, label="Electric submodel energy")
            plt.plot(ex, g_energy_norm, "-s", linewidth=2.2, markersize=3, label="Gas submodel energy")
            plt.xlabel("Neurodynamic Iteration (effective step)", fontweight="bold")
            plt.ylabel("Normalized Energy Value", fontweight="bold")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            save_publication_figure(plt.gcf(), os.path.join(out_dir, f"lower_nd_energy_components_{target_system}"))
            plt.close()

    # GFPP distributed-consensus convergence plot.  Prefer the relative
    # disagreement between the independent local proposals before the
    # coordinator projection.  This is scale-safe and is the same quantity used
    # by the DND stopping test.
    residual_col = None
    for candidate in (
        "gt_proposal_relative_consensus_residual",
        "gt_relative_consensus_residual",
        "gt_consensus_residual",
        "max_gt_consensus_residual",
    ):
        if candidate in df.columns:
            residual_col = candidate
            break

    if residual_col is not None:
        res_df = df.dropna(subset=[residual_col]).copy()
        if len(res_df) > 0:
            rx = res_df["effective_step"].astype(float).to_numpy()
            ry = np.maximum(res_df[residual_col].astype(float).to_numpy(), 1e-16)

            plt.figure(figsize=DOUBLE_COLUMN_FIGSIZE)
            plt.semilogy(
                rx,
                ry,
                "-o",
                linewidth=2.2,
                markersize=3,
                label="GFPP proposal consensus residual",
            )

            tol_col = (
                "gt_relative_consensus_tolerance"
                if "gt_relative_consensus_tolerance" in res_df.columns
                else "gt_consensus_tolerance"
            )
            if tol_col in res_df.columns:
                tol_series = res_df[tol_col].dropna()
                if len(tol_series) > 0:
                    plt.axhline(
                        float(tol_series.iloc[-1]),
                        linestyle="--",
                        linewidth=1.8,
                        label="Convergence tolerance",
                    )

            plt.xlabel("Neurodynamic Iteration (effective step)", fontweight="bold")
            plt.ylabel("Relative Coupling Residual", fontweight="bold")
            plt.grid(True, which="both", alpha=0.3)
            plt.legend()
            plt.tight_layout()
            save_publication_figure(plt.gcf(), os.path.join(out_dir, f"lower_nd_coupling_residual_{target_system}"))
            plt.close()

    last = df.iloc[-1]
    projected_residual = (
        float(last["projected_gradient_residual"])
        if "projected_gradient_residual" in df.columns and pd.notna(last["projected_gradient_residual"])
        else float(last["gradient_norm"])
    )
    print(
        "[ND convergence] "
        f"hour={record_hour} | stop={stop_reason} | "
        f"proj_res={projected_residual:.3e} | "
        f"cons_rel={float(last.get('gt_proposal_relative_consensus_residual', last.get('gt_relative_consensus_residual', float('nan')))) if pd.notna(last.get('gt_proposal_relative_consensus_residual', last.get('gt_relative_consensus_residual', float('nan')))) else float('nan'):.3e} | "
        f"rel_dE={float(last.get('energy_rel_change', last.get('energy_abs_change', float('nan')))) if pd.notna(last.get('energy_rel_change', last.get('energy_abs_change', float('nan')))) else float('nan'):.3e} | "
        f"saved to {csv_path}"
    )


def plot_fixed_timing_results(result, labels: list[str], out_dir: str, target_system: str, env: IEGSSystem) -> None:
    os.makedirs(out_dir, exist_ok=True)
    hours = _get_hours(result)
    label_to_idx = {label: i for i, label in enumerate(labels)}

    style_map = {
        "FDIA -1h, DoS same": ("--D", 2.0),
        "FDIA same, DoS -1h": ("--s", 2.0),
        "FDIA +1h, DoS same": ("-.^", 2.0),
        "FDIA same, DoS +1h": ("-.v", 2.0),
    }

    plt.figure(figsize=DOUBLE_COLUMN_FIGSIZE)
    for label in labels:
        marker, lw = style_map.get(label, ("-o", 2.0))
        idx = label_to_idx[label]
        plt.plot(hours, [_linepack_at_period_start(x) for x in result.trajectories[idx]], marker, linewidth=lw, label=label)

    plt.axhline(env.lp_replenish_ratio * 100.0, linestyle="--", linewidth=2.0, alpha=0.8, label="Replenishment Threshold")
    plt.axhline(env.lp_trip_ratio * 100.0, linestyle="-.", linewidth=2.0, alpha=0.8, label="Trip Threshold")
    plt.xlabel("Scheduling period (hour)", fontweight="bold")
    plt.ylabel("Linepack at period start (%)", fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.25),
        ncol=2,
        fontsize=7.0,
        handlelength=4.5,
        columnspacing=1.2,
        handletextpad=0.6,
    )
    plt.tight_layout()
    save_publication_figure(plt.gcf(), os.path.join(out_dir, f"timing_sensitivity_linepack_{target_system}"))
    plt.close()

    plt.figure(figsize=DOUBLE_COLUMN_FIGSIZE)
    for label in labels:
        marker, lw = style_map.get(label, ("-o", 2.0))
        idx = label_to_idx[label]
        plt.plot(hours, [x["step_shed"] for x in result.trajectories[idx]], marker, linewidth=lw, label=label)

    plt.xlabel("Time (hour)", fontweight="bold")
    plt.ylabel("Electric Load Shedding", fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.25),
        ncol=2,
        fontsize=7.0,
        handlelength=4.5,
        columnspacing=1.2,
        handletextpad=0.6,
    )
    plt.tight_layout()
    save_publication_figure(plt.gcf(), os.path.join(out_dir, f"timing_sensitivity_shedding_{target_system}"))
    plt.close()


def _run_mip_nd_optimizer(env, mode: str, initial_state: dict, args):
    if MIPNDOptimizer is None:
        raise ImportError(
            "--upper-method mip-nd requires algorithm/mip_nd_master.py. "
            "Please add the MIP-ND optimizer module before running this mode."
        )

    # Batch-wise local timing refinement settings used inside MIPNDOptimizer.
    env.mip_nd_batch_refine_step = float(getattr(args, "time_grid_step", args.mp_time_step))
    env.mip_nd_batch_refine_radius = int(getattr(args, "refine_radius", 1))

    opt = MIPNDOptimizer(
        env,
        mode=mode,
        time_step=args.mp_time_step,
        chunk_size=args.mp_chunk_size,
        seed=args.seed,
    )
    out = opt.optimize(
        initial_state,
        max_ode_steps=args.max_ode_steps,
        tolerance=args.tolerance,
    )
    return out.strategy, out.fitness, out.convergence, out.metric_curve


def _run_upper_optimizer(env, mode: str, initial_state: dict, args, seed_offset: int = 0):
    if args.upper_method == "mip-nd":
        return _run_mip_nd_optimizer(env, mode, initial_state, args)
    raise ValueError("The cleaned project only keeps --upper-method mip-nd")


def _upper_method_name(args) -> str:
    return "MIP-ND"


def _coordinated_label(args) -> str:
    return f"Coordinated ({_upper_method_name(args)})"


def _extract_fdia_vector(strat: dict) -> np.ndarray:
    if "za" in strat:
        return np.asarray(strat["za"], dtype=np.float32)
    if "FDIA" in strat:
        fdia = strat["FDIA"]
        if isinstance(fdia, (list, tuple)):
            return np.asarray(fdia[-1], dtype=np.float32)
        return np.asarray(fdia, dtype=np.float32)
    raise KeyError("Strategy does not contain 'za' or 'FDIA'.")


def run_experiment(args) -> None:
    set_seed(args.seed)

    print("=" * 88)
    print(f"{_upper_method_name(args)} IEGS attack experiment | system={args.system} | device={'cuda' if torch.cuda.is_available() else 'cpu'}")
    print("=" * 88)

    system_data = load_system_data(target_system=args.system, data_dir=args.data_dir)
    env = IEGSSystem(system_data, dt_hours=1.0)
    configure_env(env, args.system)
    lower_solver_kind = str(getattr(args, "lower_solver_kind", "mdnd"))

    if lower_solver_kind == "standard_admm":
        env.admm_rho = float(getattr(args, "admm_rho", env.admm_rho))
        env.admm_outer_iters = int(getattr(args, "admm_outer_iters", env.admm_outer_iters))
        env.admm_abs_tolerance = float(getattr(args, "admm_abs_tolerance", env.admm_abs_tolerance))
        env.admm_rel_tolerance = float(getattr(args, "admm_rel_tolerance", env.admm_rel_tolerance))
        env.admm_consensus_rel_tolerance = env.admm_rel_tolerance
        env.admm_dual_rel_tolerance = env.admm_rel_tolerance
        env.admm_min_outer_iters = int(getattr(args, "admm_min_outer_iters", env.admm_min_outer_iters))
        env.admm_convergence_patience = int(getattr(args, "admm_convergence_patience", env.admm_convergence_patience))
        env.admm_state_rel_tolerance = float(getattr(args, "admm_state_rel_tolerance", env.admm_state_rel_tolerance))
        env.admm_relaxation = float(getattr(args, "admm_relaxation", env.admm_relaxation))
        env.admm_electric_weight = float(getattr(args, "admm_electric_weight", env.admm_electric_weight))
        env.admm_gas_weight = float(getattr(args, "admm_gas_weight", env.admm_gas_weight))
        env.admm_linepack_withdrawal_fraction = float(
            getattr(args, "admm_linepack_withdrawal_fraction", env.admm_linepack_withdrawal_fraction)
        )
        env.admm_gas_security_floor_ratio = float(
            getattr(args, "admm_gas_security_floor_ratio", env.admm_gas_security_floor_ratio)
        )
        env.admm_gas_budget_margin = float(
            getattr(args, "admm_gas_budget_margin", env.admm_gas_budget_margin)
        )

    # Optional command-line overrides for MIP-ND.
    # MIP-ND upper-level parameters.
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

    env.mdnd_parallel_models = not bool(getattr(args, "disable_mdnd_parallel", False))
    env.mdnd_parallel_backend = str(getattr(args, "mdnd_parallel_backend", "auto"))
    env.mdnd_consensus_rel_tolerance = float(
        getattr(args, "mdnd_consensus_rel_tolerance", 1e-3)
    )
    env.mdnd_state_rel_tolerance = float(
        getattr(args, "mdnd_state_rel_tolerance", 1e-3)
    )
    env.mdnd_convergence_patience = max(
        1, int(getattr(args, "mdnd_convergence_patience", 5))
    )
    env.mdnd_lr_decay = max(0.0, float(getattr(args, "mdnd_lr_decay", 0.08)))
    env.mdnd_consensus_relaxation = min(
        1.0,
        max(0.0, float(getattr(args, "mdnd_consensus_relaxation", 1.0))),
    )
    env.nd_trace_effective_interval = float(
        getattr(args, "nd_check_effective_interval", 5.0)
    )
    env.mip_nd_refinement_cache_enabled = not bool(
        getattr(args, "disable_refinement_cache", False)
    )

    env.fast_dynamics = args.dynamics_mode == "fast"
    initial_state = create_initial_physical_state(env, args.system)
    # Save clean snapshots immediately after initialization.
    # These must be restored before each optimization and final scenario evaluation.
    env_snapshot = make_env_snapshot(env)
    initial_state_snapshot = clone_state_dict(initial_state)
    print(f"GFPP generators: {env.gfpp_indices}")
    print(f"GFPP-to-gas-node mapping: {env.gfpp_to_gas_node}")
    print(f"FDIA positive target buses: {env.attack_target_buses}")
    print(f"Linepack thresholds: normal={env.normal_lp_ratio*100:.1f}%, replenish={env.lp_replenish_ratio*100:.1f}%, trip={env.lp_trip_ratio*100:.1f}%")
    print(f"Upper method: {args.upper_method} ({_upper_method_name(args)})")
    if lower_solver_kind == "standard_admm":
        print("Lower-level evaluator: physics-aware consensus ADMM")
        print(
            "Standard ADMM settings: "
            f"rho={getattr(env, 'admm_rho', 10.0):.3g} | "
            f"max_iter={getattr(env, 'admm_outer_iters', 200)} | "
            f"abs_tol={getattr(env, 'admm_abs_tolerance', 1e-4):.1e} | "
            f"rel_tol={getattr(env, 'admm_rel_tolerance', 1e-3):.1e} | "
            f"state_tol={getattr(env, 'admm_state_rel_tolerance', 1e-3):.1e} | "
            f"relax={getattr(env, 'admm_relaxation', 0.65):.2f} | "
            f"patience={getattr(env, 'admm_convergence_patience', 5)}"
        )
        print("Per iteration: GFPP local QPs + complete electric/gas state replay; no ND/Adam inner iterations")
    elif lower_solver_kind == "aladin":
        print("Lower-level evaluator: physics-aware distributed ALADIN")
        print(
            "ALADIN settings: "
            f"rho={getattr(env, 'aladin_rho', 5.0):.3g} | "
            f"mu={getattr(env, 'aladin_mu', 20.0):.3g} | "
            f"max_iter={getattr(env, 'aladin_outer_iters', 60)} | "
            f"consensus_tol={getattr(env, 'aladin_consensus_rel_tolerance', 1e-3):.1e} | "
            f"step_tol={getattr(env, 'aladin_step_rel_tolerance', 1e-3):.1e} | "
            f"kkt_tol={getattr(env, 'aladin_kkt_rel_tolerance', 1e-3):.1e} | "
            f"state_tol={getattr(env, 'aladin_state_rel_tolerance', 1e-3):.1e}"
        )
        print("Local solves + second-order coordination QP; no ND/Adam or ADMM updates")
    else:
        print(f"Dynamics mode: {args.dynamics_mode} ({'short inner ODE' if args.dynamics_mode == 'fast' else 'full inner ODE'})")
        print(f"ODE reference steps: {getattr(env, 'ode_reference_steps', 'NA')} | max ODE steps: {args.max_ode_steps}")
        print(
            f"Model-distributed ND parallel: enabled={getattr(env, 'mdnd_parallel_models', True)} | "
            f"backend={getattr(env, 'mdnd_parallel_backend', 'auto')}"
        )
        print(
            "Model-distributed ND stopping: "
            f"state_rel_tol={getattr(env, 'mdnd_state_rel_tolerance', 1e-3):.1e} | "
            f"consensus_rel_tol={getattr(env, 'mdnd_consensus_rel_tolerance', 1e-3):.1e} | "
            f"patience={getattr(env, 'mdnd_convergence_patience', 5)} checks | "
            f"lr_decay={getattr(env, 'mdnd_lr_decay', 0.08):.3f} | "
            f"consensus_relax={getattr(env, 'mdnd_consensus_relaxation', 1.0):.2f} | "
            f"check_interval={getattr(env, 'nd_trace_effective_interval', 5.0)} effective steps"
        )
    print(
        f"Local refinement response cache: "
        f"enabled={getattr(env, 'mip_nd_refinement_cache_enabled', True)}"
    )
    if args.upper_method == "mip-nd":
        print(f"MIP-ND timing step: {args.mp_time_step}h | chunk size: {args.mp_chunk_size}")
        print(
            f"MIP-ND candidates: top_k={getattr(env, 'mip_nd_top_k', 'NA')} | "
            f"scales={getattr(env, 'mip_nd_fdia_scales', 'NA')} | "
            f"CPLEX time limit={getattr(env, 'mip_nd_time_limit', 'NA')}s"
        )

    start = time.time()

    print("\nOptimizing coordinated FDIA-DoS attack ...")
    restore_env_snapshot(env, env_snapshot)
    initial_state_for_coord = clone_state_dict(initial_state_snapshot)

    coord_strat, coord_fit, coord_conv, coord_metric_curve = _run_upper_optimizer(
        env,
        "coordinated",
        initial_state_for_coord,
        args,
        seed_offset=0,
    )

    print(f"\n✅ {_upper_method_name(args)} coordinated search completed in {time.time() - start:.1f}s")
    print(
        f"Coordinated: fitness={coord_fit:.2f}, "
        f"FDIA={coord_strat['T_fdia']:.2f}h, DoS={coord_strat['T_dos']:.2f}h"
    )
    print(f"Best coordinated FDIA residual meta: {coord_strat.get('meta', {})}")

    coord_za = _extract_fdia_vector(coord_strat)

    # MIP-ND performs local timing refinement inside the upper search only when
    # a candidate batch produces a new global incumbent.  Do not apply the old
    # one-shot final refinement again here; otherwise the convergence curve and
    # the final Scenario summary would refer to different search stages.
    print(
        "\n[Incumbent refinement] Local timing refinement is triggered only when "
        "a MIP-ND batch produces a new incumbent; final one-shot refinement is skipped."
    )

    coord_za = _extract_fdia_vector(coord_strat) if ("za" in coord_strat or "FDIA" in coord_strat) else coord_za

    # Final evaluation uses discrete upper-level attack times.
    # This keeps plots and summary metrics consistent with the
    # mixed-integer timing formulation in the paper.
    time_step = float(getattr(args, "time_grid_step", 1.0))
    coord_strat["T_fdia"] = _snap_to_time_grid(
        coord_strat["T_fdia"],
        step=time_step,
        t_min=float(getattr(env, "t_fdia_min", 1.0)),
        t_max=float(getattr(env, "t_fdia_max", 18.0)),
    )
    coord_strat["T_dos"] = _snap_to_time_grid(
        coord_strat["T_dos"],
        step=time_step,
        t_min=float(getattr(env, "t_dos_min", 1.0)),
        t_max=float(getattr(env, "t_dos_max", 20.0)),
    )
    coord_label = _coordinated_label(args)
    strategies = [
        {
            "name": "Baseline",
            "FDIA": np.zeros(env.power.num_buses, dtype=np.float32),
            "T_fdia": 99.0,
            "T_dos": 99.0,
        },
        {
            "name": coord_label,
            "FDIA": coord_za,
            "T_fdia": coord_strat["T_fdia"],
            "T_dos": coord_strat["T_dos"],
        },
    ]

    print("\nEvaluating baseline and coordinated-attack scenarios ...")
    restore_env_snapshot(env, env_snapshot)
    initial_state_for_final = clone_state_dict(initial_state_snapshot)
    result = simulate_strategies_batched(env, initial_state_for_final, strategies, max_ode_steps=args.max_ode_steps, tolerance=args.tolerance, return_trajectories=True)
    labels = [s["name"] for s in strategies]

    fixed_strategies = [
        {"name": coord_label, "FDIA": coord_za, "T_fdia": coord_strat["T_fdia"], "T_dos": coord_strat["T_dos"]},
        *make_fixed_timing_strategies(coord_za, coord_strat["T_fdia"], coord_strat["T_dos"]),
    ]

    print("Evaluating fixed timing sensitivity scenarios ...")
    restore_env_snapshot(env, env_snapshot)
    initial_state_for_timing = clone_state_dict(initial_state_snapshot)
    fixed_result = simulate_strategies_batched(env, initial_state_for_timing, fixed_strategies, max_ode_steps=args.max_ode_steps, tolerance=args.tolerance, return_trajectories=True)
    fixed_labels = [s["name"] for s in fixed_strategies]

    out_dir = os.path.join(args.out_dir, args.system, args.upper_method)
    os.makedirs(out_dir, exist_ok=True)

    save_trajectory_csv(result, labels, out_dir)
    save_trajectory_csv(fixed_result, fixed_labels, out_dir, prefix="timing_sensitivity")

    make_method_figures = bool(getattr(args, "make_method_figures", True))
    if make_method_figures:
        plot_main_results(
            result,
            labels,
            coord_conv,
            coord_strat,
            out_dir,
            args.system,
            env,
            metric_curve=coord_metric_curve,
        )
        plot_fixed_timing_results(fixed_result, fixed_labels, out_dir, args.system, env)
    else:
        save_upper_search_curve_csv(
            coord_conv, coord_metric_curve, out_dir, args.system
        )

    if bool(getattr(args, "record_lower_trace", True)):
        plot_lower_nd_convergence(
            env=env,
            env_snapshot=env_snapshot,
            initial_state_snapshot=initial_state_snapshot,
            coord_strat=coord_strat,
            coord_za=coord_za,
            out_dir=out_dir,
            target_system=args.system,
            args=args,
        )

    # Final evaluation metrics are recomputed after local timing refinement and time-grid snapping.
    # The optimizer fitness below is kept as the upper-level search value, while final physical
    # damages / linepack metrics are stored explicitly to avoid mixing pre-refinement fitness with
    # post-refinement attack times.
    coord_idx = labels.index(coord_label)

    summary = {
        "system": args.system,
        "upper_method": args.upper_method,
        "pop_size": args.pop_size,
        "max_iter": args.max_iter,
        "max_ode_steps": args.max_ode_steps,
        "mp_time_step": args.mp_time_step,
        "mp_chunk_size": args.mp_chunk_size,
        "mip_top_k": int(getattr(env, "mip_nd_top_k", -1)),
        "mip_time_limit": float(getattr(env, "mip_nd_time_limit", -1.0)),
        "mip_gap": float(getattr(env, "mip_nd_mip_gap", -1.0)),
        "mip_solver": "CPLEX/docplex" if args.upper_method == "mip-nd" else "NA",
        "mip_fdia_scales": list(getattr(env, "mip_nd_fdia_scales", [])),
        "time_grid_step": args.time_grid_step,
        "refine_radius": args.refine_radius,
        "dynamics_mode": args.dynamics_mode,
        "mdnd_convergence": {
            "state_rel_tolerance": float(getattr(env, "mdnd_state_rel_tolerance", 1e-3)),
            "consensus_rel_tolerance": float(getattr(env, "mdnd_consensus_rel_tolerance", 1e-3)),
            "patience": int(getattr(env, "mdnd_convergence_patience", 5)),
            "lr_decay": float(getattr(env, "mdnd_lr_decay", 0.08)),
            "consensus_relaxation": float(getattr(env, "mdnd_consensus_relaxation", 1.0)),
            "dual_enabled": bool(getattr(env, "mdnd_dual_enabled", False)),
            "check_effective_interval": float(getattr(env, "nd_trace_effective_interval", 5.0)),
        },
        "mdnd_performance_stats": copy.deepcopy(
            getattr(env, "mdnd_performance_stats", {})
        ),
        "refinement_cache_enabled": bool(
            getattr(env, "mip_nd_refinement_cache_enabled", True)
        ),
        "linepack_thresholds": {
            "normal": float(env.normal_lp_ratio),
            "replenish": float(env.lp_replenish_ratio),
            "trip": float(env.lp_trip_ratio),
        },
        "best": {
            "coordinated": {
                "optimizer_fitness": float(coord_fit),
                "optimizer_fitness_note": "Upper-level best fitness after batch-wise local timing refinement; final physical metrics are recomputed in the final scenario replay.",
                "T_fdia": float(coord_strat["T_fdia"]),
                "T_dos": float(coord_strat["T_dos"]),
                "FDIA_vector": coord_za.astype(float).tolist(),
                "fdia_l1": float(np.sum(np.abs(coord_za))),
                "fdia_linf": float(np.max(np.abs(coord_za)) if coord_za.size else 0.0),
                "meta": coord_strat.get("meta", {}),
                "refinement": coord_strat.get("refinement", {}),
                "final_total_shed": float(result.damages[coord_idx]),
                "final_min_lp_pct": float(result.metrics["min_lp_pct"][coord_idx]),
                "final_dos_lp_at_trigger_pct": float(result.metrics["dos_lp_at_trigger"][coord_idx]),
                "final_min_lp_after_dos_pct": float(result.metrics["min_lp_after_dos_pct"][coord_idx]),
            },
        },
        "damages": {label: float(dmg) for label, dmg in zip(labels, result.damages)},
        "metrics": {k: {label: float(v[i]) for i, label in enumerate(labels)} for k, v in result.metrics.items()},
        "timing_sensitivity_damages": {label: float(dmg) for label, dmg in zip(fixed_labels, fixed_result.damages)},
        "timing_sensitivity_metrics": {k: {label: float(v[i]) for i, label in enumerate(fixed_labels)} for k, v in fixed_result.metrics.items()},
    }

    if lower_solver_kind == "standard_admm":
        summary["lower_solver"] = "physics_aware_consensus_admm"
        summary.pop("mdnd_convergence", None)
        summary.pop("mdnd_performance_stats", None)
        summary["admm_convergence"] = {
            "rho": float(getattr(env, "admm_rho", 10.0)),
            "outer_iterations": int(getattr(env, "admm_outer_iters", 200)),
            "absolute_tolerance": float(getattr(env, "admm_abs_tolerance", 1e-4)),
            "relative_tolerance": float(getattr(env, "admm_rel_tolerance", 1e-3)),
            "minimum_iterations": int(getattr(env, "admm_min_outer_iters", 10)),
            "patience": int(getattr(env, "admm_convergence_patience", 5)),
            "state_relative_tolerance": float(getattr(env, "admm_state_rel_tolerance", 1e-3)),
            "relaxation": float(getattr(env, "admm_relaxation", 0.65)),
            "electric_local_weight": float(getattr(env, "admm_electric_weight", 1.0)),
            "gas_local_weight": float(getattr(env, "admm_gas_weight", 1.0)),
            "linepack_withdrawal_fraction": float(getattr(env, "admm_linepack_withdrawal_fraction", 0.12)),
            "gas_security_floor_ratio": float(getattr(env, "admm_gas_security_floor_ratio", 0.005)),
            "local_solver": "projected_boundary_QP_with_full_state_replay_each_iteration",
            "model_scope": "physics_aware_full_state_replay",
            "uses_neurodynamics": False,
            "uses_adam": False,
        }

    with open(os.path.join(out_dir, "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    lp_rep_pct = env.lp_replenish_ratio * 100.0
    lp_trip_pct = env.lp_trip_ratio * 100.0

    print("\nScenario summary")
    for label, dmg in zip(labels, result.damages):
        idx = labels.index(label)
        min_lp = float(result.metrics["min_lp_pct"][idx])
        dos_lp = float(result.metrics["dos_lp_at_trigger"][idx])

        if dos_lp < 0.0:
            window_text = "-"
        else:
            window_text = "PASS" if (dos_lp > lp_trip_pct and dos_lp <= lp_rep_pct) else "FAIL"

        print(
            f"  {label:<20} "
            f"shed={dmg:>10.2f} | "
            f"min LP={min_lp:>7.2f}% | "
            f"DoS LP={dos_lp:>7.2f}% | "
            f"window={window_text}"
        )

    print("\nTiming sensitivity summary")
    for label, dmg in zip(fixed_labels, fixed_result.damages):
        idx = fixed_labels.index(label)
        min_lp = float(fixed_result.metrics["min_lp_pct"][idx])
        dos_lp = float(fixed_result.metrics["dos_lp_at_trigger"][idx])

        if dos_lp < 0.0:
            window_text = "-"
        else:
            window_text = "PASS" if (dos_lp > lp_trip_pct and dos_lp <= lp_rep_pct) else "FAIL"

        print(
            f"  {label:<24} "
            f"shed={dmg:>10.2f} | "
            f"min LP={min_lp:>7.2f}% | "
            f"DoS LP={dos_lp:>7.2f}% | "
            f"window={window_text}"
        )

    mdnd_stats = getattr(env, "mdnd_performance_stats", {})
    if lower_solver_kind != "standard_admm" and isinstance(mdnd_stats, dict) and mdnd_stats:
        print("\nModel-distributed ND performance summary")
        print(
            f"  equilibrium calls       : {mdnd_stats.get('equilibrium_calls', 0)}"
        )
        print(
            f"  candidate-hours         : {mdnd_stats.get('candidate_hours', 0)}"
        )
        print(
            f"  raw ND steps            : {mdnd_stats.get('raw_nd_steps', 0)}"
        )
        print(
            f"  candidate-weighted steps: {mdnd_stats.get('candidate_weighted_steps', 0)}"
        )
        print(
            f"  max-step hits           : {mdnd_stats.get('max_steps_hits', 0)}"
        )
        print(
            f"  stop reasons            : {mdnd_stats.get('stop_reasons', {})}"
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", choices=["118-135"], default="118-135")
    parser.add_argument("--data-dir", default="data/")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--pop-size", type=int, default=30)
    parser.add_argument("--max-iter", type=int, default=30)
    parser.add_argument("--max-ode-steps", type=int, default=2500)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument(
        "--dynamics-mode",
        choices=["fast", "full"],
        default="full",
        help="Used by the ND runners; ignored by the standalone standard-ADMM runner.",
    )
    parser.add_argument(
        "--upper-method",
        choices=["mip-nd"],
        default="mip-nd",
        help="MIP upper-level candidate generator; the lower evaluator is selected by the runner script.",
    )
    parser.add_argument(
        "--mp-time-step",
        type=float,
        default=1.0,
        help="Time grid step in hours for MIP-ND upper timing enumeration.",
    )
    parser.add_argument(
        "--mp-chunk-size",
        type=int,
        default=8,
        help="Number of MIP-ND candidate strategies evaluated in one lower-level batch.",
    )
    parser.add_argument(
        "--mip-top-k",
        type=int,
        default=128,
        help="Number of Top-K candidate strategies generated by the MIP-ND upper master. Only used with --upper-method mip-nd.",
    )
    parser.add_argument(
        "--mip-fdia-scales",
        default="0.6,0.75,0.9,1.0",
        help="Comma-separated FDIA budget scales used by MIP-ND. Only used with --upper-method mip-nd.",
    )
    parser.add_argument(
        "--mip-time-limit",
        type=float,
        default=10.0,
        help="Time limit in seconds for each MIP solve in the MIP-ND upper master. Only used with --upper-method mip-nd.",
    )
    parser.add_argument(
        "--mip-gap",
        type=float,
        default=0.0,
        help="Relative MIP gap for the CPLEX MIP-ND upper master. Only used with --upper-method mip-nd.",
    )
    parser.add_argument(
        "--mip-cplex-log",
        action="store_true",
        help="Show CPLEX solve log for each MIP-ND upper-master solve.",
    )
    parser.add_argument(
        "--mip-allow-fallback",
        action="store_true",
        help="Allow deterministic fallback if CPLEX/docplex is unavailable. By default MIP-ND requires CPLEX.",
    )
    parser.add_argument(
        "--time-grid-step",
        type=float,
        default=1.0,
        help="Discrete scheduling time grid in hours for final evaluation and local refinement.",
    )
    parser.add_argument(
        "--refine-radius",
        type=int,
        default=1,
        help="Number of discrete time-grid steps used on each side during local timing refinement.",
    )
    parser.add_argument(
        "--mdnd-parallel-backend",
        choices=["auto", "cpu_threads", "cuda_streams", "sequential_jacobi"],
        default="auto",
        help=(
            "Execution backend for electric/gas local ND models. "
            "auto selects CUDA streams on GPU and deterministic Jacobi execution on CPU."
        ),
    )
    parser.add_argument(
        "--disable-mdnd-parallel",
        action="store_true",
        help="Disable concurrent local-model execution but keep Jacobi snapshot updates.",
    )
    parser.add_argument(
        "--mdnd-consensus-rel-tolerance",
        type=float,
        default=1e-3,
        help="Relative GFPP consensus residual tolerance for model-distributed ND.",
    )
    parser.add_argument(
        "--mdnd-state-rel-tolerance",
        type=float,
        default=1e-3,
        help="Relative projected-state residual tolerance for model-distributed ND.",
    )
    parser.add_argument(
        "--mdnd-convergence-patience",
        type=int,
        default=5,
        help="Consecutive convergence checks required before DND stops.",
    )
    parser.add_argument(
        "--mdnd-lr-decay",
        type=float,
        default=0.08,
        help="Diminishing pseudo-time learning-rate decay for model-distributed ND.",
    )
    parser.add_argument(
        "--mdnd-consensus-relaxation",
        type=float,
        default=1.0,
        help="GFPP consensus projection relaxation in [0, 1].",
    )
    parser.add_argument(
        "--nd-check-effective-interval",
        type=float,
        default=5.0,
        help="Effective pseudo-time interval between DND convergence checks.",
    )
    parser.add_argument(
        "--disable-refinement-cache",
        action="store_true",
        help="Disable exact-strategy response cache for overlapping local timing refinements.",
    )
    parser.add_argument(
        "--nd-record-hour",
        type=int,
        default=None,
        help="Physical hour used for lower-level convergence diagnostics. "
             "If omitted, the pre-DoS hour is used.",
    )
    # Optional standard-ADMM overrides.  They are used only by
    # run_admm_baseline.py; all default experiment commands remain short.
    parser.add_argument("--admm-rho", type=float, default=10.0)
    parser.add_argument("--admm-outer-iters", type=int, default=200)
    parser.add_argument("--admm-abs-tolerance", type=float, default=1e-4)
    parser.add_argument("--admm-rel-tolerance", type=float, default=1e-3)
    parser.add_argument("--admm-min-outer-iters", type=int, default=10)
    parser.add_argument("--admm-convergence-patience", type=int, default=5)
    parser.add_argument("--admm-state-rel-tolerance", type=float, default=1e-3)
    parser.add_argument("--admm-relaxation", type=float, default=0.65)
    parser.add_argument("--admm-electric-weight", type=float, default=1.0)
    parser.add_argument("--admm-gas-weight", type=float, default=1.0)
    parser.add_argument("--admm-linepack-withdrawal-fraction", type=float, default=0.12)
    parser.add_argument("--admm-gas-security-floor-ratio", type=float, default=0.005)
    parser.add_argument("--admm-gas-budget-margin", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    run_experiment(parse_args())

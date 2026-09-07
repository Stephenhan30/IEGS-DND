"""Batched 24-hour physical simulation shared by all lower-level solvers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from algorithm.ode_solver import BatchedNeurodynamicODESolver


@dataclass
class SimulationResult:
    damages: np.ndarray
    trajectories: list[list[dict]]
    metrics: dict[str, np.ndarray]


def _as_batch_state(initial_state: dict, batch_size: int) -> dict:
    return {
        k: v.unsqueeze(0).repeat(batch_size, *([1] * v.dim())).clone()
        for k, v in initial_state.items()
    }


def _strategy_fdia_tensor(
    strategies: list[dict],
    t: int,
    num_buses: int,
    device: torch.device,
) -> torch.Tensor:
    z_a = torch.zeros((len(strategies), num_buses), dtype=torch.float32, device=device)

    for i, strat in enumerate(strategies):
        if t >= float(strat.get("T_fdia", 99.0)):
            fdia = strat.get("FDIA", np.zeros(num_buses, dtype=np.float32))
            if isinstance(fdia, list):
                fdia = fdia[-1]
            z_a[i] = torch.tensor(fdia, dtype=torch.float32, device=device)

    return z_a


def load_multiplier(t: int) -> float:
    """Engineering 24-hour load multiplier.

    This is an exogenous demand profile, not a prescribed linepack curve.
    It intentionally includes both rising and falling periods so that the
    baseline trajectory can be produced by the ODE/controller dynamics rather
    than by a monotone demand ramp.
    """
    profile = np.array(
        [
            0.96, 0.95, 0.96, 0.98, 1.01, 1.05,
            1.08, 1.10, 1.07, 1.03, 0.99, 0.97,
            0.99, 1.03, 1.07, 1.10, 1.08, 1.04,
            1.01, 0.99, 0.97, 0.96, 0.95, 0.96,
        ],
        dtype=float,
    )
    idx = int(np.clip(t, 1, 24)) - 1
    return float(profile[idx])


def _min_gfpp_pressure(state: dict, gfpp_nodes: torch.Tensor) -> torch.Tensor:
    if "pi" not in state:
        batch = state[next(iter(state))].shape[0]
        return torch.zeros(batch, device=gfpp_nodes.device)
    return torch.min(state["pi"][:, gfpp_nodes], dim=1).values


# >>> [NEW-EXPORT] Extra trajectory fields used only for paper figures.
def _tensor_row_to_text(value: torch.Tensor) -> str:
    """Serialize one 1-D tensor as a compact semicolon-separated CSV field.

    This does not alter the physical model or optimization result.  It only
    exports node/pipe-level states so the plotting script can reconstruct
    mechanism and spatial-distribution figures without changing other files.
    """
    arr = value.detach().cpu().numpy().reshape(-1)
    return ";".join(f"{float(x):.10g}" for x in arr)
# <<< [NEW-EXPORT]


def _ramp_limited_source_update(
    iegs_env,
    prev_S: torch.Tensor,
    desired_S: torch.Tensor,
    lp_warning: torch.Tensor,
    s_t: torch.Tensor,
) -> torch.Tensor:
    """Ramp-limited SCADA source tracking.

    Normal zone:
        The source tracks demand and inventory restoration slowly.  This allows
        linepack to fluctuate under normal load changes and allows FDIA to
        consume linepack transiently.

    Replenishment zone:
        The source tracks faster, but still with a ramp limit.  It is not an
        instant valve-to-maximum jump.

    DoS:
        Source injection is strictly locked at previous state.
    """
    normal_track = float(getattr(iegs_env, "normal_source_tracking_rate", 0.58))
    replenish_track = float(getattr(iegs_env, "replenish_source_tracking_rate", 0.58))

    normal_ramp = float(getattr(iegs_env, "normal_source_ramp_fraction", 0.10))
    replenish_ramp = float(getattr(iegs_env, "replenish_source_ramp_fraction", 0.10))

    track_rate = torch.where(
        lp_warning,
        torch.ones_like(lp_warning, dtype=prev_S.dtype, device=prev_S.device) * replenish_track,
        torch.ones_like(lp_warning, dtype=prev_S.dtype, device=prev_S.device) * normal_track,
    )

    ramp_fraction = torch.where(
        lp_warning,
        torch.ones_like(lp_warning, dtype=prev_S.dtype, device=prev_S.device) * replenish_ramp,
        torch.ones_like(lp_warning, dtype=prev_S.dtype, device=prev_S.device) * normal_ramp,
    )

    delta_raw = desired_S - prev_S
    max_delta = torch.clamp(torch.abs(prev_S), min=1.0) * ramp_fraction.unsqueeze(-1)
    delta_limited = torch.maximum(torch.minimum(delta_raw, max_delta), -max_delta)
    ramped_S = prev_S + track_rate.unsqueeze(-1) * delta_limited

    return torch.where(s_t.unsqueeze(-1) == 1, ramped_S, prev_S)


def _source_target_with_inventory_control(
    iegs_env,
    base_S: torch.Tensor,
    expected_D: torch.Tensor,
    lp_pct_prev: torch.Tensor,
    lp_warning: torch.Tensor,
    normal_lp_ratio: float,
    lp_replenish_ratio: float,
    lp_trip_ratio: float,
    ref_lp: torch.Tensor,
    prev_lp_sum: torch.Tensor,
) -> torch.Tensor:
    """Build SCADA desired source dispatch with inventory feedback.

    The controller is not a linepack hard-code.  It uses current physical
    linepack as feedback and computes a desired gas-source injection:

        desired gas source = expected gas demand + inventory recovery flow

    Normal region: weak recovery toward normal_lp_ratio, so baseline naturally
    fluctuates around the normal inventory band instead of drifting downward.

    Replenishment region: stronger, still bounded, recovery.  The actual source
    command is applied later through tracking and ramp limits, so valves do not
    instantaneously open to maximum.
    """
    source_share = base_S / (torch.sum(base_S, dim=1, keepdim=True) + 1e-6)

    normal_restore_gain = float(getattr(iegs_env, "normal_inventory_restore_gain", 0.55))
    replenish_restore_gain = float(getattr(iegs_env, "replenish_inventory_restore_gain", 0.90))

    normal_cap = float(getattr(iegs_env, "normal_restore_flow_cap", 0.18))
    replenish_cap = float(getattr(iegs_env, "replenish_restore_flow_cap", 0.35))

    # Physical inventory target in linepack units.
    normal_lp_target = normal_lp_ratio * ref_lp
    lp_error = normal_lp_target - prev_lp_sum

    gain = torch.where(
        lp_warning,
        torch.ones_like(lp_pct_prev) * replenish_restore_gain,
        torch.ones_like(lp_pct_prev) * normal_restore_gain,
    )

    cap_fraction = torch.where(
        lp_warning,
        torch.ones_like(lp_pct_prev) * replenish_cap,
        torch.ones_like(lp_pct_prev) * normal_cap,
    )

    # Convert inventory error to equivalent gas flow.  Divide by balance_scale so
    # that a reasonable recovery command is visible after the linepack dynamic
    # scaling in the ODE solver.
    balance_scale = float(getattr(iegs_env, "linepack_balance_scale", 1.0))
    restore_flow = gain * lp_error / max(balance_scale, 1e-6)

    max_restore = torch.clamp(expected_D, min=1.0) * cap_fraction
    restore_flow = torch.maximum(torch.minimum(restore_flow, max_restore), -max_restore)

    desired_total_S = torch.clamp(expected_D + restore_flow, min=0.0)
    return source_share * desired_total_S.unsqueeze(-1)


def simulate_strategies_batched(
    iegs_env,
    initial_state: dict,
    strategies: list[dict],
    max_ode_steps: int = 3500,
    tolerance: float = 1e-4,
    return_trajectories: bool = True,
    solver: Optional[BatchedNeurodynamicODESolver] = None,
) -> SimulationResult:
    if len(strategies) == 0:
        raise ValueError("strategies must not be empty")

    batch_size = len(strategies)
    device = iegs_env.power.B_matrix.device

    if solver is None or solver.batch_size != batch_size:
        solver = BatchedNeurodynamicODESolver(
            iegs_env,
            batch_size=batch_size,
            max_ode_steps=max_ode_steps,
            tolerance=tolerance,
        )

    normal_lp_ratio = float(getattr(iegs_env, "normal_lp_ratio", 0.90))
    lp_replenish_ratio = float(getattr(iegs_env, "lp_replenish_ratio", 0.60))
    lp_trip_ratio = float(getattr(iegs_env, "lp_trip_ratio", 0.30))

    solver.initial_lp_sum = torch.sum(initial_state["lp_cur"]).to(device) / normal_lp_ratio
    solver.max_lp_vector = initial_state["lp_cur"].to(device) / normal_lp_ratio

    batched_state = _as_batch_state(initial_state, batch_size)

    base_Pg = batched_state["P_g"].clone()
    base_S = batched_state["S"].clone()
    base_f = batched_state["f_in"].clone()
    base_delta = batched_state["delta"].clone()
    base_comp = batched_state.get("comp_ratio", None)

    current_state = {k: v.clone() for k, v in batched_state.items()}
    prev_state = {k: v.clone() for k, v in batched_state.items()}

    total_damage = torch.zeros(batch_size, device=device)
    min_lp_pct = torch.ones(batch_size, device=device) * 100.0
    dos_lp_at_trigger = torch.ones(batch_size, device=device) * -1.0

    min_lp_after_dos_pct = torch.ones(batch_size, device=device) * 1.0e9
    dos_has_triggered = torch.zeros(batch_size, device=device, dtype=torch.bool)

    min_pressure_ratio = torch.ones(batch_size, device=device) * 100.0
    min_pressure_trip_ratio = torch.ones(batch_size, device=device) * 100.0
    dos_pressure_at_trigger = torch.ones(batch_size, device=device) * 100.0
    dos_pressure_trip_at_trigger = torch.ones(batch_size, device=device) * 100.0

    trajectories = [[] for _ in range(batch_size)]

    gfpp_offline_latch = torch.zeros(batch_size, dtype=torch.bool, device=device)
    dos_already_triggered = torch.zeros(batch_size, dtype=torch.bool, device=device)

    dos_vulnerable_latch = torch.zeros(batch_size, dtype=torch.bool, device=device)
    pre_dos_warning_latch = torch.zeros(batch_size, dtype=torch.bool, device=device)

    locked_S_memory = torch.zeros_like(base_S)
    locked_comp_memory = torch.zeros_like(base_comp) if base_comp is not None else None
    fdia_exposure = torch.zeros(batch_size, device=device)

    orig_power_load = iegs_env.power.base_load.clone()
    orig_gas_load = iegs_env.gas.base_gas_load.clone()
    ref_lp = solver.initial_lp_sum.to(device)

    if len(iegs_env.gfpp_indices) > 0:
        gfpp_idx_ts = iegs_env.gfpp_indices_tensor()
        gfpp_nodes = iegs_env.gfpp_gas_nodes_tensor()
    else:
        gfpp_idx_ts = torch.tensor([], dtype=torch.long, device=device)
        gfpp_nodes = torch.arange(iegs_env.gas.num_nodes, device=device)

    try:
        for t in range(1, 25):
            mult = load_multiplier(t)

            iegs_env.power.base_load = orig_power_load * mult
            iegs_env.gas.base_gas_load = orig_gas_load * mult

            current_state["P_g"] = base_Pg * mult
            current_state["delta"] = base_delta * mult
            current_state["delta_cyber"] = base_delta * mult
            current_state["f_in"] = base_f * mult
            current_state["f_out"] = base_f * mult

            z_a = _strategy_fdia_tensor(strategies, t, iegs_env.power.num_buses, device)

            fdia_intensity_now = torch.clamp(
                torch.sum(torch.abs(z_a), dim=1) / (torch.sum(iegs_env.power.base_load) + 1e-6),
                min=0.0,
                max=1.0,
            )
            fdia_exposure = torch.clamp(0.86 * fdia_exposure + fdia_intensity_now, min=0.0, max=1.0)

            s_t = torch.ones(batch_size, device=device)
            for i, strat in enumerate(strategies):
                if t >= float(strat.get("T_dos", 99.0)):
                    s_t[i] = 0.0

            lp_pct_prev = torch.sum(prev_state["lp_cur"], dim=1) / (ref_lp + 1e-6)
            lp_warning = lp_pct_prev <= lp_replenish_ratio
            lp_trip = lp_pct_prev <= lp_trip_ratio

            fdia_preconditioned = fdia_exposure >= float(getattr(iegs_env, "fdia_trip_exposure", 0.06))

            just_triggered = (s_t == 0) & (~dos_already_triggered)
            if torch.any(just_triggered):
                locked_S_memory[just_triggered] = prev_state["S"][just_triggered].clone()

                if locked_comp_memory is not None and "comp_ratio" in prev_state:
                    locked_comp_memory[just_triggered] = prev_state["comp_ratio"][just_triggered].clone()

                dos_lp_at_trigger[just_triggered] = lp_pct_prev[just_triggered] * 100.0
                dos_pressure_at_trigger[just_triggered] = 100.0
                dos_pressure_trip_at_trigger[just_triggered] = 100.0
                dos_already_triggered |= just_triggered

            pre_dos_warning_latch |= (s_t == 1) & fdia_preconditioned & lp_warning
            dos_vulnerable_latch |= (s_t == 0) & fdia_preconditioned & (lp_warning | pre_dos_warning_latch)

            # Protection decision: linepack-only trip latch.
            gfpp_offline_latch |= lp_trip

            current_base_gas = torch.sum(orig_gas_load) * mult
            if gfpp_idx_ts.numel() > 0:
                # Use previous physical generation.  FDIA-induced over-dispatch is
                # not perfectly anticipated by SCADA in the same hour.
                current_gfpp_gas = torch.sum(
                    prev_state["P_g"][:, gfpp_idx_ts] * iegs_env.psi_g[gfpp_idx_ts],
                    dim=1,
                )
            else:
                current_gfpp_gas = torch.zeros(batch_size, device=device)

            expected_D = current_base_gas + current_gfpp_gas

            desired_S = _source_target_with_inventory_control(
                iegs_env=iegs_env,
                base_S=base_S,
                expected_D=expected_D,
                lp_pct_prev=lp_pct_prev,
                lp_warning=lp_warning,
                normal_lp_ratio=normal_lp_ratio,
                lp_replenish_ratio=lp_replenish_ratio,
                lp_trip_ratio=lp_trip_ratio,
                ref_lp=ref_lp,
                prev_lp_sum=torch.sum(prev_state["lp_cur"], dim=1),
            )

            current_state["S"] = _ramp_limited_source_update(
                iegs_env=iegs_env,
                prev_S=prev_state["S"],
                desired_S=desired_S,
                lp_warning=lp_warning,
                s_t=s_t,
            )

            if "comp_ratio" in current_state:
                current_state["comp_ratio"] = prev_state["comp_ratio"].clone()

            current_state["P_d_sh"].zero_()
            current_state["P_q_sh"].zero_()
            current_state["lp_prev"] = prev_state["lp_cur"].clone()
            if "S_prev" in current_state:
                current_state["S_prev"] = prev_state["S"].clone()

            attack_params = {
                "FDIA": z_a,
                "s_t": s_t,
                "locked_S": locked_S_memory,
                "fdia_exposure": fdia_exposure,
                "dos_vulnerable": dos_vulnerable_latch.float(),
                # Used only for optional lower-level convergence tracing.
                # It does not change the physical simulation or optimization result.
                "physical_hour": int(t),
            }

            if locked_comp_memory is not None:
                attack_params["locked_comp"] = locked_comp_memory

            next_state, step_shed = solver.evolve_to_equilibrium_batched(
                current_state,
                prev_state,
                attack_params,
                gfpp_offline_latch,
            )

            noise_th = 2.0 if iegs_env.power.num_buses < 50 else 40.0
            step_shed = torch.where(step_shed < noise_th, torch.zeros_like(step_shed), step_shed)
            total_damage += step_shed

            lp_pct = torch.sum(next_state["lp_cur"], dim=1) / (ref_lp + 1e-6) * 100.0
            min_lp_pct = torch.minimum(min_lp_pct, lp_pct)

            # DoS trigger linepack must be the physical linepack at the trigger instant.
            # It has already been recorded above with lp_pct_prev when just_triggered is True.
            # Do not overwrite it here with the post-ODE linepack of the same hour; otherwise
            # the vulnerability-window check 30% < LP(T_DoS) <= 60% is shifted by one evolution step.
            dos_has_triggered = dos_has_triggered | (s_t < 0.5)

            min_lp_after_dos_pct = torch.where(
                dos_has_triggered,
                torch.minimum(min_lp_after_dos_pct, lp_pct),
                min_lp_after_dos_pct,
            )
            min_pressure = _min_gfpp_pressure(next_state, gfpp_nodes)

            # >>> [NEW-MECHANISM] Aggregate physical quantities for mechanism validation.
            if gfpp_idx_ts.numel() > 0:
                gfpp_output_total = torch.sum(next_state["P_g"][:, gfpp_idx_ts], dim=1)
                gfpp_gas_total = torch.sum(
                    next_state["P_g"][:, gfpp_idx_ts] * iegs_env.psi_g[gfpp_idx_ts],
                    dim=1,
                )
            else:
                gfpp_output_total = torch.zeros(batch_size, device=device)
                gfpp_gas_total = torch.zeros(batch_size, device=device)

            gas_source_total = torch.sum(next_state["S"], dim=1)
            total_linepack = torch.sum(next_state["lp_cur"], dim=1)
            # <<< [NEW-MECHANISM]

            # >>> [NEW-SPATIAL] Node/pipe states for spatial damage distribution.
            # Keep bus-level shedding consistent with the same noise threshold used
            # for the reported total damage: if this hour is treated as zero total
            # shedding, its bus-wise vector is also exported as zero.
            bus_shed_for_export = next_state["P_d_sh"].clone()
            zero_shed_mask = step_shed <= 0.0
            if torch.any(zero_shed_mask):
                bus_shed_for_export[zero_shed_mask] = 0.0

            max_lp_vec = torch.clamp(solver.max_lp_vector.to(device), min=1e-9)
            pipeline_lp_pct = next_state["lp_cur"] / max_lp_vec.unsqueeze(0) * 100.0

            pi_span = torch.clamp(iegs_env.gas.pi_max - iegs_env.gas.pi_min, min=1e-9)
            gas_pressure_margin_pct = (
                (next_state["pi"] - iegs_env.gas.pi_min.unsqueeze(0))
                / pi_span.unsqueeze(0)
                * 100.0
            )
            # <<< [NEW-SPATIAL]

            if return_trajectories:
                for i in range(batch_size):
                    trajectories[i].append(
                        {
                            "time": t,
                            # The DoS event is applied at the beginning of period t.
                            # Keep both time references explicitly:
                            #   lp_pct_pre  = physical linepack at the trigger instant / period start;
                            #   lp_pct_post = physical linepack after the period-t equilibrium update.
                            # lp_pct is retained as the post-update value for backward compatibility.
                            "lp_pct_pre": float((lp_pct_prev[i] * 100.0).detach().cpu().item()),
                            "lp_pct_post": float(lp_pct[i].detach().cpu().item()),
                            "lp_pct": float(lp_pct[i].detach().cpu().item()),
                            "step_shed": float(step_shed[i].detach().cpu().item()),

                            # >>> [NEW-MECHANISM] Aggregate mechanism-validation outputs.
                            "gfpp_output_mw": float(gfpp_output_total[i].detach().cpu().item()),
                            "gfpp_gas_consumption": float(gfpp_gas_total[i].detach().cpu().item()),
                            "gas_source_total": float(gas_source_total[i].detach().cpu().item()),
                            "total_linepack": float(total_linepack[i].detach().cpu().item()),
                            # <<< [NEW-MECHANISM]

                            # >>> [NEW-SPATIAL] Compact vector columns.
                            # They are strings such as "0;0;12.3;..." and are parsed
                            # by plot_four_method_bar.py.  No extra result file is needed.
                            "bus_shedding_vector_mw": _tensor_row_to_text(bus_shed_for_export[i]),
                            "gas_node_pressure_vector": _tensor_row_to_text(next_state["pi"][i]),
                            "gas_node_pressure_margin_pct_vector": _tensor_row_to_text(gas_pressure_margin_pct[i]),
                            "pipeline_linepack_vector": _tensor_row_to_text(next_state["lp_cur"][i]),
                            "pipeline_linepack_pct_vector": _tensor_row_to_text(pipeline_lp_pct[i]),
                            # <<< [NEW-SPATIAL]

                            "min_gfpp_pressure": float(min_pressure[i].detach().cpu().item()),
                            "pressure_ratio": 100.0,
                            "pressure_replenish_ratio": 100.0,
                            "pressure_trip_ratio": 100.0,
                            # Store the realized activation state used by this exact
                            # simulation step.  Plotting should read these flags instead
                            # of reconstructing activation times from rounded strategy data.
                            "fdia_active": int(torch.any(torch.abs(z_a[i]) > 1e-12).detach().cpu().item()),
                            "s_t": int(s_t[i].detach().cpu().item()),
                            "gfpp_offline": int(gfpp_offline_latch[i].detach().cpu().item()),
                            "dos_vulnerable": int(dos_vulnerable_latch[i].detach().cpu().item()),
                        }
                    )

            prev_state = {k: v.clone() for k, v in next_state.items()}
            current_state = {k: v.clone() for k, v in next_state.items()}
            current_state["lp_prev"] = current_state["lp_cur"].clone()
        min_lp_after_dos_pct_out = torch.where(
            dos_has_triggered,
            min_lp_after_dos_pct,
            torch.ones_like(min_lp_after_dos_pct) * -1.0,
        )

        post_dos_drop_pct = torch.where(
            dos_has_triggered,
            torch.clamp(dos_lp_at_trigger - min_lp_after_dos_pct, min=0.0),
            torch.ones_like(dos_lp_at_trigger) * -1.0,
        )
        metrics = {
            "min_lp_pct": min_lp_pct.detach().cpu().numpy(),
            "min_pressure_ratio": min_pressure_ratio.detach().cpu().numpy(),
            "min_pressure_trip_ratio": min_pressure_trip_ratio.detach().cpu().numpy(),
            "dos_lp_at_trigger": dos_lp_at_trigger.detach().cpu().numpy(),
            "dos_pressure_at_trigger": dos_pressure_at_trigger.detach().cpu().numpy(),
            "dos_pressure_trip_at_trigger": dos_pressure_trip_at_trigger.detach().cpu().numpy(),
            "min_lp_after_dos_pct": min_lp_after_dos_pct_out.detach().cpu().numpy(),
            "post_dos_drop_pct": post_dos_drop_pct.detach().cpu().numpy(),
        }

        return SimulationResult(total_damage.detach().cpu().numpy(), trajectories, metrics)

    finally:
        iegs_env.power.base_load = orig_power_load
        iegs_env.gas.base_gas_load = orig_gas_load

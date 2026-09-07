"""Initial operating-state construction for the IEEE 118-bus/GasLib-135 system."""
from __future__ import annotations

import torch

from env_physics.iegs_coupled import IEGSSystem


def create_initial_physical_state(env: IEGSSystem, target_system: str = "118-135") -> dict:
    device = env.power.B_matrix.device

    if target_system != "118-135":
        raise ValueError("target_system must be '118-135'")

    power_total = 4242.0
    gas_total = 3500.0
    target_total_pmax = power_total * 1.35
    gfpp_ratio = 5.0
    non_gfpp_ratio = 0.40
    kp_divisor = 85000.0
    psi_multiplier = 35.0


    if torch.sum(env.power.base_load) <= 1e-9:
        env.power.base_load[:] = power_total / env.power.num_buses
    else:
        env.power.base_load = env.power.base_load / torch.sum(env.power.base_load) * power_total

    if torch.sum(env.gas.base_gas_load) <= 1e-9:
        env.gas.base_gas_load[:] = gas_total / env.gas.num_nodes
    else:
        env.gas.base_gas_load = env.gas.base_gas_load / torch.sum(env.gas.base_gas_load) * gas_total

    B_row_sum = torch.sum(torch.abs(env.power.B_matrix), dim=1)
    isolated = B_row_sum < 1e-5
    if torch.any(isolated):
        dead_load = torch.sum(env.power.base_load[isolated])
        env.power.base_load[isolated] = 0.0
        alive = ~isolated
        env.power.base_load[alive] += dead_load / torch.sum(alive)

    total_pmax = torch.sum(env.power.gen_pmax)
    if total_pmax <= 1e-9:
        env.power.gen_pmax[:] = target_total_pmax / env.power.num_gens
    else:
        env.power.gen_pmax *= target_total_pmax / total_pmax

    env.power.gen_pmin = torch.zeros_like(env.power.gen_pmin, device=device)

    if len(env.gfpp_indices) > 0:
        non_gfpp = [i for i in range(env.power.num_gens) if i not in env.gfpp_indices]
        if non_gfpp:
            env.power.gen_pmax[non_gfpp] *= non_gfpp_ratio
        env.power.gen_pmax[env.gfpp_indices] *= gfpp_ratio
        env.power.gen_pmax *= target_total_pmax / torch.sum(env.power.gen_pmax)
        env.psi_g[env.gfpp_indices] *= psi_multiplier

    env.psi_g = env.psi_g.to(device) * 100.0
    env.power.gen_ramp_max = torch.clamp(env.power.gen_pmax * 0.25, min=10.0)
    env.power.line_capacity_max = env.power.line_capacity_max * 1000.0

    env.gas.linepack_K_p = env.gas.linepack_K_p.to(device) / kp_divisor
    env.gas.weymouth_C_p = torch.ones_like(env.gas.weymouth_C_p, device=device) * 1e-6
    env.gas.pi_min = torch.clamp(env.gas.pi_min.to(device), min=10.0)
    env.gas.pi_max = torch.maximum(env.gas.pi_max.to(device), env.gas.pi_min + 80.0)
    env.gas.pi_warn = env.gas.pi_min + 0.25 * (env.gas.pi_max - env.gas.pi_min)
    env.gas.pi_trip = env.gas.pi_min + 0.05 * (env.gas.pi_max - env.gas.pi_min)
    env.gas.S_max = torch.ones(env.gas.num_sources, device=device) * 999999.0
    env.gas.S_min = torch.zeros(env.gas.num_sources, device=device)

    total_load = torch.sum(env.power.base_load)
    pg = env.power.gen_pmax / torch.sum(env.power.gen_pmax) * total_load
    pg = torch.maximum(torch.minimum(pg, env.power.gen_pmax), env.power.gen_pmin)
    mismatch = total_load - torch.sum(pg)
    if torch.abs(mismatch) > 1e-6:
        headroom = torch.clamp(env.power.gen_pmax - pg, min=0.0)
        if torch.sum(headroom) > 1e-9:
            pg += headroom / torch.sum(headroom) * mismatch

    gen_inj = env.power.gen_to_bus_matrix @ pg
    net_inj = gen_inj - env.power.base_load
    delta = torch.linalg.pinv(env.power.B_matrix) @ net_inj
    delta = delta - delta[0]

    nodal_D = env.gas.base_gas_load.clone()
    for gen_idx in env.gfpp_indices:
        nodal_D[env.gfpp_to_gas_node[gen_idx]] += pg[gen_idx] * env.psi_g[gen_idx]

    source_reserve = float(getattr(env, "initial_source_reserve", 1.00))
    S_init = torch.ones(env.gas.num_sources, device=device) * (torch.sum(nodal_D) * source_reserve / max(1, env.gas.num_sources))
    nodal_S = env.gas.A_nw @ S_init
    gas_mismatch = nodal_D - nodal_S
    A_pipe = env.gas.A_np_minus - env.gas.A_np_plus
    f_init = torch.linalg.pinv(A_pipe) @ gas_mismatch
    f_init = torch.clamp(f_init, min=0.0)

    pressure_init_ratio = float(getattr(env, "initial_pressure_ratio", 0.70))
    pi_mid = env.gas.pi_min + pressure_init_ratio * (env.gas.pi_max - env.gas.pi_min)
    pi_mid = torch.maximum(torch.minimum(pi_mid, env.gas.pi_max), env.gas.pi_min)

    pipe_from_idx = env.gas.A_np_plus.argmax(dim=0)
    pipe_to_idx = env.gas.A_np_minus.argmax(dim=0)
    lp_init = env.gas.linepack_K_p * (pi_mid[pipe_from_idx] + pi_mid[pipe_to_idx]) / 2.0

    state = {
        "P_g": pg.clone(),
        "delta": delta.clone(),
        "delta_cyber": delta.clone(),
        "P_d_sh": torch.zeros(env.power.num_buses, device=device),
        "S": S_init.clone(),
        "pi": pi_mid.clone(),
        "f_in": f_init.clone(),
        "f_out": f_init.clone(),
        "P_q_sh": torch.zeros(env.gas.num_nodes, device=device),
        "lp_cur": lp_init.clone(),
    }
    if env.gas.num_comps > 0:
        state["comp_ratio"] = torch.ones(env.gas.num_comps, device=device)
    else:
        state["comp_ratio"] = torch.ones(1, device=device)
    state["lp_prev"] = state["lp_cur"].clone()
    state["S_prev"] = state["S"].clone()
    return state

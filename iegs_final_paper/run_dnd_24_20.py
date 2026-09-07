"""24-20 MIP-DND experiment with explicit GFPP trip / gas-linepack recovery.

This file is a drop-in replacement for the previous ``run_distributed_24_20.py``.
It is intentionally self-contained: no existing 118-135 source file is edited.

Required data file
------------------
    data/IEEE24_Belgian20.xlsx

Main corrections in this version
--------------------------------
1. The 24 h electric and gas demand profiles in the workbook are used separately.
2. The no-attack baseline uses demand-tracking gas-source dispatch and therefore
   stays close to the initial normal linepack instead of drifting strongly.
3. When aggregate linepack at the end of a scheduling period is <= 30%, the four
   GFPPs are hard-tripped from the next scheduling period onward.  Their electric
   Pmin/Pmax are forced to zero by the existing DND solver's offline mask, and
   their gas demand is explicitly removed before source control is calculated.
4. After a GFPP trip, a local gas-side emergency recovery controller is enabled.
   It is treated as a protection action and therefore is not blocked by the DoS
   lock.  Source dispatch is ramped toward native gas demand plus the inventory
   surplus required to recover linepack toward the 90% normal level.
5. The upper-level coordinated score gets no extra reward for driving linepack
   farther below the 30% trip threshold.  Once the trip threshold is crossed,
   electric damage remains the primary consequence.

Protection reset policy
-----------------------
GFPP trip is latched for the rest of the 24 h horizon (manual-reset protection).
This avoids artificial repeated trip/restart oscillations at 30%/60%.

Run
---
    python run_distributed_24_20.py

Background run
--------------
    nohup python run_distributed_24_20.py > run_24_20.log 2>&1 &
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import main as project_main
import simulation as shared_sim
import model_distributed_nd_simulation
import algorithm.mip_nd_master as mip_nd_master
from algorithm.model_distributed_nd_solver import ModelDistributedNeurodynamicODESolver
from env_physics.iegs_coupled import IEGSSystem


SYSTEM_NAME = "24-20"
DATA_FILENAME = "IEEE24_Belgian20.xlsx"
VERSION = "2026-08-31-trip-recovery-v1"

# Loader -> configure_env handoff.  Keeping the profiles here avoids depending
# on whether pandas DataFrame.attrs are preserved by the environment classes.
_POWER_HOURLY_PROFILE: Optional[np.ndarray] = None
_GAS_HOURLY_PROFILE: Optional[np.ndarray] = None

# Original project hooks are kept so importing this file does not alter the
# behavior of any non-24-20 process.
_ORIGINAL_LOAD_SYSTEM_DATA = project_main.load_system_data
_ORIGINAL_CONFIGURE_ENV = project_main.configure_env
_ORIGINAL_CREATE_INITIAL_STATE = project_main.create_initial_physical_state
_ORIGINAL_DETECT_GFPP = IEGSSystem._detect_gfpp_indices
_ORIGINAL_MIP_SCORE_BATCH = mip_nd_master.MIPNDOptimizer._score_batch


# =============================================================================
# 1. Workbook loader
# =============================================================================

def _row_text(row: pd.Series) -> list[str]:
    return [
        str(v).strip().lower()
        for v in row.tolist()
        if pd.notna(v) and str(v).strip()
    ]


def _find_header_row(raw: pd.DataFrame, required_terms: Iterable[str]) -> int:
    terms = [str(x).lower() for x in required_terms]
    for idx, row in raw.iterrows():
        vals = _row_text(row)
        if all(any(term in cell for cell in vals) for term in terms):
            return int(idx)
    raise ValueError(f"Cannot find table header containing: {terms}")


def _read_block(raw: pd.DataFrame, header_row: int, ncols: int) -> pd.DataFrame:
    """Read a stacked Excel table until the first fully blank row."""
    headers = [
        str(x).strip() if pd.notna(x) else f"col_{i}"
        for i, x in enumerate(raw.iloc[header_row, :ncols])
    ]
    rows: list[list[object]] = []
    for ridx in range(header_row + 1, len(raw)):
        vals = raw.iloc[ridx, :ncols].tolist()
        if all(pd.isna(v) or str(v).strip() == "" for v in vals):
            break
        rows.append(vals)
    return pd.DataFrame(rows, columns=headers).dropna(how="all").reset_index(drop=True)


def _num(series: pd.Series, default: float | None = None) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce")
    if default is not None:
        out = out.fillna(default)
    return out


def _require_24(arr: np.ndarray, label: str) -> np.ndarray:
    arr = np.asarray(arr, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size != 24:
        raise ValueError(f"{label} must contain exactly 24 hourly values; got {arr.size}.")
    if np.any(arr <= 0.0):
        raise ValueError(f"{label} contains non-positive hourly values.")
    return arr


def load_system_data_24_20(
    target_system: str = SYSTEM_NAME,
    data_dir: str = "data/",
) -> dict:
    """Load IEEE RTS-24 / Belgian-20 data into the existing project schema."""
    global _POWER_HOURLY_PROFILE, _GAS_HOURLY_PROFILE

    if target_system != SYSTEM_NAME:
        return _ORIGINAL_LOAD_SYSTEM_DATA(target_system=target_system, data_dir=data_dir)

    path = Path(data_dir) / DATA_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"Missing 24-20 data file: {path}\n"
            f"Put {DATA_FILENAME} in the project's data/ directory."
        )

    print(f"\n[24-20 Data] {path}")
    p_raw = pd.read_excel(path, sheet_name="IEEE RTS 24-Bus System", header=None)
    g_raw = pd.read_excel(path, sheet_name="Belgian 20-node gas system", header=None)
    coupling_raw = pd.read_excel(path, sheet_name="Coupling")

    # ---------------- Power ----------------
    branch_h = _find_header_row(p_raw, ["from", "to", "reactance"])
    gen_h = _find_header_row(p_raw, ["unit", "node", "p min", "p max"])
    load_h = _find_header_row(p_raw, ["hour", "system demand", "percentage"])

    branch_src = _read_block(p_raw, branch_h, 5)
    gen_src = _read_block(p_raw, gen_h, 9)
    load_src = _read_block(p_raw, load_h, 5)

    branch = pd.DataFrame(
        {
            "From": _num(branch_src.iloc[:, 1]),
            "To": _num(branch_src.iloc[:, 2]),
            "Reactance(p.u.)": _num(branch_src.iloc[:, 3]),
            "Capacity (MVA)": _num(branch_src.iloc[:, 4], 9999.0),
        }
    ).dropna(subset=["From", "To", "Reactance(p.u.)"])
    branch = branch.astype({"From": int, "To": int}).reset_index(drop=True)
    branch["Reactance(p.u.)"] = branch["Reactance(p.u.)"].clip(lower=1.0e-4)

    gen = pd.DataFrame(
        {
            "Node": _num(gen_src.iloc[:, 1]),
            "P min(MW)": _num(gen_src.iloc[:, 2], 0.0),
            "P max(MW)": _num(gen_src.iloc[:, 3]),
        }
    ).dropna(subset=["Node", "P max(MW)"])
    gen = gen.astype({"Node": int}).reset_index(drop=True)

    load_nodes = _num(load_src.iloc[:, 2])
    load_fraction = _num(load_src.iloc[:, 3], 0.0)
    hourly_power = _require_24(
        _num(load_src.iloc[:, 1]).dropna().to_numpy(dtype=float),
        "RTS-24 electric demand profile",
    )
    reference_power = float(np.mean(hourly_power))

    buses = sorted(
        set(branch["From"].astype(int).tolist())
        | set(branch["To"].astype(int).tolist())
        | set(gen["Node"].astype(int).tolist())
        | set(load_nodes.dropna().astype(int).tolist())
    )
    bus = pd.DataFrame({"Node": buses, "load": np.zeros(len(buses), dtype=float)})
    for node, frac in zip(load_nodes, load_fraction):
        if pd.isna(node):
            continue
        bus.loc[bus["Node"] == int(node), "load"] += reference_power * float(frac)

    # ---------------- Gas ----------------
    node_h = _find_header_row(
        g_raw, ["gas nodes", "min pressure", "max pressure", "load distribution"]
    )
    pipe_h = _find_header_row(g_raw, ["pipeline", "gnode m", "gnode n", "wmn"])
    source_h = _find_header_row(
        g_raw, ["suppliers", "source node", "minimum output", "maximum output"]
    )
    gas_load_h = _find_header_row(g_raw, ["time period", "total", "load"])

    node_src = _read_block(g_raw, node_h, 5)
    pipe_src = _read_block(g_raw, pipe_h, 5)
    source_src = _read_block(g_raw, source_h, 5)
    gas_load_src = _read_block(g_raw, gas_load_h, 2)

    hourly_gas = _require_24(
        _num(gas_load_src.iloc[:, 1]).dropna().to_numpy(dtype=float),
        "Belgian-20 gas demand profile",
    )
    reference_gas = float(np.mean(hourly_gas))

    gas_node_ids = _num(node_src.iloc[:, 0]).dropna().astype(int)
    n_gas = len(gas_node_ids)
    gas_load_fraction = _num(node_src.iloc[:, 3], 0.0).iloc[:n_gas]
    gas_node = pd.DataFrame(
        {
            "node_id": gas_node_ids.to_numpy(dtype=int),
            "load": reference_gas * gas_load_fraction.to_numpy(dtype=float),
            "min pressure": _num(node_src.iloc[:, 1], 0.0)
            .iloc[:n_gas]
            .to_numpy(dtype=float),
            "max pressure": _num(node_src.iloc[:, 2], 80.0)
            .iloc[:n_gas]
            .to_numpy(dtype=float),
        }
    )

    gas_pipe = pd.DataFrame(
        {
            "pipe_id": np.arange(len(pipe_src), dtype=int),
            "m": _num(pipe_src.iloc[:, 1]),
            "n": _num(pipe_src.iloc[:, 2]),
            "wmn": _num(pipe_src.iloc[:, 3], 1.0),
            # The workbook has no explicit pipe length / diameter.  These two
            # neutral placeholders are used only to initialize K_p and are then
            # calibrated locally in create_initial_physical_state_24_20().
            "length": np.full(len(pipe_src), 10.0),
            "diameter": np.full(len(pipe_src), 500.0),
        }
    ).dropna(subset=["m", "n"])
    gas_pipe = gas_pipe.astype({"pipe_id": int, "m": int, "n": int}).reset_index(drop=True)

    gas_source = pd.DataFrame(
        {
            "node": _num(source_src.iloc[:, 1]),
            "S_min": _num(source_src.iloc[:, 2], 0.0),
            "S_max": _num(source_src.iloc[:, 3], 1.0e6),
        }
    ).dropna(subset=["node"])
    gas_source = gas_source.astype({"node": int}).reset_index(drop=True)

    compressor = pd.DataFrame(columns=["m", "n", "c_min", "c_max"])

    # ---------------- Explicit GFPP coupling ----------------
    required = {"Generator_Unit", "Power_Bus", "Gas_Node"}
    if not required.issubset(set(coupling_raw.columns)):
        raise ValueError(f"Coupling sheet must contain columns {sorted(required)}")

    coupling = pd.DataFrame(
        {
            "GFPP_ID": _num(coupling_raw["Generator_Unit"]),
            "Power_Bus": _num(coupling_raw["Power_Bus"]),
            "Gas_Node": _num(coupling_raw["Gas_Node"]),
        }
    ).dropna()
    coupling = coupling.astype({"GFPP_ID": int, "Power_Bus": int, "Gas_Node": int})

    pmax_by_bus = gen.groupby("Node")["P max(MW)"].first().to_dict()
    coupling["P_Max"] = [float(pmax_by_bus[int(b)]) for b in coupling["Power_Bus"]]

    gas_id_to_index = {
        int(node_id): idx for idx, node_id in enumerate(gas_node["node_id"].tolist())
    }
    missing_gas = [int(n) for n in coupling["Gas_Node"] if int(n) not in gas_id_to_index]
    if missing_gas:
        raise ValueError(f"Coupling sheet references unknown gas nodes: {missing_gas}")
    coupling["Gas_Node_Index"] = [gas_id_to_index[int(n)] for n in coupling["Gas_Node"]]
    coupling["Gas_Node_Name"] = coupling["Gas_Node"].astype(str)
    coupling = coupling.sort_values("GFPP_ID").reset_index(drop=True)

    _POWER_HOURLY_PROFILE = hourly_power.copy()
    _GAS_HOURLY_PROFILE = hourly_gas.copy()

    print(
        f"    power buses={len(bus)}, generators={len(gen)}, "
        f"gas nodes={len(gas_node)}, pipes={len(gas_pipe)}, sources={len(gas_source)}"
    )
    print(
        f"    GFPP={len(coupling)} | power buses={coupling['Power_Bus'].tolist()} | "
        f"gas nodes={coupling['Gas_Node'].tolist()}"
    )
    print(
        f"    24-h mean load: power={reference_power:.3f} MW | "
        f"gas={reference_gas:.3f} Mm3/h"
    )

    return {
        "power": {"bus": bus, "branch": branch, "gen": gen},
        "gas": {
            "node": gas_node,
            "pipeline": gas_pipe,
            "source": gas_source,
            "compressor": compressor,
        },
        "wind": None,
        "coupling": {"gfpp": coupling},
    }


# =============================================================================
# 2. 4-GFPP detection and 24-20 configuration
# =============================================================================

def _detect_gfpp_indices_24_20(self: IEGSSystem) -> list[int]:
    if self.gfpp_coupling_df.empty:
        return _ORIGINAL_DETECT_GFPP(self)
    if self.power.num_buses != 24 or self.gas.num_nodes != 20:
        return _ORIGINAL_DETECT_GFPP(self)

    coupling = self.gfpp_coupling_df.sort_values("GFPP_ID").reset_index(drop=True)
    selected: list[int] = []
    for _, row in coupling.iterrows():
        bus = int(row["Power_Bus"])
        matches = self.power.gen_df.index[
            self.power.gen_df["Node"].astype(int) == bus
        ].tolist()
        if len(matches) != 1:
            raise ValueError(
                f"24-20 coupling bus {bus} must uniquely match one generator; got {matches}"
            )
        selected.append(int(matches[0]))

    if len(selected) != 4:
        raise ValueError(f"24-20 case must contain 4 GFPPs; got {len(selected)}")
    return selected


def configure_env_24_20(env: IEGSSystem, target_system: str) -> None:
    """Reuse common paper settings and add only 24-20 local calibration."""
    if target_system != SYSTEM_NAME:
        return _ORIGINAL_CONFIGURE_ENV(env, target_system)

    _ORIGINAL_CONFIGURE_ENV(env, "118-135")

    if _POWER_HOURLY_PROFILE is None or _GAS_HOURLY_PROFILE is None:
        raise RuntimeError("24-20 hourly profiles were not loaded before configure_env().")
    env.power_hourly_total_mw = np.asarray(_POWER_HOURLY_PROFILE, dtype=float).copy()
    env.gas_hourly_total_mm3h = np.asarray(_GAS_HOURLY_PROFILE, dtype=float).copy()

    # Attack / coupling strength kept close to the previous 24-20 runner.
    env.dispatch_attack_gain = 4.0
    env.fdia_max_ratio = 0.20
    env.fdia_trip_exposure = 0.05
    env.fdia_full_exposure = 0.15

    # LP thresholds used in the paper experiment.
    project_main._apply_linepack_thresholds(
        env,
        normal_ratio=0.90,
        replenish_ratio=0.60,
        trip_ratio=0.30,
    )

    # Baseline calibration: initial source exactly balances the mean operating
    # point; hourly source control below then follows actual demand instead of
    # leaving a large inventory bias.
    env.initial_pressure_ratio = 0.72
    env.initial_source_reserve = 1.00
    env.linepack_balance_scale = 0.18

    # Local source-controller parameters used by this runner.
    env.runner_normal_inventory_gain = 0.30
    env.runner_warning_inventory_gain = 0.75
    env.runner_normal_inventory_cap = 0.08
    env.runner_warning_inventory_cap = 0.25
    env.runner_normal_source_ramp_fraction = 0.50
    env.runner_warning_source_ramp_fraction = 0.65

    # After a hard trip, recover to the normal 90% inventory over roughly four
    # periods.  This controller is local protection and bypasses the remote DoS
    # lock only after the trip has occurred.
    env.runner_trip_recovery_horizon_hours = 4.0
    env.runner_trip_source_ramp_fraction = 1.00
    env.runner_trip_recovery_margin = 1.00

    # Coordinated DoS must still start inside the 30%-60% vulnerability window.
    env.mip_nd_hard_window = True

    # Do not reward numerically driving LP far below the physical trip point.
    env.runner_damage_primary_weight = 50.0
    env.runner_trip_cross_bonus = 1000.0
    env.runner_useful_lp_drop_weight = 25.0
    env.runner_timing_weight = 2.0

    # DND stopping criteria remain the same as the main experiment.
    env.mdnd_consensus_rel_tolerance = 1.0e-3
    env.mdnd_state_rel_tolerance = 1.0e-3
    env.mdnd_convergence_patience = 5
    env.nd_trace_effective_interval = 5.0

    print(f"[24-20 VERSION] {VERSION}")
    print("[24-20 FIX] baseline source-demand tracking        = ON")
    print("[24-20 FIX] hard GFPP trip at LP <= 30%            = ON")
    print("[24-20 FIX] GFPP gas demand removed after trip     = ON")
    print("[24-20 FIX] local post-trip gas recovery controller = ON")
    print("[24-20 FIX] post-trip source safety bypasses DoS    = ON")


# =============================================================================
# 3. Initial state
# =============================================================================

def create_initial_physical_state_24_20(
    env: IEGSSystem,
    target_system: str = SYSTEM_NAME,
) -> dict:
    if target_system != SYSTEM_NAME:
        return _ORIGINAL_CREATE_INITIAL_STATE(env, target_system)

    device = env.power.B_matrix.device
    power_total = float(torch.sum(env.power.base_load).detach().cpu().item())
    gas_total = float(torch.sum(env.gas.base_gas_load).detach().cpu().item())
    if power_total <= 1.0e-9 or gas_total <= 1.0e-9:
        raise ValueError("24-20 workbook produced a zero reference load.")

    # Remove any accidental load on an electrically isolated bus.
    b_row_sum = torch.sum(torch.abs(env.power.B_matrix), dim=1)
    isolated = b_row_sum < 1.0e-5
    if torch.any(isolated):
        dead_load = torch.sum(env.power.base_load[isolated])
        env.power.base_load[isolated] = 0.0
        alive = ~isolated
        if torch.any(alive):
            weights = env.power.base_load[alive].clone()
            if torch.sum(weights) <= 1.0e-9:
                weights = torch.ones_like(weights)
            env.power.base_load[alive] += dead_load * weights / torch.sum(weights)

    # Keep enough total generation and a meaningful GFPP share for the attack
    # mechanism, exactly as in the previous small-case runner.
    target_total_pmax = max(
        float(torch.sum(env.power.gen_pmax).detach().cpu().item()),
        power_total * 1.35,
    )
    env.power.gen_pmin = torch.zeros_like(env.power.gen_pmin, device=device)

    if len(env.gfpp_indices) > 0:
        non_gfpp = [i for i in range(env.power.num_gens) if i not in env.gfpp_indices]
        if non_gfpp:
            env.power.gen_pmax[non_gfpp] *= 0.60
        env.power.gen_pmax[env.gfpp_indices] *= 3.00
        env.power.gen_pmax *= target_total_pmax / torch.sum(env.power.gen_pmax)

        env.psi_g[:] = 0.0
        env.psi_g[env.gfpp_indices] = 8.0e-3

    env.power.gen_ramp_max = torch.clamp(env.power.gen_pmax * 0.25, min=10.0)
    # Retain the previous runner's feasibility scaling for the small RTS case.
    env.power.line_capacity_max = env.power.line_capacity_max * 10.0

    # Belgian-20 workbook lacks pipe length/diameter; calibrate only the linepack
    # coefficient generated from those neutral placeholders.
    env.gas.linepack_K_p = env.gas.linepack_K_p.to(device) / 25000.0
    env.gas.weymouth_C_p = torch.clamp(env.gas.weymouth_C_p.to(device), min=1.0e-3)
    env.gas.pi_min = torch.clamp(env.gas.pi_min.to(device), min=10.0)
    env.gas.pi_max = torch.maximum(env.gas.pi_max.to(device), env.gas.pi_min + 20.0)
    env.gas.pi_warn = env.gas.pi_min + 0.25 * (env.gas.pi_max - env.gas.pi_min)
    env.gas.pi_trip = env.gas.pi_min + 0.05 * (env.gas.pi_max - env.gas.pi_min)
    env.gas.S_min = torch.clamp(env.gas.S_min.to(device), min=0.0)
    env.gas.S_max = torch.maximum(env.gas.S_max.to(device), env.gas.S_min + 1.0)

    # Mean-load electric dispatch.
    total_load = torch.sum(env.power.base_load)
    pg = env.power.gen_pmax / torch.sum(env.power.gen_pmax) * total_load
    pg = torch.maximum(torch.minimum(pg, env.power.gen_pmax), env.power.gen_pmin)
    mismatch = total_load - torch.sum(pg)
    if torch.abs(mismatch) > 1.0e-6:
        if mismatch > 0:
            headroom = torch.clamp(env.power.gen_pmax - pg, min=0.0)
            if torch.sum(headroom) > 1.0e-9:
                pg += headroom / torch.sum(headroom) * mismatch
        else:
            reducible = torch.clamp(pg - env.power.gen_pmin, min=0.0)
            if torch.sum(reducible) > 1.0e-9:
                pg += reducible / torch.sum(reducible) * mismatch
    pg = torch.maximum(torch.minimum(pg, env.power.gen_pmax), env.power.gen_pmin)

    gen_inj = env.power.gen_to_bus_matrix @ pg
    net_inj = gen_inj - env.power.base_load
    delta = torch.linalg.pinv(env.power.B_matrix) @ net_inj
    delta = delta - delta[0]

    # Mean-load gas demand including GFPP fuel.
    nodal_d = env.gas.base_gas_load.clone()
    for gen_idx in env.gfpp_indices:
        nodal_d[env.gfpp_to_gas_node[gen_idx]] += pg[gen_idx] * env.psi_g[gen_idx]

    # IMPORTANT baseline fix: no artificial +4% initial gas reserve.  At the
    # mean operating point total source equals total demand; linepack therefore
    # starts at a genuine balanced normal state.
    total_required = torch.sum(nodal_d) * float(getattr(env, "initial_source_reserve", 1.0))
    source_cap = torch.clamp(env.gas.S_max - env.gas.S_min, min=1.0e-6)
    source_share = source_cap / torch.sum(source_cap)
    s_init = env.gas.S_min + source_share * torch.clamp(
        total_required - torch.sum(env.gas.S_min), min=0.0
    )
    s_init = torch.minimum(s_init, env.gas.S_max)

    src_gap = total_required - torch.sum(s_init)
    if src_gap > 1.0e-6:
        headroom = torch.clamp(env.gas.S_max - s_init, min=0.0)
        if torch.sum(headroom) > 1.0e-9:
            s_init += headroom / torch.sum(headroom) * src_gap
            s_init = torch.minimum(s_init, env.gas.S_max)

    nodal_s = env.gas.A_nw @ s_init
    gas_mismatch = nodal_d - nodal_s
    a_pipe = env.gas.A_np_minus - env.gas.A_np_plus
    f_init = torch.linalg.pinv(a_pipe) @ gas_mismatch
    f_init = torch.clamp(f_init, min=0.0)

    pressure_ratio = float(getattr(env, "initial_pressure_ratio", 0.72))
    pi_mid = env.gas.pi_min + pressure_ratio * (env.gas.pi_max - env.gas.pi_min)
    pi_mid = torch.maximum(torch.minimum(pi_mid, env.gas.pi_max), env.gas.pi_min)

    pipe_from_idx = env.gas.A_np_plus.argmax(dim=0)
    pipe_to_idx = env.gas.A_np_minus.argmax(dim=0)
    lp_init = env.gas.linepack_K_p * (
        pi_mid[pipe_from_idx] + pi_mid[pipe_to_idx]
    ) / 2.0

    state = {
        "P_g": pg.clone(),
        "delta": delta.clone(),
        "delta_cyber": delta.clone(),
        "P_d_sh": torch.zeros(env.power.num_buses, device=device),
        "S": s_init.clone(),
        "pi": pi_mid.clone(),
        "f_in": f_init.clone(),
        "f_out": f_init.clone(),
        "P_q_sh": torch.zeros(env.gas.num_nodes, device=device),
        "lp_cur": lp_init.clone(),
    }
    state["comp_ratio"] = (
        torch.ones(env.gas.num_comps, device=device)
        if env.gas.num_comps > 0
        else torch.ones(1, device=device)
    )
    state["lp_prev"] = state["lp_cur"].clone()
    state["S_prev"] = state["S"].clone()

    print(
        f"[24-20 init] power={power_total:.3f} MW | gas={gas_total:.3f} Mm3/h | "
        f"GFPP={len(env.gfpp_indices)} | initial LP={float(torch.sum(lp_init)):.6f}"
    )
    print(
        f"[24-20 init] source={float(torch.sum(s_init)):.6f} | "
        f"demand(including GFPP)={float(torch.sum(nodal_d)):.6f}"
    )
    return state


# =============================================================================
# 4. Source dispatch helpers used by the corrected 24-h simulation
# =============================================================================

def _dispatch_sources_for_total(env: IEGSSystem, total_target: torch.Tensor) -> torch.Tensor:
    """Allocate a batch of total gas-source targets over physical source bounds."""
    device = env.gas.S_min.device
    total_target = total_target.to(device=device, dtype=torch.float32).reshape(-1)
    batch = int(total_target.numel())

    s_min = env.gas.S_min.unsqueeze(0).expand(batch, -1)
    s_max = env.gas.S_max.unsqueeze(0).expand(batch, -1)
    span = torch.clamp(s_max - s_min, min=0.0)
    span_sum = torch.sum(span, dim=1, keepdim=True)

    base_sum = torch.sum(s_min, dim=1)
    remaining = torch.clamp(total_target - base_sum, min=0.0)
    share = torch.where(
        span_sum > 1.0e-9,
        span / (span_sum + 1.0e-12),
        torch.ones_like(span) / max(1, env.gas.num_sources),
    )
    out = s_min + share * remaining.unsqueeze(-1)
    out = torch.maximum(torch.minimum(out, s_max), s_min)

    # Fill any residual caused by upper-bound clipping, iteratively but cheaply.
    for _ in range(4):
        gap = total_target - torch.sum(out, dim=1)
        need = gap > 1.0e-7
        if not torch.any(need):
            break
        headroom = torch.clamp(s_max - out, min=0.0)
        head_sum = torch.sum(headroom, dim=1, keepdim=True)
        add_share = torch.where(
            head_sum > 1.0e-9,
            headroom / (head_sum + 1.0e-12),
            torch.zeros_like(headroom),
        )
        out = out + add_share * torch.clamp(gap, min=0.0).unsqueeze(-1)
        out = torch.maximum(torch.minimum(out, s_max), s_min)
    return out


def _ramp_sources(
    env: IEGSSystem,
    prev_s: torch.Tensor,
    desired_s: torch.Tensor,
    fraction: torch.Tensor,
) -> torch.Tensor:
    """Per-scenario source ramp projection."""
    span = torch.clamp(env.gas.S_max - env.gas.S_min, min=1.0)
    max_delta = fraction.reshape(-1, 1) * span.unsqueeze(0)
    delta = desired_s - prev_s
    out = prev_s + torch.maximum(torch.minimum(delta, max_delta), -max_delta)
    return torch.maximum(
        torch.minimum(out, env.gas.S_max.unsqueeze(0)),
        env.gas.S_min.unsqueeze(0),
    )


def _corrected_source_target(
    env: IEGSSystem,
    expected_demand: torch.Tensor,
    prev_lp_sum: torch.Tensor,
    ref_lp: torch.Tensor,
    lp_warning: torch.Tensor,
    gfpp_offline: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return total source target and ramp fraction for each scenario.

    Above 60%, source follows demand and applies only a weak inventory correction.
    Between 30% and 60%, the correction is stronger.  After hard trip, GFPP fuel
    demand has already been removed and the emergency controller creates the
    source surplus needed to restore LP toward 90% over a finite horizon.
    """
    dt = max(float(env.gas.dt), 1.0e-9)
    scale = max(float(getattr(env, "linepack_balance_scale", 1.0)), 1.0e-9)
    normal_ratio = float(getattr(env, "normal_lp_ratio", 0.90))
    normal_target = ref_lp * normal_ratio

    inventory_gap = normal_target - prev_lp_sum
    equivalent_flow_gap = inventory_gap / (dt * scale)

    normal_gain = float(getattr(env, "runner_normal_inventory_gain", 0.30))
    warning_gain = float(getattr(env, "runner_warning_inventory_gain", 0.75))
    gain = torch.where(
        lp_warning,
        torch.full_like(expected_demand, warning_gain),
        torch.full_like(expected_demand, normal_gain),
    )

    correction = gain * equivalent_flow_gap
    normal_cap = float(getattr(env, "runner_normal_inventory_cap", 0.08))
    warning_cap = float(getattr(env, "runner_warning_inventory_cap", 0.25))
    cap_ratio = torch.where(
        lp_warning,
        torch.full_like(expected_demand, warning_cap),
        torch.full_like(expected_demand, normal_cap),
    )
    correction_cap = torch.clamp(expected_demand.abs(), min=1.0e-3) * cap_ratio
    correction = torch.maximum(torch.minimum(correction, correction_cap), -correction_cap)
    total_target = expected_demand + correction

    # Hard-trip recovery: target the normal inventory over H periods.  This is
    # computed from the same inventory identity used by the gas projection,
    # rather than imposing a hand-drawn linepack trajectory.
    horizon = max(float(getattr(env, "runner_trip_recovery_horizon_hours", 4.0)), 1.0)
    recovery_margin = float(getattr(env, "runner_trip_recovery_margin", 1.0))
    recovery_flow = torch.clamp(inventory_gap, min=0.0) / (horizon * dt * scale)
    recovery_total = expected_demand + recovery_margin * recovery_flow
    total_target = torch.where(gfpp_offline, recovery_total, total_target)

    min_total = torch.sum(env.gas.S_min)
    max_total = torch.sum(env.gas.S_max)
    total_target = torch.clamp(total_target, min=min_total, max=max_total)

    normal_ramp = float(getattr(env, "runner_normal_source_ramp_fraction", 0.50))
    warning_ramp = float(getattr(env, "runner_warning_source_ramp_fraction", 0.65))
    trip_ramp = float(getattr(env, "runner_trip_source_ramp_fraction", 1.00))
    ramp_fraction = torch.where(
        gfpp_offline,
        torch.full_like(expected_demand, trip_ramp),
        torch.where(
            lp_warning,
            torch.full_like(expected_demand, warning_ramp),
            torch.full_like(expected_demand, normal_ramp),
        ),
    )
    return total_target, ramp_fraction


# =============================================================================
# 5. Corrected 24-h simulation body
# =============================================================================

def simulate_strategies_24_20_batched(
    iegs_env: IEGSSystem,
    initial_state: dict,
    strategies: list[dict],
    max_ode_steps: int = 3500,
    tolerance: float = 1.0e-4,
    return_trajectories: bool = True,
    solver: Optional[ModelDistributedNeurodynamicODESolver] = None,
) -> shared_sim.SimulationResult:
    """Evaluate strategies with separate hourly profiles and hard trip/recovery."""
    if not strategies:
        raise ValueError("strategies must not be empty")

    batch_size = len(strategies)
    device = iegs_env.power.B_matrix.device

    if solver is None or solver.batch_size != batch_size:
        solver = ModelDistributedNeurodynamicODESolver(
            iegs_env,
            batch_size=batch_size,
            max_ode_steps=max_ode_steps,
            tolerance=tolerance,
        )

    power_profile = np.asarray(
        getattr(iegs_env, "power_hourly_total_mw", []), dtype=float
    )
    gas_profile = np.asarray(
        getattr(iegs_env, "gas_hourly_total_mm3h", []), dtype=float
    )
    if power_profile.size != 24 or gas_profile.size != 24:
        raise ValueError("24-20 hourly power/gas profiles are missing from the environment.")

    normal_lp_ratio = float(getattr(iegs_env, "normal_lp_ratio", 0.90))
    lp_replenish_ratio = float(getattr(iegs_env, "lp_replenish_ratio", 0.60))
    lp_trip_ratio = float(getattr(iegs_env, "lp_trip_ratio", 0.30))

    # Preserve the paper's percentage convention: the initialized normal point
    # is 90% of the nominal LP reference.  The baseline is now stabilized near
    # this point rather than drifting from 90% toward 100%.
    solver.initial_lp_sum = (
        torch.sum(initial_state["lp_cur"]).to(device) / max(normal_lp_ratio, 1.0e-9)
    )
    solver.max_lp_vector = (
        initial_state["lp_cur"].to(device) / max(normal_lp_ratio, 1.0e-9)
    )
    ref_lp = solver.initial_lp_sum.to(device)

    batched_state = shared_sim._as_batch_state(initial_state, batch_size)
    base_pg = batched_state["P_g"].clone()
    base_s = batched_state["S"].clone()
    base_delta = batched_state["delta"].clone()
    base_f_in = batched_state["f_in"].clone()
    base_f_out = batched_state["f_out"].clone()
    base_comp = batched_state.get("comp_ratio", None)

    current_state = {k: v.clone() for k, v in batched_state.items()}
    prev_state = {k: v.clone() for k, v in batched_state.items()}

    total_damage = torch.zeros(batch_size, device=device)
    min_lp_pct = torch.ones(batch_size, device=device) * 100.0
    dos_lp_at_trigger = torch.ones(batch_size, device=device) * -1.0
    min_lp_after_dos_pct = torch.ones(batch_size, device=device) * 1.0e9
    dos_has_triggered = torch.zeros(batch_size, dtype=torch.bool, device=device)

    # Keep the same metric keys expected by main.py / MIP master.
    min_pressure_ratio = torch.ones(batch_size, device=device) * 100.0
    min_pressure_trip_ratio = torch.ones(batch_size, device=device) * 100.0
    dos_pressure_at_trigger = torch.ones(batch_size, device=device) * 100.0
    dos_pressure_trip_at_trigger = torch.ones(batch_size, device=device) * 100.0

    trajectories: list[list[dict]] = [[] for _ in range(batch_size)]

    # Identify the explicit no-attack baseline.  For this row the scheduled gas
    # source is held at the demand-tracking target inside each hourly DND solve;
    # this removes solver-induced source drift while retaining the same network
    # and equilibrium equations.
    baseline_mask_np = []
    for strat in strategies:
        za0 = np.asarray(
            strat.get("FDIA", np.zeros(iegs_env.power.num_buses, dtype=np.float32)),
            dtype=float,
        ).reshape(-1)
        baseline_mask_np.append(
            float(strat.get("T_fdia", 99.0)) >= 90.0
            and float(strat.get("T_dos", 99.0)) >= 90.0
            and (za0.size == 0 or float(np.max(np.abs(za0))) <= 1.0e-12)
        )
    baseline_mask = torch.tensor(
        baseline_mask_np, dtype=torch.bool, device=device
    )

    # Manual-reset protection latch.  Once LP crossed <=30% at the previous
    # period end, all GFPPs stay offline for the remaining horizon.
    gfpp_offline_latch = torch.zeros(batch_size, dtype=torch.bool, device=device)
    dos_already_triggered = torch.zeros(batch_size, dtype=torch.bool, device=device)
    dos_vulnerable_latch = torch.zeros(batch_size, dtype=torch.bool, device=device)
    pre_dos_warning_latch = torch.zeros(batch_size, dtype=torch.bool, device=device)

    locked_s_memory = torch.zeros_like(base_s)
    locked_comp_memory = torch.zeros_like(base_comp) if base_comp is not None else None
    fdia_exposure = torch.zeros(batch_size, device=device)

    orig_power_load = iegs_env.power.base_load.clone()
    orig_gas_load = iegs_env.gas.base_gas_load.clone()
    base_power_total = float(torch.sum(orig_power_load).detach().cpu().item())
    base_gas_total = float(torch.sum(orig_gas_load).detach().cpu().item())
    if base_power_total <= 1.0e-9 or base_gas_total <= 1.0e-9:
        raise ValueError("Mean-load base totals must be positive.")

    if len(iegs_env.gfpp_indices) > 0:
        gfpp_idx = iegs_env.gfpp_indices_tensor()
        gfpp_nodes = iegs_env.gfpp_gas_nodes_tensor()
    else:
        gfpp_idx = torch.tensor([], dtype=torch.long, device=device)
        gfpp_nodes = torch.arange(iegs_env.gas.num_nodes, device=device)

    try:
        for t in range(1, 25):
            power_total_t = float(power_profile[t - 1])
            gas_total_t = float(gas_profile[t - 1])
            power_mult = power_total_t / base_power_total
            gas_mult = gas_total_t / base_gas_total

            # Independent workbook profiles.
            iegs_env.power.base_load = orig_power_load * power_mult
            iegs_env.gas.base_gas_load = orig_gas_load * gas_mult

            # Warm start the electrical state around the current load level.
            current_state["P_g"] = base_pg * power_mult
            current_state["delta"] = base_delta * power_mult
            current_state["delta_cyber"] = base_delta * power_mult

            # Keep a profile-consistent gas-flow initial guess without changing LP.
            flow_mult = max(gas_mult, 1.0e-6)
            current_state["f_in"] = base_f_in * flow_mult
            current_state["f_out"] = base_f_out * flow_mult

            z_a = shared_sim._strategy_fdia_tensor(
                strategies,
                t,
                iegs_env.power.num_buses,
                device,
            )
            fdia_intensity_now = torch.clamp(
                torch.sum(torch.abs(z_a), dim=1)
                / (torch.sum(iegs_env.power.base_load) + 1.0e-6),
                min=0.0,
                max=1.0,
            )
            fdia_exposure = torch.clamp(
                0.86 * fdia_exposure + fdia_intensity_now,
                min=0.0,
                max=1.0,
            )

            # Actual DoS state requested by the attack strategy.
            s_t = torch.ones(batch_size, device=device)
            for i, strat in enumerate(strategies):
                if t >= float(strat.get("T_dos", 99.0)):
                    s_t[i] = 0.0

            lp_pct_prev = torch.sum(prev_state["lp_cur"], dim=1) / (ref_lp + 1.0e-12)
            lp_warning = lp_pct_prev <= lp_replenish_ratio
            lp_trip = lp_pct_prev <= lp_trip_ratio

            fdia_preconditioned = fdia_exposure >= float(
                getattr(iegs_env, "fdia_trip_exposure", 0.05)
            )

            just_dos = (s_t < 0.5) & (~dos_already_triggered)
            if torch.any(just_dos):
                locked_s_memory[just_dos] = prev_state["S"][just_dos].clone()
                if locked_comp_memory is not None and "comp_ratio" in prev_state:
                    locked_comp_memory[just_dos] = prev_state["comp_ratio"][just_dos].clone()
                dos_lp_at_trigger[just_dos] = lp_pct_prev[just_dos] * 100.0
                dos_already_triggered |= just_dos

            pre_dos_warning_latch |= (
                (s_t > 0.5) & fdia_preconditioned & lp_warning
            )
            dos_vulnerable_latch |= (
                (s_t < 0.5)
                & fdia_preconditioned
                & (lp_warning | pre_dos_warning_latch)
            )

            # HARD PROTECTION: if the previous period ended at/below 30%, GFPP
            # is offline from this scheduling period onward.
            newly_offline = lp_trip & (~gfpp_offline_latch)
            gfpp_offline_latch |= lp_trip

            # Make the warm-start itself consistent with the protection state.
            if gfpp_idx.numel() > 0 and torch.any(gfpp_offline_latch):
                current_state["P_g"][:, gfpp_idx] = torch.where(
                    gfpp_offline_latch.unsqueeze(-1),
                    torch.zeros_like(current_state["P_g"][:, gfpp_idx]),
                    current_state["P_g"][:, gfpp_idx],
                )

            # Native gas demand is independent of GFPP fuel demand.
            native_gas = torch.ones(batch_size, device=device) * float(
                torch.sum(iegs_env.gas.base_gas_load).detach().cpu().item()
            )

            # CRITICAL FIX: after trip, GFPP fuel demand is zero immediately in
            # the source-control calculation.  Before trip use the current nominal
            # power warm start; FDIA-induced extra dispatch is then supplied by LP.
            if gfpp_idx.numel() > 0:
                gfpp_pg_for_demand = current_state["P_g"][:, gfpp_idx].clone()
                gfpp_pg_for_demand = torch.where(
                    gfpp_offline_latch.unsqueeze(-1),
                    torch.zeros_like(gfpp_pg_for_demand),
                    gfpp_pg_for_demand,
                )
                gfpp_gas = torch.sum(
                    gfpp_pg_for_demand * iegs_env.psi_g[gfpp_idx].unsqueeze(0),
                    dim=1,
                )
            else:
                gfpp_gas = torch.zeros(batch_size, device=device)

            expected_demand = native_gas + gfpp_gas
            prev_lp_sum = torch.sum(prev_state["lp_cur"], dim=1)
            total_s_target, ramp_fraction = _corrected_source_target(
                env=iegs_env,
                expected_demand=expected_demand,
                prev_lp_sum=prev_lp_sum,
                ref_lp=ref_lp,
                lp_warning=lp_warning,
                gfpp_offline=gfpp_offline_latch,
            )
            desired_s = _dispatch_sources_for_total(iegs_env, total_s_target)
            tracked_s = _ramp_sources(
                iegs_env,
                prev_state["S"],
                desired_s,
                ramp_fraction,
            )

            # Before trip, DoS freezes source commands at the stored trigger
            # value.  After trip, local emergency protection overrides this remote
            # lock so that native gas service can recover inventory safely.
            source_control_available = (s_t > 0.5) | gfpp_offline_latch
            current_state["S"] = torch.where(
                source_control_available.unsqueeze(-1),
                tracked_s,
                locked_s_memory,
            )

            if "comp_ratio" in current_state:
                current_state["comp_ratio"] = prev_state["comp_ratio"].clone()

            current_state["P_d_sh"].zero_()
            current_state["P_q_sh"].zero_()
            current_state["lp_prev"] = prev_state["lp_cur"].clone()
            if "S_prev" in current_state:
                current_state["S_prev"] = prev_state["S"].clone()

            # Freeze the already-computed source setpoint during the equilibrium
            # solve for (i) the explicit baseline and (ii) tripped protection
            # rows.  For tripped rows this setpoint is the LOCAL emergency
            # recovery command, not the attacker-frozen remote command.  Using
            # the solver's locked-S projection here prevents the inner DND from
            # undoing the recovery target while still solving all other states.
            protection_hold = baseline_mask | gfpp_offline_latch
            solver_locked_s = locked_s_memory.clone()
            solver_locked_s[protection_hold] = current_state["S"][protection_hold]
            s_t_solver = torch.where(
                protection_hold, torch.zeros_like(s_t), s_t
            )

            attack_params = {
                "FDIA": z_a,
                "s_t": s_t_solver,
                "locked_S": solver_locked_s,
                "fdia_exposure": fdia_exposure,
                "dos_vulnerable": dos_vulnerable_latch.float(),
                "physical_hour": int(t),
            }
            if locked_comp_memory is not None:
                solver_locked_comp = locked_comp_memory.clone()
                if "comp_ratio" in current_state:
                    solver_locked_comp[protection_hold] = current_state["comp_ratio"][
                        protection_hold
                    ]
                attack_params["locked_comp"] = solver_locked_comp

            next_state, step_shed = solver.evolve_to_equilibrium_batched(
                current_state,
                prev_state,
                attack_params,
                gfpp_offline_latch,
            )

            # Explicitly enforce exact zero GFPP power after the solve as a final
            # guard against numerical residuals.  This also guarantees zero fuel
            # use in the next scheduling period.
            if gfpp_idx.numel() > 0 and torch.any(gfpp_offline_latch):
                next_state["P_g"][:, gfpp_idx] = torch.where(
                    gfpp_offline_latch.unsqueeze(-1),
                    torch.zeros_like(next_state["P_g"][:, gfpp_idx]),
                    next_state["P_g"][:, gfpp_idx],
                )

            noise_th = 2.0 if iegs_env.power.num_buses < 50 else 40.0
            step_shed = torch.where(
                step_shed < noise_th,
                torch.zeros_like(step_shed),
                step_shed,
            )
            total_damage += step_shed

            lp_pct = torch.sum(next_state["lp_cur"], dim=1) / (ref_lp + 1.0e-12) * 100.0
            min_lp_pct = torch.minimum(min_lp_pct, lp_pct)

            dos_has_triggered |= s_t < 0.5
            min_lp_after_dos_pct = torch.where(
                dos_has_triggered,
                torch.minimum(min_lp_after_dos_pct, lp_pct),
                min_lp_after_dos_pct,
            )

            min_pressure = shared_sim._min_gfpp_pressure(next_state, gfpp_nodes)

            if return_trajectories:
                actual_source = torch.sum(next_state["S"], dim=1)
                actual_gfpp_gas = torch.zeros(batch_size, device=device)
                if gfpp_idx.numel() > 0:
                    actual_gfpp_gas = torch.sum(
                        next_state["P_g"][:, gfpp_idx]
                        * iegs_env.psi_g[gfpp_idx].unsqueeze(0),
                        dim=1,
                    )

                for i in range(batch_size):
                    trajectories[i].append(
                        {
                            "time": t,
                            "lp_pct_pre": float((lp_pct_prev[i] * 100.0).detach().cpu().item()),
                            "lp_pct_post": float(lp_pct[i].detach().cpu().item()),
                            "lp_pct": float(lp_pct[i].detach().cpu().item()),
                            "step_shed": float(step_shed[i].detach().cpu().item()),
                            "min_gfpp_pressure": float(min_pressure[i].detach().cpu().item()),
                            "pressure_ratio": 100.0,
                            "pressure_replenish_ratio": 100.0,
                            "pressure_trip_ratio": 100.0,
                            "fdia_active": int(
                                torch.any(torch.abs(z_a[i]) > 1.0e-12).detach().cpu().item()
                            ),
                            # Actual attack DoS state, not the safety-bypass state.
                            "s_t": int(s_t[i].detach().cpu().item()),
                            "source_control_available": int(
                                source_control_available[i].detach().cpu().item()
                            ),
                            "gfpp_offline": int(gfpp_offline_latch[i].detach().cpu().item()),
                            "gfpp_new_trip": int(newly_offline[i].detach().cpu().item()),
                            "dos_vulnerable": int(dos_vulnerable_latch[i].detach().cpu().item()),
                            "power_load_mw": power_total_t,
                            "gas_load_mm3h": gas_total_t,
                            "power_profile_mult": power_mult,
                            "gas_profile_mult": gas_mult,
                            "gas_source_total": float(actual_source[i].detach().cpu().item()),
                            "gfpp_gas_demand": float(actual_gfpp_gas[i].detach().cpu().item()),
                            "expected_gas_demand_pre_solve": float(
                                expected_demand[i].detach().cpu().item()
                            ),
                            "source_target_total": float(
                                total_s_target[i].detach().cpu().item()
                            ),
                            "protection_mode": int(gfpp_offline_latch[i].detach().cpu().item()),
                        }
                    )

            prev_state = {k: v.clone() for k, v in next_state.items()}
            current_state = {k: v.clone() for k, v in next_state.items()}
            current_state["lp_prev"] = current_state["lp_cur"].clone()

        min_lp_after_dos_out = torch.where(
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
            "min_lp_after_dos_pct": min_lp_after_dos_out.detach().cpu().numpy(),
            "post_dos_drop_pct": post_dos_drop_pct.detach().cpu().numpy(),
        }
        return shared_sim.SimulationResult(
            total_damage.detach().cpu().numpy(),
            trajectories,
            metrics,
        )

    finally:
        iegs_env.power.base_load = orig_power_load
        iegs_env.gas.base_gas_load = orig_gas_load


# =============================================================================
# 6. Upper-level score: no reward for LP below the trip threshold
# =============================================================================

def _score_batch_24_20(self, result, offset: int):
    """24-20 coordinated score with linepack benefit capped at the 30% trip point."""
    damage = np.asarray(result.damages, dtype=float)
    min_lp = np.asarray(result.metrics["min_lp_pct"], dtype=float)
    dos_lp = np.asarray(result.metrics["dos_lp_at_trigger"], dtype=float)
    min_after = np.asarray(
        result.metrics.get("min_lp_after_dos_pct", min_lp), dtype=float
    )

    lp_rep = float(getattr(self.env, "lp_replenish_ratio", 0.60)) * 100.0
    lp_trip = float(getattr(self.env, "lp_trip_ratio", 0.30)) * 100.0

    scores = []
    for i in range(len(damage)):
        global_idx = offset + i
        post_drop = max(0.0, float(dos_lp[i] - min_after[i])) if dos_lp[i] >= 0.0 else -1.0
        info = {
            "damage": float(damage[i]),
            "min_lp_pct": float(min_lp[i]),
            "dos_lp_at_trigger": float(dos_lp[i]),
            "post_dos_drop_pct": float(post_drop),
            "min_lp_after_dos_pct": float(min_after[i]),
        }

        if self.mode == "coordinated":
            window_pass = (dos_lp[i] > lp_trip) and (dos_lp[i] <= lp_rep)
            info["window_pass"] = bool(window_pass)
            if not window_pass:
                if bool(getattr(self.env, "mip_nd_hard_window", False)):
                    score = -np.inf
                else:
                    early_gap = max(0.0, float(dos_lp[i] - lp_rep))
                    late_gap = max(0.0, float(lp_trip - dos_lp[i]))
                    score = -1.0e8 - (early_gap + late_gap) * 1.0e5
            else:
                crossed_trip = min_after[i] <= lp_trip
                # Useful depletion stops at the trip point.  There is NO extra
                # objective benefit for 20%, 10%, or 0% once protection trips.
                useful_drop = max(
                    0.0,
                    float(dos_lp[i] - max(float(min_after[i]), lp_trip)),
                )
                target = lp_rep - float(
                    getattr(self.env, "dos_window_target_margin", 10.0)
                )
                score = (
                    float(damage[i])
                    * float(getattr(self.env, "runner_damage_primary_weight", 50.0))
                    + (float(getattr(self.env, "runner_trip_cross_bonus", 1000.0)) if crossed_trip else 0.0)
                    + useful_drop
                    * float(getattr(self.env, "runner_useful_lp_drop_weight", 25.0))
                    - abs(float(dos_lp[i]) - target)
                    * float(getattr(self.env, "runner_timing_weight", 2.0))
                )
        else:
            # Keep the original project's FDIA-only / DoS-only scoring exactly.
            # The 24-20 paper-validation request here concerns the coordinated
            # trip/recovery behavior, so there is no need to duplicate those
            # well-tested branches.
            original_scores = _ORIGINAL_MIP_SCORE_BATCH(self, result, offset)
            return original_scores

        scores.append((float(score), global_idx, info))
    return scores


# =============================================================================
# 7. Runtime hook installation and CLI
# =============================================================================

def patch_project_for_24_20() -> None:
    """Install only process-local hooks; no existing project file is edited."""
    IEGSSystem._detect_gfpp_indices = _detect_gfpp_indices_24_20

    project_main.load_system_data = load_system_data_24_20
    project_main.configure_env = configure_env_24_20
    project_main.create_initial_physical_state = create_initial_physical_state_24_20

    # Replace only the central 24-h evaluator captured by the model-distributed
    # wrapper.  The actual lower equilibrium solver remains the project's DND.
    model_distributed_nd_simulation._central_simulate_strategies_batched = (
        simulate_strategies_24_20_batched
    )
    sim = model_distributed_nd_simulation.simulate_strategies_model_distributed_nd_batched
    project_main.simulate_strategies_batched = sim
    mip_nd_master.simulate_strategies_batched = sim

    # Process-local upper scoring fix.
    mip_nd_master.MIPNDOptimizer._score_batch = _score_batch_24_20


def _parse_args_for_24_20():
    """Reuse the original CLI even though its --system choices contain only 118-135."""
    original = sys.argv[:]
    filtered = [original[0]]
    skip_next = False
    for arg in original[1:]:
        if skip_next:
            skip_next = False
            continue
        if arg == "--system":
            skip_next = True
            continue
        if arg.startswith("--system="):
            continue
        filtered.append(arg)

    try:
        sys.argv = filtered
        args = project_main.parse_args()
    finally:
        sys.argv = original

    args.system = SYSTEM_NAME
    args.make_method_figures = True
    args.record_lower_trace = True
    if args.out_dir == "results":
        args.out_dir = "results_distributed"
    return args


def main() -> None:
    patch_project_for_24_20()
    args = _parse_args_for_24_20()

    print("=" * 92)
    print("Small-system validation : IEEE RTS-24 + Belgian 20-node gas network")
    print("Method                  : MIP + model-distributed DND")
    print(f"Runner version          : {VERSION}")
    print(f"Data                    : data/{DATA_FILENAME}")
    print("Normal / replenish / trip LP thresholds: 90% / 60% / 30%")
    print("Trip reset              : manual reset (latched through the 24 h horizon)")
    print("Other project files     : unchanged")
    print("=" * 92)

    project_main.run_experiment(args)


if __name__ == "__main__":
    main()

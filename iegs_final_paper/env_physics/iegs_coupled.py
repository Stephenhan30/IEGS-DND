"""Integrated electricity-gas system container."""
from __future__ import annotations

from typing import Optional

import pandas as pd
import torch

from .gas_network import GasNetworkQSS, default_device
from .power_grid import PowerGridDCOPF


class IEGSSystem:
    def __init__(self, system_data: dict, dt_hours: float = 1.0, device: Optional[torch.device] = None):
        self.device = device or default_device()
        self.power = PowerGridDCOPF(system_data["power"], device=self.device)
        self.gas = GasNetworkQSS(system_data["gas"], dt_hours=dt_hours, device=self.device)

        self.gfpp_coupling_df = (
            system_data.get("coupling", {}).get("gfpp", pd.DataFrame()).copy()
        )

        self.gfpp_indices = self._detect_gfpp_indices()
        self.gfpp_to_gas_node = self._detect_gfpp_to_gas_node_mapping()
        self.gfpp_power_buses = [
            int(self.power.gen_df.iloc[g]["Node"]) for g in self.gfpp_indices
        ]

        self.psi_g = torch.ones(self.power.num_gens, dtype=torch.float32, device=self.device) * 0.0002
        self.c_g = torch.ones(self.power.num_gens, dtype=torch.float32, device=self.device) * 0.05
        self.c_w = torch.ones(self.gas.num_sources, dtype=torch.float32, device=self.device) * 0.02
        self.c_d_sh = 500.0
        self.c_q_sh = 500.0
        self.lambda1 = 10000.0
        self.lambda2 = 10000.0

        # Two-threshold gas security logic used by the paper mechanism.
        # Normal linepack is represented around normal_lp_ratio (about 90%).
        # If linepack/pressure reaches the replenish threshold, SCADA can still
        # increase gas supply while communication is alive. If DoS locks the gas
        # controls after FDIA has pushed the system into this warning zone,
        # linepack and pressure can continue declining to the trip threshold.
        self.normal_lp_ratio = 0.90
        self.lp_replenish_ratio = 0.82
        self.lp_trip_ratio = 0.60
        # Backward-compatible aliases used by some plotting/fitness code.
        self.lp_warn_ratio = self.lp_replenish_ratio
        self.lp_shutdown_ratio = self.lp_trip_ratio
        self.replenish_recovery_rate = 0.08
        self.protected_lp_margin = 0.03
        self.protected_pressure_margin = 0.12
        # Additional post-DoS inventory drain used only for a valid coordinated
        # deadlock: it represents demand growth and GFPP over-consumption that
        # cannot be matched because source/compressor commands are locked.
        self.deadlock_drain_rate = 0.10 if self.power.num_buses < 50 else 0.07
        # Minimum available GFPP output ratio in the low-pressure derating band
        # just above the trip threshold. At or below trip pressure, availability
        # is set to zero.
        self.gfpp_pressure_derate_floor = 0.40
        self.dispatch_attack_gain = 2.40 if self.power.num_buses < 50 else 1.60
        self.fast_dynamics = False
        self.fdia_trip_exposure = 0.06
        self.fdia_full_exposure = 0.18

        # Buses where a positive FDIA is encouraged.  If GFPP buses have no load,
        # add the largest load buses so that the zero-sum load transfer remains feasible.
        gfpp_buses = [self.power.bus_id_to_idx[int(self.power.gen_df.iloc[g]["Node"])] for g in self.gfpp_indices]
        positive_load = torch.where(self.power.base_load > 1e-5)[0].detach().cpu().tolist()
        top_k = min(max(3, len(gfpp_buses)), self.power.num_buses)
        top_load_buses = torch.topk(self.power.base_load, k=top_k).indices.detach().cpu().tolist()
        self.attack_target_buses = sorted(set(gfpp_buses + positive_load[:0] + top_load_buses))

    def _detect_gfpp_indices(self) -> list[int]:
        gen_df = self.power.gen_df

        if not self.gfpp_coupling_df.empty:
            coupling = self.gfpp_coupling_df.sort_values("GFPP_ID").reset_index(drop=True)
            selected: list[int] = []
            for _, row in coupling.iterrows():
                bus = int(row["Power_Bus"])
                matches = gen_df.index[gen_df["Node"].astype(int) == bus].tolist()
                if len(matches) != 1:
                    raise ValueError(
                        f"耦合模板母线{bus}必须唯一对应一台发电机，实际匹配{matches}。"
                    )
                selected.append(int(matches[0]))
            if len(selected) != 12:
                raise ValueError(f"应识别12台燃气机组，实际识别{len(selected)}台。")
            return selected

        candidates: list[int] = []
        for idx, row in gen_df.reset_index(drop=True).iterrows():
            row_text = " ".join(str(v).lower() for v in row.values)
            if any(token in row_text for token in ["gas", "gfpp", "ccgt", "gt", "燃气", "天然气"]):
                candidates.append(idx)
        if candidates:
            return candidates
        raise ValueError(
            "未读取到GFPP耦合模板，也无法从发电机类型识别燃气机组；"
            "已禁用按前1/3发电机自动分配的旧逻辑。"
        )

    def _detect_gfpp_to_gas_node_mapping(self) -> dict[int, int]:
        if not self.gfpp_coupling_df.empty:
            coupling = self.gfpp_coupling_df.sort_values("GFPP_ID").reset_index(drop=True)
            if len(coupling) != len(self.gfpp_indices):
                raise ValueError("GFPP数量与耦合模板映射行数不一致。")
            mapping: dict[int, int] = {}
            for gen_idx, (_, row) in zip(self.gfpp_indices, coupling.iterrows()):
                actual_bus = int(self.power.gen_df.iloc[gen_idx]["Node"])
                expected_bus = int(row["Power_Bus"])
                if actual_bus != expected_bus:
                    raise ValueError(
                        f"GFPP发电机母线顺序不一致: {actual_bus} != {expected_bus}。"
                    )
                gas_idx = int(row["Gas_Node_Index"])
                if not 0 <= gas_idx < self.gas.num_nodes:
                    raise ValueError(f"非法气网内部索引: {gas_idx}")
                mapping[int(gen_idx)] = gas_idx
            return mapping

        gen_df = self.power.gen_df
        gas_cols = [c for c in gen_df.columns if "gas" in str(c).lower() and "node" in str(c).lower()]
        mapping: dict[int, int] = {}
        if gas_cols:
            col = gas_cols[0]
            for g in self.gfpp_indices:
                raw = gen_df.iloc[g][col]
                try:
                    raw_int = int(raw)
                    mapping[g] = self.gas.node_id_to_idx.get(raw_int, min(max(raw_int, 0), self.gas.num_nodes - 1))
                except Exception:
                    pass
        if len(mapping) == len(self.gfpp_indices):
            return mapping

        # Fallback: spread GFPPs through the middle/downstream gas network.
        n_gfpp = max(1, len(self.gfpp_indices))
        for i, gen_idx in enumerate(self.gfpp_indices):
            ratio = i / max(1, n_gfpp - 1)
            target = int(self.gas.num_nodes * 0.33 + self.gas.num_nodes * 0.60 * ratio)
            mapping[gen_idx] = min(max(target, 0), self.gas.num_nodes - 1)
        return mapping

    def gfpp_gas_nodes_tensor(self) -> torch.Tensor:
        nodes = [self.gfpp_to_gas_node[g] for g in self.gfpp_indices]
        return torch.tensor(nodes, dtype=torch.long, device=self.device)

    def gfpp_indices_tensor(self) -> torch.Tensor:
        return torch.tensor(self.gfpp_indices, dtype=torch.long, device=self.device)

    def calc_gfpp_gas_demand(self, pg: torch.Tensor) -> torch.Tensor:
        """Return nodal gas demand caused by GFPP generation.

        Supports ``pg`` shaped ``[num_gens]`` or ``[batch, num_gens]``.
        """
        if pg.dim() == 1:
            gas_req = torch.zeros(self.gas.num_nodes, device=self.device)
            for gen_idx in self.gfpp_indices:
                gas_req[self.gfpp_to_gas_node[gen_idx]] += pg[gen_idx] * self.psi_g[gen_idx]
            return gas_req
        gas_req = torch.zeros((pg.shape[0], self.gas.num_nodes), device=self.device)
        for gen_idx in self.gfpp_indices:
            gas_req[:, self.gfpp_to_gas_node[gen_idx]] += pg[:, gen_idx] * self.psi_g[gen_idx]
        return gas_req

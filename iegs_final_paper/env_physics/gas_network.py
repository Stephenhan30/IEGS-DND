"""Quasi-steady-state gas-network model."""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch


def default_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class GasNetworkQSS:
    def __init__(self, gas_data: dict, dt_hours: float = 1.0, device: Optional[torch.device] = None):
        self.device = device or default_device()
        self.dt = float(dt_hours)
        self.node_df = gas_data["node"].copy()
        self.pipe_df = gas_data["pipeline"].copy()
        self.source_df = gas_data.get("source", None)
        self.comp_df = gas_data.get("compressor", None)

        self.num_nodes = len(self.node_df)
        self.num_pipes = len(self.pipe_df)
        if self.num_nodes == 0 or self.num_pipes == 0:
            raise ValueError("Gas network must contain nodes and pipelines.")

        node_col = next((c for c in self.node_df.columns if "node" in str(c).lower()), self.node_df.columns[0])
        self.node_id_to_idx = {int(node_id): idx for idx, node_id in enumerate(self.node_df[node_col].values)}

        self._build_topology_matrices()
        self._extract_physics_params()

    def _build_topology_matrices(self) -> None:
        A_np_plus = np.zeros((self.num_nodes, self.num_pipes), dtype=np.float32)
        A_np_minus = np.zeros((self.num_nodes, self.num_pipes), dtype=np.float32)

        m_col = next((c for c in self.pipe_df.columns if str(c).lower() in {"m", "from", "node m", "from node"}), self.pipe_df.columns[1])
        n_col = next((c for c in self.pipe_df.columns if str(c).lower() in {"n", "to", "node n", "to node"}), self.pipe_df.columns[2])

        for pipe_idx, row in self.pipe_df.reset_index(drop=True).iterrows():
            m_idx = self.node_id_to_idx[int(row[m_col])]
            n_idx = self.node_id_to_idx[int(row[n_col])]
            A_np_plus[m_idx, pipe_idx] = 1.0
            A_np_minus[n_idx, pipe_idx] = 1.0

        if self.source_df is not None and len(self.source_df) > 0:
            self.num_sources = len(self.source_df)
            A_nw = np.zeros((self.num_nodes, self.num_sources), dtype=np.float32)
            src_node_col = next((c for c in self.source_df.columns if "node" in str(c).lower()), self.source_df.columns[0])
            for src_idx, row in self.source_df.reset_index(drop=True).iterrows():
                n_idx = self.node_id_to_idx[int(row[src_node_col])]
                A_nw[n_idx, src_idx] = 1.0
        else:
            self.num_sources = 1
            A_nw = np.zeros((self.num_nodes, 1), dtype=np.float32)
            A_nw[0, 0] = 1.0

        self.A_np_plus = torch.as_tensor(A_np_plus, dtype=torch.float32, device=self.device)
        self.A_np_minus = torch.as_tensor(A_np_minus, dtype=torch.float32, device=self.device)
        self.A_nw = torch.as_tensor(A_nw, dtype=torch.float32, device=self.device)
        self.pipe_from_idx = self.A_np_plus.argmax(dim=0)
        self.pipe_to_idx = self.A_np_minus.argmax(dim=0)

    def _extract_physics_params(self) -> None:
        wmn_col = next((c for c in self.pipe_df.columns if "wmn" in str(c).lower() or "weymouth" in str(c).lower()), None)
        if wmn_col is None:
            wmn = np.ones(self.num_pipes, dtype=np.float32)
        else:
            wmn = self.pipe_df[wmn_col].astype(float).fillna(1.0).to_numpy(dtype=np.float32)
        self.weymouth_C_p = torch.as_tensor(np.maximum(wmn, 1e-6), dtype=torch.float32, device=self.device)

        len_col = next((c for c in self.pipe_df.columns if "length" in str(c).lower() or "len" in str(c).lower()), None)
        diam_col = next((c for c in self.pipe_df.columns if "diameter" in str(c).lower() or "dia" in str(c).lower()), None)
        linepack = np.zeros(self.num_pipes, dtype=np.float32)
        for idx, row in self.pipe_df.reset_index(drop=True).iterrows():
            length = float(row[len_col]) if len_col is not None and len_col in row else 10.0
            diameter = float(row[diam_col]) if diam_col is not None and diam_col in row else 500.0
            linepack[idx] = max(1e-3, 1e-4 * length * diameter * diameter)
        self.linepack_K_p = torch.as_tensor(linepack, dtype=torch.float32, device=self.device)

        pmin_col = next((c for c in self.node_df.columns if "min" in str(c).lower() and "pressure" in str(c).lower()), None)
        pmax_col = next((c for c in self.node_df.columns if "max" in str(c).lower() and "pressure" in str(c).lower()), None)
        pi_min = self.node_df[pmin_col].astype(float).fillna(15.0).to_numpy(dtype=np.float32) if pmin_col else np.ones(self.num_nodes, dtype=np.float32) * 15.0
        pi_max = self.node_df[pmax_col].astype(float).fillna(250.0).to_numpy(dtype=np.float32) if pmax_col else np.ones(self.num_nodes, dtype=np.float32) * 250.0
        pi_max = np.maximum(pi_max, pi_min + 1.0)
        self.pi_min = torch.as_tensor(pi_min, dtype=torch.float32, device=self.device)
        self.pi_max = torch.as_tensor(pi_max, dtype=torch.float32, device=self.device)
        self.pi_warn = self.pi_min + 0.25 * (self.pi_max - self.pi_min)
        self.pi_trip = self.pi_min + 0.05 * (self.pi_max - self.pi_min)

        load_col = next((c for c in self.node_df.columns if "load" in str(c).lower()), None)
        base_load = self.node_df[load_col].astype(float).fillna(0.0).to_numpy(dtype=np.float32) if load_col else np.zeros(self.num_nodes, dtype=np.float32)
        self.base_gas_load = torch.as_tensor(base_load, dtype=torch.float32, device=self.device)

        if self.source_df is not None and len(self.source_df) > 0:
            smin_col = next((c for c in self.source_df.columns if "min" in str(c).lower()), None)
            smax_col = next((c for c in self.source_df.columns if "max" in str(c).lower()), None)
            smin = self.source_df[smin_col].astype(float).fillna(0.0).to_numpy(dtype=np.float32) if smin_col else np.zeros(self.num_sources, dtype=np.float32)
            smax = self.source_df[smax_col].astype(float).fillna(1e6).to_numpy(dtype=np.float32) if smax_col else np.ones(self.num_sources, dtype=np.float32) * 1e6
        else:
            smin = np.zeros(self.num_sources, dtype=np.float32)
            smax = np.ones(self.num_sources, dtype=np.float32) * 1e6
        self.S_min = torch.as_tensor(smin, dtype=torch.float32, device=self.device)
        self.S_max = torch.as_tensor(np.maximum(smax, smin + 1.0), dtype=torch.float32, device=self.device)

        if self.comp_df is not None and len(self.comp_df) > 0:
            self.num_comps = len(self.comp_df)
            cmin_col = next((c for c in self.comp_df.columns if "min" in str(c).lower()), None)
            cmax_col = next((c for c in self.comp_df.columns if "max" in str(c).lower()), None)
            cmin = self.comp_df[cmin_col].astype(float).fillna(1.0).to_numpy(dtype=np.float32) if cmin_col else np.ones(self.num_comps, dtype=np.float32)
            cmax = self.comp_df[cmax_col].astype(float).fillna(1.5).to_numpy(dtype=np.float32) if cmax_col else np.ones(self.num_comps, dtype=np.float32) * 1.5
            self.comp_ratio_min = torch.as_tensor(cmin, dtype=torch.float32, device=self.device)
            self.comp_ratio_max = torch.as_tensor(np.maximum(cmax, cmin), dtype=torch.float32, device=self.device)
        else:
            self.num_comps = 0
            self.comp_ratio_min = torch.ones(1, dtype=torch.float32, device=self.device)
            self.comp_ratio_max = torch.ones(1, dtype=torch.float32, device=self.device) * 1.5

    def calc_weymouth_residual(self, pi_vector: torch.Tensor, f_in: torch.Tensor, f_out: torch.Tensor) -> torch.Tensor:
        f_avg = 0.5 * (f_in + f_out)
        pi_m = pi_vector[..., self.pipe_from_idx]
        pi_n = pi_vector[..., self.pipe_to_idx]
        p_diff = pi_m.pow(2) - pi_n.pow(2)
        return f_avg * torch.abs(f_avg) - self.weymouth_C_p.pow(2) * torch.sign(p_diff) * torch.abs(p_diff)

    def calc_linepack_residual(self, lp_cur: torch.Tensor, pi_vector: torch.Tensor) -> torch.Tensor:
        pi_m = pi_vector[..., self.pipe_from_idx]
        pi_n = pi_vector[..., self.pipe_to_idx]
        return lp_cur - self.linepack_K_p * (pi_m + pi_n) / 2.0

    def calc_dynamic_mass_balance(self, lp_cur: torch.Tensor, lp_prev: torch.Tensor, f_in: torch.Tensor, f_out: torch.Tensor) -> torch.Tensor:
        return (lp_cur - lp_prev) / self.dt - (f_in - f_out)

    def calc_node_mass_balance(self, S: torch.Tensor, f_in: torch.Tensor, f_out: torch.Tensor, phi_gfpp: torch.Tensor, pq_sh: torch.Tensor) -> torch.Tensor:
        nodal_S = S @ self.A_nw.T if S.dim() == 2 else self.A_nw @ S
        net_pipe_flow = f_out @ self.A_np_minus.T - f_in @ self.A_np_plus.T if f_in.dim() == 2 else self.A_np_minus @ f_out - self.A_np_plus @ f_in
        return nodal_S + net_pipe_flow - (self.base_gas_load + phi_gfpp - pq_sh)

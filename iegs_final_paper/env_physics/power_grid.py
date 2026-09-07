"""DC power-network model used by the batched IEGS simulator."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


def default_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PowerGridDCOPF:
    """Lightweight DC network container.

    The class only stores the matrices and bounds needed by the neurodynamic
    optimizer.  The actual time-domain optimization is implemented in
    ``algorithm.ode_solver``.
    """

    def __init__(self, power_data: dict, base_mva: float = 100.0, device: Optional[torch.device] = None):
        self.device = device or default_device()
        self.base_mva = float(base_mva)
        self.bus_df = power_data["bus"].copy()
        self.branch_df = power_data["branch"].copy()
        self.gen_df = power_data["gen"].copy()

        self.bus_ids = self.bus_df["Node"].dropna().astype(int).to_numpy()
        self.num_buses = len(self.bus_ids)
        self.num_branches = len(self.branch_df)
        self.num_gens = len(self.gen_df)
        self.bus_id_to_idx = {int(bus_id): idx for idx, bus_id in enumerate(self.bus_ids)}

        if self.num_buses == 0:
            raise ValueError("Power system contains no buses.")
        if self.num_gens == 0:
            raise ValueError("Power system contains no generators.")

        self._build_network_matrices()
        self._extract_limits_and_costs()
        self._build_gen_to_bus_mapping()

    def _build_network_matrices(self) -> None:
        B_mat = np.zeros((self.num_buses, self.num_buses), dtype=np.float32)
        H_mat = np.zeros((self.num_branches, self.num_buses), dtype=np.float32)

        for row_idx, row in self.branch_df.reset_index(drop=True).iterrows():
            from_bus = int(row["From"])
            to_bus = int(row["To"])
            if from_bus not in self.bus_id_to_idx or to_bus not in self.bus_id_to_idx:
                continue
            i = self.bus_id_to_idx[from_bus]
            j = self.bus_id_to_idx[to_bus]
            x = max(float(row["Reactance(p.u.)"]), 1e-4)
            b_ij = self.base_mva / x
            B_mat[i, j] -= b_ij
            B_mat[j, i] -= b_ij
            B_mat[i, i] += b_ij
            B_mat[j, j] += b_ij
            H_mat[row_idx, i] = b_ij
            H_mat[row_idx, j] = -b_ij

        self.B_matrix = torch.as_tensor(B_mat, dtype=torch.float32, device=self.device)
        self.H_matrix = torch.as_tensor(H_mat, dtype=torch.float32, device=self.device)
        self.B_pinv = torch.linalg.pinv(self.B_matrix)

    def _extract_limits_and_costs(self) -> None:
        cap_col = "Capacity (MVA)" if "Capacity (MVA)" in self.branch_df.columns else self.branch_df.columns[-1]
        caps = self.branch_df[cap_col].astype(float).fillna(9999.0).to_numpy(dtype=np.float32)
        self.line_capacity_max = torch.as_tensor(caps, dtype=torch.float32, device=self.device)

        pmax_col = next((c for c in self.gen_df.columns if "max" in str(c).lower()), None)
        pmin_col = next((c for c in self.gen_df.columns if "min" in str(c).lower()), None)
        if pmax_col is None:
            pmax = np.ones(self.num_gens, dtype=np.float32) * 500.0
        else:
            pmax = self.gen_df[pmax_col].astype(float).fillna(500.0).to_numpy(dtype=np.float32)
        if pmin_col is None:
            pmin = np.zeros(self.num_gens, dtype=np.float32)
        else:
            pmin = self.gen_df[pmin_col].astype(float).fillna(0.0).to_numpy(dtype=np.float32)

        self.gen_pmax = torch.as_tensor(pmax, dtype=torch.float32, device=self.device)
        self.gen_pmin = torch.as_tensor(pmin, dtype=torch.float32, device=self.device)
        self.gen_ramp_max = torch.clamp(self.gen_pmax * 0.30, min=10.0)

        load_col = next((c for c in self.bus_df.columns if "load" in str(c).lower() or "pd" in str(c).lower()), None)
        if load_col is None:
            base_load = np.zeros(self.num_buses, dtype=np.float32)
        else:
            base_load = self.bus_df[load_col].astype(float).fillna(0.0).to_numpy(dtype=np.float32)
        self.base_load = torch.as_tensor(base_load, dtype=torch.float32, device=self.device)

    def _build_gen_to_bus_mapping(self) -> None:
        gen_to_bus = np.zeros((self.num_buses, self.num_gens), dtype=np.float32)
        for gen_idx, row in self.gen_df.reset_index(drop=True).iterrows():
            bus = int(row["Node"])
            if bus not in self.bus_id_to_idx:
                raise ValueError(f"Generator {gen_idx} is connected to unknown bus {bus}.")
            gen_to_bus[self.bus_id_to_idx[bus], gen_idx] = 1.0
        self.gen_to_bus_matrix = torch.as_tensor(gen_to_bus, dtype=torch.float32, device=self.device)

    def calculate_line_flows(self, theta_vector: torch.Tensor) -> torch.Tensor:
        if theta_vector.dim() == 1:
            return self.H_matrix @ theta_vector
        return theta_vector @ self.H_matrix.T

    def calculate_power_imbalance(
        self,
        pg_vector: torch.Tensor,
        theta_phys: torch.Tensor,
        theta_cyber: torch.Tensor,
        pd_shedding_vector: torch.Tensor,
        attack_injection: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if attack_injection is None:
            attack_injection = torch.zeros_like(pd_shedding_vector)
        gen_injections = self.gen_to_bus_matrix @ pg_vector
        physical_imbalance = gen_injections - (self.B_matrix @ theta_phys) - (self.base_load - pd_shedding_vector)
        cyber_imbalance = gen_injections - (self.B_matrix @ theta_cyber) - (self.base_load + attack_injection - pd_shedding_vector)
        return physical_imbalance, cyber_imbalance

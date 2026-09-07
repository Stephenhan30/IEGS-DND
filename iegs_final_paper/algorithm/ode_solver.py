"""Centralized batched neurodynamic lower-level solver."""
from __future__ import annotations

import torch


class BatchedNeurodynamicODESolver:
    def __init__(self, iegs_env, batch_size: int, max_ode_steps: int = 3500, tolerance: float = 1e-4):
        self.env = iegs_env
        self.batch_size = int(batch_size)
        self.device = self.env.power.B_matrix.device
        self.max_steps = max(1, int(max_ode_steps))
        self.tol = float(tolerance)
        self.initial_lp_sum = None
        self.max_lp_vector = None

        #
        # max_ode_steps should only control numerical resolution,
        # not the physical / pseudo-time evolution length.
        #
        # N_ref = 600 means:
        #   if max_ode_steps = 3500, each inner update is scaled by 600 / 3500.
        self.ode_reference_steps = int(getattr(self.env, "ode_reference_steps", 600))

        # Early stopping settings.
        self.ode_min_steps = int(getattr(self.env, "ode_min_steps", 30))
        self.convergence_patience = int(getattr(self.env, "ode_convergence_patience", 8))
        self.ode_rel_tolerance = float(getattr(self.env, "ode_rel_tolerance", self.tol))

        # Optional records for lower-level convergence plots.
        # These flags are off by default and are enabled only by main.py when
        # it runs an additional single-strategy convergence probe.
        if not hasattr(self.env, "nd_convergence_trace"):
            self.env.nd_convergence_trace = []

        self.gfpp_idx = torch.tensor(self.env.gfpp_indices, dtype=torch.long, device=self.device)
        self.non_gfpp_idx = torch.tensor(
            [i for i in range(self.env.power.num_gens) if i not in self.env.gfpp_indices],
            dtype=torch.long,
            device=self.device,
        )
        self.attack_target_buses = torch.tensor(
            getattr(self.env, "attack_target_buses", []),
            dtype=torch.long,
            device=self.device,
        )

        self.pipe_from_idx = self.env.gas.A_np_plus.argmax(dim=0)
        self.pipe_to_idx = self.env.gas.A_np_minus.argmax(dim=0)

        if self.env.gas.num_comps > 0 and self.env.gas.comp_df is not None:
            comp_df = self.env.gas.comp_df.reset_index(drop=True)
            cols = list(comp_df.columns)
            m_col = next((c for c in cols if str(c).lower() in ["m", "from", "start", "node_m"]), cols[0])
            n_col = next((c for c in cols if str(c).lower() in ["n", "to", "end", "node_n"]), cols[1 if len(cols) > 1 else 0])
            self.comp_m_idx = torch.tensor(
                [self.env.gas.node_id_to_idx[int(row[m_col])] for _, row in comp_df.iterrows()],
                dtype=torch.long,
                device=self.device,
            )
            self.comp_n_idx = torch.tensor(
                [self.env.gas.node_id_to_idx[int(row[n_col])] for _, row in comp_df.iterrows()],
                dtype=torch.long,
                device=self.device,
            )
        else:
            self.comp_m_idx = None
            self.comp_n_idx = None

    def _clone_tensor_state(self, state: dict) -> dict:
        cloned = {}
        for k, v in state.items():
            if torch.is_tensor(v):
                cloned[k] = v.clone()
            else:
                cloned[k] = v
        return cloned

    def _projected_state_residual(self, prev_state: dict, curr_state: dict) -> float:
        """Scale-safe projected-gradient residual for constrained ND diagnostics.

        The lower-level update is a negative-gradient step followed by physical
        projection/clipping.  At active bounds or DoS-locked variables, the raw
        unconstrained gradient does not have to vanish.  Therefore the practical
        constrained-gradient criterion is the size of the *projected state
        change* between two monitoring windows.

        A simple max-relative-change can be numerically misleading when a
        variable starts near zero, because the denominator becomes tiny.  Here
        each variable is normalized by max(||x_old||, ||x_new||, sqrt(n)), so
        the first point is bounded and the curve is suitable for plotting.
        """
        total_sq = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        count = 0

        # Variables that are actually evolved/projected in the lower-level ND.
        # P_d_sh is mainly set by the capacity-deficit rule, so it is excluded.
        diagnostic_keys = {
            "P_g", "delta", "delta_cyber", "S", "pi", "f_in", "f_out",
            "lp_cur", "P_q_sh", "comp_ratio",
        }

        for key in diagnostic_keys:
            if key not in prev_state or key not in curr_state:
                continue

            old_v = prev_state[key]
            new_v = curr_state[key]

            if not torch.is_tensor(old_v) or not torch.is_tensor(new_v):
                continue
            if old_v.shape != new_v.shape:
                continue
            if not torch.is_floating_point(old_v):
                continue

            old_flat = old_v.reshape(-1)
            new_flat = new_v.reshape(-1)
            diff = torch.norm(new_flat - old_flat, p=2)

            # Scale-safe denominator: avoids the 1e8 spike caused by variables
            # whose previous value is nearly zero.
            unit_scale = torch.sqrt(torch.tensor(float(old_flat.numel()), dtype=torch.float32, device=self.device))
            base = torch.maximum(torch.maximum(torch.norm(old_flat, p=2), torch.norm(new_flat, p=2)), unit_scale)
            rel = diff / (base + 1e-12)

            total_sq = total_sq + rel * rel
            count += 1

        if count <= 0:
            return 0.0

        return float(torch.sqrt(total_sq / float(count)).detach().cpu().item())

    # Backward-compatible alias used by older code paths.
    def _max_relative_state_change(self, prev_state: dict, curr_state: dict) -> float:
        return self._projected_state_residual(prev_state, curr_state)

    def _gradient_rms_norm(self, gradient: dict, s_t: torch.Tensor) -> float:
        """Root-mean-square gradient norm used by the gradient convergence test.

        This quantity corresponds to ||∇E(x)|| in a numerically normalized form.
        RMS normalization avoids the norm being dominated only by system size
        when comparing 24-bus and 118-bus cases.
        """
        total_sq = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        total_numel = 0

        for key, g in gradient.items():
            if not torch.is_tensor(g) or not torch.is_floating_point(g):
                continue

            gg = g
            if key in ["S", "comp_ratio"]:
                mask = torch.ones_like(gg)
                mask[s_t == 0] = 0.0
                gg = gg * mask

            # Electric shedding is set by the physical capacity-deficit rule in
            # this implementation and is not treated as a free ODE variable.
            if key == "P_d_sh":
                continue

            total_sq = total_sq + torch.sum(gg.reshape(-1) ** 2)
            total_numel += int(gg.numel())

        if total_numel <= 0:
            return 0.0

        rms = torch.sqrt(total_sq / float(total_numel))
        return float(rms.detach().cpu().item())

    def _energy_value_batched(self, state: dict, attack_strategy: dict, gfpp_offline_mask: torch.Tensor) -> float:
        """Compute the augmented energy E(x) used for lower-level convergence plots.

        The energy contains the operating/shedding cost and squared residuals of
        the retained electric-gas physical equations.  It is used only for
        monitoring and the energy-stability stopping check; it does not change
        the ODE update direction.
        """
        env_p = self.env.power
        env_g = self.env.gas
        L1 = float(self.env.lambda1)
        L2 = float(self.env.lambda2)

        pg = state["P_g"]
        pd_sh = state["P_d_sh"]
        pi = state["pi"]
        f_in = state["f_in"]
        f_out = state["f_out"]
        pq_sh = state["P_q_sh"]
        S = state["S"]
        lp_cur = state["lp_cur"]
        lp_prev = state.get("lp_prev", lp_cur)
        delta = state["delta"]
        delta_cyber = state.get("delta_cyber", delta)

        # Operating and shedding cost, averaged over the batch.
        cost = (
            torch.sum(pg * self.env.c_g.unsqueeze(0), dim=1)
            + torch.sum(pd_sh * self.env.c_d_sh, dim=1)
            + torch.sum(pq_sh * self.env.c_q_sh, dim=1)
            + torch.sum(S * self.env.c_w.unsqueeze(0), dim=1)
        )
        energy = torch.mean(cost)

        z_a = attack_strategy.get("FDIA", torch.zeros((self.batch_size, env_p.num_buses), device=self.device))
        gen_injections = pg @ env_p.gen_to_bus_matrix.T

        network_phys = delta @ env_p.B_matrix.T
        demand_phys = env_p.base_load.unsqueeze(0) - pd_sh
        p_imb_phys = gen_injections - network_phys - demand_phys

        network_cyber = delta_cyber @ env_p.B_matrix.T
        demand_cyber = env_p.base_load.unsqueeze(0) + z_a - pd_sh
        p_imb_cyber = gen_injections - network_cyber - demand_cyber

        phys_residual = torch.mean(p_imb_phys**2) + torch.mean(p_imb_cyber**2)

        # Gas Weymouth residual.
        f_avg = (f_in + f_out) / 2.0
        m_idx = self.pipe_from_idx
        n_idx = self.pipe_to_idx
        pi_m = pi[:, m_idx]
        pi_n = pi[:, n_idx]
        p_diff = pi_m**2 - pi_n**2
        C_sq = (env_g.weymouth_C_p**2).unsqueeze(0)
        wey_res = f_avg * torch.abs(f_avg) - C_sq * torch.sign(p_diff) * torch.abs(p_diff)
        phys_residual = phys_residual + torch.mean(wey_res**2)

        # Compressor pressure-ratio residual.
        if env_g.num_comps > 0 and "comp_ratio" in state and self.comp_m_idx is not None:
            comp_ratio = state["comp_ratio"]
            pi_comp_m = pi[:, self.comp_m_idx]
            pi_comp_n = pi[:, self.comp_n_idx]
            if comp_ratio.shape[1] == pi_comp_m.shape[1]:
                comp_res = pi_comp_n - pi_comp_m * comp_ratio
                phys_residual = phys_residual + torch.mean(comp_res**2)

        # Gas nodal mass balance.
        gas_req = self._gfpp_gas_demand(pg)
        nodal_S = S @ env_g.A_nw.T
        net_pipe_flow = f_out @ env_g.A_np_minus.T - f_in @ env_g.A_np_plus.T
        nodal_D = env_g.base_gas_load.unsqueeze(0) + gas_req - pq_sh
        gas_mass_res = nodal_S + net_pipe_flow - nodal_D
        phys_residual = phys_residual + torch.mean(gas_mass_res**2)

        # Dynamic linepack balance.
        lp_dyn_res = ((lp_cur - lp_prev) / env_g.dt) - (f_in - f_out)
        phys_residual = phys_residual + torch.mean(lp_dyn_res**2)

        # Static linepack-pressure relation.
        lp_static = env_g.linepack_K_p.unsqueeze(0) * (pi_m + pi_n) / 2.0
        lp_stat_res = lp_cur - lp_static
        phys_residual = phys_residual + torch.mean(lp_stat_res**2)

        energy = energy + 0.5 * L1 * phys_residual

        # Inequality / lock penalties.
        penalty = torch.tensor(0.0, dtype=torch.float32, device=self.device)

        line_flows = delta @ env_p.H_matrix.T
        flow_diff = torch.relu(torch.abs(line_flows) - env_p.line_capacity_max.unsqueeze(0))
        penalty = penalty + torch.mean(flow_diff**2)

        s_t = attack_strategy.get("s_t", torch.ones(self.batch_size, device=self.device))
        locked_S = attack_strategy.get("locked_S")
        if locked_S is not None:
            dos_mask = (1.0 - s_t).unsqueeze(-1)
            penalty = penalty + torch.mean((dos_mask * (S - locked_S)) ** 2)

        locked_comp = attack_strategy.get("locked_comp")
        if locked_comp is not None and "comp_ratio" in state:
            dos_mask = (1.0 - s_t).unsqueeze(-1)
            penalty = penalty + torch.mean((dos_mask * (state["comp_ratio"] - locked_comp)) ** 2)

        energy = energy + 0.5 * L2 * penalty
        return float(energy.detach().cpu().item())

    def _initial_lp_sum_tensor(self) -> torch.Tensor:
        if self.initial_lp_sum is None:
            raise ValueError("solver.initial_lp_sum must be set before simulation.")
        if isinstance(self.initial_lp_sum, torch.Tensor):
            return self.initial_lp_sum.to(self.device)
        return torch.tensor(float(self.initial_lp_sum), dtype=torch.float32, device=self.device)

    def _build_batched_pmax(self, gfpp_offline_mask: torch.Tensor) -> torch.Tensor:
        env_p = self.env.power
        pmax = env_p.gen_pmax.unsqueeze(0).repeat(self.batch_size, 1).clone()
        for gen_idx in self.env.gfpp_indices:
            pmax[gfpp_offline_mask, gen_idx] = 0.0
        return pmax

    def _build_batched_pmin(self, gfpp_offline_mask: torch.Tensor) -> torch.Tensor:
        env_p = self.env.power
        pmin = env_p.gen_pmin.unsqueeze(0).repeat(self.batch_size, 1).clone()
        for gen_idx in self.env.gfpp_indices:
            pmin[gfpp_offline_mask, gen_idx] = 0.0
        return pmin

    def _apply_fdia_redispatch(self, state: dict, attack_strategy: dict, gfpp_offline_mask: torch.Tensor) -> torch.Tensor:
        """FDIA drives GFPP over-dispatch and extra gas consumption."""
        pg = state["P_g"].clone()
        z_a = attack_strategy.get("FDIA")
        if z_a is None or self.gfpp_idx.numel() == 0:
            return pg

        positive_part = torch.relu(z_a)
        if self.attack_target_buses.numel() > 0:
            fdia_drive = torch.sum(positive_part[:, self.attack_target_buses], dim=1)
        else:
            fdia_drive = torch.sum(positive_part, dim=1)

        shift_req = fdia_drive * float(getattr(self.env, "dispatch_attack_gain", 1.0))
        if torch.max(shift_req).item() <= 1e-9:
            return pg

        gfpp_headroom = torch.clamp(
            self.env.power.gen_pmax[self.gfpp_idx].unsqueeze(0) - pg[:, self.gfpp_idx],
            min=0.0,
        )
        gfpp_headroom = torch.where(gfpp_offline_mask.unsqueeze(-1), torch.zeros_like(gfpp_headroom), gfpp_headroom)
        total_headroom = torch.sum(gfpp_headroom, dim=1) + 1e-6
        actual_shift = torch.minimum(shift_req, total_headroom)
        pg[:, self.gfpp_idx] += gfpp_headroom / total_headroom.unsqueeze(-1) * actual_shift.unsqueeze(-1)

        if self.non_gfpp_idx.numel() > 0:
            reducible = torch.clamp(
                pg[:, self.non_gfpp_idx] - self.env.power.gen_pmin[self.non_gfpp_idx].unsqueeze(0),
                min=0.0,
            )
            total_reducible = torch.sum(reducible, dim=1) + 1e-6
            red = torch.minimum(actual_shift, total_reducible)
            pg[:, self.non_gfpp_idx] -= reducible / total_reducible.unsqueeze(-1) * red.unsqueeze(-1)
        return pg

    def _gfpp_gas_demand(self, pg: torch.Tensor) -> torch.Tensor:
        gas_req = torch.zeros((pg.shape[0], self.env.gas.num_nodes), dtype=pg.dtype, device=self.device)
        for gen_idx in self.env.gfpp_indices:
            gas_node = self.env.gfpp_to_gas_node[gen_idx]
            gas_req[:, gas_node] += pg[:, gen_idx] * self.env.psi_g[gen_idx]
        return gas_req

    def _calculate_analytical_gradient_batched(self, state: dict, attack_strategy: dict, gfpp_offline_mask: torch.Tensor) -> dict:
        """Analytical gradient of the augmented energy function."""
        grad = {k: torch.zeros_like(v) for k, v in state.items()}
        env_p = self.env.power
        env_g = self.env.gas
        L1 = float(self.env.lambda1)
        L2 = float(self.env.lambda2)

        pg = state["P_g"]
        pd_sh = state["P_d_sh"]
        pi = state["pi"]
        f_in = state["f_in"]
        f_out = state["f_out"]
        pq_sh = state["P_q_sh"]
        S = state["S"]
        lp_cur = state["lp_cur"]
        lp_prev = state.get("lp_prev", lp_cur)
        delta = state["delta"]
        delta_cyber = state.get("delta_cyber", delta)

        grad["P_g"] += self.env.c_g.unsqueeze(0)
        grad["P_d_sh"] += self.env.c_d_sh
        grad["P_q_sh"] += self.env.c_q_sh
        grad["S"] += self.env.c_w.unsqueeze(0)

        z_a = attack_strategy.get("FDIA", torch.zeros((self.batch_size, env_p.num_buses), device=self.device))
        gen_injections = pg @ env_p.gen_to_bus_matrix.T

        network_phys = delta @ env_p.B_matrix.T
        demand_phys = env_p.base_load.unsqueeze(0) - pd_sh
        p_imb_phys = gen_injections - network_phys - demand_phys

        network_cyber = delta_cyber @ env_p.B_matrix.T
        demand_cyber = env_p.base_load.unsqueeze(0) + z_a - pd_sh
        p_imb_cyber = gen_injections - network_cyber - demand_cyber

        grad["P_g"] += L1 * (p_imb_phys @ env_p.gen_to_bus_matrix + p_imb_cyber @ env_p.gen_to_bus_matrix)
        grad["delta"] += L1 * (-p_imb_phys @ env_p.B_matrix)
        if "delta_cyber" in grad:
            grad["delta_cyber"] += L1 * (-p_imb_cyber @ env_p.B_matrix)
        grad["P_d_sh"] += L1 * (p_imb_phys + p_imb_cyber)

        # Gas Weymouth residual.
        f_avg = (f_in + f_out) / 2.0
        m_idx = self.pipe_from_idx
        n_idx = self.pipe_to_idx
        pi_m = pi[:, m_idx]
        pi_n = pi[:, n_idx]
        p_diff = pi_m**2 - pi_n**2
        C_sq = (env_g.weymouth_C_p**2).unsqueeze(0)
        wey_res = f_avg * torch.abs(f_avg) - C_sq * torch.sign(p_diff) * torch.abs(p_diff)
        d_wey_df = torch.clamp(torch.abs(f_avg), min=0.1)
        grad["f_in"] += L1 * wey_res * d_wey_df
        grad["f_out"] += L1 * wey_res * d_wey_df
        d_wey_dpi_m = -C_sq * 2.0 * pi_m
        d_wey_dpi_n = C_sq * 2.0 * pi_n
        grad["pi"].scatter_add_(1, m_idx.unsqueeze(0).expand(self.batch_size, -1), L1 * wey_res * d_wey_dpi_m)
        grad["pi"].scatter_add_(1, n_idx.unsqueeze(0).expand(self.batch_size, -1), L1 * wey_res * d_wey_dpi_n)

        # Compressor pressure ratio.
        if env_g.num_comps > 0 and "comp_ratio" in state and self.comp_m_idx is not None:
            comp_ratio = state["comp_ratio"]
            pi_comp_m = pi[:, self.comp_m_idx]
            pi_comp_n = pi[:, self.comp_n_idx]
            if comp_ratio.shape[1] == pi_comp_m.shape[1]:
                comp_res = pi_comp_n - pi_comp_m * comp_ratio
                grad["pi"].scatter_add_(1, self.comp_n_idx.unsqueeze(0).expand(self.batch_size, -1), L1 * comp_res)
                grad["pi"].scatter_add_(1, self.comp_m_idx.unsqueeze(0).expand(self.batch_size, -1), L1 * comp_res * (-comp_ratio))
                grad["comp_ratio"] += L1 * comp_res * (-pi_comp_m)
                locked_comp = attack_strategy.get("locked_comp")
                s_t = attack_strategy.get("s_t", torch.ones(self.batch_size, device=self.device))
                if locked_comp is not None:
                    dos_mask = (1.0 - s_t).unsqueeze(-1)
                    grad["comp_ratio"] += L2 * dos_mask * (comp_ratio - locked_comp)

        # Gas nodal mass balance.
        gas_req = self._gfpp_gas_demand(pg)
        nodal_S = S @ env_g.A_nw.T
        net_pipe_flow = f_out @ env_g.A_np_minus.T - f_in @ env_g.A_np_plus.T
        nodal_D = env_g.base_gas_load.unsqueeze(0) + gas_req - pq_sh
        gas_mass_res = nodal_S + net_pipe_flow - nodal_D
        grad["S"] += L1 * (gas_mass_res @ env_g.A_nw)
        grad["f_out"] += L1 * (gas_mass_res @ env_g.A_np_minus)
        grad["f_in"] += L1 * (-gas_mass_res @ env_g.A_np_plus)
        grad["P_q_sh"] += L1 * gas_mass_res

        # Dynamic linepack balance.
        lp_dyn_res = ((lp_cur - lp_prev) / env_g.dt) - (f_in - f_out)
        grad["lp_cur"] += L1 * lp_dyn_res * (1.0 / env_g.dt)
        grad["f_in"] += L1 * lp_dyn_res * (-1.0)
        grad["f_out"] += L1 * lp_dyn_res

        # Static linepack-pressure relation.
        lp_static = env_g.linepack_K_p.unsqueeze(0) * (pi_m + pi_n) / 2.0
        lp_stat_res = lp_cur - lp_static
        grad["lp_cur"] += L1 * lp_stat_res
        term = L1 * lp_stat_res * (-env_g.linepack_K_p.unsqueeze(0) / 2.0)
        grad["pi"].scatter_add_(1, m_idx.unsqueeze(0).expand(self.batch_size, -1), term)
        grad["pi"].scatter_add_(1, n_idx.unsqueeze(0).expand(self.batch_size, -1), term)

        # Line capacity soft penalty.
        line_flows = delta @ env_p.H_matrix.T
        flow_diff = torch.abs(line_flows) - env_p.line_capacity_max.unsqueeze(0)
        violating = flow_diff > 0
        if violating.any():
            sign_flow = torch.sign(line_flows)
            dG_ddelta = sign_flow.unsqueeze(-1) * env_p.H_matrix.unsqueeze(0)
            violating_flow_diff = (flow_diff * violating.float()).unsqueeze(-1)
            grad["delta"] += L2 * torch.sum(violating_flow_diff * dG_ddelta, dim=1)

        # DoS lock penalty for gas source.
        s_t = attack_strategy.get("s_t", torch.ones(self.batch_size, device=self.device))
        locked_S = attack_strategy.get("locked_S")
        if locked_S is not None:
            dos_mask = (1.0 - s_t).unsqueeze(-1)
            grad["S"] += L2 * dos_mask * (S - locked_S)
        return grad

    def _apply_physical_bounds_batched(self, state: dict, prev_state: dict, attack_strategy: dict, s_t: torch.Tensor, gfpp_offline_mask: torch.Tensor) -> dict:
        env_p = self.env.power
        env_g = self.env.gas

        state["pi"] = torch.maximum(torch.minimum(state["pi"], env_g.pi_max.unsqueeze(0)), env_g.pi_min.unsqueeze(0))
        state["delta"] = torch.clamp(state["delta"], min=-10.0, max=10.0)
        if "delta_cyber" in state:
            state["delta_cyber"] = torch.clamp(state["delta_cyber"], min=-10.0, max=10.0)
        state["f_in"] = torch.clamp(state["f_in"], min=0.0, max=50000.0)
        state["f_out"] = torch.clamp(state["f_out"], min=0.0, max=50000.0)
        state["P_d_sh"] = torch.maximum(torch.minimum(state["P_d_sh"], env_p.base_load.unsqueeze(0)), torch.zeros_like(state["P_d_sh"]))
        state["P_q_sh"] = torch.maximum(torch.minimum(state["P_q_sh"], env_g.base_gas_load.unsqueeze(0) * 0.05), torch.zeros_like(state["P_q_sh"]))

        if "locked_S" in attack_strategy:
            locked_S = attack_strategy["locked_S"]
            bounded_S = torch.maximum(torch.minimum(state["S"], env_g.S_max.unsqueeze(0)), env_g.S_min.unsqueeze(0))
            state["S"] = torch.where(s_t.unsqueeze(-1) == 1, bounded_S, locked_S)
        else:
            state["S"] = torch.maximum(torch.minimum(state["S"], env_g.S_max.unsqueeze(0)), env_g.S_min.unsqueeze(0))

        if "comp_ratio" in state:
            if env_g.num_comps > 0:
                bounded_comp = torch.maximum(torch.minimum(state["comp_ratio"], env_g.comp_ratio_max.unsqueeze(0)), env_g.comp_ratio_min.unsqueeze(0))
            else:
                bounded_comp = torch.clamp(state["comp_ratio"], min=1.0, max=1.5)
            if "locked_comp" in attack_strategy:
                locked_comp = attack_strategy["locked_comp"]
                state["comp_ratio"] = torch.where(s_t.unsqueeze(-1) == 1, bounded_comp, locked_comp)
            else:
                state["comp_ratio"] = bounded_comp

        batched_pmax = self._build_batched_pmax(gfpp_offline_mask)
        batched_pmin = self._build_batched_pmin(gfpp_offline_mask)
        state["P_g"] = torch.maximum(torch.minimum(state["P_g"], batched_pmax), batched_pmin)

        # Ramp limits.
        delta_pg = state["P_g"] - prev_state["P_g"]
        rmax = env_p.gen_ramp_max.unsqueeze(0)
        state["P_g"] = prev_state["P_g"] + torch.maximum(torch.minimum(delta_pg, rmax), -rmax)
        state["P_g"] = torch.maximum(torch.minimum(state["P_g"], batched_pmax), batched_pmin)

        # Aggregate linepack dynamics from total S-D imbalance.
        gas_req = self._gfpp_gas_demand(state["P_g"])
        total_D = torch.sum(env_g.base_gas_load.unsqueeze(0) + gas_req - state["P_q_sh"], dim=1)
        total_S = torch.sum(state["S"], dim=1)
        lp_total_target = torch.sum(prev_state["lp_cur"], dim=1) + (total_S - total_D) * env_g.dt * float(getattr(self.env, "linepack_balance_scale", 1.0))
        max_lp_sum = self._initial_lp_sum_tensor()
        lp_total_target = torch.clamp(lp_total_target, min=max_lp_sum * 0.005, max=max_lp_sum * 1.05)
        cur_sum = torch.sum(torch.clamp(state["lp_cur"], min=1e-6), dim=1) + 1e-6
        state["lp_cur"] = state["lp_cur"] * (lp_total_target / cur_sum).unsqueeze(-1)
        if self.max_lp_vector is not None:
            max_vec = self.max_lp_vector.to(self.device).unsqueeze(0)
            state["lp_cur"] = torch.maximum(torch.minimum(state["lp_cur"], max_vec), torch.ones_like(state["lp_cur"]) * 1e-6)
        else:
            state["lp_cur"] = torch.clamp(state["lp_cur"], min=1e-6)

        # Electric load shedding only from generation capacity deficit after GFPP outage.
        total_available_pmax = torch.sum(batched_pmax, dim=1)
        total_load = torch.sum(env_p.base_load)
        deficit_ratio = torch.clamp(torch.relu(total_load - total_available_pmax) / (total_load + 1e-6), min=0.0, max=1.0)
        forced_shedding = env_p.base_load.unsqueeze(0) * deficit_ratio.unsqueeze(-1)
        state["P_d_sh"] = torch.maximum(state["P_d_sh"], forced_shedding)
        state["P_d_sh"] = torch.minimum(state["P_d_sh"], env_p.base_load.unsqueeze(0))

        state["delta"][:, 0] = 0.0
        if "delta_cyber" in state:
            state["delta_cyber"][:, 0] = 0.0
        return state

    @torch.no_grad()
    def evolve_to_equilibrium_batched(self, current_state: dict, prev_state: dict, attack_strategy: dict, gfpp_offline_mask: torch.Tensor):
        state = {k: v.clone().to(self.device) for k, v in current_state.items()}
        prev_state = {k: v.clone().to(self.device) for k, v in prev_state.items()}
        s_t = attack_strategy.get("s_t", torch.ones(self.batch_size, device=self.device))

        # Optional convergence trace.  The trace follows the paper-style
        # criteria: gradient norm and energy-function stabilization.
        physical_hour = attack_strategy.get("physical_hour", -1)
        try:
            physical_hour_int = int(physical_hour)
        except Exception:
            physical_hour_int = -1

        record_enabled = bool(getattr(self.env, "record_nd_convergence", False))
        record_hour = getattr(self.env, "record_nd_hour", None)
        if record_hour is not None:
            try:
                record_enabled = record_enabled and (physical_hour_int == int(record_hour))
            except Exception:
                record_enabled = False

        if record_enabled and not hasattr(self.env, "nd_convergence_trace"):
            self.env.nd_convergence_trace = []

        grad_tolerance = float(getattr(self.env, "ode_grad_tolerance", self.ode_rel_tolerance))
        energy_tolerance = float(getattr(self.env, "ode_energy_tolerance", 1e-4))
        energy_patience = int(getattr(self.env, "ode_energy_convergence_patience", self.convergence_patience))

        state["P_g"] = self._apply_fdia_redispatch(state, attack_strategy, gfpp_offline_mask)

        # State snapshot used for the projected-gradient residual.
        # It measures how much the physically projected state changes between
        # two monitoring windows.  This is the constrained-version numerical
        # realization of the Chapter-3 gradient criterion.
        state_history = self._clone_tensor_state(state)

        m = {k: torch.zeros_like(v) for k, v in state.items()}
        v = {k: torch.zeros_like(v) for k, v in state.items()}
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        is_large_system = self.env.power.num_buses > 50
        if getattr(self.env, "fast_dynamics", False):
            n_steps = min(self.max_steps, int(getattr(self.env, "fast_inner_steps_cap", 80)))
        else:
            n_steps = self.max_steps
        n_steps = max(1, int(n_steps))

        # Local pseudo-time scaling.  If fast mode caps the actual loop length,
        # use the actual loop length to keep the pseudo-time horizon consistent.
        local_dtau_scale = float(self.ode_reference_steps) / float(max(n_steps, 1))

        lr_map = {
            "P_g": 0.20 if is_large_system else 0.30,
            "S": 0.03 if is_large_system else 0.05,
            "f_in": 0.03 if is_large_system else 0.05,
            "f_out": 0.03 if is_large_system else 0.05,
            "pi": 0.03 if is_large_system else 0.05,
            "delta": 0.05 if is_large_system else 0.08,
            "delta_cyber": 0.05,
            "comp_ratio": 0.01,
            "P_d_sh": 0.20,
            "P_q_sh": 0.05,
            "lp_cur": 0.05,
        }

        previous_energy_value = None
        energy_stable_count = 0
        stop_reason = "max_steps_reached"
        # Check / record every reference-equivalent effective interval.
        # This only makes the convergence trace denser; it does not change the
        # neural-dynamic update rule or the final physical response.
        trace_effective_interval = float(getattr(self.env, "nd_trace_effective_interval", 10.0))
        check_interval = max(1, int(round(trace_effective_interval / max(local_dtau_scale, 1e-12))))
        min_check_step = max(self.ode_min_steps, check_interval)

        for step in range(1, n_steps + 1):
            gradient = self._calculate_analytical_gradient_batched(state, attack_strategy, gfpp_offline_mask)
            for key in gradient.keys():
                mask = torch.ones_like(gradient[key])
                if key in ["S", "comp_ratio"]:
                    mask[s_t == 0] = 0.0
                # P_d_sh is set by capacity deficit, not by free ODE optimization.
                if key == "P_d_sh":
                    mask[:] = 0.0

                grad_clipped = torch.clamp(gradient[key], min=-1e6, max=1e6) * mask
                m[key] = beta1 * m[key] + (1.0 - beta1) * grad_clipped
                v[key] = beta2 * v[key] + (1.0 - beta2) * (grad_clipped**2)
                m_hat = m[key] / (1.0 - beta1**step)
                v_hat = v[key] / (1.0 - beta2**step)

                # The decay schedule is expressed in reference pseudo-time units,
                # not in raw loop index units.  This keeps 600-step and 3500-step
                # runs comparable.
                effective_step = step * local_dtau_scale
                current_lr = lr_map.get(key, 0.05) / (1.0 + 0.001 * effective_step)

                # Critical fix: scale the inner ODE update by local_dtau_scale.
                # Larger max_ode_steps now means smaller numerical steps, not a
                # longer pseudo-time evolution.
                state[key] -= local_dtau_scale * current_lr * (m_hat / (torch.sqrt(v_hat) + eps))

            state = self._apply_physical_bounds_batched(state, prev_state, attack_strategy, s_t, gfpp_offline_mask)

            # Early stopping based on relative movement over reference-equivalent
            # intervals.  This avoids falsely stopping merely because individual
            # micro-steps are small when max_ode_steps is large.
            if step >= min_check_step and step % check_interval == 0:
                # Paper-style convergence diagnostics:
                #   1) gradient criterion: ||grad E(x)||_2 <= epsilon;
                #   2) energy-stability criterion: |E(x_{tau+dt}) - E(x_tau)| <= mu
                #      for consecutive checking windows.
                check_gradient = self._calculate_analytical_gradient_batched(state, attack_strategy, gfpp_offline_mask)
                gradient_norm = self._gradient_rms_norm(check_gradient, s_t)
                energy_value = self._energy_value_batched(state, attack_strategy, gfpp_offline_mask)

                if previous_energy_value is None:
                    energy_abs_change_raw = float("nan")
                    energy_rel_change = float("nan")
                    energy_pass = False
                else:
                    energy_abs_change_raw = abs(float(energy_value) - float(previous_energy_value))
                    energy_rel_change = energy_abs_change_raw / (abs(float(previous_energy_value)) + 1e-12)
                    energy_pass = energy_rel_change <= energy_tolerance

                previous_energy_value = energy_value

                # Projected-gradient residual:
                # after the negative-gradient step and physical projection,
                # if the state barely changes between two monitoring windows,
                # the constrained neural dynamics has reached a numerical
                # stationary response.  This avoids falsely requiring the raw
                # unconstrained gradient to vanish at active physical bounds.
                projected_gradient_residual = self._projected_state_residual(state_history, state)
                projected_gradient_pass = projected_gradient_residual <= self.ode_rel_tolerance

                gradient_pass = gradient_norm <= grad_tolerance

                if energy_pass:
                    energy_stable_count += 1
                else:
                    energy_stable_count = 0

                if record_enabled:
                    self.env.nd_convergence_trace.append(
                        {
                            "physical_hour": int(physical_hour_int),
                            "raw_step": int(step),
                            "effective_step": float(step * local_dtau_scale),

                            # Constrained gradient criterion used for the paper figure.
                            "projected_gradient_residual": float(projected_gradient_residual),
                            "projected_gradient_tolerance": float(self.ode_rel_tolerance),
                            "projected_gradient_converged": bool(projected_gradient_pass),

                            # Raw / normalized gradient diagnostics kept for transparency.
                            "gradient_norm": float(gradient_norm),
                            "energy_value": float(energy_value),

                            # Energy-stability criterion for the paper figure:
                            # relative |E_k - E_{k-1}| / (|E_{k-1}| + eps).
                            "energy_rel_change": float(energy_rel_change) if energy_rel_change == energy_rel_change else float("nan"),
                            "energy_abs_change": float(energy_rel_change) if energy_rel_change == energy_rel_change else float("nan"),

                            "energy_abs_change_raw": float(energy_abs_change_raw) if energy_abs_change_raw == energy_abs_change_raw else float("nan"),

                            "gradient_tolerance": float(grad_tolerance),
                            "energy_tolerance": float(energy_tolerance),
                            "energy_stable_count": int(energy_stable_count),
                            "energy_patience": int(energy_patience),
                            "gradient_converged": bool(gradient_pass),
                            "energy_converged": bool(energy_stable_count >= energy_patience),
                            "stop_reason": "",
                        }
                    )

                if gradient_pass:
                    stop_reason = "gradient_converged"
                    if record_enabled and len(self.env.nd_convergence_trace) > 0:
                        self.env.nd_convergence_trace[-1]["stop_reason"] = stop_reason
                    break

                if energy_stable_count >= energy_patience:
                    stop_reason = "energy_stabilized"
                    if record_enabled and len(self.env.nd_convergence_trace) > 0:
                        self.env.nd_convergence_trace[-1]["stop_reason"] = stop_reason
                    break

                # Prepare the next monitoring-window residual.
                state_history = self._clone_tensor_state(state)

        if record_enabled and len(getattr(self.env, "nd_convergence_trace", [])) > 0:
            # If the last recorded row has no explicit stop reason, the inner
            # solver ended because the maximum number of steps was reached.
            if self.env.nd_convergence_trace[-1].get("stop_reason", "") == "":
                self.env.nd_convergence_trace[-1]["stop_reason"] = stop_reason

        state = self._apply_physical_bounds_batched(state, prev_state, attack_strategy, s_t, gfpp_offline_mask)
        state["lp_prev"] = state["lp_cur"].clone()
        state["S_prev"] = state["S"].clone()
        power_shedding = torch.sum(state["P_d_sh"], dim=1)
        return state, power_shedding

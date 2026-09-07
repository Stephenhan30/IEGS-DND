"""Block-distributed electric–gas neurodynamic solver."""
from __future__ import annotations

from typing import Iterable

import torch

from algorithm.ode_solver import BatchedNeurodynamicODESolver


class DistributedBatchedNeurodynamicODESolver(BatchedNeurodynamicODESolver):
    """Electric-gas distributed variant of the batched lower-level ND solver.

    The class is API-compatible with ``BatchedNeurodynamicODESolver``.  Existing
    simulation code can use it by passing an instance through the optional
    ``solver=...`` argument of ``simulation.simulate_strategies_batched``.
    """

    electric_keys = ("P_g", "delta", "delta_cyber", "P_d_sh")
    gas_keys = ("S", "pi", "f_in", "f_out", "lp_cur", "P_q_sh", "comp_ratio")

    def _available_keys(self, state: dict, keys: Iterable[str]) -> list[str]:
        return [k for k in keys if k in state and torch.is_tensor(state[k])]

    def _projected_state_residual_for_keys(self, prev_state: dict, curr_state: dict, keys: Iterable[str]) -> float:
        """Projected residual restricted to one local subsystem."""
        total_sq = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        count = 0

        for key in keys:
            if key not in prev_state or key not in curr_state:
                continue
            old_v = prev_state[key]
            new_v = curr_state[key]
            if not torch.is_tensor(old_v) or not torch.is_tensor(new_v):
                continue
            if old_v.shape != new_v.shape or not torch.is_floating_point(old_v):
                continue

            old_flat = old_v.reshape(-1)
            new_flat = new_v.reshape(-1)
            diff = torch.norm(new_flat - old_flat, p=2)
            unit_scale = torch.sqrt(torch.tensor(float(old_flat.numel()), dtype=torch.float32, device=self.device))
            base = torch.maximum(torch.maximum(torch.norm(old_flat, p=2), torch.norm(new_flat, p=2)), unit_scale)
            rel = diff / (base + 1e-12)
            total_sq = total_sq + rel * rel
            count += 1

        if count <= 0:
            return 0.0
        return float(torch.sqrt(total_sq / float(count)).detach().cpu().item())

    def _mask_for_key(self, key: str, grad_value: torch.Tensor, s_t: torch.Tensor) -> torch.Tensor:
        """Return the same feasibility mask as the centralized solver."""
        mask = torch.ones_like(grad_value)
        if key in ["S", "comp_ratio"]:
            # Under DoS, source injection and compressor ratio are locked at the
            # previous physical operating point, so they are not free variables.
            mask[s_t == 0] = 0.0
        if key == "P_d_sh":
            # Electric shedding is determined by physical capacity deficit after
            # projection instead of being a free ODE state.
            mask[:] = 0.0
        return mask

    def _adam_update_block(
        self,
        state: dict,
        attack_strategy: dict,
        gfpp_offline_mask: torch.Tensor,
        s_t: torch.Tensor,
        keys: Iterable[str],
        m: dict,
        v: dict,
        step: int,
        local_dtau_scale: float,
        lr_map: dict[str, float],
        beta1: float,
        beta2: float,
        eps: float,
    ) -> tuple[dict, dict]:
        """Perform one local ND block update using the current coupled state."""
        gradient = self._calculate_analytical_gradient_batched(state, attack_strategy, gfpp_offline_mask)
        effective_step = step * local_dtau_scale

        for key in self._available_keys(state, keys):
            if key not in gradient:
                continue
            if not torch.is_floating_point(gradient[key]):
                continue

            mask = self._mask_for_key(key, gradient[key], s_t)
            grad_clipped = torch.clamp(gradient[key], min=-1e6, max=1e6) * mask

            m[key] = beta1 * m[key] + (1.0 - beta1) * grad_clipped
            v[key] = beta2 * v[key] + (1.0 - beta2) * (grad_clipped**2)
            m_hat = m[key] / (1.0 - beta1**step)
            v_hat = v[key] / (1.0 - beta2**step)
            # A diminishing pseudo-time step is required for the projected
            # distributed dynamics to settle instead of oscillating with a
            # nearly constant Adam step.  The model-distributed runner sets a
            # stronger default decay; legacy solvers keep the old 0.001 value.
            lr_decay = float(getattr(self.env, "mdnd_lr_decay", 0.001))
            current_lr = lr_map.get(key, 0.05) / (1.0 + lr_decay * effective_step)

            state[key] -= local_dtau_scale * current_lr * (m_hat / (torch.sqrt(v_hat) + eps))

        return state, gradient

    def _apply_electric_projection_only(
        self,
        state: dict,
        prev_state: dict,
        gfpp_offline_mask: torch.Tensor,
    ) -> dict:
        """Projection for electric local variables only."""
        env_p = self.env.power

        state["delta"] = torch.clamp(state["delta"], min=-10.0, max=10.0)
        if "delta_cyber" in state:
            state["delta_cyber"] = torch.clamp(state["delta_cyber"], min=-10.0, max=10.0)

        state["P_d_sh"] = torch.maximum(
            torch.minimum(state["P_d_sh"], env_p.base_load.unsqueeze(0)),
            torch.zeros_like(state["P_d_sh"]),
        )

        batched_pmax = self._build_batched_pmax(gfpp_offline_mask)
        batched_pmin = self._build_batched_pmin(gfpp_offline_mask)
        state["P_g"] = torch.maximum(torch.minimum(state["P_g"], batched_pmax), batched_pmin)

        delta_pg = state["P_g"] - prev_state["P_g"]
        rmax = env_p.gen_ramp_max.unsqueeze(0)
        state["P_g"] = prev_state["P_g"] + torch.maximum(torch.minimum(delta_pg, rmax), -rmax)
        state["P_g"] = torch.maximum(torch.minimum(state["P_g"], batched_pmax), batched_pmin)

        # Electric load shedding from capacity deficit after GFPP outage.
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

    def _apply_gas_projection_only(
        self,
        state: dict,
        prev_state: dict,
        attack_strategy: dict,
        s_t: torch.Tensor,
    ) -> dict:
        """Projection for gas local variables only."""
        env_g = self.env.gas

        state["pi"] = torch.maximum(torch.minimum(state["pi"], env_g.pi_max.unsqueeze(0)), env_g.pi_min.unsqueeze(0))
        state["f_in"] = torch.clamp(state["f_in"], min=0.0, max=50000.0)
        state["f_out"] = torch.clamp(state["f_out"], min=0.0, max=50000.0)
        state["P_q_sh"] = torch.maximum(
            torch.minimum(state["P_q_sh"], env_g.base_gas_load.unsqueeze(0) * 0.05),
            torch.zeros_like(state["P_q_sh"]),
        )

        if "locked_S" in attack_strategy:
            locked_S = attack_strategy["locked_S"]
            bounded_S = torch.maximum(torch.minimum(state["S"], env_g.S_max.unsqueeze(0)), env_g.S_min.unsqueeze(0))
            state["S"] = torch.where(s_t.unsqueeze(-1) == 1, bounded_S, locked_S)
        else:
            state["S"] = torch.maximum(torch.minimum(state["S"], env_g.S_max.unsqueeze(0)), env_g.S_min.unsqueeze(0))

        if "comp_ratio" in state:
            if env_g.num_comps > 0:
                bounded_comp = torch.maximum(
                    torch.minimum(state["comp_ratio"], env_g.comp_ratio_max.unsqueeze(0)),
                    env_g.comp_ratio_min.unsqueeze(0),
                )
            else:
                bounded_comp = torch.clamp(state["comp_ratio"], min=1.0, max=1.5)

            if "locked_comp" in attack_strategy:
                locked_comp = attack_strategy["locked_comp"]
                state["comp_ratio"] = torch.where(s_t.unsqueeze(-1) == 1, bounded_comp, locked_comp)
            else:
                state["comp_ratio"] = bounded_comp

        # Aggregate dynamic linepack projection using the current electric-side
        # GFPP output as exchanged coupling information.
        gas_req = self._gfpp_gas_demand(state["P_g"])
        total_D = torch.sum(env_g.base_gas_load.unsqueeze(0) + gas_req - state["P_q_sh"], dim=1)
        total_S = torch.sum(state["S"], dim=1)
        lp_total_target = torch.sum(prev_state["lp_cur"], dim=1) + (
            total_S - total_D
        ) * env_g.dt * float(getattr(self.env, "linepack_balance_scale", 1.0))

        max_lp_sum = self._initial_lp_sum_tensor()
        lp_total_target = torch.clamp(lp_total_target, min=max_lp_sum * 0.005, max=max_lp_sum * 1.05)
        cur_sum = torch.sum(torch.clamp(state["lp_cur"], min=1e-6), dim=1) + 1e-6
        state["lp_cur"] = state["lp_cur"] * (lp_total_target / cur_sum).unsqueeze(-1)

        if self.max_lp_vector is not None:
            max_vec = self.max_lp_vector.to(self.device).unsqueeze(0)
            state["lp_cur"] = torch.maximum(
                torch.minimum(state["lp_cur"], max_vec),
                torch.ones_like(state["lp_cur"]) * 1e-6,
            )
        else:
            state["lp_cur"] = torch.clamp(state["lp_cur"], min=1e-6)

        return state

    def _coupling_summary(self, state: dict) -> dict[str, float]:
        """Small diagnostic summary of exchanged coupling variables."""
        out: dict[str, float] = {}
        if self.gfpp_idx.numel() > 0:
            gfpp_pg = torch.sum(state["P_g"][:, self.gfpp_idx], dim=1)
            gas_req = self._gfpp_gas_demand(state["P_g"])
            out["mean_gfpp_power"] = float(torch.mean(gfpp_pg).detach().cpu().item())
            out["mean_gfpp_gas_demand"] = float(torch.mean(torch.sum(gas_req, dim=1)).detach().cpu().item())
        out["mean_linepack"] = float(torch.mean(torch.sum(state["lp_cur"], dim=1)).detach().cpu().item())
        return out

    @torch.no_grad()
    def evolve_to_equilibrium_batched(
        self,
        current_state: dict,
        prev_state: dict,
        attack_strategy: dict,
        gfpp_offline_mask: torch.Tensor,
    ):
        state = {k: v.clone().to(self.device) for k, v in current_state.items()}
        prev_state = {k: v.clone().to(self.device) for k, v in prev_state.items()}
        s_t = attack_strategy.get("s_t", torch.ones(self.batch_size, device=self.device))

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

        # FDIA-induced redispatch is the electric subsystem's initial response.
        state["P_g"] = self._apply_fdia_redispatch(state, attack_strategy, gfpp_offline_mask)
        state = self._apply_electric_projection_only(state, prev_state, gfpp_offline_mask)

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

        trace_effective_interval = float(getattr(self.env, "nd_trace_effective_interval", 10.0))
        check_interval = max(1, int(round(trace_effective_interval / max(local_dtau_scale, 1e-12))))
        min_check_step = max(self.ode_min_steps, check_interval)

        electric_keys = self._available_keys(state, self.electric_keys)
        gas_keys = self._available_keys(state, self.gas_keys)

        for step in range(1, n_steps + 1):
            # 1) Electric local ND update.
            state, _ = self._adam_update_block(
                state=state,
                attack_strategy=attack_strategy,
                gfpp_offline_mask=gfpp_offline_mask,
                s_t=s_t,
                keys=electric_keys,
                m=m,
                v=v,
                step=step,
                local_dtau_scale=local_dtau_scale,
                lr_map=lr_map,
                beta1=beta1,
                beta2=beta2,
                eps=eps,
            )
            state = self._apply_electric_projection_only(state, prev_state, gfpp_offline_mask)

            # Exchange P_g -> GFPP gas demand.  No explicit assignment is needed:
            # the gas residual uses ``state["P_g"]`` through ``_gfpp_gas_demand``.

            # 2) Gas local ND update.
            state, _ = self._adam_update_block(
                state=state,
                attack_strategy=attack_strategy,
                gfpp_offline_mask=gfpp_offline_mask,
                s_t=s_t,
                keys=gas_keys,
                m=m,
                v=v,
                step=step,
                local_dtau_scale=local_dtau_scale,
                lr_map=lr_map,
                beta1=beta1,
                beta2=beta2,
                eps=eps,
            )
            state = self._apply_gas_projection_only(state, prev_state, attack_strategy, s_t)

            # Exchange gas state -> next electric local update / next physical hour.
            # The linepack-based GFPP outage latch is still managed by the outer
            # simulation, preserving the original protection mechanism.

            if step >= min_check_step and step % check_interval == 0:
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

                projected_gradient_residual = self._projected_state_residual(state_history, state)
                electric_projected_residual = self._projected_state_residual_for_keys(state_history, state, electric_keys)
                gas_projected_residual = self._projected_state_residual_for_keys(state_history, state, gas_keys)

                projected_gradient_pass = projected_gradient_residual <= self.ode_rel_tolerance
                gradient_pass = gradient_norm <= grad_tolerance

                if energy_pass:
                    energy_stable_count += 1
                else:
                    energy_stable_count = 0

                coupling = self._coupling_summary(state)

                if record_enabled:
                    self.env.nd_convergence_trace.append(
                        {
                            "solver_type": "distributed_electric_gas_nd",
                            "physical_hour": int(physical_hour_int),
                            "raw_step": int(step),
                            "effective_step": float(step * local_dtau_scale),
                            "projected_gradient_residual": float(projected_gradient_residual),
                            "electric_projected_residual": float(electric_projected_residual),
                            "gas_projected_residual": float(gas_projected_residual),
                            "projected_gradient_tolerance": float(self.ode_rel_tolerance),
                            "projected_gradient_converged": bool(projected_gradient_pass),
                            "gradient_norm": float(gradient_norm),
                            "energy_value": float(energy_value),
                            "energy_rel_change": float(energy_rel_change) if energy_rel_change == energy_rel_change else float("nan"),
                            "energy_abs_change": float(energy_rel_change) if energy_rel_change == energy_rel_change else float("nan"),
                            "energy_abs_change_raw": float(energy_abs_change_raw) if energy_abs_change_raw == energy_abs_change_raw else float("nan"),
                            "gradient_tolerance": float(grad_tolerance),
                            "energy_tolerance": float(energy_tolerance),
                            "energy_stable_count": int(energy_stable_count),
                            "energy_patience": int(energy_patience),
                            "gradient_converged": bool(gradient_pass),
                            "energy_converged": bool(energy_stable_count >= energy_patience),
                            "mean_gfpp_power": coupling.get("mean_gfpp_power", 0.0),
                            "mean_gfpp_gas_demand": coupling.get("mean_gfpp_gas_demand", 0.0),
                            "mean_linepack": coupling.get("mean_linepack", 0.0),
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

                state_history = self._clone_tensor_state(state)

        if record_enabled and len(getattr(self.env, "nd_convergence_trace", [])) > 0:
            if self.env.nd_convergence_trace[-1].get("stop_reason", "") == "":
                self.env.nd_convergence_trace[-1]["stop_reason"] = stop_reason

        # Final full projection keeps the returned state exactly compatible with
        # the original solver's public output convention.
        state = self._apply_physical_bounds_batched(state, prev_state, attack_strategy, s_t, gfpp_offline_mask)
        state["lp_prev"] = state["lp_cur"].clone()
        state["S_prev"] = state["S"].clone()
        power_shedding = torch.sum(state["P_d_sh"], dim=1)
        return state, power_shedding

"""Physics-aware two-block consensus ADMM lower-level solver."""
from __future__ import annotations

import math
import time
from typing import Optional

import torch


class StandardADMMEvaluationSolver:
    """Physics-aware two-block consensus ADMM.

    Each coordination iteration solves the GFPP boundary subproblems and then
    recomputes the full electric and gas operating states.  The stopping test
    therefore includes both boundary residuals and the relative change of the
    complete 118-bus/135-node state.
    """

    solver_name = "physics_aware_consensus_admm"

    def __init__(
        self,
        iegs_env,
        batch_size: int,
        max_ode_steps: int = 2500,
        tolerance: float = 1e-4,
    ) -> None:
        self.env = iegs_env
        self.batch_size = int(batch_size)
        self.device = iegs_env.power.B_matrix.device
        self.max_steps = int(max_ode_steps)
        self.tol = float(tolerance)

        self.gfpp_idx = iegs_env.gfpp_indices_tensor()
        all_gen = torch.arange(iegs_env.power.num_gens, device=self.device)
        gfpp_mask = torch.zeros(iegs_env.power.num_gens, dtype=torch.bool, device=self.device)
        if self.gfpp_idx.numel() > 0:
            gfpp_mask[self.gfpp_idx] = True
        self.non_gfpp_idx = all_gen[~gfpp_mask]

        self.attack_target_buses = torch.as_tensor(
            list(getattr(iegs_env, "attack_target_buses", [])),
            dtype=torch.long,
            device=self.device,
        )

        # Fixed-rho standard ADMM parameters.  No adaptive-rho or inner solver.
        self.admm_rho = float(getattr(iegs_env, "admm_rho", 10.0))
        self.admm_outer_iters = min(
            self.max_steps,
            max(1, int(getattr(iegs_env, "admm_outer_iters", 200))),
        )
        self.admm_abs_tolerance = float(
            getattr(iegs_env, "admm_abs_tolerance", max(self.tol, 1e-5))
        )
        self.admm_rel_tolerance = float(
            getattr(iegs_env, "admm_rel_tolerance", 1e-3)
        )
        self.admm_consensus_rel_tolerance = float(
            getattr(iegs_env, "admm_consensus_rel_tolerance", self.admm_rel_tolerance)
        )
        self.admm_dual_rel_tolerance = float(
            getattr(iegs_env, "admm_dual_rel_tolerance", self.admm_rel_tolerance)
        )
        self.admm_min_outer_iters = max(
            1, int(getattr(iegs_env, "admm_min_outer_iters", 10))
        )
        self.admm_convergence_patience = max(
            1, int(getattr(iegs_env, "admm_convergence_patience", 5))
        )
        self.admm_state_rel_tolerance = float(
            getattr(iegs_env, "admm_state_rel_tolerance", 1e-3)
        )
        # Relaxed consensus update is a standard ADMM variant.  A value below
        # one prevents the low-dimensional boundary variables from appearing to
        # converge before the full electric/gas state has settled.
        self.admm_relaxation = min(1.0, max(0.05, float(
            getattr(iegs_env, "admm_relaxation", 0.65)
        )))

        self.admm_electric_weight = max(
            1e-9, float(getattr(iegs_env, "admm_electric_weight", 1.0))
        )
        self.admm_gas_weight = max(
            1e-9, float(getattr(iegs_env, "admm_gas_weight", 1.0))
        )

        # Linearized gas-local model parameters.  The available linepack release
        # is rate-limited, and the aggregate inventory is never allowed below the
        # same numerical floor used by the physical replay.
        self.admm_linepack_withdrawal_fraction = max(
            0.0,
            float(getattr(iegs_env, "admm_linepack_withdrawal_fraction", 0.12)),
        )
        self.admm_gas_security_floor_ratio = min(
            1.0,
            max(0.0, float(getattr(iegs_env, "admm_gas_security_floor_ratio", 0.005))),
        )
        self.admm_gas_budget_margin = float(
            getattr(iegs_env, "admm_gas_budget_margin", 0.0)
        )

        # Set by simulation.py before the first physical hour.
        self.initial_lp_sum: Optional[torch.Tensor] = None
        self.max_lp_vector: Optional[torch.Tensor] = None


    @staticmethod
    def _clone_tensor_state(state: dict) -> dict:
        return {
            key: value.detach().clone() if torch.is_tensor(value) else value
            for key, value in state.items()
        }

    @staticmethod
    def _full_state_relative_change(previous: dict, current: dict) -> float:
        keys = (
            "P_g", "delta", "delta_cyber", "P_d_sh",
            "S", "pi", "f_in", "f_out", "lp_cur",
            "P_q_sh", "comp_ratio",
        )
        worst = 0.0
        for key in keys:
            if key not in previous or key not in current:
                continue
            a, b = previous[key], current[key]
            if not (torch.is_tensor(a) and torch.is_tensor(b)) or a.shape != b.shape:
                continue
            scale = torch.clamp(torch.max(torch.abs(a)), min=1.0)
            rel = torch.max(torch.abs(b - a)) / scale
            worst = max(worst, float(rel.detach().cpu().item()))
        return worst

    def _gas_pressure_upper_bound(
        self, state: dict, nominal_upper: torch.Tensor
    ) -> torch.Tensor:
        """Pressure-aware GFPP deliverability bound.

        Above the warning pressure the nominal unit limit is unchanged.  Below
        the warning pressure the gas-side copy is reduced continuously, so the
        gas local response depends on the 135-node pressure state rather than
        only on an aggregate gas budget.
        """
        if self._num_gfpp() == 0 or "pi" not in state:
            return nominal_upper
        nodes = torch.as_tensor(
            [self.env.gfpp_to_gas_node[i] for i in self.env.gfpp_indices],
            dtype=torch.long, device=self.device,
        )
        pi = state["pi"][:, nodes]
        pi_warn = self.env.gas.pi_warn[nodes].unsqueeze(0)
        pi_min = self.env.gas.pi_min[nodes].unsqueeze(0)
        factor = torch.clamp(
            (pi - pi_min) / (pi_warn - pi_min + 1e-9), min=0.05, max=1.0
        )
        return nominal_upper * factor

    def _num_gfpp(self) -> int:
        return int(self.gfpp_idx.numel())

    def _zeros_gt(self) -> torch.Tensor:
        return torch.zeros(
            (self.batch_size, self._num_gfpp()),
            dtype=torch.float32,
            device=self.device,
        )

    def _initial_lp_sum_tensor(self) -> torch.Tensor:
        if self.initial_lp_sum is None:
            raise RuntimeError("simulation.py must set solver.initial_lp_sum before use")
        value = self.initial_lp_sum.to(self.device)
        if value.dim() == 0:
            return value
        return value.reshape(-1)[0]

    def _build_batched_pmax(self, offline: torch.Tensor) -> torch.Tensor:
        pmax = self.env.power.gen_pmax.unsqueeze(0).repeat(self.batch_size, 1).clone()
        if self._num_gfpp() > 0:
            pmax[:, self.gfpp_idx] = torch.where(
                offline.unsqueeze(-1),
                torch.zeros_like(pmax[:, self.gfpp_idx]),
                pmax[:, self.gfpp_idx],
            )
        return pmax

    def _build_batched_pmin(self, offline: torch.Tensor) -> torch.Tensor:
        pmin = self.env.power.gen_pmin.unsqueeze(0).repeat(self.batch_size, 1).clone()
        if self._num_gfpp() > 0:
            pmin[:, self.gfpp_idx] = torch.where(
                offline.unsqueeze(-1),
                torch.zeros_like(pmin[:, self.gfpp_idx]),
                pmin[:, self.gfpp_idx],
            )
        return pmin

    @staticmethod
    def _project_box_weighted_upper(
        center: torch.Tensor,
        lower: torch.Tensor,
        upper: torch.Tensor,
        weights: torch.Tensor,
        cap: torch.Tensor,
        iterations: int = 32,
    ) -> torch.Tensor:
        """Euclidean projection onto box and weighted upper half-space.

        Solves
            min_x 0.5 ||x-center||^2
            s.t.  lower <= x <= upper,
                  weights^T x <= cap.
        """
        x = torch.maximum(torch.minimum(center, upper), lower)
        if x.numel() == 0:
            return x

        if weights.dim() == 1:
            weights = weights.unsqueeze(0).expand_as(x)
        cap = cap.reshape(-1)
        active = torch.sum(weights * x, dim=1) > cap + 1e-8
        if not torch.any(active):
            return x

        lam_lo = torch.zeros(center.shape[0], dtype=center.dtype, device=center.device)
        lam_hi = torch.ones_like(lam_lo)

        # Expand the upper multiplier until every active row satisfies the cap.
        for _ in range(80):
            trial = torch.maximum(
                torch.minimum(center - lam_hi.unsqueeze(-1) * weights, upper),
                lower,
            )
            still_high = active & (torch.sum(weights * trial, dim=1) > cap)
            if not torch.any(still_high):
                break
            lam_hi = torch.where(still_high, lam_hi * 2.0, lam_hi)

        for _ in range(iterations):
            lam_mid = 0.5 * (lam_lo + lam_hi)
            trial = torch.maximum(
                torch.minimum(center - lam_mid.unsqueeze(-1) * weights, upper),
                lower,
            )
            too_high = torch.sum(weights * trial, dim=1) > cap
            lam_lo = torch.where(active & too_high, lam_mid, lam_lo)
            lam_hi = torch.where(active & (~too_high), lam_mid, lam_hi)

        projected = torch.maximum(
            torch.minimum(center - lam_hi.unsqueeze(-1) * weights, upper),
            lower,
        )
        return torch.where(active.unsqueeze(-1), projected, x)

    @staticmethod
    def _project_box_sum_equality(
        center: torch.Tensor,
        lower: torch.Tensor,
        upper: torch.Tensor,
        target_sum: torch.Tensor,
        iterations: int = 32,
    ) -> torch.Tensor:
        """Projection onto a box with a feasible row-wise sum equality."""
        if center.numel() == 0:
            return center

        lower_sum = torch.sum(lower, dim=1)
        upper_sum = torch.sum(upper, dim=1)
        target = torch.maximum(torch.minimum(target_sum.reshape(-1), upper_sum), lower_sum)

        # x(lambda)=clip(center-lambda, lower, upper) is monotone in lambda.
        lam_lo = torch.min(center - upper, dim=1).values - 1.0
        lam_hi = torch.max(center - lower, dim=1).values + 1.0

        for _ in range(iterations):
            lam_mid = 0.5 * (lam_lo + lam_hi)
            x_mid = torch.maximum(
                torch.minimum(center - lam_mid.unsqueeze(-1), upper),
                lower,
            )
            sum_mid = torch.sum(x_mid, dim=1)
            # Sum too large -> increase lambda.
            lam_lo = torch.where(sum_mid > target, lam_mid, lam_lo)
            lam_hi = torch.where(sum_mid > target, lam_hi, lam_mid)

        return torch.maximum(
            torch.minimum(center - lam_hi.unsqueeze(-1), upper),
            lower,
        )

    def _apply_fdia_redispatch(
        self,
        pg: torch.Tensor,
        attack_strategy: dict,
        offline: torch.Tensor,
    ) -> torch.Tensor:
        """Build the electric operator's attacked dispatch reference."""
        out = pg.clone()
        z_a = attack_strategy.get("FDIA")
        if z_a is None or self._num_gfpp() == 0:
            return out

        z_a = z_a.to(self.device)
        positive = torch.relu(z_a)
        if self.attack_target_buses.numel() > 0:
            drive = torch.sum(positive[:, self.attack_target_buses], dim=1)
        else:
            drive = torch.sum(positive, dim=1)

        requested = drive * float(getattr(self.env, "dispatch_attack_gain", 1.0))
        if torch.max(requested).item() <= 1e-12:
            return out

        pmax = self._build_batched_pmax(offline)
        pmin = self._build_batched_pmin(offline)
        headroom = torch.clamp(pmax[:, self.gfpp_idx] - out[:, self.gfpp_idx], min=0.0)
        total_headroom = torch.sum(headroom, dim=1)
        actual = torch.minimum(requested, total_headroom)
        share = headroom / (total_headroom.unsqueeze(-1) + 1e-12)
        out[:, self.gfpp_idx] += share * actual.unsqueeze(-1)

        if self.non_gfpp_idx.numel() > 0:
            reducible = torch.clamp(
                out[:, self.non_gfpp_idx] - pmin[:, self.non_gfpp_idx],
                min=0.0,
            )
            total_reducible = torch.sum(reducible, dim=1)
            reduction = torch.minimum(actual, total_reducible)
            out[:, self.non_gfpp_idx] -= (
                reducible / (total_reducible.unsqueeze(-1) + 1e-12)
            ) * reduction.unsqueeze(-1)
        return out

    def _gfpp_bounds(
        self,
        prev_state: dict,
        offline: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._num_gfpp() == 0:
            z = self._zeros_gt()
            return z, z
        pmax = self._build_batched_pmax(offline)[:, self.gfpp_idx]
        pmin = self._build_batched_pmin(offline)[:, self.gfpp_idx]
        ramp = self.env.power.gen_ramp_max[self.gfpp_idx].unsqueeze(0)
        prev = prev_state["P_g"][:, self.gfpp_idx]
        lower = torch.maximum(pmin, prev - ramp)
        upper = torch.minimum(pmax, prev + ramp)
        lower = torch.minimum(lower, upper)
        return lower, upper

    def _electric_gfpp_budget(
        self,
        lower_all: torch.Tensor,
    ) -> torch.Tensor:
        total_load = torch.sum(self.env.power.base_load).expand(self.batch_size)
        if self.non_gfpp_idx.numel() == 0:
            non_min = torch.zeros_like(total_load)
        else:
            non_min = torch.sum(lower_all[:, self.non_gfpp_idx], dim=1)
        return torch.clamp(total_load - non_min, min=0.0)

    def _gas_delivery_budget(
        self,
        state: dict,
        prev_state: dict,
    ) -> torch.Tensor:
        """Linearized aggregate gas available for GFPP consumption."""
        source = torch.sum(state["S"], dim=1)
        base_demand = torch.sum(self.env.gas.base_gas_load).expand(self.batch_size)

        ref_lp = self._initial_lp_sum_tensor()
        prev_lp = torch.sum(prev_state["lp_cur"], dim=1)
        floor = self.admm_gas_security_floor_ratio * ref_lp
        physically_available_lp = torch.clamp(prev_lp - floor, min=0.0)
        rate_available_lp = self.admm_linepack_withdrawal_fraction * ref_lp
        release_lp = torch.minimum(
            physically_available_lp,
            torch.ones_like(physically_available_lp) * rate_available_lp,
        )

        scale = max(float(getattr(self.env, "linepack_balance_scale", 1.0)), 1e-9)
        dt = max(float(self.env.gas.dt), 1e-9)
        release_flow = release_lp / (scale * dt)

        budget = source + release_flow - base_demand + self.admm_gas_budget_margin
        return torch.clamp(budget, min=0.0)

    def _residual_record(
        self,
        p_e: torch.Tensor,
        p_g: torch.Tensor,
        z: torch.Tensor,
        z_old: torch.Tensor,
        u_e: torch.Tensor,
        u_g: torch.Tensor,
    ) -> dict[str, float]:
        if p_e.numel() == 0:
            return {
                "admm_primal_residual": 0.0,
                "admm_dual_residual": 0.0,
                "admm_gt_consensus_residual": 0.0,
                "admm_relative_primal_residual": 0.0,
                "admm_relative_dual_residual": 0.0,
                "admm_relative_consensus_residual": 0.0,
                "admm_primal_threshold": self.admm_abs_tolerance,
                "admm_dual_threshold": self.admm_abs_tolerance,
                "admm_consensus_rel_tolerance": self.admm_consensus_rel_tolerance,
                "admm_dual_rel_tolerance": self.admm_dual_rel_tolerance,
            }

        r_e = p_e - z
        r_g = p_g - z
        primal = torch.sqrt(torch.sum(r_e**2) + torch.sum(r_g**2))
        consensus = torch.max(torch.abs(p_e - p_g))
        dual = self.admm_rho * math.sqrt(2.0) * torch.linalg.vector_norm(z - z_old)

        n_local = max(1, p_e.numel())
        x_norm = torch.maximum(torch.linalg.vector_norm(p_e), torch.linalg.vector_norm(p_g))
        z_norm = math.sqrt(2.0) * torch.linalg.vector_norm(z)
        eps_primal = math.sqrt(2.0 * n_local) * self.admm_abs_tolerance + self.admm_consensus_rel_tolerance * torch.maximum(x_norm, z_norm)

        # Standard dual threshold uses the concatenated scaled multipliers;
        # u_e and u_g are opposite under consensus and must not be summed,
        # otherwise their cancellation would make the threshold spuriously tiny.
        y_norm = self.admm_rho * torch.sqrt(
            torch.sum(u_e**2) + torch.sum(u_g**2)
        )
        eps_dual = math.sqrt(float(n_local)) * self.admm_abs_tolerance + self.admm_dual_rel_tolerance * y_norm

        one = torch.tensor(1.0, dtype=p_e.dtype, device=self.device)
        boundary_scale = torch.maximum(
            one,
            torch.maximum(
                torch.max(torch.abs(z)),
                torch.maximum(torch.max(torch.abs(p_e)), torch.max(torch.abs(p_g))),
            ),
        )
        dual_scale = torch.maximum(one, torch.max(torch.abs(z_old)))

        rel_primal = torch.max(
            torch.max(torch.abs(r_e)), torch.max(torch.abs(r_g))
        ) / boundary_scale
        rel_consensus = consensus / boundary_scale
        rel_dual = torch.max(torch.abs(z - z_old)) / dual_scale

        return {
            "admm_primal_residual": float(primal.detach().cpu().item()),
            "admm_dual_residual": float(dual.detach().cpu().item()),
            "admm_gt_consensus_residual": float(consensus.detach().cpu().item()),
            "admm_relative_primal_residual": float(rel_primal.detach().cpu().item()),
            "admm_relative_dual_residual": float(rel_dual.detach().cpu().item()),
            "admm_relative_consensus_residual": float(rel_consensus.detach().cpu().item()),
            "admm_primal_threshold": float(eps_primal.detach().cpu().item()),
            "admm_dual_threshold": float(eps_dual.detach().cpu().item()),
            "admm_consensus_rel_tolerance": float(self.admm_consensus_rel_tolerance),
            "admm_dual_rel_tolerance": float(self.admm_dual_rel_tolerance),
        }

    def _run_standard_admm(
        self,
        state: dict,
        prev_state: dict,
        attack_strategy: dict,
        offline: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, int]:
        if self._num_gfpp() == 0:
            z = self._zeros_gt()
            return z, z, z, z, "admm_no_coupling_variables", 0

        pmax_all = self._build_batched_pmax(offline)
        pmin_all = self._build_batched_pmin(offline)
        ramp_all = self.env.power.gen_ramp_max.unsqueeze(0)
        prev_pg = prev_state["P_g"]
        lower_all = torch.maximum(pmin_all, prev_pg - ramp_all)
        upper_all = torch.minimum(pmax_all, prev_pg + ramp_all)
        lower_all = torch.minimum(lower_all, upper_all)

        attacked_pg = self._apply_fdia_redispatch(state["P_g"], attack_strategy, offline)
        p_ref_e = torch.maximum(
            torch.minimum(attacked_pg[:, self.gfpp_idx], upper_all[:, self.gfpp_idx]),
            lower_all[:, self.gfpp_idx],
        )
        # Independent gas-operator reference.  The gas subsystem must not
        # copy the attacked electric-side target; otherwise p_e and p_g can be
        # identical from the first ADMM iteration and the common GFPP residual
        # becomes identically zero even though the ADMM dual residual is still
        # evolving.  Use the previously physically supplied GFPP output, which
        # is also the convention used by the ALADIN implementation.
        p_ref_g = torch.maximum(
            torch.minimum(prev_pg[:, self.gfpp_idx], upper_all[:, self.gfpp_idx]),
            lower_all[:, self.gfpp_idx],
        )

        lower_e = lower_all[:, self.gfpp_idx]
        upper_e = upper_all[:, self.gfpp_idx]
        # The gas-side copy represents the same physical GFPP units and
        # therefore uses the same unit/ramp lower and upper limits.
        lower_g = lower_e.clone()
        upper_g = upper_e.clone()

        electric_cap = self._electric_gfpp_budget(lower_all)
        gas_cap = self._gas_delivery_budget(state, prev_state)
        psi = self.env.psi_g[self.gfpp_idx].to(self.device)
        ones = torch.ones_like(psi)

        reference_scale = torch.maximum(
            torch.ones(p_ref_e.shape[0], dtype=p_ref_e.dtype, device=self.device),
            torch.maximum(
                torch.max(torch.abs(p_ref_e), dim=1).values,
                torch.max(torch.abs(p_ref_g), dim=1).values,
            ),
        )
        reference_gap = torch.max(torch.abs(p_ref_e - p_ref_g), dim=1).values
        relative_reference_gap = reference_gap / reference_scale

        # Start each physical-hour subproblem with zero scaled multipliers.
        # This is standard ADMM initialization; z is warm-started from the
        # previous feasible boundary value when available.
        if "z_gt" in prev_state and torch.is_tensor(prev_state["z_gt"]):
            z = prev_state["z_gt"].clone().to(self.device)
            if z.shape != p_ref_e.shape:
                z = 0.5 * (p_ref_e + p_ref_g)
        else:
            z = 0.5 * (p_ref_e + p_ref_g)
        z = torch.maximum(torch.minimum(z, upper_e), lower_e)
        u_e = torch.zeros_like(z)
        u_g = torch.zeros_like(z)

        record_enabled = bool(getattr(self.env, "record_nd_convergence", False))
        physical_hour = int(attack_strategy.get("physical_hour", -1))
        record_hour = getattr(self.env, "record_nd_hour", None)
        if record_hour is not None:
            record_enabled = record_enabled and physical_hour == int(record_hour)
        if record_enabled and not hasattr(self.env, "nd_convergence_trace"):
            self.env.nd_convergence_trace = []

        stable_count = 0
        stop_reason = "admm_max_iterations_reached"
        p_e = p_ref_e.clone()
        p_g = p_ref_g.clone()
        last_iter = 0
        previous_full_state = self._clone_tensor_state(state)
        start_time = time.perf_counter()

        for k in range(1, self.admm_outer_iters + 1):
            last_iter = k
            # Electric local QP followed by a complete 118-bus dispatch replay.
            center_e = (
                self.admm_electric_weight * p_ref_e
                + self.admm_rho * (z - u_e)
            ) / (self.admm_electric_weight + self.admm_rho)
            p_e = self._project_box_weighted_upper(
                center_e, lower_e, upper_e, ones, electric_cap
            )
            state = self._dispatch_electric_state(
                state, prev_state, attack_strategy, offline, p_e
            )

            # Gas local QP uses both aggregate gas inventory and the current
            # pressure at the twelve GFPP connection nodes.  The complete
            # 135-node gas state is replayed at every outer iteration.
            gas_cap = self._gas_delivery_budget(state, prev_state)
            dynamic_upper_g = torch.minimum(
                upper_g, self._gas_pressure_upper_bound(state, upper_g)
            )
            dynamic_lower_g = torch.minimum(lower_g, dynamic_upper_g)
            center_g = (
                self.admm_gas_weight * p_ref_g
                + self.admm_rho * (z - u_g)
            ) / (self.admm_gas_weight + self.admm_rho)
            p_g = self._project_box_weighted_upper(
                center_g, dynamic_lower_g, dynamic_upper_g, psi, gas_cap
            )
            state = self._dispatch_gas_state(
                state, prev_state, attack_strategy, p_g
            )

            # Relaxed scaled-form consensus and dual updates.  The relaxation
            # is not a plotting device; it is a standard stabilized ADMM step.
            z_old = z.clone()
            z_candidate = 0.5 * (p_e + u_e + p_g + u_g)
            z = (1.0 - self.admm_relaxation) * z_old + self.admm_relaxation * z_candidate
            z = torch.maximum(torch.minimum(z, upper_e), lower_e)
            u_e = u_e + p_e - z
            u_g = u_g + p_g - z

            residuals = self._residual_record(p_e, p_g, z, z_old, u_e, u_g)
            full_state_rel = self._full_state_relative_change(previous_full_state, state)
            previous_full_state = self._clone_tensor_state(state)

            primal_ok = (
                residuals["admm_relative_primal_residual"]
                <= residuals["admm_consensus_rel_tolerance"]
            )
            dual_ok = (
                residuals["admm_relative_dual_residual"]
                <= residuals["admm_dual_rel_tolerance"]
            )
            consensus_ok = (
                residuals["admm_relative_consensus_residual"]
                <= residuals["admm_consensus_rel_tolerance"]
            )
            state_ok = full_state_rel <= self.admm_state_rel_tolerance
            enough = k >= self.admm_min_outer_iters
            all_ok = bool(enough and primal_ok and dual_ok and consensus_ok and state_ok)
            stable_count = stable_count + 1 if all_ok else 0

            if record_enabled:
                self.env.nd_convergence_trace.append(
                    {
                        "solver_type": self.solver_name,
                        "model_scope": "physics_aware_full_state_replay",
                        "physical_hour": physical_hour,
                        "admm_iteration": k,
                        "elapsed_time_s": float(time.perf_counter() - start_time),
                        "admm_rho": float(self.admm_rho),
                        "admm_relaxation": float(self.admm_relaxation),
                        "admm_electric_weight": float(self.admm_electric_weight),
                        "admm_gas_weight": float(self.admm_gas_weight),
                        "admm_gas_budget": float(torch.mean(gas_cap).detach().cpu().item()),
                        "admm_electric_budget": float(torch.mean(electric_cap).detach().cpu().item()),
                        "admm_reference_gap": float(torch.max(reference_gap).detach().cpu().item()),
                        "admm_relative_reference_gap": float(torch.max(relative_reference_gap).detach().cpu().item()),
                        "admm_full_state_relative_residual": float(full_state_rel),
                        "admm_state_rel_tolerance": float(self.admm_state_rel_tolerance),
                        "admm_convergence_stable_count": int(stable_count),
                        "admm_convergence_patience": int(self.admm_convergence_patience),
                        "admm_primal_pass": bool(primal_ok),
                        "admm_dual_pass": bool(dual_ok),
                        "admm_consensus_pass": bool(consensus_ok),
                        "admm_state_pass": bool(state_ok),
                        "admm_all_residuals_pass": bool(all_ok),
                        **residuals,
                        "stop_reason": "",
                    }
                )

            if stable_count >= self.admm_convergence_patience:
                stop_reason = (
                    f"admm_full_state_residuals_stable_{self.admm_convergence_patience}_iterations"
                )
                if record_enabled:
                    self.env.nd_convergence_trace[-1]["stop_reason"] = stop_reason
                break

        if record_enabled and self.env.nd_convergence_trace:
            if not self.env.nd_convergence_trace[-1].get("stop_reason"):
                self.env.nd_convergence_trace[-1]["stop_reason"] = stop_reason

        # Return a single physically feasible shared boundary point.  These are
        # upper constraints, so sequential projection by reduction preserves all
        # previously satisfied delivery limits.
        coupled = torch.maximum(torch.minimum(z, upper_e), lower_e)
        coupled = self._project_box_weighted_upper(
            coupled, lower_e, upper_e, ones, electric_cap
        )
        coupled = self._project_box_weighted_upper(
            coupled, lower_g, upper_g, psi, gas_cap
        )
        return coupled, u_e, u_g, p_g, stop_reason, last_iter

    def _dispatch_electric_state(
        self,
        state: dict,
        prev_state: dict,
        attack_strategy: dict,
        offline: torch.Tensor,
        coupled_gt: torch.Tensor,
    ) -> dict:
        env_p = self.env.power
        pmax = self._build_batched_pmax(offline)
        pmin = self._build_batched_pmin(offline)
        ramp = env_p.gen_ramp_max.unsqueeze(0)
        lower = torch.maximum(pmin, prev_state["P_g"] - ramp)
        upper = torch.minimum(pmax, prev_state["P_g"] + ramp)
        lower = torch.minimum(lower, upper)

        pg_ref = self._apply_fdia_redispatch(state["P_g"], attack_strategy, offline)
        pg = torch.maximum(torch.minimum(pg_ref, upper), lower)
        if self._num_gfpp() > 0:
            pg[:, self.gfpp_idx] = coupled_gt

        total_load = torch.sum(env_p.base_load).expand(self.batch_size)
        gfpp_total = torch.sum(coupled_gt, dim=1) if self._num_gfpp() else torch.zeros_like(total_load)

        if self.non_gfpp_idx.numel() > 0:
            non_ref = pg_ref[:, self.non_gfpp_idx]
            non_lower = lower[:, self.non_gfpp_idx]
            non_upper = upper[:, self.non_gfpp_idx]
            required_non = total_load - gfpp_total
            feasible_non = torch.maximum(
                torch.minimum(required_non, torch.sum(non_upper, dim=1)),
                torch.sum(non_lower, dim=1),
            )
            pg[:, self.non_gfpp_idx] = self._project_box_sum_equality(
                non_ref, non_lower, non_upper, feasible_non
            )

        actual_total = torch.sum(pg, dim=1)
        shortage = torch.clamp(total_load - actual_total, min=0.0)
        shed_ratio = torch.clamp(shortage / (total_load + 1e-12), min=0.0, max=1.0)
        state["P_d_sh"] = env_p.base_load.unsqueeze(0) * shed_ratio.unsqueeze(-1)
        state["P_g"] = pg

        gen_injection = pg @ env_p.gen_to_bus_matrix.T
        physical_rhs = gen_injection - (env_p.base_load.unsqueeze(0) - state["P_d_sh"])
        theta = physical_rhs @ env_p.B_pinv.T
        theta = theta - theta[:, :1]
        state["delta"] = theta

        z_a = attack_strategy.get("FDIA")
        if z_a is None:
            z_a = torch.zeros_like(state["P_d_sh"])
        cyber_rhs = gen_injection - (
            env_p.base_load.unsqueeze(0) + z_a - state["P_d_sh"]
        )
        theta_cyber = cyber_rhs @ env_p.B_pinv.T
        theta_cyber = theta_cyber - theta_cyber[:, :1]
        state["delta_cyber"] = theta_cyber
        return state

    def _dispatch_gas_state(
        self,
        state: dict,
        prev_state: dict,
        attack_strategy: dict,
        coupled_gt: torch.Tensor,
    ) -> dict:
        env_g = self.env.gas
        s_t = attack_strategy.get(
            "s_t", torch.ones(self.batch_size, dtype=torch.float32, device=self.device)
        )

        bounded_s = torch.maximum(
            torch.minimum(state["S"], env_g.S_max.unsqueeze(0)),
            env_g.S_min.unsqueeze(0),
        )
        locked_s = attack_strategy.get("locked_S")
        if locked_s is not None:
            bounded_locked = torch.maximum(
                torch.minimum(locked_s.to(self.device), env_g.S_max.unsqueeze(0)),
                env_g.S_min.unsqueeze(0),
            )
            state["S"] = torch.where(
                s_t.unsqueeze(-1) > 0.5, bounded_s, bounded_locked
            )
        else:
            state["S"] = bounded_s

        if "comp_ratio" in state:
            if env_g.num_comps > 0:
                bounded_comp = torch.maximum(
                    torch.minimum(
                        state["comp_ratio"], env_g.comp_ratio_max.unsqueeze(0)
                    ),
                    env_g.comp_ratio_min.unsqueeze(0),
                )
            else:
                bounded_comp = torch.clamp(state["comp_ratio"], 1.0, 1.5)
            locked_comp = attack_strategy.get("locked_comp")
            if locked_comp is not None:
                state["comp_ratio"] = torch.where(
                    s_t.unsqueeze(-1) > 0.5,
                    bounded_comp,
                    locked_comp.to(self.device),
                )
            else:
                state["comp_ratio"] = bounded_comp

        state["P_q_sh"] = torch.maximum(
            torch.minimum(
                state["P_q_sh"], env_g.base_gas_load.unsqueeze(0) * 0.05
            ),
            torch.zeros_like(state["P_q_sh"]),
        )

        gas_req = torch.zeros(
            (self.batch_size, env_g.num_nodes),
            dtype=torch.float32,
            device=self.device,
        )
        for local_j, gen_idx in enumerate(self.env.gfpp_indices):
            gas_node = self.env.gfpp_to_gas_node[gen_idx]
            gas_req[:, gas_node] += coupled_gt[:, local_j] * self.env.psi_g[gen_idx]

        total_demand = torch.sum(
            env_g.base_gas_load.unsqueeze(0) + gas_req - state["P_q_sh"], dim=1
        )
        total_source = torch.sum(state["S"], dim=1)
        scale = float(getattr(self.env, "linepack_balance_scale", 1.0))
        lp_target = torch.sum(prev_state["lp_cur"], dim=1) + (
            total_source - total_demand
        ) * env_g.dt * scale

        ref_lp = self._initial_lp_sum_tensor()
        lp_target = torch.clamp(lp_target, min=ref_lp * 0.005, max=ref_lp * 1.05)
        prev_lp = torch.clamp(prev_state["lp_cur"], min=1e-9)
        prev_sum = torch.sum(prev_lp, dim=1)
        lp_cur = prev_lp * (lp_target / (prev_sum + 1e-12)).unsqueeze(-1)

        if self.max_lp_vector is not None:
            max_vec = self.max_lp_vector.to(self.device).unsqueeze(0)
            lp_cur = torch.maximum(
                torch.minimum(lp_cur, max_vec), torch.ones_like(lp_cur) * 1e-9
            )
        else:
            lp_cur = torch.clamp(lp_cur, min=1e-9)
        state["lp_cur"] = lp_cur

        # Enforce the dynamic inventory identity with a symmetric flow update.
        flow_avg = 0.5 * (state["f_in"] + state["f_out"])
        delta_lp = (lp_cur - prev_state["lp_cur"]) / max(float(env_g.dt), 1e-9)
        state["f_in"] = torch.clamp(flow_avg + 0.5 * delta_lp, min=0.0, max=50000.0)
        state["f_out"] = torch.clamp(flow_avg - 0.5 * delta_lp, min=0.0, max=50000.0)

        # Linearized pressure-linepack replay.  It is diagnostic only in the
        # current simulation, but keeps the returned gas state physically scaled.
        lp_ratio = lp_target / (prev_sum + 1e-12)
        pi = prev_state["pi"] * lp_ratio.unsqueeze(-1)
        state["pi"] = torch.maximum(
            torch.minimum(pi, env_g.pi_max.unsqueeze(0)),
            env_g.pi_min.unsqueeze(0),
        )

        state["P_gt_g"] = coupled_gt.clone()
        state["lp_prev"] = state["lp_cur"].clone()
        state["S_prev"] = state["S"].clone()
        return state

    @torch.no_grad()
    def evolve_to_equilibrium_batched(
        self,
        current_state: dict,
        prev_state: dict,
        attack_strategy: dict,
        gfpp_offline_mask: torch.Tensor,
    ) -> tuple[dict, torch.Tensor]:
        state = {
            key: value.clone().to(self.device) if torch.is_tensor(value) else value
            for key, value in current_state.items()
        }
        prev = {
            key: value.clone().to(self.device) if torch.is_tensor(value) else value
            for key, value in prev_state.items()
        }
        offline = gfpp_offline_mask.to(self.device).bool()

        coupled, u_e, u_g, p_g_local, stop_reason, iterations = self._run_standard_admm(
            state, prev, attack_strategy, offline
        )

        state = self._dispatch_electric_state(
            state, prev, attack_strategy, offline, coupled
        )
        state = self._dispatch_gas_state(
            state, prev, attack_strategy, coupled
        )

        state["z_gt"] = coupled.clone()
        state["u_e_gt"] = u_e.clone()
        state["u_g_gt"] = u_g.clone()
        state["P_gt_g_local"] = p_g_local.clone()
        state["admm_iterations"] = torch.full(
            (self.batch_size, 1),
            float(iterations),
            dtype=torch.float32,
            device=self.device,
        )
        state["admm_stop_code"] = torch.full(
            (self.batch_size, 1),
            1.0 if (stop_reason.startswith("admm_residuals_stable_") or stop_reason.startswith("admm_full_state_residuals_stable_")) else 0.0,
            dtype=torch.float32,
            device=self.device,
        )

        power_shedding = torch.sum(state["P_d_sh"], dim=1)
        return state, power_shedding


# Backward-compatible name used by the existing simulation wrapper.
ADMMDistributedEvaluationSolver = StandardADMMEvaluationSolver

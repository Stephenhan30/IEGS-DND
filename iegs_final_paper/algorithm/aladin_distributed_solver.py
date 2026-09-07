"""Physics-aware distributed ALADIN lower-level solver."""
from __future__ import annotations

import math
import time
from typing import Optional

import torch


class ALADINEvaluationSolver:
    """Physics-aware two-region ALADIN evaluator.

    Local boundary solves are followed by full electric and gas state replays at
    every coordination iteration.  Convergence therefore requires consensus,
    step, KKT, and complete-state stabilization.
    """

    solver_name = "physics_aware_distributed_aladin"

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

        # Standard ALADIN parameters.
        self.aladin_rho = max(1e-9, float(getattr(iegs_env, "aladin_rho", 5.0)))
        self.aladin_mu = max(1e-9, float(getattr(iegs_env, "aladin_mu", 20.0)))
        self.aladin_outer_iters = min(
            self.max_steps,
            max(1, int(getattr(iegs_env, "aladin_outer_iters", 40))),
        )
        self.aladin_alpha = min(
            1.0, max(1e-3, float(getattr(iegs_env, "aladin_alpha", 0.45)))
        )
        self.aladin_hessian_regularization = max(
            1e-10, float(getattr(iegs_env, "aladin_hessian_regularization", 1e-6))
        )
        self.aladin_active_tolerance = max(
            1e-10, float(getattr(iegs_env, "aladin_active_tolerance", 1e-6))
        )
        self.aladin_trust_region_fraction = max(
            1e-3, float(getattr(iegs_env, "aladin_trust_region_fraction", 0.50))
        )
        self.aladin_local_bisection_iters = max(
            16, int(getattr(iegs_env, "aladin_local_bisection_iters", 48))
        )

        self.aladin_consensus_rel_tolerance = float(
            getattr(iegs_env, "aladin_consensus_rel_tolerance", 1e-3)
        )
        self.aladin_step_rel_tolerance = float(
            getattr(iegs_env, "aladin_step_rel_tolerance", 1e-3)
        )
        self.aladin_kkt_rel_tolerance = float(
            getattr(iegs_env, "aladin_kkt_rel_tolerance", 1e-3)
        )
        self.aladin_min_outer_iters = max(
            1, int(getattr(iegs_env, "aladin_min_outer_iters", 10))
        )
        self.aladin_convergence_patience = max(
            1, int(getattr(iegs_env, "aladin_convergence_patience", 5))
        )
        self.aladin_state_rel_tolerance = float(
            getattr(iegs_env, "aladin_state_rel_tolerance", 1e-3)
        )

        self.aladin_electric_weight = max(
            1e-9, float(getattr(iegs_env, "aladin_electric_weight", 1.0))
        )
        self.aladin_gas_weight = max(
            1e-9, float(getattr(iegs_env, "aladin_gas_weight", 1.0))
        )
        # A small positive value makes the gas local model a genuine nonlinear
        # program while preserving convexity and inexpensive exact solution.
        self.aladin_gas_loss_coefficient = max(
            0.0, float(getattr(iegs_env, "aladin_gas_loss_coefficient", 1e-4))
        )

        self.aladin_linepack_withdrawal_fraction = max(
            0.0,
            float(getattr(iegs_env, "aladin_linepack_withdrawal_fraction", 0.12)),
        )
        self.aladin_gas_security_floor_ratio = min(
            1.0,
            max(0.0, float(getattr(iegs_env, "aladin_gas_security_floor_ratio", 0.005))),
        )
        self.aladin_gas_budget_margin = float(
            getattr(iegs_env, "aladin_gas_budget_margin", 0.0)
        )

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
        floor = self.aladin_gas_security_floor_ratio * ref_lp
        physically_available_lp = torch.clamp(prev_lp - floor, min=0.0)
        rate_available_lp = self.aladin_linepack_withdrawal_fraction * ref_lp
        release_lp = torch.minimum(
            physically_available_lp,
            torch.ones_like(physically_available_lp) * rate_available_lp,
        )

        scale = max(float(getattr(self.env, "linepack_balance_scale", 1.0)), 1e-9)
        dt = max(float(self.env.gas.dt), 1e-9)
        release_flow = release_lp / (scale * dt)

        budget = source + release_flow - base_demand + self.aladin_gas_budget_margin
        return torch.clamp(budget, min=0.0)


    @staticmethod
    def _project_box_sum_upper_with_dual(
        center: torch.Tensor,
        lower: torch.Tensor,
        upper: torch.Tensor,
        cap: torch.Tensor,
        quadratic_weight: float,
        iterations: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Solve a box-constrained quadratic with one sum upper constraint."""
        a = max(float(quadratic_weight), 1e-12)
        cap_eff = torch.maximum(cap.reshape(-1), torch.sum(lower, dim=1))
        x0 = torch.maximum(torch.minimum(center, upper), lower)
        active = torch.sum(x0, dim=1) > cap_eff + 1e-9
        if not torch.any(active):
            return x0, torch.zeros(center.shape[0], dtype=center.dtype, device=center.device)

        lo = torch.zeros(center.shape[0], dtype=center.dtype, device=center.device)
        hi = torch.ones_like(lo)
        for _ in range(80):
            trial = torch.maximum(
                torch.minimum(center - hi.unsqueeze(-1) / a, upper), lower
            )
            too_high = active & (torch.sum(trial, dim=1) > cap_eff)
            if not torch.any(too_high):
                break
            hi = torch.where(too_high, hi * 2.0, hi)

        for _ in range(iterations):
            mid = 0.5 * (lo + hi)
            trial = torch.maximum(
                torch.minimum(center - mid.unsqueeze(-1) / a, upper), lower
            )
            too_high = torch.sum(trial, dim=1) > cap_eff
            lo = torch.where(active & too_high, mid, lo)
            hi = torch.where(active & (~too_high), mid, hi)

        x = torch.maximum(
            torch.minimum(center - hi.unsqueeze(-1) / a, upper), lower
        )
        return torch.where(active.unsqueeze(-1), x, x0), torch.where(active, hi, torch.zeros_like(hi))

    def _gas_constraint_value(self, x: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
        gas = x * psi.unsqueeze(0)
        return torch.sum(gas, dim=1) + 0.5 * self.aladin_gas_loss_coefficient * torch.sum(gas * gas, dim=1)

    def _gas_constraint_gradient(self, x: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
        return psi.unsqueeze(0) + self.aladin_gas_loss_coefficient * psi.pow(2).unsqueeze(0) * x

    def _project_box_nonlinear_gas_with_dual(
        self,
        center: torch.Tensor,
        lower: torch.Tensor,
        upper: torch.Tensor,
        psi: torch.Tensor,
        cap: torch.Tensor,
        quadratic_weight: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Exact KKT solve of the convex nonlinear gas local subproblem."""
        a = max(float(quadratic_weight), 1e-12)
        cap_eff = torch.maximum(cap.reshape(-1), self._gas_constraint_value(lower, psi))
        x0 = torch.maximum(torch.minimum(center, upper), lower)
        active = self._gas_constraint_value(x0, psi) > cap_eff + 1e-9
        if not torch.any(active):
            return x0, torch.zeros(center.shape[0], dtype=center.dtype, device=center.device)

        psi_b = psi.unsqueeze(0)
        psi2_b = psi.pow(2).unsqueeze(0)
        kappa = self.aladin_gas_loss_coefficient

        def primal(nu: torch.Tensor) -> torch.Tensor:
            denom = a + nu.unsqueeze(-1) * kappa * psi2_b
            numer = a * center - nu.unsqueeze(-1) * psi_b
            raw = numer / torch.clamp(denom, min=1e-12)
            return torch.maximum(torch.minimum(raw, upper), lower)

        lo = torch.zeros(center.shape[0], dtype=center.dtype, device=center.device)
        hi = torch.ones_like(lo)
        for _ in range(100):
            trial = primal(hi)
            too_high = active & (self._gas_constraint_value(trial, psi) > cap_eff)
            if not torch.any(too_high):
                break
            hi = torch.where(too_high, hi * 2.0, hi)

        for _ in range(self.aladin_local_bisection_iters):
            mid = 0.5 * (lo + hi)
            trial = primal(mid)
            too_high = self._gas_constraint_value(trial, psi) > cap_eff
            lo = torch.where(active & too_high, mid, lo)
            hi = torch.where(active & (~too_high), mid, hi)

        x = primal(hi)
        return torch.where(active.unsqueeze(-1), x, x0), torch.where(active, hi, torch.zeros_like(hi))

    @staticmethod
    def _apply_block_hessian_inverse(
        v_e: torch.Tensor,
        v_g: torch.Tensor,
        free_e: torch.Tensor,
        free_g: torch.Tensor,
        a_e: torch.Tensor,
        a_g: torch.Tensor,
        mu: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the inverse of the free-variable 2x2 block Hessian."""
        both = free_e & free_g
        only_e = free_e & (~free_g)
        only_g = free_g & (~free_e)

        y_e = torch.zeros_like(v_e)
        y_g = torch.zeros_like(v_g)
        det = torch.clamp(a_e * a_g - float(mu) ** 2, min=1e-12)
        y_e = torch.where(both, (a_g * v_e + float(mu) * v_g) / det, y_e)
        y_g = torch.where(both, (float(mu) * v_e + a_e * v_g) / det, y_g)
        y_e = torch.where(only_e, v_e / torch.clamp(a_e, min=1e-12), y_e)
        y_g = torch.where(only_g, v_g / torch.clamp(a_g, min=1e-12), y_g)
        return y_e, y_g

    def _solve_coordination_qp(
        self,
        x_e: torch.Tensor,
        x_g: torch.Tensor,
        grad_e: torch.Tensor,
        grad_g: torch.Tensor,
        hess_e: torch.Tensor,
        hess_g: torch.Tensor,
        multiplier: torch.Tensor,
        lower_e: torch.Tensor,
        upper_e: torch.Tensor,
        lower_g: torch.Tensor,
        upper_g: torch.Tensor,
        electric_cap: torch.Tensor,
        gas_cap: torch.Tensor,
        psi: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Solve the ALADIN coordination QP with active local constraints."""
        r = x_e - x_g
        mu = float(self.aladin_mu)
        active_tol = float(self.aladin_active_tolerance)

        scale_e = 1.0 + torch.maximum(torch.abs(lower_e), torch.abs(upper_e))
        scale_g = 1.0 + torch.maximum(torch.abs(lower_g), torch.abs(upper_g))
        active_bound_e = ((x_e - lower_e) <= active_tol * scale_e) | ((upper_e - x_e) <= active_tol * scale_e)
        active_bound_g = ((x_g - lower_g) <= active_tol * scale_g) | ((upper_g - x_g) <= active_tol * scale_g)
        free_e = ~active_bound_e
        free_g = ~active_bound_g

        a_e = hess_e + mu
        a_g = hess_g + mu
        rhs_e = -(grad_e + multiplier + mu * r)
        rhs_g = -(grad_g - multiplier - mu * r)
        d0_e, d0_g = self._apply_block_hessian_inverse(
            rhs_e, rhs_g, free_e, free_g, a_e, a_g, mu
        )

        electric_value = torch.sum(x_e, dim=1)
        electric_active = electric_value >= electric_cap - active_tol * (1.0 + torch.abs(electric_cap))
        gas_value = self._gas_constraint_value(x_g, psi)
        gas_active = gas_value >= gas_cap - active_tol * (1.0 + torch.abs(gas_cap))

        c1_e = free_e.to(x_e.dtype) * electric_active.unsqueeze(-1).to(x_e.dtype)
        c1_g = torch.zeros_like(x_g)
        c2_e = torch.zeros_like(x_e)
        c2_g = self._gas_constraint_gradient(x_g, psi) * free_g.to(x_g.dtype) * gas_active.unsqueeze(-1).to(x_g.dtype)

        h_c1_e, h_c1_g = self._apply_block_hessian_inverse(
            c1_e, c1_g, free_e, free_g, a_e, a_g, mu
        )
        h_c2_e, h_c2_g = self._apply_block_hessian_inverse(
            c2_e, c2_g, free_e, free_g, a_e, a_g, mu
        )

        m11 = torch.sum(c1_e * h_c1_e + c1_g * h_c1_g, dim=1)
        m12 = torch.sum(c1_e * h_c2_e + c1_g * h_c2_g, dim=1)
        m22 = torch.sum(c2_e * h_c2_e + c2_g * h_c2_g, dim=1)
        b1 = torch.sum(c1_e * d0_e + c1_g * d0_g, dim=1)
        b2 = torch.sum(c2_e * d0_e + c2_g * d0_g, dim=1)

        nu1 = torch.zeros_like(b1)
        nu2 = torch.zeros_like(b2)
        both = electric_active & gas_active
        only1 = electric_active & (~gas_active)
        only2 = gas_active & (~electric_active)
        eps = 1e-12
        nu1 = torch.where(only1, b1 / torch.clamp(m11, min=eps), nu1)
        nu2 = torch.where(only2, b2 / torch.clamp(m22, min=eps), nu2)
        det = torch.clamp(m11 * m22 - m12 * m12, min=eps)
        nu1_both = (m22 * b1 - m12 * b2) / det
        nu2_both = (-m12 * b1 + m11 * b2) / det
        nu1 = torch.where(both, nu1_both, nu1)
        nu2 = torch.where(both, nu2_both, nu2)

        corr_e = c1_e * nu1.unsqueeze(-1) + c2_e * nu2.unsqueeze(-1)
        corr_g = c1_g * nu1.unsqueeze(-1) + c2_g * nu2.unsqueeze(-1)
        h_corr_e, h_corr_g = self._apply_block_hessian_inverse(
            corr_e, corr_g, free_e, free_g, a_e, a_g, mu
        )
        d_e = d0_e - h_corr_e
        d_g = d0_g - h_corr_g

        # A bounded trust region is a standard globalization safeguard.
        radius_e = self.aladin_trust_region_fraction * torch.clamp(upper_e - lower_e, min=1.0)
        radius_g = self.aladin_trust_region_fraction * torch.clamp(upper_g - lower_g, min=1.0)
        d_e = torch.maximum(torch.minimum(d_e, radius_e), -radius_e)
        d_g = torch.maximum(torch.minimum(d_g, radius_g), -radius_g)

        slack = r + d_e - d_g
        qp_multiplier = multiplier + mu * slack
        return d_e, d_g, qp_multiplier

    def _projected_kkt_residual(
        self,
        x_e: torch.Tensor,
        x_g: torch.Tensor,
        grad_e: torch.Tensor,
        grad_g: torch.Tensor,
        multiplier: torch.Tensor,
        lower_e: torch.Tensor,
        upper_e: torch.Tensor,
        lower_g: torch.Tensor,
        upper_g: torch.Tensor,
        electric_cap: torch.Tensor,
        gas_cap: torch.Tensor,
        psi: torch.Tensor,
    ) -> torch.Tensor:
        e_trial, _ = self._project_box_sum_upper_with_dual(
            x_e - grad_e - multiplier,
            lower_e,
            upper_e,
            electric_cap,
            quadratic_weight=1.0,
            iterations=self.aladin_local_bisection_iters,
        )
        g_trial, _ = self._project_box_nonlinear_gas_with_dual(
            x_g - grad_g + multiplier,
            lower_g,
            upper_g,
            psi,
            gas_cap,
            quadratic_weight=1.0,
        )
        return torch.maximum(
            torch.max(torch.abs(x_e - e_trial), dim=1).values,
            torch.max(torch.abs(x_g - g_trial), dim=1).values,
        )

    def _run_aladin(
        self,
        state: dict,
        prev_state: dict,
        attack_strategy: dict,
        offline: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, int]:
        if self._num_gfpp() == 0:
            z = self._zeros_gt()
            return z, z, z, z, z, "aladin_no_coupling_variables", 0

        pmax_all = self._build_batched_pmax(offline)
        pmin_all = self._build_batched_pmin(offline)
        ramp_all = self.env.power.gen_ramp_max.unsqueeze(0)
        prev_pg = prev_state["P_g"]
        lower_all = torch.maximum(pmin_all, prev_pg - ramp_all)
        upper_all = torch.minimum(pmax_all, prev_pg + ramp_all)
        lower_all = torch.minimum(lower_all, upper_all)

        attacked_pg = self._apply_fdia_redispatch(state["P_g"], attack_strategy, offline)
        lower_e = lower_all[:, self.gfpp_idx]
        upper_e = upper_all[:, self.gfpp_idx]
        lower_g = lower_e.clone()
        upper_g = upper_e.clone()
        p_ref_e = torch.maximum(torch.minimum(attacked_pg[:, self.gfpp_idx], upper_e), lower_e)
        # Independent gas-operator reference: preserve the previous physically
        # supplied GFPP output instead of copying the attacked electric target.
        p_ref_g = torch.maximum(torch.minimum(prev_pg[:, self.gfpp_idx], upper_g), lower_g)

        electric_cap = self._electric_gfpp_budget(lower_all)
        gas_cap = self._gas_delivery_budget(state, prev_state)
        psi = self.env.psi_g[self.gfpp_idx].to(self.device)

        reference_scale = torch.maximum(
            torch.ones(p_ref_e.shape[0], dtype=p_ref_e.dtype, device=self.device),
            torch.maximum(
                torch.max(torch.abs(p_ref_e), dim=1).values,
                torch.max(torch.abs(p_ref_g), dim=1).values,
            ),
        )
        reference_gap = torch.max(torch.abs(p_ref_e - p_ref_g), dim=1).values
        relative_reference_gap = reference_gap / reference_scale

        if "z_e_gt" in prev_state and torch.is_tensor(prev_state["z_e_gt"]) and prev_state["z_e_gt"].shape == p_ref_e.shape:
            z_e = prev_state["z_e_gt"].clone().to(self.device)
        else:
            z_e = p_ref_e.clone()
        if "z_g_gt" in prev_state and torch.is_tensor(prev_state["z_g_gt"]) and prev_state["z_g_gt"].shape == p_ref_g.shape:
            z_g = prev_state["z_g_gt"].clone().to(self.device)
        else:
            z_g = p_ref_g.clone()
        if "lambda_gt" in prev_state and torch.is_tensor(prev_state["lambda_gt"]) and prev_state["lambda_gt"].shape == p_ref_e.shape:
            multiplier = prev_state["lambda_gt"].clone().to(self.device)
        else:
            multiplier = torch.zeros_like(p_ref_e)

        record_enabled = bool(getattr(self.env, "record_nd_convergence", False))
        physical_hour = int(attack_strategy.get("physical_hour", -1))
        record_hour = getattr(self.env, "record_nd_hour", None)
        if record_hour is not None:
            record_enabled = record_enabled and physical_hour == int(record_hour)
        if record_enabled and not hasattr(self.env, "nd_convergence_trace"):
            self.env.nd_convergence_trace = []

        stable_count = 0
        stop_reason = "aladin_max_iterations_reached"
        x_e = p_ref_e.clone()
        x_g = p_ref_g.clone()
        last_iter = 0
        previous_full_state = self._clone_tensor_state(state)
        start_time = time.perf_counter()

        a_local_e = self.aladin_electric_weight + self.aladin_rho
        a_local_g = self.aladin_gas_weight + self.aladin_rho
        for k in range(1, self.aladin_outer_iters + 1):
            last_iter = k
            center_e = (
                self.aladin_electric_weight * p_ref_e
                + self.aladin_rho * z_e
                - multiplier
            ) / a_local_e
            x_e, nu_e = self._project_box_sum_upper_with_dual(
                center_e, lower_e, upper_e, electric_cap,
                quadratic_weight=a_local_e,
                iterations=self.aladin_local_bisection_iters,
            )
            state = self._dispatch_electric_state(
                state, prev_state, attack_strategy, offline, x_e
            )

            gas_cap = self._gas_delivery_budget(state, prev_state)
            dynamic_upper_g = torch.minimum(
                upper_g, self._gas_pressure_upper_bound(state, upper_g)
            )
            dynamic_lower_g = torch.minimum(lower_g, dynamic_upper_g)
            center_g = (
                self.aladin_gas_weight * p_ref_g
                + self.aladin_rho * z_g
                + multiplier
            ) / a_local_g
            x_g, nu_g = self._project_box_nonlinear_gas_with_dual(
                center_g, dynamic_lower_g, dynamic_upper_g, psi, gas_cap,
                quadratic_weight=a_local_g,
            )
            state = self._dispatch_gas_state(
                state, prev_state, attack_strategy, x_g
            )

            grad_e = self.aladin_electric_weight * (x_e - p_ref_e)
            grad_g = self.aladin_gas_weight * (x_g - p_ref_g)
            hess_e = torch.ones_like(x_e) * (
                self.aladin_electric_weight + self.aladin_hessian_regularization
            )
            hess_g = torch.ones_like(x_g) * (
                self.aladin_gas_weight + self.aladin_hessian_regularization
            ) + nu_g.unsqueeze(-1) * self.aladin_gas_loss_coefficient * psi.pow(2).unsqueeze(0)

            d_e, d_g, qp_multiplier = self._solve_coordination_qp(
                x_e, x_g, grad_e, grad_g, hess_e, hess_g, multiplier,
                lower_e, upper_e, dynamic_lower_g, dynamic_upper_g,
                electric_cap, gas_cap, psi,
            )

            z_e_old = z_e.clone()
            z_g_old = z_g.clone()
            alpha = float(self.aladin_alpha)
            z_e = x_e + alpha * d_e
            z_g = x_g + alpha * d_g
            multiplier = multiplier + alpha * (qp_multiplier - multiplier)

            consensus = torch.max(torch.abs(x_e - x_g), dim=1).values
            step = torch.maximum(
                torch.max(torch.abs(z_e - z_e_old), dim=1).values,
                torch.max(torch.abs(z_g - z_g_old), dim=1).values,
            )
            kkt = self._projected_kkt_residual(
                x_e, x_g, grad_e, grad_g, multiplier,
                lower_e, upper_e, dynamic_lower_g, dynamic_upper_g,
                electric_cap, gas_cap, psi,
            )
            scale = torch.maximum(
                torch.ones_like(consensus),
                torch.maximum(
                    torch.max(torch.abs(x_e), dim=1).values,
                    torch.max(torch.abs(x_g), dim=1).values,
                ),
            )
            step_scale = torch.maximum(
                torch.ones_like(step),
                torch.maximum(
                    torch.max(torch.abs(z_e_old), dim=1).values,
                    torch.max(torch.abs(z_g_old), dim=1).values,
                ),
            )
            rel_consensus = consensus / scale
            rel_step = step / step_scale
            rel_kkt = kkt / scale
            full_state_rel = self._full_state_relative_change(previous_full_state, state)
            previous_full_state = self._clone_tensor_state(state)

            consensus_ok = bool(torch.max(rel_consensus).item() <= self.aladin_consensus_rel_tolerance)
            step_ok = bool(torch.max(rel_step).item() <= self.aladin_step_rel_tolerance)
            kkt_ok = bool(torch.max(rel_kkt).item() <= self.aladin_kkt_rel_tolerance)
            state_ok = bool(full_state_rel <= self.aladin_state_rel_tolerance)
            all_ok = bool(
                k >= self.aladin_min_outer_iters
                and consensus_ok and step_ok and kkt_ok and state_ok
            )
            stable_count = stable_count + 1 if all_ok else 0

            if record_enabled:
                self.env.nd_convergence_trace.append({
                    "solver_type": self.solver_name,
                    "model_scope": "physics_aware_full_state_replay",
                    "physical_hour": physical_hour,
                    "aladin_iteration": k,
                    "elapsed_time_s": float(time.perf_counter() - start_time),
                    "aladin_rho": float(self.aladin_rho),
                    "aladin_mu": float(self.aladin_mu),
                    "aladin_alpha": float(self.aladin_alpha),
                    "aladin_gas_loss_coefficient": float(self.aladin_gas_loss_coefficient),
                    "aladin_electric_budget": float(torch.mean(electric_cap).detach().cpu().item()),
                    "aladin_gas_budget": float(torch.mean(gas_cap).detach().cpu().item()),
                    "aladin_reference_gap": float(torch.max(reference_gap).detach().cpu().item()),
                    "aladin_relative_reference_gap": float(torch.max(relative_reference_gap).detach().cpu().item()),
                    "aladin_consensus_residual": float(torch.max(consensus).detach().cpu().item()),
                    "aladin_step_residual": float(torch.max(step).detach().cpu().item()),
                    "aladin_kkt_residual": float(torch.max(kkt).detach().cpu().item()),
                    "aladin_relative_consensus_residual": float(torch.max(rel_consensus).detach().cpu().item()),
                    "aladin_relative_step_residual": float(torch.max(rel_step).detach().cpu().item()),
                    "aladin_relative_kkt_residual": float(torch.max(rel_kkt).detach().cpu().item()),
                    "aladin_full_state_relative_residual": float(full_state_rel),
                    "aladin_consensus_rel_tolerance": float(self.aladin_consensus_rel_tolerance),
                    "aladin_step_rel_tolerance": float(self.aladin_step_rel_tolerance),
                    "aladin_kkt_rel_tolerance": float(self.aladin_kkt_rel_tolerance),
                    "aladin_state_rel_tolerance": float(self.aladin_state_rel_tolerance),
                    "aladin_convergence_stable_count": int(stable_count),
                    "aladin_convergence_patience": int(self.aladin_convergence_patience),
                    "aladin_consensus_pass": consensus_ok,
                    "aladin_step_pass": step_ok,
                    "aladin_kkt_pass": kkt_ok,
                    "aladin_state_pass": state_ok,
                    "aladin_all_residuals_pass": all_ok,
                    "stop_reason": "",
                })

            if stable_count >= self.aladin_convergence_patience:
                stop_reason = f"aladin_full_state_residuals_stable_{self.aladin_convergence_patience}_iterations"
                if record_enabled:
                    self.env.nd_convergence_trace[-1]["stop_reason"] = stop_reason
                break

        if record_enabled and self.env.nd_convergence_trace and not self.env.nd_convergence_trace[-1].get("stop_reason"):
            self.env.nd_convergence_trace[-1]["stop_reason"] = stop_reason

        # Shared point for physical replay; both local upper constraints are
        # enforced by reduction, which preserves previously satisfied limits.
        coupled = 0.5 * (x_e + x_g)
        coupled = torch.maximum(torch.minimum(coupled, upper_e), lower_e)
        coupled, _ = self._project_box_sum_upper_with_dual(
            coupled, lower_e, upper_e, electric_cap,
            quadratic_weight=1.0,
            iterations=self.aladin_local_bisection_iters,
        )
        coupled, _ = self._project_box_nonlinear_gas_with_dual(
            coupled, lower_g, upper_g, psi, gas_cap,
            quadratic_weight=1.0,
        )
        return coupled, z_e, z_g, multiplier, x_g, stop_reason, last_iter
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

        coupled, z_e, z_g, multiplier, p_g_local, stop_reason, iterations = self._run_aladin(
            state, prev, attack_strategy, offline
        )

        state = self._dispatch_electric_state(
            state, prev, attack_strategy, offline, coupled
        )
        state = self._dispatch_gas_state(
            state, prev, attack_strategy, coupled
        )

        state["z_gt"] = coupled.clone()
        state["z_e_gt"] = z_e.clone()
        state["z_g_gt"] = z_g.clone()
        state["lambda_gt"] = multiplier.clone()
        state["P_gt_g_local"] = p_g_local.clone()
        state["aladin_iterations"] = torch.full(
            (self.batch_size, 1),
            float(iterations),
            dtype=torch.float32,
            device=self.device,
        )
        state["aladin_stop_code"] = torch.full(
            (self.batch_size, 1),
            1.0 if (stop_reason.startswith("aladin_residuals_stable_") or stop_reason.startswith("aladin_full_state_residuals_stable_")) else 0.0,
            dtype=torch.float32,
            device=self.device,
        )

        power_shedding = torch.sum(state["P_d_sh"], dim=1)
        return state, power_shedding


ALADINDistributedEvaluationSolver = ALADINEvaluationSolver

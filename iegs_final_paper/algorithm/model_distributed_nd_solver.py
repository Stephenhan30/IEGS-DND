"""Model-distributed electric–gas neurodynamic solver with GFPP coordination."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Iterable

import torch

from algorithm.distributed_ode_solver import DistributedBatchedNeurodynamicODESolver


class ModelDistributedCouplingMixin:
    """Mixin that turns electric-gas block ND into model-distributed ND.

    The mixin assumes that the base class provides the physical residuals,
    projections, FDIA redispatch, and ADAM-style block update helpers used by
    ``DistributedBatchedNeurodynamicODESolver``.
    """

    model_gas_keys = ("S", "pi", "f_in", "f_out", "lp_cur", "P_q_sh", "comp_ratio", "P_gt_g")

    def _mdnd_init_params(self) -> None:
        self.mdnd_rho = float(getattr(self.env, "mdnd_rho", 10.0))

        # Absolute residual is retained for diagnostics/backward compatibility,
        # but stopping is based on dimensionless relative residuals.  This avoids
        # forcing a 118-bus/135-node system to satisfy an unrealistically tiny
        # absolute GFPP mismatch such as 1e-4 MW.
        self.mdnd_consensus_tolerance = float(
            getattr(self.env, "mdnd_consensus_tolerance", self.tol)
        )
        self.mdnd_consensus_rel_tolerance = float(
            getattr(self.env, "mdnd_consensus_rel_tolerance", 1e-3)
        )
        self.mdnd_state_rel_tolerance = float(
            getattr(self.env, "mdnd_state_rel_tolerance", 1e-3)
        )
        self.mdnd_convergence_patience = max(
            1, int(getattr(self.env, "mdnd_convergence_patience", 5))
        )

        # Stabilized consensus coordinator.  Each electric/gas branch first
        # computes a private proposal from the same iteration-k snapshot.  The
        # coordinator then projects the two GFPP proposals onto a common,
        # electrically feasible consensus point.  The *pre-projection proposal
        # mismatch* is used as the distributed convergence residual, so an exact
        # projection does not make the stopping test trivially pass.
        self.mdnd_consensus_relaxation = float(
            getattr(self.env, "mdnd_consensus_relaxation", 1.0)
        )
        self.mdnd_consensus_relaxation = min(
            1.0, max(0.0, self.mdnd_consensus_relaxation)
        )
        self.mdnd_dual_enabled = bool(getattr(self.env, "mdnd_dual_enabled", False))
        self.mdnd_dual_damping = float(getattr(self.env, "mdnd_dual_damping", 0.90))
        self.mdnd_dual_clip = float(getattr(self.env, "mdnd_dual_clip", 1.0e4))
        self._lambda_gt = None

    def _num_gfpp(self) -> int:
        return int(self.gfpp_idx.numel())

    def _zeros_gt(self, batch_size: int | None = None) -> torch.Tensor:
        b = self.batch_size if batch_size is None else int(batch_size)
        return torch.zeros((b, self._num_gfpp()), dtype=torch.float32, device=self.device)

    def _ensure_model_coupling_state(self, state: dict, prev_state: dict | None = None) -> dict:
        """Add the gas-side GFPP output and coupling multiplier if absent."""
        n_gt = self._num_gfpp()
        if n_gt <= 0:
            state["P_gt_g"] = self._zeros_gt()
            state["lambda_gt"] = self._zeros_gt()
            self._lambda_gt = self._zeros_gt()
            return state

        if "P_gt_g" not in state or not torch.is_tensor(state["P_gt_g"]):
            if prev_state is not None and "P_gt_g" in prev_state and torch.is_tensor(prev_state["P_gt_g"]):
                state["P_gt_g"] = prev_state["P_gt_g"].clone().to(self.device)
            else:
                state["P_gt_g"] = state["P_g"][:, self.gfpp_idx].clone()
        else:
            state["P_gt_g"] = state["P_gt_g"].clone().to(self.device)

        if self._lambda_gt is None or self._lambda_gt.shape != state["P_gt_g"].shape:
            if prev_state is not None and "lambda_gt" in prev_state and torch.is_tensor(prev_state["lambda_gt"]):
                self._lambda_gt = prev_state["lambda_gt"].clone().to(self.device)
            else:
                self._lambda_gt = torch.zeros_like(state["P_gt_g"])

        state["lambda_gt"] = self._lambda_gt.clone()
        return state

    def _p_gt_e(self, state: dict) -> torch.Tensor:
        if self._num_gfpp() <= 0:
            return self._zeros_gt(state[next(iter(state))].shape[0])
        return state["P_g"][:, self.gfpp_idx]

    def _p_gt_g(self, state: dict) -> torch.Tensor:
        if "P_gt_g" not in state:
            return self._p_gt_e(state).clone()
        return state["P_gt_g"]

    def _lambda(self, state: dict | None = None) -> torch.Tensor:
        if state is not None and "lambda_gt" in state and torch.is_tensor(state["lambda_gt"]):
            return state["lambda_gt"].to(self.device)
        if self._lambda_gt is None:
            return self._zeros_gt()
        return self._lambda_gt

    def _coupling_residual_tensor(self, state: dict) -> torch.Tensor:
        return self._p_gt_e(state) - self._p_gt_g(state)

    def _relative_consensus_residual_from_proposals(
        self,
        p_e: torch.Tensor,
        p_g: torch.Tensor,
    ) -> float:
        """Return the relative GFPP mismatch of two *independent* proposals.

        This helper deliberately accepts the electric- and gas-side proposal
        tensors directly.  It must be called before the coordinator overwrites
        the two local copies with a shared consensus value.  Using ``state``
        after consensus projection would make the recorded residual identically
        zero and would produce a false horizontal convergence curve.
        """
        if p_e.numel() <= 0 or p_g.numel() <= 0:
            return 0.0
        one = torch.tensor(1.0, dtype=p_e.dtype, device=p_e.device)
        scale = torch.maximum(
            one,
            torch.maximum(
                torch.max(torch.abs(p_e)),
                torch.max(torch.abs(p_g)),
            ),
        )
        rel = torch.max(torch.abs(p_e - p_g)) / (scale + 1e-12)
        return float(rel.detach().cpu().item())

    def _relative_consensus_residual(
        self,
        state: dict,
        residual: torch.Tensor | None = None,
    ) -> float:
        """Scale-safe GFPP coupling residual used by the DND stop test.

        r_rel = ||P_gt_e - P_gt_g||_inf /
                max(1, ||P_gt_e||_inf, ||P_gt_g||_inf)
        """
        if residual is None:
            residual = self._coupling_residual_tensor(state)
        if residual.numel() <= 0:
            return 0.0

        p_e = self._p_gt_e(state)
        p_g = self._p_gt_g(state)
        one = torch.tensor(1.0, dtype=p_e.dtype, device=p_e.device)
        scale = torch.maximum(
            one,
            torch.maximum(
                torch.max(torch.abs(p_e)),
                torch.max(torch.abs(p_g)),
            ),
        )
        rel = torch.max(torch.abs(residual)) / (scale + 1e-12)
        return float(rel.detach().cpu().item())

    def _update_coupling_multiplier(
        self,
        state: dict,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Update the optional damped coupling multiplier.

        The previous implementation accumulated ``lambda += rho*r`` after every
        single local Adam step.  That is not a valid method-of-multipliers inner
        solve and caused the multiplier to grow while the GFPP mismatch stayed
        almost constant.  The stabilized default uses primal consensus
        projection and keeps the dual term disabled.  A bounded, damped dual
        update remains available for sensitivity tests.
        """
        r = self._coupling_residual_tensor(state) if residual is None else residual
        if self._lambda_gt is None or self._lambda_gt.shape != r.shape:
            self._lambda_gt = torch.zeros_like(r)

        if self.mdnd_dual_enabled:
            self._lambda_gt = (
                self.mdnd_dual_damping * self._lambda_gt + self.mdnd_rho * r
            )
            self._lambda_gt = torch.clamp(
                self._lambda_gt,
                min=-self.mdnd_dual_clip,
                max=self.mdnd_dual_clip,
            )
        else:
            self._lambda_gt.zero_()

        state["lambda_gt"] = self._lambda_gt.clone()
        return r

    def _coordinate_gfpp_consensus(
        self,
        state: dict,
        prev_state: dict,
        gfpp_offline_mask: torch.Tensor | None = None,
    ) -> dict:
        """Project electric/gas GFPP proposals onto one feasible consensus.

        The consensus point is the average of both local proposals, clipped by
        generator capacity and ramp constraints.  With relaxation=1 (default),
        both local copies are set to the same shared point after every
        coordination barrier.  Values below 1 provide under-relaxation.
        """
        if self._num_gfpp() <= 0:
            return state

        p_e = self._p_gt_e(state)
        p_g = self._p_gt_g(state)
        z = 0.5 * (p_e + p_g)

        pmax_all = self._build_batched_pmax(gfpp_offline_mask)
        pmin_all = self._build_batched_pmin(gfpp_offline_mask)
        pmax = pmax_all[:, self.gfpp_idx]
        pmin = pmin_all[:, self.gfpp_idx]

        ramp = self.env.power.gen_ramp_max[self.gfpp_idx].unsqueeze(0)
        prev_p = prev_state["P_g"][:, self.gfpp_idx]
        lower = torch.maximum(pmin, prev_p - ramp)
        upper = torch.minimum(pmax, prev_p + ramp)
        z = torch.maximum(torch.minimum(z, upper), lower)

        alpha = self.mdnd_consensus_relaxation
        p_e_new = (1.0 - alpha) * p_e + alpha * z
        p_g_new = (1.0 - alpha) * p_g + alpha * z

        state["P_g"] = state["P_g"].clone()
        state["P_g"][:, self.gfpp_idx] = p_e_new
        state["P_gt_g"] = p_g_new
        return self._project_model_coupling_state(state, gfpp_offline_mask)

    def _gfpp_gas_demand_from_gt_power(self, p_gt: torch.Tensor) -> torch.Tensor:
        """Gas-node demand produced by the gas-side GFPP output variable."""
        batch = p_gt.shape[0]
        gas_req = torch.zeros((batch, self.env.gas.num_nodes), dtype=p_gt.dtype, device=self.device)
        if self._num_gfpp() <= 0:
            return gas_req
        for local_j, gen_idx in enumerate(self.env.gfpp_indices):
            gas_node = self.env.gfpp_to_gas_node[gen_idx]
            gas_req[:, gas_node] += p_gt[:, local_j] * self.env.psi_g[gen_idx]
        return gas_req

    def _gas_mass_residual_with_gfpp_power(self, state: dict, p_gt: torch.Tensor) -> torch.Tensor:
        env_g = self.env.gas
        nodal_S = state["S"] @ env_g.A_nw.T
        net_pipe_flow = state["f_out"] @ env_g.A_np_minus.T - state["f_in"] @ env_g.A_np_plus.T
        gas_req = self._gfpp_gas_demand_from_gt_power(p_gt)
        nodal_D = env_g.base_gas_load.unsqueeze(0) + gas_req - state["P_q_sh"]
        return nodal_S + net_pipe_flow - nodal_D

    def _project_model_coupling_state(self, state: dict, gfpp_offline_mask: torch.Tensor | None = None) -> dict:
        """Clamp the gas-side GFPP output variable to physical GFPP bounds."""
        if self._num_gfpp() <= 0 or "P_gt_g" not in state:
            return state
        pmax = self.env.power.gen_pmax[self.gfpp_idx].unsqueeze(0).repeat(self.batch_size, 1)
        pmin = torch.zeros_like(pmax)
        if gfpp_offline_mask is not None:
            pmax = torch.where(gfpp_offline_mask.unsqueeze(-1), torch.zeros_like(pmax), pmax)
        state["P_gt_g"] = torch.maximum(torch.minimum(state["P_gt_g"], pmax), pmin)
        if self._lambda_gt is not None:
            state["lambda_gt"] = self._lambda_gt.clone()
        return state

    def _model_coupling_summary(self, state: dict) -> dict[str, float]:
        out: dict[str, float] = {}
        if self._num_gfpp() <= 0:
            out["mean_gt_consensus_residual"] = 0.0
            out["max_gt_consensus_residual"] = 0.0
            return out
        r = self._coupling_residual_tensor(state)
        out["mean_gt_consensus_residual"] = float(torch.mean(torch.abs(r)).detach().cpu().item())
        out["max_gt_consensus_residual"] = float(torch.max(torch.abs(r)).detach().cpu().item())
        out["mean_power_side_gfpp"] = float(torch.mean(torch.sum(self._p_gt_e(state), dim=1)).detach().cpu().item())
        out["mean_gas_side_gfpp"] = float(torch.mean(torch.sum(self._p_gt_g(state), dim=1)).detach().cpu().item())
        return out

    def _projected_state_residual_for_keys(self, prev_state: dict, curr_state: dict, keys: Iterable[str]) -> float:
        # The inherited implementation already handles arbitrary key lists; keep
        # this method for compatibility when the base class is the block solver.
        return super()._projected_state_residual_for_keys(prev_state, curr_state, keys)

    def _projected_state_residual(self, prev_state: dict, curr_state: dict) -> float:
        base_res = super()._projected_state_residual(prev_state, curr_state)
        if "P_gt_g" not in prev_state or "P_gt_g" not in curr_state:
            return base_res
        extra = self._projected_state_residual_for_keys(prev_state, curr_state, ["P_gt_g"])
        return float((base_res**2 + extra**2) ** 0.5 / (2.0**0.5))

    def _energy_value_batched(self, state: dict, attack_strategy: dict, gfpp_offline_mask: torch.Tensor) -> float:
        state = self._ensure_model_coupling_state(state)
        energy = super()._energy_value_batched(state, attack_strategy, gfpp_offline_mask)
        L1 = float(self.env.lambda1)

        # Replace the inherited gas mass-balance residual based on P_g by a gas
        # model residual based on P_gt_g.
        old_res = self._gas_mass_residual_with_gfpp_power(state, self._p_gt_e(state))
        new_res = self._gas_mass_residual_with_gfpp_power(state, self._p_gt_g(state))
        delta_energy = 0.5 * L1 * (torch.mean(new_res**2) - torch.mean(old_res**2))

        # Add augmented-Lagrangian coupling energy.
        r = self._coupling_residual_tensor(state)
        lam = self._lambda(state)
        coupling_energy = torch.mean(torch.sum(lam * r + 0.5 * self.mdnd_rho * r**2, dim=1)) if r.numel() else torch.tensor(0.0, device=self.device)

        return float(float(energy) + float(delta_energy.detach().cpu().item()) + float(coupling_energy.detach().cpu().item()))

    def _calculate_analytical_gradient_batched(self, state: dict, attack_strategy: dict, gfpp_offline_mask: torch.Tensor) -> dict:
        state = self._ensure_model_coupling_state(state)
        grad = super()._calculate_analytical_gradient_batched(state, attack_strategy, gfpp_offline_mask)
        L1 = float(self.env.lambda1)
        env_g = self.env.gas

        if "P_gt_g" not in grad:
            grad["P_gt_g"] = torch.zeros_like(state["P_gt_g"])
        if "lambda_gt" in grad:
            grad["lambda_gt"].zero_()

        # Replace gas-model residual contributions: inherited block solver uses
        # power-side P_g inside gas mass balance.  Remove that contribution from
        # gas-local variables and add the one based on gas-side P_gt_g.
        old_res = self._gas_mass_residual_with_gfpp_power(state, self._p_gt_e(state))
        new_res = self._gas_mass_residual_with_gfpp_power(state, self._p_gt_g(state))
        diff_res = new_res - old_res

        if "S" in grad:
            grad["S"] += L1 * (diff_res @ env_g.A_nw)
        if "f_out" in grad:
            grad["f_out"] += L1 * (diff_res @ env_g.A_np_minus)
        if "f_in" in grad:
            grad["f_in"] += L1 * (-(diff_res @ env_g.A_np_plus))
        if "P_q_sh" in grad:
            grad["P_q_sh"] += L1 * diff_res

        # Gas-side GFPP variable appears in gas mass balance as demand.  For a
        # GFPP connected to gas node n, d residual_n / d P_gt_g = -psi.
        if self._num_gfpp() > 0:
            for local_j, gen_idx in enumerate(self.env.gfpp_indices):
                gas_node = self.env.gfpp_to_gas_node[gen_idx]
                grad["P_gt_g"][:, local_j] += -L1 * new_res[:, gas_node] * self.env.psi_g[gen_idx]

        # Augmented-Lagrangian consistency gradient.
        r = self._coupling_residual_tensor(state)
        lam = self._lambda(state)
        cgrad = lam + self.mdnd_rho * r
        if self._num_gfpp() > 0:
            grad["P_g"][:, self.gfpp_idx] += cgrad
            grad["P_gt_g"] += -cgrad
        return grad


class ModelDistributedNeurodynamicODESolver(
    ModelDistributedCouplingMixin,
    DistributedBatchedNeurodynamicODESolver,
):
    """QSS/dynamic-linepack model-distributed ND lower solver."""

    solver_name = "model_distributed_qss_neurodynamic"
    gas_keys = ModelDistributedCouplingMixin.model_gas_keys

    def __init__(self, iegs_env, batch_size: int, max_ode_steps: int = 3500, tolerance: float = 1e-4):
        super().__init__(iegs_env, batch_size=batch_size, max_ode_steps=max_ode_steps, tolerance=tolerance)
        self._mdnd_init_params()

    def _apply_gas_projection_only(
        self,
        state: dict,
        prev_state: dict,
        attack_strategy: dict,
        s_t: torch.Tensor,
    ) -> dict:
        state = super()._apply_gas_projection_only(state, prev_state, attack_strategy, s_t)
        state = self._project_model_coupling_state(state)
        return state

    def _resolve_parallel_backend(self) -> str:
        """Resolve the execution backend for electric/gas local solves.

        ``auto`` uses two CUDA streams on GPU and deterministic Jacobi execution
        on CPU.  Explicit ``cpu_threads`` is retained for machines where PyTorch
        thread oversubscription is controlled externally.  All implementations use a Jacobi-style snapshot: the
        electric and gas submodels read exactly the same iteration-k state and
        their disjoint local results are merged only after both tasks finish.
        """
        if not bool(getattr(self.env, "mdnd_parallel_models", True)):
            return "sequential_jacobi"

        requested = str(getattr(self.env, "mdnd_parallel_backend", "auto")).strip().lower()
        aliases = {
            "threads": "cpu_threads",
            "thread": "cpu_threads",
            "cuda": "cuda_streams",
            "streams": "cuda_streams",
            "sequential": "sequential_jacobi",
        }
        requested = aliases.get(requested, requested)

        if requested == "auto":
            return "cuda_streams" if self.device.type == "cuda" else "sequential_jacobi"
        if requested == "cuda_streams":
            if self.device.type != "cuda" or not torch.cuda.is_available():
                return "cpu_threads"
            return requested
        if requested in {"cpu_threads", "sequential_jacobi"}:
            return requested
        raise ValueError(
            "env.mdnd_parallel_backend must be one of "
            "{'auto', 'cpu_threads', 'cuda_streams', 'sequential_jacobi'}"
        )

    def _make_branch_local_state(
        self,
        state_snapshot: dict,
        owned_keys: Iterable[str],
    ) -> dict:
        """Create one local branch state without cloning the whole system state.

        Non-owned tensors are shared read-only from the iteration-k snapshot.
        Only variables owned by the local operator are cloned because only those
        variables can be changed by the local ND/Adam step and its projection.
        This removes three full-state copies per DND iteration while preserving
        the Jacobi information pattern.
        """
        owned = set(owned_keys)
        local: dict = {}
        for key, value in state_snapshot.items():
            if torch.is_tensor(value) and key in owned:
                local[key] = value.clone()
            else:
                local[key] = value
        return local

    @torch.no_grad()
    def _run_local_model_branch(
        self,
        branch: str,
        state_snapshot: dict,
        prev_state: dict,
        attack_strategy: dict,
        gfpp_offline_mask: torch.Tensor,
        s_t: torch.Tensor,
        keys: list[str],
        m_local: dict,
        v_local: dict,
        step: int,
        local_dtau_scale: float,
        lr_map: dict[str, float],
        beta1: float,
        beta2: float,
        eps: float,
    ) -> dict:
        """Solve one local model from a read-only iteration snapshot.

        The returned dictionary is private to the branch.  Only the variables
        listed in ``keys`` are merged into the coordinated state afterwards.
        This prevents electric/gas write races and implements a true Jacobi
        information pattern for the two distributed operators.
        """
        local_state = self._make_branch_local_state(state_snapshot, keys)
        local_state, _ = self._adam_update_block(
            state=local_state,
            attack_strategy=attack_strategy,
            gfpp_offline_mask=gfpp_offline_mask,
            s_t=s_t,
            keys=keys,
            m=m_local,
            v=v_local,
            step=step,
            local_dtau_scale=local_dtau_scale,
            lr_map=lr_map,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
        )

        if branch == "electric":
            local_state = self._apply_electric_projection_only(
                local_state, prev_state, gfpp_offline_mask
            )
        elif branch == "gas":
            local_state = self._apply_gas_projection_only(
                local_state, prev_state, attack_strategy, s_t
            )
        else:
            raise ValueError(f"Unknown local model branch: {branch}")

        local_state = self._project_model_coupling_state(
            local_state, gfpp_offline_mask
        )
        return local_state

    def _merge_parallel_model_states(
        self,
        state_snapshot: dict,
        electric_state: dict,
        gas_state: dict,
        electric_keys: list[str],
        gas_keys: list[str],
    ) -> dict:
        """Merge disjoint local decisions after the parallel barrier."""
        # Shallow-copy the dictionary only.  All tensors in state_snapshot are
        # read-only during the merge; branch-owned tensors below replace the
        # corresponding references.
        merged = dict(state_snapshot)
        for key in electric_keys:
            if key in electric_state:
                merged[key] = electric_state[key]
        for key in gas_keys:
            if key in gas_state:
                merged[key] = gas_state[key]

        # The multiplier belongs to the coordinator, not to either local model.
        # It is updated exactly once after the two branches have synchronized.
        if "lambda_gt" in state_snapshot:
            merged["lambda_gt"] = state_snapshot["lambda_gt"].clone()
        return merged

    @torch.no_grad()
    def evolve_to_equilibrium_batched(
        self,
        current_state: dict,
        prev_state: dict,
        attack_strategy: dict,
        gfpp_offline_mask: torch.Tensor,
    ):
        state = {k: v.clone().to(self.device) for k, v in current_state.items() if torch.is_tensor(v)}
        prev_state = {k: v.clone().to(self.device) for k, v in prev_state.items() if torch.is_tensor(v)}
        state = self._ensure_model_coupling_state(state, prev_state)
        prev_state = self._ensure_model_coupling_state(prev_state)
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

        # Electric model's initial response to FDIA.
        state["P_g"] = self._apply_fdia_redispatch(state, attack_strategy, gfpp_offline_mask)
        state = self._apply_electric_projection_only(state, prev_state, gfpp_offline_mask)
        state = self._project_model_coupling_state(state, gfpp_offline_mask)

        state_history = self._clone_tensor_state(state)
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
            "P_gt_g": 0.20 if is_large_system else 0.30,
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
        relative_stable_count = 0
        stop_reason = "max_steps_reached"
        trace_effective_interval = float(getattr(self.env, "nd_trace_effective_interval", 10.0))
        check_interval = max(1, int(round(trace_effective_interval / max(local_dtau_scale, 1e-12))))
        min_check_step = max(self.ode_min_steps, check_interval)

        electric_keys = self._available_keys(state, self.electric_keys)
        gas_keys = self._available_keys(state, self.gas_keys)

        # Separate ADAM memories guarantee that each local operator owns only
        # its private algorithmic state.  No branch writes another branch's
        # first/second moments.
        m_e = {k: torch.zeros_like(state[k]) for k in electric_keys}
        v_e = {k: torch.zeros_like(state[k]) for k in electric_keys}
        m_g = {k: torch.zeros_like(state[k]) for k in gas_keys}
        v_g = {k: torch.zeros_like(state[k]) for k in gas_keys}

        parallel_backend = self._resolve_parallel_backend()
        cpu_executor = (
            ThreadPoolExecutor(max_workers=2, thread_name_prefix="mdnd-local")
            if parallel_backend == "cpu_threads"
            else None
        )
        electric_stream = None
        gas_stream = None
        if parallel_backend == "cuda_streams":
            electric_stream = torch.cuda.Stream(device=self.device)
            gas_stream = torch.cuda.Stream(device=self.device)

        if not bool(getattr(self, "_mdnd_parallel_backend_printed", False)):
            print(
                f"[Model-distributed ND] electric/gas local models: "
                f"parallel_backend={parallel_backend}"
            )
            self._mdnd_parallel_backend_printed = True

        try:
            for step in range(1, n_steps + 1):
                # 1) Freeze iteration-k information shared by both local
                #    operators.  Both branches must read this same snapshot.
                state_k = dict(state)

                branch_args = dict(
                    state_snapshot=state_k,
                    prev_state=prev_state,
                    attack_strategy=attack_strategy,
                    gfpp_offline_mask=gfpp_offline_mask,
                    s_t=s_t,
                    step=step,
                    local_dtau_scale=local_dtau_scale,
                    lr_map=lr_map,
                    beta1=beta1,
                    beta2=beta2,
                    eps=eps,
                )

                # 2) Electric and gas local model updates execute in
                #    parallel.  CPU: two persistent worker threads.
                #    CUDA: two independent streams.
                if parallel_backend == "cpu_threads":
                    assert cpu_executor is not None
                    electric_future = cpu_executor.submit(
                        self._run_local_model_branch,
                        branch="electric",
                        keys=electric_keys,
                        m_local=m_e,
                        v_local=v_e,
                        **branch_args,
                    )
                    gas_future = cpu_executor.submit(
                        self._run_local_model_branch,
                        branch="gas",
                        keys=gas_keys,
                        m_local=m_g,
                        v_local=v_g,
                        **branch_args,
                    )
                    electric_state = electric_future.result()
                    gas_state = gas_future.result()

                elif parallel_backend == "cuda_streams":
                    assert electric_stream is not None and gas_stream is not None
                    current_stream = torch.cuda.current_stream(device=self.device)
                    electric_stream.wait_stream(current_stream)
                    gas_stream.wait_stream(current_stream)

                    with torch.cuda.stream(electric_stream):
                        electric_state = self._run_local_model_branch(
                            branch="electric",
                            keys=electric_keys,
                            m_local=m_e,
                            v_local=v_e,
                            **branch_args,
                        )
                    with torch.cuda.stream(gas_stream):
                        gas_state = self._run_local_model_branch(
                            branch="gas",
                            keys=gas_keys,
                            m_local=m_g,
                            v_local=v_g,
                            **branch_args,
                        )

                    current_stream.wait_stream(electric_stream)
                    current_stream.wait_stream(gas_stream)

                else:  # deterministic Jacobi fallback, same information pattern
                    electric_state = self._run_local_model_branch(
                        branch="electric",
                        keys=electric_keys,
                        m_local=m_e,
                        v_local=v_e,
                        **branch_args,
                    )
                    gas_state = self._run_local_model_branch(
                        branch="gas",
                        keys=gas_keys,
                        m_local=m_g,
                        v_local=v_g,
                        **branch_args,
                    )

                # 3) Barrier + merge: only local variables are collected.
                #    Then the coordinator updates the coupling multiplier once.
                if self._num_gfpp() > 0:
                    p_e_proposal = (
                        electric_state["P_g"][:, self.gfpp_idx]
                        .detach()
                        .clone()
                    )
                    p_g_proposal = (
                        gas_state["P_gt_g"]
                        .detach()
                        .clone()
                    )
                    proposal_r = p_e_proposal - p_g_proposal
                    proposal_norm = float(
                        torch.max(torch.abs(proposal_r)).detach().cpu().item()
                    )
                    proposal_rel = self._relative_consensus_residual_from_proposals(
                        p_e_proposal, p_g_proposal
                    )
                    proposal_power_sum = float(
                        torch.mean(torch.sum(p_e_proposal, dim=1))
                        .detach()
                        .cpu()
                        .item()
                    )
                    proposal_gas_sum = float(
                        torch.mean(torch.sum(p_g_proposal, dim=1))
                        .detach()
                        .cpu()
                        .item()
                    )
                else:
                    proposal_r = self._zeros_gt()
                    proposal_norm = 0.0
                    proposal_rel = 0.0
                    proposal_power_sum = 0.0
                    proposal_gas_sum = 0.0

                state = self._merge_parallel_model_states(
                    state_snapshot=state_k,
                    electric_state=electric_state,
                    gas_state=gas_state,
                    electric_keys=electric_keys,
                    gas_keys=gas_keys,
                )
                state = self._project_model_coupling_state(state, gfpp_offline_mask)

                state = self._coordinate_gfpp_consensus(
                    state, prev_state, gfpp_offline_mask
                )
                self._update_coupling_multiplier(state, proposal_r)

                post_r = self._coupling_residual_tensor(state)
                post_norm = (
                    float(torch.max(torch.abs(post_r)).detach().cpu().item())
                    if post_r.numel()
                    else 0.0
                )
                post_rel = self._relative_consensus_residual(state, post_r)

                # Use proposal mismatch for the non-trivial convergence test.
                consensus_norm = proposal_norm
                consensus_rel = proposal_rel

                # Always record/check the first coordination round.  With an
                # exact consensus projection (relaxation=1), the first local
                # electric/gas proposals can have a large mismatch and the
                # coordinator can remove it immediately.  If recording starts
                # only after ``min_check_step``, every saved r_GT value can be
                # zero, which incorrectly appears as a constant convergence
                # curve.
                if step == 1 or (
                    step >= min_check_step and step % check_interval == 0
                ):
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

                    # Primary DND stopping criteria use dimensionless relative
                    # state change and relative GFPP consensus residual.
                    state_rel_pass = (
                        projected_gradient_residual
                        <= self.mdnd_state_rel_tolerance
                    )
                    consensus_rel_pass = (
                        consensus_rel
                        <= self.mdnd_consensus_rel_tolerance
                    )
                    projected_gradient_pass = projected_gradient_residual <= self.ode_rel_tolerance
                    consensus_pass = consensus_norm <= self.mdnd_consensus_tolerance
                    gradient_pass = gradient_norm <= grad_tolerance

                    if state_rel_pass and consensus_rel_pass:
                        relative_stable_count += 1
                    else:
                        relative_stable_count = 0

                    if energy_pass:
                        energy_stable_count += 1
                    else:
                        energy_stable_count = 0
    
                    coupling = self._model_coupling_summary(state)
    
                    if record_enabled:
                        self.env.nd_convergence_trace.append(
                            {
                                "solver_type": self.solver_name,
                                "physical_hour": int(physical_hour_int),
                                "raw_step": int(step),
                                "effective_step": float(step * local_dtau_scale),
                                "projected_gradient_residual": float(projected_gradient_residual),
                                "electric_projected_residual": float(electric_projected_residual),
                                "gas_projected_residual": float(gas_projected_residual),
                                "gt_consensus_residual": float(consensus_norm),
                                "gt_proposal_consensus_residual": float(proposal_norm),
                                "gt_postcoord_consensus_residual": float(post_norm),
                                "gt_consensus_tolerance": float(self.mdnd_consensus_rel_tolerance),
                                "gt_consensus_converged": bool(consensus_rel_pass),
                                "gt_absolute_consensus_tolerance": float(self.mdnd_consensus_tolerance),
                                "gt_absolute_consensus_converged": bool(consensus_pass),
                                "gt_relative_consensus_residual": float(consensus_rel),
                                "gt_proposal_relative_consensus_residual": float(proposal_rel),
                                "gt_postcoord_relative_consensus_residual": float(post_rel),
                                "mean_power_side_gfpp_proposal": float(proposal_power_sum),
                                "mean_gas_side_gfpp_proposal": float(proposal_gas_sum),
                                "gt_relative_consensus_tolerance": float(self.mdnd_consensus_rel_tolerance),
                                "gt_relative_consensus_converged": bool(consensus_rel_pass),
                                "state_relative_residual": float(projected_gradient_residual),
                                "state_relative_tolerance": float(self.mdnd_state_rel_tolerance),
                                "state_relative_converged": bool(state_rel_pass),
                                "relative_stable_count": int(relative_stable_count),
                                "relative_stable_patience": int(self.mdnd_convergence_patience),
                                "projected_gradient_tolerance": float(self.mdnd_state_rel_tolerance),
                                "projected_gradient_converged": bool(state_rel_pass),
                                "legacy_ode_projected_tolerance": float(self.ode_rel_tolerance),
                                "legacy_ode_projected_converged": bool(projected_gradient_pass),
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
                                **coupling,
                                "stop_reason": "",
                            }
                        )
    
                    # The DND solve stops only after the relative state change
                    # and relative GFPP consensus residual remain below their
                    # tolerances for several consecutive monitoring checks.
                    # Raw gradient and energy criteria remain diagnostic so the
                    # convergence plots can still report them.
                    if relative_stable_count >= self.mdnd_convergence_patience:
                        stop_reason = (
                            f"relative_state_and_consensus_stable_"
                            f"{self.mdnd_convergence_patience}_checks"
                        )
                        if record_enabled and len(self.env.nd_convergence_trace) > 0:
                            self.env.nd_convergence_trace[-1]["stop_reason"] = stop_reason
                        break

                    state_history = self._clone_tensor_state(state)
        finally:
            if cpu_executor is not None:
                cpu_executor.shutdown(wait=True)


        if record_enabled and len(getattr(self.env, "nd_convergence_trace", [])) > 0:
            if self.env.nd_convergence_trace[-1].get("stop_reason", "") == "":
                self.env.nd_convergence_trace[-1]["stop_reason"] = stop_reason

        # Aggregate real DND work statistics over all batched lower-level calls.
        stats = getattr(self.env, "mdnd_performance_stats", None)
        if not isinstance(stats, dict):
            stats = {
                "equilibrium_calls": 0,
                "candidate_hours": 0,
                "raw_nd_steps": 0,
                "candidate_weighted_steps": 0,
                "max_steps_hits": 0,
                "stop_reasons": {},
            }
        stats["equilibrium_calls"] = int(stats.get("equilibrium_calls", 0)) + 1
        stats["candidate_hours"] = int(stats.get("candidate_hours", 0)) + int(self.batch_size)
        stats["raw_nd_steps"] = int(stats.get("raw_nd_steps", 0)) + int(step)
        stats["candidate_weighted_steps"] = int(stats.get("candidate_weighted_steps", 0)) + int(step) * int(self.batch_size)
        if stop_reason == "max_steps_reached":
            stats["max_steps_hits"] = int(stats.get("max_steps_hits", 0)) + 1
        stop_counts = dict(stats.get("stop_reasons", {}))
        stop_counts[stop_reason] = int(stop_counts.get(stop_reason, 0)) + 1
        stats["stop_reasons"] = stop_counts
        stats["last_step_count"] = int(step)
        stats["last_stop_reason"] = str(stop_reason)
        self.env.mdnd_performance_stats = stats

        state = self._apply_physical_bounds_batched(state, prev_state, attack_strategy, s_t, gfpp_offline_mask)
        state = self._project_model_coupling_state(state, gfpp_offline_mask)
        state["lp_prev"] = state["lp_cur"].clone()
        state["S_prev"] = state["S"].clone()
        if self._lambda_gt is not None:
            state["lambda_gt"] = self._lambda_gt.clone()
        power_shedding = torch.sum(state["P_d_sh"], dim=1)
        return state, power_shedding

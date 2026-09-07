from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
from types import SimpleNamespace

import numpy as np

from attacker.fdia_generator import FDIAGenerator
from simulation import simulate_strategies_batched


@dataclass
class MIPNDResult:
    strategy: dict
    fitness: float
    convergence: list
    metric_curve: list


class MIPNDOptimizer:
    """
    MIP-ND optimizer.

    Upper master:
        A CPLEX mixed-integer linear programming (MIP) model jointly selects
        candidate FDIA injection time, DoS trigger time, and a stealthy FDIA
        load-deviation vector under attack-budget constraints.  Repeated CPLEX MIP
        solves with no-good timing cuts generate a finite Top-K candidate set.

    Lower subproblem:
        Each candidate strategy is evaluated by simulate_strategies_batched(),
        where the lower-level IEGS physical response is solved by the
        neurodynamic ODE solver.

    Difference from algorithm/mp_nd_master.py:
        This file is the extra MIP-ND method.  The original MP-ND file is not modified.

    Important modeling note:
        The upper MIP is a candidate generator based on a linear surrogate,
        not an exact global MINLP reformulation of the full nonlinear dynamic
        IEGS attack problem.  The final strategy is selected by the true
        lower-level neurodynamic response.
    """

    def __init__(
        self,
        env,
        mode: str = "coordinated",
        time_step: float = 1.0,
        chunk_size: int = 64,
        seed: int = 42,
    ):
        if mode not in {"coordinated", "fdia_only", "dos_only"}:
            raise ValueError("mode must be coordinated, fdia_only, or dos_only")

        self.env = env
        self.mode = mode
        self.time_step = max(float(time_step), 1e-9)
        self.chunk_size = max(1, int(chunk_size))
        self.rng = np.random.default_rng(seed)
        self._mip_warning_printed = False
        self._fdia_warning_printed = False

        # Exact-strategy response cache.  It is used mainly by the all-batch
        # local timing refinement, where neighboring 3x3 windows overlap heavily.
        self._response_cache: dict[tuple, dict] = {}
        self._cache_hits = 0
        self._cache_misses = 0

    # ------------------------------------------------------------------
    # Basic utilities
    # ------------------------------------------------------------------
    def _time_grid(self, t_min: float, t_max: float) -> list[float]:
        """Return an inclusive discrete scheduling grid."""
        t_min = float(t_min)
        t_max = float(t_max)
        if t_max < t_min:
            return []

        vals: list[float] = []
        k_max = int(math.floor((t_max - t_min) / self.time_step + 1e-9))
        for k in range(k_max + 1):
            v = t_min + k * self.time_step
            if v <= t_max + 1e-9:
                vals.append(float(np.round(v, 10)))

        if not vals or abs(vals[-1] - t_max) > 1e-8:
            vals.append(float(t_max))

        out: list[float] = []
        seen = set()
        for v in vals:
            key = round(float(v), 10)
            if key not in seen:
                out.append(float(v))
                seen.add(key)
        return out

    def _target_buses(self) -> list[int]:
        power = self.env.power
        target_buses = list(getattr(self.env, "attack_target_buses", []))
        if not target_buses:
            target_buses = [
                power.bus_id_to_idx[int(power.gen_df.iloc[g]["Node"])]
                for g in self.env.gfpp_indices
            ]
        return [int(b) for b in target_buses]

    def _fdia_scales(self) -> list[float]:
        raw = getattr(self.env, "mip_nd_fdia_scales", [0.60, 0.75, 0.90, 1.00])
        if isinstance(raw, str):
            vals = [float(x.strip()) for x in raw.split(",") if x.strip()]
        else:
            vals = [float(x) for x in raw]
        vals = [float(np.clip(x, 1e-6, 1.0)) for x in vals]
        vals = sorted(set(round(x, 6) for x in vals))
        return vals or [1.0]

    def _base_load_np(self) -> tuple[np.ndarray, float]:
        base_load = self.env.power.base_load.detach().cpu().numpy().astype(float)
        positive = base_load[base_load > 0.0]
        mean_load = float(np.mean(positive)) if positive.size else 1.0
        return base_load, mean_load

    def _project_and_meta(self, za: np.ndarray, r_d_max: float) -> tuple[np.ndarray, dict]:
        fdia_gen = FDIAGenerator(self.env.power)
        legal, meta = fdia_gen.check_and_project_fdia(za, r_d_max)
        return legal.astype(np.float32), dict(meta)

    # ------------------------------------------------------------------
    # Upper MIP candidate model
    # ------------------------------------------------------------------
    def _timing_items(self) -> list[dict]:
        """Create the discrete timing alternatives used by the MIP binary variables."""
        t_fdia_grid = self._time_grid(
            float(getattr(self.env, "t_fdia_min", 1.0)),
            float(getattr(self.env, "t_fdia_max", 18.0)),
        )
        t_dos_grid = self._time_grid(
            float(getattr(self.env, "t_dos_min", 1.0)),
            float(getattr(self.env, "t_dos_max", 20.0)),
        )
        min_gap = float(getattr(self.env, "min_attack_gap", 1.0))

        items: list[dict] = []

        if self.mode == "fdia_only":
            center = float(
                getattr(
                    self.env,
                    "mip_nd_fdia_center",
                    0.5 * (min(t_fdia_grid) + max(t_fdia_grid)) if t_fdia_grid else 1.0,
                )
            )
            for tf in t_fdia_grid:
                # Prefer representative middle/early FDIA times, while still allowing no-good cuts to generate alternatives.
                score = 100.0 - 4.0 * abs(float(tf) - center)
                items.append({"T_fdia": float(tf), "T_dos": 99.0, "score": float(score)})
            return items

        if self.mode == "dos_only":
            center = float(
                getattr(
                    self.env,
                    "mip_nd_dos_center",
                    0.5 * (min(t_dos_grid) + max(t_dos_grid)) if t_dos_grid else 1.0,
                )
            )
            for td in t_dos_grid:
                score = 100.0 - 4.0 * abs(float(td) - center)
                items.append({"T_fdia": 99.0, "T_dos": float(td), "score": float(score)})
            return items

        # coordinated mode
        target_gap = float(getattr(self.env, "mip_nd_target_gap", 5.0))
        dos_center = float(
            getattr(
                self.env,
                "mip_nd_dos_center",
                0.5 * (min(t_dos_grid) + max(t_dos_grid)) if t_dos_grid else 1.0,
            )
        )
        fdia_center = float(getattr(self.env, "mip_nd_fdia_center", dos_center - target_gap))

        for tf in t_fdia_grid:
            for td in t_dos_grid:
                if td < tf + min_gap:
                    continue
                gap = float(td - tf)
                # Linear-surrogate timing preference.  The true response is still judged by the lower ND solver.
                score = (
                    120.0
                    - 8.0 * abs(gap - target_gap)
                    - 3.0 * abs(float(td) - dos_center)
                    - 1.0 * abs(float(tf) - fdia_center)
                    + 0.2 * float(td)
                )
                items.append({"T_fdia": float(tf), "T_dos": float(td), "score": float(score)})

        return items

    def _solve_one_mip(
        self,
        scale: float,
        timing_items: list[dict],
        excluded_item_ids: set[int],
    ) -> Optional[tuple[np.ndarray, dict, int]]:
        """
        Solve one CPLEX MIP candidate-generation problem.

        Variables:
            z_i     continuous FDIA load deviation on bus i
            y_j     binary timing-alternative selector

        Constraints:
            sum_i z_i = 0
            sum_j y_j = 1
            y_j = 0 for timing alternatives already excluded by no-good cuts
            bus-wise FDIA bounds
            target-bus FDIA direction constraints

        Objective:
            maximize target-bus FDIA drive + timing surrogate score

        Solver:
            IBM ILOG CPLEX through docplex.mp.model.Model.
        """
        try:
            from docplex.mp.model import Model
        except Exception as exc:
            msg = (
                "[MIP-ND] CPLEX/docplex is unavailable. Install IBM CPLEX runtime and docplex, "
                "for example: pip install docplex cplex. "
                f"Original import error: {exc}"
            )
            if bool(getattr(self.env, "mip_nd_allow_fallback", False)):
                if not self._mip_warning_printed:
                    print(msg)
                    print("[MIP-ND] Falling back to deterministic non-CPLEX candidate generation because env.mip_nd_allow_fallback=True.")
                    self._mip_warning_printed = True
                return None
            raise RuntimeError(msg) from exc

        n_bus = int(self.env.power.num_buses)
        n_items = len(timing_items)
        if n_items <= 0:
            return None

        base_load, mean_load = self._base_load_np()
        full_r_max = float(getattr(self.env, "fdia_max_ratio", 0.20))
        scaled_r_max = full_r_max * float(scale)

        # FDIA bounds. In DoS-only mode, FDIA is forced to zero.
        z_lb = np.zeros(n_bus, dtype=float)
        z_ub = np.zeros(n_bus, dtype=float)
        if self.mode != "dos_only":
            for i in range(n_bus):
                load_i = max(float(base_load[i]), 0.05 * mean_load)
                b = scaled_r_max * load_i
                z_lb[i] = -b
                z_ub[i] = b

        target_buses = self._target_buses()
        target_set = set(target_buses)
        base_scale = max(float(np.max(np.abs(base_load))) if base_load.size else 1.0, 1.0)

        fdia_weight = float(getattr(self.env, "mip_nd_fdia_weight", 1.0))
        timing_weight = float(getattr(self.env, "mip_nd_timing_weight", 1.0))
        time_limit = float(getattr(self.env, "mip_nd_time_limit", 10.0))
        mip_gap = float(getattr(self.env, "mip_nd_mip_gap", 0.0))

        timing_scores = np.array([float(x["score"]) for x in timing_items], dtype=float)
        if timing_scores.size > 0:
            timing_scores = timing_scores - float(np.min(timing_scores))
            denom = float(np.max(timing_scores)) + 1e-9
            timing_scores = timing_scores / denom * base_scale

        mdl = Model(name=f"mip_nd_{self.mode}_scale_{scale:.3f}")
        mdl.context.solver.log_output = bool(getattr(self.env, "mip_nd_cplex_log", False))

        if time_limit > 0:
            mdl.parameters.timelimit = time_limit
        if mip_gap >= 0:
            mdl.parameters.mip.tolerances.mipgap = mip_gap

        z = [
            mdl.continuous_var(lb=float(z_lb[i]), ub=float(z_ub[i]), name=f"z_{i}")
            for i in range(n_bus)
        ]
        y = [mdl.binary_var(name=f"y_{j}") for j in range(n_items)]

        # Stealthy zero-sum FDIA load deviation and one timing alternative.
        mdl.add_constraint(mdl.sum(z) == 0.0, ctname="fdia_zero_sum")
        mdl.add_constraint(mdl.sum(y) == 1.0, ctname="choose_one_timing")

        # No-good timing cuts from previous CPLEX solves.
        for j in excluded_item_ids:
            jj = int(j)
            if 0 <= jj < n_items:
                mdl.add_constraint(y[jj] == 0, ctname=f"nogood_timing_{jj}")

        # Target buses should carry nonnegative load-deviation direction in FDIA modes.
        if self.mode != "dos_only" and len(target_buses) > 0:
            for b in target_buses:
                mdl.add_constraint(z[int(b)] >= 0.0, ctname=f"target_nonnegative_{int(b)}")

        # Objective: target-bus FDIA drive + timing surrogate score.
        obj_terms = []
        if self.mode != "dos_only":
            for b in target_buses:
                obj_terms.append(fdia_weight * z[int(b)])
            # A small reward for negative compensation on non-target buses supports zero-sum stealth.
            for i in range(n_bus):
                if i not in target_set:
                    obj_terms.append(-0.02 * fdia_weight * z[i])

        for j in range(n_items):
            obj_terms.append(timing_weight * float(timing_scores[j]) * y[j])

        mdl.maximize(mdl.sum(obj_terms))

        try:
            sol = mdl.solve(log_output=bool(getattr(self.env, "mip_nd_cplex_log", False)))
        except Exception as exc:
            print(f"[MIP-ND] CPLEX solve failed at scale={scale:.2f}: {exc}")
            return None

        if sol is None:
            print(
                f"[MIP-ND] CPLEX did not return a candidate at scale={scale:.2f}. "
                f"status={mdl.solve_status}"
            )
            return None

        z_raw = np.array([float(sol.get_value(z[i])) for i in range(n_bus)], dtype=np.float32)
        y_val = np.array([float(sol.get_value(y[j])) for j in range(n_items)], dtype=float)
        item_id = int(np.argmax(y_val))

        # Safety: if the selected y was numerically excluded, reject.
        if item_id in excluded_item_ids:
            return None

        cplex_objective = float(sol.objective_value)

        if self.mode == "dos_only":
            za = np.zeros(n_bus, dtype=np.float32)
            meta: dict = {
                "fdia_source": "zero_for_dos_only",
                "mip_solver": "CPLEX/docplex",
                "cplex_objective": cplex_objective,
                "cplex_selected_item": int(item_id),
            }
        else:
            za, meta = self._project_and_meta(z_raw, full_r_max)
            meta.update(
                {
                    "fdia_source": "cplex_mip_master",
                    "fdia_scale": float(scale),
                    "mip_solver": "CPLEX/docplex",
                    "cplex_objective": cplex_objective,
                    "cplex_selected_item": int(item_id),
                    "cplex_solve_status": str(mdl.solve_status),
                    "cplex_mip_gap": mip_gap,
                    "cplex_time_limit": time_limit,
                }
            )

        mdl.end()
        return za, meta, item_id

    # ------------------------------------------------------------------
    # Fallback candidate generator
    # ------------------------------------------------------------------
    def _fallback_fdia_vector(self, scale: float = 1.0) -> tuple[np.ndarray, dict]:
        power = self.env.power
        n_bus = power.num_buses
        base_load, mean_load = self._base_load_np()
        full_r_max = float(getattr(self.env, "fdia_max_ratio", 0.20))
        scaled_r_max = full_r_max * float(scale)

        za = np.zeros(n_bus, dtype=np.float32)
        target_buses = self._target_buses()

        for b in target_buses:
            load_b = max(float(base_load[b]), 0.05 * mean_load)
            za[b] = scaled_r_max * load_b

        non_targets = [i for i in range(n_bus) if i not in target_buses]
        if non_targets:
            weights = base_load[non_targets].astype(float)
            if np.sum(weights) <= 1e-9:
                weights = np.ones(len(non_targets), dtype=float)
            weights = weights / np.sum(weights)
            total_pos = float(np.sum(za[target_buses])) if target_buses else 0.0
            for k, b in enumerate(non_targets):
                za[b] = -total_pos * weights[k]

        legal, meta = self._project_and_meta(za, full_r_max)
        meta.update({"fdia_source": "fallback_directional_zero_sum", "fdia_scale": float(scale)})
        return legal, meta

    def _fallback_generate_candidates(self, timing_items: list[dict], total_k: int) -> list[dict]:
        print("[MIP-ND] Using fallback Top-K candidate generation.")
        n_bus = self.env.power.num_buses
        timing_sorted = sorted(enumerate(timing_items), key=lambda x: x[1]["score"], reverse=True)

        candidates: list[dict] = []
        if self.mode == "dos_only":
            for item_id, item in timing_sorted[:total_k]:
                candidates.append(
                    {
                        "name": f"MIP-ND fallback DoS item={item_id}",
                        "FDIA": np.zeros(n_bus, dtype=np.float32),
                        "T_fdia": 99.0,
                        "T_dos": float(item["T_dos"]),
                        "za": np.zeros(n_bus, dtype=np.float32),
                        "meta": {"upper_source": "fallback_timing"},
                    }
                )
            return candidates

        scales = self._fdia_scales()
        per_scale = max(1, int(math.ceil(total_k / max(len(scales), 1))))
        for scale in scales:
            za, meta = self._fallback_fdia_vector(scale=scale)
            for item_id, item in timing_sorted[:per_scale]:
                candidates.append(
                    {
                        "name": f"MIP-ND fallback scale={scale:.2f} item={item_id}",
                        "FDIA": za,
                        "T_fdia": float(item["T_fdia"]),
                        "T_dos": float(item["T_dos"]),
                        "za": za,
                        "meta": dict(meta, upper_source="fallback_timing"),
                    }
                )
                if len(candidates) >= total_k:
                    break
            if len(candidates) >= total_k:
                break
        return candidates

    def _deduplicate_candidates(self, candidates: list[dict]) -> list[dict]:
        out: list[dict] = []
        seen = set()
        for c in candidates:
            za = np.asarray(c.get("za", c.get("FDIA")), dtype=float)
            # Avoid exact duplicates while still allowing different FDIA scales/timing combinations.
            key = (
                round(float(c.get("T_fdia", 99.0)), 6),
                round(float(c.get("T_dos", 99.0)), 6),
                tuple(np.round(za, 6).tolist()),
            )
            if key not in seen:
                out.append(c)
                seen.add(key)
        return out

    def _generate_candidates(self) -> list[dict]:
        timing_items = self._timing_items()
        if len(timing_items) == 0:
            return []

        total_k = int(getattr(self.env, "mip_nd_top_k", getattr(self.env, "mp_nd_max_fdia_vectors", 32)))
        total_k = max(1, total_k)

        candidates: list[dict] = []

        if self.mode == "dos_only":
            scales = [1.0]
        else:
            scales = self._fdia_scales()

        per_scale_k = max(1, int(math.ceil(total_k / max(len(scales), 1))))

        for scale in scales:
            excluded: set[int] = set()
            local_count = 0

            while local_count < per_scale_k and len(excluded) < len(timing_items) and len(candidates) < total_k:
                solved = self._solve_one_mip(scale=scale, timing_items=timing_items, excluded_item_ids=excluded)
                if solved is None:
                    # If the MIP is unavailable or fails, fall back for the full set.
                    if len(candidates) == 0:
                        return self._deduplicate_candidates(self._fallback_generate_candidates(timing_items, total_k))
                    break

                za, meta, item_id = solved
                excluded.add(int(item_id))
                item = timing_items[int(item_id)]
                candidates.append(
                    {
                        "name": (
                            f"MIP-ND {self.mode} scale={scale:.2f} "
                            f"tf={float(item['T_fdia']):.2f} td={float(item['T_dos']):.2f}"
                        ),
                        "FDIA": za,
                        "T_fdia": float(item["T_fdia"]),
                        "T_dos": float(item["T_dos"]),
                        "za": za,
                        "meta": dict(meta, upper_source="mip_master", timing_score=float(item["score"])),
                    }
                )
                local_count += 1

        candidates = self._deduplicate_candidates(candidates)
        if len(candidates) == 0:
            candidates = self._deduplicate_candidates(self._fallback_generate_candidates(timing_items, total_k))

        print(
            f"[MIP-ND] mode={self.mode} | generated_candidates={len(candidates)} | "
            f"timing_items={len(timing_items)} | scales={scales}"
        )
        return candidates

    # ------------------------------------------------------------------
    # Lower-response scoring and optimization loop
    # ------------------------------------------------------------------
    def _score_batch(self, result, offset: int) -> list[Tuple[float, int, dict]]:
        damage = np.asarray(result.damages, dtype=float)
        min_lp = np.asarray(result.metrics["min_lp_pct"], dtype=float)
        dos_lp = np.asarray(result.metrics["dos_lp_at_trigger"], dtype=float)

        lp_rep = float(getattr(self.env, "lp_replenish_ratio", 0.60)) * 100.0
        lp_trip = float(getattr(self.env, "lp_trip_ratio", 0.30)) * 100.0

        post_drop = np.asarray(
            result.metrics.get("post_dos_drop_pct", np.maximum(0.0, dos_lp - min_lp)),
            dtype=float,
        )
        post_drop = np.where(post_drop < 0.0, 0.0, post_drop)

        min_lp_after_dos = np.asarray(
            result.metrics.get("min_lp_after_dos_pct", min_lp),
            dtype=float,
        )

        scores: list[Tuple[float, int, dict]] = []

        for i in range(len(damage)):
            global_idx = offset + i
            info = {
                "damage": float(damage[i]),
                "min_lp_pct": float(min_lp[i]),
                "dos_lp_at_trigger": float(dos_lp[i]),
                "post_dos_drop_pct": float(post_drop[i]),
                "min_lp_after_dos_pct": float(min_lp_after_dos[i]),
            }

            if self.mode == "coordinated":
                # QSS and strict-steady models must not use the same LP-window
                # scoring semantics.  In the QSS model, LP(T_dos) is a dynamic
                # state and the 30%-60% coordination window is meaningful.  In
                # the strict steady model, the reported LP percentage is only a
                # pressure-derived equivalent-inventory diagnostic.  Therefore,
                # steady-model optimization maximizes attack consequence and
                # uses the diagnostic only as a tie-breaker, never as a hard
                # dynamic window constraint.
                steady_objective = bool(getattr(self.env, "mip_nd_steady_objective", False))

                if steady_objective:
                    info["window_pass"] = None
                    info["steady_objective"] = True
                    steady_damage_weight = float(
                        getattr(self.env, "steady_damage_weight", 1000.0)
                    )
                    steady_stress_weight = float(
                        getattr(self.env, "steady_stress_weight", 5.0)
                    )
                    score = (
                        damage[i] * steady_damage_weight
                        + max(0.0, 100.0 - min_lp[i]) * steady_stress_weight
                        + max(0.0, post_drop[i])
                    )
                else:
                    window_pass = (dos_lp[i] > lp_trip) and (dos_lp[i] <= lp_rep)
                    info["window_pass"] = bool(window_pass)

                    hard_window = bool(getattr(self.env, "mip_nd_hard_window", False))
                    if not window_pass:
                        if hard_window:
                            score = -np.inf
                        else:
                            early_gap = max(0.0, dos_lp[i] - lp_rep)
                            late_gap = max(0.0, lp_trip - dos_lp[i])
                            score = -1.0e8 - (early_gap + late_gap) * 1.0e5 + damage[i] * 0.1
                    else:
                        target = lp_rep - float(getattr(self.env, "dos_window_target_margin", 10.0))
                        score = (
                            damage[i] * float(getattr(self.env, "coordinated_damage_weight", 50.0))
                            + post_drop[i] * float(getattr(self.env, "post_dos_drop_weight", 1500.0))
                            + max(0.0, 100.0 - min_lp[i]) * float(getattr(self.env, "coordinated_min_lp_weight", 0.1))
                            - abs(dos_lp[i] - target) * 20.0
                        )
                        score += max(0.0, lp_trip - min_lp_after_dos[i]) * 200.0

            elif self.mode == "fdia_only":
                margin = float(getattr(self.env, "single_attack_trip_margin_pct", 3.0))
                safe_trip = lp_trip + margin
                if damage[i] > 1e-6 or min_lp[i] <= safe_trip:
                    score = -1.0e8 - damage[i] * 10.0 - max(0.0, safe_trip - min_lp[i]) * 1.0e5
                else:
                    target_lp = float(getattr(self.env, "single_attack_target_lp", 45.0))
                    score = 1000.0 - abs(min_lp[i] - target_lp) * 10.0

            else:  # dos_only
                margin = float(getattr(self.env, "single_attack_trip_margin_pct", 3.0))
                safe_trip = lp_trip + margin
                if damage[i] > 1e-6 or min_lp[i] <= safe_trip:
                    score = -1.0e8 - damage[i] * 10.0 - max(0.0, safe_trip - min_lp[i]) * 1.0e5
                else:
                    target_lp = float(getattr(self.env, "single_attack_target_lp", 45.0))
                    score = 1000.0 - abs(min_lp[i] - target_lp) * 10.0

            scores.append((float(score), global_idx, info))

        return scores

    def _strategy_response_cache_key(self, strategy: dict) -> tuple:
        za = np.asarray(
            strategy.get("za", strategy.get("FDIA")),
            dtype=np.float32,
        )
        za_key = np.round(za.astype(np.float64), 6).astype(np.float32).tobytes()
        return (
            round(float(strategy.get("T_fdia", 99.0)), 8),
            round(float(strategy.get("T_dos", 99.0)), 8),
            za_key,
        )

    def _cache_result_batch(self, strategies: list[dict], result) -> None:
        if not bool(getattr(self.env, "mip_nd_refinement_cache_enabled", True)):
            return
        damages = np.asarray(result.damages, dtype=float)
        metric_arrays = {
            key: np.asarray(values, dtype=float)
            for key, values in result.metrics.items()
        }
        for i, strategy in enumerate(strategies):
            payload = {
                "damage": float(damages[i]),
                "metrics": {
                    key: float(values[i])
                    for key, values in metric_arrays.items()
                },
            }
            self._response_cache[self._strategy_response_cache_key(strategy)] = payload

    def _result_from_cached_payloads(self, payloads: list[dict]):
        metric_keys: set[str] = set()
        for payload in payloads:
            metric_keys.update(payload.get("metrics", {}).keys())
        metrics = {
            key: np.asarray(
                [payload.get("metrics", {}).get(key, np.nan) for payload in payloads],
                dtype=float,
            )
            for key in sorted(metric_keys)
        }
        damages = np.asarray([payload["damage"] for payload in payloads], dtype=float)
        return SimpleNamespace(damages=damages, metrics=metrics, trajectories=[])

    def _evaluate_with_response_cache(
        self,
        initial_state: dict,
        strategies: list[dict],
        max_ode_steps: int,
        tolerance: float,
    ):
        """Evaluate only cache misses and rebuild results in original order.

        Cache keys include the full rounded FDIA vector and both attack times, so
        different FDIA scales never alias.  The cache is local to one optimizer
        instance and one physical initial state.
        """
        use_cache = bool(getattr(self.env, "mip_nd_refinement_cache_enabled", True))
        if not use_cache:
            return simulate_strategies_batched(
                self.env,
                initial_state,
                strategies,
                max_ode_steps=max_ode_steps,
                tolerance=tolerance,
                return_trajectories=False,
            )

        payloads: list[dict | None] = [None] * len(strategies)
        miss_positions: list[int] = []
        miss_strategies: list[dict] = []

        for i, strategy in enumerate(strategies):
            key = self._strategy_response_cache_key(strategy)
            cached = self._response_cache.get(key)
            if cached is None:
                self._cache_misses += 1
                miss_positions.append(i)
                miss_strategies.append(strategy)
            else:
                self._cache_hits += 1
                payloads[i] = cached

        if miss_strategies:
            miss_result = simulate_strategies_batched(
                self.env,
                initial_state,
                miss_strategies,
                max_ode_steps=max_ode_steps,
                tolerance=tolerance,
                return_trajectories=False,
            )
            self._cache_result_batch(miss_strategies, miss_result)
            for pos, strategy in zip(miss_positions, miss_strategies):
                payloads[pos] = self._response_cache[
                    self._strategy_response_cache_key(strategy)
                ]

        return self._result_from_cached_payloads([p for p in payloads if p is not None])

    def _build_batch_refinement_candidates(self, strategy: dict) -> list[dict]:
        """Build a local timing neighborhood around one coordinated strategy.

        The FDIA vector is kept unchanged.  Only T_fdia and T_dos are shifted
        on the same discrete time grid used by the upper MIP master.
        """
        if self.mode != "coordinated":
            return [dict(strategy)]

        step = max(
            float(getattr(self.env, "mip_nd_batch_refine_step", self.time_step)),
            1e-9,
        )
        radius = max(
            0,
            int(getattr(self.env, "mip_nd_batch_refine_radius", 1)),
        )

        t_fdia_min = float(getattr(self.env, "t_fdia_min", 1.0))
        t_fdia_max = float(getattr(self.env, "t_fdia_max", 18.0))
        t_dos_min = float(getattr(self.env, "t_dos_min", 1.0))
        t_dos_max = float(getattr(self.env, "t_dos_max", 20.0))
        min_gap = float(getattr(self.env, "min_attack_gap", 1.0))

        base_t_fdia = float(strategy.get("T_fdia", 99.0))
        base_t_dos = float(strategy.get("T_dos", 99.0))
        za = np.asarray(
            strategy.get("za", strategy.get("FDIA")),
            dtype=np.float32,
        )

        candidates: list[dict] = []
        seen: set[tuple[float, float]] = set()

        for i in range(-radius, radius + 1):
            for j in range(-radius, radius + 1):
                t_fdia = float(
                    np.clip(base_t_fdia + i * step, t_fdia_min, t_fdia_max)
                )
                t_dos = float(
                    np.clip(base_t_dos + j * step, t_dos_min, t_dos_max)
                )

                # Keep FDIA before DoS and satisfy the minimum attack gap.
                if t_dos < t_fdia + min_gap - 1e-9:
                    continue

                key = (round(t_fdia, 8), round(t_dos, 8))
                if key in seen:
                    continue
                seen.add(key)

                meta = dict(strategy.get("meta", {}))
                meta.update(
                    {
                        "batch_refinement_candidate": True,
                        "refine_fdia_offset": float(i * step),
                        "refine_dos_offset": float(j * step),
                    }
                )

                candidates.append(
                    {
                        "name": (
                            f"MIP-ND batch refine "
                            f"tf={t_fdia:.2f} td={t_dos:.2f}"
                        ),
                        "FDIA": za.copy(),
                        "za": za.copy(),
                        "T_fdia": t_fdia,
                        "T_dos": t_dos,
                        "meta": meta,
                    }
                )

        return candidates or [dict(strategy)]

    def _refine_batch_best(
        self,
        initial_state: dict,
        strategy: dict,
        base_score: float,
        base_info: dict,
        max_ode_steps: int,
        tolerance: float,
        batch_number: int,
    ) -> tuple[dict, float, dict, bool]:
        """Locally refine the best candidate of one evaluated batch.

        Returns:
            strategy, score, info, improved
        """
        if self.mode != "coordinated":
            return dict(strategy), float(base_score), dict(base_info), False

        local_candidates = self._build_batch_refinement_candidates(strategy)
        local_result = self._evaluate_with_response_cache(
            initial_state=initial_state,
            strategies=local_candidates,
            max_ode_steps=max_ode_steps,
            tolerance=tolerance,
        )
        local_scores = self._score_batch(local_result, offset=0)

        local_best_score = -np.inf
        local_best_idx: Optional[int] = None
        local_best_info: dict = {}

        for score, local_idx, info in local_scores:
            if np.isfinite(score) and score > local_best_score:
                local_best_score = float(score)
                local_best_idx = int(local_idx)
                local_best_info = dict(info)

        # No feasible local candidate under a hard-window configuration.
        if local_best_idx is None:
            return dict(strategy), float(base_score), dict(base_info), False

        # Keep the original batch winner unless refinement is strictly better.
        if np.isfinite(base_score) and local_best_score <= float(base_score) + 1e-9:
            return dict(strategy), float(base_score), dict(base_info), False

        refined = dict(local_candidates[local_best_idx])
        refined_meta = dict(refined.get("meta", {}))
        refined_meta.update(
            {
                "batch_refinement_applied": True,
                "batch_refinement_round": int(batch_number),
                "parent_T_fdia": float(strategy.get("T_fdia", 99.0)),
                "parent_T_dos": float(strategy.get("T_dos", 99.0)),
            }
        )
        refined["meta"] = refined_meta

        local_best_info["batch_refinement_applied"] = True
        local_best_info["batch_refinement_round"] = int(batch_number)

        return refined, float(local_best_score), local_best_info, True

    def _attach_final_meta(self, best: dict) -> dict:
        out = dict(best)
        n_bus = self.env.power.num_buses

        if self.mode == "dos_only":
            out["FDIA"] = np.zeros(n_bus, dtype=np.float32)
            out["za"] = np.zeros(n_bus, dtype=np.float32)
            out["T_fdia"] = 99.0
            out["meta"] = dict(out.get("meta", {}), fdia_final_checked=True)
            return out

        za = np.asarray(out.get("za", out.get("FDIA")), dtype=np.float32)
        full_r_max = float(getattr(self.env, "fdia_max_ratio", 0.20))
        legal, meta = self._project_and_meta(za, full_r_max)

        if np.linalg.norm(legal - za) > 1e-5:
            out["FDIA"] = legal
            out["za"] = legal
        else:
            out["FDIA"] = za
            out["za"] = za

        merged_meta = dict(out.get("meta", {}))
        merged_meta.update(meta)
        merged_meta["fdia_final_checked"] = True
        out["meta"] = merged_meta
        return out

    def optimize(self, initial_state: dict, max_ode_steps: int = 1000, tolerance: float = 1e-4) -> MIPNDResult:
        """Evaluate candidates in batches but record one convergence point per candidate.

        Computation remains batched for efficiency.  With ``chunk_size=8`` and
        128 generated candidates, the lower solver is still called on 16 raw
        batches, but ``metric_curve`` contains 128 points:

            candidate 1, 2, ..., 8   -> batch 1
            candidate 9, ..., 16     -> batch 2
            ...
            candidate 121, ..., 128  -> batch 16

        At the end of each batch, the batch winner is locally refined.  The
        refinement result replaces the last curve point of that batch (x=8,
        16, ..., 128), so local refinement is reflected without adding extra
        artificial points beyond the number of evaluated MIP candidates.
        """
        start = time.time()
        candidates = self._generate_candidates()

        if len(candidates) == 0:
            raise RuntimeError(f"MIP-ND generated no candidates for mode={self.mode}")

        print(
            f"[MIP-ND] evaluating mode={self.mode} | "
            f"candidates={len(candidates)} | chunk={self.chunk_size}"
        )

        best_score = -np.inf
        best_strategy: Optional[dict] = None
        best_info: dict = {}
        convergence: list[float] = []
        metric_curve: list[dict] = []

        total_batches = int(math.ceil(len(candidates) / self.chunk_size))

        for batch_number, start_idx in enumerate(
            range(0, len(candidates), self.chunk_size),
            start=1,
        ):
            end_idx = min(start_idx + self.chunk_size, len(candidates))
            chunk = candidates[start_idx:end_idx]

            # ----------------------------------------------------------
            # 1) Evaluate this chunk in one batched lower-level call.
            # ----------------------------------------------------------
            result = simulate_strategies_batched(
                self.env,
                initial_state,
                chunk,
                max_ode_steps=max_ode_steps,
                tolerance=tolerance,
                return_trajectories=False,
            )
            # Raw candidate responses are also reusable by local timing
            # refinement when the same strategy reappears in an overlapping
            # neighborhood.
            self._cache_result_batch(chunk, result)

            scores = self._score_batch(result, offset=start_idx)

            # ----------------------------------------------------------
            # 2) Scan the already-computed batch results candidate by candidate.
            #    This does NOT add lower-level solves; it only increases the
            #    convergence-curve recording resolution from one point per batch
            #    to one point per candidate.
            # ----------------------------------------------------------
            batch_best_score = -np.inf
            batch_best_idx: Optional[int] = None
            batch_best_info: dict = {}

            # Fallback anchor for hard-window cases where all candidates in the
            # batch are infeasible.  The least violating candidate is still used
            # as the center of the local timing neighborhood.
            fallback_idx: Optional[int] = None
            fallback_info: dict = {}
            fallback_key: Optional[tuple[float, float]] = None

            lp_rep = float(getattr(self.env, "lp_replenish_ratio", 0.60)) * 100.0
            lp_trip = float(getattr(self.env, "lp_trip_ratio", 0.30)) * 100.0

            for local_pos, (score, global_idx, info) in enumerate(scores, start=1):
                # Batch raw winner.
                if np.isfinite(score) and score > batch_best_score:
                    batch_best_score = float(score)
                    batch_best_idx = int(global_idx)
                    batch_best_info = dict(info)

                # Fallback local-refinement anchor for coordinated attacks.
                if self.mode == "coordinated":
                    dos_lp = float(info.get("dos_lp_at_trigger", -1.0))
                    gap = max(0.0, dos_lp - lp_rep) + max(0.0, lp_trip - dos_lp)
                    key = (float(gap), -float(info.get("damage", 0.0)))
                    if fallback_key is None or key < fallback_key:
                        fallback_key = key
                        fallback_idx = int(global_idx)
                        fallback_info = dict(info)

                # Update the global historical incumbent using this raw candidate.
                if np.isfinite(score) and score > best_score:
                    best_score = float(score)
                    best_strategy = dict(candidates[int(global_idx)])
                    best_info = dict(info)

                # Record exactly one curve point for this evaluated candidate.
                evaluated_count = int(start_idx + local_pos)
                convergence.append(float(best_score))

                row = {
                    "evaluated": evaluated_count,
                    "batch": int(batch_number),
                    "candidate_in_batch": int(local_pos),
                    "best_score": float(best_score),
                    "batch_local_refined": False,
                }

                if best_strategy is not None:
                    row["best_T_fdia"] = float(best_strategy.get("T_fdia", 99.0))
                    row["best_T_dos"] = float(best_strategy.get("T_dos", 99.0))

                if best_info:
                    row.update({f"best_{k}": v for k, v in best_info.items()})
                    if "damage" in best_info:
                        row["best_feasible_load_shedding"] = float(best_info["damage"])

                metric_curve.append(row)

            # ----------------------------------------------------------
            # 3) Select and locally refine the winner of this batch.
            # ----------------------------------------------------------
            if batch_best_idx is not None:
                batch_strategy = dict(candidates[batch_best_idx])
                batch_score = float(batch_best_score)
                batch_info = dict(batch_best_info)
            elif fallback_idx is not None:
                batch_strategy = dict(candidates[fallback_idx])
                batch_score = -np.inf
                batch_info = dict(fallback_info)
            else:
                batch_strategy = None
                batch_score = -np.inf
                batch_info = {}

            refined_this_batch = False
            if self.mode == "coordinated" and batch_strategy is not None:
                (
                    batch_strategy,
                    batch_score,
                    batch_info,
                    refined_this_batch,
                ) = self._refine_batch_best(
                    initial_state=initial_state,
                    strategy=batch_strategy,
                    base_score=batch_score,
                    base_info=batch_info,
                    max_ode_steps=max_ode_steps,
                    tolerance=tolerance,
                    batch_number=batch_number,
                )

            # ----------------------------------------------------------
            # 4) Refined batch winner competes with the global incumbent.
            # ----------------------------------------------------------
            if batch_strategy is not None and batch_score > best_score:
                best_score = float(batch_score)
                best_strategy = dict(batch_strategy)
                best_info = dict(batch_info)

            # ----------------------------------------------------------
            # 5) Reflect batch refinement at the last candidate point of this
            #    batch, rather than appending an extra x=129,130,... point.
            # ----------------------------------------------------------
            if metric_curve:
                final_row = {
                    "evaluated": int(end_idx),
                    "batch": int(batch_number),
                    "candidate_in_batch": int(end_idx - start_idx),
                    "best_score": float(best_score),
                    "batch_local_refined": bool(refined_this_batch),
                }

                if best_strategy is not None:
                    final_row["best_T_fdia"] = float(best_strategy.get("T_fdia", 99.0))
                    final_row["best_T_dos"] = float(best_strategy.get("T_dos", 99.0))

                if best_info:
                    final_row.update({f"best_{k}": v for k, v in best_info.items()})
                    if "damage" in best_info:
                        final_row["best_feasible_load_shedding"] = float(best_info["damage"])

                metric_curve[-1] = final_row
                convergence[-1] = float(best_score)

            refine_text = " | refined" if refined_this_batch else ""
            print(
                f"[MIP-ND] evaluated {end_idx}/{len(candidates)} | "
                f"batch={batch_number}/{total_batches} | "
                f"best={best_score:.2f}{refine_text}"
            )

        if best_strategy is None:
            raise RuntimeError(
                f"MIP-ND failed to select a best candidate for mode={self.mode}"
            )

        best = self._attach_final_meta(best_strategy)
        best_meta = dict(best.get("meta", {}))
        best_meta.update({f"lower_{k}": v for k, v in best_info.items()})
        best["meta"] = best_meta

        elapsed = time.time() - start
        if bool(getattr(self.env, "mip_nd_refinement_cache_enabled", True)):
            total_cache_queries = self._cache_hits + self._cache_misses
            hit_rate = (
                100.0 * self._cache_hits / total_cache_queries
                if total_cache_queries > 0
                else 0.0
            )
            print(
                f"[MIP-ND cache] hits={self._cache_hits} | "
                f"misses={self._cache_misses} | hit_rate={hit_rate:.1f}%"
            )
        print(f"[MIP-ND] completed in {elapsed:.1f}s | best_score={best_score:.2f}")
        print(
            f"[MIP-ND] best strategy: "
            f"T_fdia={best.get('T_fdia', 99.0):.2f}h, "
            f"T_dos={best.get('T_dos', 99.0):.2f}h"
        )
        if best.get("meta"):
            print(f"[MIP-ND] best meta: {best['meta']}")

        return MIPNDResult(
            strategy=best,
            fitness=float(best_score),
            convergence=convergence,
            metric_curve=metric_curve,
        )


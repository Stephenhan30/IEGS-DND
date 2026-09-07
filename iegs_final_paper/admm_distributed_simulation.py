"""Simulation wrapper for the ADMM lower-level solver."""
from __future__ import annotations

from typing import Optional

from algorithm.admm_distributed_solver import StandardADMMEvaluationSolver
from simulation import SimulationResult
from simulation import simulate_strategies_batched as _shared_simulation


def simulate_strategies_admm_distributed_batched(
    iegs_env,
    initial_state: dict,
    strategies: list[dict],
    max_ode_steps: int = 2500,
    tolerance: float = 1e-4,
    return_trajectories: bool = True,
    solver: Optional[StandardADMMEvaluationSolver] = None,
) -> SimulationResult:
    """Evaluate strategies with the standard consensus-ADMM solver."""
    if not strategies:
        raise ValueError("strategies must not be empty")

    batch_size = len(strategies)
    if solver is None or solver.batch_size != batch_size:
        solver = StandardADMMEvaluationSolver(
            iegs_env,
            batch_size=batch_size,
            max_ode_steps=max_ode_steps,
            tolerance=tolerance,
        )

    return _shared_simulation(
        iegs_env=iegs_env,
        initial_state=initial_state,
        strategies=strategies,
        max_ode_steps=max_ode_steps,
        tolerance=tolerance,
        return_trajectories=return_trajectories,
        solver=solver,
    )


simulate_strategies_batched = simulate_strategies_admm_distributed_batched

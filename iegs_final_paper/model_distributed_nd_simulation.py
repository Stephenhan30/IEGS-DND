"""Simulation wrapper for the model-distributed neurodynamic solver."""
from __future__ import annotations

from typing import Optional

from algorithm.model_distributed_nd_solver import ModelDistributedNeurodynamicODESolver
from simulation import SimulationResult
from simulation import simulate_strategies_batched as _central_simulate_strategies_batched


def simulate_strategies_model_distributed_nd_batched(
    iegs_env,
    initial_state: dict,
    strategies: list[dict],
    max_ode_steps: int = 3500,
    tolerance: float = 1e-4,
    return_trajectories: bool = True,
    solver: Optional[ModelDistributedNeurodynamicODESolver] = None,
) -> SimulationResult:
    """Evaluate strategies with the model-distributed QSS ND lower solver."""
    if len(strategies) == 0:
        raise ValueError("strategies must not be empty")

    batch_size = len(strategies)
    if solver is None or solver.batch_size != batch_size:
        solver = ModelDistributedNeurodynamicODESolver(
            iegs_env,
            batch_size=batch_size,
            max_ode_steps=max_ode_steps,
            tolerance=tolerance,
        )

    return _central_simulate_strategies_batched(
        iegs_env=iegs_env,
        initial_state=initial_state,
        strategies=strategies,
        max_ode_steps=max_ode_steps,
        tolerance=tolerance,
        return_trajectories=return_trajectories,
        solver=solver,
    )


# Alias with the same name as the original simulation API.  This is convenient
simulate_strategies_batched = simulate_strategies_model_distributed_nd_batched

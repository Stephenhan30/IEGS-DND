"""Save comparison-method convergence traces without generating figures."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

import main as project_main
from admm_distributed_simulation import simulate_strategies_admm_distributed_batched
from aladin_distributed_simulation import simulate_strategies_aladin_distributed_batched


def _record_trace(
    env,
    env_snapshot: dict,
    initial_state_snapshot: dict,
    coord_strat: dict,
    coord_za: np.ndarray,
    out_dir: str,
    target_system: str,
    args,
    *,
    method: str,
    simulator: Callable,
    filename: str,
    iteration_column: str,
) -> None:
    """Replay one coordinated strategy and save the solver trace as CSV/JSON."""
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)

    requested_hour = getattr(args, "nd_record_hour", None)
    record_hour = (
        int(requested_hour)
        if requested_hour is not None
        else max(1, int(np.ceil(float(coord_strat.get("T_dos", 1.0)))) - 1)
    )

    project_main.restore_env_snapshot(env, env_snapshot)
    env.record_nd_convergence = True
    env.record_nd_hour = record_hour
    env.nd_convergence_trace = []

    strategy = [{
        "name": f"Coordinated {method} convergence probe",
        "FDIA": coord_za,
        "T_fdia": float(coord_strat["T_fdia"]),
        "T_dos": float(coord_strat["T_dos"]),
    }]
    initial_state = project_main.clone_state_dict(initial_state_snapshot)

    try:
        simulator(
            env,
            initial_state,
            strategy,
            max_ode_steps=args.max_ode_steps,
            tolerance=args.tolerance,
            return_trajectories=False,
        )
        trace = list(getattr(env, "nd_convergence_trace", []))
    finally:
        env.record_nd_convergence = False
        env.record_nd_hour = None

    if not trace:
        print(f"[Warning] No {method} convergence trace was recorded.")
        return

    df = pd.DataFrame(trace)
    csv_path = output / filename.format(system=target_system)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    last = df.iloc[-1].to_dict()
    summary = {
        "method": method,
        "physical_hour": record_hour,
        "samples": int(len(df)),
        "iterations": int(float(last.get(iteration_column, len(df)))),
        "stop_reason": str(last.get("stop_reason", "unknown")),
        "trace_csv": csv_path.name,
    }
    json_path = csv_path.with_name(csv_path.stem + "_summary.json")
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[{method} trace] hour={record_hour} | samples={len(df)} | "
        f"stop={summary['stop_reason']} | saved={csv_path}"
    )


def save_admm_convergence_results(
    env, env_snapshot, initial_state_snapshot, coord_strat, coord_za,
    out_dir, target_system, args,
) -> None:
    _record_trace(
        env, env_snapshot, initial_state_snapshot, coord_strat, coord_za,
        out_dir, target_system, args,
        method="MIP-ADMM",
        simulator=simulate_strategies_admm_distributed_batched,
        filename="admm_convergence_{system}.csv",
        iteration_column="admm_iteration",
    )


def save_aladin_convergence_results(
    env, env_snapshot, initial_state_snapshot, coord_strat, coord_za,
    out_dir, target_system, args,
) -> None:
    _record_trace(
        env, env_snapshot, initial_state_snapshot, coord_strat, coord_za,
        out_dir, target_system, args,
        method="MIP-ALADIN",
        simulator=simulate_strategies_aladin_distributed_batched,
        filename="aladin_convergence_{system}.csv",
        iteration_column="aladin_iteration",
    )

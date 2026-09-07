"""Validate the 12 gas-fired generator coupling records."""
from __future__ import annotations

from env_physics.iegs_coupled import IEGSSystem
from utils import load_system_data

EXPECTED_BUSES = [6, 8, 18, 19, 25, 49, 59, 61, 72, 73, 90, 99]
EXPECTED_GAS_NODES = [20, 78, 115, 9, 55, 126, 71, 46, 26, 28, 17, 45]


def main() -> None:
    data = load_system_data("118-135", "data/")
    env = IEGSSystem(data)
    coupling = data["coupling"]["gfpp"].sort_values("GFPP_ID")

    buses = [int(v) for v in coupling["Power_Bus"].tolist()]
    gas_nodes = [int(v) for v in coupling["Gas_Node"].tolist()]
    if len(env.gfpp_indices) != 12:
        raise AssertionError(f"GFPP count mismatch: {len(env.gfpp_indices)}")
    if buses != EXPECTED_BUSES:
        raise AssertionError(f"GFPP power buses mismatch: {buses}")
    if gas_nodes != EXPECTED_GAS_NODES:
        raise AssertionError(f"GFPP gas nodes mismatch: {gas_nodes}")

    print("GFPP coupling validation: PASS")
    print(f"GFPP count: {len(env.gfpp_indices)}")
    print(f"GFPP generator indices (0-based): {env.gfpp_indices}")
    print(f"GFPP power buses: {buses}")
    print(f"GFPP gas-node numbers (1-based): {gas_nodes}")
    print(f"GFPP gas-node names: {coupling['Gas_Node_Name'].tolist()}")
    print(f"GFPP gas-node internal indices: {[env.gfpp_to_gas_node[g] for g in env.gfpp_indices]}")


if __name__ == "__main__":
    main()

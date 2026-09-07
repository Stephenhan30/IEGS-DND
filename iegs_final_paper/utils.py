"""Data-loading utilities for the IEEE 118-Gas135 IEGS case."""
from __future__ import annotations

import os
import warnings
from typing import Iterable

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")


def safe_pad(values, target_len: int, default_val: float):
    clean = [x for x in values if pd.notnull(x)]
    if len(clean) >= target_len:
        return clean[:target_len]
    return clean + [default_val] * (target_len - len(clean))


def get_col(df: pd.DataFrame, keywords: Iterable[str]) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=float)
    keys = [str(k).lower() for k in keywords]
    for key in keys:
        for col in df.columns:
            if pd.notnull(col) and key in str(col).lower():
                return df[col]
    return pd.Series([np.nan] * len(df))


def extract_table_from_stacked(df_raw: pd.DataFrame, required_keywords: Iterable[str]) -> pd.DataFrame:
    if df_raw is None or df_raw.empty:
        return pd.DataFrame()
    reqs = [str(k).lower() for k in required_keywords]
    for idx, row in df_raw.iterrows():
        row_strs = [str(x).lower().replace("\n", " ").strip() for x in row.values if pd.notnull(x)]
        if not row_strs:
            continue
        if all(any(req in val for val in row_strs) for req in reqs):
            df_sub = df_raw.iloc[idx + 1 :].copy().reset_index(drop=True)
            df_sub.columns = row.values
            empty_mask = df_sub.isnull().all(axis=1)
            if empty_mask.any():
                first_empty = int(np.argmax(empty_mask.values))
                if first_empty > 0:
                    df_sub = df_sub.iloc[:first_empty]
            return df_sub.dropna(how="all")
    return pd.DataFrame()


def get_sheet(xls: pd.ExcelFile, keywords: Iterable[str]) -> pd.DataFrame:
    for sheet in xls.sheet_names:
        if any(str(k).lower() in sheet.lower() for k in keywords):
            return pd.read_excel(xls, sheet_name=sheet).dropna(how="all")
    return pd.DataFrame()


def _require_file(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"缺少数据文件: {path}")


def load_system_data(target_system: str = "118-135", data_dir: str = "data/") -> dict:
    print(f"\n[Data Loader] 解析算例: IEEE {target_system} | 数据目录: {data_dir}")
    system_params = {"power": {}, "gas": {}, "wind": None, "coupling": {}}

    p_bus_df = p_branch_df = p_gen_df = p_load_df = p_coupling_df = pd.DataFrame()
    g_pipe_df = g_node_df = g_source_df = g_sink_df = g_innode_df = g_comp_df = pd.DataFrame()

    if target_system == "118-135":
        p_path = os.path.join(data_dir, "IEEE118_Completed_with_Coupling.xlsx")
        g_path = os.path.join(data_dir, "GasLib-135.xlsx")
        _require_file(p_path)
        _require_file(g_path)
        p_xls = pd.ExcelFile(p_path)
        g_xls = pd.ExcelFile(g_path)

        p_bus_df = get_sheet(p_xls, ["bus", "节点", "母线"])
        p_branch_df = get_sheet(p_xls, ["branch", "支路", "line"])
        p_gen_df = get_sheet(p_xls, ["gen", "发电机", "machine"])
        p_load_df = get_sheet(p_xls, ["load", "负荷"])
        p_coupling_df = get_sheet(p_xls, ["coupling", "耦合参数模板"])

        g_source_df = get_sheet(g_xls, ["气源节点_source", "source"])
        g_sink_df = get_sheet(g_xls, ["负荷节点_sink", "sink"])
        g_innode_df = get_sheet(g_xls, ["内部节点_innode", "innode"])
        g_node_df = g_innode_df
        g_pipe_df = get_sheet(g_xls, ["pipe", "管道"])
        g_comp_df = get_sheet(g_xls, ["compressor", "comp", "压缩"])

    else:
        raise ValueError("target_system 只能是 '118-135'")

    # ---------------- Power system ----------------
    bus_col_main = pd.to_numeric(get_col(p_bus_df, ["name", "bus", "node", "unnamed: 0"]), errors="coerce").dropna().astype(int).tolist()
    branch_f = pd.to_numeric(get_col(p_branch_df, ["from", "f"]), errors="coerce").dropna().astype(int).tolist()
    branch_t = pd.to_numeric(get_col(p_branch_df, ["to", "t"]), errors="coerce").dropna().astype(int).tolist()
    gen_b = pd.to_numeric(get_col(p_gen_df, ["bus", "node"]), errors="coerce").dropna().astype(int).tolist()
    load_b = pd.to_numeric(get_col(p_load_df, ["bus", "node"]), errors="coerce").dropna().astype(int).tolist() if not p_load_df.empty else []

    all_power_buses = sorted(set(bus_col_main + branch_f + branch_t + gen_b + load_b))
    if not all_power_buses:
        raise ValueError("无法从 Excel 中解析电力母线。请检查表头。")
    clean_bus_df = pd.DataFrame({"Node": all_power_buses, "load": 0.0})

    if not p_load_df.empty:
        load_buses = pd.to_numeric(get_col(p_load_df, ["bus", "node"]), errors="coerce")
        load_ps = pd.to_numeric(get_col(p_load_df, ["p_mw", "p ", "pd", "load"]), errors="coerce")
        for _, row in pd.DataFrame({"bus": load_buses, "p": load_ps}).dropna().iterrows():
            clean_bus_df.loc[clean_bus_df["Node"] == int(row["bus"]), "load"] += float(row["p"])
    else:
        bus_buses = pd.to_numeric(get_col(p_bus_df, ["bus", "node", "name"]), errors="coerce")
        bus_loads = pd.to_numeric(get_col(p_bus_df, ["pd", "load", "p_mw", "mw"]), errors="coerce").fillna(0.0)
        for _, row in pd.DataFrame({"bus": bus_buses, "p": bus_loads}).dropna().iterrows():
            clean_bus_df.loc[clean_bus_df["Node"] == int(row["bus"]), "load"] += float(row["p"])
    system_params["power"]["bus"] = clean_bus_df

    clean_branch_df = pd.DataFrame(
        {
            "From": pd.to_numeric(get_col(p_branch_df, ["from", "f"]), errors="coerce"),
            "To": pd.to_numeric(get_col(p_branch_df, ["to", "t"]), errors="coerce"),
            "Reactance(p.u.)": pd.to_numeric(get_col(p_branch_df, ["x_", "reactance", "x", "x "]), errors="coerce"),
            "Capacity (MVA)": pd.to_numeric(get_col(p_branch_df, ["rate", "capacity", "limit"]), errors="coerce").fillna(9999.0),
        }
    ).dropna()
    clean_branch_df = clean_branch_df.astype({"From": int, "To": int})
    clean_branch_df["Reactance(p.u.)"] = clean_branch_df["Reactance(p.u.)"].clip(lower=1e-4)
    system_params["power"]["branch"] = clean_branch_df

    clean_gen_df = pd.DataFrame(
        {
            "Node": pd.to_numeric(get_col(p_gen_df, ["bus", "node"]), errors="coerce"),
            "P max(MW)": pd.to_numeric(get_col(p_gen_df, ["p_mw", "max", "pmax"]), errors="coerce"),
            "P min(MW)": pd.to_numeric(get_col(p_gen_df, ["pmin", "min"]), errors="coerce").fillna(0.0),
        }
    ).dropna()
    clean_gen_df = clean_gen_df.astype({"Node": int})

    # Preserve possible fuel/coupling columns if they exist in source sheet.
    for col in p_gen_df.columns:
        col_s = str(col).lower()
        if any(k in col_s for k in ["fuel", "type", "gfpp", "gas node", "gas_node"]):
            vals = p_gen_df[col].reset_index(drop=True).iloc[: len(clean_gen_df)].to_numpy()
            clean_gen_df[str(col)] = vals
    system_params["power"]["gen"] = clean_gen_df

    # ---------------- Explicit GFPP coupling template ----------------
    # The workbook is authoritative for the gas-fired generator set.  The old
    # fallback (first num_gens//3 generators) incorrectly turns 53 generators
    # into 17 GFPPs.  Gas_Node is a one-based external node number following
    # the GasLib workbook order: 6 sources, 99 sinks, then 30 internal nodes.
    if p_coupling_df.empty:
        raise ValueError(
            "缺少 Coupling_耦合参数模板，无法确定燃气机组。"
        )

    coupling = pd.DataFrame(
        {
            "GFPP_ID": pd.to_numeric(
                get_col(p_coupling_df, ["gfpp_id", "燃机编号"]),
                errors="coerce",
            ),
            "Power_Bus": pd.to_numeric(
                get_col(p_coupling_df, ["power_bus", "电网母线"]),
                errors="coerce",
            ),
            "Gas_Node": pd.to_numeric(
                get_col(p_coupling_df, ["gas_node", "气网节点"]),
                errors="coerce",
            ),
            "P_Max": pd.to_numeric(
                get_col(p_coupling_df, ["p_max", "最大发电功率"]),
                errors="coerce",
            ),
        }
    ).dropna()
    coupling = coupling.astype(
        {"GFPP_ID": int, "Power_Bus": int, "Gas_Node": int, "P_Max": float}
    ).sort_values("GFPP_ID").reset_index(drop=True)

    if len(coupling) != 12:
        raise ValueError(f"耦合模板应包含12台燃气机组，实际为{len(coupling)}台。")
    if coupling["GFPP_ID"].duplicated().any() or coupling["Power_Bus"].duplicated().any():
        raise ValueError("耦合模板包含重复的GFPP编号或电网母线。")

    gen_buses = clean_gen_df["Node"].astype(int)
    missing_buses = [b for b in coupling["Power_Bus"].tolist() if int(b) not in set(gen_buses)]
    if missing_buses:
        raise ValueError(f"耦合模板母线没有对应发电机: {missing_buses}")

    # ---------------- Gas system ----------------
    pipe_m = get_col(g_pipe_df, ["gnode m", "start", "from", "node 1", "f", "m"]).astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
    pipe_n = get_col(g_pipe_df, ["gnode n", "end", "to", "node 2", "t", "n"]).astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
    valid_pipe = (pipe_m != "nan") & (pipe_n != "nan") & (pipe_m != "") & (pipe_n != "")
    pipe_m, pipe_n = pipe_m[valid_pipe], pipe_n[valid_pipe]
    if pipe_m.empty:
        raise ValueError("无法从 Excel 中解析天然气管道。请检查表头。")

    all_gas_nodes = set(pipe_m.tolist() + pipe_n.tolist())
    if not g_comp_df.empty:
        comp_m = get_col(g_comp_df, ["inlet", "start", "from", "m"]).astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
        comp_n = get_col(g_comp_df, ["outlet", "end", "to", "n"]).astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
        all_gas_nodes.update([x for x in comp_m.tolist() + comp_n.tolist() if x and str(x).lower() != "nan"])
    all_gas_nodes_sorted = sorted(n for n in all_gas_nodes if n and str(n).lower() != "nan")
    node_str_to_int = {name: idx for idx, name in enumerate(all_gas_nodes_sorted)}
    num_gas_nodes = len(all_gas_nodes_sorted)

    def _clean_ids(df: pd.DataFrame, candidates: Iterable[str]) -> list[str]:
        return (
            get_col(df, candidates)
            .dropna()
            .astype(str)
            .str.replace(r"\.0$", "", regex=True)
            .str.strip()
            .tolist()
        )

    workbook_node_names = (
        _clean_ids(g_source_df, ["source_id"])
        + _clean_ids(g_sink_df, ["sink_id"])
        + _clean_ids(g_innode_df, ["node_id"])
    )
    if len(workbook_node_names) != 135:
        raise ValueError(
            f"GasLib工作簿应包含135个节点，实际解析为{len(workbook_node_names)}个。"
        )
    if len(set(workbook_node_names)) != 135:
        raise ValueError("GasLib节点名称存在重复。")
    unknown_names = [name for name in workbook_node_names if name not in node_str_to_int]
    if unknown_names:
        raise ValueError(f"GasLib节点未出现在管网拓扑中: {unknown_names[:10]}")

    gas_indices: list[int] = []
    gas_names: list[str] = []
    for external_id in coupling["Gas_Node"].astype(int).tolist():
        if not 1 <= external_id <= len(workbook_node_names):
            raise ValueError(f"耦合模板气网节点{external_id}不在1..135范围内。")
        node_name = workbook_node_names[external_id - 1]
        gas_names.append(node_name)
        gas_indices.append(int(node_str_to_int[node_name]))
    coupling["Gas_Node_Index"] = gas_indices
    coupling["Gas_Node_Name"] = gas_names
    system_params["coupling"]["gfpp"] = coupling

    node_load_list = pd.to_numeric(get_col(g_node_df, ["load"]), errors="coerce").dropna().tolist()
    node_pmin_list = pd.to_numeric(get_col(g_node_df, ["min"]), errors="coerce").dropna().tolist()
    node_pmax_list = pd.to_numeric(get_col(g_node_df, ["max"]), errors="coerce").dropna().tolist()
    system_params["gas"]["node"] = pd.DataFrame(
        {
            "node_id": [node_str_to_int[n] for n in all_gas_nodes_sorted],
            "load": safe_pad(node_load_list, num_gas_nodes, 1.0),
            "min pressure": safe_pad(node_pmin_list, num_gas_nodes, 15.0),
            "max pressure": safe_pad(node_pmax_list, num_gas_nodes, 100.0),
        }
    )

    system_params["gas"]["pipeline"] = pd.DataFrame(
        {
            "pipe_id": range(len(pipe_m)),
            "m": pipe_m.map(node_str_to_int),
            "n": pipe_n.map(node_str_to_int),
            "wmn": pd.to_numeric(get_col(g_pipe_df, ["wmn", "weymouth", "c_p"]), errors="coerce").fillna(1.0),
            "length": pd.to_numeric(get_col(g_pipe_df, ["length", "len"]), errors="coerce").fillna(10.0),
            "diameter": pd.to_numeric(get_col(g_pipe_df, ["diameter", "dia"]), errors="coerce").fillna(500.0),
        }
    ).dropna().astype({"pipe_id": int, "m": int, "n": int})

    if not g_source_df.empty:
        source_node_col = get_col(g_source_df, ["node", "injection", "id", "source", "bus"])
        source_nodes_raw = source_node_col.dropna().astype(str).str.replace(r"\.0$", "", regex=True).str.strip().tolist()
        mapped_sources = [node_str_to_int[sn] for sn in source_nodes_raw if sn in node_str_to_int]
        if mapped_sources:
            s_min_vals = pd.to_numeric(get_col(g_source_df, ["min"]), errors="coerce").fillna(0.0).tolist()
            s_max_vals = pd.to_numeric(get_col(g_source_df, ["max"]), errors="coerce").fillna(999999.0).tolist()
            system_params["gas"]["source"] = pd.DataFrame(
                {"node": mapped_sources, "S_min": safe_pad(s_min_vals, len(mapped_sources), 0.0), "S_max": safe_pad(s_max_vals, len(mapped_sources), 999999.0)}
            )
        else:
            system_params["gas"]["source"] = pd.DataFrame({"node": [0], "S_min": [0.0], "S_max": [999999.0]})
    else:
        system_params["gas"]["source"] = pd.DataFrame({"node": [0], "S_min": [0.0], "S_max": [999999.0]})

    if not g_comp_df.empty:
        system_params["gas"]["compressor"] = pd.DataFrame(
            {
                "m": get_col(g_comp_df, ["inlet", "start", "from", "m"]).astype(str).str.replace(r"\.0$", "", regex=True).str.strip().map(node_str_to_int),
                "n": get_col(g_comp_df, ["outlet", "end", "to", "n"]).astype(str).str.replace(r"\.0$", "", regex=True).str.strip().map(node_str_to_int),
                "c_min": pd.to_numeric(get_col(g_comp_df, ["min"]), errors="coerce").fillna(1.0),
                "c_max": pd.to_numeric(get_col(g_comp_df, ["max"]), errors="coerce").fillna(1.5),
            }
        ).dropna().astype({"m": int, "n": int})
    else:
        system_params["gas"]["compressor"] = pd.DataFrame(columns=["m", "n", "c_min", "c_max"])

    print(
        f"    ✅ 解析成功 [{target_system}]: 电网母线 {len(system_params['power']['bus'])} 个, "
        f"发电机 {len(system_params['power']['gen'])} 台 | 气网节点 {len(system_params['gas']['node'])} 个, "
        f"管道 {len(system_params['gas']['pipeline'])} 条, 气源 {len(system_params['gas']['source'])} 个。"
    )
    print(
        "    ✅ 耦合模板: GFPP 12 台 | "
        f"电网母线 {coupling['Power_Bus'].tolist()} | "
        f"气网节点编号 {coupling['Gas_Node'].tolist()}"
    )
    return system_params

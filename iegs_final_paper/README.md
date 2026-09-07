# IEGS FDIA–DoS 协同攻击实验代码

本项目用于 IEEE 118 节点电力系统与 GasLib-135 天然气系统的 FDIA–DoS 协同攻击仿真。上层采用 MIP 生成候选攻击策略，下层分别采用集中式神经动力学（CND）、分布式神经动力学（DND）、ADMM 和 ALADIN 进行评估。

## 1. 目录结构

```text
.
├── algorithm/                         # 下层求解器与 MIP 主问题
├── attacker/                          # FDIA 构造
├── data/                              # IEEE 118 与 GasLib-135 数据
├── env_physics/                       # 电网、气网及耦合系统模型
├── initial_state.py                   # 初始运行状态
├── simulation.py                      # 统一的 24 h 仿真流程
├── main.py                            # 公共实验流程与 DND 绘图代码
├── model_distributed_nd_simulation.py # DND 仿真入口
├── admm_distributed_simulation.py     # ADMM 仿真入口
├── aladin_distributed_simulation.py   # ALADIN 仿真入口
├── comparison_trace_export.py         # ADMM/ALADIN 收敛结果导出，不绘图
├── run_distributed_main.py            # 本文方法：MIP–DND
├── run_centralized_baseline.py        # 对比方法：MIP–CND
├── run_admm_baseline.py               # 对比方法：MIP–ADMM
├── run_aladin_baseline.py             # 对比方法：MIP–ALADIN
├── run_all_four_method_experiments.py # 依次运行四种方法并生成综合图
├── plot_four_method_bar.py            # 唯一的四方法综合绘图脚本
├── plot_utils.py                      # DND 与综合图的公共绘图设置
└── validate_gfpp_coupling.py          # 12 台燃气机组耦合校验
```

压缩包不包含结果目录、日志、缓存、临时文件或稳态对比代码。

## 2. 环境安装

建议使用 Python 3.10，并在独立虚拟环境中安装依赖：

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

MIP 主问题依赖 IBM ILOG CPLEX。安装 Python 包后，还需保证本机 CPLEX 许可证可用。

## 3. 数据校验

首次运行前检查 12 台燃气机组的电–气耦合关系：

```bash
python validate_gfpp_coupling.py
```

## 4. 运行方法

### 4.1 本文方法 MIP–DND

```bash
python run_distributed_main.py --system 118-135 --upper-method mip-nd
```

默认输出到 `results_distributed/118-135/mip-nd/`。DND 保留单方法物理轨迹、上层搜索和下层收敛图，同时输出综合绘图所需的 CSV/JSON 数据。

### 4.2 三种对比方法

```bash
python run_centralized_baseline.py --system 118-135 --upper-method mip-nd
python run_admm_baseline.py --system 118-135 --upper-method mip-nd
python run_aladin_baseline.py --system 118-135 --upper-method mip-nd
```

结果分别写入 `results_centralized/`、`results_admm/` 和 `results_aladin/`。三种对比方法不生成单方法图片，只保存计算结果与 `plot_four_method_bar.py` 所需数据：

- CND：`run_summary.json`、场景轨迹和汇总指标。
- ADMM：在上述结果基础上，保存 `admm_convergence_*.csv`。
- ALADIN：在上述结果基础上，保存 `aladin_convergence_*.csv`。
- ADMM 和 ALADIN 均保存 `upper_search_curve_*.csv`，用于综合收敛图和雷达图。

### 4.3 一次运行四种方法并生成综合图

```bash
python run_all_four_method_experiments.py --system 118-135 --upper-method mip-nd
```

脚本依次运行 CND、DND、ADMM 和 ALADIN，然后调用 `plot_four_method_bar.py`，将综合图片和对应 CSV 保存到 `results_method_compare/118-135/`。

仅运行四种方法而不生成综合图：

```bash
python run_all_four_method_experiments.py --no-plot --system 118-135 --upper-method mip-nd
```

已有四种方法结果时，可单独重新绘制综合图：

```bash
python plot_four_method_bar.py --system 118-135
```

## 5. 常用参数

查看公共参数：

```bash
python run_distributed_main.py --help
```

主要参数：

- `--max-ode-steps`：下层求解最大离散步数。
- `--tolerance`：下层停止容差。
- `--mip-top-k`：MIP 候选策略数量。
- `--mp-chunk-size`：候选策略批量评估大小。
- `--mip-fdia-scales`：FDIA 强度候选集合。
- `--nd-record-hour`：下层收敛轨迹记录时段；默认取 DoS 前一时段。
- `--seed`：随机种子。
- `--out-dir`：结果根目录。单独运行四方法脚本时，建议省略该参数以使用各方法的默认目录。

示例：

```bash
python run_distributed_main.py \
  --system 118-135 \
  --upper-method mip-nd \
  --max-ode-steps 2500 \
  --tolerance 1e-4 \
  --mip-top-k 128 \
  --mp-chunk-size 8 \
  --seed 42
```

## 6. 综合绘图所需文件

`plot_four_method_bar.py` 从四个标准结果目录读取以下文件：

```text
results_distributed/118-135/mip-nd/
├── run_summary.json
├── trajectories.csv
├── upper_search_curve_118-135.csv
└── lower_nd_convergence_118-135.csv

results_centralized/118-135/mip-nd/
├── run_summary.json
└── trajectories.csv

results_admm/118-135/mip-nd/
├── run_summary.json
├── trajectories.csv
├── upper_search_curve_118-135.csv
└── admm_convergence_118-135.csv

results_aladin/118-135/mip-nd/
├── run_summary.json
├── trajectories.csv
├── upper_search_curve_118-135.csv
└── aladin_convergence_118-135.csv
```

综合脚本生成：

- 四方法攻击效果与管存对比图。
- 四方法管存—切负荷轨迹图。
- DND、ADMM 和 ALADIN 的上层搜索与下层停止残差图。
- 三种分布式方法的通信数据量图。
- 三种分布式方法综合雷达图。

所有综合图仅保存为 PNG，并同步保存对应 CSV。

## 7. 输出管理

结果目录、日志、缓存和 Python 编译文件已写入 `.gitignore`。重新打包代码前可删除所有 `results*` 目录，不影响源代码运行。

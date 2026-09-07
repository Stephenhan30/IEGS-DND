# IEGS FDIA–DoS 协同攻击实验代码

本项目用于复现电–气综合能源系统（IEGS）中的 FDIA–DoS 协同攻击实验。主算例采用 **IEEE 118-bus / GasLib-135** 系统，上层采用 MIP 生成候选攻击策略，下层分别采用集中式神经动力学（CND）、分布式神经动力学（DND）、ADMM 和 ALADIN 进行系统响应评估。

当前代码还包括：

- 118–135 四方法对比实验；
- FDIA-only / DoS-only / Coordinated 消融实验；
- MIP 候选集大小敏感性实验；
- IEEE RTS-24 / Belgian-20 的 24–20 DND 扩展实验；
- 论文综合绘图与攻击机理验证图。

---

## 1. 目录结构

```text
.
├── algorithm/                         # MIP、DND、CND、ADMM、ALADIN 求解器
├── attacker/                          # FDIA 构造
├── env_physics/                       # 电网、气网与电-气耦合物理模型
├── data/
│   ├── IEEE118_Completed_with_Coupling.xlsx
│   ├── GasLib-135.xlsx
│   └── IEEE24_Belgian20.xlsx
│
├── initial_state.py                   # 初始状态构造
├── simulation.py                      # 统一 24 h 物理仿真
├── main.py                            # 118–135 公共实验流程与参数入口
├── model_distributed_nd_simulation.py # DND 下层仿真
├── admm_distributed_simulation.py     # ADMM 下层仿真
├── aladin_distributed_simulation.py   # ALADIN 下层仿真
├── comparison_trace_export.py         # ADMM/ALADIN 收敛轨迹导出
│
├── run_distributed_main.py            # 本文方法：MIP–DND
├── run_centralized_baseline.py        # 对比方法：MIP–CND
├── run_admm_baseline.py               # 对比方法：MIP–ADMM
├── run_aladin_baseline.py             # 对比方法：MIP–ALADIN
├── run_all_four_method_experiments.py # 顺序运行四方法
├── run_ablation_118_135.py            # FDIA / DoS 消融实验
├── run_dnd_24_20.py                   # 24–20 DND 扩展实验
├── run_mip_topk_sensitivity_1.py        # MIP 候选集大小敏感性
│
├── plot_four_method_bar.py            # 论文综合绘图脚本
├── plot_utils.py                      # 公共绘图工具
├── validate_gfpp_coupling.py          # GFPP 电-气耦合校验
├── requirements.txt
└── README.md
```

---

## 2. Python 环境

建议使用 **Python 3.10**。

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows：

```bash
.venv\Scripts\activate
```

安装依赖：

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

当前 `requirements.txt`：

```text
numpy==1.24.4
pandas==2.3.3
openpyxl==3.1.5
torch==2.10.0
cplex==22.1.2.1
docplex==2.32.264
matplotlib==3.10.7
```

MIP 上层默认使用 **IBM ILOG CPLEX**。除安装 Python 包外，还需要本机存在可用的 CPLEX 许可证。

仅做本地测试且 CPLEX 不可用时，可在支持的脚本后添加：

```bash
--mip-allow-fallback
```

正式论文实验建议使用 CPLEX 结果。

---

## 3. 运行前数据校验

118–135 主实验首次运行前，可检查 12 台燃气机组的电–气耦合关系：

```bash
python validate_gfpp_coupling.py
```

---

# 4. 118–135 主实验

## 4.1 本文方法：MIP–DND

```bash
python run_distributed_main.py
```

等价的显式写法：

```bash
python run_distributed_main.py --system 118-135 --upper-method mip-nd
```

默认结果目录：

```text
results_distributed/118-135/mip-nd/
```

主要输出包括：

```text
run_summary.json
trajectories.csv
upper_search_curve_118-135.csv
lower_nd_convergence_118-135.csv
```

在当前 Situation-2 版本中，DND `trajectories.csv` 还会保存攻击机理绘图所需的额外物理量，例如 GFPP 总出力、气源注入等。

---

## 4.2 CND、ADMM、ALADIN 对比实验

```bash
python run_centralized_baseline.py
python run_admm_baseline.py
python run_aladin_baseline.py
```

结果目录分别为：

```text
results_centralized/118-135/mip-nd/
results_admm/118-135/mip-nd/
results_aladin/118-135/mip-nd/
```

其中：

- CND 保存最终结果与物理轨迹；
- ADMM 额外保存 `admm_convergence_118-135.csv`；
- ALADIN 额外保存 `aladin_convergence_118-135.csv`；
- ADMM、ALADIN 均保存上层搜索曲线，供综合收敛图使用。

---

## 4.3 四种方法顺序运行

```bash
python run_all_four_method_experiments.py
```

该脚本依次运行：

```text
CND → DND → ADMM → ALADIN
```

如果只运行四种算法、不自动绘图：

```bash
python run_all_four_method_experiments.py --no-plot
```

由于完整四方法实验耗时较长，也可以分别运行四个脚本，全部完成后再单独执行绘图脚本。

---

# 5. 综合绘图

已有四方法结果后直接运行：

```bash
python plot_four_method_bar.py
```

默认读取：

```text
results_distributed/118-135/mip-nd/
results_centralized/118-135/mip-nd/
results_admm/118-135/mip-nd/
results_aladin/118-135/mip-nd/
results_ablation/118-135/mip-nd/
```

默认输出：

```text
results_method_compare/118-135/
```

当前论文版主要生成以下图片：

```text
load_shedding_comparison.png
linepack_trajectories.png
algorithm_performance.png
exchanged_data_volume.png
three_distributed_methods_radar_118-135.png
attack_ablation_linepack.png
attack_mechanism.png
```

同时保存与每张图对应的 CSV 数据。

### 当前版式

- `load_shedding_comparison.png`：单栏、纵向 2×1；
- `linepack_trajectories.png`：双栏、2×2；
- `algorithm_performance.png`：单栏、纵向收敛图；
- `exchanged_data_volume.png`：单栏；
- 雷达图：单栏；
- `attack_ablation_linepack.png`：单栏；
- `attack_mechanism.png`：双栏、横向 1×3。

当前版本**不生成** `spatial_damage_distribution.png`。

### 攻击机理图的数据要求

`attack_mechanism.png` 使用 DND 的 Baseline 与 Coordinated attack 轨迹，包含：

1. GFPP output；
2. gas-source injection；
3. total linepack。

如果绘图时提示缺少类似下面的字段：

```text
gfpp_output_mw
gas_source_total
```

说明当前 `results_distributed/.../trajectories.csv` 是旧结果。使用 Situation-2 的 `simulation.py` / DND 代码重新运行：

```bash
python run_distributed_main.py
```

然后重新绘图即可。

---

# 6. 消融实验

消融实验固定同一个最优 Coordinated strategy，仅移除不同攻击阶段，用于比较：

```text
Baseline
FDIA-only
DoS-only
Coordinated
```

运行：

```bash
python run_ablation_118_135.py
```

结果目录：

```text
results_ablation/118-135/mip-nd/
```

主要文件：

```text
attack_ablation_summary.csv
attack_ablation_summary.json
attack_ablation_trajectories.csv
attack_ablation_118-135.png
search_curve_coordinated.csv
search_curve_fdia_only.csv
search_curve_dos_only.csv
```

综合绘图脚本还会基于这些结果生成：

```text
attack_ablation_linepack.png
```

该图比较 Baseline、FDIA-only、DoS-only 和 Coordinated 四种情形下的管存变化。

---

# 7. 24–20 扩展实验

24–20 算例使用：

```text
data/IEEE24_Belgian20.xlsx
```

该实验由单独脚本运行，不修改 118–135 主实验入口：

```bash
python run_dnd_24_20.py
```

脚本内部会自动设置：

```text
system = 24-20
out_dir = results_distributed
```

默认结果目录：

```text
results_distributed/24-20/mip-nd/
```

24–20 当前版本包含 GFPP trip 与管存恢复逻辑：当管存触发保护阈值后，GFPP 离线，并通过气侧保护控制恢复管存；保护采用锁存策略，在该 24 h 仿真周期内不自动重启 GFPP。

---

# 8. MIP 候选集大小敏感性

论文当前采用的候选集敏感性结果为：

```text
K = 32, 64, 128, 256
```

运行脚本：

```bash
python run_mip_topk_sensitivity_1.py
```

论文中只使用以下候选集敏感性结果目录与汇总文件：

```text
results_mip_topk_sensitivity_1/
results_mip_topk_sensitivity_1/mip_topk_sensitivity_summary.csv
```

本实验考察的候选集大小为 **K = 32、64、128、256**。实验定义以 `run_mip_topk_sensitivity_1.py` 中的
`K_VALUES = (32, 64, 128, 256)` 为准。该汇总文件用于记录不同 K 下的攻击效果、管存指标、攻击时刻及相关统计结果。

> 注意：不使用 `results_mip_topk_sensitivity/` 目录中的另一套候选集敏感性结果。

---

# 9. 常用参数

查看完整参数：

```bash
python run_distributed_main.py --help
```

主实验常用参数：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `--max-ode-steps` | 2500 | DND 最大离散步数 |
| `--tolerance` | 1e-4 | ND 基础停止容差 |
| `--mip-top-k` | 128 | MIP 上层候选策略数量 |
| `--mp-chunk-size` | 8 | 下层批量评估候选数量 |
| `--mip-fdia-scales` | 0.6,0.75,0.9,1.0 | FDIA 候选预算比例 |
| `--mip-time-limit` | 10 s | 每次 CPLEX MIP 求解时间上限 |
| `--mip-gap` | 0 | CPLEX 相对 MIP gap |
| `--mdnd-consensus-rel-tolerance` | 1e-3 | DND GFPP 共识相对残差阈值 |
| `--mdnd-state-rel-tolerance` | 1e-3 | DND 状态相对残差阈值 |
| `--mdnd-convergence-patience` | 5 | 连续满足停止条件次数 |
| `--nd-record-hour` | 自动 | 下层收敛轨迹记录时段 |
| `--seed` | 42 | 随机种子 |

示例：

```bash
python run_distributed_main.py \
  --max-ode-steps 2500 \
  --tolerance 1e-4 \
  --mip-top-k 128 \
  --mp-chunk-size 8 \
  --seed 42
```

---

# 10. 后台运行

Linux 服务器上推荐使用 `nohup`。

### DND

```bash
nohup python -u run_distributed_main.py > dnd.log 2>&1 &
```

### CND

```bash
nohup python -u run_centralized_baseline.py > cnd.log 2>&1 &
```

### ADMM

```bash
nohup python -u run_admm_baseline.py > admm.log 2>&1 &
```

### ALADIN

```bash
nohup python -u run_aladin_baseline.py > aladin.log 2>&1 &
```

### 24–20

```bash
nohup python -u run_dnd_24_20.py > 24_20.log 2>&1 &
```

### 消融实验

```bash
nohup python -u run_ablation_118_135.py > ablation.log 2>&1 &
```

### 候选集敏感性

```bash
nohup python -u run_mip_topk_sensitivity_1.py > mip_topk_sensitivity.log 2>&1 &
```

查看运行状态：

```bash
ps -ef | grep python
```

实时查看日志：

```bash
tail -f dnd.log
```

---

# 11. 推荐复现实验顺序

如果需要从零复现论文中的主要结果，建议按下面顺序运行：

```bash
python validate_gfpp_coupling.py

python run_distributed_main.py
python run_centralized_baseline.py
python run_admm_baseline.py
python run_aladin_baseline.py

python run_ablation_118_135.py

python plot_four_method_bar.py
```

扩展实验再分别运行：

```bash
python run_dnd_24_20.py
python run_mip_topk_sensitivity_1.py
```

---

# 12. 结果清理

`.gitignore` 默认忽略：

```text
__pycache__/
*.py[cod]
*.log
results*/
paper_figures_custom/
```

因此结果文件、日志和 Python 缓存不会进入 Git。

如果需要重新打包纯代码，可删除：

```text
results*/
*.log
__pycache__/
```

删除这些文件不会影响源代码运行。

---

## 最简运行命令汇总

```bash
# 118-135 四方法
python run_distributed_main.py
python run_centralized_baseline.py
python run_admm_baseline.py
python run_aladin_baseline.py

# 消融实验
python run_ablation_118_135.py

# 综合绘图
python plot_four_method_bar.py

# 24-20 DND
python run_dnd_24_20.py

# 候选集敏感性实验
python run_mip_topk_sensitivity_1.py
```

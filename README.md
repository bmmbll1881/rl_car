# rl_autodrive — 可调试的强化学习自动驾驶训练与评测方案

一套**零重型依赖**（只需要 `python` + `numpy`，出图/出视频只用 `Pillow`）的自动驾驶强化学习训练/评测框架：
自带 2D 测试环境 **MiniDrive**（含**动态随机交通**）、可插拔算法接口（已实现 **DQN** 与 **PPO**）、统一评估协议、
CSV 指标日志、学习曲线、**全过程行驶视频**，以及数值梯度检验的单元测试。

> 设计取向：**可调试性优先**。所有梯度都是手写的（约 100 行 `nn.py`），
> 每个 episode / 每一步都能被完整回放和检查，一次 25 万步训练在普通笔记本上 70 秒跑完，
> 所以"改一个超参 → 看曲线"的循环足够快。

## 0. 实测结果（开箱即用的结论）

> **当前版本（本轮）**：3 车道 / 24 辆 NPC / **无对抗车**；自车 **v_max 20 m/s**；
> 任务目标 = **最快且不撞车地跑完一圈**（奖励以时间成本为主）。

**只想跑起来、不想读代码**：看 **[docs/独立运行手册.docx](docs/独立运行手册.docx)**（训练/评估/出图/出视频/出报告的完整命令手册）。

详细方案与实验报告见 **[docs/强化学习自动驾驶训练方案与实验报告-时间目标版.docx](docs/强化学习自动驾驶训练方案与实验报告-时间目标版.docx)**。

协议：同一环境 / 同一 7 动作 / 同一 **25 万步**预算 / 3 个随机种子；**验证集**（种子 10000–10004）选 checkpoint，
**独立测试集**（种子 20000–20009，10 episodes）报数。

**有交通（24 辆 NPC，5.5–8 m/s）**

| 指标 | DQN | **PPO** | RARL（风险感知） | 脚本参考驾驶员 | 随机 |
|---|---|---|---|---|---|
| 成功率 | 0.43 ± 0.21 | **0.63 ± 0.05** | 0.60 ± 0.24 | 0.80 | 0.00 |
| 成功圈速 (s) | 75.6 ± 5.1 | **73.4 ± 1.1** | 77.7 ± 1.7 | 55.0 | — |
| 碰撞率 | 0.33 | 0.37 | **0.33** | 0.20 | 1.00 |
| 平均车速 (m/s) | 5.90 | 6.15 | 5.79 | 8.48 | — |
| 横向 RMS (m) | 2.77 | 1.60 | 1.45 | 2.33 | — |
| 首次成功率≥0.8 (步) | 未达到 | **90k** | 240k | — | — |
| 吞吐 (步/秒) / 单次训练 | 2,158 / 116 s | 2,065 / 121 s | 1,050 / 238 s | — | — |

一句话：**三种算法都学会了"安全完赛"（0.43–0.63），但都没跑赢脚本专家（55 s）**——撞车重罚（−150）让 RL 收敛到
"慢而稳"：车速 5.8–6.2 m/s，专家 8.5 m/s。PPO 最稳（成功率标准差 0.05）；RARL 把风险权重压在**车辆**风险上之后
碰撞率最低（0.33）、会走侧车道超车，但圈速仍慢 4.3 s。**两次真实的排错**：(1) 最初"每步 −1、撞车 −200"让策略学会
尽早撞车（0 成功率），改成"每步 −0.1、撞车 −150"后恢复；(2) 风险权重原本压在"总风险"上（2/3 是路缘项，最小值在
道路中心线），策略被驯化成守中线的龟速车，改压车辆风险后侧车道使用率 3%→24%、成功率 0.47→0.60——详见报告 §4.3、§5.3。

**RARL 扩展实现状态（2026-11-20）**：`TTCCollisionRisk`、动作条件风险、PPO 动作掩码、
`lane_aware` 车道特征、`overtake_lock` 战术锁定以及 `reset_episode` / `observe_transition`
评估钩子均已实现，**全部默认关闭**，默认策略输入仍是 48 维、旧检查点可直接加载。
超车闭环的显式配置是 `configs/rarl_overtake.json`（72 维输入），已跑通 4k / 20k 步训练冒烟、
独立评估与超车诊断；**20k 步没有学会开车（成功率 0），因此这里不报告任何性能结论**，
命令与实测数字见 §3“超车训练闭环”。后续的**超车 curriculum**（慢速交通 + 初始前车，
`configs/env_overtake_stage1.json` / `stage2.json`）把“追上被挡前车”变成 10/10 episode 都会发生的事件，
并在 stage 1 累计训练 270k 步：评估中 85% 的被挡步确实在缩短间距、完成 23 次计划变道，
但**成功率仍为 0、真正超过前车 0 次**，详见 §3“超车 curriculum”。

主要产物：
* 视频（早期版本，10 fps 实时，各一圈）：`figs/video_ppo_lap.avi`、`figs/video_dqn_lap.avi`
* 三车道环境图：`figs/track_map.png`（赛道 + 车道线 + 24 辆车）
* 图表：`curve_eval_success.png`、`bar_final_metrics.png`、`trace_gap.png`（与前车距离）、
  `trace_speed.png`、`action_usage.png`、`ppo_instability.png`
* 对比表：`figs/summary.md`（有交通）、`figs_empty/summary.md`（空赛道）

## 1. 快速开始

> 本机（以及任何没有系统 Python 的机器）**不需要安装 Python、不需要激活环境，也不依赖 Codex**：
> 项目自带 `runtime\python`（CPython 3.12.14 + numpy + Pillow），并已创建 `.venv`。
> 用 `run.cmd` 执行即可（见 §1.0）。

独立运行验证（2026-11-20）：在普通 Windows 进程中直接使用项目内 `run.cmd` 完成了
47 项单元测试、最小训练冒烟、三模型集成评估、整圈视频渲染和逐帧 AVI 校验；
源码中没有引用 Codex 解释器或缓存路径。

### 1.0 独立运行环境（不依赖 DeepSeek Harness）

| 路径 | 内容 | 体积 |
|---|---|---|
| `runtime\python\` | 随项目携带的 **CPython 3.12.14**，其 `Lib\site-packages` 内已带 numpy 2.3.5 / Pillow 12.3.0 | ~103 MB |
| `.venv\` | 由 `runtime\python` 创建的虚拟环境（`--system-site-packages`），`pyvenv.cfg` 只指向项目内的 runtime | ~11 MB |
| `run.cmd` / `run.ps1` | 用 `.venv` 的解释器执行任意脚本，**无需激活** | — |
| `scripts\setup_venv.ps1` | 创建／重建（`-Rebuild`）虚拟环境；离线也能用 | — |

```powershell
# 推荐：一条命令跑任意脚本（批处理不受执行策略限制）
.\run.cmd train.py --algo ppo --seed 0 --steps 250000
.\run.cmd aggregate.py
.\run.cmd -m unittest discover -s tests

# 或者直接调解释器（路径无空格，任何终端都行）
.\.venv\Scripts\python.exe train.py --algo ppo --seed 0 --steps 250000

# 想用 activate 也行（.ps1 受执行策略限制时需要先放开）
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1

# 换机器 / 移动目录 / 环境损坏时重建（离线，基于项目内 runtime）
powershell -ExecutionPolicy Bypass -File scripts\setup_venv.ps1 -Rebuild
```

要点：

* `sys.path` 里**没有任何 DSH 路径**，`python -c "import sys;print(sys.base_prefix)"` 返回项目内的 `runtime\python`；
* 只依赖 numpy + Pillow，没有 torch / gym / matplotlib，也不需要 GPU；
* 若目录被移动，`.venv\pyvenv.cfg` 里的绝对路径会失效 → 跑一次 `setup_venv.ps1 -Rebuild` 即可（10 秒内，离线）；
* `runtime/` 与 `.venv/` 已写入 `.gitignore`，不会进版本库。

### 1.1 常用命令

```bash
# 单元测试（梯度检验 + 环境/交通用例）
run.cmd -m unittest discover -s tests -v

# 跑一个算法（默认 24 辆 NPC 车、3 车道）
run.cmd train.py --algo dqn --seed 0 --steps 250000
run.cmd train.py --algo ppo --seed 0 --steps 250000
run.cmd train.py --algo ppo --seed 0 --steps 250000 --obstacles 0     # 空路消融

# 复现完整实验矩阵（2 算法 × 5 种子，同预算）
run.cmd scripts/run_batch.py --seeds 0 1 2 3 4 --steps 250000
run.cmd scripts/run_batch.py --seeds 0 1 2 3 4 --steps 250000 --obstacles 0 --out runs_empty

# 汇总成对比图 + 指标表
run.cmd aggregate.py

# 全过程视频
run.cmd scripts/make_video.py --run runs/ppo_s0 --out figs/video_ppo_lap.avi
run.cmd scripts/check_avi.py figs/video_ppo_lap.avi        # 容器 + 逐帧校验

# 单独评测，导出 GIF / 逐步 trace
run.cmd evaluate.py --run runs/ppo_s0 --ckpt ckpt_best.npz --episodes 10 --gif figs/ppo_drive.gif
run.cmd evaluate.py --run runs/dqn_s0 --trace out/dqn_trace.csv --episode 0

# 脚本化对照（不需要训练）
run.cmd evaluate.py --algo expert --episodes 10
```

> 如果你自己装了 Python 3.8+，把上面的 `run.cmd` 换成 `python` 即可，先 `pip install -r requirements.txt`。

输出目录：

```
runs/<tag>/
  config.json      环境 + 算法 + 预算 + 评估种子的完整快照
  episodes.csv     每个训练 episode 一行（回报、距离、速度、loss、eps ...）
  evals.csv        每 10k 步一次的贪心评估（固定验证种子 10000..10004）
  ckpt_best.npz    验证集最优参数（按 eval return 选）
  ckpt_final.npz   训练结束时的参数
  train.log        控制台完整转录
  summary.json     验证集 / 独立测试集(20000..20009) / 参考基线的汇总
figs/              对比曲线、柱状图、GIF、汇总表
```

---

## 2. 环境：MiniDrive（2D + 动态交通）

| 项目 | 取值 | 说明 |
|---|---|---|
| 任务 | 绕闭合赛道跑完整一圈，且**不撞其他车** | 出界／撞车＝终止，超时＝失败 |
| 赛道 | 459.3 m 闭合曲线（Catmull-Rom，432 点） | 最小转弯半径 10.6 m |
| **车道** | **3 车道 × 3.2 m ＝ 9.6 m 路面** | 车道中心 −3.2 / 0 / +3.2 m，画面上有虚线车道线 |
| 车辆模型 | 运动学自行车 + 摩擦限幅偏航率 | `dθ/dt = v/L·tanδ`，且 `v·ω ≤ a_lat_max` |
| 车体参数 | L=2.6 m，δ_max=0.45 rad，**v_max=20 m/s** | 加速 5.0 / 制动 10.0 m/s²，侧向加速度上限 9 m/s² |
| **其他车辆** | **24 辆**（约每 19 m 一辆），**每辆速度独立**（5.5–8.0 m/s + 各自相位的慢波动），**按车道行驶**，车道内漂移 ±0.22 m | `n_obstacles` 可设 0 做消融；同车道跟车减速保证 NPC 不重叠；内含 **2 组两车并排** + **2 处三车并排** |
| 三车并排 | 允许，但**会在主车驶近时自动让道** | 每步检查"三条车道在自车前方 32 m 内是否都被占"，若是则加速让最易通行那条车道的头车，约 **3 s** 内打开一条车道 |
| 控制周期 | dt = 0.1 s，max_steps = 1200（120 s） | 有交通时圈速变长 |
| 观测 | **29 维**：9 线激光雷达 + 最近 4 车 ×(距离, 方位 cos/sin, **接近速度**) + 速度 + 横向偏差 + 航向偏差(sin, cos) | 全部归一化 |
| 激光雷达 | 9 束，±70°，量程 20 m | 与赛道边界求交（向量化射线-线段相交） |
| 他车探测 | 30 m 半径，含**沿视线的相对速度** | 没有这一项就是 POMDP，学不会跟车 |
| 动作 | **7 个离散动作** | 硬左/左/直/右/硬右/加速/刹车 |
| 奖励（时间目标） | `−0.1`（时间成本）+ `0.02·Δs` − `0.05·(e/W)²`；出界/撞车 **−150**、完赛 **+20** | 成功圈 ≈ −20、超时 ≈ −111、撞车 ≈ −160，即 **完赛 > 超时 > 撞车** |
| 随机化 | 起点横向 ±1.5 m、航向 ±0.12 rad；NPC 车道/位置/速度随机 | 防止死记硬背 |
| 确定性 | 给定 seed 完全可复现（含 NPC 生成与运动） | 有用例保证 |

为什么这样设计（见报告 §1）：赛道足够长（需要速度规划）、弯道足够急（必须刹车）、
三条车道让"选一条空闲车道超车"成为可学、可解释的决策，因此"学得好不好"可以用
**成功率 + 碰撞率 + 最近车距** 三个低方差指标直接衡量。

环境 API 与 Gymnasium 一致（`reset()->(obs,info)`、`step()->(obs,r,term,trunc,info)`），
把 `MiniDriveEnv` 换成 `gymnasium.make("CarRacing-v3")` 只需包一层同样接口的 wrapper。

---

## 3. 算法：可插拔

所有算法实现 `autodrive/algos/base.py` 的四个钩子：

```python
act(obs, greedy)                        -> int
update(obs, a, r, next_obs, done, info) -> dict | None
state_dict() / load_state_dict()        -> npz 存档
stats()                                 -> 日志附加字段
```

训练循环、日志、评估、可视化**完全不知道**具体算法是谁。新增算法 = 新写一个文件 +
在 `algos/__init__.py` 的 `REGISTRY` 里加一行。这正是"换算法只需导入 py 文件"的实现方式。

### DQN（`autodrive/algos/dqn.py`）

| 超参数 | 值 |
|---|---|
| 网络 | 29 → 64 → 64 → 7（tanh），Q 值线性输出 |
| 学习率 / 优化器 | 5e-4 / Adam（梯度裁剪 10） |
| 折扣 γ / 目标更新 | 0.99 / Polyak τ=0.01 |
| 探索 | ε: 1.0 → 0.05，40k 步线性衰减 |
| 回放池 / 批大小 | 100k / 64 |
| 更新频率 | 每 4 个环境步 1 次梯度步 |
| 技巧 | Double DQN + Huber loss（δ=1） |
| 可选 | `reward_scale`（默认 1.0，把奖励乘系数后再进回放池） |

> **为什么它的成功率不高**：实测 25 万步后 Q 值只到 −7.3，而真实回报约 −182；同一状态下 7 个动作的 Q 差只有 0.09
> —— critic 学会了"世界一直以 −0.1/步 走下去"（自举不动点 ≈ −10），撞车 −150 没学进去，argmax 等于噪声。
> 根因是**奖励尺度 + Huber δ=1 + ε 在 40k 步就停**，不是参数没调好（含一次失败的修正实验记录）。
> 详见报告 §2.1。

### PPO（`autodrive/algos/ppo.py`）

| 超参数 | 值 |
|---|---|
| 网络 | actor 29→64→64→7（softmax），critic 29→64→64→1（线性输出） |
| 学习率 / 优化器 | 3e-4 / Adam（梯度裁剪 0.5） |
| γ / λ（GAE） | 0.99 / 0.95 |
| clip / epoch / minibatch | 0.2 / 8 / 256 |
| rollout 长度 | 2048 步 |
| 熵系数 / 价值系数 | 0.01 / 0.5 |
| 优势 | 批内标准化 |

### RARL（`autodrive/algos/rarl.py` + `autodrive/risk.py`）

移植自 [jingyunWu-23/risk](https://github.com/jingyunWu-23/risk)（*Risk-Aware RL for CARLA*）。
**本轮已按要求移除对抗车**，只保留"风险场 + 高斯混合 belief + 风险惩罚"，因此它现在是 **风险感知 PPO**。
原仓库是 CARLA + torch + TD3/ACA-LSTM；本项目用 **numpy + 离散动作** 复现了它的两个核心机制：

| 原仓库 | 本项目对应 |
|---|---|
| `rarl/risk/risk_field.py`：道路风险 `w/max(半宽−\|横向\|, margin)²` + 车道中心项，车辆风险 = 高斯模式椭圆（概率缩放 + 不确定性膨胀） | `autodrive/risk.py::TimeVaryingRiskField`（同一组公式与权重，作用在赛道坐标上） |
| `rarl/belief/gm_bayes.py`：7 个横向机动模式的 GM 贝叶斯 belief（IMM 混合 + CV 预测 + 卡尔曼更新 + 模式重加权） | `autodrive/risk.py::GaussianMixtureBelief`（状态改为世界坐标 `[x,y,vx,vy]`，模式仍为 ±0/0.5/1.0/1.5 m/s，模式轴向量化） |
| ~~`a/` 的对抗训练（adv/ego/joint + 对手车）~~ | **本轮移除**：没有对手车，`env.set_adversary` 不再被调用（测试断言绑定后不得接管任何 NPC） |

| 超参数 | 值 |
|---|---|
| 策略输入（默认） | 环境观测 29 + 风险特征 4（road / vehicle / total / margin）+ belief 特征 3×5 = **48 维** |
| 策略网络 | 与 PPO 相同：48→64→64→7；开启任一可选观测块时输入相应加宽 |
| 奖励（默认） | `r − 1.0 · vehicle_risk − 0.05 · road_risk`（**权重压在车辆风险上**；压总风险会退化成守中线） |
| 风险评估车辆数 | 最近 4 辆；belief 只跟踪最近 6 辆（原仓库最多 10 辆） |
| 可选观测块 | `hazard_mode` / `ttc_features` / `action_risk` / `lane_aware` / `lane_clearance_features`，全部默认关闭或不改变默认输入（见下表） |

**可选开关（默认保持 48 维旧行为）**

| 开关 | 默认 | 作用 |
|---|---|---|
| `hazard_mode` | `"field"` | `"ttc"` 时改用 `TTCCollisionRisk` 的纵向 TTC 危险度做奖励塑形 |
| `ttc_features` | `true`（仅 TTC 模式生效） | 追加 `[hazard, proximity]` 2 维；`hazard_mode="field"` 时不追加 |
| `action_risk` | `false` | 追加 7 维动作条件风险；`safety_shield` 可据此屏蔽高风险动作 |
| `safety_shield` | `false` | 前向 TTC 危险时禁掉所有非刹车动作；后向危险时禁掉刹车；**战术锁被它否决时会放弃变道并刹车**（见下） |
| `lane_aware` | `false` | 追加 9 维车道特征；`overtake_lock` 与车道变更奖励依赖它 |
| `overtake_lock` | `false` | 有安全相邻车道时锁定发起变道的硬转向动作，完成后进入冷却 |
| `lane_change_bonus` | `1.0` | **只在计划中的安全变道真正完成的那一步**支付一次；不再按“打了转向”逐步付费 |
| `lane_clearance_features` | `false` | 追加 6 维车道前/后净空，输入 48 → 54 |

TTC 模式下的奖励为 `r − 1.0 · (ttc_hazard + dense_weight · vehicle_risk) − 0.05 · road_risk`；
`dense_weight` 从 `dense_risk_weight` 线性衰减到 `dense_risk_end`（`dense_risk_decay_steps` 步内）。
这些开关只影响观测/奖励/动作掩码，不改变环境本身。

### 超车训练闭环（`configs/rarl_overtake.json`）

目标不是"多加观测"，而是让**奖励对一次安全、已完成的变道**给出可学信号。显式配置（72 维输入）：

| 键 | 值 | 作用 |
|---|---|---|
| `hazard_mode` / `ttc_features` | `ttc` / `true` | 纵向 TTC 危险度进入观测与奖励 |
| `action_risk` | `true` | 7 维动作条件风险；`action_risk_weight=0.05` |
| `safety_shield` | `true` | 前向危险时只允许刹车；否决战术锁 |
| `lane_aware` | `true` | 9 维车道特征（左右/本车道前后间隙、方向、最佳间隙、可用标志） |
| `lane_clearance_features` | `true` | 6 维车道级前后净空 |
| `overtake_lock` | `true` | 机会出现时锁定硬转向直到变道完成，冷却 20 步 |
| `lane_change_bonus` | `1.0` | **每次通过校验的计划变道只付一次** |
| `lane_change_min_progress` | `12.0` m | 两次奖励之间至少要有这么多沿轨道前进量（防左右摆动刷分） |
| `lane_change_min_front_gap` | `8.0` m | 变道完成后目标车道前方仍需有这么多净空，否则不算完成 |
| `overtake_lock_steps` | `40` | 战术锁超过这么多步仍未完成则放弃，避免一直硬转向 |
| `safe_progress_weight` | `0.2` | 每米**通过安全校验**的前进奖励（`prog06` 对照为 `0.6`） |
| `safe_progress_min_gap` | `12.0` m | 本车道前车近于此距离时不付前进奖励（防跟车冲刺） |
| `safe_speed_target` | `8.0` m/s | 超过该速度的位移不再计奖 |
| `overtake_window_steps` | `60` | 变道完成后开启的"安全前进"加成窗口 |
| `overtake_progress_multiplier` | `2.0` | 窗口内前进奖励的倍率，直到原阻挡车辆被甩到身后 |

奖励（TTC 模式）：

```
r − 1.0·ttc_hazard − 0.05·road_risk − 0.05·action_risk
  + lane_change_bonus · [通过校验的计划变道完成]
  + safe_progress_weight · min(ds, safe_speed_target·dt) · [hazard/动作风险/前车间距均达标]
      · (窗口内再乘 overtake_progress_multiplier)
```

- "计划变道" = 动作前的 `lane_aware` 判定同时满足：本车道前车 < 18 m、相邻车道无车在侧（≥7 m）、
  相邻车道前间隙比本车道多 ≥5 m。
- "通过校验的完成" = 跨过车道边界进入 `_overtake_target`，**并且**完成后目标车道侧向净空 ≥7 m、
  前向净空 ≥8 m、距上次奖励已前进 ≥12 m、战术锁未被安全屏蔽否决/超时放弃。
  因此**左右摆动不付钱**（160k 步评估里 90 次变道只有 6 次拿到奖励），**切进/别车也不付钱**。
- 前进奖励按 `info["ds"]`（沿轨道真实前进量）支付，所以转向本身一分钱不给。
- 安全屏蔽现在对战术锁同样有效：若锁定的硬转向被 TTC/动作风险否决，锁被放弃并改为刹车。
- **已知漏洞（未修，属于风险栈而不是奖励）**：`TTCCollisionRisk` 把每个模式的危险度乘上该模式的
  信念权重（7 个模式约 1/7），因此单辆车的 `ttc_front_hazard` 上限约 0.14，`shield_hazard_threshold=0.5`
  和 `action_risk_mask_threshold=0.25` 实际都不会触发（`diagnose_risk` 的 `front_hazard_steps` 恒为 0）。
  新奖励因此改用几何间距 `safe_progress_min_gap` 作为真正生效的安全闸门。

超车诊断指标（`OvertakeTracker`，`episodes.csv` / `evals.csv` / `evaluate.py` / `diagnose_risk.py` 共用）：
机会步数、朝目标变道比例、保持比例、刹车比例、变道次数、朝计划方向完成次数、奖励触发次数，
外加成功率与碰撞率。口径是"策略自己的 `lane_aware` 判断"（不是纯几何间隙）。

```bash
# 项目内解释器（.venv 在本机已失效，见 §1.0）
./runtime/python/python.exe train.py --algo rarl --seed 0 --steps 4096 \
    --eval-every 2048 --eval-episodes 2 --final-eval-episodes 2 --test-episodes 2 \
    --config configs/rarl_overtake.json --tag rarl_overtake_smoke
./runtime/python/python.exe train.py --algo rarl --seed 0 --steps 20000 \
    --eval-every 5000 --eval-episodes 3 --final-eval-episodes 3 --test-episodes 3 \
    --config configs/rarl_overtake.json --tag rarl_overtake_20k
./runtime/python/python.exe evaluate.py --run runs/rarl_overtake_20k --ckpt ckpt_final.npz \
    --episodes 5 --seed-base 20000
./runtime/python/python.exe scripts/diagnose_risk.py --run runs/rarl_overtake_20k \
    --ckpt ckpt_final.npz --episodes 5 --seed-base 20000
```

实测（2026-11-20，单机、普通 CPU；数字为真实运行结果，不是估计）：

| 运行 | 步数 | 结果 |
|---|---|---|
| `rarl_overtake_smoke` | 4 096 | 配置被 `train.py` 接受，36.8 s（111 步/s），验证/测试评估跑通 |
| `rarl_overtake_20k` | 20 000 | 116 episodes，161.4 s（124 步/s）；4 个 episode 出现机会（共 63 机会步），5 次计划变道完成并触发奖励 |
| 独立评估（种子 20000–20004） | — | 成功率 0、碰撞率 0.6、平均速度 3.8 m/s |
| `diagnose_risk`（同上） | — | 机会 0、变道 29 次（朝计划 0）、屏蔽触发 0、最大前向 TTC 危险 0.19 |
| 对照 `control_rarl_20k`（默认 48 维 RARL） | 20 000 | 成功率同样为 0，648 步/s |

### 2026-11-20 奖励返工后的实测（单 seed，held-out 种子 20000–20009，各 10 回合）

同环境 `configs/env_overtake_stage1.json`、同 seed 0，`--eval-episodes 10 --seed-base 20000`：

| 运行 | 预算 | 行进步数/s | 距离 | 速度 | 成功率 | 碰撞率 | 机会步 | 朝目标 | 变道 | 通过校验 | 甩开阻挡车 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `rarl_ovl20k_legacy`（旧奖励） | 20 k | 3184 | 22.3 m | 2.01 | 0 | 1.00 | 86 | 0.98 | 14 | 6 | 0 |
| `rarl_ovl20k_hardened`（只加校验） | 20 k | 3184 | 22.3 m | 2.01 | 0 | 1.00 | 86 | 0.98 | 14 | 6 | 0 |
| `rarl_ovl20k_safeprog`（+前进奖励 0.2） | 20 k | 3373 | 12.0 m | 0.81 | 0 | 1.00 | 33 | 1.00 | 11 | 2 | 0 |
| `rarl_ovl100k_legacy` | 100 k | 6789 | 30.9 m | 0.45 | 0 | 1.00 | 24 | 1.00 | 28 | 1 | 0 |
| `rarl_ovl100k_safeprog` | 100 k | 5158 | 45.1 m | 1.02 | 0 | 1.00 | 58 | 0.86 | 38 | 3 | 0 |
| `rarl_ovl200k_prog06` @160k（前进奖励 0.6） | 160 k | 9950 | 183.1 m | 1.76 | 0 | 0.60 | 89 | 0.88 | 90 | 6 | 2（含 1 次安全超车） |

- `20k legacy` 与 `20k hardened` 指标完全一致：20k 预算下策略还没学会摆动，校验条件一次都没否掉，
  说明校验不是靠"扣钱"起作用，而是等策略真的开始乱摆（160k：90 次变道只有 6 次通过）才生效。
- `prog06` 是唯一跑出"超车后甩开阻挡车"的运行（`diagnose_risk`: `safe_overtake_observed=true`，
  10 回合里 1 次安全完成），但**成功率仍为 0、碰撞率 0.5、平均速度 1.8 m/s**，远未达到可用水平。
- 三个 20k/100k 运行都退化成低速爬行；把前进奖励从 0.2 提到 0.6 并训练到 160k 才把速度拉到 1.8 m/s，
  说明**奖励尺度与预算同时是瓶颈**：`r_crash=-150`、`w_step=-0.1/step` 使"慢慢爬"与"跑完一圈"
  （约 −47）差距不大，单 seed 无法据此断言奖励已经解决问题。

结论只说机制，不说性能：奖励/诊断/评估链路在真实 rollout 中确实触发过（训练期 5 次奖励），
但 20k 步的智能体还不会开车（离开路面/撞车），评估 episode 里连超车机会都遇不到。

未验证 / 已知限制：
- **未做 ≥100k 步训练**；本配置每步要跑 2×7 次动作风险 rollout（10 步 × 28 个交通模式），
  实测 111–124 步/s（净空边界提到 rollout 外层之前只有 46 步/s），250k 步约 35–40 分钟。
- 未跑多 seed，未验证收敛后的超车成功率/圈速；评估期机会为 0，与自车平均 3.8 m/s
  （慢于 NPC 的 5.5–8 m/s、追不上前车）一致，但未做因果消融。
- `overtake_lock` 下"朝目标变道比例"按构造接近 1（锁强制动作），该指标只有在关锁时才能反映策略偏好。
- 超车特征来自仿真状态（`obstacles.s/lane`），不是真实传感器。

### 超车 curriculum：先让自车追得上（2026-11-20）

20k 冒烟暴露的第一个问题不是奖励，而是**追不上**：默认 NPC 5.5–8 m/s，训练早期的自车只有 3.8 m/s，
评估里连"前方被挡"都遇不到（机会 0）。curriculum 因此只改环境侧，不改动作表、不删安全约束：

| 阶段 | 环境文件 | NPC 速度 (m/s) | NPC 数 | 自车 `v_max` | 初始前车 | 相邻车道交通 |
|---|---|---|---|---|---|---|
| 1 慢速追车 | `configs/env_overtake_stage1.json` | 2.5–4.0 | 12 | 10 | 20 m | 2 组并排 + 1 堵三车墙 |
| 2 中等交通 | `configs/env_overtake_stage2.json` | 4.0–6.0 | 18 | 13 | 20 m | 2 组并排 + 2 堵墙 |
| 3（可选）原口径 | 不用 `--env-json` | 5.5–8.0 | 24 | 16 | 无 | 默认 |

- 新增环境参数 `npc_lead_gap`（默认 `0` = 关闭）：`reset` 时把一辆车移到**自车所在车道**前方指定距离，
  速度取 `npc_v_min`，并保持同车道 `same_lane_gap` 间距规则（放不下就放弃，不会叠车）。
- 取 `20 m` 而不是 14–16 m，是因为 `lane_block_distance=18`：小于 18 m 会在自车**静止**时立刻触发
  `overtake_lock` 的满舵，episode 起步就被强制转向并偏出路面；20 m 让自车先加速，再在 ~7 m/s 时进入机会。
- 碰撞惩罚（−150）、`safety_shield`、`action_risk` 掩码、`lane_clearance` 与变道侧向净空判定全部保留；
  只改 `npc_v_* / v_max / n_obstacles / npc_block_range / npc_lead_gap`。

**环境侧验证（脚本驾驶员，10 回合，种子 20000–20009；几何口径：本车道前车 < 18 m）**

| 配置 | 自车均速 | 被挡步数 | 其中在追赶 | 平均前车间隙 | 被挡的 episode |
|---|---|---|---|---|---|
| stage 1 | 7.06 m/s | 1 896 | 1 122（59%） | 12.6 m | 10/10 |
| stage 2 | 7.70 m/s | 1 790 | 940（53%） | 13.0 m | 10/10 |

即：自车确实比 NPC 快，能在被挡住时持续缩短间距；相邻车道仍有前后车（2 组并排 + 1 堵三车墙），不是空路。

```bash
# 环境侧检查（脚本驾驶员，报告被挡/追赶步数；不训练）
./runtime/python/python.exe scripts/check_curriculum.py \
    --env-json configs/env_overtake_stage1.json --episodes 10 --seed-base 20000
./runtime/python/python.exe scripts/check_curriculum.py \
    --env-json configs/env_overtake_stage2.json --episodes 10 --seed-base 20000

# 阶段 1 训练（两段续训，共 250k）
./runtime/python/python.exe train.py --algo rarl --seed 0 --steps 125000 \
    --config configs/rarl_overtake_curriculum.json \
    --env-json configs/env_overtake_stage1.json \
    --eval-every 25000 --eval-episodes 3 --final-eval-episodes 3 --test-episodes 3 \
    --tag rarl_overtake_curr_s1
./runtime/python/python.exe train.py --algo rarl --seed 0 --steps 125000 \
    --init-checkpoint runs/rarl_overtake_curr_s1/ckpt_final.npz \
    --config configs/rarl_overtake_curriculum.json \
    --env-json configs/env_overtake_stage1.json \
    --eval-every 62500 --eval-episodes 3 --final-eval-episodes 3 --test-episodes 3 \
    --tag rarl_overtake_curr_s1b
# 用当前（含 safe_progress / 通过窗口）算法配置再微调 20k
./runtime/python/python.exe train.py --algo rarl --seed 0 --steps 20000 \
    --init-checkpoint runs/rarl_overtake_curr_s1b/ckpt_final.npz \
    --config configs/rarl_overtake_curriculum.json \
    --env-json configs/env_overtake_stage1.json \
    --eval-every 10000 --eval-episodes 3 --final-eval-episodes 3 --test-episodes 3 \
    --tag rarl_overtake_curr_s1c
# 独立 10 回合评估 + 诊断
./runtime/python/python.exe evaluate.py --run runs/rarl_overtake_curr_s1c \
    --ckpt ckpt_best.npz --episodes 10 --seed-base 20000
./runtime/python/python.exe scripts/diagnose_risk.py --run runs/rarl_overtake_curr_s1c \
    --ckpt ckpt_best.npz --episodes 10 --seed-base 20000
```

**实测（2026-11-20，单机 CPU，单 seed；真实数字，未做多 seed）**

| 运行 | 步数 | 结果 |
|---|---|---|
| `rarl_overtake_curr_s1` + `_s1b` | 250k | 训练后期 episode 能安全行驶 ~300 m（1200 步限时内未跑完一圈），评估仍 0 成功；两段吞吐 421 / 435 步/s（当时的算法配置尚无 `safe_progress_*` / 通过窗口） |
| `rarl_overtake_curr_s1c`（+20k 微调） | 270k 累计 | 评估均速 2.82 m/s；训练 377 步/s（与另一进程共享 CPU） |
| 独立评估（10 eps） | — | **成功率 0**、碰撞率 0.30、crash 率 0.80（含出界）；机会 360 步、朝目标 0.92、完成计划变道 23 次、奖励 21 次、**通过前车 0 次** |
| `diagnose_risk`（10 eps） | — | 被挡 360 步（9/10 episode）、其中 85% 步自车快于前车、平均被挡间隙 11.4 m；屏蔽改写 4.5% 动作；加速度动作被选中 0 次 |

结论（只陈述机制，不报性能）：

1. curriculum 解决了"追不上"：9/10 评估 episode 真的进入了被挡状态，且 85% 的被挡步在**缩短**间距，
   说明自车速度已经高于 NPC；
2. 计划变道链路在新奖励下确实触发（23 次完成、21 次奖励、窗口 993 步），但**没有一次真正超过前车**，
   80% 的 episode 以 crash 结束（其中碰撞 30%）—— 当前瓶颈是"边开边超"的驾驶能力，不是环境或信号缺失；
3. 吞吐从 124 步/s 提到 420–440 步/s（`action_risk_horizon` 1.0→0.4，且 `action_risks` 的交通模式循环
   改为向量化、逐项公式不变，`tests/test_rarl.py` 的动作风险用例全部通过）。

**未完成 / 未验证**

- **stage 2 没有训练**：stage 1 的 270k 仍未达到可迁移的驾驶水平，只做了 2k 步冒烟（配置被 `train.py`
  接受，18 辆 NPC 4–6 m/s、2 组并排 + 2 堵墙、379 步/s）。因此不能声称"两阶段课程有效"。
- 只有 seed 0；成功率 0，不能对外报任何超车性能；也未做"关掉 curriculum 的同预算对照"。
- s1/s1b 与 s1c 用的算法配置不同（后者才含 `safe_progress_*` 与通过窗口），因此 270k 不是一条同配置曲线，
  只能说明"环境侧追赶成立、链路触发过、仍未学会超车"。
- 与默认 5.5–8 m/s / 24 辆的原始口径相比，curriculum 的交通更慢、更稀；它的作用是把"超车"变成
  episode 内必然发生的事件，**不是**最终评测口径。
- 慢速交通会让"必须超车才能前进"变成硬约束：默认环境里弱策略还能跟在 5–6 m/s 车流后面拿里程，
  在 stage 1 里跟车只能到 2.5–4 m/s，因此**同样的步数下成功率反而更低**，这与上面的 0 成功率一致。

**车道级净空观测（`lane_clearance_features`，默认关闭）**

- 6 维 = `[左邻车道, 本车道, 右邻车道] × [前方间隙, 后方间隙]`，均为沿赛道弧长的距离，
  截断到 `lane_clearance_range` 后除以它（1.0 = 该方向至少这么远，空车道也是 1.0）。
  赛道上不存在的车道（2 车道、道路边缘）记为 `-1.0`，因此正常值域是 `[0, 1]`。
- 目的是让策略区分"前车在本车道"与"前车在相邻车道"，而不是仅靠激光雷达/最近车辆距离。
- **数据来源是仿真状态，不是真实传感器**：前车间隙用 `obstacles.lane_free_distance`，
  后车间隙由 `obstacles.s` / `obstacles.lane` 按同一弧长约定计算。这组特征不能作为
  真实车辆可获得的感知证据。
- 默认关闭保证默认策略输入仍是 48 维，**旧 RARL 检查点（`pol_meta[0] == 48`）可直接加载**。
  开启后加载旧检查点会抛出明确的 `ValueError`（列出检查点维度、当前维度及开关状态）；
  用关闭开关的默认配置加载旧检查点，`tests/test_lane_clearance.py` 有对应用例。

```json
{ "lane_clearance_features": true, "lane_clearance_range": 30.0 }
```

**传感器融合观测层（`sensor_fusion`，默认关闭）**

`autodrive/sensors.py` 提供 `SensorFusion`（别名 `LaneRadar`）。它只读环境的**可解释状态**
（自车弧长位置/速度、每辆 NPC 的 `s`/`lane`/`v`、车道中心线），不用相机图像、不读渲染帧。

- 三条车道行 `[左邻, 本车道, 右邻]`，每行 7 个数：前车间隙、前车闭合速度（自车 − 前车）、
  后车间隙、后车闭合速度（后车 − 自车）、前/后接触 TTC（扣掉 4.3 m 车长）、**安全窗口**（秒）。
  安全窗口 = 任一侧间隙缩到最小安全间距所需时间的最小值，饱和于 `sensor_ttc_horizon`。
- 4 个标量：`safe_to_accelerate`、目标方向、目标车道安全窗口、`target_available`。
  合计 `3×7+4 = 25` 维，值域 `[0,1]`（速度项 `[-1,1]`），不存在的车道整行记 `-1`。
- `sensor_fusion: true` 时策略输入 48 → 73；与 `lane_aware`(+9)、`lane_clearance_features`(+6)、
  TTC(+2)、`action_risk`(+7) 同时开启时为 97 维。默认关闭，**旧 48 维检查点照常加载**
  （已用 `runs/control_rarl_20k/ckpt_final.npz` 实测）；开启后加载旧检查点会抛出列出维度的 `ValueError`。
- `lane_aware` / `lane_clearance_features` / `overtake_lock` 现在都消费同一个 `SensorFrame`，
  不再各自遍历一遍 NPC；`tests/test_sensors.py` 用 24 个用例覆盖间隙/闭合速度/TTC/安全窗口/
  缺失车道/检查点维度/“观测跟随传感器”。
- 奖励仍只来自**安全完成的计划变道**（通过 `lane_change_min_progress` 防来回摆动刷分）和
  **变道/超车后的持续前进**（`progress_reward_only_in_overtake_window`，窗口内 `safe_progress_weight`
  才生效）；转向动作本身不给分。`lane_change_bonus_requires_pass` 可把奖励进一步推迟到“真正越过前车”。

课程入口：`scripts/train_overtake_curriculum.py`（easy → stage1 → stage2，checkpoint 续训，
最后对每个阶段跑独立 10 回合评估 + `diagnose_risk`）。实测（seed 0，真实数字）：

| 运行 | 步数 | 结果 |
|---|---|---|
| `rarl_overtake_sensor_20k` | 20 000 | 7 次计划变道完成并触发奖励，**通过前车 0 次** |
| `rarl_overtake_sensor_40k`（easy/stage1/stage2） | 40 000 | 均速 0.2–1.9 m/s，**0 次通过**，评估仍以 crash 为主 |
| `rarl_overtake_sensor_v5`（含锁的最低速度 3 m/s） | 20 000 | 出现 **1 次通过**（seed 20004，104 步、45 m），但同 episode 碰撞，**安全通过 0 次** |
| `rarl_overtake_sensor_v6` | 48 000 | 各阶段 0 次通过、0 次安全通过 |

结论：传感器层与“安全完成才付费”的奖励链路都真实触发过（变道完成、窗口、通过判定），但 20k–48k 步内
策略仍学不会边开边超；**没有任何一次安全完成超车**，不能对外报超车性能。瓶颈是驾驶能力，不是观测缺失。

```bash
run.cmd train.py --algo rarl --seed 0 --steps 250000 --config configs/rarl.json
```

> 说明：本方法不再有对手车，`evaluate.py` / `make_video.py` 仍会调用 `bind_env`（只保存环境引用，不接管任何 NPC）。

两算法**共用**同一环境、同一动作集、同一预算（环境步数）、同一评估种子，因此可比。

---

## 4. 可调试性

| 手段 | 命令 / 位置 | 用途 |
|---|---|---|
| 数值梯度检验 | `tests/test_suite.py::TestNeuralNet/TestPPOGradient` | 用中心差分逐元素核对 MLP 反向传播、熵梯度、**完整 clipped PPO 损失**的梯度 |
| 环境 & 交通用例 | `TestEnvironment` / `TestTraffic` | 同 seed 复现、观测维度恒定、出界／撞车必终止、**NPC 不会生成在自车身上**、NPC 会动且不越界 |
| 逐步 trace | `python evaluate.py --run runs/ppo_s0 --trace out/t.csv` | 每步 state/action/reward/进度/前车距离 落 CSV |
| 控制台逐步打印 | `python train.py --algo dqn --debug-trace 40` | 训练时逐 step 打印观测与奖励，排查奖励 bug |
| **全过程视频** | `python scripts/make_video.py --run runs/ppo_s0 --out x.avi` | 每一控制步一帧、10 fps 实时；画面上有他车、激光雷达、车速/回报/**前车距离** |
| 视频容器校验 | `python scripts/check_avi.py x.avi` | RIFF 结构、各 chunk 尺寸、帧数、索引偏移、**逐帧 JPEG 解码** |
| 行为诊断图 | `figs/trace_gap.png`、`figs/action_usage.png` | 跟车间距是否稳定、动作是否退化 |
| 脚本化上界 | `--algo expert` | 纯追踪 + 曲率限速 + 简单避让，用来证明环境可解、给出参照线 |
| 随机对照 | `--algo random` | 确认奖励尺度（随机策略应接近 0 分） |
| 指标日志 | `runs/*/episodes.csv`, `evals.csv` | loss / entropy / approx_kl / clip_frac / **碰撞率** / **最近车距** / 速度 / 横向 RMS |
| 训练-验证-测试分离 | 10k 一次验证集评估；最终在 20000+ 的**独立测试种子**上报告 | 避免"用验证集挑模型又用验证集报数" |

---

## 5. 目录

```
autodrive/
  env.py            MiniDrive 环境（动力学、奖励、观测、交通接入）
  obstacles.py      ObstacleSet：NPC 车辆生成 / 运动 / 碰撞 / 最近车查询 / 射线求交
  track.py          赛道几何：Catmull-Rom 中心线、边界、射线求交、投影
  nn.py             numpy MLP + 手写反向传播 + Adam
  video.py          MJPEG-AVI 视频写出（无 ffmpeg 依赖）
  baselines.py      PurePursuitDriver（脚本化参考驾驶员，含避让）
  logging_utils.py  CSV 日志 + 统一评估协议（含碰撞率/最近车距）
  render.py         Pillow 渲染（帧、GIF、折线图、柱状图）
  risk.py           椭圆风险场 + GM belief + 多模态 TTC 风险
  sensors.py        SensorFusion / LaneRadar：车道级间隙、闭合速度、TTC、安全窗口（无相机）
  algos/            base.py / dqn.py / ppo.py / rarl.py / __init__.py(注册表)
train.py            训练 CLI（--obstacles N 切换交通）
evaluate.py         评测 CLI（指标表、trace、GIF、视频）
aggregate.py        多 run 汇总（对比图 + summary.csv/md）
scripts/            run_batch.py / run_parallel.py / make_video.py / check_avi.py
                    make_debug_figs.py / make_metric_curves.py（指标曲线）
                    eval_curve.py（每个 checkpoint 跑 100 个 episode 的专项评估 + 曲线）
                    diagnose_risk.py（TTC 触发、动作掩码与超车机会/变道统计）
                    check_curriculum.py（脚本驾驶员下"被挡/追赶"步数，验证 curriculum 环境）
                    eval_ensemble.py（多 seed 策略概率/多数投票集成）
                    smoke_test.py / trace_episode.py / tune_expert.py / make_strips.py
                    setup_venv.ps1（创建/重建独立虚拟环境）
tests/               test_suite.py（梯度/PPO/环境/交通）+ test_rarl.py（风险场/belief/TTC/动作与车道风险）+ test_lane_clearance.py（车道级净空特征、检查点维度兼容）
configs/            dqn.json / ppo.json / rarl.json / rarl_lowrisk.json / rarl_overtake.json（超车闭环，72 维）
                    env_overtake_stage1.json / env_overtake_stage2.json（超车 curriculum 的环境覆盖）
                    rarl_overtake_curriculum.json（curriculum 用算法配置）
run.cmd / run.ps1   用项目内 .venv 执行任意脚本（无需激活）
runtime\python\     随项目携带的 CPython 3.12（含 numpy/Pillow）——见 §1.0
.venv\              虚拟环境（基于 runtime\python 创建）
```

> `runs_probe/probe_dqn_eps/` 是报告中"放慢探索后 DQN 仍然失败"的对照实验记录。

## 6. 复现性

* 环境、算法、评估全部由 `seed` 决定；`config.json` 完整记录，可直接 diff 两次实验。
* 每个 run 独立进程，跑批互不影响。
* 硬件无关（纯 numpy，无 GPU/无随机并行归约），`steps/s` 在 3k–5k 量级。

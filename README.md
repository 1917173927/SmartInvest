# SmartInvest (StockAnalysis)

> **严谨、可复现、时点隔离（Point-in-Time）的个人量化投资研究与自动化挂单决策系统。**  
> 知识库直接兼容 **Obsidian Vault**，并提供工业级 Python CLI 工具链 `stock`。

---

## 🧭 系统核心设计哲学与免训练架构

### 模型的胜率是如何实现的？需要手动训练吗？
**结论：你不需要手动训练深度学习模型，系统开箱即用，并由“时序通用大模型 + 公理化量化因子 + 动态滚动自适应校准”三层闭环驱动：**

```
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │                          SmartInvest 三层量化预测与决策引擎                            │
 └───────────────────────────────────────────┬────────────────────────────────────────────┘
                                             │
             ┌───────────────────────────────┼───────────────────────────────┐
             ▼                               ▼                               ▼
   【第 1 层：时序通用大模型】       【第 2 层：公理化金融因子】       【第 3 层：滚动在线自适应校准】
   (Zero-Shot Foundation Model)     (Axiomatic Quant Factors)       (Walk-Forward Calibration)
   - 基于 Amazon Chronos-2 预训练   - Yang-Zhang(2000) 极小方差波动率 - 自动跟踪历史滚动分位数损失
   - 跨行业海量真实时序学习通用特征  - Asness(2019) QMJ 质量四维模型  - 动态反向损失定权 (Inverse Loss)
   - 输出完整概率分布 (q10/q50/q90) - Blitz(2013) 残差特质动量平滑度 - 连续预测偏误触发熔断与自动降级
   - 无需用户从零训练权重           - 经典闭式解析公式，杜绝过拟合   - 可复现基准见 docs/BACKTEST_BENCHMARK.md
```

1. **时序通用大模型（Zero-Shot Foundation Models）**：
   - 默认采用 `amazon/chronos-2` 时序大模型。它已经在全球跨领域的数十亿条时间序列上预训练完毕，具备极强的 Zero-Shot 概率预测能力，直接读取历史 OHLCV 序列并输出概率分布，**无需在本地进行耗时巨大的反向传播训练**。
2. **公理化金融因子与数学法则（无需拟合，杜绝过拟合与 P-Hacking）**：
   - 波动率采用 **Yang-Zhang (2000)** 极小方差漂移无关估计器；
   - 基本面质量采用 **Asness et al. (2019) QMJ (Quality Minus Junk)** 四维标准化得分；
   - 趋势动量采用 **Blitz (2013)** 残差特质动量与趋势平滑度；
   - 仓位计算采用 **Fractional Kelly (分数凯利公式)**。
   - 所有因子均直接使用国际顶级期刊严格推导的解析数学公式，**没有待调超参数，从物理上杜绝了对历史噪音的曲线拟合**。
3. **滚动样本外在线自适应校准（Walk-Forward Calibration）**：
   - 每次生成预测时固化 SHA-256 事前凭证（Receipts），到期后自动核对实际收益；
   - 系统计算 Chronos 与随机游走基线的分位数损失（Pinball Loss），按逆损失自适应分配集成权重：
     $$w_{chronos} = \frac{\text{Loss}_{base}^{-1}}{\text{Loss}_{base}^{-1} + \text{Loss}_{chronos}^{-1}} \in [0.2, 0.8]$$
   - 一旦模型连续出现预测回撤，系统会自动触发降级保护，将权重回退至保守基线。

---

## 🤖 Agent 协作与实盘操作 SOP 指南

为确保人类投资者及各类 AI Agent 对齐工作进度与操作规范，系统制定了以下 **4 个标准化操作流程 (SOP)**：

```
 ┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
 │ SOP 1: 盘前挂单 │ ───>  │ SOP 2: 盘中执行 │ ───>  │ SOP 3: 盘后深度 │ ───>  │ SOP 4: 调仓复盘 │
 │ (09:00 - 09:25) │       │ (09:30 - 15:00) │       │ (18:30+)        │       │ (周末/定期)     │
 └─────────────────┘       └─────────────────┘       └─────────────────┘       └─────────────────┘
```

### SOP 1：工作日盘前准备（09:00 – 09:25）
1. **自动触发**：macOS 守护进程于 09:00 自动生成晨报，并在 Mac 桌面弹出提醒。
2. **人工/Agent 复核**：
   - 查看 `06-自动分析/最新盘前挂单晨报.md` 或在终端运行：
     ```bash
     uv run stock morning --capital 100000
     ```
3. **券商 App 预埋单**：
   - 根据表格中的挂单价格、建议手数与下单类型：
     - **① 首笔底仓**：在 9:15 集合竞价前挂入限价买单；
     - **② 强支撑加仓**：设置“价格回踩支撑上沿自动买入”的券商条件单；
     - **③ 极限买点**：在深度折价下沿预埋低位限价单；
     - **🛑 硬止损线**：在券商 App 设置“跌破止损线自动清仓”的条件预警。

### SOP 2：盘中被动执行与监控（09:30 – 15:00）
1. **纪律原则**：盘中由券商系统自动触发条件单被动成交，**严禁盯盘追涨杀跌**。
2. **盘中临时敏感性推演**（若市场发生意外异动）：
   ```bash
   uv run stock size CN:601318 --capital 100000
   uv run stock scenario CN:601318 --margin-delta 0.05
   ```

### SOP 3：每日盘后全流程分析（18:30+）
1. **自动触发**：macOS 守护进程于 18:30 自动执行 `stock auto`。
2. **执行链条**：
   `依赖检查 → 行情/分红增量同步 → 新闻与宏观刷新 → 到期回执自动核验 → 条件校准 → 多周期模型推演 → 图表渲染与 Obsidian 研报整理`
3. **成果查阅**：
   - 打开 Obsidian 浏览 `00-交易总览.md` 与 `06-自动分析/最新摘要.md` 查看今日多周期冲突决策卡与行业暴露。

### SOP 4：新增标的、策略回测与周度复盘
1. **添加新标的**：
   ```bash
   uv run stock add CN:600519 --name "贵州茅台" --sector "消费" --role "core" --fair-pe 25.0 --fair-pb 6.0
   ```
2. **执行严格的非重叠 Walk-Forward 样本外回测**：
   ```bash
   uv run stock evaluate --backtest CN:600519 --horizon-days 20
   ```
3. **跨标的横向比对矩阵**：
   ```bash
   uv run stock compare CN:601318 CN:601398 CN:600519
   ```

---

## 💻 完整 CLI 命令速查表 (Command Reference)

| 命令 | 用途与核心参数 |
|---|---|
| `stock morning` | 🌅 **盘前挂单晨报**：输出全标的 3 阶挂单价、建议手数与止损线（支持桌面弹窗） |
| `stock size SYMBOL` | 🎯 **实盘仓位精算器**：`--capital 100000` `--risk-budget 0.02` 计算精确手数与回撤金额 |
| `stock compare S1 S2...` | ⚖️ **跨标的比对矩阵**：横向比对估值折扣、多周期信号与优先建仓排序 |
| `stock dash` | 📊 **终端实时决策看板**：Rich Table 呈现全资产四周期信号与数据质量 |
| `stock scenario SYMBOL` | 🔬 **What-If 情景推演**：模拟盈利预期调整与安全边际变化对买入价的影响 |
| `stock add SYMBOL` | ➕ **向导式添加标的**：设定行业、组合角色（core/satellite）、估值模型与基准倍数 |
| `stock auto` | ⚡ **一键全流程自动批处理**：无交互执行同步、研报生成、回执核对与图表更新 |
| `stock analyze SYMBOL` | 📝 **单标的多周期分析**：`--horizon short\|medium\|long\|value\|all` |
| `stock evaluate` | 🔍 **到期回执核验与回测**：`--backtest SYMBOL --horizon-days 20` 执行严格样本外盲测 |
| `stock portfolio` | 💼 **组合风控硬约束检查**：核查单股上限、行业集中度、现金底线与组合回撤 |
| `stock sync SYMBOL` | 🔄 **按需数据同步**：同步日线 OHLCV、公司分红除权除息与财务数据 |
| `stock backfill` | 📥 **历史数据深度回填**：分块拉取 10 年以上原始历史数据并安全写入 SQLite |
| `stock doctor` | 🩺 **环境与依赖诊断**：检查 Python 3.12、数据库、Chronos 权重与 API 连通性 |

---

## ⚙️ 环境安装与定时任务部署

### 1. 基础环境配置
要求 Python 3.12 与 [uv](https://docs.astral.sh/uv/)。

```bash
git clone git@github.com:1917173927/SmartInvest.git
cd SmartInvest
cp stock-analysis.example.toml stock-analysis.toml

# 安装完整依赖（数据源、Chronos 时序模型、绘图引擎与开发套件）
uv sync --extra data --extra forecast --extra charts --extra dev

# 验证运行环境
uv run stock doctor
```

### 2. macOS 后台自动化守护进程一键安装
运行自带的配置脚本，会自动在 macOS 系统中注册两大无缝定时守护进程：
```bash
uv run python scripts/install_launchd.py
```
* **`com.stockanalysis.morning.plist`**：每个工作日 **09:00** 自动计算全标的挂单网格并推送桌面通知；
* **`com.stockanalysis.daily.plist`**：每日 **18:30** 自动执行盘后全流程数据同步与深度模型计算。

---

## 🛡️ 数据边界与时点隔离保障 (Strict Point-in-Time)

1. **时点物理隔离（No Look-Ahead Bias）**：
   - 每次分析与回测只截取 $t \le \text{as\_of}$ 历史切片，未来的开高低收和未发布财报在内存中完全不存在。
2. **纯样本外非重叠封测（Purged Walk-Forward）**：
   - 步长等于预测期限（`stride = horizon_days`），测试窗口互不重叠，彻底消除自相关性。
   - 详细回测规范、股票池定义、摩擦成本与最新样本外复现数据统一记录于 [docs/BACKTEST_BENCHMARK.md](docs/BACKTEST_BENCHMARK.md)。
3. **不可篡改的 SHA-256 事前预测凭证（Receipts）**：
   - 预测在发出时即计算哈希并固化存入 SQLite，未来到期后自动核验真实盈亏，确保无任何后验作弊。

---

## 📂 Obsidian 知识库工作流

1. 从 `00-交易总览.md` 开始查看全局资产与 Dataview 动态监控看板；
2. 每日盘前打开 `06-自动分析/最新盘前挂单晨报.md` 执行挂单；
3. 盘后阅读 `06-自动分析/最新摘要.md` 查看四周期信号冲突与行业暴露；
4. 真实操作前调用 `05-模板/操作记录模板.md` 记录买入理由与失效条件；
5. 每周复盘纪律，每月更新 `01-持仓/` 资产快照。

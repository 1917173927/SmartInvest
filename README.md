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
     $$
     w_{chronos} = \frac{\text{Loss}_{base}^{-1}}{\text{Loss}_{base}^{-1} + \text{Loss}_{chronos}^{-1}} \in [0.2, 0.8]
     $$
   - 一旦模型连续出现预测回撤，系统会自动触发降级保护，将权重回退至保守基线。

---

## 🧭 实盘操作 SOP 指南

盘中下单前请先阅读 [盘中操作建议与执行指南](docs/INTRADAY_EXECUTION_GUIDE.md)，明确区分
盘中执行价、日线分析基准、账户总资产、可用现金和条件触发建议。

为确保数据口径、操作条件和复盘记录保持一致，系统制定了以下 **4 个标准化操作流程 (SOP)**：

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
     uv run stock morning --capital 51546.80
     ```
3. **券商 App 预埋单**：
   - 根据表格中的挂单价格、建议手数与下单类型：
     - **① 首笔底仓**：仅当建议股数大于 0 且仓位未达上限时，设置不高于首笔价的限价单；
     - **② 强支撑加仓**：确认第一档成交、仓位仍有额度且基本面未恶化后，再设置第二档条件单；
     - **③ 极限买点**：确认前两档成交和剩余额度后，才考虑第三档低位限价单；
     - **🛑 逻辑失效参考线**：设置到价预警；触发后先取消加仓并复核连续收盘与基本面条件，不自动清仓。

### SOP 2：盘中被动执行与监控（09:30 – 15:00）

1. **纪律原则**：盘中由券商系统自动触发条件单被动成交，**严禁盯盘追涨杀跌**。
2. **盘中仓位复核**：
   ```bash
   # 自动读取配置中的账户总资产和已有持仓，并优先尝试 Longbridge 盘中报价
   uv run stock size CN:601318

   # 持仓尚未写回配置时，可临时覆盖当前持股数
   uv run stock size CN:601318 --held-shares 300

   # 基本面敏感性推演
   uv run stock scenario CN:601318 --margin-delta 0.05
   ```
3. **执行口径**：
   - `盘中执行价` 只用于判断当前是否触发，以及估算已有持仓市值；
   - 行情源依次为 Longbridge OpenAPI、腾讯、AKShare、Yahoo，并显示实际来源和报价时间；
   - 报价超过 15 分钟或数据源失败时只作参考，系统会暂停下单判断；
   - `--price` 是离线复现参数，人工价格不再被标成券商实时行情，也不能恢复执行判断；
   - 真正下单前仍需在券商盘口核对买一/卖一、可用资金和已成交数量；
   - 三档挂单价继续锚定已完成的日线分析基准，不随盘中上涨而抬高；
   - `--capital` 表示账户总资产，不是可用现金；可用现金需在券商端另行核验；
   - 已有仓位达到目标上限时，系统输出 0 股并明确提示“不新增买入”。

### SOP 3：每日盘后全流程分析（18:30+）

1. **自动触发**：macOS 守护进程于 18:30 自动执行 `stock auto`。
2. **手动启动与 CLI 选择**：
   ```bash
   # 终端直接运行时默认进入菜单：选择全部/部分标的，以及运行模式
   uv run stock auto

   # 跳过菜单，直接按配置顺序轮流分析全部标的
   uv run stock auto --no-interactive

   # 非交互地只分析指定标的；--symbol 可重复
   uv run stock auto --symbol CN:601318 --symbol CN:601398

   # 快速模式：保留确定性分析，关闭 LLM 与 Chronos
   uv run stock auto --no-interactive --no-llm --no-chronos
   ```
   `-i` 是 `--interactive` 的缩写，通常无需再写；仅在管道或其他非交互环境中需要强制
   打开菜单时使用。指定 `--symbol`、启用 `--verbose`，或在 launchd/cron 中运行时，菜单
   默认关闭，避免后台任务等待输入。
   “完整”表示请求启用 LLM 与 Chronos；若 API 凭据、模型或证据不可用，标的的确定性分析
   仍会继续，但终端会计入“子任务失败/跳过”，详细原因写入运行日志，不会伪装成完整成功。
   在交互终端中默认显示“预检 → 逐只同步 → 回执 → 校准状态 → 逐只分析 → 组合报告”
   总进度条；可用 `--progress` 强制显示或 `--no-progress` 关闭。
3. **全部轮流分析口径**：
   - 默认读取 `stock-analysis.toml` 的全部 `[assets]`，按配置顺序逐只执行，不并行争抢数据库、模型内存或行情连接；
   - 单只失败会记录原因并继续下一只，不会中断整个批次；
   - 高成本 walk-forward 校准受每次运行名额和时间上限约束并按日期轮换，但每只同步成功的标的仍会完成当次多周期分析。
4. **执行链条**：
   `依赖检查 → 行情/分红增量同步 → 新闻与宏观刷新 → 到期回执自动核验 → 条件校准 → 多周期模型推演 → 图表渲染与 Obsidian 研报整理`
5. **成果查阅**：
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

| 命令                       | 用途与核心参数                                                                                 |
| -------------------------- | ---------------------------------------------------------------------------------------------- |
| `stock morning`          | 🌅**盘前交易纪律晨报**：全标的轮询；需要减仓时输出卖出股数并停止生成买入挂单，否则输出 3 阶买入网格 |
| `stock size SYMBOL`      | 🎯**实盘仓位精算器**：读取总资产和已有持仓，优先获取 Longbridge 盘中报价，输出买入、减仓或退出计划 |
| `stock compare S1 S2...` | ⚖️**跨标的比对矩阵**：横向比对估值折扣、多周期信号与优先建仓排序                       |
| `stock dash`             | 📊**终端决策看板**：基于本地日线数据呈现全资产四周期信号、数据日期与质量                  |
| `stock scenario SYMBOL`  | 🔬**What-If 情景推演**：模拟盈利预期调整与安全边际变化对买入价的影响                     |
| `stock add SYMBOL`       | ➕**向导式添加标的**：设定行业、组合角色（core/satellite）、估值模型与基准倍数           |
| `stock auto`             | ⚡**全流程自动分析**：默认轮流分析全部配置标的并显示进度；`-i` 菜单选择，`-s` 指定标的 |
| `stock analyze SYMBOL`   | 📝**单标的多周期分析**：`--horizon short\|medium\|long\|value\|all`                        |
| `stock evaluate`         | 🔍**到期回执核验与回测**：`--backtest SYMBOL --horizon-days 20` 执行严格样本外盲测     |
| `stock portfolio`        | 💼**组合风控硬约束检查**：核查单股上限、行业集中度、现金底线与组合回撤                   |
| `stock sync SYMBOL`      | 🔄**按需数据同步**：同步日线 OHLCV、公司分红除权除息与财务数据                           |
| `stock backfill`         | 📥**历史数据深度回填**：分块拉取 10 年以上原始历史数据并安全写入 SQLite                  |
| `stock doctor`           | 🩺**环境与依赖诊断**：检查 Python 3.12、数据库、Chronos 权重与 API 连通性                |

---

## ⚙️ 环境安装与定时任务部署

### 1. 基础环境配置

要求 Python 3.12 与 [uv](https://docs.astral.sh/uv/)。

```bash
git clone git@github.com:1917173927/SmartInvest.git
cd SmartInvest
cp stock-analysis.example.toml stock-analysis.toml

# 默认安装完整运行依赖：数据源、Chronos 时序模型与绘图引擎
# --extra dev 仅增加 Ruff、Pytest 等开发工具
uv sync --extra dev

# 验证运行环境
uv run stock doctor
```

此后运行 `uv run stock auto` 时，uv 会先核对锁文件与默认依赖；行情、Chronos 和图表组件
不再依赖额外的 `--extra` 参数。旧的 `data`、`forecast`、`charts` extra 名称仍保留兼容。
LLM 还需要 `.env` 中有效的 API 凭据；Python 依赖安装无法修复接口返回的 401。
LLM 与嵌入请求会对 DNS/连接错误、HTTP 429 和 5xx 进行有限指数退避重试；401 等配置错误
不会重试。`SEC_USER_AGENT` 只有填写真实联系方式后才会在 `stock doctor` 中显示为可用。
项目 `.env` 中的模型服务地址、模型名和凭据优先于终端中可能残留的同名环境变量，确保
手动 CLI 与 launchd 使用同一套 Key；其他非服务变量仍保留系统环境优先级。

### 2. macOS 盘中行情：Longbridge OpenAPI

本项目只调用 Longbridge 的只读行情命令，不调用资产、持仓或交易接口。OpenAPI 令牌由官方
CLI 保存在用户目录，不写入仓库或 `stock-analysis.toml`。

```bash
# 安装官方 CLI；新版 Homebrew 首次使用第三方 tap 时需要显式信任该 cask
brew tap longbridge/tap
brew trust --cask longbridge/tap/longbridge-terminal
brew install --cask longbridge-terminal

# 浏览器 OAuth 授权并核验状态
longbridge auth login --client-name SmartInvest
longbridge auth status

# 独立验证中国平安行情
longbridge quote 601318.SH --format json
longbridge intraday 601318.SH --format json
```

正常测算无需输入价格：

```bash
uv run stock size CN:601318
```

系统使用 `quote` 获取最新价，并以 `intraday` 返回的最新分钟时间校验新鲜度。CLI 未安装、
未登录、行情不可用或报价超过 15 分钟时自动尝试下一数据源；所有来源都失败时使用历史日线
且暂停执行判断。`longbridge auth status` 若显示 `CN_Basic / 15-min Delay`，代表当前账户没有
A 股实时权限，系统不会把这档行情标成实时；需在 Longbridge 开通实时权限或使用中国大陆
网络后重新核验。A 股适配器默认使用官方 `.cn` 接入点，无需在每次运行前手工设置
`LONGBRIDGE_REGION`；若显式设置该变量，则以用户配置为准。

首次实盘测算前，必须在本地 `stock-analysis.toml` 填写真实账户资产和当前持股数：

```toml
[portfolio]
cn_account_assets = 51546.80
cn_account_assets_as_of = "2026-09-01"

[assets."CN:601318"]
name = "中国平安"
role = "core"
current_shares = 300
```

真实配置已被 Git 忽略。账户资产或持仓变化后应及时更新；也可在单次测算中使用
`--capital`、`--held-shares` 临时覆盖。

`stock analyze`、`stock auto`、`stock dash`、`stock compare`、`stock size` 和 `stock morning`
会优先读取 `01-持仓` 中最新的持仓快照；只有快照缺失或无法解析完整总资产时，才回退到
`stock-analysis.toml`。`stock morning` 不传 `--capital` 时也使用这一口径，不再默认假设
账户总资产为 100,000 元。

当当前仓位超过核心/卫星仓位上限，或多周期判断触发“减仓/回避”“退出”时，报告会生成
独立的“减仓与退出计划”，包括当前仓位、纪律目标、建议卖出股数、卖出后持股和参考回笼
资金。A 股部分减仓按 100 股整手向上取整，清仓时允许卖出全部余股。触发退出计划后系统会
取消买入建议，避免同一份报告同时建议买入和卖出；系统不会自动下单，执行前仍须以券商
实时盘口复核可卖数量、未成交委托与成交回报。

执行口径是确定性的：仅超出角色上限时降到上限；“减仓/回避”默认降到当前仓位的一半
（可用 `risk.reduction_target_fraction` 调整）；“回避/重审退出”目标为零；单次跌破技术
失效线只会暂停买入并要求复核，不会直接触发清仓。

### 3. macOS 后台自动化守护进程一键安装

运行自带的配置脚本，会自动在 macOS 系统中注册两大无缝定时守护进程：

```bash
uv run python scripts/install_launchd.py
```

* **`com.stockanalysis.morning.plist`**：每个工作日 **09:00** 自动计算全标的挂单网格并推送桌面通知；
* **`com.stockanalysis.daily.plist`**：每日 **18:30** 自动执行盘后全流程数据同步与深度模型计算。

launchd 使用无交互入口，不显示终端进度条，也不会等待菜单输入；运行明细写入
`.stock-analysis/auto.log` 与 `06-自动分析/运行日志/`。

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

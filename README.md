# StockAnalysis

个人投资记录、概率预测与交易纪律知识库。目录可以直接作为 Obsidian Vault，也包含一个名为 `stock` 的 Python CLI。

系统的特色不是“让 AI 猜目标价”，而是：将带来源和日期的 LLM 事件因子接入 Chronos-2，保存每次预测回执，到期后用真实结果校准；同一标的分别给出短线、中线、长线和价值判断。

> 本项目用于个人研究与复盘，不构成收益承诺，不连接券商，不自动下单。

## 快速开始

要求 Python 3.12 与 [uv](https://docs.astral.sh/uv/)。

```bash
git clone git@github.com:1917173927/SmartInvest.git
cd SmartInvest
cp stock-analysis.example.toml stock-analysis.toml

# 完整本地能力：数据源、Chronos、静态图表与测试
uv sync --extra data --extra forecast --extra charts --extra dev

uv run stock doctor
uv run stock sync CN:601318
uv run stock backfill                 # 尝试补齐数据源可提供的全部历史
uv run stock analyze CN:601318 --horizon all --skip-chronos
uv run stock portfolio
uv run stock evaluate
# 首次建立 Chronos/基线校准样本（默认 20 日，需模型已缓存）
uv run stock calibrate --symbol CN:601318 --horizon-days 20 --max-windows 100 --step 1
# 无交互批量运行（读取 stock-analysis.toml 中的全部标的）
uv run stock auto
```

Chronos 默认模型为 `amazon/chronos-2`。首次实际运行会下载模型；若未安装、下载失败或被校准规则禁用，系统自动回退到随机游走与历史波动率基线。系统优先读取本地模型缓存；首次下载默认避开在部分个人网络中不稳定的 Xet 传输层。

## LLM 与证据

复制 `.env.example` 为项目根目录的 `.env` 并填写所需变量，程序会自动读取它；也可以继续使用终端环境变量。`.env` 已加入 `.gitignore`，不会提交。

```bash
export OPENAI_API_KEY="..."
export OPENAI_BASE_URL="https://api.openai.com/v1"
export STOCK_ANALYSIS_MODEL="你的 OpenAI-compatible 模型"
export STOCK_ANALYSIS_EMBEDDING_MODEL="可选嵌入模型"
```

使用 `.env` 时不需要每次重复 `export`；定时任务也会读取同一个文件。不要把密钥写入 Markdown 报告。

DeepSeek 账户可直接用于事件抽取对话模型，例如将 `OPENAI_BASE_URL` 设为
`https://api.deepseek.com`、`STOCK_ANALYSIS_MODEL` 设为 DeepSeek 控制台中可用的模型名。
`STOCK_ANALYSIS_EMBEDDING_MODEL` 可以留空；系统会自动使用 SQLite FTS。若另有向量服务，
可单独填写 `STOCK_ANALYSIS_EMBEDDING_BASE_URL`、`STOCK_ANALYSIS_EMBEDDING_API_KEY` 和对应模型，
不会影响 DeepSeek 对话配置。

Gemini 提供 `gemini-embedding-001`（文本）和 `gemini-embedding-2`（多模态）嵌入模型，
本项目已支持其原生批量嵌入接口。配置 `GEMINI_API_KEY` 与 `GEMINI_EMBEDDING_MODEL` 即可，
它只用于把公告/财报转换为向量并改善证据检索，不参与股价预测或估值算术。

LLM 不会自动浏览网络。把已经核验、带发布日期的 Markdown/TXT 传给分析命令：

```bash
uv run stock analyze CN:601318 \
  --horizon medium \
  --evidence ./研究资料/中国平安-2026Q2.md
```

证据文件优先使用如下 frontmatter：

```markdown
---
date: 2026-08-20
source_url: https://example.com/original-filing
---
```

LLM 只能生成白名单事件，且事件生效日不得早于证据发布日期。没有证据、来源或日期的内容不会成为时间模型协变量。

## 命令

- `stock doctor`：检查配置、数据库、可选依赖与 API。
- `stock sync SYMBOL`：按需同步原始日线、公司行动和可用财务事实。
- `stock backfill`：分块获取接口可提供的最早历史并安全 upsert；`--recent --years 10` 可限制范围。
- `stock analyze SYMBOL --horizon short|medium|long|value|all`：生成分析与预测回执。
- `stock portfolio`：读取最新 `01-持仓/*-持仓快照.md`，检查仓位、行业和现金硬约束。
- `stock evaluate`：核对到期回执。
- `stock auto`：无交互执行“依赖检查 → 条件同步/补缺 → 新闻宏观刷新 → 到期回执核对 → 条件校准 → 技术/多因素分析 → 图表与组合报告”；每项任务写入 SQLite 审计表，达到样本、TTL 或冷却条件后自动跳过。
- `stock evaluate --backtest CN:601318 --horizon-days 20`：执行至少三年历史的 walk-forward 基线回测。
- `stock evaluate --backtest CN:601318 --with-chronos`：显式把 Chronos 加入历史回测，耗时明显增加。
- 校准样本不足时可加 `--step 1 --max-windows 100 --with-chronos`；默认步长等于预测期限，样本会较少但窗口不重叠。
- `stock calibrate`：先补齐历史数据，再按指定标的/期限执行 walk-forward 校准；它不微调模型参数，也不把未来数据带入历史窗口。

报告写入 `06-自动分析`，个股文件名为“日期-市场代码-股票名称-周期”，图表位于对应个股的 `charts/`。技术图包含 ATR 聚类支撑/压力带、K 线、MA、成交量、MACD 和 RSI；概率图单独展示 5/10/20/60/120 日中位路径、名义 80% 与近似 50% 分位区间。行情、新闻、宏观、回执、校准与自动任务状态统一保存在 `.stock-analysis/analysis.sqlite3`。运行数据默认不提交 Git。

### macOS 定时运行

`scripts/com.stockanalysis.daily.plist.template` 是每天 18:30（本地时间）运行的 launchd 模板。安装脚本会自动代入当前项目与 Python 路径：

```bash
uv run python scripts/install_launchd.py
```

日志和失败详情写入 `.stock-analysis/auto.log`、`.stock-analysis/launchd.*.log`；任务只生成研究报告，不会自动下单。
如需在网络或模型不可用时强制轻量运行，可在 launchd 环境中设置
`STOCK_AUTO_USE_CHRONOS=0`、`STOCK_AUTO_USE_LLM=0`；默认值读取 `[automation]` 配置。

## 数据边界

- AKShare 是 A/H 股和基金的首选适配器；A 股在东财接口异常时自动尝试腾讯/新浪接口，最后才回退到 yfinance。所有接口都属于研究数据接口，不是交易所级行情保证。
- 个股新闻使用 AKShare 暴露的东方财富公开接口，必须带发布日期和来源 URL；宏观/商品默认覆盖 WTI、黄金、美元、美债 10 年期、SHIBOR 和沪深 300，并以 yfinance 或 AKShare 保存到本地缓存。
- 美股财务事实可通过 SEC EDGAR 获取。设置 `SEC_USER_AGENT="项目名 联系邮箱"` 后启用。
- 原始 OHLCV 与分红/拆股事件分开保存。回测不会直接采用可能随未来公司行动变化的动态前复权快照。
- 数据少于约三年、超过十天未更新或出现无法解释的巨大跳变时降为 C 级；C 级数据不会触发买入建议。
- 免费接口结构可能变化，适配器失败时保留缓存并在报告中标记，不静默伪造数据。

## 投资与风险规则

`stock-analysis.toml` 保存已确认的个人参数：最大组合回撤 25%、核心单股 35%、卫星仓 15%、行业 60%、现金底线 5%、短线单笔风险 1%。该文件和个人持仓均被 Git 忽略；仓库只提供脱敏的 `stock-analysis.example.toml`。

决策层遵循“先好公司、再好价格”：质量、治理、现金流不合格时，低估值不能触发买入；PE、PB、股息率交叉验证，买入价必须带安全边际；仓位硬约束不会被 LLM 或 Chronos 覆盖。

支撑/压力强度是局部高低点触达、成交量、ATR 与时间衰减形成的启发式分数，不是守住概率，也不直接提高买入评分。概率扇形图只在明确的预测期限上使用模型输出，中间交易日是可视化插值；实验期限未完成区间覆盖率校准时必须结合状态标签阅读。

## Obsidian 工作流

1. 从 [[00-交易总览]] 开始。
2. 自动运行后先读 [[06-自动分析/最新摘要]]，再按需打开个股、组合或回测详情。
3. 操作前用 [[05-模板/操作记录模板]] 写逻辑、风险和失效条件。
4. 成交后补录均价、数量和计划偏差。
5. 每周复盘纪律，每月更新持仓快照；自动任务会持续整理报告目录。

架构与扩展边界参见 [ARCHITECTURE.md](ARCHITECTURE.md)。

# 架构与维护边界

## 数据流

```text
Longbridge（只读盘中报价）/ 腾讯 / AKShare / 东方财富 / yfinance / SEC
          │
          ▼
data：原始行情、公司行动、财务事实、新闻/宏观、时间点校验、SQLite
          │
          ├──────────────┬────────────────┐
          ▼              ▼                ▼
forecast             research        indicators
Chronos-2 + 基线      FTS/向量 + LLM   MA/MACD/RSI/宏观
          │              │                │
          └──────────────┴───────┬────────┘
                                 ▼
decision：质量、估值、多因素评分、安全边际、硬约束、多周期冲突卡
                                 │
                         ┌───────┴────────┐
                         ▼                ▼
                    charts            cli/automation
                    静态图表           报告、回执、任务审计
```

`src/stock_analysis` 保留五个核心职责，并用三个小型支持模块隔离可选能力：

- `data.py`：唯一的数据与持久化边界；Longbridge CLI 仅用于读取行情，不读取账户或调用交易接口。
- `forecast.py`：概率预测、集成权重、回执评估与 walk-forward。
- `research.py`：证据入库、全文/向量检索、LLM 事件校验。
- `decision.py`：确定性规则、风险门槛、组合解析和报告模型。
- `cli.py` 与 `automation.py`：编排、无交互批处理与用户接口，不承载投资算法；`automation.py` 仅是 CLI 的可复用运行器。
- `context.py`：公开新闻、商品与宏观序列的 TTL 刷新，不执行投资判断。
- `indicators.py`：纯函数式技术指标、宏观暴露与 ATR 枢轴区间，禁止网络访问。
- `charts.py`：可选 Matplotlib 静态输出；技术图和概率扇形图分离，不改变模型结果。

`__init__.py` 与 `__main__.py` 只负责包入口。

## 不变量

1. 历史分析只能读取 `published_at <= as_of` 的文档和 `trade_date <= as_of` 的行情。
2. LLM 输出不是事实来源；每个事件必须引用本地证据 ID。
3. LLM 不做估值算术；合理价值由确定性代码或显式人工区间计算。
4. Chronos 只输出概率分布，不产生自动交易指令。
5. 数据 C 级、资金非余钱、使用杠杆或突破仓位硬约束时，模型分数不能触发买入。
6. 模型失败必须可降级，基础 CLI 不依赖 PyTorch、Chronos 或 API。
7. SQLite 是事实与审计源；Markdown/SVG 是可重建的阅读层，不反向覆盖数据库。
8. 每次自动运行及其 `executed/skipped/failed` 任务都必须有唯一运行编号。
9. 支撑/压力强度不得表述为概率；插值概率路径不得伪装成逐日模型输出。

## 扩展方式

- 新数据源：实现 `MarketDataProvider` 或独立上下文适配器，不得绕过原始数据与来源字段。
- 新预测模型：实现 `Forecaster`，先加入 walk-forward，再考虑获得非零权重。
- 新事件类型：同时修改枚举、配置白名单、提示词、校验与测试。
- 新估值模型：在 `decision.py` 增加行业分支；不得用一个通用 PE 规则覆盖银行、保险和周期股。

## 明确不做

v1 不做全市场扫描、盘中分钟线存储、网页界面、模型微调、强化学习、多 Agent、TimesFM、独立向量数据库、券商账户连接或自动下单。Longbridge 分时数据只用于校验当前报价时间，不写入历史数据库；操作系统定时器只负责启动 `stock auto`，不进入核心业务层。

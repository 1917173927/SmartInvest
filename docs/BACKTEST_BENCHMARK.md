# SmartInvest 量化回测基准规范与复现报告模板 (Backtest Benchmark Protocol)

> **版本**：v1.0  
> **适用范围**：所有模型升级、因子引入或策略参数变更的样本外（Out-of-Sample）回测验收与审计。  
> **核心原则**：严格时点隔离（Point-in-Time）、非重叠滚动窗口（Purged Walk-Forward）、包含摩擦成本、提供置信区间与可复现哈希。

---

## 1. 回测设计与环境规范

所有在 SmartInvest 中引用的胜率、盈亏比与收益率指标，必须基于以下标准化环境进行统计与复现：

| 维度 | 标准规范 | 说明 |
|---|---|---|
| **回测方法** | **Purged Walk-Forward OOS** | 步进式滚动样本外盲测，训练/预测/评估单向流动 |
| **测试窗口步长** | `stride = horizon_days` | **非重叠步长**（例如 20 日预测期限则步长为 20 日），消除自相关性与数据污染 |
| **最小样本量** | $\ge 60$ 窗口（推荐 $\ge 100$） | 保证统计大样本性质，禁止仅凭 5~10 次交易宣称高胜率 |
| **交易摩擦成本** | 单边佣金 $0.025\%$ + 印花税 $0.05\%$ (卖出) + 滑点 $0.10\%$ | 往返交易摩擦总计扣除 $0.20\%$ |
| **置信区间** | **Wilson 95% 双侧置信区间** | 使用 Wilson Score Interval，完整公式见第 2 节 |
| **基线对照组** | 1) 随机游走基线 (Random Walk)<br>2) 买入持有基线 (Buy & Hold) | 模型策略期望收益与夏普比率必须显著战胜基线 |

---

## 2. 统计指标与公式定义

1. **样本外胜率 (Out-of-Sample Win Rate)**：
   $$\text{Win Rate} = \frac{\sum_{i=1}^N \mathbb{I}(R_{\text{actual}, i} \times R_{\text{pred}, i} > 0)}{N}$$
2. **盈亏比 (Profit Factor)**：
   $$\text{Profit Factor} = \frac{\sum \text{Gains}}{\sum |\text{Losses}|}$$
3. **单笔期望收益率 (Expected Value per Trade, EV)**：
   $$\text{EV} = (\text{Win Rate} \times \bar{R}_{\text{win}}) - ((1 - \text{Win Rate}) \times |\bar{R}_{\text{loss}}|)$$
4. **Wilson 95% 置信区间 (Wilson Score Interval)**：
   $$CI_{95\%} = \frac{\hat{p} + \frac{z^2}{2n} \pm z \sqrt{\frac{\hat{p}(1-\hat{p})}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}} \quad (z = 1.96)$$

---

## 3. 标准回测报告模版 (Benchmark Report Template)

> [!note] 模版说明
> 以下表格内容为**报告结构与格式规范占位符（Placeholder）**，并非全标的固化结论。针对具体标的与特定时间范围的真实审计结果，须按第 4 节命令独立运行后据实填入。

```markdown
# 📊 [标的代码/名称] 样本外回测基准报告 (示例模版)

## 📌 实验元数据与可复现哈希 (Reproducibility Metadata)
- **回测标的**：`[示例: CN:601318 中国平安]`
- **数据时间跨度**：`[示例: 2020-01-01 至 2026-08-31]`
- **预测期限 (Horizon)**：`[示例: 20 交易日]`
- **Git Commit SHA**：`[当前提交哈希]`
- **配置哈希 (Config SHA-256)**：`[stock-analysis.toml 的 SHA-256]`
- **数据源版本**：AKShare / yfinance + 本地 SQLite 缓存
- **运行命令**：`uv run stock evaluate --backtest <SYMBOL> --horizon-days <DAYS>`

## 📈 核心指标对照表 (vs 基线) [占位示例]
| 指标 | 本策略 (SmartInvest Multi-Factor) | 买入持有基线 (Buy & Hold) | 随机游走基线 (Random Walk) |
|---|---:|---:|---:|
| **样本外窗口数 (N)** | [示例: 60 个非重叠窗口] | [示例: 60 个非重叠窗口] | [示例: 60 个非重叠窗口] |
| **样本外胜率 (Win Rate)** | **[待测算 %]** | [基线 %] | [基线 %] |
| **95% 置信区间 (Wilson CI)** | **[[下界 %, 上界 %]]** | [[下界 %, 上界 %]] | [[下界 %, 上界 %]] |
| **盈亏比 (Profit Factor)** | **[待测算]** | [基线] | [基线] |
| **单笔平均期望收益 (EV)** | **[待测算 %]** | [基线 %] | [基线 %] |
| **最大回撤 (Max Drawdown)** | **[待测算 %]** | [基线 %] | [基线 %] |
| **年化夏普比率 (Sharpe)** | **[待测算]** | [基线] | [基线] |

## 🛡️ 摩擦成本与滑点扣除说明
- 往返佣金与税费：按 0.10% 扣除
- 冲击滑点：按 0.10% 扣除
```

---

## 4. 独立复现指南 (Step-by-Step Reproduction)

### 场景 A：离线确定性复现（完全基于本地已有 SQLite 历史数据，无外部 API 依赖）
当本地数据库已包含所需历史数据时，可直接进行纯离线、确定性的样本外推演：

```bash
# 1. 检出待评估的 Git 提交
git checkout <COMMIT_SHA>

# 2. 执行确定性非重叠 Walk-Forward 样本外回测（步长等于预测期限）
uv run stock evaluate --backtest CN:601318 --horizon-days 20 --step 20
```

### 场景 B：在线全量刷新与补齐（需要外部行情数据源接口）
若需要在全新环境中拉取最新交易所历史行情，执行：

```bash
# 1. 同步日线与分红除权事实
uv run stock sync CN:601318

# 2. 分块回填历史数据
uv run stock backfill --recent --years 5

# 3. 运行样本外回测
uv run stock evaluate --backtest CN:601318 --horizon-days 20 --step 20
```

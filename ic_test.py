"""
简单因子回测框架
目标：多股票、多因子，输出IC/ICIR/收益曲线
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
# 1. 配置
# ─────────────────────────────────────────

TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "TSLA", "NVDA", "JPM", "GS", "BAC",
    "JNJ", "PFE", "UNH", "XOM", "CVX",
    "WMT", "HD", "MCD", "KO", "PEP"
]

START = "2024-01-01"
END   = "2026-04-01"
# START = "2019-01-01"
# END   = "2023-12-31"

# ─────────────────────────────────────────
# 2. 下载数据
# ─────────────────────────────────────────

print("正在下载数据...")
raw = yf.download(TICKERS, start=START, end=END, auto_adjust=True)
close  = raw["Close"].dropna(how="all")
volume = raw["Volume"].dropna(how="all")
print(f"数据下载完成：{close.shape[0]} 个交易日，{close.shape[1]} 只股票\n")


# ─────────────────────────────────────────
# 3. 定义因子
#    每个因子是一个函数：接收 close/volume DataFrame，返回同形状的因子值 DataFrame
# ─────────────────────────────────────────

def factor_reversal_5(close, volume):
    """5天价格反转：过去5天跌得多的给高分"""
    return -close.diff(5)

def factor_reversal_20(close, volume):
    """20天价格反转"""
    return -close.diff(20)

def factor_momentum_20(close, volume):
    """20天动量：过去20天涨得多的给高分"""
    return close.pct_change(20)

def factor_volume_spike(close, volume):
    """成交量异常放大：今天成交量 / 过去20天均量"""
    return volume / volume.rolling(20).mean()

def factor_vol_adjusted_reversal(close, volume):
    """波动率调整后的反转：反转信号 / 过去10天波动率"""
    reversal = -close.diff(5)
    volatility = close.pct_change().rolling(10).std()
    return reversal / (volatility + 1e-8)


FACTORS = {
    "reversal_5d":           factor_reversal_5,
    "reversal_20d":          factor_reversal_20,
    "momentum_20d":          factor_momentum_20,
    "volume_spike":          factor_volume_spike,
    "vol_adjusted_reversal": factor_vol_adjusted_reversal,
}


# ─────────────────────────────────────────
# 4. 计算未来收益率（明天的收益）
# ─────────────────────────────────────────

returns_next = close.pct_change(1).shift(-1)


# ─────────────────────────────────────────
# 5. 回测函数
# ─────────────────────────────────────────

def backtest_factor(factor_values, returns_next):
    """
    输入：
        factor_values : DataFrame，每天每只股票的因子值
        returns_next  : DataFrame，每天每只股票明天的收益率
    输出：
        summary : dict，包含平均IC、ICIR等
        daily_ic: Series，每天的IC
    """
    # 每天截面IC
    daily_ic = factor_values.corrwith(returns_next, axis=1)
    daily_ic = daily_ic.dropna()

    mean_ic = daily_ic.mean()
    std_ic  = daily_ic.std()
    icir    = mean_ic / std_ic if std_ic != 0 else 0
    ic_positive_rate = (daily_ic > 0).mean()

    # 逐年IC
    yearly_ic = daily_ic.groupby(daily_ic.index.year).mean()

    summary = {
        "平均IC":      round(mean_ic, 4),
        "IC标准差":    round(std_ic, 4),
        "ICIR":        round(icir, 4),
        "IC>0比例":    f"{ic_positive_rate:.1%}",
        "逐年IC":      yearly_ic.round(4).to_dict(),
    }

    return summary, daily_ic


# ─────────────────────────────────────────
# 6. 运行所有因子
# ─────────────────────────────────────────

results   = {}
daily_ics = {}

print("=" * 55)
print(f"{'因子':<25} {'平均IC':>8} {'ICIR':>8} {'IC>0':>8}")
print("=" * 55)

for name, fn in FACTORS.items():
    factor_values = fn(close, volume)
    summary, daily_ic = backtest_factor(factor_values, returns_next)
    results[name]   = summary
    daily_ics[name] = daily_ic

    print(f"{name:<25} {summary['平均IC']:>8.4f} {summary['ICIR']:>8.4f} {summary['IC>0比例']:>8}")

print("=" * 55)


# ─────────────────────────────────────────
# 7. 打印逐年IC
# ─────────────────────────────────────────

print("\n逐年平均IC：")
years = sorted({y for r in results.values() for y in r["逐年IC"].keys()})
header = f"{'因子':<25}" + "".join(f"{y:>8}" for y in years)
print(header)
print("-" * len(header))
for name, summary in results.items():
    row = f"{name:<25}"
    for y in years:
        val = summary["逐年IC"].get(y, float("nan"))
        row += f"{val:>8.4f}"
    print(row)


# ─────────────────────────────────────────
# 8. 画图
# ─────────────────────────────────────────

fig, axes = plt.subplots(2, 1, figsize=(14, 10))
fig.suptitle("Factor IC Analysis", fontsize=16, fontweight="bold")

# 图1：每个因子的IC累积曲线
ax1 = axes[0]
for name, daily_ic in daily_ics.items():
    cumulative_ic = daily_ic.cumsum()
    ax1.plot(cumulative_ic.index, cumulative_ic.values, label=name, linewidth=1.5)
ax1.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax1.set_title("Cumulative IC over Time")
ax1.set_ylabel("Cumulative IC")
ax1.legend(fontsize=8)
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax1.grid(alpha=0.3)

# 图2：逐年IC柱状图
ax2 = axes[1]
yearly_df = pd.DataFrame({
    name: summary["逐年IC"]
    for name, summary in results.items()
}).T
yearly_df.plot(kind="bar", ax=ax2, width=0.7)
ax2.axhline(0, color="black", linewidth=0.8)
ax2.set_title("Yearly IC by Factor")
ax2.set_ylabel("IC")
ax2.set_xlabel("")
ax2.legend(fontsize=8)
ax2.tick_params(axis="x", rotation=0)
ax2.grid(alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("factor_analysis.png", dpi=150, bbox_inches="tight")
print("\n图表已保存为 factor_analysis.png")
plt.show()
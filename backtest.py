"""
Multi-stock, multi-factor backtesting framework.
Metrics: IC, RankIC, ICIR, RankICIR, IC>0 Rate
"""

import warnings

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────
# 1. Configuration
# ─────────────────────────────────────────

TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "TSLA", "NVDA", "JPM", "GS", "BAC",
    "JNJ", "PFE", "UNH", "XOM", "CVX",
    "WMT", "HD", "MCD", "KO", "PEP",
]

START = "2019-01-01"
END   = "2023-12-31"


# ─────────────────────────────────────────
# 2. Data download
# ─────────────────────────────────────────

print("Downloading data...")
raw    = yf.download(TICKERS, start=START, end=END, auto_adjust=True)
close  = raw["Close"].dropna(how="all")
volume = raw["Volume"].dropna(how="all")
print(f"Done: {close.shape[0]} trading days, {close.shape[1]} stocks\n")


# ─────────────────────────────────────────
# 3. Factor definitions
#    Each factor: (close, volume) -> DataFrame of cross-sectional scores
#    Higher score = predict stock will outperform tomorrow
# ─────────────────────────────────────────

def factor_reversal_5(close, volume):
    """5-day price reversal: stocks that fell more get higher scores"""
    return -close.diff(5)


def factor_reversal_20(close, volume):
    """20-day price reversal"""
    return -close.diff(20)


def factor_momentum_20(close, volume):
    """20-day momentum: stocks that rose more get higher scores"""
    return close.pct_change(20)


def factor_volume_spike(close, volume):
    """Volume spike: today's volume relative to 20-day average"""
    return volume / volume.rolling(20).mean()


def factor_vol_adjusted_reversal(close, volume):
    """Volatility-adjusted reversal: reversal signal normalized by 10-day realized vol"""
    reversal   = -close.diff(5)
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
# 4. Forward returns (next-day)
# ─────────────────────────────────────────

returns_next = close.pct_change(1).shift(-1)


# ─────────────────────────────────────────
# 5. Backtesting function
# ─────────────────────────────────────────

def backtest_factor(factor_values: pd.DataFrame, returns_next: pd.DataFrame) -> tuple:
    """
    Compute IC, RankIC, ICIR, RankICIR and yearly breakdowns.

    Args:
        factor_values : cross-sectional factor scores (dates x stocks)
        returns_next  : next-day returns aligned to factor_values

    Returns:
        summary      : dict of aggregate metrics
        daily_ic     : Series of daily Pearson IC values
        daily_rankic : Series of daily Spearman RankIC values
    """
    # ── IC (Pearson) ──────────────────────────────────────────
    daily_ic = factor_values.corrwith(returns_next, axis=1).dropna()
    mean_ic  = daily_ic.mean()
    std_ic   = daily_ic.std()
    icir     = mean_ic / std_ic if std_ic > 1e-8 else 0.0
    ic_pos   = (daily_ic > 0).mean()

    # ── RankIC (Spearman) ─────────────────────────────────────
    # Rank stocks cross-sectionally each day (pct=True -> uniform [0,1] scale)
    # then compute Pearson correlation on the ranks
    factor_ranked  = factor_values.rank(axis=1, pct=True)
    returns_ranked = returns_next.rank(axis=1, pct=True)
    daily_rankic   = factor_ranked.corrwith(returns_ranked, axis=1).dropna()
    mean_rankic    = daily_rankic.mean()
    std_rankic     = daily_rankic.std()
    rankicir       = mean_rankic / std_rankic if std_rankic > 1e-8 else 0.0
    rankic_pos     = (daily_rankic > 0).mean()

    # ── Yearly breakdown ──────────────────────────────────────
    yearly_ic     = daily_ic.groupby(daily_ic.index.year).mean().round(4).to_dict()
    yearly_rankic = daily_rankic.groupby(daily_rankic.index.year).mean().round(4).to_dict()

    summary = {
        "IC":              round(mean_ic, 4),
        "RankIC":          round(mean_rankic, 4),
        "ICIR":            round(icir, 4),
        "RankICIR":        round(rankicir, 4),
        "IC>0":            f"{ic_pos:.1%}",
        "RankIC>0":        f"{rankic_pos:.1%}",
        "Yearly_IC":       yearly_ic,
        "Yearly_RankIC":   yearly_rankic,
    }

    return summary, daily_ic, daily_rankic


# ─────────────────────────────────────────
# 6. Run all factors
# ─────────────────────────────────────────

results       = {}
daily_ics     = {}
daily_rankics = {}

print("=" * 75)
print(f"{'Factor':<25} {'IC':>8} {'RankIC':>8} {'ICIR':>8} {'RankICIR':>10} {'IC>0':>7}")
print("=" * 75)

for name, fn in FACTORS.items():
    factor_values                   = fn(close, volume)
    summary, daily_ic, daily_rankic = backtest_factor(factor_values, returns_next)
    results[name]                   = summary
    daily_ics[name]                 = daily_ic
    daily_rankics[name]             = daily_rankic

    print(
        f"{name:<25} "
        f"{summary['IC']:>8.4f} "
        f"{summary['RankIC']:>8.4f} "
        f"{summary['ICIR']:>8.4f} "
        f"{summary['RankICIR']:>10.4f} "
        f"{summary['IC>0']:>7}"
    )

print("=" * 75)


# ─────────────────────────────────────────
# 7. Yearly breakdown tables
# ─────────────────────────────────────────

years  = sorted({y for r in results.values() for y in r["Yearly_IC"].keys()})
header = f"{'Factor':<25}" + "".join(f"{y:>8}" for y in years)
divider = "-" * len(header)

print("\nYearly IC:")
print(header)
print(divider)
for name, summary in results.items():
    row = f"{name:<25}"
    for y in years:
        row += f"{summary['Yearly_IC'].get(y, float('nan')):>8.4f}"
    print(row)

print("\nYearly RankIC:")
print(header)
print(divider)
for name, summary in results.items():
    row = f"{name:<25}"
    for y in years:
        row += f"{summary['Yearly_RankIC'].get(y, float('nan')):>8.4f}"
    print(row)


# ─────────────────────────────────────────
# 8. Plots (2x2 grid)
#    Top row    : cumulative IC / RankIC curves
#    Bottom row : yearly IC / RankIC bar charts
# ─────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Factor IC Analysis", fontsize=16, fontweight="bold")

date_fmt = mdates.DateFormatter("%Y")

# Top-left: Cumulative IC
ax = axes[0, 0]
for name, daily_ic in daily_ics.items():
    cum = daily_ic.cumsum()
    ax.plot(cum.index, cum.values, label=name, linewidth=1.5)
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_title("Cumulative IC (Pearson)")
ax.set_ylabel("Cumulative IC")
ax.legend(fontsize=7)
ax.xaxis.set_major_formatter(date_fmt)
ax.grid(alpha=0.3)

# Top-right: Cumulative RankIC
ax = axes[0, 1]
for name, daily_rankic in daily_rankics.items():
    cum = daily_rankic.cumsum()
    ax.plot(cum.index, cum.values, label=name, linewidth=1.5)
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_title("Cumulative RankIC (Spearman)")
ax.set_ylabel("Cumulative RankIC")
ax.legend(fontsize=7)
ax.xaxis.set_major_formatter(date_fmt)
ax.grid(alpha=0.3)

# Bottom-left: Yearly IC bar chart
ax = axes[1, 0]
yearly_ic_df = pd.DataFrame(
    {name: s["Yearly_IC"] for name, s in results.items()}
).T
yearly_ic_df.plot(kind="bar", ax=ax, width=0.7)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_title("Yearly IC by Factor")
ax.set_ylabel("IC")
ax.set_xlabel("")
ax.legend(fontsize=7)
ax.tick_params(axis="x", rotation=15)
ax.grid(alpha=0.3, axis="y")

# Bottom-right: Yearly RankIC bar chart
ax = axes[1, 1]
yearly_rankic_df = pd.DataFrame(
    {name: s["Yearly_RankIC"] for name, s in results.items()}
).T
yearly_rankic_df.plot(kind="bar", ax=ax, width=0.7)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_title("Yearly RankIC by Factor")
ax.set_ylabel("RankIC")
ax.set_xlabel("")
ax.legend(fontsize=7)
ax.tick_params(axis="x", rotation=15)
ax.grid(alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("factor_analysis.png", dpi=150, bbox_inches="tight")
print("\nChart saved as factor_analysis.png")
plt.show()
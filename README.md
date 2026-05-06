# Alpha Research: LLM-Driven Quantitative Factor Mining

A quantitative factor research framework that combines traditional backtesting with LLM-driven factor generation. Inspired by [CogAlpha (Liu et al., 2025)](https://arxiv.org/abs/2511.18850).

---

## Key Finding

> **Market regime drives factor effectiveness.**
> Momentum factors dominated in trending markets (2019–2023), while reversal factors recovered strongly after Trump's tariff-driven volatility in 2025–2026. The LLM-generated factors independently confirmed this pattern — the best factor's IC improved from 0.0035 (2024) to 0.0225 (2025).

---

## Results

### Factor IC Comparison: 2019–2023 vs 2024–2026

| Period | Best Factor | Mean IC | ICIR |
|--------|------------|---------|------|
| 2019–2023 | momentum_20d | +0.030 | +0.38 |
| 2024–2026 | reversal_20d | +0.037 | +0.45 |

**2019–2023: Momentum dominates**

![Factor IC Analysis 2019-2023](results/factor_analysis_2019_2023.png)

- `momentum_20d` cumulative IC reached 25+, far outperforming all other factors
- Reversal factors near zero — consistent with the 2021 bull market suppressing mean-reversion signals

**2024–2026: Regime shift after Liberation Day**

![Factor IC Analysis 2024-2026](results/factor_analysis_2024_2026.png)

- `reversal_20d` IC surged to **0.037** in 2026 (highest across all periods)
- `momentum_20d` IC turned **negative** — trend-following broke down under policy uncertainty
- Pattern aligns with S&P 500 dropping ~10% in April 2025, then rallying sharply after tariff pause

### LLM Factor Search Results

3 rounds × 3 factors = 9 candidates generated and backtested automatically.

| Rank | Factor | IC | ICIR | Round |
|------|--------|----|------|-------|
| 1 | volume_surge_price_recovery_signal | +0.0136 | +0.0375 | R3 |
| 2 | volume_price_divergence_reversal | +0.0128 | +0.0360 | R2 |
| 3 | price_momentum_volume_confirmation | -0.0093 | -0.0311 | R2 |

**Best Factor:**
```
((close / close.rolling(10).mean() - 1) * -1) * (volume / volume.rolling(20).mean()).rolling(3).mean().rank(axis=1, pct=True)
```
Logic: Stocks with larger deviation below 10-day MA (oversold) AND sustained volume increase over 3 days → higher probability of institutional accumulation and mean reversion.

**Iterative improvement worked:** Round 1 factors ranked 8–9; Round 3 factors ranked 1–2. The LLM successfully refined signals across iterations.

---

## Architecture

```
Market Hypothesis
      │
      ▼
┌─────────────────────┐
│   LLM Factor        │  claude-sonnet-4-6
│   Generator         │  Generates factor expressions + logic
└─────────┬───────────┘
          │ factor expressions
          ▼
┌─────────────────────┐
│   Backtesting       │  yfinance + pandas
│   Engine            │  Computes IC, ICIR, yearly breakdown
└─────────┬───────────┘
          │ IC / ICIR results
          ▼
┌─────────────────────┐
│   Feedback Loop     │  Best factors fed back to LLM
│                     │  LLM analyzes and refines
└─────────┬───────────┘
          │ improved factors
          ▼
    Next Round (×3)
```

---

## Project Structure

```
alpha-research/
├── README.md
├── backtest.py            # Multi-factor backtesting framework
├── llm_factor_search.py   # LLM-driven factor generation loop
└── results/
    ├── factor_analysis_2019_2023.png
    ├── factor_analysis_2024_2026.png
    └── factor_search_*.json
```

---

## How to Run

**1. Install dependencies**
```bash
pip install anthropic yfinance pandas numpy matplotlib
```

**2. Set API key**
```bash
export ANTHROPIC_API_KEY="your-key-here"
```

**3. Run traditional backtesting**
```bash
python backtest.py
```

**4. Run LLM factor search**
```bash
python llm_factor_search.py
```

---

## Methodology

### Factor Evaluation Metrics

| Metric | Formula | Threshold |
|--------|---------|-----------|
| IC (Information Coefficient) | `corr(factor, next_day_return)` | > 0.02 |
| ICIR | `mean(IC) / std(IC)` | > 0.3 |
| IC > 0 Rate | Fraction of days with positive IC | > 52% |

### Factors Studied

| Factor | Expression | Logic |
|--------|-----------|-------|
| reversal_5d | `-close.diff(5)` | 5-day price reversal |
| reversal_20d | `-close.diff(20)` | 20-day price reversal |
| momentum_20d | `close.pct_change(20)` | 20-day momentum |
| volume_spike | `volume / volume.rolling(20).mean()` | Abnormal volume |
| vol_adjusted_reversal | `-close.diff(5) / close.pct_change().rolling(10).std()` | Volatility-normalized reversal |

### LLM Iteration Design

Each round feeds prior results back to the LLM with explicit guidance:
- Which factor performed best and why
- Direction to improve vs. explore new signals
- Constraints to avoid degenerate expressions

---

## Key Observations

**1. Regime switching is real and measurable**
The same factor can go from IC=+0.030 to IC=-0.020 depending on market conditions. Ignoring regime is the most common mistake in naive backtesting.

**2. LLM iteration improves factor quality**
Round 1 ICIR averaged ~0.01. Round 3 ICIR reached 0.037. Structured feedback consistently pushed the search in productive directions.

**3. Volume + price combinations outperformed pure price signals**
The top 2 LLM-generated factors both combined price deviation and volume patterns, suggesting institutional flow signals add incremental information beyond price alone.

---

## References

- Liu et al. (2025). *CogAlpha: Cognitive Alpha Mining with LLM-based Multi-Agent Framework*. arXiv:2511.18850
- Kakushadze (2016). *101 Formulaic Alphas*. arXiv:1601.00991
- Wikipedia: [2025 stock market crash](https://en.wikipedia.org/wiki/2025_stock_market_crash)

---

## Author

Jamie Ren · Statistics & Physics (University of Toronto) · M.S. Information Science (Trine University)

*Built as part of quantitative research skill development for quant researcher roles.*
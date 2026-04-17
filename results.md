# 📊 Results — Bitcoin (BTC-USD)

**Asset:** Bitcoin (BTC-USD)  
**Source:** btc-usd-max.csv (CoinGecko historical export)  
**Period:** 2013-04-28 → 2026-04-17 (4,737 days)  
**Observations:** 4,735 log return data points  

---

## 🔹 Descriptive Statistics

| Statistic | Value |
|-----------|-------|
| Mean | 0.001335 |
| Std Dev | 0.037455 |
| Min | -0.433714 |
| Max | 0.287099 |
| Skewness | -0.4934 |
| Kurtosis | 9.3778 (excess; Normal = 0) |

> **Key insight:** Excess kurtosis of 9.38 is extreme — nearly 10× the normal distribution. Bitcoin exhibits dramatically fat tails, meaning crashes and rallies of ±20-40% occur far more often than any Gaussian model predicts. The negative skewness (-0.49) indicates crashes tend to be sharper than rallies.

---

## 🔹 ADF Test (Augmented Dickey-Fuller)

| Series | ADF Statistic | p-value | Result |
|--------|-------------|---------|--------|
| Prices | -1.0225 | 0.7450 | Non-stationary (unit root) ✔ |
| Log Returns | -18.4751 | 0.0000 | Stationary ✔ |

**Conclusion:**
- Prices are non-stationary with p = 0.745 → they wander freely (supports RWH) ✔
- Returns are stationary with ADF = -18.47 → strong mean-reversion in returns ✔
- **Both results support the Random Walk framework**

---

## 🔹 Autocorrelation (Ljung-Box Q Test)

- Significant lags: **Lags 6 through 30** (cumulative Q-test p < 0.05)
- First significant lag: **Lag 6** (Q = 16.03, p = 0.0136)
- Lags 1–5: individually **not significant** (near-term returns appear Independent)

**Conclusion:**
- **Autocorrelation detected at medium lags** ✘ (Against RWH)
- Short-term (1–5 day) returns show no significant dependence
- Beyond 1 week, cumulative serial correlations become statistically significant
- This suggests weekly/multi-day momentum structures in BTC pricing

---

## 🔹 Runs Test (Wald-Wolfowitz)

| Metric | Value |
|--------|-------|
| Z-statistic | 3.0376 |
| p-value | 0.0024 |

**Conclusion:**
- p = 0.0024 ≪ 0.05 → **Strongly rejects randomness** ✘
- Positive Z = 3.04 indicates **fewer runs than expected** — consecutive same-sign returns cluster together
- This is the statistical signature of **momentum behavior**: up-days tend to follow up-days, down-days follow down-days
- **Rejects the Random Walk Hypothesis**

---

## 🔹 Variance Ratio Test (Lo-MacKinlay 1988)

| Horizon (k) | VR(k) | z-statistic | Significant? |
|-------------|-------|-------------|-------------|
| 2 | 0.9902 | -0.6771 | ✔ No |
| 5 | 0.9990 | -0.0304 | ✔ No |
| 10 | 1.0747 | 1.5228 | ✔ No |
| 20 | 1.2273 | 3.1469 | ✘ **Yes** (rejects RWH) |
| 30 | 1.3210 | 3.5820 | ✘ **Yes** (rejects RWH) |

**Conclusion:**
- At short horizons (k = 2, 5): VR ≈ 1.0 — variance scales linearly ✔
- At longer horizons (k = 20, 30): VR climbs to 1.22–1.32 — **positive autocorrelation**
- VR > 1 = **momentum** (trend-following behavior confirmed)
- The VR profile curve rises monotonically from ~0.99 to ~1.44 at k=50
- **Bitcoin shows short-term randomness but long-term momentum rejection of RWH**

---

## 📋 Summary Scorecard

| Test | Verdict |
|------|---------|
| ADF (prices non-stationary?) | ✔ Supports RWH |
| ADF (returns stationary?) | ✔ Supports RWH |
| Autocorrelation (no sig lags?) | ✘ Against RWH |
| Runs Test (random sequence?) | ✘ Against RWH |
| Variance Ratio (VR ≈ 1?) | ✘ Against RWH |
| **Overall Score** | **2/5 — REJECTED** |
# ⚙️ Implementation Guide — Random Walk Hypothesis (BTC-USD)

**Script:** `rw_btc_test.py`  
**Data:** `btc-usd-max.csv` (CoinGecko, 2013–2026)  
**Libraries:** pandas, numpy, matplotlib, statsmodels  

---

## 📥 Step 1 — Data Loading

```python
import pandas as pd
import numpy as np

df = pd.read_csv("btc-usd-max.csv")
df["snapped_at"] = pd.to_datetime(df["snapped_at"])
df = df.sort_values("snapped_at").set_index("snapped_at")
df = df.dropna(subset=["price"])
prices = df["price"]
```

> **Result:** 4,736 rows loaded, spanning 2013-04-28 → 2026-04-17.

---

## 🔄 Step 2 — Log Return Calculation

```python
log_ret = np.log(prices / prices.shift(1)).dropna()
# 4,735 return observations
```

> **Why log returns?** They are time-additive and normalize exponential price growth, making them suitable for stationarity and autocorrelation tests.

---

## 📊 Step 3 — Descriptive Statistics

```python
mean  = log_ret.mean()     # 0.001335
std   = log_ret.std()      # 0.037455
skew  = log_ret.skew()     # -0.4934   (left-skewed)
kurt  = log_ret.kurtosis() # 9.3778    (extreme fat tails)
```

> Kurtosis of **9.38** confirms Bitcoin returns are **leptokurtic** — extreme events occur far more often than a Normal distribution predicts.

---

## 🧪 Step 4a — Augmented Dickey-Fuller (ADF) Test

```python
from statsmodels.tsa.stattools import adfuller

adf_price   = adfuller(prices)   # p = 0.7450  → Non-stationary ✔
adf_returns = adfuller(log_ret)  # p = 0.0000  → Stationary ✔

print(f"Price p-value  : {adf_price[1]:.4f}")
print(f"Returns p-value: {adf_returns[1]:.6f}")
```

**Null hypothesis H₀:** The series has a unit root (non-stationary).

| Series | p-value | Decision |
|--------|---------|----------|
| Prices | 0.7450 | Fail to reject H₀ → Non-stationary (unit root exists) ✔ |
| Returns | 0.0000 | Reject H₀ → Stationary ✔ |

---

## 📉 Step 4b — Autocorrelation / Ljung-Box Q Test

```python
from statsmodels.tsa.stattools import acf

acf_vals, confint, qstat, pvals = acf(
    log_ret, nlags=30, qstat=True, fft=True, alpha=0.05
)
sig_lags = [i+1 for i, p in enumerate(pvals) if p < 0.05]
# → [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 ...]
```

**Null hypothesis H₀:** No autocorrelation up to lag k (white noise).

| Finding | Value |
|---------|-------|
| First significant lag | Lag 6 |
| Ljung-Box Q (lag 6) | 16.03 |
| p-value | 0.0136 |
| Verdict | Autocorrelation detected ✘ |

---

## 🎲 Step 4c — Wald-Wolfowitz Runs Test

```python
from statsmodels.sandbox.stats.runs import runstest_1samp

median_ret   = float(log_ret.median())
runs_z, runs_p = runstest_1samp(log_ret.values, cutoff=median_ret)
# Z = 3.0376,  p = 0.0024
```

**Null hypothesis H₀:** The sequence of +/− returns is purely random.

| Metric | Value |
|--------|-------|
| Z-statistic | 3.0376 |
| p-value | 0.0024 |
| Verdict | Non-random (momentum clustering) ✘ |

> Positive Z → fewer runs than expected → same-sign returns cluster → **momentum effect**.

---

## ⚖️ Step 4d — Variance Ratio Test (Lo-MacKinlay 1988)

```python
def variance_ratio_test(rets, k):
    arr  = rets.values
    n    = len(arr)
    var1 = np.var(arr, ddof=1)
    ret_k = pd.Series(arr).rolling(window=k).sum().dropna().values
    vark  = np.var(ret_k, ddof=1)
    vr    = vark / (k * var1)
    se    = np.sqrt(2*(2*k - 1)*(k - 1) / (3*k*n))   # IID SE
    z     = (vr - 1) / se
    return vr, z

for k in [2, 5, 10, 20, 30]:
    vr, z = variance_ratio_test(log_ret, k)
    print(f"k={k:2d}: VR={vr:.4f}  z={z:.4f}")
```

**Null hypothesis H₀:** VR(k) = 1 for all horizons (random walk holds).

| k | VR(k) | z-stat | Reject H₀? |
|---|-------|--------|------------|
| 2 | 0.9902 | -0.677 | No ✔ |
| 5 | 0.9990 | -0.030 | No ✔ |
| 10 | 1.0747 | 1.523 | No ✔ |
| 20 | 1.2273 | 3.147 | **Yes ✘** |
| 30 | 1.3210 | 3.582 | **Yes ✘** |

> VR profile rises from ~1.0 at k=2 to ~1.44 at k=50 — a clear upward trend indicating **long-horizon positive autocorrelation (momentum)**.

---

## 📈 Outputs Generated

| File | Description |
|------|-------------|
| `btc_eda_plots.png` | 5-panel EDA: price, returns, histogram, ACF, rolling volatility |
| `btc_variance_ratio.png` | VR(k) profile curve from k=2 to k=50 |
| `results.md` | Full test results and interpretation |
| `analysis.md` | Final verdict and insights |
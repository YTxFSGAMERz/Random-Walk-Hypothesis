# 🧠 Final Analysis — Bitcoin (BTC-USD)

**Asset:** Bitcoin (BTC-USD)  
**Data:** 4,735 daily observations (2013-04-28 → 2026-04-17)  

---

## 📌 Key Findings

| Observation | Detail |
|-------------|--------|
| Price series | Non-stationary with unit root (ADF p = 0.745) |
| Return series | Strongly stationary (ADF p ≈ 0.000) |
| Return distribution | Heavy-tailed — kurtosis = 9.38, skew = -0.49 |
| Autocorrelation | Significant at lags 6–30 (Ljung-Box Q rejects white noise) |
| Runs structure | Non-random — momentum clusters detected (Z = 3.04, p = 0.0024) |
| Short-term VR | Near 1.0 at k=2,5 (appears random in short horizons) |
| Long-term VR | Rises to 1.32 at k=30 (positive autocorrelation at longer horizons) |
| Volatility | Extreme clustering visible — early 2014, 2018, 2020 |

---

## 📊 Test-by-Test Summary

| Test | H₀ | Result | RWH Verdict |
|------|----|--------|-------------|
| ADF (prices) | Unit root exists | p=0.745 → Fail to reject | ✔ Supports |
| ADF (returns) | Unit root exists | p=0.000 → Reject | ✔ Supports |
| Ljung-Box ACF | No autocorrelation | Sig. lags 6–30 | ✘ Against |
| Runs Test | Random sequence | p=0.0024 → Reject | ✘ Against |
| Variance Ratio | VR(k)=1 all k | Rejects at k=20,30 | ✘ Against |

---

## ⚖️ Verdict

The Random Walk Hypothesis is:

- [ ] Fully Supported  
- [ ] Partially Supported  
- [x] **Rejected (2/5 tests support RWH)**  

---

## 📉 Interpretation

**What the tests tell us together:**

- The **ADF test** confirms that price levels behave like a random walk — they do not revert to any fixed mean. This is the *structural* behavior expected of a randomly wandering price.
- However, the **Runs Test** (Z = 3.04, p = 0.0024) reveals that consecutive return signs are **not independent** — Bitcoin tends to rally in streaks and crash in streaks. This is the statistical fingerprint of **momentum**.
- The **Ljung-Box ACF test** confirms that beyond 5 trading days, returns are serially correlated. There are predictable structures at weekly and multi-week horizons.
- The **Variance Ratio** profile makes this conclusive: as holding period k grows from 2 to 50 days, the VR climbs from ~0.99 to ~1.44. A pure random walk would stay flat at 1.0. This monotonically rising profile is the hallmark of **long-horizon positive autocorrelation**.

---

## 💡 Market Efficiency Implication

Bitcoin's market is **not weak-form efficient** over the 2013–2026 window. The statistical evidence suggests:

1. **Short-term (1–5 days):** Behaves close to a random walk — noise dominates.
2. **Medium-term (6–20 days):** Autocorrelation becomes detectable — potential momentum signals.
3. **Long-term (20+ days):** Strong momentum structure — markets trend and exhibit boom/bust cycles.

This is consistent with known BTC market behavior (bull/bear cycles). Retail-driven speculation, lower institutional arbitrage capacity (especially pre-2020), and sentiment-driven trading all contribute to these inefficiencies.

---

## 📉 Insight

- Crypto markets show **much higher inefficiency** than mature equity markets (e.g., AAPL showed 3/5 vs BTC's 2/5)
- Volatility clustering is severe — standard RWH assumes constant variance, which is massively violated
- The fat-tailed return distribution (kurtosis = 9.38) makes standard tests conservative — true departures from randomness are likely even stronger than reported
- **Future work:** Apply GARCH(1,1) to model conditional heteroskedasticity, then re-run VR and ACF tests on GARCH-filtered residuals for a noise-adjusted verdict

---

## 🔬 Comparison: BTC vs AAPL

| Metric | BTC-USD | AAPL |
|--------|---------|------|
| Tests supporting RWH | 2/5 | 3/5 |
| Kurtosis | 9.38 | 5.69 |
| Runs Test p-value | 0.0024 ✘ | 0.3537 ✔ |
| VR at k=20 | 1.23 ✘ | 0.85 ✘ |
| RWH Verdict | **Rejected** | **Partially Supported** |

Conclusion: Bitcoin is **significantly less efficient** than Apple stock, with stronger momentum, higher tail risk, and more pronounced serial dependence across all tested horizons.
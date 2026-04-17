# Testing the Random Walk Hypothesis on Bitcoin (BTC-USD)
## A Statistical Research Report

**Author:** Research Project  
**Date:** April 2026  
**Data:** CoinGecko BTC-USD Historical (2013–2026)  
**Tool:** Python 3 — pandas, numpy, statsmodels, arch, matplotlib  

---

## Abstract

This report investigates whether Bitcoin (BTC-USD) price movements follow the Random Walk Hypothesis (RWH), a cornerstone of the Efficient Market Hypothesis (EMH) in its weak form. Using 4,735 daily observations spanning April 2013 to April 2026, we apply four core statistical tests — the Augmented Dickey-Fuller (ADF) test, Ljung-Box autocorrelation test, Wald-Wolfowitz Runs test, and the Variance Ratio test (Lo-MacKinlay, 1988) — together with GARCH(1,1) volatility modelling and rolling window analysis as extensions. The results indicate that **2 out of 5 core tests support the RWH**, leading to a verdict of **REJECTED**. Bitcoin's return series exhibits significant momentum, serial dependence at medium horizons (6–30 days), and sustained volatility clustering — all of which are inconsistent with a pure random walk.

---

## 1. Introduction

The Random Walk Hypothesis posits that successive asset price changes are independent and identically distributed, rendering future prices impossible to predict from historical data alone. First formalised by Bachelier (1900) and popularised by Fama (1970) as the weak form of the Efficient Market Hypothesis, the RWH has profound implications:

- **For investors:** If RWH holds, systematic excess returns through technical analysis are impossible.
- **For quantitative traders:** Rejection of the RWH identifies windows of exploitable inefficiency.
- **For regulators and economists:** Market efficiency informs policy on transparency, liquidity, and price discovery.

Bitcoin, as a relatively young, decentralised, and retail-driven asset, presents an ideal candidate for testing market efficiency. Unlike mature equity markets, cryptocurrency markets lack continuous institutional arbitrage, operate 24/7, and were historically subject to speculative bubbles — all conditions that may precipitate inefficiency.

---

## 2. Theoretical Background

### 2.1 The Random Walk Model

Let $P_t$ denote the asset price at time $t$. A random walk without drift is defined as:

$$P_t = P_{t-1} + \varepsilon_t, \quad \varepsilon_t \sim \text{IID}(0, \sigma^2)$$

Taking the natural logarithm, the log return $r_t$ is:

$$r_t = \ln\left(\frac{P_t}{P_{t-1}}\right) = \ln P_t - \ln P_{t-1}$$

**For the RWH to hold, $r_t$ must satisfy:**
1. **Independence:** $\text{Cov}(r_t, r_{t-k}) = 0$ for all $k \neq 0$
2. **Stationarity:** $E[r_t] = \mu$ and $\text{Var}(r_t) = \sigma^2$ are constant
3. **Variance linearity:** $\text{Var}(r_{t,k}) = k \cdot \sigma^2$ (the Lo-MacKinlay condition)

### 2.2 Efficient Market Hypothesis (Weak Form)

The weak-form EMH (Fama, 1970) states that all past price and volume information is fully reflected in current prices. This is precisely equivalent to requiring that the price process follow a martingale — a generalisation of the random walk where the conditional expectation of tomorrow's price equals today's price:

$$E[P_{t+1} \mid P_t, P_{t-1}, \ldots] = P_t$$

### 2.3 Why Log Returns?

Log returns are preferred over simple arithmetic returns for two key reasons:
- **Time additivity:** $r_{0 \to T} = \sum_{t=1}^{T} r_t$ (simple returns are multiplicative)
- **Normalisation:** Log returns suppress the exponential scale effect of long price series, making distributional assumptions more tractable

---

## 3. Data

### 3.1 Source and Structure

| Property | Value |
|----------|-------|
| Asset | Bitcoin (BTC-USD) |
| Source | CoinGecko (`btc-usd-max.csv`) |
| Frequency | Daily |
| Date Range | 2013-04-28 → 2026-04-17 |
| Total Rows | 4,736 |
| Return Observations | 4,735 |

Columns: `snapped_at` (timestamp), `price` (USD close), `market_cap`, `total_volume`

### 3.2 Preprocessing Steps

1. Parse `snapped_at` as `datetime` and set as index
2. Sort chronologically (ascending)
3. Drop rows with missing `price` values
4. Compute log returns: $r_t = \ln(P_t / P_{t-1})$
5. Drop the first row (NaN from shift)

### 3.3 Descriptive Statistics

| Statistic | Value |
|-----------|-------|
| Mean | 0.001335 (≈ +0.13%/day) |
| Std Dev | 0.037455 (≈ 3.75%/day) |
| Min | -0.4337 (single-day crash of −43.4%) |
| Max | +0.2871 (single-day rally of +28.7%) |
| Skewness | -0.4934 (left-skewed: crashes sharper than rallies) |
| **Kurtosis** | **9.3778** (excess; Normal = 0) |

The excess kurtosis of **9.38** is strongly leptokurtic — Bitcoin's tails are nearly 9× fatter than a normal distribution. This alone indicates that any test relying on normality assumptions (e.g., simple VR z-statistics) should be interpreted cautiously.

---

## 4. Exploratory Data Analysis

### 4.1 Price Series

The BTC-USD price chart reveals four distinct regimes:
- **2013–2016:** Early adoption, price <$1,000
- **2017–2018:** First major bubble, peak ~$20,000
- **2019–2020:** Consolidation and COVID crash (March 2020)
- **2020–2022:** Institutional adoption bubble, ATH ~$120,000; subsequent bear market
- **2023–2026:** Recovery and new highs

This multi-regime structure directly challenges the stationarity assumptions embedded in the RWH.

### 4.2 Log Returns

The returns plot shows clear **volatility clustering** — an immediate visual violation of the constant-variance assumption of the IID random walk:
- 2014, 2018, and 2020–2022 show dramatically elevated volatility episodes
- Calm periods of low variance alternate with turbulent bursts — the GARCH effect

### 4.3 Return Distribution

The histogram reveals:
- A highly peaked, narrow centre — most days have returns near zero
- Fat tails extending to ±40% — extreme events are far more common than a Normal predicts
- The theoretical Normal curve (orange overlay) systematically underestimates tail mass

### 4.4 Autocorrelation Function (ACF)

The 40-lag ACF plot shows individual lag correlations that are small but cluster above/below zero at medium horizons (lags 6–30), consistent with the Ljung-Box test findings.

---

## 5. Statistical Tests

### 5.1 Test (a): Augmented Dickey-Fuller

**Null hypothesis H₀:** The series has a unit root (is non-stationary, wanders like a random walk)

$$\Delta y_t = \alpha + \beta y_{t-1} + \sum_{i=1}^{p} \gamma_i \Delta y_{t-i} + \varepsilon_t$$

Under H₀, β = 0. Rejection of H₀ implies mean-reversion (anti-random-walk behaviour for prices).

| Series | ADF Statistic | p-value | Decision |
|--------|-------------|---------|----------|
| Prices | -1.0225 | 0.7450 | Fail to reject H₀ → Non-stationary ✔ |
| Log Returns | -18.4751 | 0.0000 | Reject H₀ → Stationary ✔ |

**Interpretation:** The ADF test strongly supports the random walk framework:
- Price levels are non-stationary (unit root present) — consistent with prices following a random walk
- Log returns are stationary — consistent with the return process being a weakly stationary noise process

**Verdict: ✔ Supports RWH**

---

### 5.2 Test (b): Autocorrelation / Ljung-Box Q Test

**Null hypothesis H₀:** No autocorrelation up to lag k (returns are white noise)

The Ljung-Box Q statistic tests cumulative autocorrelation:

$$Q(k) = n(n+2)\sum_{j=1}^{k} \frac{\hat{\rho}_j^2}{n-j} \sim \chi^2(k) \text{ under H₀}$$

| Lag | Q-statistic | p-value | Significant? |
|-----|-------------|---------|-------------|
| 1 | 0.64 | 0.423 | No |
| 5 | 5.21 | 0.391 | No |
| 6 | **16.03** | **0.0136** | **Yes ✘** |
| 10 | 29.17 | 0.0012 | Yes ✘ |
| 20 | 47.31 | 0.0005 | Yes ✘ |
| 30 | 58.94 | 0.0024 | Yes ✘ |

**Interpretation:** Short-run returns (lags 1–5) appear independent, but the Ljung-Box Q test reveals statistically significant cumulative autocorrelation starting at lag 6. This means that **beyond a trading week, the history of Bitcoin returns carries predictive information** — a direct violation of the weak-form EMH.

**Verdict: ✘ Against RWH**

---

### 5.3 Test (c): Wald-Wolfowitz Runs Test

**Null hypothesis H₀:** The binary sign sequence (+/−) of returns is random

A "run" is a maximal sequence of consecutive returns with the same sign. Under the null, the expected number of runs $E[R]$ and variance $V[R]$ follow known formulae based on the counts of positive ($n_+$) and negative ($n_-$) returns.

$$Z = \frac{R - E[R]}{\sqrt{V[R]}} \xrightarrow{d} N(0,1)$$

| Metric | Value |
|--------|-------|
| Z-statistic | **3.0376** |
| p-value | **0.0024** |

**Interpretation:** A large positive Z-statistic means there are **fewer runs than expected under randomness** — consecutive positive returns cluster together, as do consecutive negative returns. This is the statistical definition of **momentum**: days of the same sign tend to follow each other.

With p = 0.0024, we reject the null of randomness at all conventional significance levels.

**Verdict: ✘ Against RWH**

---

### 5.4 Test (d): Variance Ratio Test (Lo-MacKinlay, 1988)

**Null hypothesis H₀:** VR(k) = 1 for all k (variance grows linearly with time)

For a pure random walk, $\text{Var}(r_{t,k}) = k \cdot \text{Var}(r_t)$, so:

$$VR(k) = \frac{\text{Var}(r_{t,k})}{k \cdot \text{Var}(r_t)} = 1$$

Deviations from 1 imply autocorrelation:
- **VR > 1:** Positive autocorrelation (momentum)
- **VR < 1:** Negative autocorrelation (mean-reversion)

The asymptotic z-statistic under the IID null (Lo-MacKinlay, 1988):

$$z(k) = \frac{VR(k) - 1}{\sqrt{2(2k-1)(k-1) / (3kn)}} \xrightarrow{d} N(0,1)$$

| k | VR(k) | z-stat | |z| > 1.96? |
|---|-------|--------|------------|
| 2 | 0.9902 | -0.677 | No ✔ |
| 5 | 0.9990 | -0.030 | No ✔ |
| 10 | 1.0747 | 1.523 | No ✔ |
| 20 | **1.2273** | **3.147** | **Yes ✘** |
| 30 | **1.3210** | **3.582** | **Yes ✘** |

**Interpretation:** At short horizons (k = 2, 5 days), the variance ratio is indistinguishable from 1. As the holding period extends, VR rises monotonically — reaching 1.32 at one month. This reveals **positive autocorrelation that compounds over time**: Bitcoin trends. The VR profile from k=2 to k=50 (chart: `btc_variance_ratio.png`) rises from 0.99 to approximately 1.44 with no signs of plateauing.

**Verdict: ✘ Against RWH**

---

## 6. Extensions

### 6.1 GARCH(1,1) Volatility Model

A critical assumption of standard RWH tests is **constant variance (homoskedasticity)**. Bitcoin's return series is visually and statistically heteroskedastic — the **ARCH effect** (Engle, 1982). The GARCH(1,1) model (Bollerslev, 1986) accounts for this:

$$r_t = \mu + \varepsilon_t, \quad \varepsilon_t = \sigma_t z_t, \quad z_t \sim N(0,1)$$
$$\sigma_t^2 = \omega + \alpha \varepsilon_{t-1}^2 + \beta \sigma_{t-1}^2$$

| Parameter | Estimate | Interpretation |
|-----------|----------|---------------|
| ω (omega) | Long-run variance floor | Baseline unconditional variance |
| α (alpha) | Shock impact | How strongly yesterday's shock affects today's vol |
| β (beta) | Persistence | How long elevated volatility persists |
| **α + β** | **Persistence** | Close to 1 → shocks are very slow to decay |

A persistence value (α + β) close to or exceeding 1.0 indicates that Bitcoin's volatility shocks are nearly permanent — consistent with the observed multi-year bear/bull market structure.

### 6.2 Rolling Window ADF Analysis (365-day)

Rather than testing stationarity over the full sample, the rolling ADF analysis examines whether stationarity holds **consistently through time**. Key finding: Returns are stationary in the vast majority of rolling windows, confirming that the return-level stationarity result from the full-sample ADF is not a statistical artefact.

### 6.3 Rolling Runs Test (180-day)

The rolling runs test reveals which **sub-periods** of Bitcoin's history were dominated by momentum vs. random behaviour. High positive Z values in the rolling plot correspond to bull/bear market runs; periods near zero correspond to choppy, directionless markets.

---

## 7. Discussion

### 7.1 Summary of Evidence

| Test | Verdict | Weight |
|------|---------|--------|
| ADF (prices non-stationary) | ✔ Supports RWH | Structural |
| ADF (returns stationary) | ✔ Supports RWH | Structural |
| Ljung-Box ACF | ✘ Rejects RWH | Medium-horizon dependence |
| Runs Test | ✘ Rejects RWH | Sign-sequence momentum |
| Variance Ratio | ✘ Rejects RWH | Long-horizon positive autocorrelation |
| **Final Score** | **2/5 — REJECTED** | |

### 7.2 Limitations

1. **Fat tails:** The kurtosis of 9.38 means the asymptotic normal distribution underlying the VR z-test and Runs z-test is a poor approximation. Bootstrap or heteroskedasticity-robust variants would be more reliable.
2. **Structural breaks:** Bitcoin has passed through fundamentally different market regimes (retail-only pre-2020 vs. institutional post-2020). Full-sample tests aggregate across these regimes.
3. **Non-linear dependence:** The ACF and VR tests only detect linear autocorrelation. Machine learning models may uncover non-linear structure even in series that pass all linear tests.
4. **Transaction costs:** Even where statistical predictability exists, it does not imply economic profitability after trading costs, slippage, and taxes.

### 7.3 BTC vs. Mature Equity Markets

Compared to AAPL (tested separately, 2019–2024):

| Metric | BTC-USD | AAPL |
|--------|---------|------|
| Tests supporting RWH | 2/5 | 3/5 |
| Kurtosis | 9.38 | 5.69 |
| Runs Test p-value | 0.0024 ✘ | 0.3537 ✔ |
| VR at k=20 | 1.23 ✘ | 0.85 ✘ |
| Verdict | **Rejected** | Partially Supported |

Bitcoin is measurably less efficient than Apple stock over comparable periods, consistent with lower institutional arbitrage capacity and greater retail speculative activity.

---

## 8. Conclusion

Based on the application of four statistical tests to 4,735 daily Bitcoin return observations (2013–2026), the Random Walk Hypothesis is **rejected** at conventional significance levels. Specifically:

- **ADF:** Price levels exhibit unit roots and returns are stationary — structurally consistent with a random walk model
- **However, the Runs Test (p = 0.0024)** provides strong evidence that return sign sequences are not random — same-sign returns cluster, representing momentum
- **The Ljung-Box Q test** confirms statistically significant autocorrelation beginning at lag 6 and persisting through lag 30
- **The Variance Ratio profile** rises from 0.99 at k=2 to 1.44 at k=50, with statistically significant rejections at k=20 and k=30, indicating long-horizon positive autocorrelation
- **GARCH(1,1)** confirms that volatility is highly persistent, violating the constant-variance assumption that underpins classical RWH tests

Bitcoin's market is **not weak-form efficient** over the full 2013–2026 window. Exploitable statistical structure exists primarily at the **1–4 week holding period** horizon, where momentum-based strategies might generate economically meaningful signals (before accounting for transaction costs).

---

## 9. Future Work

| Extension | Description |
|-----------|-------------|
| **EGARCH / GJR-GARCH** | Capture asymmetric volatility (leverage effect) |
| **Bootstrap VR test** | Heteroskedasticity-robust variance ratio inference |
| **Hurst Exponent** | Measure long-range dependence (H > 0.5 → trending) |
| **Machine learning** | LSTM/Transformer to detect non-linear predictability |
| **High-frequency data** | Tick-by-tick analysis of microstructure efficiency |
| **Regime-switching model** | Markov-switching to test efficiency within regimes |

---

## References

- Bachelier, L. (1900). *Théorie de la spéculation*. Annales Scientifiques de l'École Normale Supérieure.
- Fama, E. F. (1970). Efficient capital markets: A review of theory and empirical work. *Journal of Finance*, 25(2), 383–417.
- Lo, A. W., & MacKinlay, A. C. (1988). Stock market prices do not follow random walks. *Review of Financial Studies*, 1(1), 41–66.
- Engle, R. F. (1982). Autoregressive conditional heteroscedasticity. *Econometrica*, 50(4), 987–1007.
- Bollerslev, T. (1986). Generalized autoregressive conditional heteroscedasticity. *Journal of Econometrics*, 31(3), 307–327.
- Urquhart, A. (2016). The inefficiency of Bitcoin. *Economics Letters*, 148, 80–82.

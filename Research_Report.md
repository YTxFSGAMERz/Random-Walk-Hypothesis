# Testing the Random Walk Hypothesis on Financial Time Series Using Statistical Methods

## 1. Objective

The primary objective of this research is to rigorously test whether the price movements of a selected financial asset follow a **Random Walk**. Specifically, the analysis seeks to determine if asset returns are **Independent and Identically Distributed (I.I.D.)** and whether the Efficient Market Hypothesis (EMH) holds under its weak form.

In financial economics, the Random Walk Hypothesis (RWH) asserts that asset prices evolve according to a random process, implying that future price steps cannot be predicted based on historical price sequences. Verifying this hypothesis is fundamental for market participants, as it defines whether technical analysis and historical chart patterns have any statistical validity or if the market is efficiently pricing all past information.

---

## 2. Theoretical Foundation

### The Random Walk Hypothesis
The Random Walk Hypothesis suggests that changes in stock prices have the same distribution and are independent of each other. Therefore, the past movement or trend of a stock price or market cannot be used to predict its future movement. In short, prices take a "random walk."

### Relationship with the Efficient Market Hypothesis (EMH)
The RWH is heavily intertwined with the **Weak Form of the Efficient Market Hypothesis**. The weak-form EMH states that all past market prices and data are fully reflected in current securities prices. Therefore, if the weak-form EMH holds, the price path should perfectly mimic a random walk, preventing anyone from generating consistent excess returns using historical price data.

### Mathematical Formulation
Let $P_t$ be the price of an asset at time $t$. The actual price process can be modeled as:
$$ P_t = P_{t-1} + \epsilon_t $$
where $\epsilon_t$ is a random error term (white noise) with a mean of zero.

Because financial prices compounded over time, it is standard practice to use the natural logarithm of prices, $p_t = \ln(P_t)$. The **log return** $r_t$ over a single period is defined as:
$$ r_t = p_t - p_{t-1} = \ln\left(\frac{P_t}{P_{t-1}}\right) $$

For a pure random walk, we assume:
1. **Independence:** $\text{Cov}(r_t, r_{t-k}) = 0$ for all $k \neq 0$.
2. **Stationarity:** The mean and variance of $r_t$ are constant over time.
3. **Normality (Optional but common):** $r_t \sim N(\mu, \sigma^2)$.

---

## 3. Data Collection

To conduct a robust statistical test, you must gather reliable historical market data.

* **Asset Selection:** Choose a highly liquid financial asset. For this guide, we will consider **Bitcoin (BTC-USD)** or a high-volume stock like **Apple (AAPL)**.
* **Frequency & Length:** Daily closing prices are preferred. Intraday data is exceptionally noisy, while monthly data provides too few observations. Collect at least **3 to 5 years** of data to capture various market cycles.
* **Required Columns:** At a minimum, your dataset should contain:
  * `Date`: The timestamp of the observation.
  * `Close` (or `Adj Close`): The closing price representing the asset's value for the period.
  * `Volume` (Optional - useful for broader liquidity analysis).

---

## 4. Data Preprocessing

Raw financial data is rarely ready for immediate statistical testing. Apply the following steps:

1. **Format Datetime:** Ensure the `Date` column is cast to a strict datetime object and set as the index.
2. **Chronological Sorting:** Sort the data chronologically (oldest to newest) to prevent look-ahead bias in statistical calculations.
3. **Handle Missing Values:** Drop or linearly interpolate `NaN` values, which commonly occur during market holidays or API request failures.
4. **Calculate Log Returns:** Transform raw prices into continuously compounded returns (log returns).

> [!NOTE]
> **Why Log Returns?** 
> Log returns are preferred over raw percentage returns because they are time-additive ($r_{0 \rightarrow 2} = r_{0 \rightarrow 1} + r_{1 \rightarrow 2}$), making them mathematically convenient for statistical modeling. They also normalize the exponential compounding effect of asset prices.

---

## 5. Exploratory Data Analysis (EDA)

Before applying rigid statistical tests, visualize the data to build intuition.

* **Price Series Plot:** Observe $P_t$ over time. Does it look like a smooth trend, or erratic noise?
* **Log Returns Plot:** Plot $r_t$ over time. A stationary process should oscillate evenly around zero. 
* **Histogram:** Check if the returns mirror a normal distribution (bell curve). Financial assets often exhibit "fat tails" (leptokurtosis), meaning extreme events happen more frequently than a normal distribution predicts.
* **Volatility Clustering:** Observe the returns plot. You will notice periods of high turbulence followed by periods of calm, an effect known as volatility clustering (a direct violation of strict RWH assumptions).

---

## 6. Statistical Tests

We apply four distinct tests to systematically evaluate the RWH.

### (a) Autocorrelation Test (ACF)
The Autocorrelation Function (ACF) measures the linear correlation between the series and its own lagged values. If the market perfectly follows a random walk, the ACF at all lags $k \geq 1$ should be statistically indistinguishable from zero.

### (b) Augmented Dickey-Fuller (ADF) Test
The ADF test checks for the presence of a unit root (non-stationarity) in the series. 
* **Null Hypothesis ($H_0$):** The series possesses a unit root (is non-stationary).
* **Price Application:** Asset prices *should* fail to reject $H_0$, proving they wander randomly.
* **Return Application:** Asset returns *should* reject $H_0$, proving that returns are stationary.

### (c) Runs Test
A non-parametric test to check the randomness of the sequence by analyzing "runs" (consecutive sequences of identical signs, e.g., `+ + - - - +`).
* Assesses whether the number of observed runs differs significantly from the statistically expected number of runs. Too few runs hint at momentum; too many hint at mean-reversion.

### (d) Variance Ratio Test
The core tenet of a random walk is that the variance of returns scales linearly with time. For example, the variance of 2-day returns should safely be exactly twice the variance of 1-day returns.
* **Logic:** If $VR(k) = \frac{\text{Var}(r_{k\text{-day}})}{k \times \text{Var}(r_{\text{1-day}})} \approx 1$, the hypothesis holds. Deviations from 1 indicate autocorrelation.

---

## 7. Implementation (Python)

The following modular code executes all required operations using established data science libraries.

```python
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller
from statsmodels.sandbox.stats.runs import runstest_1samp
import scipy.stats as stats
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. & 2. Data Collection & Preprocessing
# ==========================================
symbol = "BTC-USD"
print(f"Fetching data for {symbol}...")
data = yf.download(symbol, start="2019-01-01", end="2024-01-01", progress=False)

# Clean and sort
data = data.sort_index()
data = data.dropna()

# Calculate log returns
data['Log_Return'] = np.log(data['Close'] / data['Close'].shift(1))
data = data.dropna()

print("\n--- Summary Statistics ---")
print(data['Log_Return'].describe())
print(f"Skewness: {data['Log_Return'].skew():.4f}")
print(f"Kurtosis: {data['Log_Return'].kurtosis():.4f}")

# ==========================================
# 3. Exploratory Data Analysis (EDA)
# ==========================================
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
ax[0].plot(data.index, data['Close'], color='blue')
ax[0].set_title(f"{symbol} Closing Price")
ax[0].grid(True, alpha=0.3)

ax[1].plot(data.index, data['Log_Return'], color='red', alpha=0.7)
ax[1].set_title(f"{symbol} Log Returns")
ax[1].grid(True, alpha=0.3)
plt.show()

# ==========================================
# 4. Statistical Tests
# ==========================================

# (a) Autocorrelation
plt.figure(figsize=(10, 4))
plot_acf(data['Log_Return'], lags=30, zero=False)
plt.title(f"Autocorrelation Function (ACF) of {symbol} Log Returns")
plt.show()

# (b) Augmented Dickey-Fuller Test
print("\n--- Augmented Dickey-Fuller (ADF) Test ---")
adf_price = adfuller(data['Close'])
print(f"Prices ADF Statistic: {adf_price[0]:.4f} | p-value: {adf_price[1]:.4f}")

adf_returns = adfuller(data['Log_Return'])
print(f"Returns ADF Statistic: {adf_returns[0]:.4f} | p-value: {adf_returns[1]:.4f}")

# (c) Runs Test
print("\n--- Runs Test ---")
# Using median to distinguish runs (+ / - relative to median)
median_ret = data['Log_Return'].median()
runs_stat, runs_p = runstest_1samp(data['Log_Return'], cutoff=median_ret)
print(f"Runs Test Z-Statistic: {runs_stat:.4f} | p-value: {runs_p:.4f}")

# (d) Lo-MacKinlay Variance Ratio Test (Simplified 2 and 5 periods)
print("\n--- Variance Ratio Test ---")
def variance_ratio(returns, k):
    var_1 = np.var(returns, ddof=1)
    ret_k = returns.rolling(window=k).sum().dropna()
    var_k = np.var(ret_k, ddof=1)
    vr = var_k / (k * var_1)
    return vr

vr_2 = variance_ratio(data['Log_Return'], 2)
vr_5 = variance_ratio(data['Log_Return'], 5)
print(f"Variance Ratio (k=2): {vr_2:.4f}")
print(f"Variance Ratio (k=5): {vr_5:.4f}")
```

---

## 8. Results Interpretation

After running the Python pipeline, interpret the results systematically:

### Autocorrelation (ACF)
* **Interpretation:** Look at the lag plot. If most lags fall within the shaded confidence interval (usually 95%), the series lacks significant linear autocorrelation. 
* **Significance:** A purely random walk will have entirely insignificant lags. If an asset exhibits strong correlation at lag 1, it suggests short-term momentum or mean reversion.

### Augmented Dickey-Fuller Test
* **Price Series Result:** Generally, the p-value will be $> 0.05$. We fail to reject the null hypothesis, concluding that the *price* level is non-stationary and exhibits a unit root (supports random walk).
* **Returns Series Result:** Generally, the p-value will be $\approx 0.000$. We reject the null hypothesis, concluding that the *log returns* are stationary (supports random walk model assumption).

### Runs Test
* **Interpretation:** The null hypothesis assumes the sequence is entirely random.
* **Significance:** If the p-value is $> 0.05$, we fail to reject randomness. If p-value $< 0.05$ and Z-Statistic is negative, there are fewer runs than expected (momentum effect). If positive, there are more runs than expected (mean-reversion).

### Variance Ratio Test
* **Interpretation:** If the RWH holds, VR(k) should equal exactly 1.0.
* **Significance:** A VR $> 1$ indicates positive autocorrelation (trend-following / momentum). A VR $< 1$ indicates negative autocorrelation (mean-reversion). Small deviations are normal, but extreme variance ratios reject the Random Walk Hypothesis.

---

## 9. Discussion

While a financial time series might display "random walk-like" linearity, several core considerations expose the limits of these strict statistical tests:

* **Limitations of Tests:** Linear autocorrelation tests (ACF, VR) cannot detect non-linear predictability. Machine learning models often find patterns in data that linear statistical tests brand as noise.
* **Volatility Clustering:** Mandelbrot famously highlighted that "large changes tend to be followed by large changes, and small changes by small changes." Constant Variance (Homoskedasticity), a core assumption of standard RWH tests, is heavily violated by the financial markets' localized volatility. To address this, heteroskedasticity-robust tests are often mathematically necessary.
* **Market Anomalies & Asset Differences:** Cryptocurrencies like Bitcoin often fail the Runs test due to excessive momentum cycles (bubbles), whereas highly mature large-cap stocks like AAPL often align much closer with strict randomness, guided by relentless institutional arbitrage. 

---

## 10. Conclusion

By systematically deploying the ACF, ADF, Runs, and Variance Ratio tests, a researcher can objectively gauge market efficiency. An asset that perfectly matches the expectation of these tests confirms the Weak-Form Efficient Market Hypothesis: predicting its future based strictly on its past is mathematically equivalent to flipping a coin. 

If test parameters (such as the Runs Test p-value or a VR ratio consistently $>1$) show non-randomness, it implies temporary pricing inefficiencies that quantitative algorithmic trading models and hedge funds specifically aim to exploit.

---

## 11. Extensions

For future research, testing the RWH can be expanded through several progressive paradigms:

1. **GARCH Modeling:** Rather than standard variance models, implementing Generalized Autoregressive Conditional Heteroskedasticity (GARCH) allows modeling of the volatility clustering itself.
2. **High-Frequency Data:** Analyzing tick-by-tick micro-structure data. At millisecond intervals, the Random Walk Hypothesis often fails due to order-book imbalances and market-maker spread geometries.
3. **Machine Learning Predictability:** Deploying Recurrent Neural Networks (LSTMs) or Transformer models to test if deep non-linear feature abstractions can predict future returns where linear tests fail.

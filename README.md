# 📊 Random Walk Hypothesis Research Project

## 🎯 Objective
This project tests the **Random Walk Hypothesis** on Bitcoin (BTC-USD) using rigorous statistical methods across 13 years of daily price data.

We investigate whether price movements are:
- Random
- Independent
- Unpredictable

**Verdict: RWH REJECTED — Bitcoin is NOT weak-form efficient.**

---

## 🧠 Theory

The Random Walk Hypothesis states that:
> Future price movements cannot be predicted using past price data.

It is closely related to the **Efficient Market Hypothesis (Weak Form)**.

Mathematical formulation:
```
P_t = P_(t-1) + ε_t      where ε_t ~ IID(0, σ²)
r_t = ln(P_t / P_(t-1))  ← log return
```

---

## ⚙️ Methodology

1. ✅ Data Collection — `btc-usd-max.csv` (CoinGecko, 2013–2026)
2. ✅ Data Preprocessing — datetime parsing, sorting, NaN removal
3. ✅ Log Return Calculation — 4,735 observations
4. ✅ Exploratory Data Analysis — 5-panel chart generated
5. ✅ Statistical Testing:
   - ✅ Augmented Dickey-Fuller (ADF)
   - ✅ Autocorrelation / Ljung-Box Q
   - ✅ Wald-Wolfowitz Runs Test
   - ✅ Variance Ratio (Lo-MacKinlay 1988)
6. ✅ Extensions:
   - ✅ GARCH(1,1) volatility modelling
   - ✅ Rolling window ADF + Runs analysis

---

## 📂 Project Structure

```
Random Walk Hypothesis/
│── btc-usd-max.csv          ← Raw data (CoinGecko)
│── rw_btc_test.py           ← Main statistical test pipeline
│── rw_extensions.py         ← GARCH + rolling window analysis
│── btc_eda_plots.png        ← EDA visualisations
│── btc_variance_ratio.png   ← VR(k) profile chart
│── btc_garch_plots.png      ← GARCH volatility + rolling analysis
│── Research_Report.md       ← Full academic report
│── implementations.md       ← Code documentation
│── results.md               ← Test results and tables
│── analysis.md              ← Final verdict and insights
│── skills.md                ← Skills developed
│── README.md                ← This file
```

---

## 🚀 How to Run

```bash
# Step 1 — Run core statistical tests
python rw_btc_test.py

# Step 2 — Run GARCH + rolling window extensions
python rw_extensions.py
```

**Requirements:**
```bash
pip install pandas numpy matplotlib statsmodels arch yfinance
```

---

## 📊 Key Results

| Test | Result | RWH Verdict |
|------|--------|-------------|
| ADF (prices) | p = 0.7450 | ✔ Supports |
| ADF (returns) | p = 0.0000 | ✔ Supports |
| Ljung-Box ACF | Sig. lags 6–30 | ✘ Rejects |
| Runs Test | Z=3.04, p=0.0024 | ✘ Rejects |
| Variance Ratio | VR=1.32 at k=30 | ✘ Rejects |
| **Final Score** | **2/5** | **❌ REJECTED** |

---

## 📝 Tasks Roadmap

### 🔹 Phase 1: Setup
- [x] Create project structure
- [x] Load dataset (BTC-USD from btc-usd-max.csv)
- [x] Inspect columns and data types

### 🔹 Phase 2: Data Preprocessing
- [x] Convert timestamps to datetime
- [x] Sort data chronologically
- [x] Handle missing values
- [x] Extract closing price

### 🔹 Phase 3: Feature Engineering
- [x] Compute log returns
- [x] Drop NaN values
- [x] Validate return distribution

### 🔹 Phase 4: Exploratory Data Analysis
- [x] Plot price series
- [x] Plot log returns
- [x] Generate histogram
- [x] Compute summary statistics
- [x] Generate rolling volatility chart

### 🔹 Phase 5: Statistical Testing
- [x] Autocorrelation / Ljung-Box Q Test
- [x] Augmented Dickey-Fuller Test
- [x] Wald-Wolfowitz Runs Test
- [x] Variance Ratio Test (Lo-MacKinlay)

### 🔹 Phase 6: Interpretation
- [x] Analyze each test result
- [x] Compare outcomes across tests
- [x] Evaluate randomness vs. momentum

### 🔹 Phase 7: Conclusion
- [x] Accept or reject Random Walk Hypothesis → **REJECTED**
- [x] Document findings in results.md + analysis.md
- [x] Suggest improvements (GARCH, rolling analysis)

### 🔹 Phase 8: Extensions
- [x] GARCH(1,1) volatility modelling
- [x] Rolling window ADF stationarity analysis
- [x] Rolling Runs Test across time

---

## 📈 Expected Outcome

✅ Determined that Bitcoin does **not** follow a random walk  
✅ Identified market inefficiencies (momentum at 6–30 day horizons)  
✅ Built statistical foundation for quantitative trading strategy research  
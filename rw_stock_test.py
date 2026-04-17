"""
=============================================================================
Testing the Random Walk Hypothesis on a Stock Using Statistical Methods
=============================================================================
Asset   : Apple Inc. (AAPL)
Period  : 2019-01-01 to 2024-12-31 (~6 years of daily data)
Tests   : ADF, ACF/Ljung-Box, Runs Test, Variance Ratio
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")                       # non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from statsmodels.tsa.stattools import adfuller, acf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.sandbox.stats.runs import runstest_1samp
import yfinance as yf
import os, warnings
warnings.filterwarnings("ignore")

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ────────────────────────────────────────────────────────
# 1.  DATA COLLECTION
# ────────────────────────────────────────────────────────
SYMBOL = "AAPL"
START  = "2019-01-01"
END    = "2024-12-31"

print(f"{'='*60}")
print(f"  Random Walk Hypothesis Test  —  {SYMBOL}")
print(f"  Period: {START} → {END}")
print(f"{'='*60}\n")

print("[1/6] Downloading data …")
raw = yf.download(SYMBOL, start=START, end=END, progress=False)
raw = raw.sort_index().dropna()

# Flatten MultiIndex columns if present (yfinance ≥ 0.2.x)
if isinstance(raw.columns, pd.MultiIndex):
    raw.columns = raw.columns.get_level_values(0)

prices = raw["Close"].copy()
print(f"      {len(prices)} trading days loaded.\n")

# ────────────────────────────────────────────────────────
# 2.  DATA PREPROCESSING — Log Returns
# ────────────────────────────────────────────────────────
print("[2/6] Computing log returns …")
log_ret = np.log(prices / prices.shift(1)).dropna()
print(f"      {len(log_ret)} return observations.\n")

# ────────────────────────────────────────────────────────
# 3.  EXPLORATORY DATA ANALYSIS
# ────────────────────────────────────────────────────────
print("[3/6] Generating EDA plots …")

fig = plt.figure(figsize=(16, 10), facecolor="#0d1117")
gs  = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.30)

# — Price series
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(prices.index, prices.values, color="#58a6ff", linewidth=0.8)
ax1.set_title(f"{SYMBOL} — Daily Closing Price", color="white", fontsize=12, fontweight="bold")
ax1.set_facecolor("#161b22"); ax1.tick_params(colors="white")
ax1.grid(alpha=0.15, color="white")
for spine in ax1.spines.values(): spine.set_color("#30363d")

# — Log returns
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(log_ret.index, log_ret.values, color="#f97583", linewidth=0.4, alpha=0.85)
ax2.set_title(f"{SYMBOL} — Daily Log Returns", color="white", fontsize=12, fontweight="bold")
ax2.axhline(0, color="#8b949e", linewidth=0.5, linestyle="--")
ax2.set_facecolor("#161b22"); ax2.tick_params(colors="white")
ax2.grid(alpha=0.15, color="white")
for spine in ax2.spines.values(): spine.set_color("#30363d")

# — Histogram
ax3 = fig.add_subplot(gs[1, 0])
ax3.hist(log_ret.values, bins=80, color="#7ee787", edgecolor="#0d1117", alpha=0.85, density=True)
# overlay normal fit
mu, sigma = log_ret.mean(), log_ret.std()
x_range = np.linspace(log_ret.min(), log_ret.max(), 300)
ax3.plot(x_range, (1/(sigma*np.sqrt(2*np.pi)))*np.exp(-0.5*((x_range-mu)/sigma)**2),
         color="#f0883e", linewidth=1.5, label="Normal fit")
ax3.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="white")
ax3.set_title("Return Distribution vs Normal", color="white", fontsize=12, fontweight="bold")
ax3.set_facecolor("#161b22"); ax3.tick_params(colors="white")
ax3.grid(alpha=0.15, color="white")
for spine in ax3.spines.values(): spine.set_color("#30363d")

# — ACF
ax4 = fig.add_subplot(gs[1, 1])
plot_acf(log_ret, lags=30, zero=False, ax=ax4,
         color="#d2a8ff", vlines_kwargs={"colors": "#d2a8ff"})
ax4.set_title("ACF of Log Returns (30 lags)", color="white", fontsize=12, fontweight="bold")
ax4.set_facecolor("#161b22"); ax4.tick_params(colors="white")
ax4.grid(alpha=0.15, color="white")
for spine in ax4.spines.values(): spine.set_color("#30363d")

eda_path = os.path.join(OUT_DIR, "eda_plots.png")
fig.savefig(eda_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"      Saved → {eda_path}\n")

# ────────────────────────────────────────────────────────
# 4.  DESCRIPTIVE STATISTICS
# ────────────────────────────────────────────────────────
print("[4/6] Descriptive statistics of log returns …")
desc = log_ret.describe()
skewness = float(log_ret.skew())
kurt     = float(log_ret.kurtosis())     # excess kurtosis
print(f"      Mean     : {desc['mean']:.6f}")
print(f"      Std Dev  : {desc['std']:.6f}")
print(f"      Min      : {desc['min']:.6f}")
print(f"      Max      : {desc['max']:.6f}")
print(f"      Skewness : {skewness:.4f}")
print(f"      Kurtosis : {kurt:.4f}  (excess; Normal = 0)")
print()

# ────────────────────────────────────────────────────────
# 5.  STATISTICAL TESTS
# ────────────────────────────────────────────────────────
print("[5/6] Running statistical tests …\n")
alpha = 0.05

# ── (a) Augmented Dickey-Fuller ──────────────────────
adf_p_price   = adfuller(prices)[1]
adf_stat_price = adfuller(prices)[0]
adf_p_ret     = adfuller(log_ret)[1]
adf_stat_ret  = adfuller(log_ret)[0]

print("  ┌──────────────────────────────────────────────┐")
print("  │  (a) Augmented Dickey-Fuller Test             │")
print("  ├──────────────────────────────────────────────┤")
print(f"  │  Prices  — ADF stat: {adf_stat_price:>10.4f}  p = {adf_p_price:.4f}  │")
verdict_p = "Non-stationary (unit root) ✔" if adf_p_price > alpha else "Stationary ✘"
print(f"  │           → {verdict_p:<32s}│")
print(f"  │  Returns — ADF stat: {adf_stat_ret:>10.4f}  p = {adf_p_ret:.6f}│")
verdict_r = "Stationary ✔" if adf_p_ret < alpha else "Non-stationary ✘"
print(f"  │           → {verdict_r:<32s}│")
print("  └──────────────────────────────────────────────┘\n")

# ── (b) Autocorrelation / Ljung-Box ──────────────────
acf_vals, confint, qstat, pvals = acf(log_ret, nlags=20, qstat=True,
                                       fft=True, alpha=0.05)
sig_lags = [i+1 for i, p in enumerate(pvals) if p < alpha]

print("  ┌──────────────────────────────────────────────┐")
print("  │  (b) Autocorrelation / Ljung-Box Q Test       │")
print("  ├──────────────────────────────────────────────┤")
if sig_lags:
    lag_str = ", ".join(str(l) for l in sig_lags[:10])
    if len(sig_lags) > 10:
        lag_str += " …"
    print(f"  │  Significant lags (p<0.05): {lag_str:<17s}│")
    print(f"  │  → Autocorrelation DETECTED                 │")
else:
    print(f"  │  No significant lags at 5% level             │")
    print(f"  │  → No autocorrelation (supports RWH) ✔       │")
print("  └──────────────────────────────────────────────┘\n")

# ── (c) Runs Test ────────────────────────────────────
median_ret = float(log_ret.median())
runs_z, runs_p = runstest_1samp(log_ret.values, cutoff=median_ret)

print("  ┌──────────────────────────────────────────────┐")
print("  │  (c) Wald-Wolfowitz Runs Test                 │")
print("  ├──────────────────────────────────────────────┤")
print(f"  │  Z-statistic : {runs_z:>8.4f}                      │")
print(f"  │  p-value     : {runs_p:>8.4f}                      │")
if runs_p < alpha:
    print(f"  │  → Sequence is NON-random  ✘                 │")
else:
    print(f"  │  → Sequence appears random (supports RWH) ✔  │")
print("  └──────────────────────────────────────────────┘\n")

# ── (d) Variance Ratio Test ─────────────────────────
def variance_ratio(rets, k):
    """Simple VR(k) = Var(k-period returns) / [k * Var(1-period returns)]"""
    arr = rets.values
    n   = len(arr)
    var1 = np.var(arr, ddof=1)
    # k-period overlapping returns
    ret_k = pd.Series(arr).rolling(window=k).sum().dropna().values
    vark  = np.var(ret_k, ddof=1)
    vr    = vark / (k * var1)
    # Asymptotic z-stat under IID null  (Lo-MacKinlay 1988)
    se    = np.sqrt(2*(2*k - 1)*(k - 1) / (3*k*n))
    z     = (vr - 1) / se
    return vr, z

vr2, z2 = variance_ratio(log_ret, 2)
vr5, z5 = variance_ratio(log_ret, 5)
vr10, z10 = variance_ratio(log_ret, 10)
vr20, z20 = variance_ratio(log_ret, 20)

print("  ┌──────────────────────────────────────────────┐")
print("  │  (d) Variance Ratio Test (Lo-MacKinlay)       │")
print("  ├──────────────────────────────────────────────┤")
print(f"  │   k=2  : VR = {vr2:.4f}   z = {z2:>7.4f}             │")
print(f"  │   k=5  : VR = {vr5:.4f}   z = {z5:>7.4f}             │")
print(f"  │   k=10 : VR = {vr10:.4f}   z = {z10:>7.4f}             │")
print(f"  │   k=20 : VR = {vr20:.4f}   z = {z20:>7.4f}             │")
print(f"  │  (VR ≈ 1 supports RWH; |z| > 1.96 rejects)  │")
any_reject = any(abs(z) > 1.96 for z in [z2, z5, z10, z20])
if any_reject:
    print(f"  │  → At least one horizon REJECTS RWH  ✘       │")
else:
    print(f"  │  → All horizons support RWH  ✔               │")
print("  └──────────────────────────────────────────────┘\n")

# ────────────────────────────────────────────────────────
# 6.  SUMMARY TABLE
# ────────────────────────────────────────────────────────
print("[6/6] Final verdict …\n")
results = {
    "ADF (prices non-stationary?)":   "✔ Supports RWH" if adf_p_price > alpha else "✘ Against RWH",
    "ADF (returns stationary?)":      "✔ Supports RWH" if adf_p_ret   < alpha else "✘ Against RWH",
    "Autocorrelation (no sig lags?)": "✘ Against RWH"  if sig_lags            else "✔ Supports RWH",
    "Runs Test (random sequence?)":   "✔ Supports RWH" if runs_p     > alpha  else "✘ Against RWH",
    "Variance Ratio (VR ≈ 1?)":      "✘ Against RWH"  if any_reject          else "✔ Supports RWH",
}

support_count = sum(1 for v in results.values() if "Supports" in v)
total = len(results)

print("  ╔═══════════════════════════════════════════════════╗")
print("  ║        RANDOM WALK HYPOTHESIS — RESULTS TABLE     ║")
print("  ╠═══════════════════════════════════════════════════╣")
for test, verdict in results.items():
    print(f"  ║  {test:<35s} {verdict:<14s}║")
print("  ╠═══════════════════════════════════════════════════╣")
print(f"  ║  Score: {support_count}/{total} tests support the Random Walk     ║")

if support_count == total:
    final = "FULLY SUPPORTED"
elif support_count >= 3:
    final = "PARTIALLY SUPPORTED"
else:
    final = "REJECTED"

print(f"  ║  Verdict:  {final:<39s}║")
print("  ╚═══════════════════════════════════════════════════╝")
print()
print("Done. All figures saved to:", OUT_DIR)

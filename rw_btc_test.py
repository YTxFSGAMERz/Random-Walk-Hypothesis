"""
=============================================================================
Testing the Random Walk Hypothesis on Financial Time Series
=============================================================================
Asset   : Bitcoin (BTC-USD)
Source  : btc-usd-max.csv (local CoinGecko export)
Tests   : ADF, ACF/Ljung-Box, Runs Test, Variance Ratio (Lo-MacKinlay)
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from statsmodels.tsa.stattools import adfuller, acf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.sandbox.stats.runs import runstest_1samp
import os, warnings, textwrap
warnings.filterwarnings("ignore")

OUT = os.path.dirname(os.path.abspath(__file__))

# ═══════════════════════════════════════════════════════
#  1.  LOAD LOCAL CSV
# ═══════════════════════════════════════════════════════
CSV_PATH = os.path.join(OUT, "btc-usd-max.csv")

print("=" * 62)
print("   Random Walk Hypothesis — Bitcoin (BTC-USD)")
print("   Data source: btc-usd-max.csv")
print("=" * 62, "\n")

print("[1/7] Loading CSV …")
df = pd.read_csv(CSV_PATH)
df["snapped_at"] = pd.to_datetime(df["snapped_at"])
df = df.sort_values("snapped_at").set_index("snapped_at")
df = df.dropna(subset=["price"])
print(f"      Rows loaded     : {len(df)}")
print(f"      Date range      : {df.index[0].date()} → {df.index[-1].date()}")
print(f"      Total span      : {(df.index[-1] - df.index[0]).days} days\n")

prices = df["price"]

# ═══════════════════════════════════════════════════════
#  2.  LOG RETURNS
# ═══════════════════════════════════════════════════════
print("[2/7] Computing log returns …")
log_ret = np.log(prices / prices.shift(1)).dropna()
print(f"      Return observations: {len(log_ret)}\n")

# ═══════════════════════════════════════════════════════
#  3.  DESCRIPTIVE STATISTICS
# ═══════════════════════════════════════════════════════
print("[3/7] Descriptive statistics …")
mean_r   = float(log_ret.mean())
std_r    = float(log_ret.std())
min_r    = float(log_ret.min())
max_r    = float(log_ret.max())
skew_r   = float(log_ret.skew())
kurt_r   = float(log_ret.kurtosis())

stats_table = f"""
      ┌────────────────────────────────┐
      │  Mean        : {mean_r:>12.6f}     │
      │  Std Dev     : {std_r:>12.6f}     │
      │  Min         : {min_r:>12.6f}     │
      │  Max         : {max_r:>12.6f}     │
      │  Skewness    : {skew_r:>12.4f}     │
      │  Kurtosis    : {kurt_r:>12.4f}     │
      │  (excess; Normal = 0)          │
      └────────────────────────────────┘
"""
print(stats_table)

# ═══════════════════════════════════════════════════════
#  4.  PLOTS  (dark-themed, publication quality)
# ═══════════════════════════════════════════════════════
print("[4/7] Generating EDA plots …")

BG    = "#0d1117"
CARD  = "#161b22"
EDGE  = "#30363d"
TXT   = "white"

def style_ax(ax):
    ax.set_facecolor(CARD)
    ax.tick_params(colors=TXT, labelsize=8)
    ax.grid(alpha=0.12, color=TXT)
    for s in ax.spines.values():
        s.set_color(EDGE)

fig = plt.figure(figsize=(18, 12), facecolor=BG)
gs  = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.30)

# (A) Price series
ax1 = fig.add_subplot(gs[0, 0:2])
ax1.plot(prices.index, prices.values, color="#58a6ff", lw=0.8)
ax1.set_title("BTC-USD — Daily Price (Full History)", color=TXT, fontsize=13, fontweight="bold")
ax1.set_ylabel("Price (USD)", color=TXT)
style_ax(ax1)

# (B) Log returns
ax2 = fig.add_subplot(gs[0, 2])
ax2.plot(log_ret.index, log_ret.values, color="#f97583", lw=0.35, alpha=0.85)
ax2.axhline(0, color="#8b949e", lw=0.5, ls="--")
ax2.set_title("Daily Log Returns", color=TXT, fontsize=13, fontweight="bold")
style_ax(ax2)

# (C) Histogram + Normal overlay
ax3 = fig.add_subplot(gs[1, 0])
ax3.hist(log_ret.values, bins=100, color="#7ee787", edgecolor=BG, alpha=0.85, density=True)
x_r = np.linspace(log_ret.min(), log_ret.max(), 400)
ax3.plot(x_r, (1/(std_r*np.sqrt(2*np.pi)))*np.exp(-0.5*((x_r - mean_r)/std_r)**2),
         color="#f0883e", lw=1.8, label="Normal fit")
ax3.legend(facecolor=CARD, edgecolor=EDGE, labelcolor=TXT)
ax3.set_title("Return Distribution vs Normal", color=TXT, fontsize=13, fontweight="bold")
style_ax(ax3)

# (D) ACF
ax4 = fig.add_subplot(gs[1, 1])
plot_acf(log_ret, lags=40, zero=False, ax=ax4, color="#d2a8ff",
         vlines_kwargs={"colors": "#d2a8ff"})
ax4.set_title("ACF of Log Returns (40 lags)", color=TXT, fontsize=13, fontweight="bold")
style_ax(ax4)

# (E) Rolling 30-day volatility
ax5 = fig.add_subplot(gs[1, 2])
rolling_vol = log_ret.rolling(30).std() * np.sqrt(365)
ax5.fill_between(rolling_vol.index, rolling_vol.values, color="#bc8cff", alpha=0.5)
ax5.plot(rolling_vol.index, rolling_vol.values, color="#d2a8ff", lw=0.6)
ax5.set_title("30-Day Rolling Annualized Volatility", color=TXT, fontsize=13, fontweight="bold")
ax5.set_ylabel("Volatility", color=TXT)
style_ax(ax5)

fig.suptitle("Random Walk Hypothesis — Bitcoin (BTC-USD) Exploratory Analysis",
             color="#58a6ff", fontsize=16, fontweight="bold", y=0.98)

eda_path = os.path.join(OUT, "btc_eda_plots.png")
fig.savefig(eda_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"      Saved → {eda_path}\n")

# ═══════════════════════════════════════════════════════
#  5.  STATISTICAL TESTS
# ═══════════════════════════════════════════════════════
print("[5/7] Running statistical tests …\n")
ALPHA = 0.05

# ── (a) Augmented Dickey-Fuller ──────────────────────
adf_price    = adfuller(prices)
adf_returns  = adfuller(log_ret)

print("  ┌─────────────────────────────────────────────────────┐")
print("  │  TEST (a): Augmented Dickey-Fuller                   │")
print("  ├─────────────────────────────────────────────────────┤")
print(f"  │  Prices  — ADF stat: {adf_price[0]:>10.4f}   p = {adf_price[1]:.4f}        │")
vp = "Non-stationary (unit root) ✔" if adf_price[1] > ALPHA else "Stationary ✘ (unexpected)"
print(f"  │            → {vp:<38s}│")
print(f"  │  Returns — ADF stat: {adf_returns[0]:>10.4f}   p = {adf_returns[1]:.6f}  │")
vr = "Stationary ✔" if adf_returns[1] < ALPHA else "Non-stationary ✘"
print(f"  │            → {vr:<38s}│")
print("  └─────────────────────────────────────────────────────┘\n")

# ── (b) Autocorrelation / Ljung-Box ──────────────────
acf_vals, confint, qstat, pvals = acf(log_ret, nlags=30, qstat=True,
                                       fft=True, alpha=0.05)
sig_lags = [i+1 for i, p in enumerate(pvals) if p < ALPHA]

print("  ┌─────────────────────────────────────────────────────┐")
print("  │  TEST (b): Autocorrelation / Ljung-Box Q             │")
print("  ├─────────────────────────────────────────────────────┤")
if sig_lags:
    lag_str = ", ".join(str(l) for l in sig_lags[:12])
    if len(sig_lags) > 12: lag_str += " …"
    print(f"  │  Significant lags: {lag_str:<33s}│")
    first_q = qstat[sig_lags[0]-1]
    first_p = pvals[sig_lags[0]-1]
    print(f"  │  First sig lag {sig_lags[0]}: Q={first_q:.2f}, p={first_p:.4f}           │")
    print(f"  │  → Autocorrelation DETECTED  ✘                     │")
else:
    print(f"  │  No significant lags at 5% level                    │")
    print(f"  │  → No autocorrelation (supports RWH) ✔              │")
print("  └─────────────────────────────────────────────────────┘\n")

# ── (c) Runs Test ────────────────────────────────────
median_ret = float(log_ret.median())
runs_z, runs_p = runstest_1samp(log_ret.values, cutoff=median_ret)

print("  ┌─────────────────────────────────────────────────────┐")
print("  │  TEST (c): Wald-Wolfowitz Runs Test                  │")
print("  ├─────────────────────────────────────────────────────┤")
print(f"  │  Z-statistic : {runs_z:>9.4f}                           │")
print(f"  │  p-value     : {runs_p:>9.4f}                           │")
if runs_p < ALPHA:
    z_interp = "momentum (fewer runs)" if runs_z > 0 else "mean-reversion (more runs)"
    print(f"  │  → NON-random sequence  ✘  ({z_interp})│")
else:
    print(f"  │  → Sequence appears random ✔ (supports RWH)         │")
print("  └─────────────────────────────────────────────────────┘\n")

# ── (d) Variance Ratio (Lo-MacKinlay 1988) ──────────
def variance_ratio_test(rets, k):
    """VR(k) with asymptotic z under IID null."""
    arr = rets.values
    n   = len(arr)
    var1 = np.var(arr, ddof=1)
    ret_k = pd.Series(arr).rolling(window=k).sum().dropna().values
    vark  = np.var(ret_k, ddof=1)
    vr    = vark / (k * var1)
    se    = np.sqrt(2*(2*k - 1)*(k - 1) / (3*k*n))
    z     = (vr - 1) / se
    return vr, z

horizons = [2, 5, 10, 20, 30]
vr_results = {k: variance_ratio_test(log_ret, k) for k in horizons}

print("  ┌─────────────────────────────────────────────────────┐")
print("  │  TEST (d): Variance Ratio (Lo-MacKinlay 1988)        │")
print("  ├─────────────────────────────────────────────────────┤")
for k, (vr_val, z_val) in vr_results.items():
    sig_marker = "✘" if abs(z_val) > 1.96 else "✔"
    print(f"  │   k={k:<3d}: VR = {vr_val:.4f}   z = {z_val:>8.4f}   {sig_marker}            │")
print(f"  │  (VR ≈ 1 supports RWH; |z| > 1.96 rejects at 5%)   │")
reject_ks = [k for k, (_, z) in vr_results.items() if abs(z) > 1.96]
if reject_ks:
    print(f"  │  → Rejects at k = {reject_ks}              │")
else:
    print(f"  │  → All horizons support RWH  ✔                      │")
print("  └─────────────────────────────────────────────────────┘\n")

# ═══════════════════════════════════════════════════════
#  6.  VARIANCE RATIO PLOT
# ═══════════════════════════════════════════════════════
print("[6/7] Generating Variance Ratio plot …")

fig2, ax_vr = plt.subplots(figsize=(10, 5), facecolor=BG)
ks_plot = list(range(2, 51))
vrs_plot = [variance_ratio_test(log_ret, k)[0] for k in ks_plot]

ax_vr.plot(ks_plot, vrs_plot, color="#58a6ff", lw=2, marker="o", markersize=3)
ax_vr.axhline(1.0, color="#f97583", lw=1.5, ls="--", label="VR = 1.0 (Random Walk)")
ax_vr.fill_between(ks_plot, 1.0, vrs_plot, alpha=0.15, color="#58a6ff")
ax_vr.set_xlabel("Holding Period (k days)", color=TXT, fontsize=11)
ax_vr.set_ylabel("Variance Ratio VR(k)", color=TXT, fontsize=11)
ax_vr.set_title("Variance Ratio Profile — BTC-USD", color=TXT, fontsize=14, fontweight="bold")
ax_vr.legend(facecolor=CARD, edgecolor=EDGE, labelcolor=TXT)
style_ax(ax_vr)

vr_path = os.path.join(OUT, "btc_variance_ratio.png")
fig2.savefig(vr_path, dpi=150, bbox_inches="tight", facecolor=fig2.get_facecolor())
plt.close(fig2)
print(f"      Saved → {vr_path}\n")

# ═══════════════════════════════════════════════════════
#  7.  FINAL VERDICT
# ═══════════════════════════════════════════════════════
print("[7/7] Final verdict …\n")

tests = {
    "ADF: prices non-stationary?":   adf_price[1] > ALPHA,
    "ADF: returns stationary?":      adf_returns[1] < ALPHA,
    "ACF: no significant lags?":     len(sig_lags) == 0,
    "Runs: random sequence?":        runs_p > ALPHA,
    "VR: all horizons VR ≈ 1?":      len(reject_ks) == 0,
}

support = sum(tests.values())
total   = len(tests)

print("  ╔════════════════════════════════════════════════════════════╗")
print("  ║      RANDOM WALK HYPOTHESIS — FINAL RESULTS TABLE         ║")
print("  ║      Asset: Bitcoin (BTC-USD)                              ║")
print(f"  ║      Period: {df.index[0].date()} → {df.index[-1].date()}                   ║")
print("  ╠════════════════════════════════════════════════════════════╣")
for test_name, passed in tests.items():
    icon = "✔ Supports RWH" if passed else "✘ Against RWH "
    print(f"  ║  {test_name:<35s}  {icon:<20s}  ║")
print("  ╠════════════════════════════════════════════════════════════╣")
print(f"  ║  Score: {support}/{total} tests support the Random Walk Hypothesis      ║")

if support == total:
    verdict = "FULLY SUPPORTED"
elif support >= 3:
    verdict = "PARTIALLY SUPPORTED"
else:
    verdict = "REJECTED"

print(f"  ║  >>>  Verdict:  {verdict:<42s}║")
print("  ╚════════════════════════════════════════════════════════════╝")
print()
print(f"  Charts saved to: {OUT}")
print("  Done.\n")

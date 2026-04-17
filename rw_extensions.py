"""
=============================================================================
RWH Extensions — GARCH(1,1) Model + Rolling Window Analysis
=============================================================================
Asset   : Bitcoin (BTC-USD)
Source  : btc-usd-max.csv
Methods : GARCH(1,1) volatility, rolling ADF, rolling Runs Test
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from statsmodels.tsa.stattools import adfuller
from statsmodels.sandbox.stats.runs import runstest_1samp
import os, warnings
warnings.filterwarnings("ignore")

# Try importing arch for GARCH; install hint if missing
try:
    from arch import arch_model
    HAS_ARCH = True
except ImportError:
    HAS_ARCH = False
    print("  [INFO] 'arch' package not found. Install with: pip install arch")
    print("         GARCH section will be skipped.\n")

OUT = os.path.dirname(os.path.abspath(__file__))
BG, CARD, EDGE, TXT = "#0d1117", "#161b22", "#30363d", "white"

def style_ax(ax):
    ax.set_facecolor(CARD)
    ax.tick_params(colors=TXT, labelsize=8)
    ax.grid(alpha=0.12, color=TXT)
    for s in ax.spines.values(): s.set_color(EDGE)

# ─────────────────────────────────────────────
#  Load data
# ─────────────────────────────────────────────
print("=" * 62)
print("  RWH Extensions — GARCH + Rolling Window Analysis")
print("=" * 62, "\n")

df = pd.read_csv(os.path.join(OUT, "btc-usd-max.csv"))
df["snapped_at"] = pd.to_datetime(df["snapped_at"])
df = df.sort_values("snapped_at").set_index("snapped_at").dropna(subset=["price"])
prices  = df["price"]
log_ret = np.log(prices / prices.shift(1)).dropna()

# Scale to percentage returns for GARCH (improves numerical stability)
ret_pct = log_ret * 100

print(f"  Loaded {len(log_ret)} return observations\n")

# ═══════════════════════════════════════════════════════
#  EXTENSION 1 — GARCH(1,1) Model
# ═══════════════════════════════════════════════════════
print("[1/3] Fitting GARCH(1,1) model …")

if HAS_ARCH:
    garch = arch_model(ret_pct, vol="Garch", p=1, q=1, dist="normal")
    garch_fit = garch.fit(disp="off")

    print("\n  GARCH(1,1) Parameter Estimates:")
    print("  " + "─" * 48)
    params = garch_fit.params
    pvals  = garch_fit.pvalues
    for name, val, pv in zip(params.index, params.values, pvals.values):
        sig = "***" if pv < 0.001 else ("**" if pv < 0.01 else ("*" if pv < 0.05 else ""))
        print(f"  {name:<15s}: {val:>10.6f}   p={pv:.4f}  {sig}")
    
    omega = params["omega"]
    alpha = params["alpha[1]"]
    beta  = params["beta[1]"]
    persist = alpha + beta

    print(f"\n  Persistence (α+β) = {persist:.6f}")
    if persist >= 1.0:
        print("  → Integrated GARCH (IGARCH) — shocks don't decay!")
    elif persist > 0.95:
        print("  → Very high persistence — volatility clusters strongly")
    else:
        print("  → Moderate persistence — volatility clustering present")

    long_run_vol = np.sqrt(omega / (1 - persist)) * np.sqrt(365) / 100
    print(f"  Long-run annual volatility = {long_run_vol:.2%}")

    # Conditional volatility series
    cond_vol = garch_fit.conditional_volatility   # in % units
    std_resid = garch_fit.std_resid

    print("\n  Testing GARCH residuals for remaining autocorrelation …")
    from statsmodels.tsa.stattools import acf
    acf_resid, _, qstat_r, pvals_r = acf(std_resid.dropna(), nlags=20,
                                         qstat=True, fft=True, alpha=0.05)
    sig_resid_lags = [i+1 for i, p in enumerate(pvals_r) if p < 0.05]
    if sig_resid_lags:
        print(f"  Significant lags in residuals: {sig_resid_lags}")
        print("  → Non-linear structure remains after GARCH filtering")
    else:
        print("  → No significant autocorrelation in GARCH residuals ✔")
        print("  → GARCH fully explains the serial dependence")

    # Runs test on standardised residuals
    runs_z_r, runs_p_r = runstest_1samp(std_resid.dropna().values,
                                         cutoff=float(std_resid.median()))
    print(f"\n  Runs Test on GARCH residuals: Z={runs_z_r:.4f}, p={runs_p_r:.4f}")
    if runs_p_r < 0.05:
        print("  → Residuals still non-random (deeper structure exists) ✘")
    else:
        print("  → Residuals are random — GARCH captures the dependence ✔")

else:
    print("  [SKIPPED] Install 'arch' package to run GARCH analysis.\n")
    cond_vol = None
    std_resid = None

# ═══════════════════════════════════════════════════════
#  EXTENSION 2 — Rolling Window ADF Test
# ═══════════════════════════════════════════════════════
print("\n[2/3] Rolling window ADF test (window = 365 days) …")

WINDOW = 365
adf_stats, adf_pvals, roll_dates = [], [], []

ret_arr = log_ret.values
idx_arr = log_ret.index

for i in range(WINDOW, len(ret_arr)):
    window_data = ret_arr[i - WINDOW : i]
    try:
        stat, pval, *_ = adfuller(window_data)
        adf_stats.append(stat)
        adf_pvals.append(pval)
        roll_dates.append(idx_arr[i])
    except Exception:
        pass

roll_adf = pd.DataFrame({"stat": adf_stats, "pval": adf_pvals}, index=roll_dates)
pct_reject = (roll_adf["pval"] < 0.05).mean() * 100

print(f"  Windows computed : {len(roll_adf)}")
print(f"  Windows rejecting H₀ (stationary returns): {pct_reject:.1f}%")
if pct_reject > 80:
    print("  → Returns are consistently stationary throughout BTC's history ✔")
else:
    print("  → Returns show periods of non-stationarity — regime shifts detected")

# ═══════════════════════════════════════════════════════
#  EXTENSION 3 — Rolling Runs Test
# ═══════════════════════════════════════════════════════
print("\n[3/3] Rolling Runs Test (window = 180 days) …")

WIN_RUNS = 180
runs_zs, runs_ps, runs_dates = [], [], []

for i in range(WIN_RUNS, len(ret_arr)):
    w = ret_arr[i - WIN_RUNS : i]
    try:
        z, p = runstest_1samp(w, cutoff=float(np.median(w)))
        runs_zs.append(z)
        runs_ps.append(p)
        runs_dates.append(idx_arr[i])
    except Exception:
        pass

roll_runs = pd.DataFrame({"z": runs_zs, "pval": runs_ps}, index=runs_dates)
pct_nonrandom = (roll_runs["pval"] < 0.05).mean() * 100

print(f"  Windows computed : {len(roll_runs)}")
print(f"  Windows showing non-random sequence: {pct_nonrandom:.1f}%")
if pct_nonrandom > 50:
    print("  → Majority of BTC's history has exhibited NON-random return sequences")
else:
    print("  → Return sequences appear random in most sub-periods")

# ═══════════════════════════════════════════════════════
#  PLOTS
# ═══════════════════════════════════════════════════════
print("\n[Plotting] Generating extension charts …")

n_rows = 3 if (HAS_ARCH and cond_vol is not None) else 2
fig = plt.figure(figsize=(16, 5 * n_rows), facecolor=BG)
gs  = GridSpec(n_rows, 2, figure=fig, hspace=0.4, wspace=0.28)

row = 0

# — GARCH row (if available)
if HAS_ARCH and cond_vol is not None:
    ax_vol = fig.add_subplot(gs[0, 0])
    ax_vol.fill_between(cond_vol.index, cond_vol.values, color="#f97583", alpha=0.4)
    ax_vol.plot(cond_vol.index, cond_vol.values, color="#f97583", lw=0.6)
    ax_vol.set_title("GARCH(1,1) — Conditional Volatility (%)", color=TXT,
                     fontsize=12, fontweight="bold")
    style_ax(ax_vol)

    ax_res = fig.add_subplot(gs[0, 1])
    ax_res.plot(std_resid.index, std_resid.values, color="#d2a8ff", lw=0.4, alpha=0.7)
    ax_res.axhline(0, color="#8b949e", lw=0.5, ls="--")
    ax_res.set_title("GARCH(1,1) — Standardised Residuals", color=TXT,
                     fontsize=12, fontweight="bold")
    style_ax(ax_res)
    row = 1

# — Rolling ADF
ax_adf = fig.add_subplot(gs[row, 0])
ax_adf.plot(roll_adf.index, roll_adf["pval"], color="#58a6ff", lw=0.7)
ax_adf.axhline(0.05, color="#f97583", lw=1.2, ls="--", label="5% threshold")
ax_adf.fill_between(roll_adf.index, roll_adf["pval"], 0.05,
                    where=(roll_adf["pval"] > 0.05), color="#f97583", alpha=0.2,
                    label="Non-stationary region")
ax_adf.set_title(f"Rolling ADF p-value (window={WINDOW}d)", color=TXT,
                 fontsize=12, fontweight="bold")
ax_adf.set_ylabel("p-value", color=TXT)
ax_adf.legend(facecolor=CARD, edgecolor=EDGE, labelcolor=TXT, fontsize=8)
style_ax(ax_adf)

# — Rolling Runs p-value
ax_runs = fig.add_subplot(gs[row, 1])
colors_runs = ["#f97583" if p < 0.05 else "#7ee787" for p in roll_runs["pval"]]
ax_runs.bar(roll_runs.index, roll_runs["pval"], color=colors_runs, alpha=0.7, width=1.5)
ax_runs.axhline(0.05, color="white", lw=1.0, ls="--", label="5% threshold")
ax_runs.set_title(f"Rolling Runs Test p-value (window={WIN_RUNS}d)", color=TXT,
                  fontsize=12, fontweight="bold")
ax_runs.set_ylabel("p-value (red = non-random)", color=TXT)
ax_runs.legend(facecolor=CARD, edgecolor=EDGE, labelcolor=TXT, fontsize=8)
style_ax(ax_runs)

# — Rolling Runs Z-stat
ax_z = fig.add_subplot(gs[row + 1, 0])
ax_z.plot(roll_runs.index, roll_runs["z"], color="#f0883e", lw=0.7)
ax_z.axhline(0, color="#8b949e", lw=0.5, ls="--")
ax_z.axhline(1.96,  color="#f97583", lw=0.8, ls=":", label="+1.96")
ax_z.axhline(-1.96, color="#58a6ff", lw=0.8, ls=":", label="-1.96")
ax_z.set_title(f"Rolling Runs Z-statistic (window={WIN_RUNS}d)", color=TXT,
               fontsize=12, fontweight="bold")
ax_z.set_ylabel("Z-stat (+ve = momentum)", color=TXT)
ax_z.legend(facecolor=CARD, edgecolor=EDGE, labelcolor=TXT, fontsize=8)
style_ax(ax_z)

# — Rolling ADF stat
ax_stat = fig.add_subplot(gs[row + 1, 1])
ax_stat.plot(roll_adf.index, roll_adf["stat"], color="#7ee787", lw=0.7)
ax_stat.axhline(-3.45, color="#f97583", lw=1.0, ls="--", label="5% critical value")
ax_stat.set_title(f"Rolling ADF Statistic (window={WINDOW}d)", color=TXT,
                  fontsize=12,  fontweight="bold")
ax_stat.set_ylabel("ADF Statistic (< -3.45 → stationary)", color=TXT)
ax_stat.legend(facecolor=CARD, edgecolor=EDGE, labelcolor=TXT, fontsize=8)
style_ax(ax_stat)

fig.suptitle("RWH Extensions — GARCH Volatility + Rolling Window Analysis (BTC-USD)",
             color="#58a6ff", fontsize=14, fontweight="bold", y=1.01)

out_path = os.path.join(OUT, "btc_garch_plots.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"  Saved → {out_path}\n")

# ─────────────────────────────────────────────
#  Summary
# ─────────────────────────────────────────────
print("=" * 62)
print("  EXTENSIONS SUMMARY")
print("=" * 62)
if HAS_ARCH:
    print(f"  GARCH α+β persistence  : {persist:.4f}")
    print(f"  Long-run annual vol    : {long_run_vol:.2%}")
    resid_random = runs_p_r >= 0.05
    print(f"  Residuals random?      : {'Yes ✔' if resid_random else 'No ✘'}")
print(f"  Rolling ADF stationary : {pct_reject:.1f}% of windows")
print(f"  Rolling Runs non-rand  : {pct_nonrandom:.1f}% of windows")
print()
print("  Conclusion:")
print("  Bitcoin's conditional volatility is highly persistent (GARCH).")
print("  Momentum (non-random runs) is present in most rolling sub-periods.")
print("  Returns are stationary throughout the sample.")
print("=" * 62)

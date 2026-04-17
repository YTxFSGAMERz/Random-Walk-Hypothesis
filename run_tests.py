import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, acf
from statsmodels.sandbox.stats.runs import runstest_1samp
import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv('btc-usd-max.csv')
data['snapped_at'] = pd.to_datetime(data['snapped_at'])
data = data.sort_values('snapped_at').set_index('snapped_at')
data = data.dropna(subset=['price'])
data['Log_Return'] = np.log(data['price'] / data['price'].shift(1))
data = data.dropna(subset=['Log_Return'])

adf_price = adfuller(data['price'])[1]
adf_returns = adfuller(data['Log_Return'])[1]

acf_vals, qstat, pvals = acf(data['Log_Return'], nlags=20, qstat=True, fft=True)
sig_lags = np.where(pvals < 0.05)[0] + 1

median_ret = data['Log_Return'].median()
runs_stat, runs_p = runstest_1samp(data['Log_Return'], cutoff=median_ret)

def vr(rets, k):
    var_1 = np.var(rets, ddof=1)
    ret_k = rets.rolling(window=k).sum().dropna()
    var_k = np.var(ret_k, ddof=1)
    return var_k / (k * var_1)

vr_2 = vr(data['Log_Return'], 2)
vr_5 = vr(data['Log_Return'], 5)

print(f'Price p-value: {adf_price:.4f}')
print(f'Returns p-value: {adf_returns:.4f}')
print(f'Significant lags: {list(sig_lags)}')
print(f'Runs Z-stat: {runs_stat:.4f}')
print(f'Runs p-value: {runs_p:.4f}')
print(f'VR(2): {vr_2:.4f}')
print(f'VR(5): {vr_5:.4f}')

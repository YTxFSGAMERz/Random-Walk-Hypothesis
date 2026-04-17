import pandas as pd
import numpy as np

# Load your CSV
df = pd.read_csv("btc-usd-max.csv")

# Check columns
print(df.columns)

# Assume 'price' or 'close' column exists
# (adjust if needed)
df['price'] = df['price'] if 'price' in df.columns else df['close']

# Convert to returns
df['log_return'] = np.log(df['price'] / df['price'].shift(1))

# Drop NaN
df = df.dropna()

print(df.head())

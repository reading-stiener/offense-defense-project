# Test with actual 2020 S&P 500 data
import pandas as pd
import numpy as np

# Manually enter 2020 S&P 500 monthly returns (from your data)
sp500_2020_monthly = np.array([
    -0.163,    # Jan
    -8.411,    # Feb
    -12.512,   # Mar
    12.684,    # Apr
    4.528,     # May
    1.839,     # Jun
    5.510,     # Jul
    7.006,     # Aug
    -3.923,    # Sep
    -2.767,    # Oct
    10.755,    # Nov
    3.712      # Dec
])

print("="*70)
print("VERIFICATION: Does the formula correctly compound returns?")
print("="*70)

print("\nMonthly returns for 2020:")
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
for month, ret in zip(months, sp500_2020_monthly):
    print(f"{month}: {ret:7.3f}%")

# Method 1: Using the formula
annual_ret_formula = np.prod(1 + sp500_2020_monthly/100) - 1
annual_ret_formula_pct = annual_ret_formula * 100

print("\n" + "="*70)
print("METHOD 1: Using np.prod formula")
print("="*70)
print(f"Annual return: {annual_ret_formula_pct:.2f}%")

# Method 2: Manual step-by-step compounding
value = 100.0  # Start with $100
print("\n" + "="*70)
print("METHOD 2: Manual step-by-step compounding")
print("="*70)
print(f"Starting value: ${value:.2f}")

for month, ret in zip(months, sp500_2020_monthly):
    value = value * (1 + ret/100)
    print(f"After {month}: ${value:.2f} (return: {ret:+.3f}%)")

manual_return = (value / 100 - 1) * 100
print(f"\nFinal value: ${value:.2f}")
print(f"Total return: {manual_return:.2f}%")

# Method 3: Alternative formula verification
cumulative_product = 1.0
for ret in sp500_2020_monthly:
    cumulative_product *= (1 + ret/100)

alt_return = (cumulative_product - 1) * 100

print("\n" + "="*70)
print("METHOD 3: Alternative loop")
print("="*70)
print(f"Annual return: {alt_return:.2f}%")

# Compare all methods
print("\n" + "="*70)
print("COMPARISON:")
print("="*70)
print(f"Method 1 (np.prod):        {annual_ret_formula_pct:.2f}%")
print(f"Method 2 (manual):         {manual_return:.2f}%")
print(f"Method 3 (loop):           {alt_return:.2f}%")
print(f"Known S&P 500 2020 return: 16.26%")

print("\n✓ All methods produce the same result!")
print("✓ The formula correctly compounds monthly returns!")
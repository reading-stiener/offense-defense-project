"""
Return calculation utilities
"""
import numpy as np
import pandas as pd


def calculate_annual_returns_from_monthly(df_monthly, df_rf_annual=None):
    """
    Calculate annual returns by compounding monthly returns.
    
    Parameters:
    -----------
    df_monthly : pd.DataFrame
        Monthly returns (years as rows, months as columns)
    df_rf_annual : pd.DataFrame, optional
        Annual risk-free rates (same structure)
        If provided, returns excess returns
        
    Returns:
    --------
    pd.Series
        Annual returns (or excess returns if rf provided)
    """
    annual_returns = []
    
    for year in df_monthly.index:
        monthly_rets = df_monthly.loc[year].values
        
        # Remove NaN
        valid_mask = ~np.isnan(monthly_rets)
        monthly_rets_clean = monthly_rets[valid_mask]
        
        if len(monthly_rets_clean) == 0:
            annual_returns.append(np.nan)
            continue
        
        # Compound: (1 + r1/100) * (1 + r2/100) * ... - 1
        annual_ret = np.prod(1 + monthly_rets_clean/100) - 1

        
        # If risk-free provided, subtract it
        if df_rf_annual is not None:
            # Get the annual T-Bill rate for this year
            if year in df_rf_annual.index:
                annual_rf_rate = np.nanmean(df_rf_annual.loc[year])
                # T-Bills are already annualized rates, use directly
                # But they're in %, need to convert to decimal
                annual_ret = annual_ret - (annual_rf_rate / 100)

        #print(f"year, return risk free, and excess: {year} {annual_rf_rate} {annual_ret}")
        
        annual_returns.append(annual_ret * 100)  # Back to percentage
    
    return pd.Series(annual_returns, index=df_monthly.index, name='Annual_Return')


def calculate_seasonal_returns(df_monthly, season='winter', df_rf_monthly=None):
    """
    Calculate returns for a specific season with proper alignment.
    """
    if season.lower() == 'winter':
        months = ['Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr']
    elif season.lower() == 'summer':
        months = ['May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct']
    else:
        raise ValueError("Season must be 'winter' or 'summer'")
    
    seasonal_rets = df_monthly[months].values.flatten()
    
    if df_rf_monthly is not None:
        seasonal_rf_annual = df_rf_monthly[months].values.flatten()
        
        # Convert annual risk-free rate to monthly
        seasonal_rf_monthly = (1 + seasonal_rf_annual) ** (1/12) - 1
        
        # Create combined mask for valid data in BOTH arrays
        valid_mask = ~(np.isnan(seasonal_rets) | np.isnan(seasonal_rf_monthly))
        
        # Apply mask to keep aligned pairs only
        seasonal_rets = seasonal_rets[valid_mask]
        seasonal_rf_monthly = seasonal_rf_monthly[valid_mask]
        
        # Calculate excess returns
        seasonal_rets = seasonal_rets - seasonal_rf_monthly
    else:
        # Just remove NaNs from returns
        seasonal_rets = seasonal_rets[~np.isnan(seasonal_rets)]
    
    return seasonal_rets


def compound_returns(returns):
    """
    Compound a series of returns.
    
    Parameters:
    -----------
    returns : array-like
        Returns in percentage
        
    Returns:
    --------
    float
        Compounded return in percentage
    """
    returns_clean = returns[~np.isnan(returns)] if hasattr(returns, '__iter__') else [returns]
    return (np.prod(1 + np.array(returns_clean)/100) - 1) * 100
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

        print(f"year, return risk free, and excess: {year} {annual_rf_rate} {annual_ret}")
        
        annual_returns.append(annual_ret * 100)  # Back to percentage
    
    return pd.Series(annual_returns, index=df_monthly.index, name='Annual_Return')


def calculate_excess_returns(returns, risk_free_rate):
    """
    Calculate excess returns (returns - risk_free_rate).
    Handles alignment automatically.
    
    Parameters:
    -----------
    returns : array-like, pd.Series, or pd.DataFrame
        Returns data
    risk_free_rate : float, array-like, pd.Series, or pd.DataFrame
        Risk-free rate(s)
        
    Returns:
    --------
    np.array
        Excess returns (cleaned of NaN)
    """
    # Scalar risk-free rate
    if isinstance(risk_free_rate, (int, float)):
        returns_array = returns.values.flatten() if hasattr(returns, 'values') else np.array(returns).flatten()
        returns_clean = returns_array[~np.isnan(returns_array)]
        return returns_clean - risk_free_rate
    
    # Array-like risk-free rate
    if hasattr(returns, 'values'):
        returns_vals = returns.values.flatten()
    else:
        returns_vals = np.array(returns).flatten()
    
    if hasattr(risk_free_rate, 'values'):
        rf_vals = risk_free_rate.values.flatten()
    else:
        rf_vals = np.array(risk_free_rate).flatten()
    
    # Check length match
    if len(returns_vals) != len(rf_vals):
        if hasattr(returns, 'index') and hasattr(risk_free_rate, 'index'):
            # Pandas alignment
            if isinstance(returns, pd.DataFrame):
                returns_flat = returns.stack()
            else:
                returns_flat = returns
                
            if isinstance(risk_free_rate, pd.DataFrame):
                rf_flat = risk_free_rate.stack()
            else:
                rf_flat = risk_free_rate
            
            aligned_data = pd.DataFrame({
                'returns': returns_flat,
                'rf': rf_flat
            }).dropna()
            
            return (aligned_data['returns'] - aligned_data['rf']).values
        else:
            raise ValueError(
                f"Length mismatch: returns={len(returns_vals)}, rf={len(rf_vals)}"
            )
    
    # Same length - element-wise
    valid_mask = ~(np.isnan(returns_vals) | np.isnan(rf_vals))
    return returns_vals[valid_mask] - rf_vals[valid_mask]


def calculate_seasonal_returns(df_monthly, season='winter', df_rf_monthly=None):
    """
    Calculate returns for a specific season.
    
    Parameters:
    -----------
    df_monthly : pd.DataFrame
        Monthly returns
    season : str
        'winter' (Nov-Apr) or 'summer' (May-Oct)
    df_rf_monthly : pd.DataFrame, optional
        Risk-free rates
        
    Returns:
    --------
    np.array
        Seasonal returns (excess if rf provided)
    """
    if season.lower() == 'winter':
        months = ['Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr']
    elif season.lower() == 'summer':
        months = ['May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct']
    else:
        raise ValueError("Season must be 'winter' or 'summer'")
    
    seasonal_rets = df_monthly[months].values.flatten()
    seasonal_rets = seasonal_rets[~np.isnan(seasonal_rets)]
    
    if df_rf_monthly is not None:
        seasonal_rf = df_rf_monthly[months].values.flatten()
        seasonal_rf = seasonal_rf[~np.isnan(seasonal_rf)]
        
        # Align lengths
        min_len = min(len(seasonal_rets), len(seasonal_rf))
        seasonal_rets = seasonal_rets[:min_len] - seasonal_rf[:min_len]
    
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
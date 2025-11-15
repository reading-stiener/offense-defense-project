"""
Statistical analysis functions including Deflated Sharpe Ratio
"""
import numpy as np
from scipy import stats
from scipy.stats import norm


def calculate_statistics(returns):
    """
    Calculate comprehensive statistics for a return series.
    
    Parameters:
    -----------
    returns : array-like
        Array of returns
        
    Returns:
    --------
    dict
        Dictionary of statistics
    """
    returns = np.array(returns)
    
    stats_dict = {
        'count': len(returns),
        'mean': np.mean(returns),
        'median': np.median(returns),
        'std': np.std(returns, ddof=1),
        'min': np.min(returns),
        'max': np.max(returns),
        'skewness': stats.skew(returns),
        'kurtosis': stats.kurtosis(returns, fisher=True),  # Excess kurtosis
        #'sharpe_ratio': calculate_sharpe_ratio(returns),
        'positive_pct': (returns > 0).sum() / len(returns) * 100
    }
    
    return stats_dict


# Calculate annual returns by compounding monthly returns
def calculate_annual_returns(df_monthly, df_rf_monthly):
    """
    Calculate annual returns from monthly data.
    
    Parameters:
    -----------
    df_monthly : pd.DataFrame
        Monthly returns (years as index, months as columns)
    df_rf_monthly : pd.DataFrame
        Monthly risk-free rates (same structure)
        
    Returns:
    --------
    pd.Series
        Annual returns (excess of risk-free)
    """
    annual_returns = []
    
    for year in df_monthly.index:
        # Get monthly returns for this year
        monthly_rets = df_monthly.loc[year].values
        monthly_rf = df_rf_monthly.loc[year].values
        
        # Remove NaN
        valid_mask = ~(np.isnan(monthly_rets) | np.isnan(monthly_rf))
        monthly_rets_clean = monthly_rets[valid_mask]
        monthly_rf_clean = monthly_rf[valid_mask]
        
        # Compound to annual return
        # (1 + r1/100) * (1 + r2/100) * ... - 1
        annual_ret = np.prod(1 + monthly_rets_clean/100) - 1
        annual_rf = np.prod(1 + monthly_rf_clean/100) - 1
        
        # Store excess return (as percentage)
        annual_returns.append((annual_ret - annual_rf) * 100)
    
    return pd.Series(annual_returns, index=df_monthly.index)



def deflated_sharpe_ratio(sr_hat, T, skew, kurt, N, var_sr):
    """
    Calculate the Deflated Sharpe Ratio (DSR).
    
    Based on Bailey & López de Prado (2014):
    "The Deflated Sharpe Ratio: Correcting for Selection Bias, 
    Backtest Overfitting and Non-Normality"
    
    Parameters:
    -----------
    sr_hat : float
        Estimated Sharpe ratio of the strategy
    T : int
        Number of observations (sample length)
    skew : float
        Skewness of returns
    kurt : float
        Excess kurtosis of returns
    N : int
        Number of independent trials
    var_sr : float
        Variance of Sharpe ratios across all trials
        
    Returns:
    --------
    float
        Deflated Sharpe Ratio (probability that true SR > 0)
    """
    # Euler-Mascheroni constant
    emc = 0.5772156649
    
    # Expected maximum Sharpe ratio under null hypothesis (SR = 0)
    maxZ = (1 - emc) * norm.ppf(1 - 1/N) + emc * norm.ppf(1 - 1/(N * np.e))
    sr0 = np.sqrt(var_sr) * maxZ
    
    # Adjustment for non-normality
    adjustment = 1 - skew * sr_hat + (kurt/4) * (sr_hat ** 2)
    
    # Deflated Sharpe Ratio
    dsr = norm.cdf((sr_hat - sr0) * np.sqrt(T - 1) / np.sqrt(adjustment))
    
    return dsr


def expected_max_sharpe_ratio(N, var_sr, mean_sr=0):
    """
    Calculate the expected maximum Sharpe ratio after N trials.
    
    Parameters:
    -----------
    N : int
        Number of independent trials
    var_sr : float
        Variance of Sharpe ratios across trials
    mean_sr : float
        Mean Sharpe ratio (default 0 for null hypothesis)
        
    Returns:
    --------
    float
        Expected maximum SR
    """
    emc = 0.5772156649
    maxZ = (1 - emc) * norm.ppf(1 - 1/N) + emc * norm.ppf(1 - 1/(N * np.e))
    return mean_sr + np.sqrt(var_sr) * maxZ


def seasonal_ttest(winter_returns, summer_returns):
    """
    Perform t-test comparing winter vs summer returns.
    
    Parameters:
    -----------
    winter_returns : array-like
        Winter season returns
    summer_returns : array-like
        Summer season returns
        
    Returns:
    --------
    dict
        Dictionary with test results
    """
    t_stat, p_value = stats.ttest_ind(winter_returns, summer_returns)
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt((np.var(winter_returns, ddof=1) + 
                          np.var(summer_returns, ddof=1)) / 2)
    cohens_d = (np.mean(winter_returns) - np.mean(summer_returns)) / pooled_std
    
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'significant_05': p_value < 0.05,
        'significant_01': p_value < 0.01,
        'cohens_d': cohens_d,
        'mean_difference': np.mean(winter_returns) - np.mean(summer_returns)
    }
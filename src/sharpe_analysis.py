"""
Sharpe ratio and deflated Sharpe ratio calculations
"""
import numpy as np
from scipy.stats import norm


def calculate_sharpe_ratio(excess_returns, frequency='annual'):
    """
    Calculate Sharpe Ratio.
    
    Parameters:
    -----------
    returns : array-like, pd.Series, or pd.DataFrame
        Returns (in %)
        If frequency='annual', these should be annual excess returns
        If frequency='monthly', these should be monthly excess returns
    frequency : str
        'annual' or 'monthly'
        If 'monthly', result is annualized (× √12)
        
    Returns:
    --------
    float
        Sharpe ratio (annualized)
    """
    
    mean_excess = np.mean(excess_returns)
    std_excess = np.std(excess_returns, ddof=1)
    
    if std_excess == 0:
        return 0.0
    
    sharpe = mean_excess / std_excess
    
    # Annualize if monthly
    if frequency.lower() == 'monthly':
        sharpe = sharpe * np.sqrt(12)
    
    return sharpe


def calculate_sharpe_ratio_by_season(df_monthly, df_rf_monthly, season='winter'):
    """
    Calculate Sharpe ratio for a specific season.
    
    Parameters:
    -----------
    df_monthly : pd.DataFrame
        Monthly returns
    df_rf_monthly : pd.DataFrame
        Monthly risk-free rates
    season : str
        'winter' or 'summer'
        onthly_rets = df_monthly.loc[year].values
        
        # Remove NaN
        valid_mask = ~np.isnan(monthly_rets)
        monthly_rets_clean = monthly_rets[valid_mask]
        
        if len(monthly_rets_clean) == 0:
            annual_returns.append(np.nan)
            continue
        
        # Compound: (1 + r1/100) * (1 + r2/100) * ... - 1
    Returns:
    --------
    float
        Annualized Sharpe ratio for the season
    """
    from returns_calculator import calculate_seasonal_returns
    
    seasonal_excess = calculate_seasonal_returns(df_monthly, season, df_rf_monthly)
    
    if len(seasonal_excess) == 0:
        return 0.0
    
    sharpe_monthly = np.mean(seasonal_excess) / np.std(seasonal_excess, ddof=1)
    return sharpe_monthly * np.sqrt(12)


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
        Use number of ANNUAL observations, not monthly
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
    
    # Prevent negative adjustment
    if adjustment <= 0:
        adjustment = 0.01
    
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
        Expected maximum SR under the null hypothesis
    """
    emc = 0.5772156649
    maxZ = (1 - emc) * norm.ppf(1 - 1/N) + emc * norm.ppf(1 - 1/(N * np.e))
    return mean_sr + np.sqrt(var_sr) * maxZ


def probabilistic_sharpe_ratio(sr_hat, T, sr_benchmark, skew=0, kurt=0):
    """
    Calculate Probabilistic Sharpe Ratio (PSR).
    
    Probability that true SR exceeds a benchmark SR.
    
    Parameters:
    -----------
    sr_hat : float
        Estimated Sharpe ratio
    T : int
        Number of observations
    sr_benchmark : float
        Benchmark Sharpe ratio to compare against
    skew : float
        Skewness (default 0 for normal)
    kurt : float
        Excess kurtosis (default 0 for normal)
        
    Returns:
    --------
    float
        PSR (probability that true SR > benchmark)
    """
    # Adjustment for non-normality
    adjustment = 1 - skew * sr_hat + (kurt/4) * (sr_hat ** 2)
    
    if adjustment <= 0:
        adjustment = 0.01
    
    # PSR calculation
    psr = norm.cdf((sr_hat - sr_benchmark) * np.sqrt(T - 1) / np.sqrt(adjustment))
    
    return psr
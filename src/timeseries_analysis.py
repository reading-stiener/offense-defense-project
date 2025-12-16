import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# ==============================================================================
# STRATEGY CALCULATIONS (PRESERVING TIME STRUCTURE)
# ==============================================================================

def create_datetime_index(df_monthly):
    """
    Create a proper datetime index from year index and month columns.
    
    Parameters:
    -----------
    df_monthly : pd.DataFrame
        DataFrame with year index and month columns
        
    Returns:
    --------
    pd.DatetimeIndex
        Datetime index for the data
    """
    month_map = {
        'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
        'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
    }
    
    dates = []
    for year in df_monthly.index:
        for month in df_monthly.columns:
            if month in month_map:
                dates.append(pd.Timestamp(year=year, month=month_map[month], day=1))
    
    return pd.DatetimeIndex(dates)


def reshape_to_timeseries(df_monthly: pd.DataFrame):
    """
    Reshape wide format (years x months) to time series.
    
    Parameters:
    -----------
    df_monthly : pd.DataFrame
        Wide format with year index and month columns
        
    Returns:
    --------
    pd.Series
        Time series with datetime index
    """
    # Stack to convert to long format
    ts = df_monthly.stack()
    
    # Create proper datetime index
    month_map = {
        'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
        'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
    }
    
    dates = [pd.Timestamp(year=year, month=month_map[month], day=1) 
             for year, month in ts.index]
    
    ts.index = pd.DatetimeIndex(dates)
    ts = ts.sort_index()
    
    return ts


def calculate_sp500_returns_ts(df_sp500, df_rf):
    """
    Calculate S&P 500 excess returns as time series.
    
    Parameters:
    -----------
    df_sp500 : pd.DataFrame
        S&P 500 monthly returns (wide format)
    df_rf : pd.DataFrame
        Risk-free rates monthly (wide format)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns:
        - 'gross': Gross returns
        - 'excess': Excess returns (gross - rf)
        - 'rf': Risk-free rate (for reference)
    """
    # Convert to time series
    sp500_ts = reshape_to_timeseries(df_sp500)
    rf_ts = reshape_to_timeseries(df_rf)

    
    # Convert RF from annual to monthly
    # rf_monthly_ts = ((1 + rf_annual_ts/100) ** (1/12) - 1)*100
    
    # Align indices and calculate excess returns
    sp500_ts, rf_monthly_ts = sp500_ts.align(rf_ts, join='inner')

     # Create output dataframe
    results = pd.DataFrame(index=sp500_ts.index)
    results['rf'] = rf_monthly_ts
    results['gross'] = sp500_ts
    results['excess'] = sp500_ts - rf_monthly_ts
    
    return results


def calculate_smga_returns_ts(df_sp500, df_rf):
    """
    Calculate SMGA strategy returns as time series.
    - Long S&P 500 during Nov-Apr
    - Risk-free rate during May-Oct
    
    Parameters:
    -----------
    df_sp500 : pd.DataFrame
        S&P 500 monthly returns (wide format)
    df_rf : pd.DataFrame
        Risk-free rates monthly (wide format)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns:
        - 'gross': Gross returns
        - 'excess': Excess returns (gross - rf)
        - 'rf': Risk-free rate (for reference)
    """

    # Convert to time series
    sp500_ts = reshape_to_timeseries(df_sp500)
    rf_ts = reshape_to_timeseries(df_rf)
    # rf_monthly_ts = ((1 + rf_annual_ts/100) ** (1/12) - 1)*100
    
    # Align indices
    sp500_ts, rf_monthly_ts = sp500_ts.align(rf_ts, join='inner')
    
    # Create output dataframe
    results = pd.DataFrame(index=sp500_ts.index)
    results['rf'] = rf_monthly_ts
    results['gross'] = np.nan
    results['excess'] = np.nan
    
    for date in sp500_ts.index:
        month = date.month

        if month in [11, 12, 1, 2, 3, 4]:  # Nov-Apr: invested in S&P 500
            results.loc[date, 'gross'] = sp500_ts[date]
            results.loc[date, 'excess'] = sp500_ts[date] - rf_monthly_ts[date]
        else:  # May-Oct: invested in risk-free asset
            results.loc[date, 'gross'] = rf_monthly_ts[date]
            results.loc[date, 'excess'] = 0.0  # RF - RF = 0
    
    return results


def calculate_sector_rotation_returns_ts(df_cyclical, df_defensive, df_rf):
    """
    Calculate Sector Rotation strategy as time series.
    - Long cyclicals during Nov-Apr
    - Long defensives during May-Oct
    
    Parameters:
    -----------
    df_cyclical : pd.DataFrame
        Consumer Discretionary returns (wide format)
    df_defensive : pd.DataFrame
        Consumer Staples returns (wide format)
    df_rf : pd.DataFrame
        Risk-free rates ANNUALIZED (wide format)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns:
        - 'gross': Gross returns
        - 'excess': Excess returns (gross - rf)
        - 'rf': Risk-free rate (for reference)
    """
    # Convert to time series
    cyclical_ts = reshape_to_timeseries(df_cyclical)
    defensive_ts = reshape_to_timeseries(df_defensive)
    rf_ts = reshape_to_timeseries(df_rf)
    
    # Align all series
    combined = pd.DataFrame({
        'cyclical': cyclical_ts,
        'defensive': defensive_ts,
        'rf': rf_ts
    })
    combined = combined.dropna()

    results = pd.DataFrame(index=combined.index)
    results['rf'] = combined['rf']
    results['gross'] = np.nan
    results['excess'] = np.nan
    
    
    for date in combined.index:
        month = date.month
        
        if month in [11, 12, 1, 2, 3, 4]:  # Nov-Apr: Long cyclicals
            results.loc[date, 'gross'] = combined.loc[date, 'cyclical']
            results.loc[date, 'excess'] = combined.loc[date, 'cyclical'] - combined.loc[date, 'rf']
        else:  # May-Oct: Long defensives
            results.loc[date, 'gross'] = combined.loc[date, 'defensive']
            results.loc[date, 'excess'] = combined.loc[date, 'defensive'] - combined.loc[date, 'rf']
    
    
    return results


def calculate_szne_returns_ts(df_discretionary, df_industrials, df_tech, df_materials,
                              df_staples, df_healthcare, df_rf):
    """
    Calculate SZNE (Pacer) strategy returns as time series.
    
    Strategy:
    - Nov-Apr (Favorable): 25% each in Discretionary, Industrials, Tech, Materials (100% cyclical)
    - May-Oct (Unfavorable): 50% Staples, 50% Healthcare (100% defensive)
    
    Parameters:
    -----------
    df_discretionary : pd.DataFrame
        Consumer Discretionary returns (wide format)
    df_industrials : pd.DataFrame
        Industrials returns (wide format)
    df_tech : pd.DataFrame
        Information Technology returns (wide format)
    df_materials : pd.DataFrame
        Materials returns (wide format)
    df_staples : pd.DataFrame
        Consumer Staples returns (wide format)
    df_healthcare : pd.DataFrame
        Healthcare returns (wide format)
    df_rf : pd.DataFrame
        Risk-free rates ANNUALIZED (wide format)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns:
        - 'gross': Gross returns
        - 'excess': Excess returns (gross - rf)
        - 'rf': Risk-free rate (for reference)
    """
    # Convert to time series
    discretionary_ts = reshape_to_timeseries(df_discretionary)
    industrials_ts = reshape_to_timeseries(df_industrials)
    tech_ts = reshape_to_timeseries(df_tech)
    materials_ts = reshape_to_timeseries(df_materials)
    staples_ts = reshape_to_timeseries(df_staples)
    healthcare_ts = reshape_to_timeseries(df_healthcare)
    rf_ts = reshape_to_timeseries(df_rf)
    
    # Align all series
    combined = pd.DataFrame({
        'discretionary': discretionary_ts,
        'industrials': industrials_ts,
        'tech': tech_ts,
        'materials': materials_ts,
        'staples': staples_ts,
        'healthcare': healthcare_ts,
        'rf': rf_ts
    })
    combined = combined.dropna()
    
    # Create results DataFrame
    results = pd.DataFrame(index=combined.index)
    results['rf'] = combined['rf']
    results['gross'] = np.nan
    results['excess'] = np.nan
    
    # Calculate strategy returns for each date
    for date in combined.index:
        month = date.month
        
        if month in [11, 12, 1, 2, 3, 4]:  # Nov-Apr: Cyclical portfolio
            # Equal weight: 25% each in 4 cyclical sectors
            portfolio_return = (
                0.25 * combined.loc[date, 'discretionary'] +
                0.25 * combined.loc[date, 'industrials'] +
                0.25 * combined.loc[date, 'tech'] +
                0.25 * combined.loc[date, 'materials']
            )
            results.loc[date, 'gross'] = portfolio_return
            results.loc[date, 'excess'] = portfolio_return - combined.loc[date, 'rf']
            
        else:  # May-Oct: Defensive portfolio
            # Equal weight: 50% each in 2 defensive sectors
            portfolio_return = (
                0.50 * combined.loc[date, 'staples'] +
                0.50 * combined.loc[date, 'healthcare']
            )
            results.loc[date, 'gross'] = portfolio_return
            results.loc[date, 'excess'] = portfolio_return - combined.loc[date, 'rf']
    
    return results


def calculate_modified_szne_returns_ts(
    # Offense sectors
    df_discretionary, df_industrials, df_tech, df_materials,
    df_communication, df_financials, df_energy,
    # Defense sectors
    df_staples, df_healthcare, df_utilities, df_realestate,
    # Market data
    df_rf,
    df_interest_rates,  # NEW: Interest rate time series
    # Rate thresholds
    high_rate_threshold=4,  # 4% annual
    low_rate_threshold=2,   # 2% annual
    # Configuration
    include_realestate=False,
     # Timing parameters
    offense_start_month=10,  # Mid-October
    offense_start_day=15,
    defense_start_month=4,   # Mid-April
    defense_start_day=15,
):
    """
    Calculate SZNE returns with interest rate regime adjustments.
    
    Interest Rate Logic:
    - HIGH RATES (>4%): Favor Financials, Energy (benefit from rates)
    - LOW RATES (<2%): Favor Tech, Communication (growth sectors)
    - NORMAL RATES (2-4%): Standard equal weighting
    
    Parameters:
    -----------
    df_interest_rates : pd.DataFrame
        Interest rate time series (wide format)
        Can be Fed Funds Rate, 10Y Treasury, or your T-Bill rate
    high_rate_threshold : float
        Threshold for "high rate" regime (annualized, as decimal)
    low_rate_threshold : float
        Threshold for "low rate" regime (annualized, as decimal)
    """
    # Convert all to time series
    discretionary_ts = reshape_to_timeseries(df_discretionary)
    industrials_ts = reshape_to_timeseries(df_industrials)
    tech_ts = reshape_to_timeseries(df_tech)
    materials_ts = reshape_to_timeseries(df_materials)
    communication_ts = reshape_to_timeseries(df_communication)
    financials_ts = reshape_to_timeseries(df_financials)
    energy_ts = reshape_to_timeseries(df_energy)
    
    staples_ts = reshape_to_timeseries(df_staples)
    healthcare_ts = reshape_to_timeseries(df_healthcare)
    utilities_ts = reshape_to_timeseries(df_utilities)
    realestate_ts = reshape_to_timeseries(df_realestate)
    
    rf_ts = reshape_to_timeseries(df_rf)
    
    interest_rates_ts = reshape_to_timeseries(df_interest_rates)
    
    # Combine all series
    combined = pd.DataFrame({
        'discretionary': discretionary_ts,
        'industrials': industrials_ts,
        'tech': tech_ts,
        'materials': materials_ts,
        'communication': communication_ts,
        'financials': financials_ts,
        'energy': energy_ts,
        'staples': staples_ts,
        'healthcare': healthcare_ts,
        'utilities': utilities_ts,
        'realestate': realestate_ts,
        'rf': rf_ts,
        'interest_rate': interest_rates_ts
    })
    combined = combined.dropna()
    
    # Create results DataFrame
    results = pd.DataFrame(index=combined.index)
    results['rf'] = combined['rf']
    results['interest_rate'] = combined['interest_rate']
    results['rate_regime'] = 'Normal'  # Track which regime we're in
    results['gross'] = np.nan
    results['excess'] = np.nan
    
    # Calculate returns for each date
    for date in combined.index:
        month = date.month
        current_rate = combined.loc[date, 'interest_rate']
        
        # Determine rate regime
        if current_rate >= high_rate_threshold:
            rate_regime = 'High'
        elif current_rate <= low_rate_threshold:
            rate_regime = 'Low'
        else:
            rate_regime = 'Normal'
        
        results.loc[date, 'rate_regime'] = rate_regime

        # Determine if we're in offense or defense period
        # Offense: Mid-Oct through Mid-Apr
        # Defense: Mid-Apr through Mid-Oct
        
        if date.month > offense_start_month or date.month < defense_start_month:
            # Clearly in offense season
            in_offense = True
        elif date.month > defense_start_month and date.month < offense_start_month:
            # Clearly in defense season
            in_offense = False
        elif date.month == offense_start_month:
            # October - check day
            in_offense = date.day >= offense_start_day
        elif date.month == defense_start_month:
            # April - check day
            in_offense = date.day < defense_start_day
        else:
            # Shouldn't reach here, but default to standard months
            in_offense = date.month in [11, 12, 1, 2, 3, 4]
        
        # OFFENSE PERIOD (Nov-Apr)
        if in_offense:
            
            # if rate_regime == 'High':
            #     # HIGH RATES: Overweight Financials (benefit from rates), add Energy
            #     offense_weights = {
            #         'discretionary': 0.15,      # Keep reasonable - consumer spending
            #         'industrials': 0.20,        # OVERWEIGHT - capex benefits
            #         'tech': 0.15,               # Reduce but don't kill (Mag 7 too important)
            #         'materials': 0.20,          # OVERWEIGHT - commodities/inflation hedge
            #         'communication': 0.10,      # Underweight - rate sensitive
            #         'financials': 0.20,         # OVERWEIGHT - net interest margin benefits
            #         'energy': 0.00              # REMOVE - too volatile, doesn't help
            #     }
            
            if rate_regime == 'Low' or rate_regime == 'High':
                # LOW RATES: Overweight Tech, Communication (growth benefits)
                offense_weights = {
                    'discretionary': 0.20,      # OVERWEIGHT - consumer confidence high
                    'industrials': 0.15,        # Moderate weight
                    'tech': 0.30,               # OVERWEIGHT - growth premium expands
                    'materials': 0.10,          # Underweight - less inflation pressure
                    'communication': 0.20,      # OVERWEIGHT - growth sector
                    'financials': 0.05,         # Underweight - NIM compression
                    'energy': 0.00              # REMOVE - doesn't fit low-rate regime
                }
            
            else:  # Normal rates
                # NORMAL: Equal weight across 6 main sectors (no Energy)
                offense_weights = {
                    'discretionary': 1/6,
                    'industrials': 1/6,
                    'tech': 1/6,
                    'materials': 1/6,
                    'communication': 1/6,
                    'financials': 1/6,
                    'energy': 0.0
                }
            
            portfolio_return = (
                offense_weights['discretionary'] * combined.loc[date, 'discretionary'] +
                offense_weights['industrials'] * combined.loc[date, 'industrials'] +
                offense_weights['tech'] * combined.loc[date, 'tech'] +
                offense_weights['materials'] * combined.loc[date, 'materials'] +
                offense_weights['communication'] * combined.loc[date, 'communication'] +
                offense_weights['financials'] * combined.loc[date, 'financials'] +
                offense_weights['energy'] * combined.loc[date, 'energy']
            )
        
        # DEFENSE PERIOD (May-Oct)
        else:
            
            # if rate_regime == 'High':
            #     # HIGH RATES: Favor Utilities (income), reduce Real Estate (hurt by rates)
            #     if include_realestate:
            #         defense_weights = {
            #             'staples': 0.30,
            #             'healthcare': 0.30,
            #             'utilities': 0.35,      # OVERWEIGHT - income benefits
            #             'realestate': 0.05      # Underweight - hurt by high rates
            #         }
            #     else:
            #         defense_weights = {
            #             'staples': 1/3,
            #             'healthcare': 1/3,
            #             'utilities': 1/3,
            #             'realestate': 0.0
            #         }
            
            if rate_regime == 'Low' or rate_regime == 'High':
                # LOW RATES: Can increase Real Estate (benefits from low rates)
                if include_realestate:
                    defense_weights = {
                        'staples': 0.25,
                        'healthcare': 0.25,
                        'utilities': 0.20,      # Slightly underweight
                        'realestate': 0.30      # OVERWEIGHT - benefits from low rates
                    }
                else:
                    defense_weights = {
                        'staples': 1/3,
                        'healthcare': 1/3,
                        'utilities': 1/3,
                        'realestate': 0.0
                    }
            
            else:  # Normal rates
                # NORMAL: Equal weight
                if include_realestate:
                    defense_weights = {
                        'staples': 0.25,
                        'healthcare': 0.25,
                        'utilities': 0.25,
                        'realestate': 0.25
                    }
                else:
                    defense_weights = {
                        'staples': 1/3,
                        'healthcare': 1/3,
                        'utilities': 1/3,
                        'realestate': 0.0
                    }
            
            portfolio_return = (
                defense_weights['staples'] * combined.loc[date, 'staples'] +
                defense_weights['healthcare'] * combined.loc[date, 'healthcare'] +
                defense_weights['utilities'] * combined.loc[date, 'utilities'] +
                defense_weights['realestate'] * combined.loc[date, 'realestate']
            )
        
        results.loc[date, 'gross'] = portfolio_return
        results.loc[date, 'excess'] = portfolio_return - combined.loc[date, 'rf']
    
    return results



def calculate_long_short_returns_ts(df_cyclical, df_defensive, df_rf):
    """
    Calculate Long-Short strategy as time series.
    - Nov-Apr: Long cyclicals, Short defensives
    - May-Oct: Long defensives, Short cyclicals
    
    Parameters:
    -----------
    df_cyclical : pd.DataFrame
        Consumer Discretionary returns (wide format)
    df_defensive : pd.DataFrame
        Consumer Staples returns (wide format)
    df_rf : pd.DataFrame
        Risk-free rates ANNUALIZED (wide format)
        
    Returns:
    --------
    pd.Series
        Time series of excess returns with datetime index
    """
    # Convert to time series
    cyclical_ts = reshape_to_timeseries(df_cyclical)
    defensive_ts = reshape_to_timeseries(df_defensive)
    rf_ts = reshape_to_timeseries(df_rf)
    # rf_monthly_ts = (1 + rf_annual_ts) ** (1/12) - 1
    
    # Align all series
    combined = pd.DataFrame({
        'cyclical': cyclical_ts,
        'defensive': defensive_ts,
        'rf': rf_ts
    })
    combined = combined.dropna()
    
    results = pd.DataFrame(index=combined.index)
    results['rf'] = combined['rf']
    results['gross'] = np.nan
    results['excess'] = np.nan
    
    for date in combined.index:
        month = date.month
        
        if month in [11, 12, 1, 2, 3, 4]:  # Nov-Apr: Long cyc, Short def
            long_short_return = combined.loc[date, 'cyclical'] - combined.loc[date, 'defensive']
            results.loc[date, 'gross'] = long_short_return
            results.loc[date, 'excess'] = long_short_return - combined.loc[date, 'rf']
        else:  # May-Oct: Long def, Short cyc
            long_short_return = combined.loc[date, 'defensive'] - combined.loc[date, 'cyclical']
            results.loc[date, 'gross'] = long_short_return
            results.loc[date, 'excess'] = long_short_return - combined.loc[date, 'rf']

    return results


# ==============================================================================
# PERIOD FILTERING
# ==============================================================================

def filter_period(returns_ts, start_year=None, end_year=None, exclude_years=None):
    """
    Filter time series to specific period.
    
    Parameters:
    -----------
    returns_ts : pd.Series
        Time series with datetime index
    start_year : int, optional
        Start year (inclusive)
    end_year : int, optional
        End year (inclusive)
    exclude_years : list, optional
        Years to exclude (e.g., [2008])
        
    Returns:
    --------
    pd.Series
        Filtered time series
    """
    filtered = returns_ts.copy()
    
    if start_year is not None:
        filtered = filtered[filtered.index.year >= start_year]
    
    if end_year is not None:
        filtered = filtered[filtered.index.year <= end_year]
    
    if exclude_years is not None:
        for year in exclude_years:
            filtered = filtered[filtered.index.year != year]
    
    return filtered


# ==============================================================================
# SUMMARY STATISTICS
# ==============================================================================

def calculate_statistics_ts(returns_df, strategy_name):
    """
    Calculate summary statistics for a return series.
    
    Parameters:
    -----------
    returns_df : pd.DataFrame
        DataFrame with 'gross', 'excess', 'rf' columns
    strategy_name : str
        Name of strategy
        
    Returns:
    --------
    dict
        Dictionary of statistics
    """
    gross_returns = returns_df['gross'].dropna()
    excess_returns = returns_df['excess'].dropna()
    rf_returns = returns_df['rf'].dropna()
    
    mean_monthly = gross_returns.mean()
    median_monthly = gross_returns.median()
    std_monthly = gross_returns.std()
    
    # Sharpe ratio using excess returns
    if std_monthly != 0 and not np.isnan(std_monthly):
        sharpe = (excess_returns.mean() / std_monthly) * np.sqrt(12)
    else:
        sharpe = 0
    
    # Sortino ratio
    downside_returns = gross_returns[gross_returns < 0]
    downside_std = downside_returns.std()
    sortino = (excess_returns.mean() / downside_std) * np.sqrt(12) if downside_std != 0 and not np.isnan(downside_std) else 0
    
    # # P-value (t-test against zero for excess returns)
    # if len(excess_returns) > 1:
    #     t_stat, p_value = stats.ttest_1samp(excess_returns, 0)
    # else:
    #     p_value = np.nan
    
    return {
        'Strategy': strategy_name,
        'Mean (%)': mean_monthly,
        'Median (%)': median_monthly,
        'Std Dev (%)': std_monthly,
        'Sharpe': sharpe,
        'Sortino': sortino,
        #'P-value': p_value,
        'N': len(gross_returns),
        'Start': gross_returns.index.min().strftime('%Y-%m'),
        'End': gross_returns.index.max().strftime('%Y-%m')
    }


def create_summary_table_ts(strategies_dict, period_name="Full Period"):
    """
    Create summary table for all strategies.
    
    Parameters:
    -----------
    strategies_dict : dict
        Dictionary with strategy names as keys and time series as values
    period_name : str
        Name of the period
        
    Returns:
    --------
    pd.DataFrame
        Summary statistics table
    """
    stats_list = []
    
    for strategy_name, returns_ts in strategies_dict.items():
        stats = calculate_statistics_ts(returns_ts, strategy_name)
        stats_list.append(stats)
    
    summary_df = pd.DataFrame(stats_list)
    summary_df = summary_df.set_index('Strategy')
    
    print(f"\n{'='*80}")
    print(f"{period_name}")
    print(f"{'='*80}")
    print(summary_df.to_string())
    
    return summary_df


# ==============================================================================
# VISUALIZATION
# ==============================================================================



def plot_cumulative_returns(strategies_dict, title="Cumulative Returns"):
    """
    Plot cumulative returns for all strategies.
    
    Parameters:
    -----------
    strategies_dict : dict
        Dictionary with strategy names as keys and return time series as values
    title : str
        Plot title
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    
    for strategy_name, returns_ts in strategies_dict.items():
        # Calculate cumulative returns
        cum_returns = 1000 * (1 + returns_ts/100).cumprod()
        
        # Plot
        ax.plot(cum_returns.index, cum_returns.values, label=strategy_name, linewidth=2)
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Cumulative Return', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_cumulative_returns(strategies_dict, title="Cumulative Returns", 
                           rf_data=None, high_rate_threshold=4):
    """
    Plot cumulative returns for all strategies with optional interest rate shading.
    
    Parameters:
    -----------
    strategies_dict : dict
        Dictionary with strategy names as keys and DataFrames as values
        Each DataFrame should have 'gross' and 'rf' columns
    title : str
        Plot title
    rf_data : pd.Series, optional
        Time series of risk-free rates (monthly). Will shade high rate periods.
    high_rate_threshold : float
        Threshold for defining "high" rates (annualized, as decimal). Default 0.04 (4%)
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Plot cumulative returns for each strategy
    for strategy_name, returns_df in strategies_dict.items():
        # Calculate cumulative returns from gross column
        cum_returns = 1000 * (1 + returns_df/100).cumprod()
        
        # Plot
        ax.plot(cum_returns.index, cum_returns.values, label=strategy_name, linewidth=2)
    
    # Overlay interest rate periods if provided
    if rf_data is not None:
        # align the data series
        

        # Annualize the monthly rates
        rf_annual = ((1 + rf_data/100) ** 12 - 1)*100

        fig, ax1 = plt.subplots(figsize=(10, 5))
        rf_annual.plot(ax=ax1, label="Annualized Risk-Free Rate", color="#1f77b4")
        ax1.set_title("Annualized Risk-Free Rate Over Time")
        ax1.set_ylabel("Rate")
        ax1.grid(True, alpha=0.3)
        ax1.legend(frameon=False)
     
        # Create boolean mask for high rates
        high_rate_mask = rf_annual >= high_rate_threshold
        
        # Find contiguous regions using diff
        # Where mask changes from False to True (start of high period)
        # Where mask changes from True to False (end of high period)
        mask_diff = high_rate_mask.astype(int).diff()
        
        starts = high_rate_mask.index[mask_diff == 1]  # Transitions to high
        ends = high_rate_mask.index[mask_diff == -1]   # Transitions to low
        
        # Handle edge cases
        if high_rate_mask.iloc[0]:
            starts = starts.insert(0, high_rate_mask.index[0])
        if high_rate_mask.iloc[-1]:
            ends = ends.insert(len(ends), high_rate_mask.index[-1])
        
        # Shade all high rate periods
        for start, end in zip(starts, ends):
            ax.axvspan(start, end, alpha=0.2, color='red', 
                      label='High Rate (>4%)' if start == starts[0] else '')
        
        # Add threshold info
        ax.text(0.02, 0.98, f'High Rate Threshold: {high_rate_threshold}%',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
                fontsize=9)
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Cumulative Return (Growth of $1000)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_rolling_sharpe(strategies_dict, window=36, title="Rolling 3-Year Sharpe Ratio"):
    """
    Plot rolling Sharpe ratios for all strategies.
    
    Parameters:
    -----------
    strategies_dict : dict
        Dictionary with strategy names as keys and return time series as values
    window : int
        Rolling window size in months (default 36 = 3 years)
    title : str
        Plot title
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    
    for strategy_name, returns_ts in strategies_dict.items():
        # Calculate rolling Sharpe
        rolling_mean = returns_ts.rolling(window=window).mean()
        rolling_std = returns_ts.rolling(window=window).std()
        rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(12)
        
        # Plot
        ax.plot(rolling_sharpe.index, rolling_sharpe.values, label=strategy_name, linewidth=2)
    
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Sharpe Ratio', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_drawdowns(strategies_dict, title="Drawdowns"):
    """
    Plot drawdowns for all strategies.
    
    Parameters:
    -----------
    strategies_dict : dict
        Dictionary with strategy names as keys and return time series as values
    title : str
        Plot title
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    
    for strategy_name, returns_ts in strategies_dict.items():
        # Calculate cumulative returns
        cum_returns = (1 + returns_ts).cumprod()
        
        # Calculate running maximum
        running_max = cum_returns.expanding().max()
        
        # Calculate drawdown
        drawdown = (cum_returns - running_max) / running_max
        
        # Plot
        ax.plot(drawdown.index, drawdown.values * 100, label=strategy_name, linewidth=2)
    
    ax.fill_between(ax.get_xlim(), 0, -100, alpha=0.1, color='red')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Drawdown (%)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_monthly_returns_heatmap(returns_ts, title="Monthly Returns Heatmap"):
    """
    Create heatmap of monthly returns by year.
    
    Parameters:
    -----------
    returns_ts : pd.Series
        Time series of monthly returns
    title : str
        Plot title
    """
    # Ensure numeric dtype and drop non-numeric/NaN entries to avoid object arrays in imshow
    returns_ts = pd.to_numeric(returns_ts, errors='coerce').dropna()

    # Reshape to year x month format
    returns_pivot = returns_ts.to_frame('return')
    returns_pivot['year'] = returns_pivot.index.year
    returns_pivot['month'] = returns_pivot.index.month
    
    heatmap_data = returns_pivot.pivot(index='year', columns='month', values='return') * 100
    
    # Create month labels
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    im = ax.imshow(heatmap_data.values, cmap='RdYlGn', aspect='auto', vmin=-10, vmax=10)
    
    # Set ticks
    ax.set_xticks(np.arange(len(month_labels)))
    ax.set_yticks(np.arange(len(heatmap_data.index)))
    ax.set_xticklabels(month_labels)
    ax.set_yticklabels(heatmap_data.index)
    
    # Rotate month labels
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Monthly Return (%)', rotation=270, labelpad=20)
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Year', fontsize=12)
    
    plt.tight_layout()
    plt.show()


# ==============================================================================
# MAIN ANALYSIS
# ==============================================================================

def main_analysis_ts(df_sp500, df_discretionary, df_staples, df_tbills):
    """
    Run complete analysis with time series preservation.
    
    Parameters:
    -----------
    df_sp500 : pd.DataFrame
        S&P 500 monthly returns
    df_discretionary : pd.DataFrame
        Consumer Discretionary monthly returns
    df_staples : pd.DataFrame
        Consumer Staples monthly returns
    df_tbills : pd.DataFrame
        T-Bills (annualized rates)
        
    Returns:
    --------
    dict
        Dictionary of strategy time series
    """
    
    print("=" * 80)
    print("REPLICATING: An Update on Sector Rotation in SMGA Strategy")
    print("WITH TIME SERIES PRESERVATION")
    print("=" * 80)
    
    # Calculate returns for each strategy
    print("\nCalculating strategy returns...")
    
    sp500_ts = calculate_sp500_returns_ts(df_sp500, df_tbills)
    smga_ts = calculate_smga_returns_ts(df_sp500, df_tbills)
    sector_rotation_ts = calculate_sector_rotation_returns_ts(
        df_discretionary, df_staples, df_tbills
    )
    long_short_ts = calculate_long_short_returns_ts(
        df_discretionary, df_staples, df_tbills
    )
    
    # Create dictionary of strategies
    strategies_full = {
        'S&P 500': sp500_ts,
        'SMGA': smga_ts,
        'Sector Rotation': sector_rotation_ts,
        'Long-Short': long_short_ts
    }
    
    # Full period analysis
    summary_full = create_summary_table_ts(strategies_full, "FULL PERIOD")
    
    # Period 1: 1993-2007
    strategies_p1 = {
        name: filter_period(ts, start_year=1993, end_year=2007)
        for name, ts in strategies_full.items()
    }
    summary_p1 = create_summary_table_ts(strategies_p1, "PERIOD 1: 1993-2007")
    
    # Period 2: 2009-2023 (excluding 2008)
    strategies_p2 = {
        name: filter_period(ts, start_year=2009, end_year=2023)
        for name, ts in strategies_full.items()
    }
    summary_p2 = create_summary_table_ts(strategies_p2, "PERIOD 2: 2009-2023 (Excluding 2008)")
    
    # Visualizations
    print("\nGenerating visualizations...")
    plot_cumulative_returns(strategies_full, "Cumulative Returns (Full Period)")
    plot_rolling_sharpe(strategies_full)
    plot_drawdowns(strategies_full)
    
    # Individual strategy heatmaps
    plot_monthly_returns_heatmap(smga_ts, "SMGA Strategy: Monthly Returns by Year")
    
    return strategies_full, {
        'full': summary_full,
        'period1': summary_p1,
        'period2': summary_p2
    }


# ==============================================================================
# USAGE EXAMPLE
# ==============================================================================

if __name__ == "__main__":
    # Load your data
    # df_sp500 = pd.read_csv('sp500_monthly.csv', index_col=0)
    # df_discretionary = pd.read_csv('discretionary_monthly.csv', index_col=0)
    # df_staples = pd.read_csv('staples_monthly.csv', index_col=0)
    # df_tbills = pd.read_csv('tbills_monthly.csv', index_col=0)
    
    # Run analysis
    # strategies_ts, summaries = main_analysis_ts(df_sp500, df_discretionary, df_staples, df_tbills)
    
    # Access specific strategy time series
    # smga_returns = strategies_ts['SMGA']
    
    # Filter to specific period
    # smga_2010s = filter_period(smga_returns, start_year=2010, end_year=2019)
    
    pass

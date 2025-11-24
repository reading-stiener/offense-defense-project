# ===============================================================================
# FAMA-FRENCH REGRESSION ANALYSIS
# ===============================================================================

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns


def align_strategy_with_factors(strategy_returns, ff_factors):
    """
    Align strategy returns with Fama-French factors.
    
    Parameters:
    -----------
    strategy_returns : pd.Series
        Time series of strategy excess returns
    ff_factors : pd.DataFrame
        Fama-French factors with datetime index
        
    Returns:
    --------
    pd.DataFrame
        Aligned data with strategy returns and factors
    """
    # Ensure numeric and consistent column names
    rename_map = {'rm_rf': 'Mkt-RF', 'smb': 'SMB', 'hml': 'HML', 'rf': 'RF'}
    ff_clean = (ff_factors.rename(columns=rename_map)
                           .apply(pd.to_numeric, errors='coerce'))

    strategy_clean = pd.to_numeric(strategy_returns, errors='coerce')

    # Combine on index, drop rows with missing values
    data = pd.concat([strategy_clean.rename('Strategy'), ff_clean], axis=1).dropna()
    
    return data


def run_fama_french_regression(strategy_returns, ff_factors, model_type='3factor'):
    """
    Run Fama-French regression analysis.
    
    Parameters:
    -----------
    strategy_returns : pd.Series
        Time series of strategy excess returns (already minus risk-free rate)
    ff_factors : pd.DataFrame
        Fama-French factors with columns: Mkt-RF, SMB, HML, (RMW, CMA for 5-factor)
    model_type : str
        '3factor' for FF3 (Market, SMB, HML) or '5factor' for FF5
        
    Returns:
    --------
    dict
        Dictionary containing:
        - model: statsmodels regression results object
        - alpha: monthly alpha (intercept)
        - alpha_annual: annualized alpha
        - beta_market: market beta
        - beta_smb: SMB loading
        - beta_hml: HML loading
        - t_stats: t-statistics for all coefficients
        - p_values: p-values for all coefficients
        - r_squared: R-squared
        - adj_r_squared: Adjusted R-squared
    """
    # Align data
    data = align_strategy_with_factors(strategy_returns, ff_factors)

    
    # Dependent variable (strategy excess returns)
    y = data['Strategy']
    
    # Independent variables
    if model_type == '3factor':
        factor_cols = ['Mkt-RF', 'SMB', 'HML']
    elif model_type == '5factor':
        factor_cols = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
    else:
        raise ValueError("model_type must be '3factor' or '5factor'")
    
    # Check if factors exist
    available_factors = [col for col in factor_cols if col in data.columns]
    if len(available_factors) != len(factor_cols):
        missing = set(factor_cols) - set(available_factors)
        raise ValueError(f"Missing factors: {missing}")
    
    X = data[factor_cols]
    
    # Add constant (this is alpha)
    X = sm.add_constant(X)
    
    # Run OLS regression
    model = sm.OLS(y, X).fit()
    
    # Extract results
    results = {
        'model': model,
        'alpha': model.params['const'],
        'alpha_annual': model.params['const'] * 12,
        'beta_market': model.params['Mkt-RF'],
        'beta_smb': model.params['SMB'] if 'SMB' in model.params else None,
        'beta_hml': model.params['HML'] if 'HML' in model.params else None,
        't_stats': model.tvalues.to_dict(),
        'p_values': model.pvalues.to_dict(),
        'r_squared': model.rsquared,
        'adj_r_squared': model.rsquared_adj,
        'n_obs': int(model.nobs)
    }
    
    # Add 5-factor loadings if applicable
    if model_type == '5factor':
        results['beta_rmw'] = model.params.get('RMW', None)
        results['beta_cma'] = model.params.get('CMA', None)
    
    return results


def create_regression_summary_table(regression_results_dict):
    """
    Create a summary table from multiple regression results.
    
    Parameters:
    -----------
    regression_results_dict : dict
        Dictionary with strategy names as keys and regression results as values
        
    Returns:
    --------
    pd.DataFrame
        Summary table with key statistics
    """
    summary_data = []
    
    for strategy_name, results in regression_results_dict.items():
        row = {
            'Strategy': strategy_name,
            'Alpha (monthly %)': results['alpha'],
            'Alpha (annual %)': results['alpha_annual'],
            'Alpha p-value': results['p_values']['const'],
            'Beta (Market)': results['beta_market'],
            'Beta (SMB)': results['beta_smb'],
            'Beta (HML)': results['beta_hml'],
            'R²': results['r_squared'],
            'Adj R²': results['adj_r_squared'],
            'N': results['n_obs']
        }
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data).set_index('Strategy')
    
    return summary_df


def plot_regression_diagnostics(strategy_returns, regression_results, strategy_name):
    """
    Create diagnostic plots for regression analysis.
    
    Parameters:
    -----------
    strategy_returns : pd.Series
        Strategy excess returns
    regression_results : dict
        Results from run_fama_french_regression()
    strategy_name : str
        Name of strategy for plot titles
    """
    model = regression_results['model']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Fitted vs Actual
    ax = axes[0, 0]
    ax.scatter(model.fittedvalues, model.fittedvalues + model.resid, alpha=0.6)
    ax.axhline(y=0, color='r', linestyle='--', linewidth=1)
    ax.set_xlabel('Fitted Values', fontsize=10)
    ax.set_ylabel('Actual Returns', fontsize=10)
    ax.set_title(f'{strategy_name}: Fitted vs Actual', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add R-squared
    ax.text(0.05, 0.95, f'R² = {model.rsquared:.4f}', 
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 2. Residuals vs Fitted
    ax = axes[0, 1]
    ax.scatter(model.fittedvalues, model.resid, alpha=0.6)
    ax.axhline(y=0, color='r', linestyle='--', linewidth=1)
    ax.set_xlabel('Fitted Values', fontsize=10)
    ax.set_ylabel('Residuals', fontsize=10)
    ax.set_title(f'{strategy_name}: Residual Plot', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 3. Q-Q Plot
    ax = axes[1, 0]
    sm.graphics.qqplot(model.resid, line='45', ax=ax, alpha=0.6)
    ax.set_title(f'{strategy_name}: Q-Q Plot', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 4. Residuals histogram
    ax = axes[1, 1]
    ax.hist(model.resid, bins=30, alpha=0.7, edgecolor='black')
    ax.axvline(x=0, color='r', linestyle='--', linewidth=1)
    ax.set_xlabel('Residuals', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.set_title(f'{strategy_name}: Residual Distribution', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()


def plot_factor_loadings(regression_results_dict):
    """
    Plot factor loadings (betas) across multiple strategies.
    
    Parameters:
    -----------
    regression_results_dict : dict
        Dictionary with strategy names as keys and regression results as values
    """
    # Extract factor loadings
    strategies = list(regression_results_dict.keys())
    market_betas = [regression_results_dict[s]['beta_market'] for s in strategies]
    smb_betas = [regression_results_dict[s]['beta_smb'] for s in strategies]
    hml_betas = [regression_results_dict[s]['beta_hml'] for s in strategies]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(strategies))
    width = 0.25
    
    ax.bar(x - width, market_betas, width, label='Market', alpha=0.8)
    ax.bar(x, smb_betas, width, label='SMB (Size)', alpha=0.8)
    ax.bar(x + width, hml_betas, width, label='HML (Value)', alpha=0.8)
    
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.axhline(y=1, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    
    ax.set_xlabel('Strategy', fontsize=12)
    ax.set_ylabel('Factor Loading (Beta)', fontsize=12)
    ax.set_title('Factor Loadings Across Strategies', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()


def plot_alpha_comparison(regression_results_dict):
    """
    Plot alphas (risk-adjusted excess returns) across strategies.
    
    Parameters:
    -----------
    regression_results_dict : dict
        Dictionary with strategy names as keys and regression results as values
    """
    # Extract alphas and p-values
    strategies = list(regression_results_dict.keys())
    alphas = [regression_results_dict[s]['alpha_annual'] * 100 for s in strategies]
    p_values = [regression_results_dict[s]['p_values']['const'] for s in strategies]
    
    # Color by significance
    colors = ['green' if p < 0.05 else 'orange' if p < 0.10 else 'red' 
              for p in p_values]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars = ax.bar(strategies, alphas, color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(y=0, color='black', linewidth=1)
    
    # Add significance stars
    for i, (alpha, p_val) in enumerate(zip(alphas, p_values)):
        if p_val < 0.01:
            stars = '***'
        elif p_val < 0.05:
            stars = '**'
        elif p_val < 0.10:
            stars = '*'
        else:
            stars = ''
        
        if stars:
            y_pos = alpha + (0.5 if alpha > 0 else -0.5)
            ax.text(i, y_pos, stars, ha='center', fontsize=14, fontweight='bold')
    
    ax.set_xlabel('Strategy', fontsize=12)
    ax.set_ylabel('Annual Alpha (%)', fontsize=12)
    ax.set_title('Risk-Adjusted Excess Returns (Alpha) by Strategy', 
                 fontsize=14, fontweight='bold')
    ax.set_xticklabels(strategies, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add legend for significance
    from matplotlib.patches import Rectangle
    legend_elements = [
        Rectangle((0, 0), 1, 1, fc='green', alpha=0.7, label='p < 0.05 **'),
        Rectangle((0, 0), 1, 1, fc='orange', alpha=0.7, label='p < 0.10 *'),
        Rectangle((0, 0), 1, 1, fc='red', alpha=0.7, label='p ≥ 0.10 (not sig.)')
    ]
    ax.legend(handles=legend_elements, loc='best')
    
    plt.tight_layout()
    plt.show()


def plot_rolling_alpha(strategy_returns, ff_factors, window=36, strategy_name='Strategy'):
    """
    Plot rolling alpha over time.
    
    Parameters:
    -----------
    strategy_returns : pd.Series
        Strategy excess returns
    ff_factors : pd.DataFrame
        Fama-French factors
    window : int
        Rolling window in months (default 36 = 3 years)
    strategy_name : str
        Name for plot title
    """
    # Align data
    data = align_strategy_with_factors(strategy_returns, ff_factors)
    
    # Calculate rolling alpha
    rolling_alphas = []
    dates = []
    
    for i in range(window, len(data)):
        # Get window of data
        window_data = data.iloc[i-window:i]
        
        y = window_data['Strategy']
        X = window_data[['Mkt-RF', 'SMB', 'HML']]
        X = sm.add_constant(X)
        
        # Run regression
        model = sm.OLS(y, X).fit()
        
        # Store alpha (annualized)
        rolling_alphas.append(model.params['const'] * 12)
        dates.append(data.index[i])
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(dates, rolling_alphas, linewidth=2, color='blue')
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1)
    ax.fill_between(dates, 0, rolling_alphas, alpha=0.3)
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Annual Alpha (%)', fontsize=12)
    ax.set_title(f'{strategy_name}: Rolling {window}-Month Alpha', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def print_regression_summary(results, strategy_name):
    """
    Print formatted regression results.
    
    Parameters:
    -----------
    results : dict
        Results from run_fama_french_regression()
    strategy_name : str
        Name of strategy
    """
    print("=" * 80)
    print(f"FAMA-FRENCH REGRESSION RESULTS: {strategy_name}")
    print("=" * 80)
    print(f"\nAlpha (Monthly):     {results['alpha']:>8.4f}%  (p={results['p_values']['const']:.4f})")
    print(f"Alpha (Annual):      {results['alpha_annual']:>8.4f}%")
    
    sig = ""
    if results['p_values']['const'] < 0.01:
        sig = " ***"
    elif results['p_values']['const'] < 0.05:
        sig = " **"
    elif results['p_values']['const'] < 0.10:
        sig = " *"
    print(f"Significance:        {sig}")
    
    print(f"\nFactor Loadings:")
    print(f"  Market (Beta):     {results['beta_market']:>8.4f}  (p={results['p_values']['Mkt-RF']:.4f})")
    print(f"  SMB (Size):        {results['beta_smb']:>8.4f}  (p={results['p_values']['SMB']:.4f})")
    print(f"  HML (Value):       {results['beta_hml']:>8.4f}  (p={results['p_values']['HML']:.4f})")
    
    print(f"\nModel Fit:")
    print(f"  R-squared:         {results['r_squared']:>8.4f}")
    print(f"  Adj. R-squared:    {results['adj_r_squared']:>8.4f}")
    print(f"  Observations:      {results['n_obs']:>8.0f}")
    
    print("\n" + "=" * 80)

"""
Data loading and cleaning utilities for seasonal analysis project
"""
import pandas as pd
import numpy as np

def load_sector_data(file_path, sheet_name, skiprows=6, start_year=None, end_year=None, numrows=None):
    """
    Load and clean sector data from Excel file.
    
    Parameters:
    -----------
    file_path : str
        Path to Excel file
    sheet_name : str
        Name of sheet to load (e.g., 'S&P500', 'Discretionary', 'Staples')
    skiprows: int
        Number of rows to skip
    start_year : int, optional
        Filter data from this year onwards
    end_year : int, optional
        Filter data up to this year
        
    Returns:
    --------
    pd.DataFrame
        Cleaned dataframe with years as index, months as columns
    """
    # Read raw data
    
    df_raw = pd.read_excel(file_path, sheet_name=sheet_name, skiprows=skiprows)
    
    # Clean up
    if numrows: 
        df = df_raw.iloc[:numrows, 1:]  # Drop first unnamed column, special case for factor models
    else:
        df = df_raw.iloc[:, 1:] 
    df.columns = ['Year', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # Keep only rows with valid years
    df = df[pd.to_numeric(df['Year'], errors='coerce').notna()]
    df['Year'] = df['Year'].astype(int)
    
    # Remove summary rows (decade averages, etc.)
    df = df[df['Year'] < 2030]
    
    # Apply year filters if provided
    if start_year:
        df = df[df['Year'] >= start_year]
    if end_year:
        df = df[df['Year'] <= end_year]
    
    # Set year as index
    df = df.set_index('Year')
    
    return df


def load_multiple_sectors(file_path, sector_list, start_year=None, end_year=None):
    """
    Load multiple sectors at once.
    
    Parameters:
    -----------
    file_path : str
        Path to Excel file
    sector_list : list of str
        List of sheet names to load
    start_year, end_year : int, optional
        Year range filters
        
    Returns:
    --------
    dict
        Dictionary with sector names as keys, DataFrames as values
    """
    data_dict = {}
    
    for sector in sector_list:
        try:
            df = load_sector_data(file_path, sector, start_year, end_year)
            data_dict[sector] = df
            print(f"✓ Loaded {sector}: {df.shape[0]} years, {df.index.min()}-{df.index.max()}")
        except Exception as e:
            print(f"✗ Error loading {sector}: {e}")
    
    return data_dict


def get_all_returns(df):
    """
    Flatten all monthly returns into a single array (removes NaN).
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with monthly returns
        
    Returns:
    --------
    np.array
        Flattened array of all returns
    """
    all_returns = df.values.flatten()
    return all_returns[~np.isnan(all_returns)]


def get_seasonal_returns(df, season='winter'):
    """
    Get returns for a specific season.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with monthly returns
    season : str
        'winter' (Nov-Apr) or 'summer' (May-Oct)
        
    Returns:
    --------
    np.array
        Array of seasonal returns
    """
    if season.lower() == 'winter':
        months = ['Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr']
    elif season.lower() == 'summer':
        months = ['May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct']
    else:
        raise ValueError("Season must be 'winter' or 'summer'")
    
    seasonal_returns = df[months].values.flatten()
    seasonal_returns = pd.to_numeric(seasonal_returns, errors='coerce')
    return seasonal_returns[~np.isnan(seasonal_returns)]


def calculate_sector_ratio(df_numerator, df_denominator):
    """
    Calculate ratio of two sectors (e.g., Discretionary/Staples).
    Returns a DataFrame with the ratio for each month.
    
    Parameters:
    -----------
    df_numerator : pd.DataFrame
        Numerator sector returns
    df_denominator : pd.DataFrame
        Denominator sector returns
        
    Returns:
    --------
    pd.DataFrame
        Ratio values (same structure as input)
    """
    # Align indices (use common years only)
    common_index = df_numerator.index.intersection(df_denominator.index)
    
    df_num = df_numerator.loc[common_index]
    df_den = df_denominator.loc[common_index]
    
    # Calculate ratio (disc return - staples return gives relative performance)
    # Or could use (1 + disc/100) / (1 + staples/100) for true ratio
    ratio = df_num - df_den  # Simpler: outperformance measure
    
    return ratio
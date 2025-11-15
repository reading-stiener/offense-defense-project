"""
Visualization utilities for seasonal analysis
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_seasonal_comparison(winter_returns, summer_returns, title="Seasonal Comparison"):
    """
    Create comprehensive seasonal comparison plots.
    
    Parameters:
    -----------
    winter_returns : array-like
        Winter season returns
    summer_returns : array-like
        Summer season returns
    title : str
        Main title for the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Overlaid histograms
    axes[0, 0].hist(winter_returns, bins=30, alpha=0.6, 
                    label=f'Winter (μ={np.mean(winter_returns):.2f}%)', color='blue')
    axes[0, 0].hist(summer_returns, bins=30, alpha=0.6, 
                    label=f'Summer (μ={np.mean(summer_returns):.2f}%)', color='red')
    axes[0, 0].axvline(np.mean(winter_returns), color='blue', linestyle='--', linewidth=2)
    axes[0, 0].axvline(np.mean(summer_returns), color='red', linestyle='--', linewidth=2)
    axes[0, 0].set_xlabel('Monthly Return (%)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # 2. Box plots
    data_for_box = [winter_returns, summer_returns]
    box = axes[0, 1].boxplot(data_for_box, labels=['Winter\n(Nov-Apr)', 'Summer\n(May-Oct)'],
                              patch_artist=True)
    box['boxes'][0].set_facecolor('lightblue')
    box['boxes'][1].set_facecolor('lightcoral')
    axes[0, 1].set_ylabel('Monthly Return (%)')
    axes[0, 1].set_title('Box Plot Comparison')
    axes[0, 1].axhline(0, color='black', linestyle='--', alpha=0.5)
    axes[0, 1].grid(alpha=0.3)
    
    # 3. Violin plots
    parts = axes[1, 0].violinplot([winter_returns, summer_returns], 
                                   positions=[1, 2], showmeans=True, showmedians=True)
    axes[1, 0].set_xticks([1, 2])
    axes[1, 0].set_xticklabels(['Winter', 'Summer'])
    axes[1, 0].set_ylabel('Monthly Return (%)')
    axes[1, 0].set_title('Violin Plot Comparison')
    axes[1, 0].axhline(0, color='black', linestyle='--', alpha=0.5)
    axes[1, 0].grid(alpha=0.3)
    
    # 4. Summary statistics table
    axes[1, 1].axis('off')
    
    stats_data = [
        ['Metric', 'Winter', 'Summer', 'Difference'],
        ['Mean', f'{np.mean(winter_returns):.2f}%', f'{np.mean(summer_returns):.2f}%', 
         f'{np.mean(winter_returns) - np.mean(summer_returns):.2f}%'],
        ['Median', f'{np.median(winter_returns):.2f}%', f'{np.median(summer_returns):.2f}%',
         f'{np.median(winter_returns) - np.median(summer_returns):.2f}%'],
        ['Std Dev', f'{np.std(winter_returns):.2f}%', f'{np.std(summer_returns):.2f}%', ''],
        ['% Positive', f'{(winter_returns > 0).sum() / len(winter_returns) * 100:.1f}%',
         f'{(summer_returns > 0).sum() / len(summer_returns) * 100:.1f}%', '']
    ]
    
    table = axes[1, 1].table(cellText=stats_data, cellLoc='center', loc='center',
                            colWidths=[0.25, 0.25, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Color header row
    for i in range(4):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_monthly_averages(df, title="Average Returns by Month"):
    """
    Plot average returns for each month with seasonal coloring.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with monthly returns
    title : str
        Plot title
    """
    winter_months = ['Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr']
    month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    month_avgs = [df[month].mean() for month in month_order]
    colors = ['blue' if month in winter_months else 'red' for month in month_order]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(month_order, month_avgs, color=colors, alpha=0.7, edgecolor='black')
    plt.axhline(0, color='black', linestyle='-', linewidth=1)
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Average Return (%)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3, axis='y')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='blue', alpha=0.7, label='Winter (Nov-Apr)'),
                      Patch(facecolor='red', alpha=0.7, label='Summer (May-Oct)')]
    plt.legend(handles=legend_elements)
    
    plt.tight_layout()
    plt.show()
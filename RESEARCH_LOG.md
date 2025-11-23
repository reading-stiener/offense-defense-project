# Offense vs Defense Seasonality Research Project - Context & Progress

## Project Overview
Comprehensive quantitative finance research examining market seasonality and sector rotation patterns, with particular focus on validating the "Sell in May and Go Away" strategy using rigorous statistical methods.

## Core Research Questions
1. How can we validate "Sell in May and Go Away"? Performance of long Nov-Apr, cash May-Oct?
2. Can we quantify Consumer Discretionary vs Consumer Staples as market condition indicators?
3. How do inflation and interest rate conditions affect these patterns?
4. How can we improve the SZNE ETF structure based on our analysis?

## Dataset
- **Time Period:** 20+ years of monthly data
- **Data Includes:**
  - S&P 500 sector performance
  - Inflation rates
  - Interest rates
  - T-Bills (3-month Treasury rates)
  - SZNE ETF performance

### CRITICAL DATA CONSTRAINT
⚠️ **Dataset contains PRICE RETURNS, not TOTAL RETURNS**
- Systematically understates defensive sector performance by ~2-3% annually
- Missing dividend yields (defensive sectors like Consumer Staples pay higher dividends)
- Must account for this in analysis and conclusions

## Methodology: Deflated Sharpe Ratio (DSR)

### Why DSR?
Traditional backtesting systematically inflates performance through:
1. **Selection bias** - testing multiple strategies, reporting only winners
2. **Multiple testing** - more trials = higher probability of false positives
3. **Non-normal returns** - short samples and non-normal distributions inflate metrics

### Key Insight
Random samples contain patterns. With enough trials, you'll ALWAYS find a "profitable" strategy, even in random data. This is backtest overfitting.

**DSR corrects for:**
- Selection bias from multiple testing
- Non-normal return distributions  
- Sample size effects

### DSR Formula Components
```
DSR = Φ[(SR_hat - SR_0) * sqrt(T) / sqrt(1 - γ̂₃*SR_hat + (γ̂₄-1)/4*SR_hat²)]
```

Where:
- `SR_hat` = estimated Sharpe Ratio of selected strategy
- `SR_0` = expected maximum SR under null hypothesis (accounts for N trials)
- `T` = sample length
- `γ̂₃` = skewness of returns
- `γ̂₄` = kurtosis of returns
- `N` = number of independent trials conducted
- `V[{SR_n}]` = variance across trials' estimated SRs

**Critical:** SR_0 increases with number of trials (N). Must track ALL trials, not just winners.

## Technical Decisions & Architecture

### Code Structure (Modular Approach)
- `data_loader.py` - Load and validate data
- `returns_calculator.py` - Calculate returns, handle T-Bills properly
- `sharpe_analysis.py` - SR and DSR calculations
- `statistics.py` - Statistical tests and validation
- `visualization.py` - Matplotlib plotting functions

**Philosophy:** Reusable components, not copy-paste analysis

### T-Bills Handling
**IMPORTANT:** T-Bills data represents ANNUALIZED rates
- Must convert to monthly/period rates for excess return calculations
- Formula: `monthly_rate = (1 + annual_rate)^(1/12) - 1`
- Or for monthly data: `monthly_rate = annual_rate / 12` (approximation)
- Used as risk-free rate for excess return: `excess_return = portfolio_return - rf_rate`

### Data Validation Strategy
- Verify calculations against known benchmarks (e.g., 2020 S&P 500 returns)
- Mathematical validation of formulas before applying to novel analysis
- Check data types (numpy array compatibility)

## Research Phases

### Phase 1: Data Validation (IN PROGRESS)
- [ ] Load all data sources correctly
- [ ] Verify T-Bills rate conversion
- [ ] Calculate monthly returns properly
- [ ] Validate against known benchmarks
- [ ] Handle missing data/edge cases

### Phase 2: Seasonal Pattern Analysis
- [ ] Test "Sell in May" strategy performance
- [ ] Calculate Sharpe ratios for seasonal vs buy-and-hold
- [ ] Track number of strategy variations tested (N)
- [ ] Calculate variance of strategy SRs

### Phase 3: Sector Rotation Analysis  
- [ ] Consumer Discretionary (offense) vs Consumer Staples (defense) patterns
- [ ] Correlation with bull/bear markets
- [ ] Alternative defensive sectors (Utilities, Real Estate)
- [ ] Account for dividend yield bias in data

### Phase 4: Macro Conditions Analysis
- [ ] Segment by inflation regimes (high/low)
- [ ] Segment by interest rate environments (rising/falling)
- [ ] Interaction effects

### Phase 5: DSR Calculations & Validation
- [ ] Calculate DSR for selected strategies
- [ ] Determine statistical significance at 95% confidence
- [ ] Compare to naive Sharpe ratio conclusions
- [ ] Document all trials conducted

### Phase 6: SZNE ETF Analysis & Recommendations
- [ ] Analyze existing SZNE structure
- [ ] Propose improvements based on findings
- [ ] Consider timing, sectors, rebalancing

## Key Statistical Concepts

### Multiple Testing Problem
If you test N strategies at α=0.05 significance:
- Probability of at least one false positive increases with N
- After ~20 trials at 95% confidence, false positives become EXPECTED
- Holdout validation does NOT solve this - it assumes single trial

### Expected Maximum Sharpe Ratio
```
E[max{SR_n}] = E[{SR_n}] + sqrt(V[{SR_n}]) * ((1-γ)*Φ^(-1)(1-1/N) + γ*Φ^(-1)(1-1/(N*e)))
```
Where γ ≈ 0.5772 (Euler-Mascheroni constant)

**Key Point:** Expected max SR grows with N even if true SR = 0!

### Memory Effects & Backtest Overfitting
Financial series have memory (mean reversion, momentum)
- Overfitting finds extreme random patterns in-sample
- Memory "undoes" these patterns out-of-sample
- Result: Backtest overfitting → LOSS maximization (not just underperformance)

### Optimal Stopping (Secretary Problem)
How many trials should we run?
- Sample ~37% (1/e) of theoretically justified configurations
- Continue testing until finding one that beats all previous
- Minimizes false positive probability while finding near-optimal

## Current Technical Status

### Completed
- ✅ Python environment setup
- ✅ Understanding DSR methodology
- ✅ Modular architecture design
- ✅ T-Bills calculation approach defined
- ✅ Data structure planning

### In Progress
- 🔄 Data loading implementation
- 🔄 Returns calculation (handling T-Bills conversion)
- 🔄 Debugging numpy data type issues
- 🔄 Matplotlib visualization setup

### Blockers/Issues
- Data type compatibility with numpy operations
- Proper T-Bills rate conversion verification
- Setting up visualization workflows

## Important Reminders

1. **Track ALL trials** - N must include everything tested, not just reported
2. **Price returns only** - Defensive sectors artificially underperform by 2-3%
3. **T-Bills are annualized** - Must convert for monthly calculations  
4. **DSR > 0.95 needed** - For 95% confidence after accounting for multiple testing
5. **Sample size matters** - Longer history = more statistical power
6. **Non-normality matters** - Skewness and kurtosis inflate naive SR

## Key References

- **Bailey & López de Prado (2014)** - "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality"
  - Core methodology paper
  - See project files: `deflatedsharpe1.pdf`
  
- **Project Scope** - See `OffenceVsDefenseProjectScope_MPF241.pdf`

## Next Steps
1. Complete data loading and validation
2. Verify T-Bills conversion with test calculations
3. Implement basic Sharpe ratio calculations
4. Set up visualization for seasonal patterns
5. Begin tracking trial count for DSR

## Notes for Future Sessions
- When continuing work, reference this file for context
- Update progress checkboxes as phases complete
- Document any new technical decisions or insights
- Track the running count of trials (N) in a separate log
- Note any deviations from planned approach with rationale

---
*Last Updated: 11/22/2025*
*Current Phase: Data Validation*
*Status: Setting up infrastructure and debugging data processing*

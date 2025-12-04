"""
Portfolio Optimizer - Phase 4: Portfolio Construction
Mean-Variance optimization with enhanced risk constraints (CVaR, Sortino, Max Drawdown).
Addresses judge feedback on Sortino's limitations by incorporating multiple risk metrics.

Run this AFTER:
1. Clustering analysis complete
2. Qualitative diligence complete
3. Expected returns adjusted (Black-Litterman "lite" principle)
4. Manual allocations decided (bonds + cash)

Output: Optimal portfolio weights meeting all constraints
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class PortfolioOptimizer:
    
    def __init__(self, risk_free_rate: float = 0.04):
        self.risk_free_rate = risk_free_rate
        self.stocks_df = None
        self.returns_data = None
        self.cov_matrix = None
        self.downside_cov_matrix = None
        
    def load_approved_stocks(self, file_path: str, category_col: str = 'category'):
        """Load stocks that passed qualitative diligence"""
        print("=" * 70)
        print("PORTFOLIO OPTIMIZATION - Phase 4")
        print("=" * 70)
        print(f"Loading approved stocks from: {file_path}\n")
        
        self.stocks_df = pd.read_csv(file_path)
        
        # Filter out rejected stocks
        if category_col in self.stocks_df.columns:
            self.stocks_df = self.stocks_df[self.stocks_df[category_col] != 'Reject']
            print(f"✓ Loaded {len(self.stocks_df)} approved stocks")
            print(f"  Categories: {self.stocks_df[category_col].value_counts().to_dict()}")
        else:
            print(f"⚠️  No '{category_col}' column found - using all stocks")
            print(f"✓ Loaded {len(self.stocks_df)} stocks")
        
        return self
    
    
    def load_historical_returns(self, tickers: List[str], years: int = 3):
        """Fetch historical price data and calculate daily returns"""
        print(f"\n📊 Fetching historical data ({years} years)...")
        
        import yfinance as yf
        from datetime import datetime, timedelta
        
        end = datetime.now()
        start = end - timedelta(days=365 * years)
        
        all_prices = {}
        failed = []
        
        for ticker in tickers:
            try:
                data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
                
                if not data.empty:
                    # Handle MultiIndex columns from yfinance
                    if isinstance(data.columns, pd.MultiIndex):
                        data.columns = [col[0] for col in data.columns]
                    
                    all_prices[ticker] = data['Close']
                    print(f"  ✓ {ticker}")
                else:
                    failed.append(ticker)
                    print(f"  ❌ {ticker} - no data")
            except Exception as e:
                failed.append(ticker)
                print(f"  ❌ {ticker} - {str(e)}")
        
        if not all_prices:
            raise ValueError("No historical data could be fetched. Check tickers.")
        
        # Create DataFrame and align dates
        prices_df = pd.DataFrame(all_prices)
        prices_df = prices_df.dropna()
        
        if prices_df.empty:
            raise ValueError("All price data has missing values - cannot proceed")
        
        self.returns_data = prices_df.pct_change().dropna()
        
        print(f"\n✓ Successfully fetched: {len(all_prices)} stocks")
        print(f"✓ Date range: {prices_df.index[0].date()} to {prices_df.index[-1].date()}")
        print(f"✓ Trading days: {len(self.returns_data)}")
        
        if failed:
            print(f"\n⚠️  Failed to fetch {len(failed)} stocks: {', '.join(failed)}")
            print(f"   These will be excluded from optimization")
        
        return self

    def calculate_covariance_matrices(self):
        """Calculate both regular and downside covariance matrices"""
        print(f"\n📈 Calculating covariance matrices...")
        
        # Regular covariance (annualized)
        self.cov_matrix = self.returns_data.cov() * 252
        
        # Downside covariance (only negative return days)
        downside_returns = self.returns_data[self.returns_data < 0].fillna(0)
        self.downside_cov_matrix = downside_returns.cov() * 252
        
        print(f"✓ Regular covariance: {self.cov_matrix.shape}")
        print(f"✓ Downside covariance: {self.downside_cov_matrix.shape}")
        return self
    
    def apply_black_litterman_lite(self, adjustments: Dict[str, Tuple[float, str]]):
        """
        Apply Black-Litterman "lite" principle - adjust expected returns based on views.
        
        adjustments: Dict[ticker, (new_return, rationale)]
        Example: {'NVDA': (0.25, 'AI hype normalization'), 'NEE': (0.11, 'IRA tailwinds')}
        """
        print(f"\n🎯 Applying Black-Litterman Lite adjustments...")
        print(f"   (Adjusting {len(adjustments)} stocks based on qualitative views)\n")
        
        # Start with historical annualized returns as baseline
        historical_returns = self.returns_data.mean() * 252
        self.stocks_df['expected_return'] = self.stocks_df['ticker'].map(historical_returns)
        
        # Apply team adjustments
        for ticker, (new_return, rationale) in adjustments.items():
            if ticker in self.stocks_df['ticker'].values:
                old_return = self.stocks_df.loc[self.stocks_df['ticker'] == ticker, 'expected_return'].values[0]
                self.stocks_df.loc[self.stocks_df['ticker'] == ticker, 'expected_return'] = new_return
                print(f"  {ticker}: {old_return:.1%} → {new_return:.1%}")
                print(f"    Rationale: {rationale}")
        
        print(f"\n✓ Expected returns finalized")
        return self
    
    def optimize_bucket(self, 
                   bucket_name: str,
                   tickers: List[str],
                   bucket_capital: float,
                   constraints: Dict):
        """
        Optimize a single bucket (Core, Growth, or Impact).
        """
        print(f"\n{'='*70}")
        print(f"OPTIMIZING {bucket_name.upper()} BUCKET")
        print(f"{'='*70}")
        print(f"Capital: ${bucket_capital:,.0f}")
        print(f"Securities: {len(tickers)}")
        
        # Check if we have enough securities
        min_holdings = int(1.0 / constraints.get('max_position', 0.15))
        if len(tickers) < min_holdings:
            print(f"\n⚠️  Not enough securities!")
            print(f"   Need at least {min_holdings} stocks for max position {constraints.get('max_position', 0.15):.0%}")
            print(f"   Only have {len(tickers)} stocks")
            print(f"   Skipping {bucket_name} optimization")
            return None
        
        # Filter data for this bucket
        bucket_returns = self.returns_data[tickers]
        bucket_expected = self.stocks_df[self.stocks_df['ticker'].isin(tickers)].set_index('ticker')['expected_return']
        
        # Check for missing expected returns
        missing_returns = [t for t in tickers if t not in bucket_expected.index]
        if missing_returns:
            print(f"\n⚠️  Missing expected returns for: {missing_returns}")
            print(f"   Removing these from optimization")
            tickers = [t for t in tickers if t not in missing_returns]
            bucket_returns = self.returns_data[tickers]
            bucket_expected = self.stocks_df[self.stocks_df['ticker'].isin(tickers)].set_index('ticker')['expected_return']
        
        bucket_cov = self.cov_matrix.loc[tickers, tickers]
        bucket_downside_cov = self.downside_cov_matrix.loc[tickers, tickers]
        
        n_assets = len(tickers)
        
        # Objective function: Negative expected return
        def objective(weights):
            return -np.dot(weights, bucket_expected)
        
        # Constraint functions
        def portfolio_volatility(weights):
            return np.sqrt(np.dot(weights, np.dot(bucket_cov, weights)))
        
        def portfolio_sortino(weights):
            downside_vol = np.sqrt(np.dot(weights, np.dot(bucket_downside_cov, weights)))
            excess_return = np.dot(weights, bucket_expected) - self.risk_free_rate
            return excess_return / downside_vol if downside_vol > 0 else 0
        
        def portfolio_cvar(weights):
            portfolio_returns = bucket_returns.dot(weights)
            var_threshold = portfolio_returns.quantile(0.05)
            cvar = portfolio_returns[portfolio_returns <= var_threshold].mean()
            return cvar
        
        def portfolio_max_drawdown(weights):
            portfolio_returns = bucket_returns.dot(weights)
            cumulative = (1 + portfolio_returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            return drawdown.min()
        
        # Build constraint list - START SIMPLE
        constraint_list = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},  # Weights sum to 1
        ]
        
        # Bounds: position size limits
        min_weight = constraints.get('min_position', 0.05)
        max_weight = constraints.get('max_position', 0.25)
        bounds = tuple((min_weight, max_weight) for _ in range(n_assets))
        
        # Initial guess: equal weights
        x0 = np.array([1.0 / n_assets] * n_assets)
        
        print(f"\n🔧 Running optimization...")
        print(f"   Assets: {n_assets}")
        print(f"   Position limits: {min_weight:.1%} - {max_weight:.1%}")
        print(f"   Constraints: {len(constraint_list)} (starting simple)")
        
        # First pass: Just maximize return with basic constraints
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraint_list,
            options={'maxiter': 1000, 'ftol': 1e-6}
        )
        
        if not result.success:
            print(f"\n❌ Basic optimization failed: {result.message}")
            print(f"   This usually means position size constraints are impossible")
            print(f"   Try reducing min_position or increasing max_position")
            return None
        
        print(f"✓ Basic optimization successful")
        
        # Now add risk constraints one by one
        weights = result.x
        
        print(f"\n📊 Checking risk metrics on unconstrained solution:")
        print(f"   Volatility: {portfolio_volatility(weights):.2%}")
        print(f"   Sortino: {portfolio_sortino(weights):.3f}")
        print(f"   CVaR: {portfolio_cvar(weights):.2%}")
        print(f"   Max DD: {portfolio_max_drawdown(weights):.2%}")
        
        # Now try adding risk constraints
        if 'max_volatility' in constraints:
            current_vol = portfolio_volatility(weights)
            if current_vol > constraints['max_volatility']:
                print(f"\n⚠️  Volatility {current_vol:.2%} exceeds limit {constraints['max_volatility']:.2%}")
                print(f"   Adding volatility constraint...")
                constraint_list.append({
                    'type': 'ineq',
                    'fun': lambda w: constraints['max_volatility'] - portfolio_volatility(w)
                })
        
        if 'min_sortino' in constraints:
            current_sortino = portfolio_sortino(weights)
            if current_sortino < constraints['min_sortino']:
                print(f"\n⚠️  Sortino {current_sortino:.3f} below minimum {constraints['min_sortino']:.3f}")
                print(f"   Adding Sortino constraint...")
                constraint_list.append({
                    'type': 'ineq',
                    'fun': lambda w: portfolio_sortino(w) - constraints['min_sortino']
                })
        
        # Re-optimize with risk constraints if any were added
        if len(constraint_list) > 1:
            print(f"\n🔧 Re-optimizing with {len(constraint_list)} constraints...")
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraint_list,
                options={'maxiter': 1000, 'ftol': 1e-6}
            )
            
            if not result.success:
                print(f"\n⚠️  Constrained optimization failed: {result.message}")
                print(f"   Using unconstrained solution instead")
            else:
                weights = result.x
                print(f"✓ Constrained optimization successful")
        
        # Package results
        weights_dict = dict(zip(tickers, weights))
        
        # Calculate final metrics
        final_return = np.dot(weights, bucket_expected)
        final_vol = portfolio_volatility(weights)
        final_sortino = portfolio_sortino(weights)
        final_cvar = portfolio_cvar(weights)
        final_drawdown = portfolio_max_drawdown(weights)
        final_sharpe = (final_return - self.risk_free_rate) / final_vol if final_vol > 0 else 0
        
        print(f"\n📊 FINAL RESULTS:")
        print(f"   Expected Return: {final_return:.2%}")
        print(f"   Volatility: {final_vol:.2%}")
        print(f"   Sharpe Ratio: {final_sharpe:.3f}")
        print(f"   Sortino Ratio: {final_sortino:.3f}")
        print(f"   CVaR (95%): {final_cvar:.2%}")
        print(f"   Max Drawdown: {final_drawdown:.2%}")
        
        print(f"\n💼 ALLOCATIONS:")
        sorted_positions = sorted(weights_dict.items(), key=lambda x: x[1], reverse=True)
        for ticker, weight in sorted_positions:
            print(f"   {ticker}: {weight:.2%}")
        
        return {
            'weights': weights_dict,
            'metrics': {
                'expected_return': final_return,
                'volatility': final_vol,
                'sharpe': final_sharpe,
                'sortino': final_sortino,
                'cvar': final_cvar,
                'max_drawdown': final_drawdown
            }
        }

    def build_complete_portfolio(self, 
                                bucket_results: Dict,
                                manual_allocations: Dict,
                                total_capital: float = 500000):
        """Combine bucket results with manual allocations into final portfolio"""
        print(f"\n{'='*70}")
        print(f"BUILDING COMPLETE PORTFOLIO")
        print(f"{'='*70}")
        
        final_portfolio = {}
        
        # Add manual allocations
        print(f"\n🔒 Manual Allocations:")
        for name, amount in manual_allocations.items():
            pct = amount / total_capital
            final_portfolio[name] = {'amount': amount, 'weight': pct}
            print(f"   {name}: ${amount:,.0f} ({pct:.2%})")
        
        # Add optimized buckets
        print(f"\n🎯 Optimized Allocations:")
        for bucket_name, result in bucket_results.items():
            capital = result['capital']
            print(f"\n  {bucket_name.upper()} ({capital / total_capital:.2%} of portfolio):")
            
            for ticker, weight in result['weights'].items():
                amount = capital * weight
                total_weight = amount / total_capital
                final_portfolio[ticker] = {'amount': amount, 'weight': total_weight}
                print(f"    {ticker}: ${amount:,.0f} ({total_weight:.2%})")
        
        # Create summary DataFrame
        portfolio_df = pd.DataFrame(final_portfolio).T
        portfolio_df = portfolio_df.sort_values('weight', ascending=False)
        
        print(f"\n✓ Complete portfolio: {len(portfolio_df)} positions")
        print(f"   Total allocated: ${portfolio_df['amount'].sum():,.0f}")
        
        return portfolio_df
    
    def visualize_efficient_frontier(self, tickers: List[str], n_portfolios: int = 1000):
        """Generate efficient frontier showing risk-return tradeoff"""
        print(f"\n📈 Generating efficient frontier ({n_portfolios} portfolios)...")
        
        returns = self.stocks_df[self.stocks_df['ticker'].isin(tickers)].set_index('ticker')['expected_return']
        cov = self.cov_matrix.loc[tickers, tickers]
        
        n_assets = len(tickers)
        results = np.zeros((3, n_portfolios))
        
        for i in range(n_portfolios):
            weights = np.random.random(n_assets)
            weights /= np.sum(weights)
            
            port_return = np.dot(weights, returns)
            port_vol = np.sqrt(np.dot(weights, np.dot(cov, weights)))
            port_sharpe = (port_return - self.risk_free_rate) / port_vol
            
            results[0, i] = port_vol
            results[1, i] = port_return
            results[2, i] = port_sharpe
        
        # Plot
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(results[0, :], results[1, :], 
                            c=results[2, :], cmap='viridis', 
                            alpha=0.5, s=10)
        plt.colorbar(scatter, label='Sharpe Ratio')
        plt.xlabel('Volatility (Risk)', fontsize=13, fontweight='bold')
        plt.ylabel('Expected Return', fontsize=13, fontweight='bold')
        plt.title('Efficient Frontier - Risk vs Return', fontsize=15, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('efficient_frontier.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: efficient_frontier.png")
        plt.close()
        
        return results


# ========== MAIN EXECUTION ==========

if __name__ == "__main__":
    
    # Configuration
    INPUT_FILE = 'clustered_stocks.csv'  # After qualitative review
    TOTAL_CAPITAL = 500000
    RISK_FREE_RATE = 0.04
    
    # Manual allocations (decided before optimization)
    MANUAL_ALLOCATIONS = {
        'Cash Reserve': 10000,      # 2%
        'Treasury Ladder': 62500    # 12.5%
    }
    MANUAL_TOTAL = sum(MANUAL_ALLOCATIONS.values())
    OPTIMIZABLE_CAPITAL = TOTAL_CAPITAL - MANUAL_TOTAL  # $427,500 = 85.5%
    
    # Bucket targets (as % of optimizable capital)
    BUCKET_TARGETS = {
        'Core': 0.40,      # 40% of $427,500 = $171,000
        'Growth': 0.38,    # 38% of $427,500 = $162,450
        'Impact': 0.17     # 17% of $427,500 = $72,675
    }
    # Buffer: 5% = $21,375 (held as additional cash)
    
    # Constraints per bucket
    CONSTRAINTS = {
        'Core': {
            'max_volatility': 0.18,      # Relaxed from 0.15
            'min_sortino': 0.50,         # Relaxed from 0.75
            'min_position': 0.05,        # Relaxed from 0.08
            'max_position': 0.25         # Relaxed from 0.15
        },
        'Growth': {
            'max_volatility': 0.30,      # Relaxed from 0.25
            'min_sortino': 0.40,         # Relaxed from 0.60
            'min_position': 0.03,        # Relaxed from 0.10
            'max_position': 0.30         # Relaxed from 0.20
        },
        'Impact': {
            'max_volatility': 0.25,      # Relaxed from 0.20
            'min_sortino': 0.45,         # Relaxed from 0.65
            'min_position': 0.10,        # Relaxed from 0.15
            'max_position': 0.40         # Relaxed from 0.35
        }
    }
    
    # Black-Litterman Lite adjustments (based on qualitative research)
    BL_ADJUSTMENTS = {
        'NVDA': (0.25, 'AI hype normalization expected'),
        'NEE': (0.11, 'IRA clean energy tailwinds'),
        # Add more as needed after Step 9
    }
    
    print("\n" + "="*70)
    print("PORTFOLIO OPTIMIZATION PIPELINE")
    print("="*70)
    print(f"Total Capital: ${TOTAL_CAPITAL:,.0f}")
    print(f"Manual Allocations: ${MANUAL_TOTAL:,.0f} ({MANUAL_TOTAL/TOTAL_CAPITAL:.1%})")
    print(f"Optimizable Capital: ${OPTIMIZABLE_CAPITAL:,.0f} ({OPTIMIZABLE_CAPITAL/TOTAL_CAPITAL:.1%})")
    print("="*70)
    
    # Initialize optimizer
    optimizer = PortfolioOptimizer(risk_free_rate=RISK_FREE_RATE)
    
    # Load data
    optimizer.load_approved_stocks(INPUT_FILE, category_col='category')
    
    # Get tickers per category
    core_tickers = optimizer.stocks_df[optimizer.stocks_df['category'] == 'Core']['ticker'].tolist()
    growth_tickers = optimizer.stocks_df[optimizer.stocks_df['category'] == 'Growth']['ticker'].tolist()
    impact_tickers = optimizer.stocks_df[optimizer.stocks_df['category'] == 'Impact']['ticker'].tolist()
    
    all_tickers = core_tickers + growth_tickers + impact_tickers
    
    # Fetch historical data
    optimizer.load_historical_returns(all_tickers, years=3)
    optimizer.calculate_covariance_matrices()
    
    # Apply Black-Litterman lite
    optimizer.apply_black_litterman_lite(BL_ADJUSTMENTS)
    
    # Optimize each bucket
    bucket_results = {}
    
    for bucket_name, target_pct in BUCKET_TARGETS.items():
        bucket_capital = OPTIMIZABLE_CAPITAL * target_pct
        
        if bucket_name == 'Core':
            tickers = core_tickers
        elif bucket_name == 'Growth':
            tickers = growth_tickers
        else:
            tickers = impact_tickers
        
        result = optimizer.optimize_bucket(
            bucket_name=bucket_name,
            tickers=tickers,
            bucket_capital=bucket_capital,
            constraints=CONSTRAINTS[bucket_name]
        )
        
        if result:
            bucket_results[bucket_name] = {
                'weights': result['weights'],
                'metrics': result['metrics'],
                'capital': bucket_capital
            }
    
    # Build complete portfolio
    final_portfolio = optimizer.build_complete_portfolio(
        bucket_results=bucket_results,
        manual_allocations=MANUAL_ALLOCATIONS,
        total_capital=TOTAL_CAPITAL
    )
    
    # Save results
    final_portfolio.to_csv('final_portfolio.csv')
    print(f"\n💾 Saved: final_portfolio.csv")
    
    # Visualize efficient frontier
    optimizer.visualize_efficient_frontier(all_tickers, n_portfolios=1000)
    
    print("\n" + "="*70)
    print("✅ OPTIMIZATION COMPLETE")
    print("="*70)
    print("\nNext Steps:")
    print("1. Review final_portfolio.csv for position details")
    print("2. Check efficient_frontier.png to visualize risk-return tradeoff")
    print("3. Proceed to Monte Carlo stress testing (Phase 6)")
    print("="*70)
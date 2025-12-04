"""
Data Collector Script - Simplified for Phase 2 Step 4
Fetches historical data and calculates required metrics for quantitative screening.
Saves results to stock_metrics.csv for clustering analysis.

Required metrics per the spec:
- Annualized return (CAGR)
- Volatility (annualized)
- Sharpe ratio
- Sortino ratio
- Max drawdown
- CVaR (95%)
- Beta vs SPY
- ROE
- Dividend yield
- P/E ratio (for context)

Note: ESG scores and ETF Expense Ratios must be added manually after running this script.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class DataCollector:
    """Simplified data collector using only yfinance (no paid APIs needed)"""
    
    def __init__(self, risk_free_rate: float = 0.04):
        """
        Args:
            risk_free_rate: Annual risk-free rate (default 4%)
        """
        self.risk_free_rate = risk_free_rate
        # Removed self._etf_cache
    
    # ========== DATA FETCHING ==========

    # Removed _load_etf_expense_ratios method

    def get_historical_prices(self, symbol: str, years: int = 3) -> pd.DataFrame:
        """Fetch historical prices using yfinance"""
        end = datetime.now()
        start = end - timedelta(days=365 * years)
        
        try:
            data = yf.download(
                symbol,
                start=start.strftime('%Y-%m-%d'),
                end=end.strftime('%Y-%m-%d'),
                progress=False,
                auto_adjust=True  # Use adjusted prices
            )
            
            if data.empty:
                print(f"  ⚠️  No data for {symbol}")
                return pd.DataFrame()
            
            # Flatten MultiIndex columns if present
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [col[0] for col in data.columns]
            
            df = data.reset_index()
            df.columns = df.columns.str.lower()
            df['date'] = pd.to_datetime(df['date'])
            
            return df[['date', 'close']].dropna()
            
        except Exception as e:
            print(f"  ❌ Failed to fetch {symbol}: {e}")
            return pd.DataFrame()
    
    def get_stock_info(self, symbol: str) -> Dict:
        """Fetch fundamental metrics from yfinance"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            quote_type = info.get('quoteType', 'EQUITY')
            is_etf = quote_type in ['ETF', 'MUTUALFUND']
            
            # Expense ratio will be manually added later, so it's None for now
            expense_ratio = None
            
            # Calculate unified quality score
            # NOTE: For ETFs, this will be None until you manually add the expense_ratio
            if is_etf:
                # The original logic was: (1 - expense_ratio) if expense_ratio else None
                # Since we don't fetch it, it defaults to None for now.
                quality_score = None 
            else:
                quality_score = info.get('returnOnEquity')
            
            return {
                'roe': info.get('returnOnEquity') if not is_etf else None,
                'expense_ratio': expense_ratio,
                'quality_score': quality_score,  # Universal metric for clustering
                'dividend_yield': info.get('dividendYield') or info.get('yield') or info.get('trailingAnnualDividendYield'),
                'pe_ratio': info.get('trailingPE') or info.get('forwardPE') if not is_etf else None,
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'is_etf': is_etf
            }
        except Exception as e:
            print(f"  ⚠️  Could not fetch info for {symbol}: {e}")
            return {}

    # ========== METRIC CALCULATIONS ==========
    
    def calculate_returns(self, prices: pd.DataFrame) -> pd.Series:
        """Calculate daily returns"""
        return prices['close'].pct_change().dropna()
    
    def calculate_annualized_return(self, prices: pd.DataFrame) -> Optional[float]:
        """Calculate CAGR"""
        if len(prices) < 2:
            return None
        
        start_price = prices['close'].iloc[0]
        end_price = prices['close'].iloc[-1]
        n_years = (prices['date'].iloc[-1] - prices['date'].iloc[0]).days / 365.25
        
        if start_price <= 0 or n_years <= 0:
            return None
        
        return (end_price / start_price) ** (1 / n_years) - 1
    
    def calculate_volatility(self, returns: pd.Series) -> Optional[float]:
        """Calculate annualized volatility"""
        if len(returns) < 2:
            return None
        return returns.std() * np.sqrt(252)
    
    def calculate_sharpe_ratio(self, returns: pd.Series) -> Optional[float]:
        """Calculate Sharpe ratio"""
        if len(returns) < 2:
            return None
        
        excess_returns = returns - (self.risk_free_rate / 252)
        mean_excess = excess_returns.mean()
        std_excess = excess_returns.std()
        
        if std_excess == 0:
            return None
        
        return (mean_excess / std_excess) * np.sqrt(252)
    
    def calculate_sortino_ratio(self, returns: pd.Series) -> Optional[float]:
        """Calculate Sortino ratio (downside deviation)"""
        if len(returns) < 2:
            return None
        
        excess_returns = returns - (self.risk_free_rate / 252)
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            return None
        
        mean_excess = excess_returns.mean()
        downside_std = downside_returns.std()
        
        if downside_std == 0:
            return None
        
        return (mean_excess / downside_std) * np.sqrt(252)
    
    def calculate_max_drawdown(self, prices: pd.DataFrame) -> Optional[float]:
        """Calculate maximum drawdown"""
        if len(prices) < 2:
            return None
        
        returns = self.calculate_returns(prices)
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        return drawdown.min()
    
    def calculate_cvar(self, returns: pd.Series, confidence: float = 0.95) -> Optional[float]:
        """Calculate CVaR (Expected Shortfall at 95% confidence)"""
        if len(returns) < 2:
            return None
        
        var_threshold = returns.quantile(1 - confidence)
        tail_returns = returns[returns <= var_threshold]
        
        if len(tail_returns) == 0:
            return None
        
        return tail_returns.mean()
    
    def calculate_beta(self, stock_prices: pd.DataFrame, benchmark_prices: pd.DataFrame) -> Optional[float]:
        """Calculate beta vs benchmark"""
        if stock_prices.empty or benchmark_prices.empty:
            return None
        
        # Merge on date
        merged = pd.merge(
            stock_prices[['date', 'close']],
            benchmark_prices[['date', 'close']],
            on='date',
            suffixes=('_stock', '_bench')
        )
        
        if len(merged) < 2:
            return None
        
        stock_returns = merged['close_stock'].pct_change().dropna()
        bench_returns = merged['close_bench'].pct_change().dropna()
        
        if len(stock_returns) < 2 or len(bench_returns) < 2:
            return None
        
        covariance = np.cov(stock_returns, bench_returns)[0, 1]
        bench_variance = bench_returns.var()
        
        if bench_variance == 0:
            return None
        
        return covariance / bench_variance
    
    def calculate_tail_ratio(self, returns: pd.Series) -> Optional[float]:
        """Calculate tail ratio: gain in best 5% / loss in worst 5%"""
        if len(returns) < 20:
            return None
        
        top_5_pct = returns.quantile(0.95)
        bottom_5_pct = returns.quantile(0.05)
        
        gains = returns[returns >= top_5_pct].mean()
        losses = abs(returns[returns <= bottom_5_pct].mean())
        
        if losses == 0:
            return None
        
        return gains / losses

    def calculate_rolling_correlation_std(self, stock_prices: pd.DataFrame, 
                                        benchmark_prices: pd.DataFrame, 
                                        window: int = 90) -> Optional[float]:
        """Calculate std dev of 90-day rolling correlation with SPY"""
        if stock_prices.empty or benchmark_prices.empty:
            return None
        
        merged = pd.merge(
            stock_prices[['date', 'close']],
            benchmark_prices[['date', 'close']],
            on='date',
            suffixes=('_stock', '_bench')
        )
        
        if len(merged) < window + 1:
            return None
        
        stock_returns = merged['close_stock'].pct_change()
        bench_returns = merged['close_bench'].pct_change()
        
        rolling_corr = stock_returns.rolling(window=window).corr(bench_returns)
        
        return rolling_corr.std()

    def calculate_liquidity_score(self, symbol: str) -> Optional[float]:
        """Calculate liquidity score: avg daily volume × price"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            avg_volume = info.get('averageVolume') or info.get('averageVolume10days')
            price = info.get('currentPrice') or info.get('previousClose')
            
            if avg_volume and price:
                return avg_volume * price
            return None
        except:
            return None
    
    # ========== MAIN COLLECTION ==========
    
    def collect_stock_metrics(self, symbol: str, benchmark_prices: pd.DataFrame, 
                         years: int = 3) -> Dict:
        """Collect all required metrics for a single stock"""
        print(f"Collecting {symbol}...", end=" ")
        
        result = {'ticker': symbol}
        
        # Get price data
        prices = self.get_historical_prices(symbol, years)
        
        if prices.empty:
            print("❌ No data")
            return result
        
        # Calculate return-based metrics
        returns = self.calculate_returns(prices)
        
        result['annualized_return'] = self.calculate_annualized_return(prices)
        result['volatility'] = self.calculate_volatility(returns)
        result['sharpe_ratio'] = self.calculate_sharpe_ratio(returns)
        result['sortino_ratio'] = self.calculate_sortino_ratio(returns)
        result['max_drawdown'] = self.calculate_max_drawdown(prices)
        result['cvar_95'] = self.calculate_cvar(returns)
        result['beta'] = self.calculate_beta(prices, benchmark_prices)
        
        # *** ADD THESE THREE NEW METRICS ***
        result['tail_ratio'] = self.calculate_tail_ratio(returns)
        result['rolling_correlation_std'] = self.calculate_rolling_correlation_std(prices, benchmark_prices)
        result['liquidity_score'] = self.calculate_liquidity_score(symbol)
        
        # Get fundamental data
        info = self.get_stock_info(symbol)
        result.update(info)
        
        print("✓")
        return result
    
    def collect_universe(self, tickers: List[str], benchmark: str = 'SPY', 
                        years: int = 3) -> pd.DataFrame:
        """Collect metrics for entire universe"""
        print("=" * 60)
        print("DATA COLLECTION - Phase 2 Step 4")
        print("=" * 60)

        # Removed: Logic to identify ETFs and call _load_etf_expense_ratios.
        
        print(f"Universe: {len(tickers)} tickers")
        print(f"Benchmark: {benchmark}")
        print(f"Period: {years} years")
        print(f"Risk-free rate: {self.risk_free_rate:.2%}")
        print("=" * 60)
        
        # Fetch benchmark once
        print(f"\nFetching {benchmark} benchmark...")
        benchmark_prices = self.get_historical_prices(benchmark, years)
        print(f"✓ Benchmark: {len(benchmark_prices)} days\n")
        
        # Collect each stock
        results = []
        for i, ticker in enumerate(tickers, 1):
            print(f"[{i}/{len(tickers)}] ", end="")
            data = self.collect_stock_metrics(ticker, benchmark_prices, years)
            results.append(data)
        
        df = pd.DataFrame(results)
        
        # Summary
        print("\n" + "=" * 60)
        print("COLLECTION COMPLETE")
        print("=" * 60)
        print(f"✓ Collected: {len(results)} stocks")
        print(f"✓ Metrics per stock: {len([c for c in df.columns if c != 'ticker'])}")
        
        # Check for missing data
        key_metrics = ['annualized_return', 'volatility', 'sharpe_ratio', 
                      'sortino_ratio', 'beta']
        missing = df[key_metrics].isna().sum()
        if missing.any():
            print("\n⚠️  Missing data:")
            for col in missing[missing > 0].index:
                print(f"  {col}: {missing[col]} stocks")
        
        print("=" * 60)
        
        return df


# ========== MAIN EXECUTION ==========

if __name__ == "__main__":
    # Configuration
    TICKERS = [
        
        'AAPL', 'NET', 'NVDA', 
        
         'ESGE', 'ICLN', 

        'OKLO', 'NLR', 'VEA', 'NXT', 'NEE', 'XYL', 'CRH', 'NLR', 'BMY', 'TYL', 'GRMN', 'GOOG', 'AVGO', 'CYBR', 'CRWD', 'AMD', 'DDOG', 'PLTR', 'NVDA', 'ABT',
        'VRTX', 'TEM', 'ISRG', 'PG', 'COST', 'WMT', 'JBGS', 'ARE', 'HASI', 'SPG', 'IEMG', 'CEG', 'AEP', 'SRE', 'DUK', 'BEP', 'FSLR'
    ]
    
    BENCHMARK = 'SPY'
    YEARS = 3
    OUTPUT_FILE = 'stock_metrics.csv'
    RISK_FREE_RATE = 0.0404  # 4% annual
    
    # Run collection
    collector = DataCollector(risk_free_rate=RISK_FREE_RATE)
    df = collector.collect_universe(TICKERS, BENCHMARK, YEARS)
    
    # Save to CSV
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n💾 Saved to: {OUTPUT_FILE}")
    
    # Preview
    print("\n📊 Preview:")
    preview_cols = ['ticker', 'annualized_return', 'volatility', 'sharpe_ratio', 
               'sortino_ratio', 'tail_ratio', 'liquidity_score', 'quality_score',
               'roe', 'dividend_yield']
    preview_cols = [c for c in preview_cols if c in df.columns]
    print(df[preview_cols].head(10).to_string(index=False))
    
    print("\n" + "=" * 60)
    print("NEXT STEPS:")
    print("=" * 60)
    print("1. ⚠️  Manually add 'expense_ratio' and 'esg_score' columns to stock_metrics.csv")
    print("   - For ETFs, calculate and fill 'quality_score' as (1 - expense_ratio) / 1.0 (max ROE is 100%)")
    print("   - Look up ESG Risk Ratings on Sustainalytics/Morningstar")
    print("   - Lower ESG score = better (0-10 = Negligible, 10-20 = Low, etc.)")
    print("2. Run clustering_analysis.py for Phase 2 Step 6")
    print("=" * 60)
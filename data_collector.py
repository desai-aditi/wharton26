"""
Data Collector Script
Fetches historical data and calculates metrics for quantitative screening.
Saves results to CSV for clustering analysis.

Run this script ONCE per screening session to avoid hitting API rate limits.
"""

import requests
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import time
import warnings
import os
from dotenv import load_dotenv
warnings.filterwarnings('ignore')

# Load environment variables from .env
load_dotenv()


class DataCollector:
    """Fetch and calculate metrics for stock screening"""
    
    def __init__(self, fmp_api_key: str, alpha_vantage_key: str = None):
        self.fmp_api_key = fmp_api_key
        self.alpha_vantage_key = alpha_vantage_key  # Deprecated, kept for backwards compatibility
        self.fmp_base_url = "https://financialmodelingprep.com/stable"
        
        # Track API calls
        self.fmp_calls = 0
        
    def _make_fmp_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        """Make FMP API request with rate limit tracking"""
        if params is None:
            params = {}
        params['apikey'] = self.fmp_api_key
        url = f"{self.fmp_base_url}{endpoint}"
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            self.fmp_calls += 1
            print(f"  [FMP calls: {self.fmp_calls}/250]")
            return response.json()
        except Exception as e:
            print(f"  ❌ FMP request failed: {e}")
            return None
    
    def _make_av_request(self, params: Dict[str, Any]) -> Optional[Any]:
        """Make yfinance call (replaces Alpha Vantage)"""
        # Deprecated - Alpha Vantage no longer used
        return None
    
    def _fetch_yfinance_metrics(self, symbol: str) -> Dict[str, Any]:
        """Fetch company metrics from yfinance (replaces Alpha Vantage)"""
        try:
            tk = yf.Ticker(symbol)
            info = tk.info or {}
            
            def safe_float(value):
                try:
                    return float(value) if value not in ['None', '', '-', 'N/A', None] else None
                except (ValueError, TypeError):
                    return None
            
            metrics = {
                'pe_ratio': safe_float(info.get('trailingPE') or info.get('forwardPE')),
                'pb_ratio': safe_float(info.get('priceToBook')),
                'dividend_yield': safe_float(info.get('dividendYield')),
                'roe': safe_float(info.get('returnOnEquity')),
                'beta_av': safe_float(info.get('beta')),
                'profit_margin': safe_float(info.get('profitMargins')),
                'expense_ratio': safe_float(info.get('expenseRatio') or info.get('netExpenseRatio') or info.get('totalExpenseRatio')),
                'net_assets': self._parse_net_assets(info.get('totalAssets') or info.get('totalNetAssets') or info.get('netAssets') or info.get('nav')),
            }
            return metrics
        except Exception as e:
            print(f"  ❌ yfinance fetch failed for {symbol}: {e}")
            return {}
    
    def _parse_net_assets(self, value) -> Optional[float]:
        """Parse net assets with suffix (K, M, B, T)"""
        if value is None or value == '' or value == '-':
            return None
        
        try:
            value = str(value).strip().upper()
            multipliers = {'K': 1e3, 'M': 1e6, 'B': 1e9, 'T': 1e12}
            
            for suffix, multiplier in multipliers.items():
                if value.endswith(suffix):
                    return float(value[:-1]) * multiplier
            
            return float(value)
        except (ValueError, TypeError, AttributeError):
            return None
    
    # ========== DATA FETCHING ==========
    
    def get_historical_prices(self, symbol: str, years: int = 3) -> pd.DataFrame:
        """Fetch historical prices using yfinance

        This replaces the Alpha Vantage implementation with `yfinance.download`.
        Returns a DataFrame with columns: `date, open, high, low, close, volume`.
        """
        end = datetime.now()
        start = end - timedelta(days=365 * years)

        try:
            data = yf.download(
                symbol,
                start=start.strftime('%Y-%m-%d'),
                end=end.strftime('%Y-%m-%d'),
                progress=False,
                auto_adjust=False,
            )
        except Exception as e:
            print(f"  ❌ yfinance download failed for {symbol}: {e}")
            return pd.DataFrame()

        if data is None or data.empty:
            print(f"  ⚠️  No price data for {symbol}")
            return pd.DataFrame()

        # Reset index and normalize column names
        df = data.reset_index().rename(
            columns={
                'Date': 'date',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Adj Close': 'adj_close',
                'Volume': 'volume',
            }
        )

        # yfinance may return a MultiIndex columns (Price, Ticker) for multi-ticker downloads.
        # For single-ticker downloads the first level contains the price names we care about.
        if getattr(df.columns, 'nlevels', 1) > 1:
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

        # Ensure datetime and proper dtypes
        df['date'] = pd.to_datetime(df['date'])
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Keep the standard set of columns
        keep_cols = [c for c in ['date', 'open', 'high', 'low', 'close', 'volume'] if c in df.columns]
        df = df[keep_cols]

        return df.sort_values('date').reset_index(drop=True)
    
    def get_company_profile(self, symbol: str) -> Dict[str, Any]:
        """Fetch company profile from FMP"""
        endpoint = "/profile"
        params = {'symbol': symbol}
        
        data = self._make_fmp_request(endpoint, params)
        if data and isinstance(data, list) and len(data) > 0:
            profile = data[0]
            return {
                'market_cap': profile.get('mktCap'),
                'sector': profile.get('sector'),
                'industry': profile.get('industry')
            }
        return {}
    
    def get_company_metrics(self, symbol: str) -> Dict:
        """Fetch company metrics from yfinance (replaces Alpha Vantage)"""
        return self._fetch_yfinance_metrics(symbol)
    
    def get_risk_free_rate(self) -> float:
        """Fetch 10-year Treasury rate from FMP"""
        to_date = datetime.now().strftime('%Y-%m-%d')
        from_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        
        endpoint = "/treasury-rates"
        params = {'from': from_date, 'to': to_date}
        
        data = self._make_fmp_request(endpoint, params)
        if data and isinstance(data, list) and len(data) > 0:
            year_10 = data[0].get('year10', 4.0)
            return year_10 / 100
        return 0.04
    
    # ========== METRIC CALCULATIONS ==========
    
    def calculate_returns(self, prices: pd.DataFrame) -> pd.Series:
        """Calculate daily returns"""
        if prices is None or prices.empty or 'close' not in prices.columns:
            return pd.Series()
        return prices['close'].pct_change().dropna()
    
    def calculate_annualized_return(self, prices: pd.DataFrame) -> Optional[float]:
        """Calculate CAGR"""
        if prices is None or prices.empty or len(prices) < 2:
            return None
        
        start_price = prices['close'].iloc[0]
        end_price = prices['close'].iloc[-1]
        
        if pd.isna(start_price) or pd.isna(end_price) or start_price <= 0:
            return None
        
        n_years = (prices['date'].iloc[-1] - prices['date'].iloc[0]).days / 365.25
        if n_years > 0:
            return (end_price / start_price) ** (1 / n_years) - 1
        return None
    
    def calculate_volatility(self, returns: pd.Series) -> Optional[float]:
        """Calculate annualized volatility"""
        if returns is None or returns.empty or len(returns) < 2:
            return None
        
        vol = returns.std()
        if pd.isna(vol):
            return None
        return vol * np.sqrt(252)
    
    def calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float) -> Optional[float]:
        """Calculate Sharpe ratio"""
        if returns is None or returns.empty or len(returns) < 2:
            return None
        if risk_free_rate is None:
            risk_free_rate = 0.04
        
        excess_returns = returns - (risk_free_rate / 252)
        mean_excess = excess_returns.mean()
        std_excess = excess_returns.std()
        
        if pd.isna(mean_excess) or pd.isna(std_excess) or std_excess == 0:
            return None
        
        return mean_excess / std_excess * np.sqrt(252)
    
    def calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float) -> Optional[float]:
        """Calculate Sortino ratio"""
        if returns is None or returns.empty or len(returns) < 2:
            return None
        if risk_free_rate is None:
            risk_free_rate = 0.04
        
        excess_returns = returns - (risk_free_rate / 252)
        downside_returns = excess_returns[excess_returns < 0]
        
        if downside_returns.empty:
            return None
        
        mean_excess = excess_returns.mean()
        downside_std = downside_returns.std()
        
        if pd.isna(mean_excess) or pd.isna(downside_std) or downside_std == 0:
            return None
        
        return mean_excess / downside_std * np.sqrt(252)
    
    def calculate_max_drawdown(self, prices: pd.DataFrame) -> Optional[float]:
        """Calculate maximum drawdown"""
        if prices is None or prices.empty or len(prices) < 2:
            return None
        
        returns = self.calculate_returns(prices)
        if returns.empty:
            return None
        
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        min_dd = drawdown.min()
        return min_dd if not pd.isna(min_dd) else None
    
    def calculate_cvar(self, returns: pd.Series, confidence: float = 0.95) -> Optional[float]:
        """Calculate CVaR (Expected Shortfall)"""
        if returns is None or returns.empty or len(returns) < 2:
            return None
        
        var_threshold = returns.quantile(1 - confidence)
        if pd.isna(var_threshold):
            return None
        
        tail_returns = returns[returns <= var_threshold]
        if tail_returns.empty:
            return None
        
        cvar = tail_returns.mean()
        return cvar if not pd.isna(cvar) else None
    
    def calculate_beta(self, stock_prices: pd.DataFrame, benchmark_prices: pd.DataFrame) -> Optional[float]:
        """Calculate beta vs benchmark"""
        if (stock_prices is None or stock_prices.empty or 
            benchmark_prices is None or benchmark_prices.empty):
            return None
        
        merged = pd.merge(
            stock_prices[['date', 'close']],
            benchmark_prices[['date', 'close']],
            on='date',
            suffixes=('_stock', '_bench')
        )
        
        if merged.empty or len(merged) < 2:
            return None
        
        stock_returns = merged['close_stock'].pct_change().dropna()
        bench_returns = merged['close_bench'].pct_change().dropna()
        
        if stock_returns.empty or bench_returns.empty:
            return None
        
        bench_var = bench_returns.var()
        if pd.isna(bench_var) or bench_var == 0:
            return None
        
        covariance = np.cov(stock_returns, bench_returns)[0][1]
        if pd.isna(covariance):
            return None
        
        return covariance / bench_var
    
    # ========== MAIN COLLECTION FUNCTION ==========
    
    def collect_stock_data(self, symbol: str, benchmark_prices: pd.DataFrame, 
                          risk_free_rate: float, years: int = 3) -> Dict[str, Any]:
        """Collect all metrics for a single stock"""
        print(f"\n📊 Collecting data for {symbol}...")
        
        result: Dict[str, Any] = {'ticker': symbol}
        
        # Fetch data
        prices = self.get_historical_prices(symbol, years)
        profile = self.get_company_profile(symbol)
        metrics = self.get_company_metrics(symbol)  # Now uses yfinance instead of Alpha Vantage
        
        # Add profile data
        result.update(profile)
        result.update(metrics)
        
        # Calculate metrics from price data
        if not prices.empty:
            returns = self.calculate_returns(prices)
            
            result['annualized_return'] = self.calculate_annualized_return(prices)
            result['volatility'] = self.calculate_volatility(returns)
            result['sharpe_ratio'] = self.calculate_sharpe_ratio(returns, risk_free_rate)
            result['sortino_ratio'] = self.calculate_sortino_ratio(returns, risk_free_rate)
            result['max_drawdown'] = self.calculate_max_drawdown(prices)
            result['cvar_95'] = self.calculate_cvar(returns, 0.95)
            
            if not benchmark_prices.empty:
                result['beta_calculated'] = self.calculate_beta(prices, benchmark_prices)
        
        print(f"  ✓ {symbol} complete")
        return result
    
    def collect_universe(self, tickers: List[str], benchmark: str = 'SPY', 
                        years: int = 3) -> pd.DataFrame:
        """Collect data for entire stock universe"""
        print("=" * 60)
        print("QUANTITATIVE SCREENING - DATA COLLECTION")
        print("=" * 60)
        print(f"Universe size: {len(tickers)} stocks")
        print(f"Benchmark: {benchmark}")
        print(f"Historical period: {years} years")
        print(f"\nEstimated API calls:")
        print(f"  FMP: ~{len(tickers)} calls (company profiles)")
        print(f"  yfinance: {len(tickers)} calls (metrics - free, no rate limits)")
        print("=" * 60)
        
        # Get risk-free rate once
        print("\n📈 Fetching risk-free rate...")
        risk_free_rate = self.get_risk_free_rate()
        print(f"  ✓ Risk-free rate: {risk_free_rate:.2%}")
        
        # Get benchmark prices once
        print(f"\n📈 Fetching {benchmark} benchmark data...")
        benchmark_prices = self.get_historical_prices(benchmark, years)
        print(f"  ✓ Benchmark data: {len(benchmark_prices)} days")
        
        # Collect data for each stock
        all_data = []
        failed_tickers = []
        
        for i, ticker in enumerate(tickers, 1):
            print(f"\n[{i}/{len(tickers)}]", end=" ")
            try:
                stock_data = self.collect_stock_data(ticker, benchmark_prices, risk_free_rate, years)
                all_data.append(stock_data)
            except Exception as e:
                print(f"  ❌ Failed: {e}")
                failed_tickers.append(ticker)
        
        # Create DataFrame
        df = pd.DataFrame(all_data)
        
        # Summary
        print("\n" + "=" * 60)
        print("COLLECTION COMPLETE")
        print("=" * 60)
        print(f"✓ Successfully collected: {len(all_data)} stocks")
        if failed_tickers:
            print(f"❌ Failed: {len(failed_tickers)} stocks - {failed_tickers}")
        print(f"\nTotal API calls:")
        print(f"  FMP: {self.fmp_calls}")
        print(f"  yfinance: {len(tickers)} (free, no rate limits)")
        print("=" * 60)
        
        return df


# ========== MAIN EXECUTION ==========

if __name__ == "__main__":
    # ===== CONFIGURATION =====
    
    
    # Your stock universe (40-45 tickers from idea generation)
    TICKERS = ["QQQ"]
    
    BENCHMARK = 'SPY'
    YEARS = 3  # Historical data period
    OUTPUT_FILE = 'stock_metrics.csv'
    
    # Load API keys from environment
    FMP_API_KEY = os.getenv('FMP_API_KEY')
    
    if not FMP_API_KEY:
        raise ValueError("FMP_API_KEY must be set in .env file")
    
    # ===== RUN COLLECTION =====
    # Alpha Vantage key no longer needed (using yfinance instead)
    collector = DataCollector(FMP_API_KEY)
    
    # Collect data
    df = collector.collect_universe(TICKERS, BENCHMARK, YEARS)
    
    # Save to CSV
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n💾 Data saved to: {OUTPUT_FILE}")
    print(f"\n📋 Columns: {list(df.columns)}")
    print(f"\n🔍 Preview:")
    print(df[['ticker', 'annualized_return', 'volatility', 'sharpe_ratio', 
              'sortino_ratio', 'roe', 'dividend_yield']].head())
    
    print("\n✅ Ready for clustering! Run clustering_analysis.py next.")
    print(f"\n📝 Note: Metrics now use yfinance (no Alpha Vantage dependency).")
    print(f"⚠️  Remember to manually add ESG Risk Ratings to {OUTPUT_FILE} before clustering!")
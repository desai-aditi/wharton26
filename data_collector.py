"""
Data Collector Script
Fetches historical data and calculates metrics for quantitative screening.
Saves results to CSV for clustering analysis.

Run this script ONCE per screening session to avoid hitting API rate limits.
"""

import requests
import pandas as pd
import numpy as np
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
    
    def __init__(self, fmp_api_key: str, alpha_vantage_key: str):
        self.fmp_api_key = fmp_api_key
        self.alpha_vantage_key = alpha_vantage_key
        self.fmp_base_url = "https://financialmodelingprep.com/stable"
        self.av_base_url = "https://www.alphavantage.co/query"
        
        # Track API calls
        self.fmp_calls = 0
        self.av_calls = 0
        
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
        """Make Alpha Vantage request with rate limit tracking"""
        params['apikey'] = self.alpha_vantage_key
        
        try:
            response = requests.get(self.av_base_url, params=params)
            response.raise_for_status()
            self.av_calls += 1
            print(f"  [AV calls: {self.av_calls}/25]")
            time.sleep(12)  # AV rate limit: 5 calls/min, so wait 12 sec between calls
            return response.json()
        except Exception as e:
            print(f"  ❌ AV request failed: {e}")
            return None
    
    # ========== DATA FETCHING ==========
    
    def get_historical_prices(self, symbol: str, years: int = 3) -> pd.DataFrame:
        """Fetch historical prices from Alpha Vantage"""
        params = {
            'function': 'TIME_SERIES_DAILY',
            'symbol': symbol,
            'outputsize': 'full',  # Get full 20+ years, then we'll filter
            'datatype': 'json'
        }
        
        data = self._make_av_request(params)
        
        if not data or 'Time Series (Daily)' not in data:
            print(f"  ⚠️  No price data for {symbol}")
            return pd.DataFrame()
        
        # Convert nested dict to DataFrame
        time_series = data['Time Series (Daily)']
        df = pd.DataFrame.from_dict(time_series, orient='index')
        
        # Rename columns (AV returns: '1. open', '2. high', etc.)
        df.columns = ['open', 'high', 'low', 'close', 'volume']
        
        # Convert index to datetime and sort
        df.index = pd.to_datetime(df.index)
        df = df.sort_index().reset_index()
        df.rename(columns={'index': 'date'}, inplace=True)
        
        # Convert price columns to float (they come as strings)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Filter to requested time period
        to_date = datetime.now()
        from_date = to_date - timedelta(days=365 * years)
        df = df[df['date'] >= from_date]
    
        return df
    
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
    
    def get_alpha_vantage_data(self, symbol: str) -> Dict:
        """
        Fetch financial data from Alpha Vantage
        Uses OVERVIEW for stocks, ETF_PROFILE for ETFs
        """
        # List of known ETFs (you can expand this)
        ETF_LIST = [
            'BND', 'AGG', 'TLT', 'IEF', 'SHV', 'BIL',  # Bonds
            'QQQ', 'VGT', 'SPY', 'VTI',  # Tech/Broad Market
            'VHT', 'XLV',  # Healthcare
            'VNQ',  # Real Estate
            'VEA', 'VWO', 'IEMG',  # International
            'VOX',  # Communications
            'ICLN', 'TAN', 'QCLN',  # Clean Energy
        ]
        
        is_etf = symbol in ETF_LIST
        
        if is_etf:
            return self._get_etf_profile(symbol) 
        else:
            return self._get_stock_overview(symbol)

    def _get_stock_overview(self, symbol: str) -> Dict:
        """Fetch stock data using OVERVIEW"""
        params = {'function': 'OVERVIEW', 'symbol': symbol}
        
        data = self._make_av_request(params)
        if not data or 'Symbol' not in data:
            return {}
        
        def safe_float(value):
            try:
                return float(value) if value not in ['None', '', '-', 'N/A'] else None
            except (ValueError, TypeError):
                return None
        
        return {
            'pe_ratio': safe_float(data.get('PERatio')),
            'pb_ratio': safe_float(data.get('PriceToBookRatio')),
            'dividend_yield': safe_float(data.get('DividendYield')),
            'roe': safe_float(data.get('ReturnOnEquityTTM')),
            'beta_av': safe_float(data.get('Beta')),
            'profit_margin': safe_float(data.get('ProfitMargin')),
            'sector': data.get('Sector'),
            'industry': data.get('Industry')
        }

    def _get_etf_profile(self, symbol: str) -> Dict:
        """Fetch ETF data using ETF_PROFILE"""
        params = {'function': 'ETF_PROFILE', 'symbol': symbol}
        
        data = self._make_av_request(params)
        if not data:
            return {}
        
        def safe_float(value):
            try:
                return float(value) if value not in ['None', '', '-', 'N/A'] else None
            except (ValueError, TypeError):
                return None
        
        # ETFs don't have ROE, PE, PB - set to None
        # But they have dividend yield and expense ratio
        return {
            'pe_ratio': None,
            'pb_ratio': None,
            'dividend_yield': safe_float(data.get('dividend_yield')),
            'roe': None,  # ETFs don't have ROE
            'beta_av': None,  # Need to calculate from price data
            'profit_margin': None,
            'expense_ratio': safe_float(data.get('net_expense_ratio')),  # ETF-specific
            'net_assets': safe_float(data.get('net_assets')),  # ETF-specific
            'sector': 'ETF',  # Tag as ETF
            'industry': data.get('asset_class', 'ETF')
        }
    
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
        av_data = self.get_alpha_vantage_data(symbol)
        
        # Add profile data
        result.update(profile)
        result.update(av_data)
        
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
        print(f"  FMP: ~{len(tickers) * 3} calls (price, profile, treasury)")
        print(f"  Alpha Vantage: {len(tickers)} calls")
        print(f"\n⚠️  Alpha Vantage will take ~{len(tickers) * 12 / 60:.1f} minutes due to rate limits")
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
        print(f"  Alpha Vantage: {self.av_calls}")
        print("=" * 60)
        
        return df


# ========== MAIN EXECUTION ==========

if __name__ == "__main__":
    # ===== CONFIGURATION =====
    
    
    # Your stock universe (40-45 tickers from idea generation)
    TICKERS = [
        # Core candidates
        'QQQ']
    
    BENCHMARK = 'SPY'
    YEARS = 3  # Historical data period
    OUTPUT_FILE = 'stock_metrics.csv'
    
    # Load API keys from environment
    FMP_API_KEY = os.getenv('FMP_API_KEY')
    ALPHA_VANTAGE_KEY = os.getenv('ALPHA_VANTAGE_KEY')
    
    if not FMP_API_KEY or not ALPHA_VANTAGE_KEY:
        raise ValueError("FMP_API_KEY and ALPHA_VANTAGE_KEY must be set in .env file")
    
    # ===== RUN COLLECTION =====
    collector = DataCollector(FMP_API_KEY, ALPHA_VANTAGE_KEY)
    
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
    print(f"\n⚠️  Remember to manually add ESG scores to {OUTPUT_FILE} before clustering!")
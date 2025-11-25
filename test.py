import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Optional, Any
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

class FMPClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://financialmodelingprep.com/stable"

    def make_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None):
        if params is None:
            params = {}
        params['apikey'] = self.api_key
        url = f"{self.base_url}{endpoint}"
        response = requests.get(url, params=params)
        try:
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Request failed: {e}")
            return None

    def get_historical_prices(self, symbol: str, from_date: str, to_date: str) -> pd.DataFrame:
        """Fetch historical EOD prices as a DataFrame."""
        endpoint = "/historical-price-eod/full"
        params = {'symbol': symbol, 'from': from_date, 'to': to_date}
        data = self.make_request(endpoint, params)
        if not data or not isinstance(data, list):
            print(f"No price data for {symbol}")
            return pd.DataFrame()
        df = pd.DataFrame(data)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        return df.sort_values('date').reset_index(drop=True)

# Usage example
if __name__ == "__main__":
    api_key = os.getenv('FMP_API_KEY')
    if not api_key:
        raise ValueError("FMP_API_KEY not found in environment variables")
    client = FMPClient(api_key=api_key)
    to_date = datetime.now().strftime('%Y-%m-%d')
    from_date = (datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d')
    prices = client.get_historical_prices("AAPL", from_date, to_date)
    print(prices.head())

"""
Example: How to create expected_returns.csv for Step 9

This shows the process of adjusting returns based on qualitative conviction.
You'll create this CSV manually or with a simple script.
"""

import pandas as pd

# Step 1: Load your stock_metrics.csv to get historical returns
stock_metrics = pd.read_csv('stock_metrics.csv')

# Step 2: Start with all stocks at their historical returns (no adjustment)
expected_returns = stock_metrics[['ticker', 'annualized_return']].copy()
expected_returns.columns = ['ticker', 'historical_return']

# Step 3: Initialize adjusted_return = historical_return (default: no change)
expected_returns['adjusted_return'] = expected_returns['historical_return']
expected_returns['rationale'] = 'No adjustment - using historical return'

# Step 4: Make your 5-7 conviction-based adjustments
# (Replace these with YOUR actual convictions from qualitative analysis!)

adjustments = {
    'NVDA': {
        'adjustment': -0.20,  # Reduce by 20% (45% → 25%)
        'rationale': 'AI hype normalization expected - current valuation unsustainable, expect mean reversion'
    },
    'AAPL': {
        'adjustment': -0.03,  # Reduce by 3%
        'rationale': 'Services growth slowing - iPhone upgrade cycles lengthening, China headwinds'
    },
    'PG': {
        'adjustment': 0.02,  # Increase by 2%
        'rationale': 'Inflation hedge with pricing power - defensive moat in uncertain macro environment'
    },
    'ICLN': {
        'adjustment': 0.05,  # Increase by 5%
        'rationale': 'Clean energy policy tailwinds - IRA + global decarbonization trends accelerating'
    },
}

# Apply adjustments
for ticker, adjustment_info in adjustments.items():
    mask = expected_returns['ticker'] == ticker
    if mask.any():
        historical = expected_returns.loc[mask, 'historical_return'].values[0]
        adjusted = historical + adjustment_info['adjustment']
        
        expected_returns.loc[mask, 'adjusted_return'] = adjusted
        expected_returns.loc[mask, 'rationale'] = adjustment_info['rationale']
        
        print(f"{ticker}: {historical:.1%} → {adjusted:.1%} ({adjustment_info['adjustment']:+.1%})")

# Step 5: Save to CSV
expected_returns.to_csv('expected_returns.csv', index=False)
print("\n✅ Saved: expected_returns.csv")
print(f"✅ Total adjustments made: {len(adjustments)}")

# Preview
print("\n📊 Preview of adjusted returns:")
print(expected_returns[expected_returns['adjusted_return'] != expected_returns['historical_return']])
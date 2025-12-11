"""
Bond Ladder Construction - Phase 4 Step 10
Builds a laddered T-bond portfolio for Connor Barwin's needs.

Strategy:
- Total allocation: $62,500 (12.5% of $500k)
- 5-7 bonds with staggered maturities
- Target yield: 4-5% average
- Maturities aligned with Connor's timeline and drawdown needs
"""

import pandas as pd

# ========== BOND LADDER STRATEGY ==========

# Connor's Timeline (Simulated):
# - Years 1-2 (2026-2028): Growth phase, no drawdowns
# - Year 3+ (2029+): $10k annual drawdowns begin
# - Year 10 (2036): Major project funding needed

# Target Maturities: 2027, 2029, 2031, 2034, 2036
# This provides liquidity every 2-3 years

BOND_LADDER = [
    {
        'bond_name': 'US Treasury 2026',
        'maturity': '2026-11-15',
        'coupon': 6.50,  # Approximate current yields
        'allocation': 10000,
        'rationale': 'Short-term liquidity - matures before first drawdown need'
    },
    {
        'bond_name': 'US Treasury 2028',
        'maturity': '2028-11-15',
        'coupon': 5.25,
        'allocation': 10000,
        'rationale': 'Aligns with Year 3 drawdown timing - provides cash when needed'
    },
    {
        'bond_name': 'US Treasury 2029',
        'maturity': '2029-02-15',
        'coupon': 5.25,
        'allocation': 12500,
        'rationale': 'Mid-term stability - covers Years 4-5 drawdown needs'
    },
    {
        'bond_name': 'US Treasury 2030',
        'maturity': '2030-05-15',
        'coupon': 6.25,
        'allocation': 10000,
        'rationale': 'Late-stage liquidity - supports Years 7-8 needs'
    },
    {
        'bond_name': 'US Treasury 2031',
        'maturity': '2031-02-15',
        'coupon': 5.375,
        'allocation': 10000,
        'rationale': 'Terminal maturity - aligns with 2036 community project funding'
    },
    {
        'bond_name': 'US Treasury 3026',
        'maturity': '2036-02-15',
        'coupon': 4.5,
        'allocation': 10000,
        'rationale': 'Terminal maturity - aligns with 2036 community project funding'
    }
]

# ========== VALIDATION ==========

def validate_bond_ladder(bonds):
    """Validate bond ladder meets requirements"""
    
    total_allocation = sum(b['allocation'] for b in bonds)
    weighted_yield = sum(b['allocation'] * b['coupon'] for b in bonds) / total_allocation
    
    print("=" * 70)
    print("BOND LADDER VALIDATION - Phase 4 Step 10")
    print("=" * 70)
    print(f"\n📊 Bond Ladder Structure:")
    print(f"  Number of bonds: {len(bonds)}")
    print(f"  Total allocation: ${total_allocation:,}")
    print(f"  Target allocation: $62,500")
    print(f"  ✓ Match: {'Yes' if total_allocation == 62500 else 'No'}")
    
    print(f"\n💰 Yield Analysis:")
    print(f"  Weighted average yield: {weighted_yield:.2f}%")
    print(f"  Target range: 4.0-5.0%")
    print(f"  ✓ Within range: {'Yes' if 4.0 <= weighted_yield <= 5.0 else 'No'}")
    
    print(f"\n📅 Maturity Ladder:")
    for bond in bonds:
        annual_income = bond['allocation'] * (bond['coupon'] / 100)
        print(f"  {bond['maturity'][:4]}: ${bond['allocation']:>6,} @ {bond['coupon']:.2f}% = ${annual_income:>6,.0f}/yr")
    
    print(f"\n📈 Annual Income from Bonds:")
    total_annual_income = sum(b['allocation'] * (b['coupon'] / 100) for b in bonds)
    print(f"  Total coupon income: ${total_annual_income:,.0f}/year")
    print(f"  Covers drawdown need: {'Yes' if total_annual_income >= 10000 else 'Partial'}")
    print(f"  Gap to cover: ${max(0, 10000 - total_annual_income):,.0f} (from dividends/growth)")
    
    print("\n" + "=" * 70)
    
    return total_allocation, weighted_yield

# ========== CREATE OUTPUT CSV ==========

def save_bond_ladder(bonds, output_file='bond_ladder.csv'):
    """Save bond ladder to CSV"""
    
    df = pd.DataFrame(bonds)
    
    # Add calculated fields
    df['annual_income'] = df['allocation'] * (df['coupon'] / 100)
    df['pct_of_portfolio'] = (df['allocation'] / 500000) * 100
    df['pct_of_bonds'] = (df['allocation'] / 62500) * 100
    
    # Reorder columns
    df = df[['bond_name', 'maturity', 'coupon', 'allocation', 
             'annual_income', 'pct_of_portfolio', 'pct_of_bonds', 'rationale']]
    
    df.to_csv(output_file, index=False)
    print(f"💾 Saved: {output_file}")
    
    return df

# ========== MAIN EXECUTION ==========

if __name__ == "__main__":
    
    print("\n🏦 Building Bond Ladder for Connor Barwin Portfolio\n")
    
    # Validate strategy
    total, avg_yield = validate_bond_ladder(BOND_LADDER)
    
    # Save to CSV
    df = save_bond_ladder(BOND_LADDER)
    
    # Display summary
    print("\n📋 Bond Ladder Details:")
    print(df.to_string(index=False))
    
    print("\n" + "=" * 70)
    print("MANUAL ALLOCATIONS SUMMARY")
    print("=" * 70)
    print(f"Bond Ladder:    $62,500  (12.5% of portfolio)")
    print(f"Cash Reserve:   $10,000  ( 2.0% of portfolio)")
    print(f"─" * 70)
    print(f"Total Manual:   $72,500  (14.5% of portfolio)")
    print(f"\nRemaining for optimization: $427,500 (85.5%)")
    print("=" * 70)
    
    print("\n✅ NEXT STEPS:")
    print("  1. Review bond_ladder.csv")
    print("  2. Verify bonds are available in simulator")
    print("  3. Document rationale for each bond choice")
    print("  4. Proceed to Step 11: Calculate Target Bucket Sizes")
    print("\n💡 TIP: Laddering provides liquidity + reinvestment flexibility")
    print("=" * 70)
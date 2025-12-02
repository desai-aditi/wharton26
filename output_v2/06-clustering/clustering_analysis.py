"""
Clustering Analysis Script - Simplified for Phase 2 Step 6
Loads stock metrics and performs K-Means clustering to classify stocks.
Creates visualizations for team review.

Run this AFTER:
1. data_collector.py has been run
2. ESG scores have been manually added to stock_metrics.csv

NOTE: ESG scores should be ESG Risk Ratings from Sustainalytics/Morningstar
where LOWER is BETTER (0-10 = Negligible, 10-20 = Low, 20-30 = Medium, etc.)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')


class ClusteringAnalysis:
    """K-Means clustering for stock classification"""
    
    def __init__(self, data_file: str):
        self.data_file = data_file
        self.df = None
        self.scaled_features = None
        self.scaler = None
        
    def load_data(self):
        """Load stock metrics CSV"""
        print("=" * 60)
        print("CLUSTERING ANALYSIS - Phase 2 Step 6")
        print("=" * 60)
        print(f"Loading: {self.data_file}")
        
        self.df = pd.read_csv(self.data_file)
        print(f"✓ Loaded {len(self.df)} stocks")
        
        # Check for ESG scores
        if 'esg_score' not in self.df.columns:
            print("\n⚠️  WARNING: 'esg_score' column not found!")
            print("   Add ESG Risk Ratings before running clustering.")
            print("   (Lower = better: Negligible 0-10, Low 10-20, Medium 20-30, etc.)")
            raise ValueError("Missing esg_score column")
        
        return self
    
    def prepare_features(self, features: list):
        """Standardize features using z-score normalization"""
        print(f"\n📊 Preparing features...")
        print(f"Features: {features}")
        
        # Check for missing columns
        missing = [f for f in features if f not in self.df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        
        # Check for missing values
        feature_data = self.df[features]
        missing_vals = feature_data.isnull().sum()
        
        if missing_vals.sum() > 0:
            print("\n⚠️  Missing values detected:")
            print(missing_vals[missing_vals > 0])
            raise ValueError("Fix missing values before clustering")
        
        print("✓ No missing values")
        
        # Standardize (z-score normalization)
        self.scaler = StandardScaler()
        self.scaled_features = self.scaler.fit_transform(feature_data)
        print(f"✓ Features standardized: {self.scaled_features.shape}")
        
        return self
    
    def run_clustering(self, n_clusters: int = 4):
        """Perform K-Means clustering with k clusters"""
        print(f"\n🎯 Running K-Means (k={n_clusters})...")
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.df['cluster'] = kmeans.fit_predict(self.scaled_features)
        
        # Calculate silhouette score (quality metric)
        silhouette = silhouette_score(self.scaled_features, self.df['cluster'])
        
        print(f"✓ Clustering complete")
        print(f"  Silhouette score: {silhouette:.3f} (higher is better)")
        print(f"\n📊 Cluster sizes:")
        print(self.df['cluster'].value_counts().sort_index())
        
        return self
    
    def analyze_clusters(self, features: list):
        """Generate cluster profiles showing average metrics"""
        print(f"\n📈 CLUSTER PROFILES")
        print("=" * 60)
        
        # Calculate mean per cluster
        profiles = self.df.groupby('cluster')[features].mean().round(3)
        print("\nAverage metrics per cluster:")
        print(profiles)
        
        # Save profiles
        profiles.to_csv('cluster_profiles.csv')
        print(f"\n💾 Saved: cluster_profiles.csv")
        
        # Create heatmap
        plt.figure(figsize=(12, 6))
        sns.heatmap(profiles.T, annot=True, fmt='.2f', cmap='RdYlGn_r',
                   cbar_kws={'label': 'Average Value'})
        plt.title('Cluster Profiles', fontsize=14, fontweight='bold')
        plt.xlabel('Cluster')
        plt.ylabel('Metric')
        plt.tight_layout()
        plt.savefig('cluster_heatmap.png', dpi=300, bbox_inches='tight')
        print(f"💾 Saved: cluster_heatmap.png")
        plt.close()
        
        # Show stocks per cluster
        print(f"\n📋 STOCKS BY CLUSTER")
        print("=" * 60)
        for cluster_id in sorted(self.df['cluster'].unique()):
            stocks = self.df[self.df['cluster'] == cluster_id]['ticker'].tolist()
            print(f"\nCluster {cluster_id} ({len(stocks)} stocks):")
            print(f"  {', '.join(stocks)}")
        
        return profiles
    
    def create_visualizations(self):
        """Create 2D scatter plots for cluster visualization"""
        print(f"\n🎨 Creating visualizations...")
        
        # Plot 1: Sharpe vs Volatility
        plt.figure(figsize=(10, 6))
        for cluster_id in sorted(self.df['cluster'].unique()):
            cluster_data = self.df[self.df['cluster'] == cluster_id]
            plt.scatter(cluster_data['volatility'], cluster_data['sharpe_ratio'],
                       label=f'Cluster {cluster_id}', s=100, alpha=0.6)
            
            # Add labels
            for _, row in cluster_data.iterrows():
                plt.annotate(row['ticker'],
                           (row['volatility'], row['sharpe_ratio']),
                           fontsize=8, alpha=0.7)
        
        plt.xlabel('Volatility (Annualized)', fontsize=12)
        plt.ylabel('Sharpe Ratio', fontsize=12)
        plt.title('Sharpe Ratio vs Volatility by Cluster', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('cluster_sharpe_volatility.png', dpi=300, bbox_inches='tight')
        print(f"  ✓ cluster_sharpe_volatility.png")
        plt.close()
        
        # Plot 2: Sortino vs Max Drawdown
        plt.figure(figsize=(10, 6))
        for cluster_id in sorted(self.df['cluster'].unique()):
            cluster_data = self.df[self.df['cluster'] == cluster_id]
            plt.scatter(cluster_data['max_drawdown'], cluster_data['sortino_ratio'],
                       label=f'Cluster {cluster_id}', s=100, alpha=0.6)
            
            for _, row in cluster_data.iterrows():
                plt.annotate(row['ticker'],
                           (row['max_drawdown'], row['sortino_ratio']),
                           fontsize=8, alpha=0.7)
        
        plt.xlabel('Max Drawdown', fontsize=12)
        plt.ylabel('Sortino Ratio', fontsize=12)
        plt.title('Sortino Ratio vs Max Drawdown by Cluster', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('cluster_sortino_drawdown.png', dpi=300, bbox_inches='tight')
        print(f"  ✓ cluster_sortino_drawdown.png")
        plt.close()
        
        print(f"\n✅ Visualizations complete!")
    
    def save_results(self, output_file: str = 'clustered_stocks.csv'):
        """Save clustered data with assignments"""
        self.df.to_csv(output_file, index=False)
        print(f"\n💾 Saved: {output_file}")
        return output_file


# ========== MAIN EXECUTION ==========

if __name__ == "__main__":
    # Configuration
    INPUT_FILE = 'stock_metrics.csv'
    OUTPUT_FILE = 'clustered_stocks.csv'
    
    # Features for clustering (from Phase 2 Step 6 spec)
    # Note: ROE excluded because ETFs don't have ROE
    FEATURES = [
        'sortino_ratio',
        'sharpe_ratio',
        'volatility',
        'max_drawdown',
        'dividend_yield',
        'esg_score',  # ESG Risk Rating (lower = better)
        'beta'
    ]
    
    N_CLUSTERS = 4  # Core, Growth, Impact, Reject
    
    # Run clustering pipeline
    print("\n🚀 Starting clustering...\n")
    
    analyzer = ClusteringAnalysis(INPUT_FILE)
    
    # Pipeline
    analyzer.load_data()
    analyzer.prepare_features(FEATURES)
    analyzer.run_clustering(n_clusters=N_CLUSTERS)
    profiles = analyzer.analyze_clusters(FEATURES)
    analyzer.create_visualizations()
    analyzer.save_results(OUTPUT_FILE)
    
    # Summary
    print("\n" + "=" * 60)
    print("CLUSTERING COMPLETE!")
    print("=" * 60)
    print("\n📁 Output files:")
    print("  1. clustered_stocks.csv - Stocks with cluster assignments")
    print("  2. cluster_profiles.csv - Average metrics per cluster")
    print("  3. cluster_heatmap.png - Profile comparison")
    print("  4. cluster_sharpe_volatility.png - 2D visualization")
    print("  5. cluster_sortino_drawdown.png - 2D visualization")
    
    print("\n📋 NEXT STEPS (Phase 3 - Qualitative Validation):")
    print("=" * 60)
    print("1. Review cluster_profiles.csv to understand each cluster")
    print("2. Look at scatter plots to see stock groupings")
    print("3. Manually review clusters and assign category labels:")
    print("   - Open clustered_stocks.csv")
    print("   - Add 'category' column with values:")
    print("     • Core (low volatility, dividend payers)")
    print("     • Growth (high Sharpe, growth potential)")
    print("     • Impact (strong ESG, values-aligned)")
    print("     • Reject (poor metrics)")
    print("4. Proceed to qualitative due diligence (Porter's 5 Forces)")
    print("\n💡 TIP: Clusters are preliminary suggestions - use your judgment!")
    print("=" * 60)
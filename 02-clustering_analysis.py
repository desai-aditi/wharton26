"""
Clustering Analysis Script - Updated for Phase 2 Step 6
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
           
        else:
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
        print(f"  Silhouette score: {silhouette:.3f}")
        if silhouette > 0.5:
            print(f"  Quality: ✅ GOOD (>0.5)")
        elif silhouette > 0.3:
            print(f"  Quality: ⚠️  FAIR (0.3-0.5)")
        else:
            print(f"  Quality: ❌ WEAK (<0.3) - consider adjusting features or k")
        
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
        print(profiles.to_string())
        
        # Save profiles
        profiles.to_csv('cluster_profiles.csv')
        print(f"\n💾 Saved: cluster_profiles.csv")
        
        # Create heatmap
        plt.figure(figsize=(14, 8))
        sns.heatmap(profiles.T, annot=True, fmt='.3f', cmap='RdYlGn_r',
                   cbar_kws={'label': 'Average Value'}, linewidths=0.5)
        plt.title('Cluster Profiles - Average Metrics per Cluster', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Cluster', fontsize=12)
        plt.ylabel('Metric', fontsize=12)
        plt.tight_layout()
        plt.savefig('cluster_heatmap.png', dpi=300, bbox_inches='tight')
        print(f"💾 Saved: cluster_heatmap.png")
        plt.close()
        
        # Show stocks per cluster with key metrics
        print(f"\n📋 STOCKS BY CLUSTER (with key metrics)")
        print("=" * 60)
        for cluster_id in sorted(self.df['cluster'].unique()):
            cluster_data = self.df[self.df['cluster'] == cluster_id]
            print(f"\n{'='*60}")
            print(f"CLUSTER {cluster_id} ({len(cluster_data)} stocks)")
            print(f"{'='*60}")
            
            # Show each stock with key metrics
            display_cols = ['ticker', 'sortino_ratio', 'volatility', 
                          'max_drawdown', 'dividend_yield', 'esg_score']
            display_cols = [c for c in display_cols if c in cluster_data.columns]
            
            print(cluster_data[display_cols].to_string(index=False))
        
        return profiles
    
    def identify_boundary_stocks(self, threshold: float = 0.3):
        """Flag stocks on cluster boundaries requiring manual review"""
        print(f"\n🔍 Identifying boundary stocks...")
        print(f"   (Using distance threshold: {threshold})")
        
        # Calculate distance from cluster centers
        kmeans = KMeans(n_clusters=len(self.df['cluster'].unique()), 
                       random_state=42, n_init=10)
        kmeans.fit(self.scaled_features)
        
        # Get distances to assigned cluster center
        distances = []
        for i, row in self.df.iterrows():
            cluster_id = row['cluster']
            center = kmeans.cluster_centers_[cluster_id]
            dist = np.linalg.norm(self.scaled_features[i] - center)
            distances.append(dist)
        
        self.df['cluster_distance'] = distances
        
        # Flag boundary stocks (far from center)
        distance_threshold = np.percentile(distances, 75)  # Top 25%
        self.df['boundary_flag'] = self.df['cluster_distance'] > distance_threshold
        
        boundary_stocks = self.df[self.df['boundary_flag']]
        
        print(f"   ✓ Identified {len(boundary_stocks)} boundary stocks")
        print(f"\n⚠️  Stocks requiring manual review:")
        print("=" * 60)
        
        if len(boundary_stocks) > 0:
            display_cols = ['ticker', 'cluster', 'cluster_distance', 
                          'sortino_ratio', 'volatility']
            display_cols = [c for c in display_cols if c in boundary_stocks.columns]
            print(boundary_stocks[display_cols].sort_values('cluster_distance', 
                  ascending=False).to_string(index=False))
        else:
            print("   (None found)")
        
        return boundary_stocks
    
    def create_visualizations(self):
        """Create 2D scatter plots for cluster visualization"""
        print(f"\n🎨 Creating visualizations...")
        
        # Define pastel color palette - red, orange, green, blue
        pastel_colors = ['#FF9999', '#FFB366', '#99CC99', '#99CCFF']
        colors = pastel_colors[:len(self.df['cluster'].unique())]
        
        # Plot 1: Sharpe vs Volatility (colored by cluster)
        plt.figure(figsize=(12, 8))
        for i, cluster_id in enumerate(sorted(self.df['cluster'].unique())):
            cluster_data = self.df[self.df['cluster'] == cluster_id]
            plt.scatter(cluster_data['volatility'], cluster_data['sharpe_ratio'],
                       label=f'Cluster {cluster_id}', s=120, alpha=0.7,
                       color=colors[i], edgecolors='black', linewidth=0.5)
            
            # Add labels
            for _, row in cluster_data.iterrows():
                plt.annotate(row['ticker'],
                           (row['volatility'], row['sharpe_ratio']),
                           fontsize=9, alpha=0.8, 
                           xytext=(5, 5), textcoords='offset points')
        
        plt.xlabel('Volatility (Annualized)', fontsize=13, fontweight='bold')
        plt.ylabel('Sharpe Ratio', fontsize=13, fontweight='bold')
        plt.title('Stock Classification: Sharpe Ratio vs Volatility', 
                 fontsize=15, fontweight='bold', pad=20)
        plt.legend(fontsize=10, framealpha=0.9)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.savefig('cluster_sharpe_volatility.png', dpi=300, bbox_inches='tight')
        print(f"  ✓ cluster_sharpe_volatility.png")
        plt.close()
        
        # Plot 2: Sortino vs Max Drawdown
        plt.figure(figsize=(12, 8))
        for i, cluster_id in enumerate(sorted(self.df['cluster'].unique())):
            cluster_data = self.df[self.df['cluster'] == cluster_id]
            plt.scatter(cluster_data['max_drawdown'], cluster_data['sortino_ratio'],
                       label=f'Cluster {cluster_id}', s=120, alpha=0.7,
                       color=colors[i], edgecolors='black', linewidth=0.5)
            
            for _, row in cluster_data.iterrows():
                plt.annotate(row['ticker'],
                           (row['max_drawdown'], row['sortino_ratio']),
                           fontsize=9, alpha=0.8,
                           xytext=(5, 5), textcoords='offset points')
        
        plt.xlabel('Max Drawdown', fontsize=13, fontweight='bold')
        plt.ylabel('Sortino Ratio', fontsize=13, fontweight='bold')
        plt.title('Stock Classification: Sortino Ratio vs Max Drawdown', 
                 fontsize=15, fontweight='bold', pad=20)
        plt.legend(fontsize=10, framealpha=0.9)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.savefig('cluster_sortino_drawdown.png', dpi=300, bbox_inches='tight')
        print(f"  ✓ cluster_sortino_drawdown.png")
        plt.close()
        
        # Plot 3: 3D visualization (if quality_score exists)
        if 'quality_score' in self.df.columns:
            from mpl_toolkits.mplot3d import Axes3D
            
            fig = plt.figure(figsize=(12, 9))
            ax = fig.add_subplot(111, projection='3d')
            
            for i, cluster_id in enumerate(sorted(self.df['cluster'].unique())):
                cluster_data = self.df[self.df['cluster'] == cluster_id]
                ax.scatter(cluster_data['volatility'], 
                          cluster_data['sharpe_ratio'],
                          cluster_data['quality_score'],
                          label=f'Cluster {cluster_id}', 
                          s=100, alpha=0.7, color=colors[i])
            
            ax.set_xlabel('Volatility', fontsize=11, fontweight='bold')
            ax.set_ylabel('Sharpe Ratio', fontsize=11, fontweight='bold')
            ax.set_zlabel('Quality Score', fontsize=11, fontweight='bold')
            ax.set_title('3D Stock Classification', fontsize=14, fontweight='bold')
            ax.legend()
            plt.tight_layout()
            plt.savefig('cluster_3d_view.png', dpi=300, bbox_inches='tight')
            print(f"  ✓ cluster_3d_view.png")
            plt.close()
        
        print(f"\n✅ Visualizations complete!")
    
    def save_results(self, output_file: str = 'clustered_stocks.csv'):
        """Save clustered data with assignments"""
        # Reorder columns for readability
        cols_order = ['ticker', 'cluster', 'boundary_flag', 'cluster_distance']
        cols_order += [c for c in self.df.columns if c not in cols_order]
        
        self.df[cols_order].to_csv(output_file, index=False)
        print(f"\n💾 Saved: {output_file}")
        return output_file


# ========== MAIN EXECUTION ==========

if __name__ == "__main__":
    # Configuration
    INPUT_FILE = 'stock_metrics.csv'
    OUTPUT_FILE = 'clustered_stocks.csv'
    
    # Features for clustering (from updated Phase 2 Step 6 spec)
    FEATURES = [
        'sortino_ratio',
        'sharpe_ratio',
        'volatility',
        'max_drawdown',
        'quality_score',     # ← Changed from 'roe' to universal quality metric
        'dividend_yield',
        'esg_score',         # ESG Risk Rating (lower = better)
        'beta'
    ]
    
    N_CLUSTERS = 4  # Core, Growth, Impact, Reject
    
    # Run clustering pipeline
    print("\n🚀 Starting clustering analysis...\n")
    
    analyzer = ClusteringAnalysis(INPUT_FILE)
    
    # Pipeline
    analyzer.load_data()
    analyzer.prepare_features(FEATURES)
    analyzer.run_clustering(n_clusters=N_CLUSTERS)
    profiles = analyzer.analyze_clusters(FEATURES)
    boundary_stocks = analyzer.identify_boundary_stocks()
    analyzer.create_visualizations()
    analyzer.save_results(OUTPUT_FILE)
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ CLUSTERING COMPLETE!")
    print("=" * 60)
    print("\n📁 Output files:")
    print("  1. clustered_stocks.csv - Stocks with cluster assignments + boundary flags")
    print("  2. cluster_profiles.csv - Average metrics per cluster")
    print("  3. cluster_heatmap.png - Profile comparison heatmap")
    print("  4. cluster_sharpe_volatility.png - 2D scatter plot")
    print("  5. cluster_sortino_drawdown.png - 2D scatter plot")
    if 'quality_score' in analyzer.df.columns:
        print("  6. cluster_3d_view.png - 3D visualization")
    
    print("\n" + "=" * 60)
    print("📋 NEXT STEPS (Phase 3 - Step 7: Manual Review)")
    print("=" * 60)
    print("\n1. Review cluster_profiles.csv to understand cluster characteristics:")
    print("   • Low volatility + high Sortino + good dividend → Core candidate")
    print("   • High Sharpe + moderate-high volatility → Growth candidate")
    print("   • Strong ESG (low score) + values alignment → Impact candidate")
    print("   • Poor metrics across board → Reject")
    print("\n2. Examine scatter plots to visualize stock groupings")
    print("\n3. Review boundary stocks (flagged in clustered_stocks.csv)")
    print("   • These stocks are far from cluster centers")
    print("   • Require extra scrutiny in qualitative review")
    print("\n4. Manually add 'category' column to clustered_stocks.csv:")
    print("   • Open file in Excel/Sheets")
    print("   • Add new column: 'category'")
    print("   • Assign preliminary labels based on cluster profiles:")
    print("     - Core (stability, income)")
    print("     - Growth (capital appreciation)")
    print("     - Impact (values alignment)")
    print("     - Reject (poor fit)")
    print("\n5. Proceed to Step 8: Qualitative Due Diligence")
    print("   • Porter's Five Forces analysis")
    print("   • Competitive moat assessment")
    print("   • Management quality review")
    print("\n💡 REMEMBER:")
    print("   Clusters are AI-suggested groupings, not final decisions.")
    print("   Use YOUR judgment to override cluster assignments when appropriate.")
    print("   Pay special attention to boundary stocks!")
    print("=" * 60)
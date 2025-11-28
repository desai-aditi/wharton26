"""
Clustering Analysis Script
Loads stock metrics and performs K-Means clustering to classify stocks.
Creates visualizations for team review.

Run this AFTER data_collector.py and after manually adding ESG Risk Ratings.

NOTE: The 'esg_score' column contains ESG RISK RATINGS from Morningstar/Sustainalytics
(not ESG scores). These are measured on a 0-100 scale where LOWER is BETTER.

Risk Rating Scale:
  Negligible: 0–9.99
  Low: 10–19.99
  Medium: 20–29.99
  High: 30–39.99
  Severe: 40+
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from typing import Optional, List
import warnings
import os
from dotenv import load_dotenv
warnings.filterwarnings('ignore')

# Load environment variables from .env
load_dotenv()


class ClusteringAnalysis:
    """Perform K-Means clustering on stock metrics"""
    
    def __init__(self, data_file: str):
        self.data_file = data_file
        self.df: Optional[pd.DataFrame] = None
        self.scaled_features: Optional[np.ndarray] = None
        self.scaler: Optional[StandardScaler] = None
        self.kmeans: Optional[KMeans] = None
        
    def load_data(self):
        """Load stock metrics from CSV"""
        print("=" * 60)
        print("CLUSTERING ANALYSIS")
        print("=" * 60)
        print(f"Loading data from: {self.data_file}")
        
        self.df = pd.read_csv(self.data_file)
        print(f"✓ Loaded {len(self.df)} stocks")
        print(f"✓ Columns: {list(self.df.columns)}")
        
        return self
    
    def prepare_features(self, feature_columns: list):
        """Select and standardize features for clustering"""
        print(f"\n📊 Preparing features for clustering...")
        print(f"Selected features: {feature_columns}")
        
        # Ensure data is loaded
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() before prepare_features()")

        # Check for missing features
        missing_cols = [col for col in feature_columns if col not in self.df.columns]
        if missing_cols:
            print(f"⚠️  WARNING: Missing columns: {missing_cols}")
            feature_columns = [col for col in feature_columns if col in self.df.columns]
        
        # Extract features
        features_df = self.df[feature_columns].copy()
        
        # Check for missing values
        print(f"\n🔍 Checking for missing values...")
        missing_counts = features_df.isnull().sum()
        if missing_counts.sum() > 0:
            print("⚠️  WARNING: Missing values detected!")
            print(missing_counts[missing_counts > 0])
            raise ValueError("Cannot proceed with missing values. Please clean data before clustering.")
        else:
            print("✓ No missing values")
        
        # Standardize features (z-score normalization)
        print("\n📏 Standardizing features...")
        self.scaler = StandardScaler()
        self.scaled_features = self.scaler.fit_transform(features_df)
        
        print(f"✓ Standardization complete")
        print(f"  Feature matrix shape: {self.scaled_features.shape}")
        
        return self
    
    def find_optimal_clusters(self, max_k: int = 7):
        """Use elbow method and silhouette score to find optimal k"""
        print(f"\n🔍 Finding optimal number of clusters (testing k=2 to {max_k})...")
        
        inertias = []
        silhouette_scores = []
        k_range = range(2, max_k + 1)
        
        # Ensure features have been prepared
        if self.scaled_features is None:
            raise ValueError("Features not prepared. Call prepare_features() before find_optimal_clusters()")

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(self.scaled_features)

            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(self.scaled_features, labels))
            
            print(f"  k={k}: inertia={kmeans.inertia_:.2f}, silhouette={silhouette_scores[-1]:.3f}")
        
        # Plot elbow curve
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(k_range, inertias, 'bo-')
        ax1.set_xlabel('Number of Clusters (k)')
        ax1.set_ylabel('Inertia (Within-cluster sum of squares)')
        ax1.set_title('Elbow Method')
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(k_range, silhouette_scores, 'ro-')
        ax2.set_xlabel('Number of Clusters (k)')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Silhouette Score by k')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('cluster_optimization.png', dpi=300, bbox_inches='tight')
        print(f"\n💾 Saved: cluster_optimization.png")
        plt.close()
        
        # Suggest optimal k
        best_k = k_range[np.argmax(silhouette_scores)]
        print(f"\n💡 Suggested k based on silhouette score: {best_k}")
        
        return best_k
    
    def run_clustering(self, n_clusters: int = 4):
        """Perform K-Means clustering"""
        print(f"\n🎯 Running K-Means clustering with k={n_clusters}...")
        # Preconditions
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() before run_clustering()")
        if self.scaled_features is None:
            raise ValueError("Features not prepared. Call prepare_features() before run_clustering()")

        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = self.kmeans.fit_predict(self.scaled_features)
        self.df['cluster'] = labels

        # Calculate silhouette score
        silhouette = silhouette_score(self.scaled_features, labels)
        
        print(f"✓ Clustering complete")
        print(f"  Silhouette score: {silhouette:.3f}")
        print(f"\n📊 Cluster distribution:")
        print(self.df['cluster'].value_counts().sort_index())
        
        return self
    
    def analyze_clusters(self, feature_columns: list):
        """Generate cluster profiles"""
        print(f"\n📈 CLUSTER PROFILES")
        print("=" * 60)
        
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() and run_clustering() before analyze_clusters()")

        # Calculate mean of each feature per cluster
        cluster_profiles = self.df.groupby('cluster')[feature_columns].mean()
        
        # Round for readability
        cluster_profiles = cluster_profiles.round(3)
        
        print("\nMean values per cluster:")
        print(cluster_profiles)
        
        # Save to CSV
        cluster_profiles.to_csv('cluster_profiles.csv')
        print(f"\n💾 Saved: cluster_profiles.csv")
        
        # Create heatmap
        plt.figure(figsize=(12, 6))
        sns.heatmap(cluster_profiles.T, annot=True, fmt='.2f', cmap='RdYlGn', 
                   center=0, cbar_kws={'label': 'Standardized Value'})
        plt.title('Cluster Profiles Heatmap', fontsize=14, fontweight='bold')
        plt.xlabel('Cluster')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.savefig('cluster_heatmap.png', dpi=300, bbox_inches='tight')
        print(f"💾 Saved: cluster_heatmap.png")
        plt.close()
        
        # Print stocks per cluster
        print(f"\n📋 STOCKS PER CLUSTER")
        print("=" * 60)
        for cluster_id in sorted(self.df['cluster'].unique()):
            stocks = self.df[self.df['cluster'] == cluster_id]['ticker'].tolist()
            print(f"\nCluster {cluster_id} ({len(stocks)} stocks):")
            print(f"  {', '.join(stocks)}")
        
        return cluster_profiles
    
    def create_visualizations(self, feature_columns: list):
        """Create scatter plots for cluster visualization"""
        print(f"\n🎨 Creating visualizations...")
        
        if self.df is None or 'cluster' not in self.df.columns:
            raise ValueError("Clustering not yet run. Call run_clustering() before create_visualizations()")

        # 2D Scatter: Sharpe vs Volatility
        if 'sharpe_ratio' in feature_columns and 'volatility' in feature_columns:
            plt.figure(figsize=(10, 6))
            
            for cluster_id in sorted(self.df['cluster'].unique()):
                cluster_data = self.df[self.df['cluster'] == cluster_id]
                plt.scatter(cluster_data['volatility'], cluster_data['sharpe_ratio'],
                          label=f'Cluster {cluster_id}', s=100, alpha=0.6)
                
                # Add ticker labels
                for _, row in cluster_data.iterrows():
                    plt.annotate(row['ticker'], 
                               (row['volatility'], row['sharpe_ratio']),
                               fontsize=8, alpha=0.7)
            
            plt.xlabel('Volatility (Annualized)', fontsize=12)
            plt.ylabel('Sharpe Ratio', fontsize=12)
            plt.title('Stock Clustering: Sharpe Ratio vs Volatility', 
                     fontsize=14, fontweight='bold')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('cluster_sharpe_volatility.png', dpi=300, bbox_inches='tight')
            print(f"  ✓ Saved: cluster_sharpe_volatility.png")
            plt.close()
        
    # 2D Scatter: Sortino vs Max Drawdown
        if 'sortino_ratio' in feature_columns and 'max_drawdown' in feature_columns:
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
            plt.title('Stock Clustering: Sortino Ratio vs Max Drawdown', 
                     fontsize=14, fontweight='bold')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('cluster_sortino_drawdown.png', dpi=300, bbox_inches='tight')
            print(f"  ✓ Saved: cluster_sortino_drawdown.png")
            plt.close()
        
        # 3D Scatter: Sharpe vs Volatility vs ESG Risk Rating (if ESG available)
        # NOTE: Lower ESG risk rating is BETTER (Negligible=0-10, Low=10-20, etc.)
        if all(col in feature_columns for col in ['sharpe_ratio', 'volatility', 'esg_score']):
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            colors = ['red', 'blue', 'green', 'orange', 'purple']
            for cluster_id in sorted(self.df['cluster'].unique()):
                cluster_data = self.df[self.df['cluster'] == cluster_id]
                ax.scatter(cluster_data['volatility'], 
                          cluster_data['sharpe_ratio'],
                          cluster_data['esg_score'],
                          label=f'Cluster {cluster_id}',
                          s=100, alpha=0.6,
                          c=colors[cluster_id % len(colors)])
            
            ax.set_xlabel('Volatility', fontsize=10)
            ax.set_ylabel('Sharpe Ratio', fontsize=10)
            ax.set_zlabel('ESG Risk Rating\n(lower=better)', fontsize=10)
            ax.set_title('3D Cluster Visualization\n(ESG Risk Ratings: Negligible 0-10, Low 10-20, Medium 20-30, High 30-40, Severe 40+)', 
                        fontsize=12, fontweight='bold')
            ax.legend()
            plt.tight_layout()
            plt.savefig('cluster_3d.png', dpi=300, bbox_inches='tight')
            print(f"  ✓ Saved: cluster_3d.png")
            plt.close()
        
        print(f"\n✅ All visualizations complete!")
    
    def save_results(self, output_file: str = 'clustered_stocks.csv'):
        """Save results with cluster assignments"""
        if self.df is None:
            raise ValueError("No results to save. Run the clustering workflow first.")

        self.df.to_csv(output_file, index=False)
        print(f"\n💾 Saved clustered results to: {output_file}")
        return output_file
    
# ========== MAIN EXECUTION ==========

if __name__ == "__main__":
    # ===== CONFIGURATION =====
    INPUT_FILE = 'stock_metrics.csv'  # File from data_collector.py (with ESG scores added)
    OUTPUT_FILE = 'clustered_stocks.csv'
    
    # Features to use for clustering (7 key metrics)
    CLUSTERING_FEATURES = [
        'sortino_ratio',
        'sharpe_ratio',
        'volatility',
        'max_drawdown',
        'dividend_yield',
        'esg_score',
        'beta_calculated'  # or 'beta_av' if using Alpha Vantage beta
    ]
    
    # Number of clusters
    N_CLUSTERS = 4
    
    # ===== RUN CLUSTERING =====
    print("\n🚀 Starting clustering analysis...\n")
    
    # Initialize
    analyzer = ClusteringAnalysis(INPUT_FILE)
    
    # Step 1: Load data
    analyzer.load_data()
    
    # Step 2: Prepare features (standardize)
    analyzer.prepare_features(CLUSTERING_FEATURES)
    
    # Step 3: Find optimal number of clusters (optional - to validate k=4)
    print("\n" + "="*60)
    response = input("Do you want to test optimal k? (y/n): ").lower()
    if response == 'y':
        suggested_k = analyzer.find_optimal_clusters(max_k=7)
        print(f"\nSuggested k: {suggested_k}")
        
        use_suggested = input(f"Use suggested k={suggested_k}? (y/n, default=n): ").lower()
        if use_suggested == 'y':
            N_CLUSTERS = suggested_k
    
    # Step 4: Run clustering with chosen k
    analyzer.run_clustering(n_clusters=N_CLUSTERS)
    
    # Step 5: Analyze clusters (generate profiles)
    cluster_profiles = analyzer.analyze_clusters(CLUSTERING_FEATURES)
    
    # Step 6: Create visualizations
    analyzer.create_visualizations(CLUSTERING_FEATURES)
    
    # Step 7: Save results
    analyzer.save_results(OUTPUT_FILE)
    
    # ===== FINAL SUMMARY =====
    print("\n" + "="*60)
    print("CLUSTERING COMPLETE!")
    print("="*60)
    print("\n📁 Output files created:")
    print("  1. clustered_stocks.csv - Stock data with cluster assignments")
    print("  2. cluster_profiles.csv - Average metrics per cluster")
    print("  3. cluster_optimization.png - Elbow curve (if you ran optimal k test)")
    print("  4. cluster_heatmap.png - Cluster profile heatmap")
    print("  5. cluster_sharpe_volatility.png - 2D scatter plot")
    print("  6. cluster_sortino_drawdown.png - 2D scatter plot")
    print("  7. cluster_3d.png - 3D scatter plot (if ESG scores present)")
    
    print("\n📋 NEXT STEPS:")
    print("  1. Review cluster_profiles.csv to understand each cluster")
    print("  2. Look at visualizations to see stock groupings")
    print("  3. Note: esg_score = ESG Risk Rating (Sustainalytics/Morningstar)")
    print("     - LOWER values are BETTER (Negligible < Low < Medium < High < Severe)")
    print("     - Scale: Negligible 0-10, Low 10-20, Medium 20-30, High 30-40, Severe 40+")
    print("  4. Manually label clusters in clustered_stocks.csv:")
    print("     - Add a 'category' column with: Core, Growth, Impact, or Reject")
    print("  5. Use labeled data for Component 2 (Risk Scoring)")
    print("\n✅ Ready to proceed to risk scoring!")
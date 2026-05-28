import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def generate_eda_plots():
    # File paths
    input_path = r'c:\Users\kanta\workspace\projects\inturn\廃業予測\data\processed\nagoya.csv'
    output_dir = r'c:\Users\kanta\workspace\projects\inturn\廃業予測\data\raw\nagoya_eda'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    df = pd.read_csv(input_path)
    
    print(f"Data shape: {df.shape}")
    
    # Select numerical columns
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # 1. Correlation Heatmap (Top 20 features correlated with is_closed)
    plt.figure(figsize=(12, 10))
    correlations = df[num_cols].corr()
    top_corr_cols = correlations['is_closed'].abs().sort_values(ascending=False).head(20).index
    top_corr = df[top_corr_cols].corr()
    
    sns.heatmap(top_corr, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
    plt.title('Top 20 Correlation Heatmap with is_closed')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap_top20.png'))
    plt.close()
    
    # 2. Distribution Histograms for selected key features
    key_features = ['rating', 'user_ratings_total', 'dist_to_nagoya_sta', 'count_restaurants_500m', 'flow_score_raw_b1.0']
    
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(key_features, 1):
        if col in df.columns:
            plt.subplot(2, 3, i)
            sns.histplot(df[col].dropna(), kde=True, bins=30)
            plt.title(f'Distribution of {col}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'distribution_key_features.png'))
    plt.close()

    # 3. Scatter plot: rating vs user_ratings_total (to observe outliers)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='rating', y='user_ratings_total', alpha=0.5)
    plt.title('Scatter Plot: Rating vs User Ratings Total')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scatter_rating_vs_reviews.png'))
    plt.close()
    
    # 4. Boxplot to observe variance / scale differences (example columns)
    plt.figure(figsize=(12, 6))
    dist_cols = [c for c in num_cols if c.startswith('dist_to_nearest_')][:5]
    sns.boxplot(data=df[dist_cols])
    plt.title('Boxplot of distance variables (Scale observation)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'boxplot_distances.png'))
    plt.close()
    
    print(f"EDA plots successfully saved to {output_dir}")

if __name__ == '__main__':
    generate_eda_plots()

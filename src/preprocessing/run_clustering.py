import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KernelDensity

def run_clustering():
    data_path = r'c:\Users\kanta\workspace\projects\inturn\廃業予測\data\processed\nagoya.csv'
    output_dir = r'c:\Users\kanta\workspace\projects\inturn\廃業予測\data\raw\nagoya_eda'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load raw data
    df = pd.read_csv(data_path)
    
    # Check what features to use for K-Means
    # Using coord + flow + density
    features_kmeans = ['latitude', 'longitude', 'flow_score_raw_b1.5', 'count_restaurants_500m']
    X_k = df[features_kmeans].dropna()
    
    scaler = StandardScaler()
    X_k_scaled = scaler.fit_transform(X_k)
    
    # 1. K-Means (Determine best K)
    print("Running K-Means...")
    best_k = 3
    best_score = -1
    scores = []
    
    for k in range(3, 9):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        preds = kmeans.fit_predict(X_k_scaled)
        score = silhouette_score(X_k_scaled, preds)
        scores.append(score)
        if score > best_score:
            best_score = score
            best_k = k
            
    print(f"Best K for K-Means: {best_k} (Silhouette Score: {best_score:.4f})")
    
    # Fit with best K
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init='auto')
    df['kmeans_cluster'] = kmeans.fit_predict(X_k_scaled)
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x='longitude', y='latitude', hue='kmeans_cluster', palette='viridis', s=15, alpha=0.6)
    plt.title(f'K-Means Clustering (k={best_k}) based on Location, Flow, and Density')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'kmeans_clusters.png'))
    plt.close()
    
    # 2. DBSCAN (Haversine distance)
    print("Running DBSCAN...")
    coords = df[['latitude', 'longitude']].dropna()
    # Convert to radians for Haversine
    coords_rad = np.radians(coords)
    
    # eps = 300 meters
    earth_radius_km = 6371.0088
    eps_rad = (300 / 1000.0) / earth_radius_km
    
    dbscan = DBSCAN(eps=eps_rad, min_samples=5, metric='haversine', algorithm='ball_tree')
    df['dbscan_cluster'] = dbscan.fit_predict(coords_rad)
    
    n_noise = list(df['dbscan_cluster']).count(-1)
    print(f"DBSCAN: found {len(set(df['dbscan_cluster'])) - (1 if -1 in df['dbscan_cluster'] else 0)} clusters.")
    print(f"DBSCAN: {n_noise} out of {len(df)} restaurants are considered noise (isolated, >300m away from clusters).")
    
    plt.figure(figsize=(10, 8))
    # Noise points in grey
    noise_df = df[df['dbscan_cluster'] == -1]
    cluster_df = df[df['dbscan_cluster'] != -1]
    plt.scatter(cluster_df['longitude'], cluster_df['latitude'], c=cluster_df['dbscan_cluster'], cmap='tab20', s=15, alpha=0.6, label='Clustered')
    plt.scatter(noise_df['longitude'], noise_df['latitude'], c='lightgrey', s=10, alpha=0.5, label='Noise (Isolated)')
    plt.title('DBSCAN Clustering (Dense Areas vs Isolated)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dbscan_clusters.png'))
    plt.close()
    
    # 3. KDE (Density Score)
    print("Running KDE...")
    # bandwidth estimation (approximate)
    kde = KernelDensity(bandwidth=0.01, metric='haversine', kernel='gaussian', algorithm='ball_tree')
    kde.fit(coords_rad)
    
    # Calculate density score for each point
    log_density = kde.score_samples(coords_rad)
    df['kde_density'] = np.exp(log_density)
    
    plt.figure(figsize=(10, 8))
    sc = plt.scatter(df['longitude'], df['latitude'], c=df['kde_density'], cmap='inferno', s=15, alpha=0.8)
    plt.colorbar(sc, label='Density Score')
    plt.title('KDE Density Plot of Restaurants')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'kde_density.png'))
    plt.close()
    
    # Closed Restaurants Distribution in Clusters
    closed_df = df[df['is_closed'] == 1]
    print("\n--- Insights on Closed Restaurants (is_closed=1) ---")
    print("K-Means Cluster distribution of Closed Restaurants:")
    print(closed_df['kmeans_cluster'].value_counts())
    
    print("\nDBSCAN Cluster distribution of Closed Restaurants (-1 is isolated):")
    print(closed_df['dbscan_cluster'].value_counts())
    
    # Save the dataframe with cluster labels for later modeling if needed
    # Since we need to follow the anti-leakage rule, we will calculate them in CV later.
    # Here we just output insights.
    print("\nFinished Clustering EDA.")

if __name__ == '__main__':
    run_clustering()

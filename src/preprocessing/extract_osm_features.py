import pandas as pd
import numpy as np
import os
import yaml
from scipy.spatial import cKDTree
from shapely.geometry import Point, LineString
import glob

def latlon_to_meters(lat, lon):
    # 名古屋付近 (35N, 137E) の簡易変換
    # 緯度1度 = 111km, 経度1度 = 91km
    return np.array([lat * 111000, lon * 91000])

def main():
    with open('config/config.yml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    input_path = os.path.join(config['data_paths']['processed'], 'nagoya_analysis_warehouse_categorized.csv')
    osm_dir = "data/raw/osm"
    output_path = os.path.join(config['data_paths']['processed'], 'nagoya_osm_features.csv')
    
    print(f"Loading store data from {input_path}...")
    stores_df = pd.read_csv(input_path)
    store_coords = latlon_to_meters(stores_df['latitude'].values, stores_df['longitude'].values).T
    
    feature_dfs = []
    
    # --- 1. Point-based Features (POIs) ---
    poi_configs = {
        "restaurants": "osm_restaurants.csv",
        "convenience": "osm_convenience.csv",
        "offices": "osm_offices.csv",
        "signals": "osm_signals.csv",
        "schools": "osm_schools.csv",
        "hospitals": "osm_hospitals.csv",
        "parking": "osm_parking.csv",
        "parks": "osm_parks.csv",
        "malls": "osm_malls.csv"
    }
    
    for feat_name, csv_file in poi_configs.items():
        csv_path = os.path.join(osm_dir, csv_file)
        if not os.path.exists(csv_path):
            print(f"Warning: {csv_path} not found. Skipping...")
            continue
            
        print(f"Processing {feat_name}...")
        df_poi = pd.read_csv(csv_path).dropna(subset=['lat', 'lon'])
        poi_coords = latlon_to_meters(df_poi['lat'].values, df_poi['lon'].values).T
        tree = cKDTree(poi_coords)
        
        # Distance to nearest
        dist, _ = tree.query(store_coords, k=1)
        stores_df[f'dist_to_nearest_{feat_name}'] = dist
        
        # Counts at radii
        for r in [300, 500, 1000]:
            stores_df[f'count_{feat_name}_{r}m'] = tree.query_ball_point(store_coords, r, return_length=True)
            
    # --- 2. Road-based Features (幹線道路) ---
    roads_path = os.path.join(osm_dir, "osm_roads_nodes.csv")
    if os.path.exists(roads_path):
        print("Processing road nodes for precise distance...")
        df_roads = pd.read_csv(roads_path).dropna(subset=['lat', 'lon'])
        road_coords = latlon_to_meters(df_roads['lat'].values, df_roads['lon'].values).T
        road_tree = cKDTree(road_coords)
        
        road_dist, _ = road_tree.query(store_coords, k=1)
        stores_df['dist_to_nearest_road'] = road_dist
        stores_df['is_along_highway'] = (road_dist <= 50).astype(int)
    else:
        print("Warning: osm_roads_nodes.csv not found. Using fallback osm_roads.csv if available.")
        fallback_path = os.path.join(osm_dir, "osm_roads.csv")
        if os.path.exists(fallback_path):
            df_roads = pd.read_csv(fallback_path).dropna(subset=['lat', 'lon'])
            road_coords = latlon_to_meters(df_roads['lat'].values, df_roads['lon'].values).T
            road_tree = cKDTree(road_coords)
            road_dist, _ = road_tree.query(store_coords, k=1)
            stores_df['dist_to_nearest_road'] = road_dist
            stores_df['is_along_highway'] = (road_dist <= 50).astype(int)
    
    # --- 3. Flags ---
    if 'dist_to_nearest_malls' in stores_df.columns:
        stores_df['is_near_mall'] = (stores_df['dist_to_nearest_malls'] <= 100).astype(int)
        
    # --- Final Output ---
    # We only keep the ID and the new features to join later
    osm_cols = [c for c in stores_df.columns if c.startswith('dist_') or c.startswith('count_') or c.startswith('is_')]
    output_df = stores_df[['id'] + osm_cols]
    
    output_df.to_csv(output_path, index=False)
    print(f"OSM features saved to {output_path}")

if __name__ == "__main__":
    main()

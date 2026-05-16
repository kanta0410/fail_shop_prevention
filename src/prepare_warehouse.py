import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt

# Configuration
OSM_LABELS_FILE = "nagoya_osm_labels.csv"
GOOGLE_DATA_FILE = "nagoya_all_massive_raw.csv"
OUTPUT_FILE = "nagoya_analysis_warehouse.csv"

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers.
    return c * r * 1000 # in meters

def main():
    print("Loading data...")
    df_osm = pd.read_csv(OSM_LABELS_FILE)
    df_google = pd.read_csv(GOOGLE_DATA_FILE)

    print(f"OSM Labels: {len(df_osm)} rows")
    print(f"Google Data: {len(df_google)} rows")

    print("Matching OSM labels with Google Places data...")
    final_rows = []
    
    # Matching logic
    for _, osm_row in df_osm.iterrows():
        lat_m = 0.001
        lng_m = 0.001
        
        # Initial filtering by BBOX
        candidates = df_google[
            (df_google['lat'] >= osm_row['lat'] - lat_m) & (df_google['lat'] <= osm_row['lat'] + lat_m) &
            (df_google['lng'] >= osm_row['lng'] - lng_m) & (df_google['lng'] <= osm_row['lng'] + lng_m)
        ]
        
        best_match = None
        min_dist = 50 
        
        for _, g_row in candidates.iterrows():
            dist = haversine(osm_row['lng'], osm_row['lat'], g_row['lng'], g_row['lat'])
            
            osm_name = str(osm_row['name']).lower()
            g_name = str(g_row['name']).lower()
            name_sim = (osm_name in g_name) or (g_name in osm_name)
            
            if name_sim and dist < 100:
                if dist < min_dist or best_match is None:
                    min_dist = dist
                    best_match = g_row
            elif dist < 30: 
                if dist < min_dist or best_match is None:
                    min_dist = dist
                    best_match = g_row
                    
        if best_match is not None:
            # Prepare row for Data Warehouse
            row = {
                "id": best_match['id'],
                "name": best_match['name'],
                "is_closed": osm_row['target_closed_osm'],
                "rating": best_match['rating'],
                "user_ratings_total": best_match['user_rating_count'],
                "price_level": best_match['price_level'],
                "latitude": best_match['lat'],
                "longitude": best_match['lng']
            }
            final_rows.append(row)

    df_warehouse = pd.DataFrame(final_rows)
    print(f"Matched {len(df_warehouse)} rows.")

    # One-hot encoding for price_level
    print("Performing one-hot encoding for price_level...")
    df_warehouse['price_level'] = df_warehouse['price_level'].fillna("UNKNOWN")
    
    categories = [
        "PRICE_LEVEL_INEXPENSIVE",
        "PRICE_LEVEL_MODERATE",
        "PRICE_LEVEL_EXPENSIVE",
        "PRICE_LEVEL_VERY_EXPENSIVE",
        "UNKNOWN"
    ]
    
    for cat in categories:
        col_name = cat.lower().replace("price_level_", "")
        df_warehouse[f"price_{col_name}"] = (df_warehouse['price_level'] == cat).astype(int)
    
    # Drop original price_level column
    df_warehouse.drop(columns=['price_level'], inplace=True)

    # Clean data types
    df_warehouse['user_ratings_total'] = df_warehouse['user_ratings_total'].fillna(0).astype(int)
    df_warehouse['is_closed'] = df_warehouse['is_closed'].astype(int)

    # Save to CSV
    df_warehouse.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"Data Warehouse CSV saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

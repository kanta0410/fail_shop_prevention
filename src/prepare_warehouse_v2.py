import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt

# Configuration
OSM_LABELS_FILE = "nagoya_osm_labels.csv"
GOOGLE_DATA_FILE = "nagoya_all_massive_raw.csv"
OUTPUT_FILE = "nagoya_analysis_warehouse.csv"

def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 
    return c * r * 1000

def main():
    print("Loading data...")
    df_google = pd.read_csv(GOOGLE_DATA_FILE)
    df_osm = pd.read_csv(OSM_LABELS_FILE)

    print(f"Initial Google Data: {len(df_google)} rows")
    print(f"OSM Labels: {len(df_osm)} rows")

    # Initialize is_closed as 0 for all Google records
    df_google['is_closed'] = 0
    
    # We will track which OSM records got matched
    matched_osm_ids = set()
    
    print("Matching Google records with OSM labels...")
    # For each Google record, see if OSM says it's closed
    for idx, g_row in df_google.iterrows():
        lat_m = 0.001
        lng_m = 0.001
        
        candidates = df_osm[
            (df_osm['lat'] >= g_row['lat'] - lat_m) & (df_osm['lat'] <= g_row['lat'] + lat_m) &
            (df_osm['lng'] >= g_row['lng'] - lng_m) & (df_osm['lng'] <= g_row['lng'] + lng_m)
        ]
        
        best_match_osm = None
        min_dist = 50
        
        for _, o_row in candidates.iterrows():
            dist = haversine(g_row['lng'], g_row['lat'], o_row['lng'], o_row['lat'])
            
            # Name matching
            g_name = str(g_row['name']).lower()
            o_name = str(o_row['name']).lower()
            name_sim = (g_name in o_name) or (o_name in g_name)
            
            if name_sim and dist < 100:
                if dist < min_dist or best_match_osm is None:
                    min_dist = dist
                    best_match_osm = o_row
            elif dist < 30:
                if dist < min_dist or best_match_osm is None:
                    min_dist = dist
                    best_match_osm = o_row
        
        if best_match_osm is not None:
            # If matched, use the OSM closure label
            df_google.at[idx, 'is_closed'] = best_match_osm['target_closed_osm']
            matched_osm_ids.add(best_match_osm['osm_id'])

    # Now, find OSM records that are CLOSED (1) but were NOT matched to any Google record
    print("Adding unmatched closed OSM records...")
    unmatched_closed_osm = df_osm[(df_osm['target_closed_osm'] == 1) & (~df_osm['osm_id'].isin(matched_osm_ids))]
    
    print(f"Unmatched closed OSM records to add: {len(unmatched_closed_osm)}")
    
    # Create rows for these unmatched closed stores
    new_rows = []
    for _, o_row in unmatched_closed_osm.iterrows():
        new_rows.append({
            "id": f"osm_{o_row['osm_id']}",
            "name": o_row['name'],
            "is_closed": 1,
            "rating": np.nan,
            "user_rating_count": 0,
            "price_level": "UNKNOWN",
            "lat": o_row['lat'],
            "lng": o_row['lng']
        })
    
    # Prepare Google records for concatenation
    df_google_subset = df_google[['id', 'name', 'is_closed', 'rating', 'user_rating_count', 'price_level', 'lat', 'lng']].copy()
    
    # Combine
    df_combined = pd.concat([df_google_subset, pd.DataFrame(new_rows)], ignore_index=True)
    
    # Rename columns to final requirement
    df_combined.rename(columns={
        "user_rating_count": "user_ratings_total",
        "lat": "latitude",
        "lng": "longitude"
    }, inplace=True)
    
    # One-hot encoding for price_level
    print("Performing one-hot encoding...")
    df_combined['price_level'] = df_combined['price_level'].fillna("UNKNOWN")
    
    categories = [
        "PRICE_LEVEL_INEXPENSIVE",
        "PRICE_LEVEL_MODERATE",
        "PRICE_LEVEL_EXPENSIVE",
        "PRICE_LEVEL_VERY_EXPENSIVE",
        "UNKNOWN"
    ]
    
    for cat in categories:
        col_name = cat.lower().replace("price_level_", "")
        df_combined[f"price_{col_name}"] = (df_combined['price_level'] == cat).astype(int)
    
    # Drop original price_level
    df_combined.drop(columns=['price_level'], inplace=True)
    
    # Clean data types
    df_combined['user_ratings_total'] = df_combined['user_ratings_total'].fillna(0).astype(int)
    df_combined['is_closed'] = df_combined['is_closed'].astype(int)

    # Save to CSV
    df_combined.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"Final Data Warehouse CSV saved to {OUTPUT_FILE} with {len(df_combined)} rows.")
    print("Closure status distribution:")
    print(df_combined['is_closed'].value_counts())

if __name__ == "__main__":
    main()

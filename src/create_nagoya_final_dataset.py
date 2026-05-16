import os
import time
import requests
import pandas as pd
import json
from math import radians, cos, sin, asin, sqrt

# Configuration
OSM_URL = "https://overpass.kumi.systems/api/interpreter"
OUTPUT_FILE = "nagoya_osm_labels.csv"
PROGRESS_FILE = "osm_progress.json"
GOOGLE_DATA_FILE = "nagoya_all_massive_raw.csv"

# Nagoya BBOX
MIN_LAT = 35.030
MAX_LAT = 35.260
MIN_LNG = 136.790
MAX_LNG = 137.050
STEP = 0.01 # 1km grids

def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 
    return c * r * 1000

def fetch_osm_data(query, timeout=180):
    for attempt in range(3):
        try:
            # Note: We use a very long timeout for historical queries
            response = requests.post(OSM_URL, data={'data': query}, timeout=timeout+10)
            if response.status_code == 200:
                return response.json().get("elements", [])
            elif response.status_code == 429:
                print("Rate limited. Sleeping 5s...")
                time.sleep(5)
            else:
                print(f"OSM Error {response.status_code}: {response.text[:100]}")
        except Exception as e:
            print(f"Request attempt {attempt+1} failed: {e}")
        time.sleep(1)
    return None

def main():
    print("=== Nagoya Restaurant Dataset Creation Pipeline ===")
    
    # Load progress
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            progress = json.load(f)
    else:
        progress = {"completed_grids": [], "osm_2025": {}, "osm_2026": {}}

    # Load existing labels if any
    if os.path.exists(OUTPUT_FILE):
        df_labels = pd.read_csv(OUTPUT_FILE)
        print(f"Loaded {len(df_labels)} existing labels.")
    
    lat_steps = int((MAX_LAT - MIN_LAT) / STEP) + 1
    lng_steps = int((MAX_LNG - MIN_LNG) / STEP) + 1
    total_grids = lat_steps * lng_steps
    
    print(f"Total grids to process: {total_grids}")
    
    lat_idx = 0
    while MIN_LAT + lat_idx * STEP < MAX_LAT:
        curr_lat = MIN_LAT + lat_idx * STEP
        lng_idx = 0
        while MIN_LNG + lng_idx * STEP < MAX_LNG:
            curr_lng = MIN_LNG + lng_idx * STEP
            grid_id = f"{lat_idx}_{lng_idx}"
            
            if grid_id in progress["completed_grids"]:
                lng_idx += 1
                continue
            
            grid_count = len(progress['completed_grids']) + 1
            print(f"Processing Grid {grid_id} ({grid_count}/{total_grids})...")
            
            # Query for 2025
            q2025 = f'[out:json][timeout:180][date:"2025-01-01T00:00:00Z"];(node["amenity"~"restaurant|cafe|bar|fast_food"]({curr_lat},{curr_lng},{curr_lat+STEP},{curr_lng+STEP});way["amenity"~"restaurant|cafe|bar|fast_food"]({curr_lat},{curr_lng},{curr_lat+STEP},{curr_lng+STEP}););out center;'
            res2025 = fetch_osm_data(q2025)
            
            if res2025 is None:
                print("Critical Error: Failed to fetch 2025 data. Retrying later.")
                time.sleep(1)
                lng_idx += 1
                continue
                
            for el in res2025:
                tags = el.get("tags", {})
                name = tags.get("name")
                if name:
                    eid = str(el["id"])
                    c = el.get("center") or el
                    progress["osm_2025"][eid] = {"name": name, "lat": c["lat"], "lng": c["lon"]}

            time.sleep(0.5)

            # Query for 2026
            q2026 = f'[out:json][timeout:180][date:"2026-01-01T00:00:00Z"];(node["amenity"~"restaurant|cafe|bar|fast_food"]({curr_lat},{curr_lng},{curr_lat+STEP},{curr_lng+STEP});way["amenity"~"restaurant|cafe|bar|fast_food"]({curr_lat},{curr_lng},{curr_lat+STEP},{curr_lng+STEP}););out center;'
            res2026 = fetch_osm_data(q2026)
            
            if res2026 is None:
                print("Critical Error: Failed to fetch 2026 data.")
                lng_idx += 1
                continue
                
            for el in res2026:
                eid = str(el["id"])
                progress["osm_2026"][eid] = True # Just mark presence

            # Mark grid as completed
            progress["completed_grids"].append(grid_id)
            
            # Periodically save progress
            if len(progress["completed_grids"]) % 5 == 0:
                with open(PROGRESS_FILE, "w") as f:
                    json.dump(progress, f)
                
                # Generate intermediate labels
                labels = []
                for eid, info in progress["osm_2025"].items():
                    labels.append({
                        "osm_id": eid,
                        "name": info["name"],
                        "lat": info["lat"],
                        "lng": info["lng"],
                        "target_closed_osm": 1 if eid not in progress["osm_2026"] else 0
                    })
                pd.DataFrame(labels).to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

            time.sleep(0.5)
            lng_idx += 1
        lat_idx += 1

    print("OSM Data Extraction Complete!")
    
    # Final Label Generation
    labels = []
    for eid, info in progress["osm_2025"].items():
        labels.append({
            "osm_id": eid,
            "name": info["name"],
            "lat": info["lat"],
            "lng": info["lng"],
            "target_closed_osm": 1 if eid not in progress["osm_2026"] else 0
        })
    df_labels = pd.DataFrame(labels)
    df_labels.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    
    # Merge with Google Data
    if os.path.exists(GOOGLE_DATA_FILE):
        print("Merging with Google Places data...")
        df_google = pd.read_csv(GOOGLE_DATA_FILE)
        final_data = []
        
        for _, osm_row in df_labels.iterrows():
            # Match by proximity and name
            lat_m = 0.001
            lng_m = 0.001
            candidates = df_google[
                (df_google['lat'] >= osm_row['lat'] - lat_m) & (df_google['lat'] <= osm_row['lat'] + lat_m) &
                (df_google['lng'] >= osm_row['lng'] - lng_m) & (df_google['lng'] <= osm_row['lng'] + lng_m)
            ]
            
            best_match = None
            min_dist = 50
            for _, g_row in candidates.iterrows():
                dist = haversine(osm_row['lng'], osm_row['lat'], g_row['lng'], g_row['lat'])
                name_sim = (str(osm_row['name']) in str(g_row['name'])) or (str(g_row['name']) in str(osm_row['name']))
                if name_sim and dist < 100:
                    if dist < min_dist or best_match is None:
                        min_dist = dist; best_match = g_row
                elif dist < 30:
                    if dist < min_dist or best_match is None:
                        min_dist = dist; best_match = g_row
            
            if best_match is not None:
                d = osm_row.to_dict()
                d.update({
                    "rating": best_match['rating'],
                    "user_rating_count": best_match['user_rating_count'],
                    "price_level": best_match['price_level']
                })
                final_data.append(d)
        
        df_final = pd.DataFrame(final_data)
        df_final.to_csv("nagoya_final_dataset.csv", index=False, encoding="utf-8-sig")
        print(f"Final dataset saved with {len(df_final)} rows.")
        print(df_final['target_closed_osm'].value_counts())

if __name__ == "__main__":
    main()

import os
import time
import requests
import pandas as pd
from math import radians, cos, sin, asin, sqrt

# OSM API Endpoint (kumi.systems works for us)
OSM_URL = "https://overpass.kumi.systems/api/interpreter"

# 名古屋市のバウンディングボックス
MIN_LAT = 35.030
MAX_LAT = 35.260
MIN_LNG = 136.790
MAX_LNG = 137.050

# サーバーのタイムアウトを防ぐため、1km四方 (0.01度) のグリッドに分割してリクエストする
STEP = 0.01

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers
    return c * r * 1000 # returns meters

def fetch_osm_grid(target_date, low_lat, low_lng, high_lat, high_lng):
    query = f"""
    [out:json][timeout:50][date:"{target_date}"];
    (
      node["amenity"~"restaurant|cafe|bar|fast_food"]({low_lat},{low_lng},{high_lat},{high_lng});
      way["amenity"~"restaurant|cafe|bar|fast_food"]({low_lat},{low_lng},{high_lat},{high_lng});
    );
    out center;
    """
    
    for attempt in range(3):
        try:
            response = requests.post(OSM_URL, data={'data': query}, timeout=30)
            if response.status_code == 200:
                data = response.json()
                return data.get("elements", [])
            else:
                print(f"OSM Error {response.status_code}")
        except Exception as e:
            print(f"Request failed: {e}")
        time.sleep(2)
    return []

def main():
    print("Starting OSM Grid Extraction for Nagoya City...")
    lat_steps = int((MAX_LAT - MIN_LAT) / STEP) + 1
    lng_steps = int((MAX_LNG - MIN_LNG) / STEP) + 1
    total_grids = lat_steps * lng_steps
    
    print(f"Total grids: {total_grids}. This will take about {total_grids * 4 / 60:.1f} minutes.")
    
    osm_2025_all = {}
    osm_2026_all = {}
    
    lat = MIN_LAT
    grid_count = 0
    
    while lat < MAX_LAT:
        lng = MIN_LNG
        while lng < MAX_LNG:
            grid_count += 1
            if grid_count % 10 == 0 or grid_count == 1:
                print(f"Processing Grid {grid_count}/{total_grids} (Lat: {lat:.3f}, Lng: {lng:.3f})...")
            
            # Fetch 2025
            elems_2025 = fetch_osm_grid("2025-01-01T00:00:00Z", lat, lng, lat + STEP, lng + STEP)
            for el in elems_2025:
                el_id = el.get("id")
                tags = el.get("tags", {})
                name = tags.get("name")
                if name:
                    c_lat = el.get("lat") or el.get("center", {}).get("lat")
                    c_lng = el.get("lon") or el.get("center", {}).get("lon")
                    if c_lat and c_lng:
                        osm_2025_all[el_id] = {"id": el_id, "name": name, "lat": c_lat, "lng": c_lng}
            
            time.sleep(1) # Rate limit protection
            
            # Fetch 2026
            elems_2026 = fetch_osm_grid("2026-01-01T00:00:00Z", lat, lng, lat + STEP, lng + STEP)
            for el in elems_2026:
                el_id = el.get("id")
                tags = el.get("tags", {})
                name = tags.get("name")
                if name:
                    c_lat = el.get("lat") or el.get("center", {}).get("lat")
                    c_lng = el.get("lon") or el.get("center", {}).get("lon")
                    if c_lat and c_lng:
                        osm_2026_all[el_id] = {"id": el_id, "name": name, "lat": c_lat, "lng": c_lng}
                        
            time.sleep(1) # Rate limit protection
            
            lng += STEP
        lat += STEP

    print(f"Total OSM places in 2025: {len(osm_2025_all)}")
    print(f"Total OSM places in 2026: {len(osm_2026_all)}")
    
    # Calculate closure labels
    dataset = []
    closed_count = 0
    operational_count = 0
    
    for pid, pinfo in osm_2025_all.items():
        is_closed = 1 if pid not in osm_2026_all else 0
        if is_closed:
            closed_count += 1
        else:
            operational_count += 1
            
        dataset.append({
            "osm_id": pid,
            "name": pinfo["name"],
            "lat": pinfo["lat"],
            "lng": pinfo["lng"],
            "target_closed_osm": is_closed
        })
        
    print(f"Generated labels: {operational_count} operational, {closed_count} closed.")
    
    df_osm = pd.DataFrame(dataset)
    df_osm.to_csv("nagoya_osm_labels.csv", index=False, encoding="utf-8-sig")
    print("Saved OSM labels to nagoya_osm_labels.csv")
    
    # Now merge with Google Places data locally to avoid API costs
    print("\nMatching with Google Places API data (nagoya_all_massive_raw.csv)...")
    try:
        df_google = pd.read_csv("nagoya_all_massive_raw.csv")
        
        # Simple matching algorithm: Same name substring OR within 30 meters
        final_data = []
        matched_count = 0
        
        for idx, osm_row in df_osm.iterrows():
            best_match = None
            min_dist = 50 # max 50 meters
            
            # Filter google places roughly by bounding box to speed up computation
            lat_margin = 0.001
            lng_margin = 0.001
            candidates = df_google[
                (df_google['lat'] >= osm_row['lat'] - lat_margin) & 
                (df_google['lat'] <= osm_row['lat'] + lat_margin) & 
                (df_google['lng'] >= osm_row['lng'] - lng_margin) & 
                (df_google['lng'] <= osm_row['lng'] + lng_margin)
            ]
            
            for g_idx, g_row in candidates.iterrows():
                dist = haversine(osm_row['lng'], osm_row['lat'], g_row['lng'], g_row['lat'])
                
                # If exact name match, be generous with distance (100m)
                name_match = (str(osm_row['name']) in str(g_row['name'])) or (str(g_row['name']) in str(osm_row['name']))
                
                if name_match and dist < 100:
                    if dist < min_dist or best_match is None:
                        min_dist = dist
                        best_match = g_row
                elif dist < 30: # Very close geographically even if name slightly differs
                    if dist < min_dist or best_match is None:
                        min_dist = dist
                        best_match = g_row
                        
            if best_match is not None:
                matched_count += 1
                row_data = osm_row.to_dict()
                row_data['google_id'] = best_match['id']
                row_data['google_name'] = best_match['name']
                row_data['rating'] = best_match['rating']
                row_data['user_rating_count'] = best_match['user_rating_count']
                row_data['price_level'] = best_match['price_level']
                final_data.append(row_data)
                
        print(f"Matched {matched_count} out of {len(df_osm)} OSM places with Google Places.")
        
        df_final = pd.DataFrame(final_data)
        df_final.to_csv("nagoya_final_dataset.csv", index=False, encoding="utf-8-sig")
        print("Saved final merged dataset to nagoya_final_dataset.csv")
        print("\n--- Final Label Distribution ---")
        print(df_final['target_closed_osm'].value_counts())
        
    except Exception as e:
        print(f"Error during merge: {e}")

if __name__ == "__main__":
    main()

import requests
import json
import time
import pandas as pd

API_KEY = ""
PLACES_URL = "https://places.googleapis.com/v1/places:searchText"

# 千種区のバウンディングボックス
BBOX = "35.145,136.920,35.195,136.995"
DATE_2025 = "2025-01-01T00:00:00Z"
DATE_2026 = "2026-01-01T00:00:00Z"

def fetch_osm_data(target_date):
    print(f"Fetching OSM data for {target_date}...")
    url = "https://overpass.kumi.systems/api/interpreter"
    
    query = f"""
    [out:json][timeout:50][date:"{target_date}"];
    (
      node["amenity"~"restaurant|cafe|bar|fast_food"]({BBOX});
      way["amenity"~"restaurant|cafe|bar|fast_food"]({BBOX});
    );
    out center;
    """
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
        'Content-Type': 'application/x-www-form-urlencoded'
    }
    
    response = requests.post(url, data={'data': query}, headers=headers)
    
    if response.status_code != 200:
        print(f"Error {response.status_code} on {target_date}: {response.text}")
        return {}
        
    data = response.json()
    elements = data.get("elements", [])
    
    restaurants = {}
    for el in elements:
        tags = el.get("tags", {})
        name = tags.get("name")
        if not name:
            continue
            
        osm_id = el.get("id")
        lat = el.get("lat") or el.get("center", {}).get("lat")
        lon = el.get("lon") or el.get("center", {}).get("lon")
        
        restaurants[osm_id] = {
            "osm_id": osm_id,
            "name": name,
            "lat": lat,
            "lon": lon,
            "category": tags.get("amenity")
        }
        
    print(f"Found {len(restaurants)} named restaurants.")
    return restaurants

def build_osm_comparison():
    data_2025 = fetch_osm_data(DATE_2025)
    data_2026 = fetch_osm_data(DATE_2026)
    
    if not data_2025:
        print("Failed to fetch 2025 data.")
        return []
        
    dataset = []
    # 2025年の全店舗をベースにする
    for osm_id, info in data_2025.items():
        # 2026年にも存在していれば 0 (存続), 存在していなければ 1 (廃業)
        is_closed_osm = 0 if osm_id in data_2026 else 1
        
        dataset.append({
            "osm_id": osm_id,
            "name": info["name"],
            "lat": info["lat"],
            "lon": info["lon"],
            "category": info["category"],
            "target_closed_osm": is_closed_osm
        })
        
    return dataset

def append_google_features(dataset):
    print(f"Fetching Google Places features for {len(dataset)} locations...")
    
    headers = {
        "X-Goog-Api-Key": API_KEY,
        "X-Goog-FieldMask": "places.id,places.displayName,places.businessStatus,places.rating,places.userRatingCount,places.priceLevel",
        "Content-Type": "application/json"
    }
    
    for idx, row in enumerate(dataset):
        # 進行状況の表示
        if (idx + 1) % 10 == 0:
            print(f"Processing {idx + 1}/{len(dataset)}...")
            
        query = f"{row['name']} 名古屋市千種区"
        payload = {
            "textQuery": query,
            "pageSize": 1,
            "locationBias": {
                "circle": {
                    "center": {"latitude": row["lat"], "longitude": row["lon"]},
                    "radius": 200.0
                }
            }
        }
        
        try:
            res = requests.post(PLACES_URL, headers=headers, json=payload)
            if res.status_code == 200:
                places = res.json().get("places", [])
                if places:
                    p = places[0]
                    row["google_place_id"] = p.get("id")
                    row["google_status"] = p.get("businessStatus")
                    row["rating"] = p.get("rating")
                    row["user_rating_count"] = p.get("userRatingCount")
                    row["price_level"] = p.get("priceLevel")
                else:
                    row["google_status"] = "NOT_FOUND"
            else:
                row["google_status"] = "API_ERROR"
        except Exception as e:
            row["google_status"] = "REQ_ERROR"
            
        # レートリミット回避
        time.sleep(1)
        
    return dataset

def main():
    # 1. OSMの2025年と2026年を比較
    osm_dataset = build_osm_comparison()
    if not osm_dataset:
        return
        
    df_osm = pd.DataFrame(osm_dataset)
    print("\n--- OSM Based Closure ---")
    print(df_osm['target_closed_osm'].value_counts())
    
    # 2. Google Places APIで特徴量を追加 (API利用制限に注意し、最大300件程度でテストする場合はスライスする)
    # df_osm = df_osm.head(300) # テスト用の場合はコメントアウト解除
    
    final_dataset = append_google_features(df_osm.to_dict('records'))
    df_final = pd.DataFrame(final_dataset)
    
    # 3. CSVに保存
    output_file = "chikusa_2025_2026_dataset.csv"
    df_final.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"\nSaved final dataset to {output_file}")
    
    if 'google_status' in df_final.columns:
        print("\n--- Google Status Counts ---")
        print(df_final['google_status'].value_counts())

if __name__ == "__main__":
    main()

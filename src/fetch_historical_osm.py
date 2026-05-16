import requests
import json
import time
import pandas as pd

API_KEY = ""
PLACES_URL = "https://places.googleapis.com/v1/places:searchText"

# 千種区の大まかなバウンディングボックス
BBOX = "35.145,136.920,35.195,136.995"
TARGET_DATE = "2022-01-01T00:00:00Z" # 2022年（コロナ後）当時のデータを取得

def fetch_osm_historical_restaurants():
    print(f"Fetching historical OSM data for date: {TARGET_DATE}...")
    overpass_url = "https://overpass-api.de/api/interpreter"
    
    overpass_query = f"""
    [out:json][timeout:50][date:"{TARGET_DATE}"];
    (
      node["amenity"~"restaurant|cafe|bar|fast_food"]({BBOX});
      way["amenity"~"restaurant|cafe|bar|fast_food"]({BBOX});
    );
    out center;
    """

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    response = requests.post(overpass_url, data=overpass_query.encode('utf-8'), headers=headers)
    
    if response.status_code != 200:
        print(f"OSM Error: {response.status_code}")
        return []
        
    data = response.json()
    elements = data.get("elements", [])
    
    restaurants = []
    for el in elements:
        tags = el.get("tags", {})
        name = tags.get("name")
        
        # 名前がない店舗はGoogleAPIで検索できないので除外
        if not name:
            continue
            
        lat = el.get("lat") or el.get("center", {}).get("lat")
        lon = el.get("lon") or el.get("center", {}).get("lon")
        
        restaurants.append({
            "osm_id": el.get("id"),
            "name": name,
            "lat": lat,
            "lon": lon,
            "category": tags.get("amenity")
        })
        
    print(f"Found {len(restaurants)} named restaurants in OSM as of {TARGET_DATE}.")
    return restaurants

def check_google_places(osm_restaurants):
    print("Cross-referencing with Google Places API...")
    
    headers = {
        "X-Goog-Api-Key": API_KEY,
        "X-Goog-FieldMask": "places.id,places.displayName,places.businessStatus,places.rating,places.userRatingCount,places.priceLevel,places.formattedAddress",
        "Content-Type": "application/json"
    }
    
    results = []
    
    # API上限に配慮し、今回はデモとして最大300件に制限
    for idx, r in enumerate(osm_restaurants[:300]):
        print(f"[{idx+1}] Checking: {r['name']} ...")
        
        # 店名と「千種区」を合わせたクエリ
        query = f"{r['name']} 名古屋市千種区"
        
        payload = {
            "textQuery": query,
            "pageSize": 3, # 関連性が高い上位3件のみ取得
            "locationBias": {
                "circle": {
                    "center": {
                        "latitude": r["lat"],
                        "longitude": r["lon"]
                    },
                    "radius": 500.0 # 500m以内のバイアス
                }
            }
        }
        
        response = requests.post(PLACES_URL, headers=headers, json=payload)
        
        if response.status_code == 200:
            data = response.json()
            places = data.get("places", [])
            
            if places:
                # 一番マッチしたものを採用
                p = places[0]
                results.append({
                    "osm_id": r["osm_id"],
                    "osm_name": r["name"],
                    "google_place_id": p.get("id"),
                    "google_name": p.get("displayName", {}).get("text"),
                    "business_status": p.get("businessStatus"),
                    "rating": p.get("rating"),
                    "user_rating_count": p.get("userRatingCount"),
                    "price_level": p.get("priceLevel"),
                    "address": p.get("formattedAddress"),
                    "lat": r["lat"],
                    "lon": r["lon"]
                })
            else:
                # 完全にヒットしない場合（APIからも消滅しているなど）
                results.append({
                    "osm_id": r["osm_id"],
                    "osm_name": r["name"],
                    "google_place_id": None,
                    "google_name": None,
                    "business_status": "NOT_FOUND_IN_GOOGLE",
                    "rating": None,
                    "user_rating_count": None,
                    "price_level": None,
                    "address": None,
                    "lat": r["lat"],
                    "lon": r["lon"]
                })
        else:
            print("Google API Error")
            
        time.sleep(1) # レートリミット回避
        
    return results

def main():
    # 1. OSMから過去のレストランを取得
    osm_data = fetch_osm_historical_restaurants()
    
    if not osm_data:
        return
        
    # 2. Google Places APIで現在の状態と特徴量を取得
    final_data = check_google_places(osm_data)
    
    # 3. データフレーム化と保存
    df = pd.DataFrame(final_data)
    output_file = "chikusa_historical_merged.csv"
    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"\nSaved {len(df)} records to {output_file}")
    
    # 簡単な集計
    print("\n--- Status Counts ---")
    print(df['business_status'].value_counts())

if __name__ == "__main__":
    main()

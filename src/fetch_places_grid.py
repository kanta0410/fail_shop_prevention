import os
import json
import time
import requests
import pandas as pd

API_KEY = ""
URL = "https://places.googleapis.com/v1/places:searchText"

# 千種区のざっくりとしたバウンディングボックス (緯度経度)
# 南西端
MIN_LAT = 35.145
MIN_LNG = 136.920
# 北東端
MAX_LAT = 35.195
MAX_LNG = 136.995

# グリッドの分割サイズ (約1km四方)
STEP_LAT = 0.010
STEP_LNG = 0.010

# APIリクエスト数の上限 ($200枠を絶対に超えないための安全装置)
# 1000リクエスト * $0.032 = $32.0 (絶対に$200を超えない)
MAX_API_REQUESTS = 1000
api_request_count = 0

def fetch_places_in_rect(low_lat, low_lng, high_lat, high_lng):
    global api_request_count
    
    headers = {
        "X-Goog-Api-Key": API_KEY,
        "X-Goog-FieldMask": "places.id,places.displayName,places.businessStatus,places.rating,places.userRatingCount,places.priceLevel,places.formattedAddress,places.location,nextPageToken",
        "Content-Type": "application/json"
    }

    places = []
    page_token = ""
    
    # 1つの短形エリアにつき最大5ページ(100件)まで取得
    for _ in range(5):
        if api_request_count >= MAX_API_REQUESTS:
            print("API request limit reached! Stopping.")
            break
            
        payload = {
            "textQuery": "飲食店", # 飲食店全般
            "pageSize": 20,
            "locationRestriction": {
                "rectangle": {
                    "low": {
                        "latitude": low_lat,
                        "longitude": low_lng
                    },
                    "high": {
                        "latitude": high_lat,
                        "longitude": high_lng
                    }
                }
            }
        }
        
        if page_token:
            payload["pageToken"] = page_token
            
        response = requests.post(URL, headers=headers, json=payload)
        api_request_count += 1
        
        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            print(response.text)
            break
            
        data = response.json()
        
        if "places" in data:
            places.extend(data["places"])
            
        page_token = data.get("nextPageToken")
        if not page_token:
            break
            
        time.sleep(1.5) # レートリミット回避
        
    return places

def main():
    print("Starting Grid Search for Chikusa-ku...")
    all_places = {}
    
    lat = MIN_LAT
    grid_count = 0
    
    while lat < MAX_LAT:
        lng = MIN_LNG
        while lng < MAX_LNG:
            if api_request_count >= MAX_API_REQUESTS:
                break
                
            grid_count += 1
            print(f"Searching grid {grid_count}... (Lat: {lat:.3f}, Lng: {lng:.3f})")
            
            places = fetch_places_in_rect(lat, lng, lat + STEP_LAT, lng + STEP_LNG)
            
            for p in places:
                pid = p.get("id")
                if pid and pid not in all_places:
                    all_places[pid] = {
                        "id": pid,
                        "name": p.get("displayName", {}).get("text"),
                        "business_status": p.get("businessStatus"),
                        "rating": p.get("rating"),
                        "user_rating_count": p.get("userRatingCount"),
                        "price_level": p.get("priceLevel"),
                        "address": p.get("formattedAddress"),
                        "lat": p.get("location", {}).get("latitude"),
                        "lng": p.get("location", {}).get("longitude")
                    }
            lng += STEP_LNG
        lat += STEP_LAT

    print(f"\n--- Scraping Completed ---")
    print(f"Total API Requests used: {api_request_count}")
    print(f"Total unique places found: {len(all_places)}")
    
    df = pd.DataFrame(list(all_places.values()))
    
    output_file = "chikusa_restaurants_grid_raw.csv"
    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"Saved to {output_file}")
    
    print("\n--- Basic Information ---")
    print(df['business_status'].value_counts(dropna=False))

if __name__ == "__main__":
    main()

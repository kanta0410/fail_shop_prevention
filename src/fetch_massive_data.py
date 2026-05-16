import os
import time
import requests
import pandas as pd

API_KEY = ""
URL = "https://places.googleapis.com/v1/places:searchText"

# 絶対に400ドルを超えないための安全リミッター ($384 = 12,000 requests)
MAX_API_REQUESTS = 12000
api_request_count = 0

# 取得対象エリアの定義
# 分析しやすく、データ密度が高い3大都市圏を指定
REGIONS = [
    {
        "name": "nagoya_all",
        "min_lat": 35.030, "max_lat": 35.260,
        "min_lng": 136.790, "max_lng": 137.050,
        "step": 0.005 # 約500m四方
    },
    {
        "name": "osaka_city",
        "min_lat": 34.580, "max_lat": 34.770,
        "min_lng": 135.370, "max_lng": 135.600,
        "step": 0.005
    },
    {
        "name": "tokyo_23wards",
        "min_lat": 35.520, "max_lat": 35.820,
        "min_lng": 139.550, "max_lng": 139.920,
        "step": 0.005
    }
]

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
            return places # 上限到達でリターン
            
        payload = {
            "textQuery": "飲食店", 
            "pageSize": 20,
            "locationRestriction": {
                "rectangle": {
                    "low": {"latitude": low_lat, "longitude": low_lng},
                    "high": {"latitude": high_lat, "longitude": high_lng}
                }
            }
        }
        
        if page_token:
            payload["pageToken"] = page_token
            
        try:
            response = requests.post(URL, headers=headers, json=payload, timeout=10)
            api_request_count += 1
            
            if response.status_code != 200:
                print(f"API Error {response.status_code}: {response.text}")
                break
                
            data = response.json()
            if "places" in data:
                places.extend(data["places"])
                
            page_token = data.get("nextPageToken")
            if not page_token:
                break
                
        except Exception as e:
            print(f"Request Exception: {e}")
            break
            
        time.sleep(0.8) # レートリミット回避
        
    return places

def process_region(region):
    global api_request_count
    name = region["name"]
    print(f"\n========== Starting Region: {name} ==========")
    
    all_places = {}
    lat = region["min_lat"]
    step = region["step"]
    
    # 進行状況表示用
    lat_steps = int((region["max_lat"] - region["min_lat"]) / step) + 1
    lng_steps = int((region["max_lng"] - region["min_lng"]) / step) + 1
    total_grids = lat_steps * lng_steps
    grid_count = 0
    
    while lat < region["max_lat"]:
        lng = region["min_lng"]
        while lng < region["max_lng"]:
            if api_request_count >= MAX_API_REQUESTS:
                print(f"HARD LIMIT REACHED! Stopping {name} immediately.")
                break
                
            grid_count += 1
            if grid_count % 50 == 0 or grid_count == 1:
                print(f"[{name}] Grid {grid_count}/{total_grids} | API Req: {api_request_count}/{MAX_API_REQUESTS}")
            
            places = fetch_places_in_rect(lat, lng, lat + step, lng + step)
            
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
                        "lng": p.get("location", {}).get("longitude"),
                        "region": name
                    }
            lng += step
        
        if api_request_count >= MAX_API_REQUESTS:
            break
        lat += step

    print(f"Finished {name}. Unique places found: {len(all_places)}")
    if len(all_places) > 0:
        df = pd.DataFrame(list(all_places.values()))
        output_file = f"{name}_massive_raw.csv"
        df.to_csv(output_file, index=False, encoding="utf-8-sig")
        print(f"Saved region data to {output_file}")

def main():
    print(f"Starting massive scraping with hard limit: {MAX_API_REQUESTS} requests (${MAX_API_REQUESTS * 0.032:.2f})")
    
    for region in REGIONS:
        if api_request_count >= MAX_API_REQUESTS:
            print("Limit reached, skipping remaining regions.")
            break
        process_region(region)
        
    print("\n=========================================")
    print("ALL SCRAPING COMPLETED OR LIMIT REACHED.")
    print(f"Total API Requests used: {api_request_count}")
    print(f"Estimated Cost: ${api_request_count * 0.032:.2f}")

if __name__ == "__main__":
    main()

import os
import time
import json
import requests
import pandas as pd
from dotenv import load_dotenv

# --- 設定 ---
INPUT_CSV = 'data/raw/nagoya_all_massive_raw.csv'
OUTPUT_CSV = 'data/raw/nagoya_google_2026_05.csv'

# 1APIキーあたりの最大リクエスト回数（安全装置）
# 1リクエスト0.025ドル相当と仮定し、170ドル分(約6800回)の手前である6500回でストップ
MAX_REQUESTS_PER_KEY = 6500 

# Field Mask（取得する項目を絞り込む）
FIELD_MASK = 'id,businessStatus,rating,userRatingCount,currentOpeningHours'

def load_api_keys():
    """環境変数からAPIキーを読み込む"""
    load_dotenv()
    keys = []
    # 複数キーがある場合はリスト化
    key1 = os.getenv('GOOGLE_PLACES_API_KEY_1')
    key2 = os.getenv('GOOGLE_PLACES_API_KEY_2')
    
    if key1: keys.append(key1)
    if key2: keys.append(key2)
    
    if not keys:
        raise ValueError("APIキーが.envファイルに設定されていません。")
    return keys

def fetch_place_details(place_id, api_key):
    """Place Details API (New) を呼び出す"""
    url = f"https://places.googleapis.com/v1/places/{place_id}"
    headers = {
        'X-Goog-Api-Key': api_key,
        'X-Goog-FieldMask': FIELD_MASK
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        
        # 完全に消去されている場合（NotFound）
        if response.status_code == 404:
            return {"status": "NOT_FOUND"}
            
        response.raise_for_status()
        return {"status": "SUCCESS", "data": response.json()}
        
    except requests.exceptions.RequestException as e:
        print(f"\n[ERROR] Request failed for {place_id}: {e}")
        return {"status": "ERROR"}

def main():
    print("=" * 60)
    print("Starting Google Places API (New) Data Fetch - 2026/05")
    print("=" * 60)
    
    api_keys = load_api_keys()
    print(f"Loaded {len(api_keys)} API key(s).")
    
    # 既存の入力データを読み込む（パースエラー行はスキップ）
    df = pd.read_csv(INPUT_CSV, on_bad_lines='skip')
    total_places = len(df)
    print(f"Total places to fetch: {total_places}")
    
    # すでに途中まで取得している場合は続きから再開する（セーブ機能）
    results = []
    processed_ids = set()
    if os.path.exists(OUTPUT_CSV):
        df_existing = pd.read_csv(OUTPUT_CSV)
        # FETCH_ERROR 以外のレコードのみを有効な処理済みとして残す
        results = [r for r in df_existing.to_dict('records') if r.get('business_status') != 'FETCH_ERROR']
        processed_ids = set(r['id'] for r in results)
        print(f"Resuming from existing file: {len(processed_ids)} already processed successfully. (Failed requests will be retried)")
    
    current_key_idx = 0
    request_count = 0
    
    print("\nStarting fetch loop...")
    for i, row in df.iterrows():
        place_id = row['id']
        
        # すでに処理済みならスキップ
        if place_id in processed_ids:
            continue
            
        # 安全リミッターチェック
        if request_count >= MAX_REQUESTS_PER_KEY:
            print(f"\n[LIMIT REACHED] Key {current_key_idx + 1} reached {request_count} requests.")
            current_key_idx += 1
            request_count = 0 # リセット
            
            if current_key_idx >= len(api_keys):
                print("\n[STOP] All available API keys have been exhausted. Stopping safely.")
                break
            else:
                print(f"Switched to API Key {current_key_idx + 1}.")
        
        # API叩く
        api_key = api_keys[current_key_idx]
        res = fetch_place_details(place_id, api_key)
        request_count += 1
        
        # 結果の解析
        result_dict = {'id': place_id}
        
        if res['status'] == "NOT_FOUND":
            result_dict['business_status'] = 'NOT_FOUND_CLOSED'
            result_dict['rating'] = None
            result_dict['user_rating_count'] = 0
            result_dict['has_opening_hours'] = 0
        elif res['status'] == "SUCCESS":
            data = res['data']
            result_dict['business_status'] = data.get('businessStatus', 'UNKNOWN')
            result_dict['rating'] = data.get('rating', None)
            result_dict['user_rating_count'] = data.get('userRatingCount', 0)
            result_dict['has_opening_hours'] = 1 if 'currentOpeningHours' in data else 0
            # 必要なら opening_hours の詳細（深夜営業フラグなど）をここでパースして追加可能
        else:
            # ERROR等
            result_dict['business_status'] = 'FETCH_ERROR'
            
        results.append(result_dict)
        
        # 進捗表示
        if (request_count) % 50 == 0:
            print(f"Processed {len(results)} / {total_places} | Current Key: {current_key_idx + 1}")
            
        # 毎回のセーブ（安全のため細かく上書き保存）
        if len(results) % 100 == 0:
            pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
            
        # API制限回避のためのわずかなスリープ
        time.sleep(0.05)
        
    # 最終セーブ
    pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
    print("\nFetch process completed safely and saved!")

if __name__ == '__main__':
    main()

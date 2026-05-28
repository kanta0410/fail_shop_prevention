import os
import sys
import json
import time
import requests
import pandas as pd

# 名古屋市の緯度経度バウンディングボックス
MIN_LAT = 35.030
MAX_LAT = 35.260
MIN_LNG = 136.790
MAX_LNG = 137.050

# 最小グリッド単位
BASE_STEP = 0.0025

# 予算管理用リミッター
MAX_REQ_KEY1 = 4800 # キー1の上限（約26,000円〜27,000円分）
MAX_REQ_KEY2 = 7300 # キー2の上限（40,000円分）

# ファイルパス
CHECKPOINT_PATH = "data/nagoya_variable_grid_checkpoint.json"
OUTPUT_CSV_PATH = "data/raw/nagoya_restaurants_variable_grid.csv"
DOTENV_PATH = ".env"

# URL
PLACES_URL = "https://places.googleapis.com/v1/places:searchText"

def load_api_keys():
    """.env ファイルから API キーをパースして取得する (python-dotenv不要)"""
    keys = {"key1": None, "key2": None}
    if not os.path.exists(DOTENV_PATH):
        print(f"Warning: {DOTENV_PATH} not found.")
        return keys
        
    with open(DOTENV_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                # コメントアウトされている行も、キーの定義であれば取得できるようにする
                # (ユーザーがコメントアウトしている可能性があるため、#を除去してパースを試みる)
                clean_line = line.lstrip("#").strip()
                if "=" in clean_line:
                    line = clean_line
                else:
                    continue
            
            if "=" in line:
                name, val = line.split("=", 1)
                name = name.strip()
                val = val.strip().strip('"').strip("'")
                if name == "GOOGLE_PLACES_API_KEY_1":
                    keys["key1"] = val
                elif name == "GOOGLE_PLACES_API_KEY_2":
                    keys["key2"] = val
                    
    # もし環境変数に直接設定されている場合はそちらを優先
    if os.environ.get("GOOGLE_PLACES_API_KEY_1"):
        keys["key1"] = os.environ.get("GOOGLE_PLACES_API_KEY_1")
    if os.environ.get("GOOGLE_PLACES_API_KEY_2"):
        keys["key2"] = os.environ.get("GOOGLE_PLACES_API_KEY_2")
        
    return keys

def get_tier(lat, lng):
    """緯度経度からその場所の密度ティア（高・中・低）を判定する"""
    # 高密度エリア (step = 0.0025): 栄・名駅周辺中心部
    if 35.130 <= lat < 35.195 and 136.870 <= lng < 136.980:
        return 1
    # 中密度エリア (step = 0.0050): 中心部の周辺地域
    elif 35.100 <= lat < 35.220 and 136.830 <= lng < 137.010:
        return 2
    # 低密度エリア (step = 0.0100): 郊外
    else:
        return 3

def generate_variable_rects():
    """
    名古屋市全体を隙間なく、かつ重複なくカバーする可変サイズの矩形リストを生成する。
    メッシュマージアルゴリズムを使用。
    """
    lat_steps = int(round((MAX_LAT - MIN_LAT) / BASE_STEP))
    lng_steps = int(round((MAX_LNG - MIN_LNG) / BASE_STEP))
    
    # 訪問済みフラグ配列
    visited = [[False for _ in range(lng_steps)] for _ in range(lat_steps)]
    rects = []
    
    for i in range(lat_steps):
        lat = MIN_LAT + i * BASE_STEP
        for j in range(lng_steps):
            lng = MIN_LNG + j * BASE_STEP
            
            if visited[i][j]:
                continue
                
            tier = get_tier(lat, lng)
            
            if tier == 1:
                # 高密度: 1x1 の矩形
                rects.append({
                    "low_lat": lat,
                    "low_lng": lng,
                    "high_lat": lat + BASE_STEP,
                    "high_lng": lng + BASE_STEP,
                    "tier": 1
                })
                visited[i][j] = True
                
            elif tier == 2:
                # 中密度: 可能な限り 2x2 でマージ
                can_merge_2x2 = True
                if i + 1 < lat_steps and j + 1 < lng_steps:
                    for di in range(2):
                        for dj in range(2):
                            if visited[i+di][j+dj] or get_tier(lat + di*BASE_STEP, lng + dj*BASE_STEP) != 2:
                                can_merge_2x2 = False
                                break
                        if not can_merge_2x2:
                            break
                else:
                    can_merge_2x2 = False
                    
                if can_merge_2x2:
                    rects.append({
                        "low_lat": lat,
                        "low_lng": lng,
                        "high_lat": lat + BASE_STEP * 2,
                        "high_lng": lng + BASE_STEP * 2,
                        "tier": 2
                    })
                    for di in range(2):
                        for dj in range(2):
                            visited[i+di][j+dj] = True
                else:
                    # マージできない境界部分は 1x1 で処理
                    rects.append({
                        "low_lat": lat,
                        "low_lng": lng,
                        "high_lat": lat + BASE_STEP,
                        "high_lng": lng + BASE_STEP,
                        "tier": 2
                    })
                    visited[i][j] = True
                    
            elif tier == 3:
                # 低密度: 可能な限り 4x4 でマージ
                can_merge_4x4 = True
                if i + 3 < lat_steps and j + 3 < lng_steps:
                    for di in range(4):
                        for dj in range(4):
                            if visited[i+di][j+dj] or get_tier(lat + di*BASE_STEP, lng + dj*BASE_STEP) != 3:
                                can_merge_4x4 = False
                                break
                        if not can_merge_4x4:
                            break
                else:
                    can_merge_4x4 = False
                    
                if can_merge_4x4:
                    rects.append({
                        "low_lat": lat,
                        "low_lng": lng,
                        "high_lat": lat + BASE_STEP * 4,
                        "high_lng": lng + BASE_STEP * 4,
                        "tier": 3
                    })
                    for di in range(4):
                        for dj in range(4):
                            visited[i+di][j+dj] = True
                else:
                    # 4x4でマージできない場合は 2x2 を試みる
                    can_merge_2x2 = True
                    if i + 1 < lat_steps and j + 1 < lng_steps:
                        for di in range(2):
                            for dj in range(2):
                                if visited[i+di][j+dj] or get_tier(lat + di*BASE_STEP, lng + dj*BASE_STEP) != 3:
                                    can_merge_2x2 = False
                                    break
                            if not can_merge_2x2:
                                break
                    else:
                        can_merge_2x2 = False
                        
                    if can_merge_2x2:
                        rects.append({
                            "low_lat": lat,
                            "low_lng": lng,
                            "high_lat": lat + BASE_STEP * 2,
                            "high_lng": lng + BASE_STEP * 2,
                            "tier": 3
                        })
                        for di in range(2):
                            for dj in range(2):
                                visited[i+di][j+dj] = True
                    else:
                        # それでもダメなら 1x1
                        rects.append({
                            "low_lat": lat,
                            "low_lng": lng,
                            "high_lat": lat + BASE_STEP,
                            "high_lng": lng + BASE_STEP,
                            "tier": 3
                        })
                        visited[i][j] = True
                        
    return rects

def fetch_places_in_rect(low_lat, low_lng, high_lat, high_lng, api_key):
    """指定した矩形エリア内の飲食店を取得する"""
    headers = {
        "X-Goog-Api-Key": api_key,
        # ユーザー希望のフィールドを指定 (Enterprise 課金となる)
        "X-Goog-FieldMask": "places.id,places.displayName,places.businessStatus,places.rating,places.userRatingCount,places.priceLevel,places.location,nextPageToken",
        "Content-Type": "application/json"
    }

    places = []
    page_token = ""
    
    # 1つのエリアにつき最大5ページ(100件)まで取得
    for page in range(5):
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
            response = requests.post(PLACES_URL, headers=headers, json=payload, timeout=15)
            
            if response.status_code == 403:
                print("API Key Error (403 Forbidden). Possibly out of quota or invalid key.")
                return None, "KEY_EXHAUSTED"
            elif response.status_code != 200:
                print(f"API Error {response.status_code}: {response.text}")
                return None, "ERROR"
                
            data = response.json()
            if "places" in data:
                places.extend(data["places"])
                
            page_token = data.get("nextPageToken")
            if not page_token:
                break
                
            time.sleep(1.0) # レートリミット回避
            
        except Exception as e:
            print(f"Request Exception: {e}")
            return None, "EXCEPTION"
            
    return places, "SUCCESS"

def save_checkpoint(completed_grids, collected_places, req_key1, req_key2, active_key_idx):
    """進捗状況をチェックポイントファイルに保存する"""
    # ディレクトリ作成
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    
    data = {
        "completed_grids": list(completed_grids),
        "collected_places": collected_places,
        "req_key1": req_key1,
        "req_key2": req_key2,
        "active_key_index": active_key_idx
    }
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    # 中間CSVも保存しておく
    save_to_csv(collected_places)

def load_checkpoint():
    """チェックポイントファイルを読み込む"""
    if os.path.exists(CHECKPOINT_PATH):
        try:
            with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                return (
                    set(data.get("completed_grids", [])),
                    data.get("collected_places", {}),
                    data.get("req_key1", 0),
                    data.get("req_key2", 0),
                    data.get("active_key_index", 1)
                )
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Starting fresh.")
    return set(), {}, 0, 0, 1

def save_to_csv(collected_places):
    """収集したデータをCSVファイルに保存する"""
    if not collected_places:
        return
        
    flat_data = []
    for pid, p in collected_places.items():
        flat_data.append({
            "id": pid,
            "name": p.get("name"),
            "business_status": p.get("business_status"),
            "rating": p.get("rating"),
            "user_rating_count": p.get("user_rating_count"),
            "price_level": p.get("price_level"),
            "lat": p.get("lat"),
            "lng": p.get("lng")
        })
        
    df = pd.DataFrame(flat_data)
    os.makedirs(os.path.dirname(OUTPUT_CSV_PATH), exist_ok=True)
    df.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")
    print(f"Saved {len(df)} unique records to {OUTPUT_CSV_PATH}")

def main():
    test_mode = "--test" in sys.argv
    if test_mode:
        print("!!! RUNNING IN TEST MODE (Max 3 requests per key) !!!")

    # 1. APIキーのロード
    keys = load_api_keys()
    key1 = keys["key1"]
    key2 = keys["key2"]
    
    print("=== API Key Status ===")
    print(f"Key 1 Available: {key1 is not None} (Length: {len(key1) if key1 else 0})")
    print(f"Key 2 Available: {key2 is not None} (Length: {len(key2) if key2 else 0})")
    
    if not key1 and not key2:
        print("Error: No API keys found in .env. Please define GOOGLE_PLACES_API_KEY_1 or 2.")
        return
        
    # 2. 矩形グリッドリストの生成
    rects = generate_variable_rects()
    total_grids = len(rects)
    print(f"\nGenerated {total_grids} variable grids covering Nagoya-shi.")
    
    # ティアごとのグリッド数の集計
    tier_counts = {1: 0, 2: 0, 3: 0}
    for r in rects:
        tier_counts[r["tier"]] += 1
    print(f"Grid Density Breakdown -> Tier 1 (High/250m): {tier_counts[1]}, Tier 2 (Med/500m): {tier_counts[2]}, Tier 3 (Low/1km): {tier_counts[3]}")

    # 3. チェックポイントのロード
    fresh_run = "--fresh" in sys.argv
    if fresh_run:
        print("!!! FRESH RUN: Ignoring existing checkpoints !!!")
        completed_grids, collected_places, req_key1, req_key2, active_key_idx = set(), {}, 0, 0, 1
    else:
        completed_grids, collected_places, req_key1, req_key2, active_key_idx = load_checkpoint()
    
    # テスト時の動的上限設定
    max_key1_limit = 3 if test_mode else MAX_REQ_KEY1
    max_key2_limit = 3 if test_mode else MAX_REQ_KEY2
    
    print(f"\n=== Progress Status ===")
    print(f"Loaded progress: {len(completed_grids)}/{total_grids} grids completed.")
    print(f"Unique places collected so far: {len(collected_places)}")
    print(f"Current Request Counts -> Key 1: {req_key1}/{max_key1_limit}, Key 2: {req_key2}/{max_key2_limit}")
    print(f"Current Active Key Index: {active_key_idx}")

    # 現在稼働中のAPIキーを設定
    current_key = key1 if active_key_idx == 1 else key2
    
    # メインループ
    save_counter = 0
    
    for idx, r in enumerate(rects):
        grid_id = f"{r['low_lat']:.4f}_{r['low_lng']:.4f}"
        
        # 既に完了しているグリッドはスキップ
        if grid_id in completed_grids:
            continue
            
        print(f"\n[{idx+1}/{total_grids}] Scanning Grid (Lat: {r['low_lat']:.4f}~{r['high_lat']:.4f}, Lng: {r['low_lng']:.4f}~{r['high_lng']:.4f}) | Tier {r['tier']}")
        
        # API上限の事前チェック
        if active_key_idx == 1 and req_key1 >= max_key1_limit:
            print(f"\n>>> API Key 1 Limit ({max_key1_limit}) reached. <<<")
            if key2:
                print("Switching active key to API Key 2...")
                active_key_idx = 2
                current_key = key2
                save_checkpoint(completed_grids, collected_places, req_key1, req_key2, active_key_idx)
            else:
                print("API Key 2 is not configured. Stopping execution for key 1 limit.")
                break
                
        if active_key_idx == 2 and req_key2 >= MAX_REQ_KEY2:
            print(f"\n>>> API Key 2 Limit ({MAX_REQ_KEY2}) reached. Stopping safely. <<<")
            break
            
        # APIリクエストの実行
        # (NextPageTokenによる複数ページ取得もこの中で処理されますが、リクエスト数は1回ごとにカウントしたい)
        # 簡略化のため、API呼び出しを内部で行い、リクエストのたびに上限をチェックする
        headers = {
            "X-Goog-Api-Key": current_key,
            "X-Goog-FieldMask": "places.id,places.displayName,places.businessStatus,places.rating,places.userRatingCount,places.priceLevel,places.location,nextPageToken",
            "Content-Type": "application/json"
        }
        
        places_in_grid = []
        page_token = ""
        key_exhausted = False
        
        for page in range(5):
            # ループの先頭で上限チェック
            if active_key_idx == 1 and req_key1 >= max_key1_limit:
                print(f"API Key 1 Limit hit during pagination. Switching to Key 2...")
                if key2:
                    active_key_idx = 2
                    current_key = key2
                    headers["X-Goog-Api-Key"] = current_key
                else:
                    print("API Key 2 is not configured. Breaking pagination.")
                    break
                    
            if active_key_idx == 2 and req_key2 >= max_key2_limit:
                print(f"API Key 2 Limit hit during pagination. Breaking.")
                break
                
            payload = {
                "textQuery": "飲食店",
                "pageSize": 20,
                "locationRestriction": {
                    "rectangle": {
                        "low": {"latitude": r["low_lat"], "longitude": r["low_lng"]},
                        "high": {"latitude": r["high_lat"], "longitude": r["high_lng"]}
                    }
                }
            }
            if page_token:
                payload["pageToken"] = page_token
                
            try:
                # リクエスト送信
                response = requests.post(PLACES_URL, headers=headers, json=payload, timeout=15)
                
                # カウンター加算
                if active_key_idx == 1:
                    req_key1 += 1
                else:
                    req_key2 += 1
                    
                if response.status_code == 403 or response.status_code == 429:
                    print(f"API Key {active_key_idx} quota exhausted or blocked (HTTP {response.status_code}).")
                    key_exhausted = True
                    break
                elif response.status_code != 200:
                    print(f"API Error {response.status_code}: {response.text}")
                    break
                    
                data = response.json()
                if "places" in data:
                    places_in_grid.extend(data["places"])
                    
                page_token = data.get("nextPageToken")
                if not page_token:
                    break
                    
                time.sleep(1.0) # レートリミット回避
                
            except Exception as e:
                print(f"Request Exception: {e}")
                break
                
        if key_exhausted:
            if active_key_idx == 1 and key2:
                print("Quota exhausted for Key 1. Swapping to Key 2 immediately...")
                active_key_idx = 2
                current_key = key2
                # スキャンをこのグリッドの最初からやり直すためにスキップせず、再度ループを回す
                save_checkpoint(completed_grids, collected_places, req_key1, req_key2, active_key_idx)
                continue
            else:
                print("Active Key quota exhausted. No fallback available. Stopping.")
                break
                
        # 取得できたプレイスの登録
        for p in places_in_grid:
            pid = p.get("id")
            if pid and pid not in collected_places:
                collected_places[pid] = {
                    "id": pid,
                    "name": p.get("displayName", {}).get("text"),
                    "business_status": p.get("businessStatus"),
                    "rating": p.get("rating"),
                    "user_rating_count": p.get("userRatingCount"),
                    "price_level": p.get("priceLevel"),
                    "lat": p.get("location", {}).get("latitude"),
                    "lng": p.get("location", {}).get("longitude")
                }
                
        print(f"-> Found {len(places_in_grid)} places (Unique total: {len(collected_places)}) | Req count -> Key 1: {req_key1}, Key 2: {req_key2}")
        
        # 走査完了としてマーク
        completed_grids.add(grid_id)
        
        # 5グリッドごとにチェックポイントをディスクに保存
        save_counter += 1
        if save_counter % 5 == 0:
            save_checkpoint(completed_grids, collected_places, req_key1, req_key2, active_key_idx)
            print("Checkpoint saved.")
            
    # ループ終了後の最終保存
    save_checkpoint(completed_grids, collected_places, req_key1, req_key2, active_key_idx)
    print("\n=== RUN COMPLETED ===")
    print(f"Completed grids: {len(completed_grids)}/{total_grids}")
    print(f"Total Unique Places collected: {len(collected_places)}")
    print(f"Final Request Counts -> Key 1: {req_key1}/{max_key1_limit}, Key 2: {req_key2}/{max_key2_limit}")
    
    if len(completed_grids) == total_grids:
        # チェックポイントファイルをクリーンアップ（オプション。今回は履歴として残す）
        print("All grids scanned successfully.")

if __name__ == "__main__":
    main()

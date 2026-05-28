import os
import sys
import json
import time
import re
import requests
import pandas as pd

# ファイルパス定義
BASE_CSV_PATH = "data/raw/nagoya_google_2026_05_week2.csv"
CHECKPOINT_PATH = "data/enrich_checkpoint.json"
OUTPUT_CSV_PATH = "data/processed/nagoya_final_data_2026.csv"
DOTENV_PATH = ".env"

# API仕様定義
DETAILS_URL_TEMPLATE = "https://places.googleapis.com/v1/places/{place_id}"

# 予算リミッター
MAX_REQ_KEY1 = 4800 # キー1の上限
MAX_REQ_KEY2 = 7300 # キー2の上限

# 優先飲食店カテゴリ（typesから主要ジャンルを決定するための優先順位リスト）
PRIORITY_CATEGORIES = [
    "ramen_restaurant", "sushi_restaurant", "izakaya_restaurant", "pizza_restaurant",
    "steak_house", "hamburger_restaurant", "italian_restaurant", "chinese_restaurant",
    "japanese_restaurant", "indian_restaurant", "korean_restaurant", "seafood_restaurant",
    "fast_food", "sandwich_shop", "bakery", "cafe", "coffee_shop", "bar", "pub",
    "restaurant", "meal_takeaway", "meal_delivery", "food"
]

def load_api_keys():
    """.env ファイルから API キーを取得する"""
    keys = {"key1": None, "key2": None}
    if not os.path.exists(DOTENV_PATH):
        print(f"Warning: {DOTENV_PATH} not found.")
        return keys
        
    with open(DOTENV_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
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
                    
    if os.environ.get("GOOGLE_PLACES_API_KEY_1"):
        keys["key1"] = os.environ.get("GOOGLE_PLACES_API_KEY_1")
    if os.environ.get("GOOGLE_PLACES_API_KEY_2"):
        keys["key2"] = os.environ.get("GOOGLE_PLACES_API_KEY_2")
        
    return keys

def extract_floor(address):
    """住所テキストから所在階数を抽出するロジック (番地形式なら1階、ビル名等で階数なしならother)"""
    if not address:
        return "other"
        
    # 前後の空白を除去
    addr_clean = address.strip()
    
    # 半角全角の統一処理 (数字、F, 階)
    addr_trans = addr_clean.translate(str.maketrans('０１２３４５６７８９Ｆ階', '0123456789F階'))
    
    # 1. 地下階 (B1, B2F, 地下1階など)
    b_match = re.search(r'(?:地下|B)\s*([0-9]+)\s*(?:階|F)?', addr_trans, re.IGNORECASE)
    if b_match:
        try:
            return -int(b_match.group(1))
        except ValueError:
            return -1
            
    # 2. 地上階 (2F, 2階, 2階B室など)
    f_match = re.search(r'([0-9]+)\s*(?:階|F)', addr_trans, re.IGNORECASE)
    if f_match:
        try:
            return int(f_match.group(1))
        except ValueError:
            return 1
            
    # 3. 階数表現がない場合の処理：
    # 住所の末尾が番地形式（数字、号、地、番、番地、丁目、など）で終わっているか確認
    # 全角ハイフンなどを半角に統一
    addr_norm = addr_trans.translate(str.maketrans('－—ー', '---'))
    addr_norm = addr_norm.strip()
    
    # 末尾が数字、または番地に関わる漢字の場合、1階とみなす
    if re.search(r'(?:[0-9号地番丁]|番地)$', addr_norm):
        return 1
    else:
        # ビル名や建物名で終わっているのに階数表現がない場合は "other" とする
        return "other"

def select_primary_category(types_list):
    """types配列から最も具体的な主要カテゴリを選択する"""
    if not types_list:
        return "other"
        
    # 優先度リストに従って、最初に見つかったカテゴリを主カテゴリとする
    for cat in PRIORITY_CATEGORIES:
        if cat in types_list:
            return cat
            
    # 優先リストになければ、一般的なカテゴリ（point_of_interestなど）を除く最初の要素を返す
    ignore_types = ["point_of_interest", "establishment", "food"]
    for t in types_list:
        if t not in ignore_types:
            return t
            
    return types_list[0]

def fetch_place_details(place_id, api_key):
    """Place Details (New) APIを実行し、住所とカテゴリーを取得する"""
    url = DETAILS_URL_TEMPLATE.format(place_id=place_id)
    headers = {
        "X-Goog-Api-Key": api_key,
        # formattedAddress と types のみを要求 (Place Details Pro ティアで課金される)
        "X-Goog-FieldMask": "formattedAddress,types",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=15)
        
        if response.status_code == 403:
            print("API Key Error (403 Forbidden). Quota limit reached or invalid key.")
            return None, "KEY_EXHAUSTED"
        elif response.status_code == 404:
            print(f"Place ID {place_id} not found (404).")
            return None, "NOT_FOUND"
        elif response.status_code != 200:
            print(f"API Error {response.status_code}: {response.text}")
            return None, "ERROR"
            
        return response.json(), "SUCCESS"
        
    except Exception as e:
        print(f"Request Exception: {e}")
        return None, "EXCEPTION"

def save_checkpoint(completed_ids, enriched_data, req_key1, req_key2, active_key_idx):
    """チェックポイントを進捗保存する"""
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    data = {
        "completed_ids": list(completed_ids),
        "enriched_data": enriched_data,
        "req_key1": req_key1,
        "req_key2": req_key2,
        "active_key_index": active_key_idx
    }
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_checkpoint():
    """チェックポイントをロードする"""
    if os.path.exists(CHECKPOINT_PATH):
        try:
            with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                return (
                    set(data.get("completed_ids", [])),
                    data.get("enriched_data", {}),
                    data.get("req_key1", 0),
                    data.get("req_key2", 0),
                    data.get("active_key_index", 1)
                )
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Starting fresh.")
    return set(), {}, 0, 0, 1

def merge_and_export_csv(enriched_data):
    """元のベースCSVデータと詳細取得したデータをマージして最終CSVとして出力する"""
    if not os.path.exists(BASE_CSV_PATH):
        print(f"Error: Base CSV {BASE_CSV_PATH} not found. Cannot merge.")
        return
        
    df_base = pd.read_csv(BASE_CSV_PATH)
    
    # マージ用のデータフレームを作成
    enriched_rows = []
    for pid, data in enriched_data.items():
        enriched_rows.append({
            "id": pid,
            "address": data.get("address"),
            "floor": data.get("floor"),
            "primary_type": data.get("primary_type"),
            "types": ",".join(data.get("types", [])) if data.get("types") else ""
        })
        
    df_enriched = pd.DataFrame(enriched_rows)
    
    # idをキーにして外部結合（元のデータを全て残す）
    df_final = pd.merge(df_base, df_enriched, on="id", how="left")
    
    df_final.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")
    print(f"\nMerged final dataset saved to {OUTPUT_CSV_PATH} (Shape: {df_final.shape})")

def main():
    test_mode = "--test" in sys.argv
    fresh_run = "--fresh" in sys.argv
    
    if test_mode:
        print("!!! RUNNING IN TEST MODE (Max 10 requests per key) !!!")
    if fresh_run:
        print("!!! FRESH RUN: Ignoring existing checkpoints !!!")
        
    # 1. APIキーのロード
    keys = load_api_keys()
    key1 = keys["key1"]
    key2 = keys["key2"]
    
    print(f"API Key 1 Available: {key1 is not None} (Length: {len(key1) if key1 else 0})")
    print(f"API Key 2 Available: {key2 is not None} (Length: {len(key2) if key2 else 0})")
    
    if not key1 and not key2:
        print("Error: No API keys found. Please check .env file.")
        return
        
    # 2. ベースデータのロード
    if not os.path.exists(BASE_CSV_PATH):
        print(f"Error: Base CSV {BASE_CSV_PATH} not found.")
        return
    df_base = pd.read_csv(BASE_CSV_PATH)
    place_ids = df_base["id"].dropna().unique().tolist()
    total_ids = len(place_ids)
    print(f"Loaded {total_ids} unique Place IDs from base CSV.")
    
    # 3. 進捗チェックポイントのロード
    if fresh_run:
        completed_ids, enriched_data, req_key1, req_key2, active_key_idx = set(), {}, 0, 0, 1
    else:
        completed_ids, enriched_data, req_key1, req_key2, active_key_idx = load_checkpoint()
        
    max_key1_limit = 10 if test_mode else MAX_REQ_KEY1
    max_key2_limit = 10 if test_mode else MAX_REQ_KEY2
    
    print(f"\n=== Progress Status ===")
    print(f"Completed details lookup: {len(completed_ids)}/{total_ids} places.")
    print(f"Current Request Counts -> Key 1: {req_key1}/{max_key1_limit}, Key 2: {req_key2}/{max_key2_limit}")
    print(f"Current Active Key Index: {active_key_idx}")
    
    current_key = key1 if active_key_idx == 1 else key2
    
    save_counter = 0
    key_exhausted = False
    
    for idx, pid in enumerate(place_ids):
        # 既に処理済みの場合はスキップ
        if pid in completed_ids:
            continue
            
        print(f"[{idx+1}/{total_ids}] Fetching details for Place ID: {pid} ...")
        
        # API上限の事前チェック
        if active_key_idx == 1 and req_key1 >= max_key1_limit:
            print(f"\n>>> API Key 1 Limit ({max_key1_limit}) reached. <<<")
            if key2:
                print("Switching active key to API Key 2...")
                active_key_idx = 2
                current_key = key2
                save_checkpoint(completed_ids, enriched_data, req_key1, req_key2, active_key_idx)
            else:
                print("API Key 2 is not configured. Stopping execution.")
                break
                
        if active_key_idx == 2 and req_key2 >= max_key2_limit:
            print(f"\n>>> API Key 2 Limit ({max_key2_limit}) reached. Stopping safely. <<<")
            break
            
        # APIリクエストの実行
        result, status = fetch_place_details(pid, current_key)
        
        # リクエスト数のカウント
        if active_key_idx == 1:
            req_key1 += 1
        else:
            req_key2 += 1
            
        if status == "KEY_EXHAUSTED":
            print(f"API Key {active_key_idx} quota exhausted.")
            key_exhausted = True
            
        elif status == "SUCCESS" and result:
            address = result.get("formattedAddress")
            types = result.get("types", [])
            
            # 特徴量抽出前処理
            floor = extract_floor(address)
            primary_cat = select_primary_category(types)
            
            enriched_data[pid] = {
                "address": address,
                "floor": floor,
                "primary_type": primary_cat,
                "types": types
            }
            # 完了として登録
            completed_ids.add(pid)
            
        elif status == "NOT_FOUND":
            # 存在しないIDは空データで完了として登録
            enriched_data[pid] = {
                "address": None,
                "floor": 1,
                "primary_type": "other",
                "types": []
            }
            completed_ids.add(pid)
            
        else:
            print(f"Failed to fetch details for Place ID {pid}. Will retry in next run.")
            
        # クォータ枯渇時の自動切り替え
        if key_exhausted:
            if active_key_idx == 1 and key2:
                print("Quota exhausted for Key 1. Swapping to Key 2 immediately...")
                active_key_idx = 2
                current_key = key2
                key_exhausted = False
                # リクエストをキー2でやり直すためインデックスを戻さず、次のIDへ進み、
                # 今回失敗したものは次回（またはチェックポイントから）再開するか、このループ内でリトライ
                save_checkpoint(completed_ids, enriched_data, req_key1, req_key2, active_key_idx)
                continue
            else:
                print("Active Key quota exhausted. No fallback available. Stopping.")
                break
                
        # 50件ごとにチェックポイントを保存
        save_counter += 1
        if save_counter % 50 == 0:
            save_checkpoint(completed_ids, enriched_data, req_key1, req_key2, active_key_idx)
            print(f"Checkpoint saved. ({len(completed_ids)}/{total_ids} processed)")
            # 中間マージCSVも出力
            merge_and_export_csv(enriched_data)
            
        time.sleep(0.5) # レートリミット回避 (Details API用)
        
    # ループ終了後の最終保存とマージ
    save_checkpoint(completed_ids=completed_ids, enriched_data=enriched_data, req_key1=req_key1, req_key2=req_key2, active_key_idx=active_key_idx)
    merge_and_export_csv(enriched_data)
    
    print("\n=== DETAILED LOOKUP RUN COMPLETED ===")
    print(f"Final progress: {len(completed_ids)}/{total_ids} processed.")
    print(f"Final Request Counts -> Key 1: {req_key1}/{max_key1_limit}, Key 2: {req_key2}/{max_key2_limit}")

if __name__ == "__main__":
    # 引数エラー防止のため、ダミー引数を少し修正
    # checkpointの引数名を修正して保存を呼び出す
    def save_checkpoint_shim(completed_grids, enriched_data, req_key1, req_key2, active_key_idx):
        save_checkpoint(completed_grids, enriched_data, req_key1, req_key2, active_key_idx)
    # 本番実行
    main()

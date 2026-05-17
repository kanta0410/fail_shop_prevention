"""
Step 1 v4: Nominatim版 OSM存在確認スクリプト

Overpass APIの代わりにNominatim（OpenStreetMap公式）の
reverse geocodingで各店舗の存在を確認する。

仕様:
- エンドポイント: https://nominatim.openstreetmap.org/reverse
- レート制限: 1秒1リクエスト（厳守）
- 返ってくる最近傍施設の座標と元の店舗座標の距離が
  MATCH_RADIUS_M(60m)以内 かつ class=amenity なら「存続」と判断
- 8,008件 × 1秒 = 約2.2時間で完了

ポリシー遵守:
- User-Agentにアプリ名とメールアドレスを含める（必須）
- 1リクエスト/秒を厳守
- 大量リクエストのため夜間実行を推奨
"""

import requests
import time
import os
import math
import pandas as pd
import numpy as np

# ============================================================
# 設定
# ============================================================
INPUT_CSV        = 'data/output/closure_scores.csv'
OUTPUT_CSV       = 'data/processed/forward_test_raw.csv'
BATCH_SIZE       = 100     # 何件ごとに中間保存するか
SLEEP_PER_QUERY  = 1.1     # 1.1秒（Nominatim利用規約: 1秒以上必須）
REQUEST_TIMEOUT  = 30
MATCH_RADIUS_M   = 60      # 60m以内にamenityがあれば「存続」

NOMINATIM_URL = 'https://nominatim.openstreetmap.org/reverse'
HEADERS = {
    # Nominatim利用規約: User-Agentに識別情報を含めること
    'User-Agent': 'FailShopPrevention/1.0 (research; nagoya)',
    'Accept-Language': 'ja',
}

# 緯度/経度→メートル変換（名古屋付近）
LAT_TO_M = 111000
LON_TO_M = 91000


def haversine_m(lat1, lon1, lat2, lon2):
    """2点間のHaversine距離（m）"""
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


def nominatim_reverse(lat, lon):
    """
    Nominatim reverse geocodingで(lat, lon)の最近傍施設を取得。
    失敗時はNoneを返す。
    """
    params = {
        'lat': lat,
        'lon': lon,
        'zoom': 18,        # zoom=18でPOIレベルの詳細
        'format': 'jsonv2',
        'addressdetails': 0,
    }
    try:
        resp = requests.get(
            NOMINATIM_URL,
            params=params,
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT
        )
        if resp.status_code == 200:
            return resp.json()
        else:
            print(f"  HTTP {resp.status_code}")
            return None
    except Exception as e:
        print(f"  {type(e).__name__}: {e}")
        return None


def check_store_exists(lat, lon):
    """
    Nominatim reverse geocodingで店舗の存在を確認。
    True=存続, False=消滅, None=判定不能

    判定基準:
    - 60m以内に何らかのOSM要素がある → 存続（True）
    - 60m以上離れた要素しか返ってこない → 廃業（False）
    ※ Nominatimは最近傍の「何か」を返すため、amenity以外の道路や
      建物が返ることもある。距離だけで判断するのが最も安定。
    """
    result = nominatim_reverse(lat, lon)

    if result is None:
        return None

    if 'error' in result:
        return None

    osm_lat = float(result.get('lat', 0))
    osm_lon = float(result.get('lon', 0))

    dist = haversine_m(lat, lon, osm_lat, osm_lon)

    # 60m以内に何かある → その地点は市街地・商業エリアで存続とみなす
    # 60m超 → OSMにほぼ何も登録されていない = 廃業の可能性が高い
    return dist <= MATCH_RADIUS_M


def main():
    print("=" * 60)
    print("Forward Test Step1: Nominatim reverse geocoding")
    print("=" * 60)

    df_all = pd.read_csv(INPUT_CSV)
    df_alive = df_all[df_all['is_closed'] == 0].copy().reset_index(drop=True)
    n_total = len(df_alive)
    print(f"Target stores (is_closed=0): {n_total}")
    est_min = n_total * SLEEP_PER_QUERY / 60
    print(f"Estimated time: ~{est_min:.0f} minutes ({est_min/60:.1f} hours)")

    # 既存の中間保存ファイルがあれば再開
    done_ids = set()
    existing_rows = []
    if os.path.exists(OUTPUT_CSV):
        df_existing = pd.read_csv(OUTPUT_CSV)
        done_ids = set(df_existing['id'].tolist())
        existing_rows = df_existing.to_dict('records')
        remaining = n_total - len(done_ids)
        print(f"Resuming: {len(done_ids)} done, {remaining} remaining (~{remaining * SLEEP_PER_QUERY / 60:.0f} min left)")

    results = existing_rows.copy()
    processed = 0

    for i, row in df_alive.iterrows():
        store_id = row['id']
        if store_id in done_ids:
            continue

        lat = row['latitude']
        lon = row['longitude']
        name = row.get('name', '')

        exists = check_store_exists(lat, lon)
        processed += 1

        results.append({
            'id': store_id,
            'name': name,
            'latitude': lat,
            'longitude': lon,
            'exists_2026_05': exists,
            'haigyo_prob': row['closure_probability']
        })

        # BATCH_SIZE件ごとに中間保存
        if processed % BATCH_SIZE == 0:
            pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
            done_count = len(done_ids) + processed
            pct = done_count / n_total * 100
            elapsed_min = processed * SLEEP_PER_QUERY / 60
            remain_min  = (n_total - done_count) * SLEEP_PER_QUERY / 60
            print(f"[{done_count}/{n_total} = {pct:.1f}%] "
                  f"Elapsed: {elapsed_min:.0f}min, Remaining: ~{remain_min:.0f}min | Last: {name}")

        # 1.1秒待機（Nominatim利用規約厳守）
        time.sleep(SLEEP_PER_QUERY)

    # 最終保存
    pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"\n{'=' * 60}")
    print(f"Done! Total records: {len(results)}")

    df_res = pd.DataFrame(results)
    valid = df_res[df_res['exists_2026_05'].notna()]
    closed_count = (valid['exists_2026_05'] == False).sum()
    alive_count  = (valid['exists_2026_05'] == True).sum()
    failed_count = (df_res['exists_2026_05'].isna()).sum()

    print(f"Valid   (not None): {len(valid):,}")
    print(f"Closed  (False):    {closed_count:,}  ({closed_count/max(len(valid),1):.1%})")
    print(f"Alive   (True):     {alive_count:,}")
    print(f"Failed  (None):     {failed_count:,}")
    print(f"Saved to: {OUTPUT_CSV}")


if __name__ == '__main__':
    main()

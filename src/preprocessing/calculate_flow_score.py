import pandas as pd
import numpy as np
import os
from sklearn.neighbors import BallTree
from sklearn.preprocessing import MinMaxScaler

def main():
    # パスの設定
    STORES_PATH = 'data/processed/nagoya_analysis_warehouse_categorized.csv'
    STATIONS_PATH = 'data/raw/nagoya_subway_stations_R6.csv'
    OUTPUT_PATH = 'data/processed/flow_features.csv'

    print("Loading data...")
    stores = pd.read_csv(STORES_PATH)
    stations = pd.read_csv(STATIONS_PATH)

    print("Step 1: Aggregating transfer stations...")
    # 乗り換え駅の合算（同じ駅名・同じ座標で集約）
    stations_agg = (
        stations
        .groupby(['駅名', '緯度', '経度'], as_index=False)
        .agg({'1日平均乗車人員_R6': 'sum'})
    )

    print("Step 2: Building BallTree for distance calculation...")
    # 駅座標をラジアンに変換（BallTree用）
    station_coords_rad = np.radians(stations_agg[['緯度', '経度']].values)

    # 店舗座標をラジアンに変換
    # storesにlat, lonが含まれている想定
    store_coords_rad = np.radians(stores[['latitude', 'longitude']].values)

    # BallTree構築
    EARTH_RADIUS_M = 6371000
    tree = BallTree(station_coords_rad, metric='haversine')

    # 2km以内の駅インデックスと距離を一括取得
    R_MAX = 2000 / EARTH_RADIUS_M  # ラジアン換算
    indices_radius, distances_radius = tree.query_radius(
        store_coords_rad,
        r=R_MAX,
        return_distance=True,
        sort_results=True
    )

    # 全店舗について、最低でも「最寄りの1駅（k=1）」を取得（距離NaNとスコア0を回避するため）
    dist_k1, idx_k1 = tree.query(store_coords_rad, k=1)

    print("Step 3: Calculating flow scores for beta = 1.0, 1.5, 2.0...")
    MIN_DIST_M = 50  # 駅直上の店舗で距離が0になるのを防ぐ下限

    betas = [1.0, 1.5, 2.0]
    score_cols = {beta: [] for beta in betas}

    for i, (idx_list, dist_list) in enumerate(zip(indices_radius, distances_radius)):
        if len(idx_list) > 0:
            # 2km圏内に駅がある場合はすべての該当駅でスコアを加算
            dist_m = dist_list * EARTH_RADIUS_M
            dist_m = np.maximum(dist_m, MIN_DIST_M)
            passengers = stations_agg['1日平均乗車人員_R6'].values[idx_list]
            for beta in betas:
                score = np.sum(passengers / (dist_m ** beta))
                score_cols[beta].append(score)
        else:
            # 2km圏内に駅がない場合は、最も近い1駅のみを使って微小なスコアを付与（完全な0を防ぐ）
            nearest_idx = idx_k1[i][0]
            nearest_dist_m = dist_k1[i][0] * EARTH_RADIUS_M
            nearest_dist_m = np.maximum(nearest_dist_m, MIN_DIST_M)
            passengers = stations_agg['1日平均乗車人員_R6'].values[nearest_idx]
            for beta in betas:
                score = passengers / (nearest_dist_m ** beta)
                score_cols[beta].append(score)

    stores['flow_score_raw_b1.0'] = score_cols[1.0]
    stores['flow_score_raw_b1.5'] = score_cols[1.5]
    stores['flow_score_raw_b2.0'] = score_cols[2.0]

    print("Step 4: Generating additional features...")
    # 最寄り駅までの距離（m）- k=1の結果を使うことでNaNを排除
    stores['dist_to_nearest_station'] = dist_k1.flatten() * EARTH_RADIUS_M

    # 最寄り駅名
    stores['nearest_station_name'] = stations_agg['駅名'].values[idx_k1.flatten()]

    # 最寄り駅の乗車人員
    stores['nearest_station_passengers'] = stations_agg['1日平均乗車人員_R6'].values[idx_k1.flatten()]

    # 乗り換え駅フラグ
    transfer_stations = [
        '名古屋','栄','金山','伏見','丸の内','久屋大通',
        '上前津','本山','八事','今池','御器所','新瑞橋','平安通'
    ]
    stores['nearest_is_transfer'] = stores['nearest_station_name'].isin(
        transfer_stations
    ).astype(int)

    # 2km以内の駅数
    stores['station_count_2km'] = [len(idx_list) for idx_list in indices_radius]

    print("Step 5: Normalizing scores...")
    # log変換（スケールが大きいため、メインのβ=1.5を採用）
    stores['flow_score_log'] = np.log1p(stores['flow_score_raw_b1.5'])

    print("Step 6: Outputting results...")
    output_cols = [
        'id',
        'flow_score_raw_b1.0',
        'flow_score_raw_b1.5',
        'flow_score_raw_b2.0',
        'flow_score_log',
        'dist_to_nearest_station',
        'nearest_station_name',
        'nearest_station_passengers',
        'nearest_is_transfer',
        'station_count_2km'
    ]

    # ディレクトリ作成
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    stores[output_cols].to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')
    print(f"Successfully generated {len(stores)} records to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()

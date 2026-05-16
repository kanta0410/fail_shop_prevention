"""
グループA・B・E 特徴量エンジニアリング
- グループA: ローデータ由来（rating, review, price）
- グループB: エリア相対値（500m圏内での相対順位・差分）
- グループE: 地理的位置系（主要拠点からの距離・都心度）
"""
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree

# ============================================================
# 定数
# ============================================================
INPUT_PATH = 'data/processed/nagoya_analysis_warehouse_final.csv'
OUTPUT_PATH = 'data/processed/nagoya_features_all.csv'

# 主要拠点座標 (lat, lon)
NAGOYA_STA   = (35.1706, 136.8816)
SAKAE        = (35.1692, 136.9084)
KANAYAMA     = (35.1430, 136.9051)
CITY_CENTER  = ((NAGOYA_STA[0] + SAKAE[0]) / 2, (NAGOYA_STA[1] + SAKAE[1]) / 2)

# 座標→メートル変換（名古屋付近）
LAT_TO_M = 111000
LON_TO_M = 91000


def haversine_m(lat1, lon1, lat2, lon2):
    """2点間のHaversine距離（m）"""
    R = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlam/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))


def add_group_a(df):
    """グループA: ローデータ由来特徴量"""
    print("Adding Group A features...")

    # rating（そのまま）
    # 既存: rating → df['rating']

    # log_review_count
    df['log_review_count'] = np.log1p(df['user_ratings_total'].fillna(0))

    # rating_x_log_review (信頼度スコア)
    df['rating_x_log_review'] = df['rating'].fillna(0) * df['log_review_count']

    # rating_div_log_review (少レビュー割引)
    df['rating_div_log_review'] = df['rating'].fillna(0) / (df['log_review_count'] + np.log(2))

    # high_rating_low_review_flag
    df['high_rating_low_review_flag'] = (
        (df['rating'] >= 4.5) & (df['user_ratings_total'].fillna(0) <= 10)
    ).astype(int)

    # price_level: 価格帯を数値エンコード（不明=0, 普通=1, 高級=2, とても高級=3）
    df['price_level'] = (
        df['price_inexpensive'] * 1 +
        df['price_moderate'] * 2 +
        df['price_expensive'] * 3 +
        df['price_very_expensive'] * 4
    )
    # price_unknownは0のまま

    # price_unknown_flag
    df['price_unknown_flag'] = df['price_unknown'].astype(int)

    return df


def add_group_b(df):
    """グループB: エリア相対値（500m圏内での相対順位・差分）"""
    print("Adding Group B features...")

    # KDTree用の座標（メートル変換）
    coords_m = np.column_stack([
        df['latitude'].values * LAT_TO_M,
        df['longitude'].values * LON_TO_M
    ])
    tree = cKDTree(coords_m)

    rating_vals     = df['rating'].fillna(df['rating'].median()).values
    review_vals     = df['user_ratings_total'].fillna(0).values
    price_vals      = df['price_level'].values

    radius_m = 500
    indices = tree.query_ball_point(coords_m, radius_m)

    rating_diff      = []
    rating_pct       = []
    review_pct       = []
    price_diff       = []

    for i, nbrs in enumerate(indices):
        # 自分自身を除く近傍
        nbrs_excl = [j for j in nbrs if j != i]

        if len(nbrs_excl) == 0:
            rating_diff.append(0.0)
            rating_pct.append(0.5)
            review_pct.append(0.5)
            price_diff.append(0.0)
            continue

        nbr_ratings  = rating_vals[nbrs_excl]
        nbr_reviews  = review_vals[nbrs_excl]
        nbr_prices   = price_vals[nbrs_excl]

        rating_diff.append(rating_vals[i] - np.mean(nbr_ratings))

        # パーセンタイル（何%より高いか）
        r_pct = np.mean(rating_vals[i] > nbr_ratings)
        rv_pct = np.mean(review_vals[i] > nbr_reviews)
        rating_pct.append(r_pct)
        review_pct.append(rv_pct)

        price_diff.append(price_vals[i] - np.mean(nbr_prices))

    df['rating_diff_500m']       = rating_diff
    df['rating_percentile_500m'] = rating_pct
    df['review_percentile_500m'] = review_pct
    df['price_diff_500m']        = price_diff

    return df


def add_group_e(df):
    """グループE: 地理的位置系特徴量"""
    print("Adding Group E features...")

    lats = df['latitude'].values
    lons = df['longitude'].values

    df['dist_to_nagoya_sta'] = haversine_m(lats, lons, *NAGOYA_STA)
    df['dist_to_sakae']      = haversine_m(lats, lons, *SAKAE)
    df['dist_to_kanayama']   = haversine_m(lats, lons, *KANAYAMA)
    df['dist_to_city_center']= haversine_m(lats, lons, *CITY_CENTER)

    # 都心度スコア（3拠点距離の逆数加重平均）
    df['urban_score'] = (
        1 / (df['dist_to_nagoya_sta'] + 1) +
        1 / (df['dist_to_sakae'] + 1) +
        1 / (df['dist_to_kanayama'] + 1)
    )

    return df


def main():
    print(f"Loading data from {INPUT_PATH}...")
    df = pd.read_csv(INPUT_PATH)
    print(f"Shape: {df.shape}")

    df = add_group_a(df)
    df = add_group_b(df)
    df = add_group_e(df)

    print(f"\nSaving to {OUTPUT_PATH}...")
    df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')
    print(f"Done. Final shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")


if __name__ == '__main__':
    main()

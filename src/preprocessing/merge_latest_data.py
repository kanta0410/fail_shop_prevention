import pandas as pd
import numpy as np
import os
import shutil

def main():
    print("============================================================")
    print("Merging 2026-05 Google Data with 2025 Feature Dataset")
    print("============================================================")

    # パス設定
    FEATURES_CSV = "data/processed/nagoya_features_all.csv"
    GOOGLE_2026_CSV = "data/raw/nagoya_google_2026_05.csv"
    BACKUP_CSV = "data/processed/nagoya_features_all_backup_2025.csv"

    if not os.path.exists(FEATURES_CSV):
        print(f"[ERROR] Features file not found: {FEATURES_CSV}")
        return
    if not os.path.exists(GOOGLE_2026_CSV):
        print(f"[ERROR] Google 2026 data file not found: {GOOGLE_2026_CSV}")
        return

    # バックアップの作成
    if not os.path.exists(BACKUP_CSV):
        print(f"Creating backup of original feature dataset to {BACKUP_CSV}...")
        shutil.copyfile(FEATURES_CSV, BACKUP_CSV)

    # データの読み込み
    df_features = pd.read_csv(FEATURES_CSV)
    df_google = pd.read_csv(GOOGLE_2026_CSV)

    print(f"Loaded {len(df_features)} records from 2025 Feature Dataset.")
    print(f"Loaded {len(df_google)} records from 2026-05 Google Dataset.")

    # マージ処理のためにキーを揃える
    # 2026データをキー 'id' でマージできるように準備
    df_google = df_google.rename(columns={
        'rating': 'rating_2026',
        'user_rating_count': 'user_rating_count_2026',
        'business_status': 'business_status_2026',
        'has_opening_hours': 'has_opening_hours_2026'
    })

    # 'is_closed' の定義: CLOSED_PERMANENTLY または NOT_FOUND_CLOSED なら 1、それ以外は 0
    df_google['is_closed_2026'] = np.where(
        df_google['business_status_2026'].isin(['CLOSED_PERMANENTLY', 'NOT_FOUND_CLOSED']), 
        1, 
        0
    )

    # features に 2026年データをマージ
    df_merged = pd.merge(df_features, df_google, on='id', how='left')

    # マージできなかった場合の処理（基本的には全件マッチするはずですが、デバッグ用に）
    missing_count = df_merged['business_status_2026'].isna().sum()
    if missing_count > 0:
        print(f"[WARNING] {missing_count} places could not be matched with 2026 Google data.")
        # マッチしなかったものは安全のため 2025年時点の is_closed を使う、または 0 とする
        df_merged['is_closed_2026'] = df_merged['is_closed_2026'].fillna(df_merged['is_closed'])
    
    # 1. 完璧な廃業フラグ (is_closed) の上書き
    df_merged['is_closed'] = df_merged['is_closed_2026'].astype(int)

    # 2. レビュー数モメンタム特徴量 (review_diff)
    # NaNは 0 として処理
    reviews_2025 = df_merged['user_ratings_total'].fillna(0)
    reviews_2026 = df_merged['user_rating_count_2026'].fillna(0)
    
    # API取得時のブレや減少によるバグを防ぐため、マイナス値は0にクリップして「純増」のみを表現
    df_merged['review_diff'] = np.maximum(0, reviews_2026 - reviews_2025)
    
    # レビュー数増加率（分母に +1 してゼロ除算を防ぐ。同様にマイナスは0にクリップ）
    df_merged['review_growth_rate'] = np.maximum(0.0, (reviews_2026 - reviews_2025) / (reviews_2025 + 1))

    # 3. 評価モメンタム特徴量 (rating_diff)
    # 評価がない（NaN）店舗同士の引き算は NaN になる
    df_merged['rating_diff'] = df_merged['rating_2026'] - df_merged['rating']

    # 4. 営業時間情報のマージ
    df_merged['has_opening_hours'] = df_merged['has_opening_hours_2026'].fillna(0).astype(int)

    # 不要になった一時的なカラムを削除して綺麗にする
    drop_cols = ['is_closed_2026', 'business_status_2026', 'rating_2026', 'user_rating_count_2026', 'has_opening_hours_2026']
    df_merged = df_merged.drop(columns=drop_cols)

    # 保存
    df_merged.to_csv(FEATURES_CSV, index=False)
    print(f"[SUCCESS] Updated feature dataset saved to {FEATURES_CSV}")
    print(f"Total rows: {len(df_merged)}")
    print(f"Number of closed restaurants (is_closed=1): {df_merged['is_closed'].sum()}")
    print(f"Average reviews added in 1.3 years: {df_merged['review_diff'].mean():.2f}")
    print(f"Average rating change: {df_merged['rating_diff'].mean():.4f}")

if __name__ == "__main__":
    main()

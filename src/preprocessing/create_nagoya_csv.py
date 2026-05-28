import os
import pandas as pd
import numpy as np

def main():
    print("Generating simplified 2026 Nagoya dataset (nagoya.csv)...")
    
    # パス定義
    features_all_path = "data/processed/nagoya_features_all.csv"
    details_2026_path = "data/processed/nagoya_final_data_2026.csv"
    output_path = "data/processed/nagoya.csv"

    if not os.path.exists(features_all_path):
        print(f"Error: {features_all_path} not found.")
        return
    if not os.path.exists(details_2026_path):
        print(f"Error: {details_2026_path} not found.")
        return

    df_features = pd.read_csv(features_all_path)
    df_details = pd.read_csv(details_2026_path)

    print(f"Original features shape: {df_features.shape}")

    # 1. OSMのみの追加店舗 (osm_ プレフィックス付きID) を除外（ただし過去に廃業と判定されていたデータは保持する）
    # is_closed_historical というカラムがあるか確認し、無ければ is_closed を使う
    closed_col = 'is_closed_historical' if 'is_closed_historical' in df_features.columns else 'is_closed'
    df_filtered = df_features[(~df_features['id'].str.startswith('osm_')) | (df_features.get(closed_col, 0) == 1)].copy()
    print(f"After filtering out active OSM-only records: {df_filtered.shape}")

    # 2. 2026年詳細データから floor と primary_type をマージ
    # 既存の衝突しそうなカラムがあれば事前に削除
    for col in ['floor', 'primary_type', 'floor_num', 'floor_is_other', 'primary_type_code']:
        if col in df_filtered.columns:
            df_filtered.drop(columns=[col], inplace=True)

    df_details_subset = df_details[['id', 'floor', 'primary_type']].copy()
    df_details_subset = df_details_subset.drop_duplicates(subset=['id'])

    # マージ
    df_merged = pd.merge(df_filtered, df_details_subset, on='id', how='left')
    print(f"After merging details: {df_merged.shape}")

    # 3. 欠損補完
    df_merged['floor'] = df_merged['floor'].fillna("other")
    df_merged['primary_type'] = df_merged['primary_type'].fillna("other")

    # 4. フロア関連の特徴量はテナントマッチング精度とZ軸の問題により全て削除
    pass

    # 5. 特徴量化 (店舗カテゴリ - Label Encoding)
    unique_types = sorted(df_merged['primary_type'].unique().tolist())
    type_to_code = {t: idx for idx, t in enumerate(unique_types)}
    df_merged['primary_type_code'] = df_merged['primary_type'].map(type_to_code)

    df_final = df_merged.copy()

    # 6. カラムのドロップ処理 (ユーザー指示)
    # ドロップ対象リスト
    drop_cols = [
        # モメンタム特徴量 (7番)
        'review_diff', 'review_growth_rate', 'rating_diff',
        # 営業時間 (6番)
        'has_opening_hours',
        # OSMカテゴリ (14番)
        'category', 'category_label', 'cat_1', 'cat_2', 'cat_3', 'cat_4', 'cat_5',
        # 多重共線性回避のための価格ダミー
        'price_unknown',
        # 中間処理用カラム
        'floor', 'primary_type'
    ]

    existing_drop_cols = [c for c in drop_cols if c in df_final.columns]
    df_final.drop(columns=existing_drop_cols, inplace=True)
    print(f"Dropped columns: {existing_drop_cols}")

    # 7. CSVに保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_final.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n[SUCCESS] New simplified 2026 dataset saved to {output_path} (Shape: {df_final.shape})")

    # 統計出力
    closed_counts = df_final['is_closed'].value_counts()
    print("\n=== Statistics for nagoya.csv ===")
    for val, cnt in closed_counts.items():
        state = "Closed (1)" if val == 1 else "Alive (0)"
        print(f"  {state}: {cnt} ({cnt/len(df_final)*100:.2f}%)")
    print(f"  Total records: {len(df_final)}")
    print(f"  Total columns: {len(df_final.columns)}")

if __name__ == "__main__":
    main()

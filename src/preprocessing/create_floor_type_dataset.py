import os
import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor

def main():
    print("Loading datasets...")
    features_2025_path = "data/processed/nagoya_features_all_backup_2025.csv"
    details_2026_path = "data/processed/nagoya_final_data_2026.csv"
    output_path = "data/processed/nagoya_features_with_floor_type_2025.csv"

    if not os.path.exists(features_2025_path):
        print(f"Error: {features_2025_path} not found.")
        return
    if not os.path.exists(details_2026_path):
        print(f"Error: {details_2026_path} not found. Please run the enrichment scan first.")
        return

    df_2025 = pd.read_csv(features_2025_path)
    df_2026 = pd.read_csv(details_2026_path)

    print(f"Original 2025 features dataset: {df_2025.shape}")

    # 1. OSMのみの追加店舗 (osm_ プレフィックス付きID) を除外
    df_2025_filtered = df_2025[~df_2025['id'].str.startswith('osm_')].copy()
    print(f"After filtering out OSM-only records: {df_2025_filtered.shape}")

    # 2. 2026年詳細データから floor と primary_type をマージ
    df_details = df_2026[['id', 'floor', 'primary_type']].copy()
    
    # 重複IDの排除 (念のため)
    df_details = df_details.drop_duplicates(subset=['id'])

    # マージ
    df_merged = pd.merge(df_2025_filtered, df_details, on='id', how='left')
    print(f"After merging details: {df_merged.shape}")

    # 3. 欠損補完
    df_merged['floor'] = df_merged['floor'].fillna("other")
    df_merged['primary_type'] = df_merged['primary_type'].fillna("other")

    # 4. 特徴量化 (フロア)
    # floor_num
    def parse_floor_num(val):
        if pd.isna(val) or val == "other":
            return 1.0
        try:
            return float(val)
        except ValueError:
            return 1.0

    df_merged['floor_num'] = df_merged['floor'].apply(parse_floor_num)
    # floor_is_other
    df_merged['floor_is_other'] = (df_merged['floor'] == "other").astype(int)

    # 5. 特徴量化 (店舗カテゴリ - Label Encoding)
    # LightGBMなどの木モデル向けに、One-Hotではなく単一の整数コードに変換
    unique_types = sorted(df_merged['primary_type'].unique().tolist())
    type_to_code = {t: idx for idx, t in enumerate(unique_types)}
    df_merged['primary_type_code'] = df_merged['primary_type'].map(type_to_code)
    
    # マッピング用辞書をログ出力または検証用に保存できると良いが、ここではそのまま結合
    df_final = df_merged.copy()

    # 多重共線性（価格ダミー）の罠を避けるため、price_unknown をドロップする
    if 'price_unknown' in df_final.columns:
        df_final.drop(columns=['price_unknown'], inplace=True)

    # 元のテキスト/不要カラムのドロップ (マージで使用した中間 floor, primary_type はドロップ、元のcategoryなどは保持)
    df_final.drop(columns=['floor', 'primary_type'], inplace=True)
    
    # CSVに保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_final.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"New dataset saved to {output_path} (Shape: {df_final.shape})")

    # --- 6. 統計分析 ---
    print("\n" + "="*40)
    print("=== STATISTICAL ANALYSIS ===")
    print("="*40)

    # 廃業店舗数の集計
    closed_counts = df_final['is_closed'].value_counts()
    closed_rate = df_final['is_closed'].mean() * 100
    print(f"\n1. Class Distribution (is_closed):")
    for val, cnt in closed_counts.items():
        state = "Closed (1)" if val == 1 else "Alive (0)"
        print(f"  {state}: {cnt} ({cnt/len(df_final)*100:.2f}%)")
    print(f"  Total records: {len(df_final)}")

    # 数値列の抽出
    # IDやカテゴリ名などのオブジェクト列、緯度経度などを除外
    exclude_cols = [
        'id', 'name', 'category', 'nearest_station_name', 'price_level', 'balanced_group'
    ]
    numeric_cols = [c for c in df_final.columns if c not in exclude_cols and not df_final[c].dtype == object]

    # 相関分析
    correlations = df_final[numeric_cols].corr()['is_closed'].sort_values(key=abs, ascending=False)
    print(f"\n2. Top 15 Feature Correlations with 'is_closed':")
    # 自分自身の相関を除外して上位15件
    top_corr = correlations.drop(labels=['is_closed']).head(15)
    for feat, corr_val in top_corr.items():
        print(f"  {feat}: {corr_val:.4f}")

    # 多重共線性 (VIF) の計算
    print(f"\n3. Multicollinearity (VIF) Check:")
    # VIF計算用データフレームの構築
    # target ('is_closed') は除外する。また、欠損値は平均値補完する。
    vif_features = [c for c in numeric_cols if c != 'is_closed']
    df_vif = df_final[vif_features].copy()
    
    # 欠損補完 (VIF計算用)
    for col in df_vif.columns:
        if df_vif[col].isna().any():
            df_vif[col] = df_vif[col].fillna(df_vif[col].mean())

    # 定数項の追加 (VIF計算には通常必要だが、バイナリ列も多いのでそのままでも算出可能。
    # すべて1の列があるとダミー変数のトラップ等で無限大になるため、注意して計算する)
    vif_data = pd.DataFrame()
    vif_data["feature"] = df_vif.columns
    
    # VIFの計算 (大規模データセットのため少し時間がかかる場合がある)
    print("  Calculating VIF values for all numeric features...")
    vif_values = []
    for i in range(df_vif.shape[1]):
        try:
            val = variance_inflation_factor(df_vif.values, i)
            vif_values.append(val)
        except Exception as e:
            vif_values.append(np.nan)
            
    vif_data["VIF"] = vif_values
    
    # VIFが高い順にソート (無限大や10以上)
    vif_sorted = vif_data.sort_values(by="VIF", ascending=False)
    
    print("\n  VIF values > 10.0 (High Multicollinearity):")
    high_vif = vif_sorted[vif_sorted["VIF"] > 10.0]
    if len(high_vif) == 0:
        print("    None")
    else:
        for idx, row in high_vif.head(25).iterrows():
            print(f"    {row['feature']}: {row['VIF']:.2f}")
            
    print("\n  VIF values < 10.0 (Acceptable Multicollinearity):")
    low_vif = vif_sorted[vif_sorted["VIF"] <= 10.0]
    for idx, row in low_vif.head(15).iterrows():
        print(f"    {row['feature']}: {row['VIF']:.2f}")

if __name__ == "__main__":
    main()

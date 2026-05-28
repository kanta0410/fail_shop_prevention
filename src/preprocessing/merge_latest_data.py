import pandas as pd
import numpy as np
import os
import shutil
import difflib

def main():
    print("============================================================")
    print("Merging 2026-05 Google Data with 2025 Feature Dataset")
    print("============================================================")

    # パス設定
    FEATURES_CSV = "data/processed/nagoya_features_all.csv"
    GOOGLE_2026_CSV = "data/raw/nagoya_google_2026_05_week2.csv"
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
    # ただし、APIで見つからなかった(NaN)場合は NaN のままにする
    df_google['is_closed_2026'] = df_google['business_status_2026'].apply(
        lambda x: 1 if x in ['CLOSED_PERMANENTLY', 'NOT_FOUND_CLOSED'] else (0 if pd.notna(x) else np.nan)
    )

    # features に 2026年データをマージ
    df_merged = pd.merge(df_features, df_google, on='id', how='left')

    # マージできなかった場合の処理
    missing_mask = df_merged['business_status_2026'].isna()
    missing_count = missing_mask.sum()
    if missing_count > 0:
        print(f"[WARNING] {missing_count} places could not be matched with 2026 Google data.")
        
        # ユーザーの指摘によるゼロトラスト対応:
        # APIで見つけられなかったデータのうち、「元々OSM等で廃業フラグが立っていた店舗(is_closed=1)」は本当に廃業している可能性が高いので残す。
        # 一方で、「元々営業中(is_closed=0)だったのにAPIで見つからなかった店舗」は、本当に廃業したのか座標ズレのノイズなのか分からないため、安全のために除外(drop)する。
        noise_mask = missing_mask & (df_merged['is_closed'] == 0)
        noise_count = noise_mask.sum()
        if noise_count > 0:
            print(f"Dropping {noise_count} noisy records (missing in API but originally marked as alive).")
            df_merged = df_merged[~noise_mask].copy()
            
        # 残ったAPI未ヒットデータ（＝元々廃業フラグが立っていたもの）は 1 にする
        df_merged['is_closed_2026'] = df_merged['is_closed_2026'].fillna(1)

    
    # 1. 完璧な廃業フラグ (is_closed) の上書きとテナント入れ替わり判定
    # まず、Google API側での廃業判定をセット
    df_merged['is_closed'] = df_merged['is_closed_2026'].astype(int)
    
    # テナント入れ替わり問題の対策:
    # 過去データ(name)と最新データ(name_2026 がもしあれば)の名前を比較し、
    # 類似度が著しく低い場合は「テナントが入れ替わった＝前の店は廃業した」と判定して1を立てる。
    if 'name_y' in df_merged.columns and 'name_x' in df_merged.columns:
        def get_similarity(row):
            if pd.isna(row['name_x']) or pd.isna(row['name_y']):
                return 1.0 # 比較できない場合はセーフとする
            return difflib.SequenceMatcher(None, str(row['name_x']), str(row['name_y'])).ratio()
            
        df_merged['name_similarity'] = df_merged.apply(get_similarity, axis=1)
        # 類似度が 0.4 未満なら別店舗（テナント入れ替わり）とみなして廃業(1)とする
        df_merged.loc[df_merged['name_similarity'] < 0.4, 'is_closed'] = 1
        
    # さらに、過去の時点ですでに廃業とされていた店舗が、誤って「営業中」に上書きされるのを防ぐ
    # (Google API側で名前が変わっていても座標マッチングで OPERATIONAL と判定されている可能性があるため)
    if 'is_closed_historical' in df_features.columns:
        df_merged.loc[df_features['is_closed_historical'] == 1, 'is_closed'] = 1
    elif 'is_closed_x' in df_merged.columns: # mergeにより接尾辞がついた場合
        df_merged.loc[df_merged['is_closed_x'] == 1, 'is_closed'] = 1
    else:
        # 古い is_closed の値を優先する (is_closed_2026で0に上書きしない)
        # df_featuresとdf_googleのマージでis_closedが重複していない場合は、結合前の値を使う
        original_closed_indices = df_features[df_features['is_closed'] == 1].index
        # on='id' のマージなので、index はおおむね保持されるか、id でマッチングさせるのが確実
        closed_ids = df_features[df_features['is_closed'] == 1]['id']
        df_merged.loc[df_merged['id'].isin(closed_ids), 'is_closed'] = 1

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

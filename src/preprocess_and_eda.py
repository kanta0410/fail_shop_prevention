import pandas as pd
import numpy as np

def load_and_preprocess(input_file, output_file):
    # データの読み込み
    try:
        df = pd.read_csv(input_file)
        print(f"Loaded {len(df)} records from {input_file}")
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        return None

    # --- 1. 欠損値の処理 ---
    
    # price_levelの欠損値は未知カテゴリとして扱うか、最頻値/中央値で埋める
    # 今回は未設定(None/NaN)が多いと考えられるため、文字列の'PRICE_LEVEL_UNKNOWN'などで一旦埋め、その後エンコーディングする
    df['price_level'] = df['price_level'].fillna('PRICE_LEVEL_UNKNOWN')
    
    # rating (星評価) が欠損している場合、レビューが0件であることが多い。
    # レビュー0件の場合は0で埋めるか、全体の中央値で埋める。今回は分析の安全のため中央値で埋める。
    rating_median = df['rating'].median()
    if pd.isna(rating_median):
        rating_median = 3.0 # データが全くない場合のフォールバック
    df['rating'] = df['rating'].fillna(rating_median)
    
    # user_rating_count (レビュー件数) が欠損している場合は0とする
    df['user_rating_count'] = df['user_rating_count'].fillna(0)

    # --- 2. 特徴量エンジニアリング（予測しやすい形へ変換） ---
    
    # ターゲット変数（正解ラベル）の作成
    # OPERATIONAL = 0 (存続), CLOSED_PERMANENTLY / CLOSED_TEMPORARILY = 1 (廃業)
    # ※今回の初期データではOPERATIONALしかない可能性があります
    def map_status(status):
        if pd.isna(status):
            return np.nan
        elif status == 'OPERATIONAL':
            return 0
        else:
            return 1
            
    df['target_closed'] = df['business_status'].apply(map_status)

    # カテゴリ変数のダミー化 (One-Hot Encoding)
    # price_level は順序尺度（安い～高い）なので、Ordinal Encodingの方が良い場合もあるが
    # LightGBMなどではそのままカテゴリとして扱うか、数値マッピングする
    price_mapping = {
        'PRICE_LEVEL_UNKNOWN': 0,
        'PRICE_LEVEL_INEXPENSIVE': 1,
        'PRICE_LEVEL_MODERATE': 2,
        'PRICE_LEVEL_EXPENSIVE': 3,
        'PRICE_LEVEL_VERY_EXPENSIVE': 4
    }
    df['price_level_num'] = df['price_level'].map(price_mapping)
    
    # --- 3. 分析用データセットの整理 ---
    
    # モデリングに使用する特徴量を整理
    features_df = df[[
        'id', 
        'name',
        'lat',
        'lng',
        'rating',
        'user_rating_count',
        'price_level',
        'price_level_num',
        'business_status',
        'target_closed'
    ]]
    
    # 保存
    features_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"Preprocessed data saved to {output_file}")
    
    return features_df

def run_eda(df):
    if df is None:
        return
        
    print("\n--- EDA: 基本統計量 ---")
    print(df.describe())
    
    print("\n--- EDA: 廃業(1) vs 存続(0) の割合 ---")
    print(df['target_closed'].value_counts(dropna=False))
    
    print("\n--- EDA: 価格帯ごとの店舗数 ---")
    print(df['price_level'].value_counts())
    
    print("\n--- EDA: 欠損値の確認 ---")
    print(df.isnull().sum())

if __name__ == "__main__":
    raw_file = "chikusa_restaurants_grid_raw.csv"
    clean_file = "chikusa_restaurants_grid_clean.csv"
    
    processed_df = load_and_preprocess(raw_file, clean_file)
    run_eda(processed_df)

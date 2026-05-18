import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

def task1_add_nearest_store_features():
    print("--- Task 1: 最も近い店舗のレビュー・繁盛具合の追加 ---")
    df = pd.read_csv('data/processed/nagoya_features_all.csv')
    
    # 緯度経度をラジアンに変換してBallTreeを作成 (Haversine距離用)
    coords = np.radians(df[['latitude', 'longitude']].values)
    tree = BallTree(coords, metric='haversine')
    
    # k=2 で検索 (1つ目は自分自身なので2つ目を取得)
    distances, indices = tree.query(coords, k=2)
    
    # 一番近い別店舗のインデックス
    nearest_idx = indices[:, 1]
    
    # 一番近い店舗の特徴量を取得して追加
    # rating_diff_500m などはあるが、一番近い店舗ズバリの値を抽出
    df['nearest_store_rating'] = df.iloc[nearest_idx]['rating'].values
    df['nearest_store_log_review_count'] = df.iloc[nearest_idx]['log_review_count'].values
    
    # 出力
    out_path = 'data/processed/nagoya_analysis_warehouse_with_nearest.csv'
    df.to_csv(out_path, index=False)
    print(f"追加完了: {out_path} に保存しました。")
    print(df[['name', 'rating', 'log_review_count', 'nearest_store_rating', 'nearest_store_log_review_count']].head())

def task2_confusion_matrix_analysis():
    print("\n--- Task 2: 混同行列の確率（False Positive等の割合）分析 ---")
    # closure_scores.csv には予測スコアと正解ラベルが入っているか確認
    try:
        scores_df = pd.read_csv('data/output/closure_scores.csv')
        # pred_prob_xgb などのカラムがある想定
        pred_col = 'pred_prob_xgb' if 'pred_prob_xgb' in scores_df.columns else 'pred_prob_lgb'
        if pred_col not in scores_df.columns:
            # カラム名を探す
            pred_cols = [c for c in scores_df.columns if 'prob' in c.lower() or 'score' in c.lower()]
            if pred_cols:
                pred_col = pred_cols[0]
            else:
                pred_col = 'closure_score'
        
        y_true = scores_df['is_closed']
        y_prob = scores_df[pred_col]
        
        # 閾値をいくつか試して混同行列を出す
        thresholds = [0.5, 0.7, 0.8, 0.9]
        for th in thresholds:
            y_pred = (y_prob >= th).astype(int)
            cm = confusion_matrix(y_true, y_pred)
            total = np.sum(cm)
            cm_prob = cm / total
            print(f"\n[閾値: {th}]")
            print(f"混同行列 (件数):\n{cm}")
            print(f"混同行列 (確率):\n{cm_prob.round(4)}")
            print(f"False Positive (廃業予測だが存続) の確率: {cm_prob[0, 1]:.4%} (件数: {cm[0, 1]})")
            print(f"True Positive (廃業予測で廃業) の確率: {cm_prob[1, 1]:.4%} (件数: {cm[1, 1]})")
    except Exception as e:
        print(f"スコアファイルの読み込みエラー: {e}")

def task3_station_analysis():
    print("\n--- Task 3: 駅の乗降客数と乗り換えフラグの分析 ---")
    df = pd.read_csv('data/processed/nagoya_features_all.csv')
    
    print("乗り換え駅かどうかの内訳:")
    print(df['nearest_is_transfer'].value_counts())
    
    print("\n乗り換え駅フラグ別の廃業率:")
    print(df.groupby('nearest_is_transfer')['is_closed'].mean())
    
    print("\n乗降客数の要約統計量 (廃業フラグ別):")
    print(df.groupby('is_closed')['nearest_station_passengers'].describe())
    
if __name__ == '__main__':
    task1_add_nearest_store_features()
    task2_confusion_matrix_analysis()
    task3_station_analysis()

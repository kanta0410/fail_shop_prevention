"""
Nagoya Restaurant Closure Prediction - EDA3 Training Pipeline
(自身のレビュー・レーティングを除外し、最も近い店舗のレビュー・レーティングのみを追加)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, log_loss
from sklearn.metrics import precision_recall_curve, roc_curve

import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTE
import shap
import joblib
import warnings
warnings.filterwarnings('ignore')

# 日本語フォント設定
try:
    plt.rcParams['font.family'] = 'MS Gothic'
except:
    pass
plt.rcParams['axes.unicode_minus'] = False

# 設定
INPUT_CSV = 'data/processed/nagoya_analysis_warehouse_with_nearest.csv'
OUTPUT_DIR = 'data/output/eda3'
os.makedirs(OUTPUT_DIR, exist_ok=True)
RANDOM_STATE = 42

# ---------------------------------------------------------
# 【特徴量リスト】
# ---------------------------------------------------------

# 連続値系（標準化対象）
CONTINUOUS_COLS = [
    # ---- 自身のレビュー関連は除外 ----
    # ---- EDA3追加: 近隣店舗のレビュー・繁盛具合 ----
    'nearest_store_rating',
    'nearest_store_log_review_count',
    
    'price_diff_500m',
    'dist_to_nearest_restaurant', 'dist_to_nearest_road', 'dist_to_nearest_convenience',
    'dist_to_nearest_signal', 'dist_to_nearest_office', 'dist_to_nearest_parking',
    'count_restaurants_300m', 'count_restaurants_500m', 'count_restaurants_1000m',
    'count_convenience_300m', 'count_convenience_500m', 'count_convenience_1000m',
    'count_office_300m', 'count_office_500m', 'count_office_1000m',
    'count_parking_300m', 'count_parking_500m', 'count_parking_1000m',
    'count_signal_300m', 'count_signal_500m', 'count_signal_1000m',
    'flow_score_raw_b1.0', 'flow_score_raw_b1.5', 'flow_score_raw_b2.0', 'flow_score_log',
    'dist_to_nearest_station', 'nearest_station_passengers', 'station_count_2km',
    'dist_to_nagoya_sta', 'dist_to_sakae', 'dist_to_kanayama', 'dist_to_city_center', 'urban_score',
    'kde_density_score', 'price_level',
]

# カテゴリ・フラグ系（標準化しない）
FLAG_COLS = [
    'price_inexpensive', 'price_moderate', 'price_expensive', 'price_very_expensive',
    'price_unknown_flag', 
    'category_label', 'cat_1', 'cat_2', 'cat_3', 'cat_4', 'cat_5',
    'is_along_highway', 'is_near_mall',
    'nearest_is_transfer',
    'kmeans_cluster_id', 'dbscan_cluster_id', 'is_dense_area',
]

ALL_FEATURES = CONTINUOUS_COLS + FLAG_COLS
TARGET_COL = 'is_closed'

def load_and_split_data():
    print(f"Loading dataset from {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    
    # クラスタリング用の特徴量は後で埋めるのでここでは仮に0を入れる
    for col in ['kmeans_cluster_id', 'dbscan_cluster_id', 'is_dense_area']:
        if col not in df.columns:
            df[col] = 0
            
    # 特徴量とターゲットの抽出
    available_cols = [c for c in ALL_FEATURES if c in df.columns]
    X = df[available_cols].fillna(0)
    y = df[TARGET_COL]
    
    # Train: 64%, Val: 16%, Test: 20%
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.20, random_state=RANDOM_STATE, stratify=y_temp)
    
    print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test, df

def apply_clustering(X_train, X_val, X_test, df):
    print("Applying clustering (Group F)...")
    
    # 1. K-Means
    cluster_features = ['flow_score_raw_b1.5', 'count_restaurants_500m']
    X_train_cluster = X_train[cluster_features].fillna(0)
    scaler_kmeans = StandardScaler()
    X_train_scaled = scaler_kmeans.fit_transform(X_train_cluster)
    
    kmeans = KMeans(n_clusters=5, random_state=RANDOM_STATE, n_init=10)
    X_train['kmeans_cluster_id'] = kmeans.fit_predict(X_train_scaled)
    
    X_val['kmeans_cluster_id'] = kmeans.predict(scaler_kmeans.transform(X_val[cluster_features].fillna(0)))
    X_test['kmeans_cluster_id'] = kmeans.predict(scaler_kmeans.transform(X_test[cluster_features].fillna(0)))
    df['kmeans_cluster_id'] = kmeans.predict(scaler_kmeans.transform(df[cluster_features].fillna(0)))
    
    # 2. DBSCAN (経度緯度のみ)
    latlon = df.loc[X_train.index, ['latitude', 'longitude']].fillna(0)
    dbscan = DBSCAN(eps=0.005, min_samples=5) 
    X_train['dbscan_cluster_id'] = dbscan.fit_predict(latlon)
    X_train['is_dense_area'] = (X_train['dbscan_cluster_id'] != -1).astype(int)
    
    X_val['dbscan_cluster_id'] = -1
    X_val['is_dense_area'] = 0
    X_test['dbscan_cluster_id'] = -1
    X_test['is_dense_area'] = 0
    
    return X_train, X_val, X_test

def scale_and_smote(X_train, X_val, X_test, y_train):
    print("Scaling and applying SMOTE...")
    scaler = StandardScaler()
    
    # Scale only continuous columns
    cont_cols = [c for c in CONTINUOUS_COLS if c in X_train.columns]
    
    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[cont_cols] = scaler.fit_transform(X_train[cont_cols])
    X_val_scaled[cont_cols] = scaler.transform(X_val[cont_cols])
    X_test_scaled[cont_cols] = scaler.transform(X_test[cont_cols])
    
    # SMOTE (訓練データのみ)
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
    
    print(f"  After SMOTE - X_train: {X_train_res.shape}")
    return X_train_res, X_val_scaled, X_test_scaled, y_train_res, scaler

def train_and_evaluate(models, X_train, y_train, X_val, y_val):
    print("Training models...")
    results = {}
    
    for name, model in models.items():
        print(f"\n--- {name} ---")
        model.fit(X_train, y_train)
        
        # 予測確率
        y_train_pred = model.predict_proba(X_train)[:, 1]
        y_val_pred = model.predict_proba(X_val)[:, 1]
        
        # 指標計算
        train_auc = roc_auc_score(y_train, y_train_pred)
        train_pr = average_precision_score(y_train, y_train_pred)
        
        val_auc = roc_auc_score(y_val, y_val_pred)
        val_pr = average_precision_score(y_val, y_val_pred)
        val_f1 = f1_score(y_val, (y_val_pred > 0.5).astype(int))
        val_ll = log_loss(y_val, y_val_pred)
        
        print(f"  [Train] AUC-ROC={train_auc:.4f}, AUC-PR={train_pr:.4f}")
        print(f"  [Val] AUC-ROC={val_auc:.4f}, AUC-PR={val_pr:.4f}, F1={val_f1:.4f}, LogLoss={val_ll:.4f}")
        
        results[name] = {
            'model': model,
            'val_auc': val_auc,
            'val_pr': val_pr,
            'val_pred': y_val_pred
        }
        
    return results

def plot_shap(model, X, model_name):
    print(f"Running SHAP analysis for {model_name}...")
    
    # Treeベースのモデルに限定
    if not isinstance(model, (lgb.LGBMClassifier, xgb.XGBClassifier, RandomForestClassifier)):
        print(f"  Skipping SHAP for {model_name} (Not a Tree model)")
        return
        
    explainer = shap.TreeExplainer(model)
    # 計算量削減のため1000件サンプリング
    X_sample = shap.sample(X, 1000)
    shap_values = explainer.shap_values(X_sample)
    
    # LightGBM/XGBoostのバージョンの違いによる戻り値の処理
    if isinstance(shap_values, list):
        shap_values = shap_values[1] # クラス1（廃業）に関するSHAP値
        
    # Bar plot (重要度ランキング)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    plt.title(f'SHAP Feature Importance ({model_name} - EDA3)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'shap_bar_{model_name.lower()}.png'), dpi=150)
    plt.close()
    
    # Beeswarm plot (分布と影響)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.title(f'SHAP Beeswarm Plot ({model_name} - EDA3)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'shap_beeswarm_{model_name.lower()}.png'), dpi=150)
    plt.close()
    
def main():
    print("=" * 60)
    print("Starting EDA3 Pipeline (With Nearest Store Features)")
    print("=" * 60)
    
    X_train, X_val, X_test, y_train, y_val, y_test, df_full = load_and_split_data()
    X_train, X_val, X_test = apply_clustering(X_train, X_val, X_test, df_full)
    
    X_train_res, X_val_scaled, X_test_scaled, y_train_res, scaler = scale_and_smote(X_train, X_val, X_test, y_train)
    
    # モデル定義
    models = {
        'LightGBM': lgb.LGBMClassifier(random_state=RANDOM_STATE, class_weight='balanced'),
        'XGBoost': xgb.XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss'),
    }
    
    results = train_and_evaluate(models, X_train_res, y_train_res, X_val_scaled, y_val)
    
    # Testセットでの最終評価とSHAP出力
    print("\n============================================================")
    print("Final evaluation on TEST set (EDA3)...")
    test_results = []
    
    for name, res in results.items():
        model = res['model']
        y_test_pred = model.predict_proba(X_test_scaled)[:, 1]
        
        auc = roc_auc_score(y_test, y_test_pred)
        pr = average_precision_score(y_test, y_test_pred)
        f1 = f1_score(y_test, (y_test_pred > 0.5).astype(int))
        ll = log_loss(y_test, y_test_pred)
        
        print(f"  [Test] {name} | AUC-ROC={auc:.4f}, AUC-PR={pr:.4f}, F1={f1:.4f}, LogLoss={ll:.4f}")
        test_results.append({'model': name, 'AUC-ROC': auc, 'AUC-PR': pr, 'F1': f1, 'LogLoss': ll})
        
        if name in ['LightGBM', 'XGBoost']:
            plot_shap(model, X_test_scaled, name)
            
    pd.DataFrame(test_results).to_csv(os.path.join(OUTPUT_DIR, 'test_metrics_eda3.csv'), index=False)
    
    print("\n============================================================")
    print("All steps completed successfully in EDA3 module!")

if __name__ == '__main__':
    main()

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve
import lightgbm as lgb
import xgboost as xgb
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

INPUT_CSV = 'data/processed/nagoya_analysis_warehouse_with_nearest.csv'
OUTPUT_DIR = 'data/output/eda3'
os.makedirs(OUTPUT_DIR, exist_ok=True)
RANDOM_STATE = 42

CONTINUOUS_COLS = [
    'nearest_store_rating', 'nearest_store_log_review_count',
    'price_diff_500m', 'dist_to_nearest_restaurant', 'dist_to_nearest_road', 'dist_to_nearest_convenience',
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

FLAG_COLS = [
    'price_inexpensive', 'price_moderate', 'price_expensive', 'price_very_expensive',
    'price_unknown_flag', 
    'category_label', 'cat_1', 'cat_2', 'cat_3', 'cat_4', 'cat_5',
    'is_along_highway', 'is_near_mall', 'nearest_is_transfer',
    'kmeans_cluster_id', 'dbscan_cluster_id', 'is_dense_area',
]
ALL_FEATURES = CONTINUOUS_COLS + FLAG_COLS
TARGET_COL = 'is_closed'

def load_and_split_data():
    df = pd.read_csv(INPUT_CSV)
    for col in ['kmeans_cluster_id', 'dbscan_cluster_id', 'is_dense_area']:
        if col not in df.columns:
            df[col] = 0
    available_cols = [c for c in ALL_FEATURES if c in df.columns]
    X = df[available_cols].fillna(0)
    y = df[TARGET_COL]
    
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.20, random_state=RANDOM_STATE, stratify=y_temp)
    return X_train, X_val, X_test, y_train, y_val, y_test, df

def apply_clustering(X_train, X_val, X_test, df):
    cluster_features = ['flow_score_raw_b1.5', 'count_restaurants_500m']
    X_train_cluster = X_train[cluster_features].fillna(0)
    scaler_kmeans = StandardScaler()
    X_train_scaled = scaler_kmeans.fit_transform(X_train_cluster)
    
    kmeans = KMeans(n_clusters=5, random_state=RANDOM_STATE, n_init=10)
    X_train['kmeans_cluster_id'] = kmeans.fit_predict(X_train_scaled)
    X_val['kmeans_cluster_id'] = kmeans.predict(scaler_kmeans.transform(X_val[cluster_features].fillna(0)))
    X_test['kmeans_cluster_id'] = kmeans.predict(scaler_kmeans.transform(X_test[cluster_features].fillna(0)))
    df['kmeans_cluster_id'] = kmeans.predict(scaler_kmeans.transform(df[cluster_features].fillna(0)))
    
    latlon = df.loc[X_train.index, ['latitude', 'longitude']].fillna(0)
    dbscan = DBSCAN(eps=0.005, min_samples=5) 
    X_train['dbscan_cluster_id'] = dbscan.fit_predict(latlon)
    X_train['is_dense_area'] = (X_train['dbscan_cluster_id'] != -1).astype(int)
    
    X_val['dbscan_cluster_id'] = -1
    X_val['is_dense_area'] = 0
    X_test['dbscan_cluster_id'] = -1
    X_test['is_dense_area'] = 0
    return X_train, X_val, X_test

def run_experiment(exp_name, use_smote, apply_class_weight):
    X_train, X_val, X_test, y_train, y_val, y_test, df_full = load_and_split_data()
    X_train, X_val, X_test = apply_clustering(X_train, X_val, X_test, df_full)
    
    scaler = StandardScaler()
    cont_cols = [c for c in CONTINUOUS_COLS if c in X_train.columns]
    
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[cont_cols] = scaler.fit_transform(X_train[cont_cols])
    X_test_scaled[cont_cols] = scaler.transform(X_test[cont_cols])
    
    if use_smote:
        smote = SMOTE(random_state=RANDOM_STATE)
        X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
    else:
        X_train_res, y_train_res = X_train_scaled, y_train

    if apply_class_weight:
        lgb_model = lgb.LGBMClassifier(random_state=RANDOM_STATE, class_weight='balanced')
        pos_weight = (len(y_train_res) - sum(y_train_res)) / sum(y_train_res)
        xgb_model = xgb.XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss', scale_pos_weight=pos_weight)
    else:
        lgb_model = lgb.LGBMClassifier(random_state=RANDOM_STATE)
        xgb_model = xgb.XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss')

    models = {'LightGBM': lgb_model, 'XGBoost': xgb_model}
    y_test_preds = {}
    
    for name, model in models.items():
        model.fit(X_train_res, y_train_res)
        y_test_preds[name] = model.predict_proba(X_test_scaled)[:, 1]
    
    plt.figure(figsize=(8, 6))
    for name, y_pred in y_test_preds.items():
        precision, recall, _ = precision_recall_curve(y_test, y_pred)
        ap = average_precision_score(y_test, y_pred)
        plt.plot(recall, precision, label=f'{name} (AP = {ap:.4f})')
    baseline = y_test.sum() / len(y_test)
    plt.axhline(y=baseline, color='gray', linestyle='--', label=f'Baseline (ratio = {baseline:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve ({exp_name})')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, f'pr_curve_{exp_name}.png'), dpi=150)
    plt.close()

    plt.figure(figsize=(8, 6))
    for name, y_pred in y_test_preds.items():
        fpr, tpr, _ = roc_curve(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_pred)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random (AUC = 0.5000)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve ({exp_name})')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, f'roc_curve_{exp_name}.png'), dpi=150)
    plt.close()
    print(f"Finished {exp_name}")

def main():
    print("Experiment 1: SMOTE only (No class_weight)")
    run_experiment("SMOTE_only", use_smote=True, apply_class_weight=False)
    
    print("\nExperiment 2: BalancedWeight only (No SMOTE)")
    run_experiment("BalancedWeight_only", use_smote=False, apply_class_weight=True)

if __name__ == '__main__':
    main()

"""
グループF: クラスタリング特徴量
- K-Means（座標＋特徴量ベース）
- DBSCAN（密度ベース・Haversine）
- KDE密度スコア

注意: Train dataのみでfitし、Val/TestにはTransformのみ適用（リーク防止）
"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KernelDensity
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')


def fit_clustering(X_train_meta, X_all_meta, train_idx, output_col_prefix=''):
    """
    KMeans・DBSCAN・KDEをtrainのみでfitし、全データにtransformを適用する。

    X_train_meta: Trainデータのfeature行列
    X_all_meta: 全データのfeature行列
    train_idx: TrainデータのDataFrame上のインデックス
    """
    results = {}

    # ========================
    # K-Means
    # ========================
    print("  Fitting K-Means...")
    # シルエットスコアで最適クラスタ数を探索（3〜8）
    scaler_km = StandardScaler()
    X_train_scaled = scaler_km.fit_transform(X_train_meta)
    X_all_scaled   = scaler_km.transform(X_all_meta)

    best_score = -1
    best_k = 5
    for k in range(3, 9):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_train_scaled)
        if len(np.unique(labels)) < 2:
            continue
        score = silhouette_score(X_train_scaled, labels, sample_size=2000, random_state=42)
        print(f"    k={k}, silhouette={score:.4f}")
        if score > best_score:
            best_score = score
            best_k = k

    print(f"  Best K={best_k} (silhouette={best_score:.4f})")
    final_km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    final_km.fit(X_train_scaled)
    results['kmeans_cluster_id'] = final_km.predict(X_all_scaled)

    # ========================
    # DBSCAN (地理的座標のみ・Haversine)
    # ========================
    print("  Fitting DBSCAN...")
    # 座標をラジアンに変換
    lat_all = X_all_meta[:, 0]
    lon_all = X_all_meta[:, 1]
    lat_tr  = X_train_meta[:, 0]
    lon_tr  = X_train_meta[:, 1]

    coords_rad_train = np.radians(np.column_stack([lat_tr, lon_tr]))
    coords_rad_all   = np.radians(np.column_stack([lat_all, lon_all]))

    EPS_M = 300  # 300m
    EARTH_R = 6371000
    eps_rad = EPS_M / EARTH_R

    db = DBSCAN(eps=eps_rad, min_samples=5, metric='haversine', algorithm='ball_tree')
    db.fit(coords_rad_train)

    # TransformはDBSCANには直接ないため、最近傍でラベル割り当て
    from sklearn.neighbors import BallTree
    tree_db = BallTree(coords_rad_train, metric='haversine')
    dist_all, nn_all = tree_db.query(coords_rad_all, k=1)

    labels_all = np.array([db.labels_[nn_all[i][0]] for i in range(len(coords_rad_all))])
    results['dbscan_cluster_id'] = labels_all
    results['is_dense_area'] = (labels_all >= 0).astype(int)

    # ========================
    # KDE密度スコア
    # ========================
    print("  Fitting KDE...")
    # 緯度経度をメートル変換（名古屋付近）
    LAT_TO_M = 111000
    LON_TO_M = 91000
    X_km_train = np.column_stack([lat_tr * LAT_TO_M, lon_tr * LON_TO_M])
    X_km_all   = np.column_stack([lat_all * LAT_TO_M, lon_all * LON_TO_M])

    kde = KernelDensity(bandwidth=500, kernel='gaussian')  # 500m帯域幅
    kde.fit(X_km_train)
    log_density = kde.score_samples(X_km_all)
    results['kde_density_score'] = np.exp(log_density)

    return results


def add_group_f(df, train_idx):
    """
    グループF特徴量を追加する。
    train_idx: Trainデータの行インデックスリスト
    """
    print("Adding Group F (Clustering) features...")

    # KMeans入力: 緯度・経度・人流スコア・競合密度
    feature_cols = ['latitude', 'longitude', 'flow_score_raw_b1.5', 'count_restaurants_500m']

    X_all   = df[feature_cols].fillna(0).values
    X_train = X_all[train_idx]

    results = fit_clustering(X_train, X_all, train_idx)

    for col, vals in results.items():
        df[col] = vals

    return df

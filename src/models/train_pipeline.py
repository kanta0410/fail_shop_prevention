"""
廃業予測モデル - メイン学習・評価パイプライン

要件定義書 Step5〜9 を一気通貫で実行:
- Step5: Train/Val/Test分割・標準化・クラスタリング
- Step6: クラスタリング特徴量（グループF）
- Step7: 不均衡対策
- Step8: 交差検証・モデル学習・比較
- Step9: SHAP分析・特徴量重要度可視化
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.build_clustering import add_group_f
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, log_loss,
    precision_recall_curve, roc_curve
)
from imblearn.over_sampling import SMOTE
import joblib

warnings.filterwarnings('ignore')

# ============================================================
# パス設定
# ============================================================
FEATURES_PATH   = 'data/processed/nagoya_features_all.csv'
OUTPUT_DIR      = 'data/output'
MODELS_DIR      = os.path.join(OUTPUT_DIR, 'models')
PLOTS_DIR       = os.path.join(OUTPUT_DIR, 'plots')

for d in [OUTPUT_DIR, MODELS_DIR, PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)

# 日本語フォント設定（matplotlib用）
# WindowsはMSゴシックを使う
try:
    plt.rcParams['font.family'] = 'MS Gothic'
except:
    pass


# ============================================================
# 使用特徴量の定義
# ============================================================
# 目的変数
TARGET = 'is_closed'

# ID・メタ情報（モデルに投入しない）
META_COLS = ['id', 'name', 'latitude', 'longitude', 'category',
             'balanced_group', 'nearest_station_name', 'is_closed']

# 標準化対象の連続値特徴量
CONTINUOUS_COLS = [
    'rating', 'log_review_count', 'rating_x_log_review', 'rating_div_log_review',
    'rating_diff_500m', 'rating_percentile_500m', 'review_percentile_500m', 'price_diff_500m',
    'dist_to_nearest_restaurants', 'dist_to_nearest_convenience', 'dist_to_nearest_offices',
    'dist_to_nearest_signals', 'dist_to_nearest_schools', 'dist_to_nearest_parking',
    'dist_to_nearest_malls', 'dist_to_nearest_road',
    'count_restaurants_300m', 'count_restaurants_500m', 'count_restaurants_1000m',
    'count_convenience_300m', 'count_convenience_500m', 'count_convenience_1000m',
    'count_offices_300m', 'count_offices_500m', 'count_offices_1000m',
    'count_signals_300m', 'count_signals_500m', 'count_signals_1000m',
    'count_schools_300m', 'count_schools_500m', 'count_schools_1000m',
    'count_parking_300m', 'count_parking_500m', 'count_parking_1000m',
    'count_malls_300m', 'count_malls_500m', 'count_malls_1000m',
    'flow_score_raw_b1.0', 'flow_score_raw_b1.5', 'flow_score_raw_b2.0', 'flow_score_log',
    'dist_to_nearest_station', 'nearest_station_passengers', 'station_count_2km',
    'dist_to_nagoya_sta', 'dist_to_sakae', 'dist_to_kanayama', 'dist_to_city_center', 'urban_score',
    'kde_density_score', 'price_level',
]

# カテゴリ・フラグ系（標準化しない）
FLAG_COLS = [
    'price_inexpensive', 'price_moderate', 'price_expensive', 'price_very_expensive',
    'price_unknown_flag', 'high_rating_low_review_flag',
    'category_label', 'cat_1', 'cat_2', 'cat_3', 'cat_4', 'cat_5',
    'is_along_highway', 'is_near_mall',
    'nearest_is_transfer',
    'kmeans_cluster_id', 'dbscan_cluster_id', 'is_dense_area',
]


def get_feature_cols(df):
    """存在するカラムのみを特徴量として返す"""
    candidates = CONTINUOUS_COLS + FLAG_COLS
    return [c for c in candidates if c in df.columns]


def evaluate(model, X, y, model_name, split_name):
    """モデルの評価指標を計算"""
    proba = model.predict_proba(X)[:, 1]
    pred  = (proba >= 0.5).astype(int)

    auc_roc = roc_auc_score(y, proba)
    auc_pr  = average_precision_score(y, proba)
    f1      = f1_score(y, pred, zero_division=0)
    ll      = log_loss(y, proba)

    result = {
        'model': model_name, 'split': split_name,
        'AUC-ROC': auc_roc, 'AUC-PR': auc_pr,
        'F1': f1, 'LogLoss': ll
    }
    print(f"  [{split_name}] AUC-ROC={auc_roc:.4f}, AUC-PR={auc_pr:.4f}, F1={f1:.4f}, LogLoss={ll:.4f}")
    return result


def plot_roc_pr(models_dict, X_val, y_val):
    """ROC曲線とPR曲線を一括プロット"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for name, model in models_dict.items():
        proba = model.predict_proba(X_val)[:, 1]

        fpr, tpr, _ = roc_curve(y_val, proba)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_val, proba):.3f})")

        prec, rec, _ = precision_recall_curve(y_val, proba)
        axes[1].plot(rec, prec, label=f"{name} (AUC-PR={average_precision_score(y_val, proba):.3f})")

    axes[0].plot([0,1],[0,1],'--', color='gray')
    axes[0].set_title('ROC Curve (Val)', fontsize=13)
    axes[0].set_xlabel('FPR'); axes[0].set_ylabel('TPR')
    axes[0].legend()

    axes[1].set_title('Precision-Recall Curve (Val)', fontsize=13)
    axes[1].set_xlabel('Recall'); axes[1].set_ylabel('Precision')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'roc_pr_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: roc_pr_curves.png")


def run_shap_analysis(model, X_val_df, model_name):
    """SHAP分析・可視化"""
    print(f"\nRunning SHAP analysis for {model_name}...")

    if model_name in ['LightGBM', 'XGBoost']:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_val_df)
        # LightGBMは binary classificationで list[pos_class]
        if isinstance(shap_values, list):
            sv = shap_values[1]
        else:
            sv = shap_values
    else:
        explainer = shap.LinearExplainer(model, X_val_df)
        sv = explainer.shap_values(X_val_df)

    # --- Bar plot（特徴量重要度ランキング） ---
    plt.figure(figsize=(10, 8))
    shap.summary_plot(sv, X_val_df, plot_type='bar', show=False, max_display=20)
    plt.title(f'{model_name} - Feature Importance (SHAP Bar)', fontsize=13)
    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, f'shap_bar_{model_name.lower()}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: shap_bar_{model_name.lower()}.png")

    # --- Beeswarm plot ---
    plt.figure(figsize=(10, 8))
    shap.summary_plot(sv, X_val_df, show=False, max_display=20)
    plt.title(f'{model_name} - SHAP Beeswarm', fontsize=13)
    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, f'shap_beeswarm_{model_name.lower()}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: shap_beeswarm_{model_name.lower()}.png")

    # SHAP値をCSVで保存
    shap_df = pd.DataFrame(sv, columns=X_val_df.columns)
    shap_df.to_csv(os.path.join(OUTPUT_DIR, f'shap_values_{model_name.lower()}.csv'), index=False)

    return sv


def main():
    # ============================================================
    # データ読み込み
    # ============================================================
    print("=" * 60)
    print("Loading feature data...")
    df = pd.read_csv(FEATURES_PATH)
    print(f"Shape: {df.shape}")
    print(f"Closed: {df[TARGET].sum()} / Total: {len(df)}")

    feature_cols = get_feature_cols(df)
    print(f"Feature columns ({len(feature_cols)}): {feature_cols}")

    X = df[feature_cols].fillna(0)
    y = df[TARGET].values

    # ============================================================
    # Step5: Train / Val / Test 分割（層化）
    # ============================================================
    print("\n" + "=" * 60)
    print("Step5: Train/Val/Test split (stratified)...")

    # 64 / 16 / 20 % 分割
    idx_all = np.arange(len(df))
    idx_tv, idx_test = train_test_split(
        idx_all, test_size=0.20, random_state=42, stratify=y
    )
    idx_train, idx_val = train_test_split(
        idx_tv, test_size=0.20, random_state=42, stratify=y[idx_tv]
    )

    # ============================================================
    # Step6: グループF クラスタリング特徴量（Train fitのみ）
    # ============================================================
    print("\n" + "=" * 60)
    print("Step6: Adding clustering features (Group F)...")
    df = add_group_f(df, idx_train)

    # クラスタリング特徴量をflag_colsに追加
    for c in ['kmeans_cluster_id', 'dbscan_cluster_id', 'is_dense_area']:
        if c not in FLAG_COLS:
            FLAG_COLS.append(c)
    if 'kde_density_score' not in CONTINUOUS_COLS:
        CONTINUOUS_COLS.append('kde_density_score')

    # 特徴量カラムを再取得
    feature_cols = get_feature_cols(df)
    X = df[feature_cols].fillna(0)

    X_train = X.iloc[idx_train]
    X_val   = X.iloc[idx_val]
    X_test  = X.iloc[idx_test]
    y_train = y[idx_train]
    y_val   = y[idx_val]
    y_test  = y[idx_test]

    print(f"  Train: {len(X_train)} ({y_train.sum()} closed)")
    print(f"  Val:   {len(X_val)}   ({y_val.sum()} closed)")
    print(f"  Test:  {len(X_test)}  ({y_test.sum()} closed)")

    # 標準化（連続値のみ）
    cont_cols = [c for c in CONTINUOUS_COLS if c in feature_cols]
    flag_cols = [c for c in FLAG_COLS       if c in feature_cols]

    scaler = StandardScaler()
    X_train_cont = scaler.fit_transform(X_train[cont_cols])
    X_val_cont   = scaler.transform(X_val[cont_cols])
    X_test_cont  = scaler.transform(X_test[cont_cols])

    X_train_scaled = pd.DataFrame(X_train_cont, columns=cont_cols, index=X_train.index)
    X_val_scaled   = pd.DataFrame(X_val_cont,   columns=cont_cols, index=X_val.index)
    X_test_scaled  = pd.DataFrame(X_test_cont,  columns=cont_cols, index=X_test.index)

    for col in flag_cols:
        X_train_scaled[col] = X_train[col].values
        X_val_scaled[col]   = X_val[col].values
        X_test_scaled[col]  = X_test[col].values

    # スケーラー保存
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.pkl'))

    # ============================================================
    # Step7: SMOTE（Trainのみ）
    # ============================================================
    print("\n" + "=" * 60)
    print("Step7: Applying SMOTE to training data...")

    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_sm, y_train_sm = smote.fit_resample(X_train_scaled, y_train)
    print(f"  After SMOTE: {len(X_train_sm)} ({y_train_sm.sum()} closed)")

    # ============================================================
    # Step8: モデル定義・学習・評価
    # ============================================================
    print("\n" + "=" * 60)
    print("Step8: Training and evaluating models...")

    models = {
        'LightGBM': lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=63,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            class_weight='balanced',
            random_state=42,
            verbose=-1
        ),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
            random_state=42,
            eval_metric='logloss',
            verbosity=0
        ),
        'LogisticRegression': LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=42,
            C=1.0
        ),
        'RandomForest': RandomForestClassifier(
            n_estimators=200,
            class_weight='balanced',
            max_depth=10,
            random_state=42,
            n_jobs=-1
        ),
    }

    all_results = []
    trained_models = {}

    for name, model in models.items():
        print(f"\n--- {name} ---")
        # SMOTEデータで学習
        model.fit(X_train_sm, y_train_sm)

        r_train = evaluate(model, X_train_scaled, y_train, name, 'Train')
        r_val   = evaluate(model, X_val_scaled,   y_val,   name, 'Val')
        all_results.extend([r_train, r_val])
        trained_models[name] = model

        joblib.dump(model, os.path.join(MODELS_DIR, f'model_{name.lower()}.pkl'))
        print(f"  Model saved.")

    # 評価結果を保存
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(OUTPUT_DIR, 'model_comparison.csv'), index=False)
    print(f"\nModel comparison saved.")

    # ROC/PR曲線プロット
    plot_roc_pr(trained_models, X_val_scaled, y_val)

    # ============================================================
    # Step9: SHAP分析（LightGBMとXGBoostのみ）
    # ============================================================
    print("\n" + "=" * 60)
    print("Step9: Running SHAP analysis...")

    for name in ['LightGBM', 'XGBoost']:
        run_shap_analysis(trained_models[name], X_val_scaled, name)

    # ============================================================
    # Test セットの最終評価（封印解除）
    # ============================================================
    print("\n" + "=" * 60)
    print("Final evaluation on TEST set...")

    test_results = []
    for name, model in trained_models.items():
        r_test = evaluate(model, X_test_scaled, y_test, name, 'Test')
        test_results.append(r_test)

    test_df = pd.DataFrame(test_results)
    test_df.to_csv(os.path.join(OUTPUT_DIR, 'test_results.csv'), index=False)
    print("\nTest results:")
    print(test_df.to_string(index=False))

    # ============================================================
    # 廃業確率スコアの出力（全データ）
    # ============================================================
    print("\n" + "=" * 60)
    print("Generating closure probability scores for all stores...")

    X_all_cont = scaler.transform(X[cont_cols])
    X_all_scaled = pd.DataFrame(X_all_cont, columns=cont_cols)
    for col in flag_cols:
        X_all_scaled[col] = X[col].values

    best_model = trained_models['LightGBM']
    df['closure_probability'] = best_model.predict_proba(X_all_scaled)[:, 1]

    score_cols = ['id', 'name', 'latitude', 'longitude', 'is_closed', 'closure_probability']
    score_cols = [c for c in score_cols if c in df.columns]
    df[score_cols].to_csv(os.path.join(OUTPUT_DIR, 'closure_scores.csv'), index=False, encoding='utf-8-sig')
    print(f"Closure scores saved to {OUTPUT_DIR}/closure_scores.csv")

    print("\n" + "=" * 60)
    print("All steps completed!")


if __name__ == '__main__':
    main()

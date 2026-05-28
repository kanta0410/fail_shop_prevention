# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import shap
import matplotlib.pyplot as plt

def main():
    print("Loading data...")
    data_path = r'c:\Users\kanta\workspace\projects\inturn\廃業予測\data\processed\nagoya_vol2.csv'
    docs_dir = r'c:\Users\kanta\workspace\projects\inturn\廃業予測\data\customer\figures'
    os.makedirs(docs_dir, exist_ok=True)
    
    df = pd.read_csv(data_path)
    
    # 目的変数と特徴量の分離
    target_col = 'is_closed'
    drop_cols = [target_col, 'id', 'name']
    features = [c for c in df.columns if c not in drop_cols]
    
    X = df[features].copy()
    y = df[target_col].copy()
    
    print(f"Dataset shape: {X.shape}")
    
    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # 不均衡対策用の class weight 算出
    ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1)
    
    # CV準備
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    models_config = {
        'LightGBM': lgb.LGBMClassifier(class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1),
        'XGBoost': xgb.XGBClassifier(scale_pos_weight=ratio, random_state=42, use_label_encoder=False, eval_metric='logloss', n_jobs=-1),
        'CatBoost': CatBoostClassifier(auto_class_weights='Balanced', random_state=42, thread_count=-1, verbose=False)
    }
    
    oof_preds = {m: np.zeros(len(X_train)) for m in models_config.keys()}
    test_preds = {m: np.zeros(len(X_test)) for m in models_config.keys()}
    
    print("Starting Cross Validation...")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        print(f"--- Fold {fold+1}/{n_splits} ---")
        X_tr, X_va = X_train.iloc[train_idx].copy(), X_train.iloc[val_idx].copy()
        y_tr, y_va = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        # 欠損値補完
        imputer = SimpleImputer(strategy='median')
        X_tr_imp = imputer.fit_transform(X_tr)
        X_va_imp = imputer.transform(X_va)
        
        # 標準化
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr_imp)
        X_va_scaled = scaler.transform(X_va_imp)
        
        for name, model in models_config.items():
            model.fit(X_tr_scaled, y_tr)
            preds = model.predict_proba(X_va_scaled)[:, 1]
            oof_preds[name][val_idx] = preds
            
    # CV結果の評価
    print("\n--- CV Results (OOF) ---")
    best_f1_thresholds = {}
    for name in models_config.keys():
        pr_auc = average_precision_score(y_train, oof_preds[name])
        precision, recall, thresholds = precision_recall_curve(y_train, oof_preds[name])
        f1_scores = 2 * recall * precision / (recall + precision + 1e-10)
        best_thresh = thresholds[np.argmax(f1_scores)] if len(thresholds) > 0 else 0.5
        best_f1 = np.max(f1_scores)
        best_f1_thresholds[name] = best_thresh
        print(f"{name} | PR-AUC: {pr_auc:.4f}, Max F1: {best_f1:.4f} (Threshold: {best_thresh:.4f})")
        
    # アンサンブル (単純平均)
    ensemble_oof = np.mean([oof_preds[name] for name in models_config.keys()], axis=0)
    ens_pr_auc = average_precision_score(y_train, ensemble_oof)
    precision, recall, thresholds = precision_recall_curve(y_train, ensemble_oof)
    f1_scores = 2 * recall * precision / (recall + precision + 1e-10)
    ens_best_thresh = thresholds[np.argmax(f1_scores)] if len(thresholds) > 0 else 0.5
    ens_best_f1 = np.max(f1_scores)
    print(f"Ensemble | PR-AUC: {ens_pr_auc:.4f}, Max F1: {ens_best_f1:.4f} (Threshold: {ens_best_thresh:.4f})")
    
    # --- 全学習データ(Train)で再学習し、テストデータを予測 ---
    print("\nTraining on full Train data for Test evaluation and SHAP...")
    imputer = SimpleImputer(strategy='median')
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)
    
    final_models = {}
    for name, model in models_config.items():
        model.fit(X_train_scaled, y_train)
        final_models[name] = model
        test_preds[name] = model.predict_proba(X_test_scaled)[:, 1]
        
    ens_test_preds = np.mean([test_preds[name] for name in models_config.keys()], axis=0)
    test_pr_auc = average_precision_score(y_test, ens_test_preds)
    
    # 評価時のF1 (CVで求めた閾値を使用)
    test_pred_labels = (ens_test_preds >= ens_best_thresh).astype(int)
    test_f1 = f1_score(y_test, test_pred_labels)
    
    print(f"Test Set Ensemble PR-AUC: {test_pr_auc:.4f}")
    print(f"Test Set Ensemble F1 Score: {test_f1:.4f}")
    
    # --- SHAP解析 (LightGBMを使用) ---
    print("\nGenerating SHAP values using LightGBM...")
    explainer = shap.TreeExplainer(final_models['LightGBM'])
    shap_values = explainer.shap_values(X_test_scaled)
    
    if isinstance(shap_values, list):
        shap_values_to_plot = shap_values[1]
    else:
        shap_values_to_plot = shap_values
        
    # SHAPサマリープロット
    plt.figure()
    shap.summary_plot(shap_values_to_plot, X_test_scaled, feature_names=features, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(docs_dir, 'shap_summary_lgbm.png'))
    plt.close()
    
    # 上位特徴量の出力
    vals = np.abs(shap_values_to_plot).mean(0)
    feature_importance = pd.DataFrame(list(zip(features, vals)), columns=['Feature','Importance'])
    feature_importance.sort_values(by=['Importance'], ascending=False, inplace=True)
    
    print("\nTop 15 Features for Interpretation:")
    print(feature_importance.head(15).to_markdown())

if __name__ == '__main__':
    main()

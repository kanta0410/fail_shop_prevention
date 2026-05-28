import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import shap
import matplotlib.pyplot as plt

def prepare_data(df):
    exclude_cols = ['id', 'name', 'nearest_station_name', 'is_closed']
    features = [c for c in df.columns if c not in exclude_cols]
    
    # We will impute 'rating' inside CV, but to simplify, we can impute it based on training set later.
    return df[features], df['is_closed'], features

def train_and_evaluate(model_name, model, X, y, use_smote=False, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    pr_aucs = []
    roc_aucs = []
    
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Impute rating
        median_rating = X_train['rating'].median()
        X_train.loc[:, 'rating'] = X_train['rating'].fillna(median_rating)
        X_val.loc[:, 'rating'] = X_val['rating'].fillna(median_rating)
        
        # Standardize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        if use_smote:
            pos_count = np.sum(y_train == 1)
            if pos_count > 1:
                k = min(5, pos_count - 1)
                smote = SMOTE(k_neighbors=k, random_state=42)
                X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
            
        # Fit
        if model_name == 'CatBoost':
            model.fit(X_train_scaled, y_train, verbose=False)
        else:
            model.fit(X_train_scaled, y_train)
            
        # Predict
        y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
        
        pr_aucs.append(average_precision_score(y_val, y_pred_proba))
        try:
            roc_aucs.append(roc_auc_score(y_val, y_pred_proba))
        except ValueError:
            pass
        
    return np.mean(pr_aucs) if pr_aucs else 0, np.mean(roc_aucs) if roc_aucs else 0

def main():
    data_path = r'c:\Users\kanta\workspace\projects\inturn\廃業予測\data\processed\nagoya.csv'
    docs_dir = r'c:\Users\kanta\workspace\projects\inturn\廃業予測\docs'
    os.makedirs(docs_dir, exist_ok=True)
    
    df = pd.read_csv(data_path)
    X, y, feature_names = prepare_data(df)
    
    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Define models
    ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1)
    
    models = {
        'LightGBM_Balanced': (lgb.LGBMClassifier(class_weight='balanced', random_state=42, n_jobs=-1), False),
        'LightGBM_SMOTE': (lgb.LGBMClassifier(random_state=42, n_jobs=-1), True),
        'XGBoost_Balanced': (xgb.XGBClassifier(scale_pos_weight=ratio, random_state=42, use_label_encoder=False, eval_metric='logloss', n_jobs=-1), False),
        'XGBoost_SMOTE': (xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', n_jobs=-1), True),
        'CatBoost_Balanced': (CatBoostClassifier(auto_class_weights='Balanced', random_state=42, thread_count=-1), False)
    }
    
    results = {}
    print("Starting CV Evaluation...")
    for name, (model, use_smote) in models.items():
        pr, roc = train_and_evaluate(name.split('_')[0], model, X_train, y_train, use_smote)
        results[name] = {'PR-AUC': pr, 'ROC-AUC': roc}
        print(f"{name} -> PR-AUC: {pr:.4f}, ROC-AUC: {roc:.4f}")
        
    # Select Best Model based on PR-AUC
    best_name = max(results, key=lambda k: results[k]['PR-AUC'])
    print(f"\nBest Model: {best_name}")
    
    # Train Best Model on full train set and evaluate on test set
    best_model, best_use_smote = models[best_name]
    
    # Final Pipeline
    median_rating = X_train['rating'].median()
    X_train_final = X_train.copy()
    X_test_final = X_test.copy()
    
    X_train_final['rating'] = X_train_final['rating'].fillna(median_rating)
    X_test_final['rating'] = X_test_final['rating'].fillna(median_rating)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_final)
    X_test_scaled = scaler.transform(X_test_final)
    
    if best_use_smote:
        pos_count = np.sum(y_train == 1)
        if pos_count > 1:
            k = min(5, pos_count - 1)
            smote = SMOTE(k_neighbors=k, random_state=42)
            X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
        
    print("Training best model on full training data...")
    if 'CatBoost' in best_name:
        best_model.fit(X_train_scaled, y_train, verbose=False)
    else:
        best_model.fit(X_train_scaled, y_train)
        
    y_test_pred = best_model.predict_proba(X_test_scaled)[:, 1]
    test_pr = average_precision_score(y_test, y_test_pred)
    print(f"Test PR-AUC: {test_pr:.4f}")
    
    # SHAP
    print("Generating SHAP values...")
    explainer = shap.TreeExplainer(best_model)
    # Using a sample to speed up SHAP if data is large, but 8000 is fine
    shap_values = explainer.shap_values(X_test_scaled)
    
    # If binary classification, SHAP might return a list [class0, class1] or just class1 depending on model
    if isinstance(shap_values, list):
        shap_values_to_plot = shap_values[1]
    else:
        shap_values_to_plot = shap_values
        
    plt.figure()
    shap.summary_plot(shap_values_to_plot, X_test_scaled, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(docs_dir, f'shap_beeswarm_{best_name}.png'))
    plt.close()
    
    plt.figure()
    shap.summary_plot(shap_values_to_plot, X_test_scaled, feature_names=feature_names, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(docs_dir, f'shap_bar_{best_name}.png'))
    plt.close()
    
    # Print top 15 features for report
    vals = np.abs(shap_values_to_plot).mean(0)
    feature_importance = pd.DataFrame(list(zip(feature_names, vals)), columns=['col_name','feature_importance_vals'])
    feature_importance.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True)
    print("\nTop 15 Features from SHAP:")
    print(feature_importance.head(15).to_markdown())

    # LIME
    from lime.lime_tabular import LimeTabularExplainer
    print("Generating LIME explanation...")
    lime_explainer = LimeTabularExplainer(X_train_scaled, feature_names=feature_names, class_names=['Alive', 'Closed'], discretize_continuous=True)
    
    # Pick a sample that is actually 'Closed'
    closed_indices = np.where(y_test.values == 1)[0]
    if len(closed_indices) > 0:
        idx = closed_indices[0]
        exp = lime_explainer.explain_instance(X_test_scaled[idx], best_model.predict_proba, num_features=10)
        exp.save_to_file(os.path.join(docs_dir, f'lime_explanation_{best_name}.html'))
        print(f"LIME explanation saved to lime_explanation_{best_name}.html")

if __name__ == '__main__':
    main()

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from imblearn.over_sampling import SMOTE

def prepare_data(df):
    exclude_cols = ['id', 'name', 'nearest_station_name', 'is_closed']
    features = [c for c in df.columns if c not in exclude_cols]
    return df[features], df['is_closed'], features

def train_and_evaluate(model, X, y, use_smote=False, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    pr_aucs = []
    roc_aucs = []
    
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        median_rating = X_train['rating'].median()
        X_train.loc[:, 'rating'] = X_train['rating'].fillna(median_rating)
        X_val.loc[:, 'rating'] = X_val['rating'].fillna(median_rating)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        if use_smote:
            pos_count = np.sum(y_train == 1)
            if pos_count > 1:
                k = min(5, pos_count - 1)
                smote = SMOTE(k_neighbors=k, random_state=42)
                X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
            
        model.fit(X_train_scaled, y_train)
        y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
        
        pr_aucs.append(average_precision_score(y_val, y_pred_proba))
        try:
            roc_aucs.append(roc_auc_score(y_val, y_pred_proba))
        except ValueError:
            pass
        
    return np.mean(pr_aucs) if pr_aucs else 0, np.mean(roc_aucs) if roc_aucs else 0

def main():
    data_path = r'c:\Users\kanta\workspace\projects\inturn\廃業予測\data\processed\nagoya.csv'
    
    df = pd.read_csv(data_path)
    X, y, feature_names = prepare_data(df)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    models = {
        'Lasso (L1)_Balanced': (LogisticRegression(penalty='l1', solver='liblinear', class_weight='balanced', random_state=42, max_iter=1000), False),
        'Lasso (L1)_SMOTE': (LogisticRegression(penalty='l1', solver='liblinear', random_state=42, max_iter=1000), True),
        'ElasticNet_Balanced': (LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, class_weight='balanced', random_state=42, max_iter=2000), False),
        'ElasticNet_SMOTE': (LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, random_state=42, max_iter=2000), True)
    }
    
    print("Starting CV Evaluation for Linear Models...")
    for name, (model, use_smote) in models.items():
        pr, roc = train_and_evaluate(model, X_train, y_train, use_smote)
        print(f"{name} -> PR-AUC: {pr:.4f}, ROC-AUC: {roc:.4f}")
        
if __name__ == '__main__':
    main()

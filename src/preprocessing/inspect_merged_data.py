import pandas as pd

def main():
    FEATURES_CSV = "data/processed/nagoya_features_all.csv"
    df = pd.read_csv(FEATURES_CSV)
    
    print("=== Merged Dataset Overview ===")
    print(f"Total Rows: {len(df)}")
    print(f"Columns: {df.columns.tolist()[:10]} ... and {len(df.columns) - 10} more")
    
    print("\n=== target (is_closed) Distribution ===")
    print(df['is_closed'].value_counts(dropna=False))
    
    print("\n=== review_diff Statistics ===")
    print(df['review_diff'].describe())
    
    print("\n=== rating_diff Statistics ===")
    print(df['rating_diff'].describe())
    
    print("\n=== has_opening_hours Distribution ===")
    print(df['has_opening_hours'].value_counts(dropna=False))
    
    # 廃業店舗(is_closed=1)と生存店舗(is_closed=0)での新特徴量の違いを確認
    print("\n=== Feature Comparison (Closed vs Operational) ===")
    comparison = df.groupby('is_closed')[['review_diff', 'rating_diff', 'has_opening_hours']].mean()
    print(comparison)

if __name__ == "__main__":
    main()

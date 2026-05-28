import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def create_nagoya_standard():
    # File paths
    input_path = r'c:\Users\kanta\workspace\projects\inturn\廃業予測\data\processed\nagoya.csv'
    output_dir = r'c:\Users\kanta\workspace\projects\inturn\廃業予測\data\warehouse'
    output_path = os.path.join(output_dir, 'nagoya_standard.csv')
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    df = pd.read_csv(input_path)
    print(f"Initial shape: {df.shape}")
    
    # 1. Missing Value Imputation
    # Only 'rating' has missing values (74 items) -> Fill with median
    if df['rating'].isnull().sum() > 0:
        median_rating = df['rating'].median()
        df['rating'] = df['rating'].fillna(median_rating)
        print(f"Imputed 'rating' missing values with median: {median_rating}")
        
    # Check if there are any other missing values
    missing_sum = df.isnull().sum().sum()
    print(f"Total missing values after imputation: {missing_sum}")
    
    # 2. Standardization
    # Exclude non-numeric, target ('is_closed'), and any ID columns if they shouldn't be scaled
    exclude_cols = ['id', 'name', 'nearest_station_name', 'is_closed']
    
    # Select columns to scale
    cols_to_scale = [col for col in df.select_dtypes(include=[np.number]).columns if col not in exclude_cols]
    
    scaler = StandardScaler()
    
    # Standardize
    df_scaled = df.copy()
    df_scaled[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    
    print(f"Standardized {len(cols_to_scale)} columns.")
    
    # 3. Save to warehouse
    df_scaled.to_csv(output_path, index=False)
    print(f"Saved standardized dataset to {output_path}")

if __name__ == '__main__':
    create_nagoya_standard()

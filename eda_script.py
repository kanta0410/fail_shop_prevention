import pandas as pd
import numpy as np

# Load data
df = pd.read_csv(r'c:\Users\kanta\workspace\projects\inturn\廃業予測\data\processed\nagoya.csv')

# Basic info
print("### Shape")
print(df.shape)
print("\n")

# Calculate summary stats for numerical columns
num_cols = df.select_dtypes(include=[np.number]).columns

summary = pd.DataFrame({
    'Dtype': df.dtypes,
    'Missing_Count': df.isnull().sum(),
    'Missing_Ratio': df.isnull().sum() / len(df) * 100,
    'Unique_Count': df.nunique()
})

num_summary = df[num_cols].describe().T
num_summary['Variance'] = df[num_cols].var()
num_summary['Skewness'] = df[num_cols].skew()
num_summary['Kurtosis'] = df[num_cols].kurtosis()

# Join
full_summary = summary.join(num_summary)

# Display options
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 1000)

print("### Summary Table")
print(full_summary.to_markdown())

# Check for outliers (e.g. values > 3 std from mean)
print("\n### Potential Outliers (> 3 std)")
outlier_counts = {}
for col in num_cols:
    mean = df[col].mean()
    std = df[col].std()
    if std > 0:
        outliers = df[(df[col] < mean - 3 * std) | (df[col] > mean + 3 * std)]
        outlier_counts[col] = len(outliers)
        
outlier_df = pd.DataFrame(list(outlier_counts.items()), columns=['Column', 'Outlier_Count'])
outlier_df['Outlier_Ratio'] = outlier_df['Outlier_Count'] / len(df) * 100
print(outlier_df[outlier_df['Outlier_Count'] > 0].sort_values('Outlier_Count', ascending=False).to_markdown())

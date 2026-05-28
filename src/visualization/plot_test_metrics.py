import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 日本語フォントの設定（必要に応じて）
# plt.rcParams['font.family'] = 'Meiryo'

# データの読み込み
input_path = r'c:\Users\kanta\workspace\projects\inturn\廃業予測\data\output\eda3\test_metrics_eda3.csv'
output_path = r'c:\Users\kanta\workspace\projects\inturn\廃業予測\data\output\eda3\test_metrics_comparison.png'

df = pd.read_csv(input_path)

# グラフの設定
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Model Performance Comparison (Test Metrics EDA3)', fontsize=16)

metrics = ['AUC-ROC', 'AUC-PR', 'F1', 'LogLoss']
colors = ['#1f77b4', '#ff7f0e']

for i, metric in enumerate(metrics):
    ax = axes[i // 2, i % 2]
    sns.barplot(x='model', y=metric, data=df, ax=ax, palette=colors)
    ax.set_title(f'{metric}')
    ax.set_xlabel('')
    ax.set_ylabel('Score')
    
    # 値をバーの上に表示
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.4f}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha = 'center', va = 'center', 
                    xytext = (0, 9), 
                    textcoords = 'offset points')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(output_path, dpi=300)
print(f'Saved plot to {output_path}')

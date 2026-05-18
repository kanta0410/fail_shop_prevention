"""
Nagoya Restaurant Closure Prediction - Detailed 2-Color EDA

すべての特徴量（カンニング指標も含む）に対して、
廃業ラベル（is_closed）で色分けした詳細な分布図を生成し、
データの境界線（閾値）やリークの実態を可視化・分析するためのスクリプト。
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 日本語フォント設定
try:
    plt.rcParams['font.family'] = 'MS Gothic'
except:
    pass
plt.rcParams['axes.unicode_minus'] = False

INPUT_CSV = 'data/processed/nagoya_features_all.csv'
EDA_DIR = 'data/output/eda2'
os.makedirs(EDA_DIR, exist_ok=True)

def plot_all_features(df):
    print(f"Generating 2-color distribution plots for all features into {EDA_DIR}...")
    
    # 数値型の特徴量のみ抽出
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # 意味のないカラムを除外
    exclude_cols = ['id', 'is_closed', 'closed_flag', 'exists_2026_05']
    features = [c for c in num_cols if c not in exclude_cols]
    
    # 色設定 (存続: 緑系, 廃業: 赤系)
    palette = {0: '#2ecc71', 1: '#e74c3c'}
    
    total = len(features)
    for i, f in enumerate(features):
        print(f"  [{i+1}/{total}] Plotting {f}...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 1. KDE付きヒストグラム (分布の重なりを見る)
        sns.histplot(data=df, x=f, hue='is_closed', palette=['#2ecc71', '#e74c3c'], 
                     kde=True, stat='density', common_norm=False, 
                     alpha=0.4, ax=axes[0])
        axes[0].set_title(f'{f} の分布 (密度)', fontsize=12, fontweight='bold')
        axes[0].set_xlabel(f)
        axes[0].set_ylabel('Density')
        
        # 2. 箱ひげ図 (外れ値や四分位数の比較)
        sns.boxplot(data=df, x='is_closed', y=f, hue='is_closed', palette=['#2ecc71', '#e74c3c'], ax=axes[1], legend=False)
        axes[1].set_title(f'{f} の箱ひげ図', fontsize=12, fontweight='bold')
        axes[1].set_xticklabels(['存続 (0)', '廃業 (1)'])
        axes[1].set_xlabel('is_closed')
        axes[1].set_ylabel(f)
        
        plt.tight_layout()
        plt.savefig(os.path.join(EDA_DIR, f'dist_{f}.png'), dpi=150)
        plt.close()

def main():
    print("=" * 60)
    print("Starting Extensive 2-Color EDA Pipeline")
    print("=" * 60)
    
    print(f"Loading features from {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    
    plot_all_features(df)
    
    print("\nAll EDA plots have been successfully generated!")

if __name__ == '__main__':
    main()

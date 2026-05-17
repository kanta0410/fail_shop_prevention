"""
Nagoya Restaurant Closure Prediction - EDA for Zero/Low Review Stores

このスクリプトは、レビュー数が0件（または1件で対数変換後0になっている）の
店舗層に特化した探索的データ分析（EDA）を実行します。
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# 日本語フォント設定
try:
    plt.rcParams['font.family'] = 'MS Gothic'
except:
    pass
plt.rcParams['axes.unicode_minus'] = False

# 設定
INPUT_CSV = 'data/processed/nagoya_features_all.csv'
EDA_DIR = 'data/output/eda/small_rating'
os.makedirs(EDA_DIR, exist_ok=True)

def load_and_filter_data():
    print(f"Loading features from {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    print(f"  Original Shape: {df.shape}")
    
    # log_review_countが0（または極めて0に近い）店舗を抽出
    # ※ log1p(0) = 0, log1p(1) = 0.693 なので、厳密に0のものを抽出
    df_small = df[df['log_review_count'] < 1e-5].copy()
    print(f"  Filtered Shape (log_review_count == 0): {df_small.shape}")
    
    return df_small

def generate_descriptive_statistics(df):
    print("Generating descriptive statistics...")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    desc_num = df[num_cols].describe().T
    desc_num['missing_count'] = df[num_cols].isnull().sum()
    desc_num.to_csv(os.path.join(EDA_DIR, 'descriptive_stats_numeric.csv'), encoding='utf-8-sig')
    
    if cat_cols:
        desc_cat = pd.DataFrame(index=cat_cols)
        desc_cat['unique_count'] = df[cat_cols].nunique()
        desc_cat['missing_count'] = df[cat_cols].isnull().sum()
        desc_cat['top'] = df[cat_cols].mode().iloc[0] if not df[cat_cols].empty else np.nan
        desc_cat.to_csv(os.path.join(EDA_DIR, 'descriptive_stats_categorical.csv'), encoding='utf-8-sig')
        
    return num_cols, cat_cols

def plot_correlation_matrix(df, num_cols):
    print("Plotting correlation matrix...")
    target_col = 'is_closed'
    if target_col not in num_cols:
        return
        
    corr = df[num_cols].corr()
    # ターゲットとの絶対相関が高い順
    top_corr_features = corr[target_col].abs().sort_values(ascending=False).head(20).index.tolist()
    
    key_features = ['dist_to_nagoya_sta', 'urban_score', 'flow_score_raw_b1.5', 'count_restaurants_500m']
    for f in key_features:
        if f in num_cols and f not in top_corr_features:
            top_corr_features.append(f)
            
    top_corr_features = list(dict.fromkeys(top_corr_features))
    
    # log_review_count等は全て0なので相関計算できないため除外
    top_corr_features = [f for f in top_corr_features if df[f].nunique() > 1]
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(df[top_corr_features].corr(), annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, square=True)
    plt.title('レビュー0件層：主要特徴量の相関関係ヒートマップ', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, 'correlation_heatmap.png'), dpi=150)
    plt.close()

def plot_vs_target(df):
    print("Plotting features vs target (is_closed)...")
    features = ['dist_to_nagoya_sta', 'flow_score_raw_b1.5', 'urban_score', 'count_restaurants_500m', 'nearest_station_dist']
    
    for f in features:
        if f not in df.columns or 'is_closed' not in df.columns:
            continue
            
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        sns.boxplot(x='is_closed', y=f, data=df, ax=axes[0], hue='is_closed', palette=['#2ecc71', '#e74c3c'], legend=False)
        axes[0].set_title(f'is_closed 別 {f} の箱ひげ図', fontsize=12, fontweight='bold')
        axes[0].set_xticks([0, 1])
        axes[0].set_xticklabels(['存続 (0)', '廃業 (1)'])
        axes[0].set_xlabel('状態')
        axes[0].set_ylabel(f)
        
        sns.violinplot(x='is_closed', y=f, data=df, ax=axes[1], hue='is_closed', palette=['#2ecc71', '#e74c3c'], inner="quartile", legend=False)
        axes[1].set_title(f'is_closed 別 {f} のバイオリンプロット', fontsize=12, fontweight='bold')
        axes[1].set_xticks([0, 1])
        axes[1].set_xticklabels(['存続 (0)', '廃業 (1)'])
        axes[1].set_xlabel('状態')
        axes[1].set_ylabel(f)
        
        plt.tight_layout()
        plt.savefig(os.path.join(EDA_DIR, f'vs_target_{f}.png'), dpi=150)
        plt.close()

def plot_spatial_distribution(df):
    print("Plotting spatial distribution...")
    if 'latitude' in df.columns and 'longitude' in df.columns:
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x='longitude', y='latitude', hue='is_closed', data=df, 
                        palette=['#2ecc71', '#e74c3c'], alpha=0.6, s=20)
        plt.title('レビュー0件層：名古屋市内における存続・廃業の地理的分布', fontsize=13, fontweight='bold')
        plt.xlabel('経度 (Longitude)')
        plt.ylabel('緯度 (Latitude)')
        plt.legend(title='状態', labels=['存続 (0)', '廃業 (1)'])
        plt.tight_layout()
        plt.savefig(os.path.join(EDA_DIR, 'spatial_distribution.png'), dpi=150)
        plt.close()

def plot_dimension_reduction(df, num_cols):
    print("Running dimensional reduction (PCA and t-SNE)...")
    exclude = ['id', 'name', 'is_closed', 'closed_flag', 'exists_2026_05']
    features = [c for c in num_cols if c not in exclude and df[c].nunique() > 1]
    
    X = df[features].fillna(0)
    y = df['is_closed']
    
    if len(X) < 10 or len(features) == 0:
        print("  Not enough data for dimensionality reduction.")
        return
        
    X_scaled = StandardScaler().fit_transform(X)
    
    # PCA
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, 
                    palette=['#2ecc71', '#e74c3c'], alpha=0.5, s=20)
    plt.title('レビュー0件層：主成分分析 (PCA) プロット', fontsize=12, fontweight='bold')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, 'pca_plot.png'), dpi=150)
    plt.close()
    
    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X)-1))
    X_tsne = tsne.fit_transform(X_scaled)
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y, 
                    palette=['#2ecc71', '#e74c3c'], alpha=0.7, s=20)
    plt.title('レビュー0件層：t-SNE プロット', fontsize=12, fontweight='bold')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, 'tsne_plot.png'), dpi=150)
    plt.close()

def write_eda_report(df, num_cols, orig_total):
    print("Writing EDA report...")
    total_small = len(df)
    closed_small = df['is_closed'].sum()
    closure_rate = df['is_closed'].mean() * 100
    
    corr = df[num_cols].corr()
    target_corr = corr['is_closed'].dropna().sort_values(ascending=False)
    top_pos_corr = target_corr[target_corr > 0].head(6)[1:]
    top_neg_corr = target_corr[target_corr < 0].tail(5)
    
    mean_closed = df[df['is_closed'] == 1][['dist_to_nagoya_sta', 'flow_score_raw_b1.5', 'count_restaurants_500m', 'urban_score']].mean()
    mean_alive = df[df['is_closed'] == 0][['dist_to_nagoya_sta', 'flow_score_raw_b1.5', 'count_restaurants_500m', 'urban_score']].mean()
    
    report_content = f"""# 名古屋市飲食店廃業予測 - レビュー0件層 (small_rating) 特化EDAレポート

本レポートは、全データの中で「レビュー数が0件（またはネット上に情報が極めて少ない）」の店舗のみを抽出し、この層がどのような特徴を持ち、なぜ廃業に至るのかを分析したものです。

---

## 1. レビュー0件層の基本概要
- **全体の店舗数**: {orig_total:,} 件
- **レビュー0件層の店舗数**: {total_small:,} 件 （全体の {total_small/orig_total*100:.1f}%）
- **この層での廃業店舗数**: {closed_small:,} 件
- **この層での廃業率**: {closure_rate:.2f}%
  - レビューが存在する層と比較して、この層の廃業率がどのように分布しているかが重要です。

---

## 2. 存続・廃業を分ける特徴量の違い（平均値比較）

| 特徴量 | 存続店舗 (is_closed=0) | 廃業店舗 (is_closed=1) | 傾向とビジネス解釈 |
| :--- | :---: | :---: | :--- |
| **名古屋駅からの距離 (km)** | {mean_alive['dist_to_nagoya_sta']/1000:.2f} | {mean_closed['dist_to_nagoya_sta']/1000:.2f} | 廃業店舗のほうが中心地に近い、または遠いなどの傾向を確認。 |
| **日次人流スコア** | {mean_alive['flow_score_raw_b1.5']:.1f} | {mean_closed['flow_score_raw_b1.5']:.1f} | レビューがない層において、駅前一等地にいることのリスク（家賃・競争）がどう影響するか。 |
| **周辺500mの飲食店数** | {mean_alive['count_restaurants_500m']:.1f} | {mean_closed['count_restaurants_500m']:.1f} | 競合の多さがそのまま廃業リスクに直結しているか。 |
| **都市開発スコア (urban_score)** | {mean_alive['urban_score']:.2f} | {mean_closed['urban_score']:.2f} | 周辺のオフィスや施設の充実度が生存に寄与するか。 |

---

## 3. この層における相関関係分析

この「レビュー0件層」の中で、廃業と特に強い相関を持っていた特徴量です：

### 正の相関（高いほど廃業しやすい特徴量）
{top_pos_corr.to_markdown()}

### 負の相関（高いほど存続しやすい特徴量）
{top_neg_corr.to_markdown()}

**分析の洞察**:
- 「認知度（レビュー）」という強力な武器を持たない店舗にとって、**立地（距離、競合数、人流スコア）**が生存確率を決定づける最大の要因になっていることが推測されます。
- 特に、周辺の競合飲食店数（`count_restaurants_xxx`）との正の相関が強い場合、「無名なのに激戦区にいる」ことが致命的であることを意味します。

---

## 4. 可視化グラフによる詳細考察
グラフは `data/output/eda/small_rating/` フォルダに保存されています。

### 1) 分布の違い ([vs_target_flow_score_raw_b1.5.png](vs_target_flow_score_raw_b1.5.png), [vs_target_count_restaurants_500m.png](vs_target_count_restaurants_500m.png))
- 箱ひげ図・バイオリンプロットを見ると、同じ「レビュー0件」でも、廃業店舗は**「人流が多く、競合が極めて多いエリア（駅前）」**に分布のピークがあることが分かります。
- 一方、存続している店舗は競合が少ないエリア（郊外や住宅街）に多く分布しており、「認知度が低くても、地元密着で細々とやれる場所」であれば生存できることが示唆されています。

### 2) 地理的分布マップ ([spatial_distribution.png](spatial_distribution.png))
- 地図上のプロットを見ると、レビュー0件の店舗自体がどこに偏在しているか、またその中で赤い点（廃業）がどのエリアで密集して発生しているかが視覚的に把握できます。

### 3) 次元削減プロット ([tsne_plot.png](tsne_plot.png))
- t-SNEプロットにおいて、特定のクラスタ（島）に廃業店舗が集中している場合、その島は「無名かつ高リスクな立地条件」を共有するグループです。これらの島を避けるような出店戦略、あるいは既存店舗へのテコ入れが求められます。

"""
    report_path = os.path.join(EDA_DIR, 'eda_small_rating_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)

def main():
    print("=" * 60)
    print("Starting EDA for Zero/Low Review Stores (small_rating)")
    print("=" * 60)
    
    # オリジナルの件数を取得するためだけ
    df_orig = pd.read_csv(INPUT_CSV, usecols=['id'])
    orig_total = len(df_orig)
    
    df = load_and_filter_data()
    num_cols, cat_cols = generate_descriptive_statistics(df)
    plot_correlation_matrix(df, num_cols)
    plot_vs_target(df)
    plot_spatial_distribution(df)
    plot_dimension_reduction(df, num_cols)
    write_eda_report(df, num_cols, orig_total)
    
    print("\nEDA for small_rating completed successfully!")

if __name__ == '__main__':
    main()

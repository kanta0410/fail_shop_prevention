"""
Step 2-5: フォワードテスト 評価・可視化パイプライン
Step1で取得したOSM存在確認データと予測スコアを結合し、
予測精度の評価・ビジネス向け可視化・サマリー出力を行う。
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import folium
from folium.plugins import HeatMap
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    classification_report, confusion_matrix,
    roc_curve, precision_recall_curve
)

# ============================================================
# 設定
# ============================================================
FORWARD_RAW_CSV = 'data/processed/forward_test_raw.csv'
OUTPUT_DIR      = 'data/output/forward_test'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 損失シミュレーション設定
AVG_LOSS_PER_STORE = 10_000_000  # 1,000万円
N_STORES_SIM = 100

# 日本語フォント設定
try:
    plt.rcParams['font.family'] = 'MS Gothic'
except:
    pass


def load_and_prepare():
    """データ読み込みと前処理"""
    print("Loading forward test data...")
    df = pd.read_csv(FORWARD_RAW_CSV)
    print(f"  Total records: {len(df)}")

    # 取得失敗を除外
    df_valid = df[df['exists_2026_05'].notna()].copy()
    failed_count = len(df) - len(df_valid)
    print(f"  Valid: {len(df_valid)} (excluded {failed_count} None records)")

    # exists_2026_05はstr('True'/'False')で保存されている可能性があるため変換
    if df_valid['exists_2026_05'].dtype == object:
        df_valid['exists_2026_05'] = df_valid['exists_2026_05'].map(
            {'True': True, 'False': False, True: True, False: False}
        )

    # 新規廃業フラグ: 2026年5月に存在しない = 廃業
    df_valid['new_haigyo_2026_05'] = (~df_valid['exists_2026_05']).astype(int)

    print(f"  New closures (2026-05): {df_valid['new_haigyo_2026_05'].sum()}")
    print(f"  Closure rate: {df_valid['new_haigyo_2026_05'].mean():.2%}")

    return df_valid


def step3_evaluate(df):
    """Step 3: 予測精度の評価"""
    print("\n" + "=" * 50)
    print("Step 3: Evaluating prediction accuracy...")

    y_true = df['new_haigyo_2026_05'].values
    y_prob = df['haigyo_prob'].values
    y_pred = (y_prob >= 0.5).astype(int)

    # すべてのクラスが存在するかチェック（廃業店舗が0件だとAUCが計算できない）
    if y_true.sum() == 0:
        print("  WARNING: No closure events found. Cannot compute AUC metrics.")
        return None, None, None

    auc_roc = roc_auc_score(y_true, y_prob)
    auc_pr  = average_precision_score(y_true, y_prob)
    f1      = classification_report(y_true, y_pred, output_dict=True)

    print(f"  AUC-ROC: {auc_roc:.4f}")
    print(f"  AUC-PR:  {auc_pr:.4f}")
    print()
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['存続', '廃業']))

    # ROC / PR 曲線を保存
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    axes[0].plot(fpr, tpr, color='#e74c3c', lw=2,
                 label=f'LightGBM (AUC={auc_roc:.3f})')
    axes[0].plot([0, 1], [0, 1], 'k--', lw=1)
    axes[0].set_title('ROC曲線 (フォワードテスト)', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('偽陽性率 (FPR)')
    axes[0].set_ylabel('真陽性率 (TPR)')
    axes[0].legend()

    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    axes[1].plot(rec, prec, color='#3498db', lw=2,
                 label=f'LightGBM (AUC-PR={auc_pr:.3f})')
    axes[1].axhline(y=y_true.mean(), color='gray', linestyle='--',
                    label=f'ベースライン ({y_true.mean():.3f})')
    axes[1].set_title('Precision-Recall曲線 (フォワードテスト)', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('再現率 (Recall)')
    axes[1].set_ylabel('適合率 (Precision)')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'roc_pr_forward.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: roc_pr_forward.png")

    return auc_roc, auc_pr, y_true.mean()


def step4_viz_quintile(df):
    """可視化①: リスクスコア別実際の廃業率（5分位）"""
    print("\nGenerating quintile chart...")

    df = df.copy()
    df['risk_quintile'] = pd.qcut(
        df['haigyo_prob'],
        q=5,
        labels=['低リスク\n(0-20%)', 'やや低\n(20-40%)',
                '中リスク\n(40-60%)', 'やや高\n(60-80%)', '高リスク\n(80-100%)']
    )

    actual_rate = df.groupby('risk_quintile', observed=True)['new_haigyo_2026_05'].mean() * 100
    colors = ['#2ecc71', '#a8e6cf', '#f9ca24', '#f0932b', '#e74c3c']

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(actual_rate.index, actual_rate.values, color=colors, edgecolor='white', linewidth=1.5)

    for bar, val in zip(bars, actual_rate.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f'{val:.1f}%', ha='center', fontsize=12, fontweight='bold')

    baseline = df['new_haigyo_2026_05'].mean() * 100
    ax.axhline(y=baseline, color='#2c3e50', linestyle='--', lw=1.5,
               label=f'全体平均: {baseline:.1f}%')
    ax.legend(fontsize=11)

    ax.set_title('リスクスコア別 実際の廃業率\n(フォワードテスト: 2026年5月検証)', fontsize=14, fontweight='bold')
    ax.set_ylabel('実際の廃業率 (%)', fontsize=12)
    ax.set_xlabel('モデルが事前に予測したリスクレベル', fontsize=12)
    ax.set_ylim(0, max(actual_rate.values) * 1.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'risk_quintile_chart.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: risk_quintile_chart.png")
    print(f"  Quintile closure rates:\n{actual_rate.to_string()}")
    return actual_rate


def step4_loss_simulation(df, baseline_rate):
    """可視化②: 損失回避シミュレーション"""
    print("\nRunning loss avoidance simulation...")

    # 低リスクゾーン（スコア下位40%）のみで出店した場合
    threshold_40 = df['haigyo_prob'].quantile(0.4)
    low_risk = df[df['haigyo_prob'] <= threshold_40]
    model_rate = low_risk['new_haigyo_2026_05'].mean()

    baseline_loss = baseline_rate * N_STORES_SIM * AVG_LOSS_PER_STORE
    model_loss    = model_rate    * N_STORES_SIM * AVG_LOSS_PER_STORE
    saved         = baseline_loss - model_loss
    reduction_pct = (saved / baseline_loss * 100) if baseline_loss > 0 else 0

    # 棒グラフ
    fig, ax = plt.subplots(figsize=(8, 6))
    labels = ['勘で出店\n（モデルなし）', 'モデル活用\n（低リスク店舗のみ）']
    values = [baseline_loss / 1e8, model_loss / 1e8]
    colors = ['#e74c3c', '#2ecc71']

    bars = ax.bar(labels, values, color=colors, width=0.5, edgecolor='white', linewidth=2)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{val:.2f}億円', ha='center', fontsize=13, fontweight='bold')

    ax.set_title(f'損失回避シミュレーション\n({N_STORES_SIM}店舗出店 / 平均損失1,000万円で試算)',
                 fontsize=13, fontweight='bold')
    ax.set_ylabel('想定損失額（億円）', fontsize=12)
    ax.set_ylim(0, max(values) * 1.3)

    diff_text = f'▼ {saved/1e8:.2f}億円削減\n({reduction_pct:.0f}%削減)'
    ax.annotate(diff_text, xy=(1, model_loss / 1e8 + 0.05), fontsize=14,
                color='#27ae60', fontweight='bold', ha='center')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'loss_simulation.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: loss_simulation.png")

    return {
        'baseline_rate': baseline_rate,
        'model_rate': model_rate,
        'baseline_loss_oku': baseline_loss / 1e8,
        'model_loss_oku': model_loss / 1e8,
        'saved_oku': saved / 1e8,
        'reduction_pct': reduction_pct
    }


def step4_heatmap(df):
    """可視化③: 予測vs実際のヒートマップ"""
    print("\nGenerating forward test heatmap...")

    center_lat = df['latitude'].mean()
    center_lon = df['longitude'].mean()

    m = folium.Map(location=[center_lat, center_lon], zoom_start=12,
                   tiles='CartoDB positron')

    # ヒートマップ（予測スコア）
    heat_data = df[['latitude', 'longitude', 'haigyo_prob']].values.tolist()
    HeatMap(
        heat_data,
        radius=20, blur=15,
        gradient={'0.4': 'blue', '0.6': 'yellow', '0.8': 'orange', '1.0': 'red'},
        name='予測リスク',
        min_opacity=0.3
    ).add_to(m)

    # 実際に廃業した店舗を黒マーカーで表示
    new_closed = df[df['new_haigyo_2026_05'] == 1]
    for _, row in new_closed.iterrows():
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=7,
            color='black', fill=True, fill_color='black', fill_opacity=0.85,
            popup=f"実際に廃業<br>予測スコア: {row['haigyo_prob']:.3f}<br>{row.get('name', '')}"
        ).add_to(m)

    folium.LayerControl().add_to(m)
    path = os.path.join(OUTPUT_DIR, 'forward_test_heatmap.html')
    m.save(path)
    print(f"  Saved: forward_test_heatmap.html")


def step5_summary(df, auc_roc, auc_pr, baseline_rate, sim_result):
    """Step 5: 最終サマリーの出力"""

    summary = f"""
====================================================
フォワードテスト サマリー
検証期間：2025年1月学習 → 2026年5月検証
====================================================
対象店舗数      : {len(df):,} 件（2025年1月時点で存続していた店舗）
新規廃業店舗数   : {int(df['new_haigyo_2026_05'].sum()):,} 件
新規廃業率       : {df['new_haigyo_2026_05'].mean():.2%}

【予測精度（フォワードテスト）】
AUC-ROC  : {f"{auc_roc:.3f}" if auc_roc else "計算不可（廃業0件）"}
AUC-PR   : {f"{auc_pr:.3f}" if auc_pr else "計算不可（廃業0件）"}
※ AUC-ROC=0.5はランダム予測、1.0は完全予測

【ビジネスインパクト（{N_STORES_SIM}店舗出店 / 平均損失1,000万円で試算）】
勘で出店した場合の廃業率     : {sim_result['baseline_rate']:.1%}
モデル活用時の廃業率          : {sim_result['model_rate']:.1%}
勘で出店した場合の想定損失    : {sim_result['baseline_loss_oku']:.2f}億円
モデル活用時の想定損失        : {sim_result['model_loss_oku']:.2f}億円
回避できた損失                : {sim_result['saved_oku']:.2f}億円（{sim_result['reduction_pct']:.0f}%削減）

【出力ファイル】
- data/output/forward_test/roc_pr_forward.png        （ROC・PR曲線）
- data/output/forward_test/risk_quintile_chart.png   （リスク分位別廃業率）
- data/output/forward_test/loss_simulation.png       （損失回避シミュレーション）
- data/output/forward_test/forward_test_heatmap.html （地図ヒートマップ）
====================================================
"""
    print(summary)

    with open(os.path.join(OUTPUT_DIR, 'forward_test_summary.txt'), 'w', encoding='utf-8') as f:
        f.write(summary)
    print("  Saved: forward_test_summary.txt")


def main():
    df = load_and_prepare()

    auc_roc, auc_pr, baseline_rate = step3_evaluate(df)

    quintile_rates = step4_viz_quintile(df)

    sim_result = step4_loss_simulation(df, baseline_rate if baseline_rate else df['new_haigyo_2026_05'].mean())

    step4_heatmap(df)

    step5_summary(df, auc_roc, auc_pr, baseline_rate, sim_result)

    print("\nAll forward test steps completed!")


if __name__ == '__main__':
    main()

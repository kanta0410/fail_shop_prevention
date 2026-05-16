"""
Step10: 廃業リスクヒートマップ生成
Foliumを使って廃業確率スコアをマップ上にヒートマップ表示する。
"""
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
import os

SCORES_PATH = 'data/output/closure_scores.csv'
OUTPUT_PATH = 'data/output/closure_risk_heatmap.html'


def main():
    print("Loading closure scores...")
    df = pd.read_csv(SCORES_PATH)
    print(f"  Stores: {len(df)}")

    # ヒートマップ用データ: [lat, lon, weight]
    heat_data = [
        [row['latitude'], row['longitude'], row['closure_probability']]
        for _, row in df.iterrows()
        if pd.notna(row['latitude']) and pd.notna(row['longitude'])
    ]

    # 名古屋市中心部を基点に地図を初期化
    center_lat = df['latitude'].mean()
    center_lon = df['longitude'].mean()

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles='CartoDB positron'
    )

    # ヒートマップレイヤー追加（青→黄→赤）
    HeatMap(
        heat_data,
        radius=20,
        blur=15,
        max_zoom=13,
        gradient={
            '0.0': 'blue',
            '0.4': 'cyan',
            '0.6': 'yellow',
            '0.8': 'orange',
            '1.0': 'red'
        },
        name='廃業リスク'
    ).add_to(m)

    # 廃業店舗（is_closed=1）を赤いマーカーで追加
    closed_df = df[df['is_closed'] == 1]
    for _, row in closed_df.iterrows():
        if pd.notna(row['latitude']):
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=4,
                color='red',
                fill=True,
                fill_color='red',
                fill_opacity=0.7,
                popup=f"{row.get('name', 'N/A')} (スコア: {row['closure_probability']:.3f})"
            ).add_to(m)

    folium.LayerControl().add_to(m)

    m.save(OUTPUT_PATH)
    print(f"Heatmap saved to {OUTPUT_PATH}")


if __name__ == '__main__':
    main()

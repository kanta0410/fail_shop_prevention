# 名古屋飲食店廃業予測プロジェクト (Nagoya Restaurant Closure Prediction)

## プロジェクト概要
名古屋エリアの飲食店の存続・廃業を予測するための機械学習プロジェクトです。
Google Places APIの店舗詳細データとOpenStreetMap (OSM) の時系列データを統合し、LightGBMを用いて廃業リスクの高い店舗を特定します。

## ディレクトリ構造
- `config/`: 設定ファイル (URL, ハイパーパラメータ等)
- `data/`: データ (※Git管理除外)
    - `raw/`: 取得した生のCSV
    - `processed/`: 前処理・特徴量生成済みのCSV
    - `output/`: モデル、予測結果、可視化画像
- `src/`: ソースコード
    - `scraping/`: データ収集スクリプト
    - `preprocessing/`: クレンジング、統合、特徴量生成
    - `models/`: 学習・推論スクリプト
- `docs/`: アイデア、メモ、ドキュメント
- `requirements.txt`: 依存ライブラリ一覧

## セットアップ
```bash
pip install -r requirements.txt
```

## 主なデータセット
- `nagoya_analysis_warehouse.csv`: 解析・学習用の統合データセット

# 名古屋市 人流・イベントデータ収集 調査レポート
**作成日:** 2026年6月9日
**目的:** 名古屋市内の人流データおよびイベントデータの定量的把握に向けた、データソース・API・ツールの網羅的調査

---

## 1. 概要・エグゼクティブサマリー

名古屋市の人流・イベントデータを定量的に取得するためのアプローチは大きく3系統に分類できる。

| 系統 | 代表例 | コスト感 | データ粒度 | リアルタイム性 |
|------|--------|---------|-----------|---------------|
| **通信キャリア系API** | NTTドコモ, KDDI, SoftBank | 有償（高） | 500m〜1kmメッシュ | 日次〜週次 |
| **公的オープンデータ** | 国交省Agoop, RESAS(終了), 名古屋市CKAN | 無償 | 1kmメッシュ | 月次〜過去データのみ |
| **イベント情報API/スクレイピング** | connpass API, nagoya-info, EventBrite | 無償〜低コスト | イベント単位 | リアルタイム |

**クオンツ的観点でのポイント:**
- 人流データの「サンプルバイアス」に注意。通信キャリア系はその社の契約者のみが母集団。NTTドコモ（約8,500万台）が母数最大で統計的偏りが小さい。
- 空間解像度と時間解像度のトレードオフが存在する。1kmメッシュ×時間帯別が現実的な最大精度。
- RESAS-APIは2025年3月24日に**サービス終了済み**。代替は別途手当が必要。

---

## 2. 人流データ（人の動き）

### 2.1 無償・公的オープンデータ系

#### ① 国土交通省 × Agoop「全国の人流オープンデータ」★最重要

| 項目 | 内容 |
|------|------|
| URL | https://www.geospatial.jp/ckan/dataset/mlit-1km-fromto |
| データ期間 | 2019年1月〜2021年12月（月次） |
| 空間粒度 | 1kmメッシュ（全国） |
| 時間粒度 | 全日/平日/休日 × 全日/昼間/夜間 |
| 提供形式 | CSV（G空間情報センターからダウンロード） |
| コスト | **無償・商用利用可** |
| API | CKANベースのREST API（https://www.geospatial.jp/ckan/api/3/）で検索・取得可 |
| 備考 | 株式会社AgoopのスマートフォンGPSデータが元データ。「発地別」も付属 |

**Pythonサンプルコード（CKAN API経由）:**
```python
import requests

CKAN_URL = "https://www.geospatial.jp/ckan"
dataset_id = "mlit-1km-fromto"

# データセット詳細取得
r = requests.get(f"{CKAN_URL}/api/3/action/package_show", params={"id": dataset_id})
resources = r.json()["result"]["resources"]

# 名古屋市域（愛知県）のCSVを抽出してダウンロード
for res in resources:
    print(res["name"], res["url"])
```

#### ② 名古屋市オープンデータカタログ（CKAN API）

| 項目 | 内容 |
|------|------|
| URL | https://odcs.bodik.jp/231002/ / https://data.bodik.jp/dataset/231002_dataset_list |
| データ内容 | 市内各種統計（人流に限らず交通・観光等） |
| API | CKAN API（`https://data.bodik.jp/api/3/`） |
| コスト | **無償** |

**CKAN APIでの名古屋市データ一括検索:**
```python
import requests

BODIK_URL = "https://data.bodik.jp"
r = requests.get(f"{BODIK_URL}/api/3/action/package_search",
    params={"q": "人流 名古屋", "rows": 50})
results = r.json()["result"]["results"]
for d in results:
    print(d["title"], d["name"])
```

#### ③ e-Stat ビッグデータポータル（ポイント型流動人口）

| 項目 | 内容 |
|------|------|
| URL | https://www.e-stat.go.jp/bigdataportal/dataintro?data_type=people_flow |
| データ内容 | スマートフォンGPSベースの流動人口（実証データ） |
| API | e-Stat API（要ユーザー登録、無償） |
| コスト | **無償** |
| 備考 | 国勢調査との接合が可能で人口推計との整合性検証に有用 |

---

### 2.2 有償民間API系（予算があれば最高精度）

#### ① NTTドコモ「モバイル空間統計」

| 項目 | 内容 |
|------|------|
| 提供元 | 株式会社NTTドコモ |
| URL | https://mobaku.jp/ |
| 空間粒度 | 500mメッシュ（一部250m） |
| 属性 | 性別・年代・居住地・就業地 |
| データ母数 | **約8,500万人（最大）** → 統計的偏りが最小 |
| 取得方法 | Web UI / CSV出力 / 要問い合わせ |
| コスト | 有償（価格は個別交渉） |

#### ② KDDI「KDDI Location Data API」★2025年11月より段階提供

| 項目 | 内容 |
|------|------|
| URL | https://biz.kddi.com/topics/2025/news/046/ |
| Phase 1（2025年11月〜） | 指定エリアの滞在データ（性別・年代・時間帯） |
| Phase 2（2026年前半〜） | 居住地・勤務地データ |
| Phase 3（2026年後半〜） | 道路単位の通行データ |
| 取得方法 | REST API（直接システム連携） |
| コスト | 有償（価格は要問い合わせ） |
| 備考 | APIとして直接連携できる点が他社と差別化 |

**Python風のAPIコール想定（仕様公開後に実装）:**
```python
import requests

headers = {"Authorization": "Bearer YOUR_API_KEY"}
payload = {
    "area": {"lat": 35.1815, "lng": 136.9066, "radius_m": 1000},  # 名古屋駅周辺
    "date_from": "2026-06-01",
    "date_to": "2026-06-07",
    "granularity": "hourly"
}
r = requests.post("https://api.kddi-location.jp/v1/stay",
                   json=payload, headers=headers)
```

#### ③ SoftBank「全国うごき統計（Agoop）」

| 項目 | 内容 |
|------|------|
| URL | https://www.data-clew.net/company-list/ugoki.html |
| 提供元 | 株式会社Agoop（SoftBank子会社） |
| 空間粒度 | 基地局ベース（可変） |
| 特徴 | 交通手段・経路の可視化、移動OD情報 |
| コスト | 有償（要問い合わせ） |

#### ④ Yahoo! JAPAN「DS.INSIGHT」

| 項目 | 内容 |
|------|------|
| 提供元 | ヤフー株式会社 |
| 特徴 | アプリ位置情報 + 検索履歴の融合。**興味・関心**も可視化可能 |
| 活用ユースケース | イベント前後の興味変化との相関分析に特に有用 |
| コスト | 有償（要問い合わせ） |

---

### 2.3 廃止・注意事項

| サービス | 状況 |
|---------|------|
| **RESAS-API** | ⚠️ **2025年3月24日にサービス終了**。APIは利用不可。Webアプリ（resas.go.jp）は引き続き閲覧可能 |
| V-RESAS | ⚠️ コロナ禍対応のため更新停止・参照用のみ |

---

## 3. イベントデータ

### 3.1 公式観光・イベント情報サイト（スクレイピング対象候補）

| サイト名 | URL | 取得方法 | コスト |
|---------|-----|---------|--------|
| 名古屋コンシェルジュ（公式） | https://www.nagoya-info.jp/event | スクレイピング（要規約確認） | 無償 |
| 名古屋コンシェルジュ・今日 | https://www.nagoya-info.jp/event/today/ | スクレイピング | 無償 |
| 名古屋市公式（観光文化） | https://www.city.nagoya.jp/kankou/ | スクレイピング | 無償 |
| Aichi Now（愛知県公式） | https://aichinow.pref.aichi.jp/events/ | スクレイピング | 無償 |
| ウォーカープラス | https://www.walkerplus.com/top/ar0623100/nagoya/ | スクレイピング（要確認） | 無償 |
| 名古屋情報通 | https://jouhou.nagoya/category/event/ | スクレイピング | 無償 |
| 久屋大通・栄 | https://www.hisayaodoripark.com/ | スクレイピング | 無償 |
| じゃらん（イベント） | https://www.jalan.net/event/230000/230200/ | スクレイピング（要確認） | 無償 |
| eあいち（愛知県） | https://www.e-aichi.jp/www/genre/1504241916503/index.html | スクレイピング | 無償 |

### 3.2 API提供あり（推奨）

#### ① EventBank API ★スクレイピング不要の推奨選択肢

| 項目 | 内容 |
|------|------|
| URL | https://api.eventbank.jp/ |
| 特徴 | 都道府県・市区町村単位でイベント情報を取得可能。33カテゴリ対応 |
| 取得方法 | REST API |
| コスト | 要確認（基本は有償） |

**PythonサンプルAPI呼び出し:**
```python
import requests

params = {
    "pref_id": "23",    # 愛知県
    "city_id": "231002", # 名古屋市
    "limit": 100,
    "api_key": "YOUR_KEY"
}
r = requests.get("https://api.eventbank.jp/v1/events", params=params)
events = r.json()["events"]
```

#### ② connpass API v2（テック系イベント）

| 項目 | 内容 |
|------|------|
| URL | https://connpass.com/about/api/ |
| 対象 | IT・エンジニア系勉強会・カンファレンス |
| 取得方法 | REST API（無料・固定IP不要） |
| コスト | **無償** |
| 備考 | スクレイピングは規約で禁止。必ずAPIを使用すること |

**PythonサンプルAPI呼び出し:**
```python
import requests

params = {
    "prefecture": "aichi",
    "keyword": "名古屋",
    "count": 100,
    "order": 2,  # 開催日順
}
r = requests.get("https://connpass.com/api/v2/event/", params=params)
events = r.json()["events"]
for e in events:
    print(e["title"], e["started_at"], e["place"])
```

#### ③ Google Places API（会場・POI情報）

| 項目 | 内容 |
|------|------|
| URL | https://developers.google.com/maps/documentation/places/web-service |
| 特徴 | 施設情報（名称・住所・混雑度・評価）の取得 |
| コスト | 有償（月200ドル分の無料クレジットあり） |
| ユースケース | イベント会場の属性情報取得、混雑度データとの接合 |

**人流×イベント接合のPythonスニペット:**
```python
import requests

def get_place_details(place_id: str, api_key: str) -> dict:
    url = "https://maps.googleapis.com/maps/api/place/details/json"
    r = requests.get(url, params={"place_id": place_id, "key": api_key,
                                   "fields": "name,geometry,rating,user_ratings_total"})
    return r.json()["result"]
```

---

## 4. 分析統合フロー（推奨アーキテクチャ）

```
【データ収集層】
┌─────────────────────────────────────────────────────────┐
│  人流データ                     │  イベントデータ          │
│  - 国交省Agoop CSV              │  - connpass API v2      │
│  - KDDI Location Data API       │  - EventBank API        │
│  - NTTドコモ モバイル空間統計   │  - nagoya-info スクレイプ│
└───────────────────┬─────────────────────┬───────────────┘
                    │                     │
【変換・正規化層】  ↓                     ↓
┌─────────────────────────────────────────────────────────┐
│  共通スキーマ: {日時, 緯度, 経度, 推定人数, イベントID} │
│  空間結合: Shapely/GeoPandas（名古屋市境界ポリゴン）    │
│  時間軸整合: pandas + pytz（JST固定）                   │
└─────────────────────────────┬───────────────────────────┘
                              ↓
【分析層】
┌─────────────────────────────────────────────────────────┐
│  - イベント前後の人流変化量（差分分析）                 │
│  - 回帰分析: イベント規模 → 人流増加数                  │
│  - 時系列予測: 人流ベースライン推定                     │
│  - 空間クラスタリング: ホットスポット検出               │
└─────────────────────────────────────────────────────────┘
```

---

## 5. スクレイピングの技術的ガイドライン

### 5.1 推奨ライブラリ

| ライブラリ | 用途 | 備考 |
|-----------|------|------|
| `requests` + `BeautifulSoup4` | 静的HTML解析 | 軽量・高速 |
| `Playwright` / `Selenium` | JS動的レンダリング対応 | SPA・React製サイト |
| `Scrapy` | 大規模スクレイピング | クロールパイプライン管理 |
| `httpx` | 非同期HTTP | asyncio環境向け |

### 5.2 スクレイピング判断フロー

```
サイトにAPIあり？
    YES → APIを使う（connpass, EventBank等）
    NO  → robots.txtを確認
            ├── 禁止されている → スクレイピング不可
            └── 許可/未記載 → 利用規約を確認
                    ├── 禁止 → 問い合わせor断念
                    └── 問題なし → スクレイピング実施
                            └── レート制限に注意（1〜5秒間隔推奨）
```

### 5.3 nagoya-info.jp スクレイピングサンプル

```python
import requests
from bs4 import BeautifulSoup
import time

BASE = "https://www.nagoya-info.jp/event"

def scrape_nagoya_events(page: int = 1) -> list[dict]:
    r = requests.get(BASE, params={"page": page},
                     headers={"User-Agent": "Mozilla/5.0 (research-bot)"})
    soup = BeautifulSoup(r.text, "html.parser")
    events = []
    for item in soup.select(".event-item"):  # セレクタは実際のDOMに合わせて変更
        events.append({
            "title": item.select_one(".event-title").text.strip(),
            "date": item.select_one(".event-date").text.strip(),
            "place": item.select_one(".event-place").text.strip(),
        })
    return events

# ページネーション対応
all_events = []
for pg in range(1, 10):
    all_events.extend(scrape_nagoya_events(pg))
    time.sleep(2)  # サーバー負荷軽減
```

---

## 6. MCPサーバー・既製ツール

現時点（2026年6月）で名古屋市人流データや名古屋イベント情報に**特化したMCPサーバーは確認されていない**。ただし以下の汎用MCPが活用可能：

| MCP/ツール | 活用シーン |
|-----------|-----------|
| `web-search` MCP | イベント情報の動的検索 |
| `filesystem` MCP | CSVダウンロードデータの読み書き |
| `python-executor` MCP | スクレイピングスクリプトの実行 |
| `puppeteer` / `playwright` MCP | JS対応サイトのスクレイピング |
| Google Maps MCP | 場所情報のリアルタイム取得 |

**自前MCPサーバー構築の推奨事項:**
1. `nagoya-event-mcp`: nagoya-info.jp, Aichi Now等を定期ポーリング → イベント情報をJSON提供
2. `jinryu-data-mcp`: Agoop CSVや各社API結果をキャッシュ・統合してメッシュデータをServe

---

## 7. データ品質・バイアスの定量評価

| データソース | サンプル率推定 | 空間誤差 | 時間遅延 | バイアス特性 |
|------------|-------------|---------|---------|------------|
| 国交省Agoop | ~30%（SB契約者比率） | ±50m（GPS） | 月次 | 若年・スマホ利用者に偏重 |
| NTTドコモ | ~40%（国内最大シェア） | ±500m（基地局） | 日次〜週次 | 高齢者比較的多い |
| KDDI API | ~30% | ±50m（GPS） | リアルタイム | 中年層比較的多い |
| connpass | 実数（登録者全数） | イベント会場単位 | リアルタイム | テック系・若年偏重 |
| nagoya-info | 全数（公式掲載のみ） | 会場住所精度 | リアルタイム | 公式認知イベントのみ |

**推定補正の推奨式（拡大係数）:**
```
真の人口 ≈ 観測人数 × (1 / キャリアシェア) × 補正係数k
補正係数k: 地域・時間帯・年齢構成に応じて0.8〜1.3程度
```

---

## 8. 優先実装ロードマップ（コスパ順）

1. **[無償・即時]** 国交省Agoop CSVダウンロード → 名古屋市域フィルタ → ベースライン人流構築
2. **[無償・即時]** connpass API v2 → 名古屋市内テックイベント自動収集パイプライン
3. **[低コスト]** nagoya-info.jp + Aichi Nowスクレイピング → イベントカレンダーDB構築
4. **[要予算]** KDDI Location Data API（Phase2以降）→ リアルタイム人流×イベント接合
5. **[要予算]** NTTドコモ モバイル空間統計 → 精度向上・クロスバリデーション
6. **[開発]** カスタムMCPサーバー構築 → エージェント間でのデータ共有標準化

---

## 9. 参照URL一覧

- 国交省Agoop人流オープンデータ: https://www.geospatial.jp/ckan/dataset/mlit-1km-fromto
- 名古屋市オープンデータ(BODIK): https://data.bodik.jp/dataset/231002_dataset_list
- 名古屋市オープンデータカタログ: https://odcs.bodik.jp/231002/
- e-Stat ビッグデータポータル: https://www.e-stat.go.jp/bigdataportal/dataintro?data_type=people_flow
- KDDI Location Data API: https://biz.kddi.com/topics/2025/news/046/
- EventBank API: https://api.eventbank.jp/
- connpass API v2: https://connpass.com/about/api/
- 名古屋コンシェルジュ（公式）: https://www.nagoya-info.jp/event
- Aichi Now（愛知県公式）: https://aichinow.pref.aichi.jp/events/
- 国交省データプラットフォーム: https://www.mlit-data.jp/
- アーバンデータチャレンジ: https://urbandata-challenge.jp/idc/data-zenkokujinryuopendata
- data-clew（人流データ比較）: https://www.data-clew.net/

---

*本レポートは2026年6月9日時点の調査に基づく。RESASのAPIサービス終了（2025年3月）等、サービス状況が変化している点に留意すること。*

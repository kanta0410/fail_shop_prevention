# エージェント実行プロンプト：名古屋市 イベント×人流 ナレッジ構築パイプライン

**バージョン:** 1.0
**作成日:** 2026-06-09
**用途:** 需要予測システムへの組み込み用ナレッジベース構築
**予算上限:** Google Places API クレジット ¥100,000（≈$670 @ 150円/ドル）

---

## ミッション定義

名古屋市内で今後1ヶ月間（実行日 〜 +30日）に開催されるイベントを網羅的に収集し、各イベントの開催地点・日時・規模を Google Places API (New) で位置情報と紐付けし、「**イベント属性 → 人流増加量の推定**」を可能にするナレッジJSONを生成する。最終出力は需要予測エージェントが直接INPUTとして受け取れる形式とする。

---

## フェーズ構成

```
Phase 1: イベントデータ収集（無償API + スクレイピング）
Phase 2: Google Places API (New) による会場エンリッチメント
Phase 3: ナレッジスキーマへの正規化・統合
Phase 4: 出力ファイル生成
```

---

## Phase 1: イベントデータ収集

### 1-1. connpass API v2（技術系イベント）

**エンドポイント:** `GET https://connpass.com/api/v2/event/`
**コスト:** 無償
**認証:** 不要

```python
import requests
from datetime import datetime, timedelta

today = datetime.now()
ym_from = today.strftime("%Y%m")
ym_to = (today + timedelta(days=30)).strftime("%Y%m")

params = {
    "prefecture": "aichi",
    "keyword_or": "名古屋",
    "ym": f"{ym_from},{ym_to}",   # 当月と翌月
    "count": 100,
    "order": 2,  # 開催日昇順
    "start": 1
}

results = []
while True:
    r = requests.get("https://connpass.com/api/v2/event/", params=params)
    data = r.json()
    results.extend(data["events"])
    if len(data["events"]) < 100:
        break
    params["start"] += 100

# 各イベントから抽出するフィールド
# title, started_at, ended_at, place, address, lat, lon, accepted, limit, event_url
```

**取得対象フィールド:**

| フィールド | 意味 | 需要予測での利用 |
|-----------|------|----------------|
| `started_at` | 開始日時 | 人流増加タイムスタンプ |
| `ended_at` | 終了日時 | 人流減少タイムスタンプ |
| `place` | 会場名 | Places APIでの検索キー |
| `address` | 住所 | ジオコーディング補完 |
| `accepted` | 参加確定者数 | 人流増加量の代理変数 |
| `limit` | 定員 | 上限スケール推定 |

---

### 1-2. nagoya-info.jp スクレイピング（公式観光・地域イベント）

**対象URL:** `https://www.nagoya-info.jp/event/`
**コスト:** 無償
**注意:** robots.txt確認済み、1リクエスト/3秒のレート制限を守ること

```python
import requests
from bs4 import BeautifulSoup
import time
from datetime import datetime, timedelta

BASE = "https://www.nagoya-info.jp"
HEADERS = {"User-Agent": "Mozilla/5.0 (research-purpose; contact: your@email.com)"}

def scrape_page(url: str) -> list[dict]:
    r = requests.get(url, headers=HEADERS, timeout=10)
    soup = BeautifulSoup(r.text, "html.parser")
    events = []

    # ※ 実際のDOMセレクタはサイト構造を確認して調整すること
    for card in soup.select(".eventlist__item, .event-card, article.event"):
        try:
            events.append({
                "source": "nagoya-info",
                "title": card.select_one("[class*='title'], h2, h3").get_text(strip=True),
                "date_str": card.select_one("[class*='date'], time").get_text(strip=True),
                "place": card.select_one("[class*='place'], [class*='venue']").get_text(strip=True) if card.select_one("[class*='place'], [class*='venue']") else "",
                "url": BASE + card.select_one("a")["href"] if card.select_one("a") else ""
            })
        except Exception:
            continue
    return events

# ページネーション
all_events = []
for page in range(1, 15):  # 最大15ページ
    url = f"{BASE}/event/?page={page}"
    batch = scrape_page(url)
    if not batch:
        break
    all_events.extend(batch)
    time.sleep(3)
```

---

### 1-3. Aichi Now スクレイピング（愛知県公式）

**対象URL:** `https://aichinow.pref.aichi.jp/events/`
**コスト:** 無償

```python
# 同様のパターンで実装。以下のカテゴリを絞り込むこと：
CATEGORY_FILTERS = [
    "nagoya",       # 名古屋市のみ
    "festival",     # 祭り
    "music",        # 音楽
    "sports",       # スポーツ
    "exhibition"    # 展覧会
]
```

---

### 1-4. フィルタリング条件（共通）

収集したイベントに対し以下の条件でフィルタリングする：

```python
from datetime import datetime, timedelta

start_date = datetime.now()
end_date = start_date + timedelta(days=30)

def is_in_scope(event: dict) -> bool:
    # 日付範囲チェック
    if event.get("start_dt"):
        if not (start_date <= event["start_dt"] <= end_date):
            return False
    # 名古屋市域チェック（住所に"名古屋"が含まれること）
    address = event.get("address", "") + event.get("place", "")
    if "名古屋" not in address and event.get("source") != "connpass":
        return False
    return True
```

---

## Phase 2: Google Places API (New) による会場エンリッチメント

### 2-1. API概要（New版）

**ベースURL:** `https://places.googleapis.com/v1/`
**認証:** `X-Goog-Api-Key: YOUR_API_KEY` ヘッダー
**重要:** フィールドマスクを必ず指定すること（未指定はエラー、かつコスト増）

**SKU別コスト目安（2025年3月以降）:**

| SKU | 含まれるフィールド | 単価（/1,000リクエスト） | ¥100,000での上限 |
|----|----------------|----------------------|----------------|
| IDs Only | place_id のみ | ~$5 | 約20,000件 |
| Location（推奨） | id, location, displayName, formattedAddress | ~$17 | 約8,800件 |
| Basic | ＋types, rating, businessStatus | ~$32 | 約4,700件 |
| Advanced | ＋openingHours, priceLevel | ~$35 | 約4,300件 |

**→ 本パイプラインでは「Location SKU」を基本とし、必要な場合のみ Basic を使う**

---

### 2-2. Nearby Search (New) で会場候補を取得

```python
import requests
import json

API_KEY = "YOUR_GOOGLE_PLACES_API_KEY"
BASE_URL = "https://places.googleapis.com/v1"

# ナゴヤ中心座標（名古屋駅）を基点に段階的に拡大
NAGOYA_BBOX = {
    "center": {"lat": 35.1706, "lng": 136.8826},
    "radius_m": 20000  # 名古屋市内ほぼカバー
}

def nearby_search_venues(lat: float, lng: float, radius_m: int,
                          included_types: list[str]) -> list[dict]:
    """会場タイプ別にNearby Searchを実行"""
    headers = {
        "X-Goog-Api-Key": API_KEY,
        # Location SKU相当のフィールドのみ指定（コスト最小化）
        "X-Goog-FieldMask": "places.id,places.displayName,places.formattedAddress,places.location,places.types,places.rating,places.userRatingCount"
    }
    body = {
        "locationRestriction": {
            "circle": {
                "center": {"latitude": lat, "longitude": lng},
                "radius": radius_m
            }
        },
        "includedTypes": included_types,
        "maxResultCount": 20,
        "languageCode": "ja"
    }
    r = requests.post(f"{BASE_URL}/places:nearbySearch",
                       json=body, headers=headers, timeout=15)
    if r.status_code != 200:
        raise Exception(f"Places API error: {r.status_code} {r.text}")
    return r.json().get("places", [])

# 対象会場タイプ（イベント開催に関連するPOI）
VENUE_TYPES_MAP = {
    "large_events": ["stadium", "arena", "convention_center", "event_venue"],
    "cultural":     ["museum", "art_gallery", "performing_arts_theater", "concert_hall"],
    "outdoor":      ["park", "amusement_park", "sports_complex"],
    "commercial":   ["shopping_mall", "department_store"],
    "transit":      ["train_station", "subway_station"]
}

# 全タイプを収集
all_venues = {}
for category, types in VENUE_TYPES_MAP.items():
    results = nearby_search_venues(
        lat=NAGOYA_BBOX["center"]["lat"],
        lng=NAGOYA_BBOX["center"]["lng"],
        radius_m=NAGOYA_BBOX["radius_m"],
        included_types=types
    )
    for venue in results:
        place_id = venue["id"]
        all_venues[place_id] = {
            "place_id": place_id,
            "name": venue["displayName"]["text"],
            "address": venue.get("formattedAddress", ""),
            "lat": venue["location"]["latitude"],
            "lng": venue["location"]["longitude"],
            "types": venue.get("types", []),
            "rating": venue.get("rating"),
            "rating_count": venue.get("userRatingCount"),
            "category": category
        }

print(f"会場マスタ構築完了: {len(all_venues)} 件")
```

---

### 2-3. Text Search (New) でイベント×会場を突合

イベント名または会場名から Place ID を特定する。

```python
def text_search_venue(query: str, location_bias_lat: float, location_bias_lng: float) -> dict | None:
    """イベントの会場名からPlace IDを特定"""
    headers = {
        "X-Goog-Api-Key": API_KEY,
        "X-Goog-FieldMask": "places.id,places.displayName,places.formattedAddress,places.location"
    }
    body = {
        "textQuery": query,
        "locationBias": {
            "circle": {
                "center": {"latitude": location_bias_lat, "longitude": location_bias_lng},
                "radius": 5000  # 5km圏内にバイアス
            }
        },
        "maxResultCount": 1,
        "languageCode": "ja"
    }
    r = requests.post(f"{BASE_URL}/places:searchText",
                       json=body, headers=headers, timeout=15)
    places = r.json().get("places", [])
    return places[0] if places else None

# イベントごとに会場のPlace IDを付与
for event in collected_events:
    query = f"{event['place']} 名古屋"
    result = text_search_venue(query, 35.1706, 136.8826)
    if result:
        event["place_id"] = result["id"]
        event["venue_lat"] = result["location"]["latitude"]
        event["venue_lng"] = result["location"]["longitude"]
        event["venue_address"] = result.get("formattedAddress", "")
    import time; time.sleep(0.1)  # レート制限（10 req/sec上限）
```

---

### 2-4. コスト管理（必須チェック）

```python
# リクエスト数トラッキング
class CostTracker:
    BUDGET_JPY = 100_000
    USD_PER_JPY = 1 / 150  # 150円/ドル想定
    SKU_COSTS = {
        "nearby_location": 17.00 / 1000,   # $17/1000リクエスト
        "text_location":   17.00 / 1000,
        "nearby_basic":    32.50 / 1000,
        "text_basic":      32.50 / 1000,
    }

    def __init__(self):
        self.counts = {k: 0 for k in self.SKU_COSTS}

    def add(self, sku: str, n: int = 1):
        self.counts[sku] += n

    def total_usd(self) -> float:
        return sum(self.counts[k] * v for k, v in self.SKU_COSTS.items())

    def total_jpy(self) -> float:
        return self.total_usd() / self.USD_PER_JPY

    def check_budget(self):
        if self.total_jpy() > self.BUDGET_JPY * 0.8:
            raise RuntimeError(f"予算80%到達: ¥{self.total_jpy():,.0f} / ¥{self.BUDGET_JPY:,}")

    def report(self):
        print(f"=== コスト集計 ===")
        for sku, count in self.counts.items():
            cost_jpy = count * self.SKU_COSTS[sku] / self.USD_PER_JPY
            print(f"  {sku}: {count}件 → ¥{cost_jpy:,.0f}")
        print(f"  合計: ¥{self.total_jpy():,.0f} / 予算¥{self.BUDGET_JPY:,}")

tracker = CostTracker()
```

**予算シミュレーション（¥100,000 = $667）:**

| 処理ステップ | リクエスト数目安 | SKU | コスト目安 |
|------------|---------------|-----|-----------|
| 会場マスタ構築（Nearby × 5タイプ） | ~100件 | Location | ~¥170 |
| イベント×会場突合（Text Search） | ~500件 | Location | ~¥850 |
| 合計 | ~600件 | | **~¥1,020（予算の約1%）** |

→ **¥100,000のクレジットで余裕を持って完結する。残クレジットは別用途に活用可。**

---

## Phase 3: ナレッジスキーマへの正規化

### 3-1. イベント × 会場 統合スキーマ

```python
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional

@dataclass
class EventKnowledgeRecord:
    # イベント基本情報
    event_id: str              # "connpass_12345" or "nagoya-info_abc"
    source: str                # "connpass" | "nagoya-info" | "aichi-now"
    title: str
    category: str              # "tech" | "music" | "festival" | "sports" | "cultural" | "other"

    # 時間軸（需要予測の主キー）
    start_dt: str              # ISO8601: "2026-06-15T14:00:00+09:00"
    end_dt: str                # ISO8601
    duration_hours: float      # 所要時間（計算値）
    day_of_week: str           # "Saturday"
    is_holiday: bool           # 祝日フラグ

    # 空間軸（人流との接合キー）
    place_name: str            # 会場名
    place_id: str              # Google Places ID（人流データとの接合に使用）
    lat: float                 # 緯度
    lng: float                 # 経度
    mesh_1km: str              # 1kmメッシュコード（国交省データとの接合）
    address: str

    # 規模推定（人流増加量の代理変数）
    capacity_est: Optional[int]      # 定員・キャパシティ推定
    attendance_est: Optional[int]    # 参加確定数（connpassのみ直接取得可）
    scale_tier: str                  # "small"(<100) | "medium"(100-1000) | "large"(>1000)

    # 人流予測パラメータ（計算値）
    expected_peak_visitors: Optional[int]   # ピーク来場者推定数
    flow_onset_minutes: int                 # イベント開始何分前から人流増加開始するか
    flow_dissipation_minutes: int           # 終了後何分で人流が元に戻るか
    affected_radius_m: int                  # 影響半径（メートル）

    # メタ
    collected_at: str          # データ収集日時
    url: str                   # 元のイベントURL

def compute_flow_params(event: dict) -> dict:
    """イベント属性から人流パラメータを推定するヒューリスティクス"""
    cap = event.get("capacity_est", 100)
    cat = event.get("category", "other")

    # 規模ティア
    if cap < 100:
        tier = "small"
        peak_visitors = int(cap * 0.8)
        radius_m = 300
    elif cap < 1000:
        tier = "medium"
        peak_visitors = int(cap * 0.9)
        radius_m = 800
    else:
        tier = "large"
        peak_visitors = int(cap * 0.95)
        radius_m = 2000

    # カテゴリ別: 人流の立ち上がり・収束時間
    flow_params = {
        "festival":  {"onset": 120, "dissipation": 180},
        "music":     {"onset": 90,  "dissipation": 60},
        "sports":    {"onset": 120, "dissipation": 90},
        "tech":      {"onset": 30,  "dissipation": 30},
        "cultural":  {"onset": 60,  "dissipation": 60},
        "other":     {"onset": 45,  "dissipation": 45},
    }
    params = flow_params.get(cat, flow_params["other"])

    return {
        "scale_tier": tier,
        "expected_peak_visitors": peak_visitors,
        "flow_onset_minutes": params["onset"],
        "flow_dissipation_minutes": params["dissipation"],
        "affected_radius_m": radius_m
    }
```

---

### 3-2. 1kmメッシュコードへの変換（国交省人流データとの接合キー）

```python
def latlon_to_1km_mesh(lat: float, lng: float) -> str:
    """緯度経度 → 1kmメッシュコード（第3次地域区画）"""
    # 1次メッシュ
    p = int(lat * 1.5)
    u = int(lng - 100)
    # 2次メッシュ
    q = int((lat * 1.5 - p) * 8)
    v = int((lng - 100 - u) * 8)
    # 3次メッシュ（1km）
    r = int(((lat * 1.5 - p) * 8 - q) * 10)
    w = int(((lng - 100 - u) * 8 - v) * 10)
    return f"{p:02d}{u:02d}{q}{v}{r}{w}"

# 使用例
mesh = latlon_to_1km_mesh(35.1706, 136.8826)  # 名古屋駅 → "53236980"
```

---

## Phase 4: 出力ファイル生成

### 4-1. メインナレッジファイル（需要予測エージェントへのINPUT）

**出力先:** `output/nagoya_event_knowledge_YYYYMM.json`

```json
{
  "meta": {
    "generated_at": "2026-06-09T12:00:00+09:00",
    "coverage_start": "2026-06-09",
    "coverage_end": "2026-07-09",
    "total_events": 312,
    "sources": {
      "connpass": 87,
      "nagoya-info": 143,
      "aichi-now": 82
    },
    "schema_version": "1.0"
  },
  "events": [
    {
      "event_id": "connpass_12345",
      "source": "connpass",
      "title": "名古屋Python勉強会 #89",
      "category": "tech",
      "start_dt": "2026-06-15T19:00:00+09:00",
      "end_dt": "2026-06-15T21:00:00+09:00",
      "duration_hours": 2.0,
      "day_of_week": "Monday",
      "is_holiday": false,
      "place_name": "名古屋市中小企業振興会館（吹上ホール）",
      "place_id": "ChIJxxxxxxxxxxxxxxxxxx",
      "lat": 35.1560,
      "lng": 136.9200,
      "mesh_1km": "53236959",
      "address": "愛知県名古屋市千種区吹上2丁目6-3",
      "capacity_est": 80,
      "attendance_est": 62,
      "scale_tier": "small",
      "expected_peak_visitors": 50,
      "flow_onset_minutes": 30,
      "flow_dissipation_minutes": 30,
      "affected_radius_m": 300,
      "collected_at": "2026-06-09T10:00:00+09:00",
      "url": "https://connpass.com/event/12345/"
    }
  ]
}
```

---

### 4-2. 需要予測エージェント向けクエリI/F

```python
def query_events_by_location_time(
    lat: float, lng: float, radius_m: int,
    dt_from: str, dt_to: str,
    knowledge: list[dict]
) -> list[dict]:
    """
    指定エリア・時間帯に影響するイベントを検索。
    需要予測エージェントはこの関数を呼び出して特徴量を取得する。
    """
    from math import radians, sin, cos, sqrt, atan2

    def haversine(lat1, lng1, lat2, lng2) -> float:
        R = 6371000
        φ1, φ2 = radians(lat1), radians(lat2)
        Δφ = radians(lat2 - lat1)
        Δλ = radians(lng2 - lng1)
        a = sin(Δφ/2)**2 + cos(φ1)*cos(φ2)*sin(Δλ/2)**2
        return R * 2 * atan2(sqrt(a), sqrt(1-a))

    from datetime import datetime
    dt_from_obj = datetime.fromisoformat(dt_from)
    dt_to_obj   = datetime.fromisoformat(dt_to)

    matching = []
    for ev in knowledge:
        # 空間フィルタ
        dist = haversine(lat, lng, ev["lat"], ev["lng"])
        if dist > radius_m + ev["affected_radius_m"]:
            continue
        # 時間フィルタ（イベント影響時間帯との重複チェック）
        ev_start = datetime.fromisoformat(ev["start_dt"]) - \
                   __import__('datetime').timedelta(minutes=ev["flow_onset_minutes"])
        ev_end   = datetime.fromisoformat(ev["end_dt"]) + \
                   __import__('datetime').timedelta(minutes=ev["flow_dissipation_minutes"])
        if ev_end < dt_from_obj or ev_start > dt_to_obj:
            continue
        matching.append({**ev, "distance_m": int(dist)})

    return sorted(matching, key=lambda x: x["start_dt"])
```

---

## 実行手順（エージェントへの指示）

```
STEP 1: 環境変数の確認
  - GOOGLE_PLACES_API_KEY が設定されていること
  - 予算トラッカーを初期化すること

STEP 2: Phase 1を実行
  - connpass API → aichi地域 × 向こう30日のイベント収集
  - nagoya-info.jp スクレイピング（1〜15ページ、3秒間隔）
  - Aichi Now スクレイピング（名古屋市フィルタ）
  - 重複排除: title + start_dt が一致するものを統合

STEP 3: Phase 2を実行
  - 会場マスタ構築（Nearby Search × 5カテゴリ）
  - イベント × 会場突合（Text Search、会場名 + "名古屋"）
  - 緯度経度未取得のイベントはスキップ（warningログ出力）
  - CostTracker.check_budget() を各リクエスト後に呼び出すこと

STEP 4: Phase 3を実行
  - compute_flow_params() で各イベントの人流パラメータを計算
  - latlon_to_1km_mesh() でメッシュコードを付与

STEP 5: Phase 4を実行
  - output/nagoya_event_knowledge_YYYYMM.json を出力
  - tracker.report() でコスト集計を出力
  - 収集件数・カバレッジサマリをログに出力

STEP 6: バリデーション
  - place_id が付与されたイベントの割合を確認（目標: 70%以上）
  - start_dt が実行日〜+30日に収まっているか確認
  - lat/lng が名古屋市域（34.9〜35.4, 136.6〜137.1）内か確認
```

---

## 制約・注意事項

| 制約 | 詳細 |
|------|------|
| Google Places API 予算上限 | ¥100,000。CostTracker で80%到達時に自動停止 |
| スクレイピング レート | nagoya-info: 1req/3秒, Aichi Now: 1req/2秒 |
| connpass API レート | 公式制限なし、念のため1req/秒以下推奨 |
| データ保持期間 | イベント終了後も90日間は保持（人流との事後照合のため） |
| 個人情報 | connpassの参加者リストは収集しないこと。集計値（人数）のみ |
| 法的 | 各サイトの利用規約を遵守。著作権のあるイベント詳細テキストは要約のみ保存 |

---

## 期待される出力サマリ

```
=== 収集結果 ===
connpass:       87件
nagoya-info:   143件
aichi-now:      82件
重複除去後:    245件

=== エンリッチメント ===
place_id付与済: 189件（77.1%）
1kmメッシュ付与: 245件（100%）

=== コスト集計 ===
Nearby Search (Location): 100件 → ¥170
Text Search (Location):   245件 → ¥416
合計: ¥586 / 予算¥100,000（0.6%使用）

=== 出力 ===
output/nagoya_event_knowledge_202606.json
```

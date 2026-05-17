import requests
import time

HEADERS = {'User-Agent': 'Mozilla/5.0 (ForwardTest/1.0; academic research)', 'Accept': 'application/json'}
AMENITY_FILTER = 'restaurant|cafe|bar|fast_food|food_court|pub|izakaya|meal_takeaway'
ENDPOINTS = [
    'https://kumi.systems/overpass/api/interpreter',
    'https://overpass.kumi.systems/api/interpreter',
    'https://overpass-api.de/api/interpreter',
]

def test_query(lat, lon, radius=30):
    q = f'[out:json][timeout:20];(node["amenity"~"{AMENITY_FILTER}"](around:{radius},{lat},{lon});way["amenity"~"{AMENITY_FILTER}"](around:{radius},{lat},{lon}););out count;'
    for ep in ENDPOINTS:
        try:
            resp = requests.post(ep, data={'data': q}, headers=HEADERS, timeout=25)
            if resp.status_code == 200:
                data = resp.json()
                elems = data.get('elements', [])
                if elems and 'tags' in elems[0]:
                    return int(elems[0]['tags'].get('total', 0))
                return len(elems)
            print(f'  [{ep}] HTTP {resp.status_code}')
        except Exception as e:
            print(f'  [{ep}] Error: {e}')
    return None

test_cases = [
    (35.1706, 136.8816, '名古屋駅周辺'),
    (35.0319, 136.9140, 'サンプル店舗1'),
    (35.0313, 136.9145, 'サンプル店舗2'),
]

for lat, lon, desc in test_cases:
    result = test_query(lat, lon)
    print(f'{desc}: count={result}')
    time.sleep(1.5)

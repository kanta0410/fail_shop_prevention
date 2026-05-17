import requests
import math
import time

NOMINATIM_URL = 'https://nominatim.openstreetmap.org/reverse'
HEADERS = {
    'User-Agent': 'FailShopPrevention/1.0 (research; nagoya)',
    'Accept-Language': 'ja',
}

test_cases = [
    (35.1706, 136.8816, '名古屋駅付近'),
    (35.1692, 136.9084, '栄付近'),
    (35.032164, 136.916, '東海市の店舗サンプル'),
]

for lat, lon, desc in test_cases:
    params = {'lat': lat, 'lon': lon, 'zoom': 18, 'format': 'jsonv2', 'addressdetails': 0}
    resp = requests.get(NOMINATIM_URL, params=params, headers=HEADERS, timeout=15)
    print(f"{desc}: HTTP {resp.status_code}")
    if resp.status_code == 200:
        d = resp.json()
        print(f"  class={d.get('class')}, type={d.get('type')}, name={d.get('name')}")
        print(f"  lat={d.get('lat')}, lon={d.get('lon')}")
        osm_lat, osm_lon = float(d.get('lat', 0)), float(d.get('lon', 0))
        R = 6371000
        phi1, phi2 = math.radians(lat), math.radians(osm_lat)
        dphi = math.radians(osm_lat - lat)
        dlam = math.radians(osm_lon - lon)
        a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
        dist = R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        print(f"  dist={dist:.1f}m")
    print()
    time.sleep(1.1)

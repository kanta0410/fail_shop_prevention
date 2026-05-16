import requests
import json

url = "https://overpass.kumi.systems/api/interpreter"
# 千種区のBBOX
BBOX = "35.145,136.920,35.195,136.995"
query = f"""
[out:json][timeout:300][date:"2025-01-01T00:00:00Z"];
(
  node["amenity"~"restaurant|cafe|bar|fast_food"]({BBOX});
  way["amenity"~"restaurant|cafe|bar|fast_food"]({BBOX});
);
out center;
"""

print(f"Requesting Chikusa-ku historical data from {url}...")
try:
    response = requests.post(url, data={'data': query}, timeout=310)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Count: {len(data.get('elements', []))}")
        with open("chikusa_osm_2025.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    else:
        print(response.text[:1000])
except Exception as e:
    print(f"Error: {e}")

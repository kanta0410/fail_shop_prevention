import requests
import json
import urllib3

# Disable SSL warnings for the Japanese endpoint which has a mismatching cert
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

url = "https://overpass.osm.jp/interpreter"
query = """
[out:json][timeout:50][date:"2025-01-01T00:00:00Z"];
(
  node["amenity"~"restaurant|cafe|bar|fast_food"](35.145,136.920,35.155,136.930);
  way["amenity"~"restaurant|cafe|bar|fast_food"](35.145,136.920,35.155,136.930);
);
out center;
"""

print(f"Testing {url} with verify=False...")
try:
    response = requests.get(url, params={'data': query}, verify=False, timeout=60)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Count: {len(data.get('elements', []))}")
    else:
        print(response.text[:500])
except Exception as e:
    print(f"Error: {e}")

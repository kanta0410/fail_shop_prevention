import requests
import json

url = "https://overpass.osm.ch/api/interpreter"
query = """
[out:json][timeout:50][date:"2025-01-01T00:00:00Z"];
(
  node["amenity"~"restaurant|cafe|bar|fast_food"](35.145,136.920,35.155,136.930);
  way["amenity"~"restaurant|cafe|bar|fast_food"](35.145,136.920,35.155,136.930);
);
out center;
"""

response = requests.get(url, params={'data': query})
print(f"Status: {response.status_code}")
if response.status_code == 200:
    data = response.json()
    print(f"Count: {len(data.get('elements', []))}")
else:
    print(response.text[:500])

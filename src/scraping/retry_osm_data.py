import requests
import pandas as pd
import time
import os

OVERPASS_URL = "https://overpass.kumi.systems/api/interpreter"
S, W, N, E = 35.05, 136.75, 35.25, 137.05

FAILED_CATEGORIES = {
    "convenience": 'node["shop"~"convenience|supermarket"](S,W,N,E); way["shop"~"convenience|supermarket"](S,W,N,E);',
    "hospitals": 'node["amenity"~"hospital|clinic|doctors"](S,W,N,E); way["amenity"~"hospital|clinic|doctors"](S,W,N,E);',
    "parks": 'node["leisure"="park"](S,W,N,E); way["leisure"="park"](S,W,N,E);',
    "malls": 'node["shop"="mall"](S,W,N,E); node["building"="mall"](S,W,N,E); way["shop"="mall"](S,W,N,E); way["building"="mall"](S,W,N,E);'
}

def fetch_category(name, query_body, output_dir):
    print(f"Retrying {name}...")
    full_query = f"""
    [out:json][timeout:300];
    (
      {query_body.replace("(S,W,N,E)", f"({S},{W},{N},{E})")}
    );
    out center;
    """
    
    try:
        response = requests.post(OVERPASS_URL, data={'data': full_query}, timeout=300)
        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            return
            
        result = response.json()
        elements = result.get('elements', [])
        data = []
        for el in elements:
            tags = el.get('tags', {})
            lat = el.get('lat') or (el.get('center', {}).get('lat'))
            lon = el.get('lon') or (el.get('center', {}).get('lon'))
            if lat and lon:
                data.append({
                    "osm_id": el.get('id'), "lat": lat, "lon": lon,
                    "name": tags.get("name", ""), "amenity": tags.get("amenity", ""),
                    "shop": tags.get("shop", ""), "highway": tags.get("highway", ""),
                    "office": tags.get("office", ""), "building": tags.get("building", ""),
                    "leisure": tags.get("leisure", ""), "capacity": tags.get("capacity", "")
                })
        
        df = pd.DataFrame(data)
        file_path = os.path.join(output_dir, f"osm_{name}.csv")
        df.to_csv(file_path, index=False, encoding='utf-8')
        print(f"Saved {len(df)} items to {file_path}")
        
    except Exception as e:
        print(f"Error: {e}")

def main():
    output_dir = "data/raw/osm"
    for name, query_body in FAILED_CATEGORIES.items():
        fetch_category(name, query_body, output_dir)
        time.sleep(10)

if __name__ == "__main__":
    main()

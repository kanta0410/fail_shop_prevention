import requests
import pandas as pd
import os

OVERPASS_URL = "https://overpass.kumi.systems/api/interpreter"
S, W, N, E = 35.05, 136.75, 35.25, 137.05

def fetch_road_nodes():
    print("Fetching road nodes (this may take a while)...")
    # Fetch all nodes belonging to main highways
    query = f"""
    [out:json][timeout:300];
    way["highway"~"primary|secondary|tertiary|trunk"]({S},{W},{N},{E});
    node(w);
    out;
    """
    
    try:
        response = requests.post(OVERPASS_URL, data={'data': query}, timeout=300)
        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            return
            
        result = response.json()
        elements = result.get('elements', [])
        
        data = []
        for el in elements:
            if el.get('type') == 'node':
                data.append({
                    "osm_id": el.get('id'),
                    "lat": el.get('lat'),
                    "lon": el.get('lon')
                })
        
        df = pd.DataFrame(data)
        output_path = "data/raw/osm/osm_roads_nodes.csv"
        df.to_csv(output_path, index=False)
        print(f"Saved {len(df)} road nodes to {output_path}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    fetch_road_nodes()

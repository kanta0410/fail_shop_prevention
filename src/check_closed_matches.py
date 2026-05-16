import pandas as pd
from math import radians, cos, sin, asin, sqrt

def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 
    return c * r * 1000

df_google = pd.read_csv("nagoya_all_massive_raw.csv")
df_osm = pd.read_csv("nagoya_osm_labels.csv")

closed_osm = df_osm[df_osm['target_closed_osm'] == 1]
print(f"Closed OSM stores: {len(closed_osm)}")

matches = 0
for _, osm_row in closed_osm.iterrows():
    lat_m = 0.001
    lng_m = 0.001
    candidates = df_google[
        (df_google['lat'] >= osm_row['lat'] - lat_m) & (df_google['lat'] <= osm_row['lat'] + lat_m) &
        (df_google['lng'] >= osm_row['lng'] - lng_m) & (df_google['lng'] <= osm_row['lng'] + lng_m)
    ]
    
    for _, g_row in candidates.iterrows():
        dist = haversine(osm_row['lng'], osm_row['lat'], g_row['lng'], g_row['lat'])
        if dist < 50:
            matches += 1
            break

print(f"Closed OSM stores found in Google data: {matches}")

import requests

endpoints = [
    'https://overpass.kumi.systems/api/interpreter',
    'https://maps.mail.ru/osm/tools/overpass/api/interpreter',
    'https://overpass-api.de/api/interpreter',
]
query = '[out:json][timeout:10];node["amenity"="restaurant"](around:100,35.1706,136.8816);out count;'
headers = {'User-Agent': 'Mozilla/5.0 (forward-test/1.0; research)'}

for ep in endpoints:
    try:
        r = requests.post(ep, data={'data': query}, headers=headers, timeout=60)
        print(f'{ep} => Status: {r.status_code}')
        if r.status_code == 200:
            print('  Body:', r.text[:200])
            break
    except Exception as e:
        print(f'{ep} => Error: {e}')

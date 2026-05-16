import os
import json
import time
import requests
import pandas as pd

API_KEY = ""
URL = "https://places.googleapis.com/v1/places:searchText"

def fetch_places(query, max_pages=50):
    headers = {
        "X-Goog-Api-Key": API_KEY,
        "X-Goog-FieldMask": "places.id,places.displayName,places.businessStatus,places.rating,places.userRatingCount,places.priceLevel,places.formattedAddress,places.location,nextPageToken",
        "Content-Type": "application/json"
    }

    places = []
    page_token = ""
    page = 0

    while page < max_pages:
        payload = {
            "textQuery": query,
            "pageSize": 20
        }
        if page_token:
            payload["pageToken"] = page_token
            
        print(f"Fetching page {page + 1}...")
        response = requests.post(URL, headers=headers, json=payload)
        
        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            print(response.text)
            break
            
        data = response.json()
        
        if "places" in data:
            places.extend(data["places"])
            print(f"Fetched {len(data['places'])} places. Total: {len(places)}")
        else:
            print("No places found in this response.")
            break
            
        page_token = data.get("nextPageToken")
        if not page_token:
            print("No more pages available.")
            break
            
        page += 1
        time.sleep(2) # To avoid rate limiting
        
    return places

def main():
    query = "名古屋市千種区 飲食店"
    print(f"Searching for: {query}")
    places_data = fetch_places(query, max_pages=100) # max 2000 results
    
    if not places_data:
        print("No data fetched.")
        return
        
    # Flatten the data
    flat_data = []
    for p in places_data:
        flat_data.append({
            "id": p.get("id"),
            "name": p.get("displayName", {}).get("text"),
            "business_status": p.get("businessStatus"),
            "rating": p.get("rating"),
            "user_rating_count": p.get("userRatingCount"),
            "price_level": p.get("priceLevel"),
            "address": p.get("formattedAddress"),
            "lat": p.get("location", {}).get("latitude"),
            "lng": p.get("location", {}).get("longitude")
        })
        
    df = pd.DataFrame(flat_data)
    
    # Save to CSV
    output_file = "chikusa_restaurants_raw.csv"
    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"Saved {len(df)} records to {output_file}")
    
    # Basic EDA print
    print("\n--- Basic Information ---")
    print(df.info())
    print("\n--- Business Status Value Counts ---")
    print(df["business_status"].value_counts(dropna=False))
    
if __name__ == "__main__":
    main()

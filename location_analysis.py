"""
location_analysis.py
Geographical analysis of restaurant distribution, ratings,
cuisines and price ranges across cities/localities.
Generates an interactive HTML map + summary CSVs in /outputs.
"""
import json
import pandas as pd

import os
_OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
os.makedirs(_OUT_DIR, exist_ok=True)
from preprocess import load_data, clean_data


def city_summary(df):
    summary = df.groupby('City').agg(
        restaurant_count=('Restaurant Name', 'count'),
        avg_rating=('Aggregate rating', 'mean'),
        avg_cost_for_two=('Average Cost for two', 'mean'),
        avg_price_range=('Price range', 'mean'),
        avg_votes=('Votes', 'mean')
    ).sort_values('restaurant_count', ascending=False)
    return summary.round(2)


def top_cuisines_by_city(df, city, top_n=5):
    subset = df[df['City'] == city]
    return subset['Primary Cuisine'].value_counts().head(top_n)


def build_map(df, sample_size=1500, out_path=os.path.join(_OUT_DIR, "restaurant_map.html")):
    """Self-contained Leaflet.js map (no extra Python geo-libraries required)."""
    if len(df) > sample_size:
        df = df.sample(sample_size, random_state=42)

    df = df.dropna(subset=['Latitude', 'Longitude'])
    points = [
        {
            "lat": row["Latitude"], "lng": row["Longitude"],
            "name": row["Restaurant Name"], "city": row["City"],
            "cuisine": row["Cuisines"], "rating": row["Aggregate rating"]
        }
        for _, row in df.iterrows()
    ]
    center_lat, center_lng = df['Latitude'].mean(), df['Longitude'].mean()

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>RestaurantIQ - Location Map</title>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.css" />
  <script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.js"></script>
  <style>html,body,#map{{height:100%;margin:0;}}</style>
</head>
<body>
<div id="map"></div>
<script>
  const points = {json.dumps(points)};
  const map = L.map('map').setView([{center_lat}, {center_lng}], 2);
  L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
    attribution: '&copy; OpenStreetMap contributors'
  }}).addTo(map);
  points.forEach(p => {{
    L.circleMarker([p.lat, p.lng], {{radius: 4, color: '#6c5ce7', fillOpacity: 0.7}})
      .addTo(map)
      .bindPopup(`<b>${{p.name}}</b><br>${{p.city}}<br>Rating: ${{p.rating}}<br>${{p.cuisine}}`);
  }});
</script>
</body>
</html>"""

    if out_path is None:
        out_path = os.path.join(_OUT_DIR, "restaurant_map.html")
    with open(out_path, "w") as f:
        f.write(html)
    print(f"Map saved to {out_path}")


if __name__ == "__main__":
    df = clean_data(load_data())

    summary = city_summary(df)
    print("Top 10 cities by restaurant count:")
    print(summary.head(10).to_string())
    summary.to_csv(os.path.join(_OUT_DIR, "city_summary.csv"))

    top_city = summary.index[0]
    print(f"\nTop cuisines in {top_city}:")
    print(top_cuisines_by_city(df, top_city))

    build_map(df)

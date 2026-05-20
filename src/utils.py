import math
import requests

def compute_heading(lat1, lon1, lat2, lon2):

    dlon = math.radians(lon2 - lon1)

    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)

    x = math.sin(dlon) * math.cos(lat2)

    y = (
        math.cos(lat1) * math.sin(lat2)
        - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    )

    brng = math.degrees(math.atan2(x, y))

    return (brng + 360) % 360


def get_metadata(lat, lon, api_key):

    url = (
        "https://maps.googleapis.com/maps/api/streetview/metadata"
        f"?location={lat},{lon}"
        f"&key={api_key}"
    )

    response = requests.get(url)

    if response.status_code != 200:

        print(f"Metadata request failed: {lat}, {lon}")

        return None

    return response.json()
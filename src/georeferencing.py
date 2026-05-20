# georeferencing.py

import os

import pandas as pd
import geopandas as gpd
import osmnx as ox
from shapely.geometry import Point

from . import config


def georeferencing(config):

    op = config.OUTPUT_DIR.format(config.LOCATIONS['building_lat'],config.LOCATIONS['building_lon'])

    os.makedirs(op, exist_ok=True)

    metadata_path = os.path.join(
        op,
        f"meta_lat_{config.LOCATIONS['building_lat']}_lng_{config.LOCATIONS['building_lon']}_p_{config.PITCH}_fov_{config.FOV}.csv"
    )

    output_path = os.path.join(
        op,
        f"fp_lat_{config.LOCATIONS['building_lat']}_lng_{config.LOCATIONS['building_lon']}_p_{config.PITCH}_fov_{config.FOV}.geojson"
    )

    metadata = pd.read_csv(metadata_path)

    results = []

    for _, row in metadata.iterrows():

        lat = row["building_lat"]
        lon = row["building_lon"]

        point = gpd.GeoSeries(
            [Point(lon, lat)],
            crs="EPSG:4326"
        ).to_crs("EPSG:3857").iloc[0]

        buildings = ox.features_from_point(
            (lat, lon),
            tags={"building": True},
            dist=1000
        ).reset_index()

        buildings = buildings.to_crs("EPSG:3857")

        buildings["distance"] = (
            buildings.geometry.centroid.distance(point)
        )

        nearest = buildings.sort_values("distance").iloc[0]

        results.append({
            "image": row["image_name"],
            "building_id": nearest.name,
            "distance_m": nearest["distance"],
            "geometry": nearest.geometry
        })

    gdf = gpd.GeoDataFrame(
        results,
        geometry="geometry",
        crs="EPSG:3857"
    ).to_crs("EPSG:4326")

    gdf.to_file(output_path, driver="GeoJSON")

    print(f"Saved: {output_path}")

    return gdf


if __name__ == "__main__":
    georeferencing(config)
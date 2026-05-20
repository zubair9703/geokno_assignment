import os
import requests
import pandas as pd

from pathlib import Path

from . import config

from .utils import get_metadata, compute_heading


def download_building_view(
    building_lat,
    building_lon,
    camera_lat,
    camera_lon,
    save_path,
    api_key,
    fov,
    pitch,
    size,
):

    # -----------------------------------------------------
    # Compute heading
    # -----------------------------------------------------

    heading = compute_heading(
            camera_lat,
            camera_lon,
            building_lat,
            building_lon
        )
        
    # -----------------------------------------------------
    # Build URL
    # -----------------------------------------------------

    url = (
        "https://maps.googleapis.com/maps/api/streetview"
        f"?size={size}"
        f"&location={camera_lat},{camera_lon}"
        f"&heading={heading}"
        f"&pitch={pitch}"
        f"&fov={fov}"
        f"&key={api_key}"
    )

    response = requests.get(url)

    if response.status_code == 200:

        with open(save_path, "wb") as f:
            f.write(response.content)

        print(f"Saved: {save_path}")

        return heading, response.content

    else:

        print(f"Failed: {save_path}")

        return None, None


# =========================================================
# PROCESS FUNCTION
# =========================================================

def process_location(
    building_lat,
    building_lon,
    output_dir,
    api_key,
    fov,
    pitch,
    size,
):
    op = output_dir.format(building_lat,building_lon)

    os.makedirs(op, exist_ok=True)
    # -----------------------------------------------------
    # Get metadata
    # -----------------------------------------------------

    metadata = get_metadata(
        building_lat,
        building_lon,
        api_key,
    )

    if metadata is None:
        return None

    if metadata.get("status") != "OK":

        print(
            f"No Street View available for "
            f"{building_lat}, {building_lon}"
        )

        return None

    # -----------------------------------------------------
    # Camera coordinates from metadata
    # -----------------------------------------------------

    camera_lat = metadata["location"]["lat"]
    camera_lon = metadata["location"]["lng"]

    pano_id = metadata.get("pano_id")
    capture_date = metadata.get("date")

    # -----------------------------------------------------
    # Image name
    # -----------------------------------------------------

    image_name = (
        f"lat_{building_lat}_lng_{building_lon}_p_{pitch}_fov_{fov}.jpg"
    )

    # image_name = image_name.replace(".", "_")

    image_path = os.path.join(
        op,
        image_name
    )

    # -----------------------------------------------------
    # Download image
    # -----------------------------------------------------

    heading, content = download_building_view(
        building_lat=building_lat,
        building_lon=building_lon,
        camera_lat=camera_lat,
        camera_lon=camera_lon,
        save_path=image_path,
        api_key=api_key,
        fov=fov,
        pitch=pitch,
        size=size,
    )
    
    meta = {

        "image_name": image_name,
        "image_path": image_path,

        "building_lat": building_lat,
        "building_lon": building_lon,

        "camera_lat": camera_lat,
        "camera_lon": camera_lon,

        "heading": heading,

        "pitch": pitch,
        "fov": fov,

        "capture_date": capture_date,
        "pano_id": pano_id,
    }

    df = pd.DataFrame([meta])

    meta_name = f"meta_lat_{building_lat}_lng_{building_lon}_p_{pitch}_fov_{fov}.csv"

    meta_path = os.path.join(
        op,
        meta_name
    )

    df.to_csv(
        meta_path,
        index=False
    )

    return df, content


# =========================================================
# MAIN PIPELINE
# =========================================================

def main():

    Path(config.OUTPUT_DIR).mkdir(
        parents=True,
        exist_ok=True
    )

    # for loc in config.LOCATIONS:

    result = process_location(

        building_lat=config.LOCATIONS["building_lat"],
        building_lon=config.LOCATIONS["building_lon"],

        output_dir=config.OUTPUT_DIR,

        api_key=config.API_KEY,

        fov=config.FOV,
        pitch=config.PITCH,
        size=config.SIZE,
    )


# =========================================================
# ENTRYPOINT
# =========================================================

if __name__ == "__main__":

    main()


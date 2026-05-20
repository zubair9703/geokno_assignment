from pathlib import Path
import base64
import mimetypes

import folium
from folium import IFrame
import pandas as pd

from .scrape_street_view import process_location
from .georeferencing import georeferencing
from .windows_detection import WindowsSegmentationPipeline
from .sign_board_classification import SignboardOCRPipeline

from . import config


class FloorNumberEstimationPipeline:

    def __init__(self, config):

        self.config = config

        self.window_pipeline = WindowsSegmentationPipeline(
            model_path=config.MODEL_PATH,
            conf=config.CONF,
            half=config.HALF,
            save=config.SAVE,
            output_dir=config.WINDOWS_DET_OP,
        )

        self.signboard_pipeline = SignboardOCRPipeline(
            model_path=config.MODEL_PATH,
            output_dir=config.CLASSIFICATION_OP,
            gpu=True,
        )

    def fetch_street_view(self):

        meta, data = process_location(
            self.config.LOCATIONS["building_lat"],
            self.config.LOCATIONS["building_lon"],
            self.config.OUTPUT_DIR,
            self.config.API_KEY,
            self.config.FOV,
            self.config.PITCH,
            self.config.SIZE,
        )

        return meta, data

    def generate_building_footprint(self):

        fp = georeferencing(self.config)

        return fp

    def get_image_path(self):

        image_path = str(
            Path(
                self.config.OUTPUT_DIR.format(
                    self.config.LOCATIONS["building_lat"],
                    self.config.LOCATIONS["building_lon"],
                )
            )
            / f"lat_{config.LOCATIONS['building_lat']}_lng_{config.LOCATIONS['building_lon']}_p_{config.PITCH}_fov_{config.FOV}.jpg"
        )

        return image_path

    def detect_windows(self):

        image_path = self.get_image_path()

        results, floor_count = self.window_pipeline.run(
            image_path,concepts=[
                "a window",
            ],
            save_visualization=True,
            save_masks=True )

        return results, floor_count, image_path

    def classify_signboard(self, image_path):

        result = self.signboard_pipeline.run(
            image_path
        )

        return result

    @staticmethod
    def encode_image(image_path):

        mime_type, _ = mimetypes.guess_type(
            image_path
        )

        if mime_type is None:
            mime_type = "image/jpeg"

        with open(image_path, "rb") as f:

            encoded = base64.b64encode(
                f.read()
            ).decode("utf-8")

        return (
            f"data:{mime_type};base64,{encoded}"
        )

    def create_map(
        self,
        gdf,
        image_column="image_path",
        output_path = 'map.html',
        polygon_color="blue",
        popup_width=400,
    ):
        """
        Add building polygons with:
        - image thumbnail popup
        - all GeoDataFrame attributes

        Parameters
        ----------
        fmap : folium.Map

        gdf : GeoDataFrame

        image_column : str
            Column containing image file paths
        """
        center = [
                gdf.geometry.centroid.y.mean(),
                gdf.geometry.centroid.x.mean(),
            ]

        fmap = folium.Map(
            location=center,
            zoom_start=19,
            # tiles="cartodbpositron",
        )

        folium.TileLayer(
            tiles="https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
            attr="Google",
            name="Google Satellite Hybrid",
            overlay=False,
            control=True,
        ).add_to(fmap)
        folium.LayerControl().add_to(fmap)

        for idx, row in gdf.iterrows():

            geom = row.geometry

            if geom.geom_type != "Polygon":
                continue

            # ==========================================================
            # IMAGE
            # ==========================================================
            image_html = ""

            if image_column in row and pd.notnull(row[image_column]):

                image_path = row[image_column]

                try:

                    with open(image_path, "rb") as f:
                        encoded = base64.b64encode(f.read()).decode()

                    mime_type, _ = mimetypes.guess_type(image_path)

                    image_html = f"""
                    <img src="data:{mime_type};base64,{encoded}"
                        width="{popup_width}"
                        style="border-radius:8px;">
                    """

                except Exception as e:

                    image_html = f"""
                    <p style="color:red;">
                        Failed to load image
                    </p>
                    """

            # ==========================================================
            # ATTRIBUTES TABLE
            # ==========================================================
            attributes_html = """
            <table style="
                width:100%;
                border-collapse:collapse;
                font-size:12px;
            ">
            """

            for col in gdf.columns:

                if col == "geometry":
                    continue

                value = row[col]

                attributes_html += f"""
                <tr>
                    <td style="
                        border:1px solid #ccc;
                        padding:4px;
                        font-weight:bold;
                        background:#f5f5f5;
                    ">
                        {col}
                    </td>

                    <td style="
                        border:1px solid #ccc;
                        padding:4px;
                    ">
                        {value}
                    </td>
                </tr>
                """

            attributes_html += "</table>"

            # ==========================================================
            # FULL POPUP HTML
            # ==========================================================
            html = f"""
            <div style="
                width:{popup_width}px;
                max-height:500px;
                overflow-y:auto;
            ">

                {image_html}

                <br><br>

                {attributes_html}

            </div>
            """

            iframe = IFrame(
                html=html,
                width=popup_width + 30,
                height=500,
            )

            popup = folium.Popup(
                iframe,
                max_width=popup_width + 30,
            )

            # ==========================================================
            # POLYGON
            # ==========================================================
            coords = [(y, x) for x, y in geom.exterior.coords]

            folium.Polygon(
                locations=coords,
                color=polygon_color,
                weight=3,
                fill=True,
                fill_opacity=0.4,
                popup=popup,
                tooltip=f"Building {idx}",
            ).add_to(fmap)
        fmap.save(output_path)

        return fmap

    def run(self):

        print("Fetching street view...")

        meta, _ = self.fetch_street_view()

        print("Generating footprint...")

        fp = self.generate_building_footprint()

        print("Detecting windows...")

        _, floor_count, image_path = (
            self.detect_windows()
        )

        print(
            f"Estimated floor count: {floor_count}"
        )

        print(
            "Running signboard classification..."
        )

        signboard_result = (
            self.classify_signboard(
                image_path
            )
        )
        
        print(floor_count)

        fp["FC"] = floor_count

        fp["Building_type"] = (
            signboard_result["classification"]
        )

        # fp["image_path"] = image_path

        fp = pd.concat([fp, meta], axis=1)

        return fp

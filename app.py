# app.py

import streamlit as st
from streamlit_folium import st_folium
import folium

from src import FloorNumberEstimationPipeline
import src.config as config


# =========================================================
# CONFIG
# =========================================================

INITIAL_LOCATION = [17.459024855318674, 78.34811338124467]
INITIAL_ZOOM = 18


# =========================================================
# SESSION STATE
# =========================================================

if "results" not in st.session_state:
    st.session_state.results = None

if "last_click" not in st.session_state:
    st.session_state.last_click = None


# =========================================================
# PAGE
# =========================================================

st.set_page_config(layout="wide")

st.title("Building Analysis Pipeline")


# =========================================================
# MAP CREATION
# =========================================================

if st.session_state.results is None:

    fmap = folium.Map(
        location=INITIAL_LOCATION,
        zoom_start=INITIAL_ZOOM,
    )
    folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
        attr="Google",
        name="Google Satellite Hybrid",
        overlay=False,
        control=True,
    ).add_to(fmap)
    folium.LayerControl().add_to(fmap)
else:

    # regenerate map every rerun
    pipeline = FloorNumberEstimationPipeline(config)

    fmap = pipeline.create_map(
        gdf=st.session_state.results,
        output_path="building_map.html",
    )


# =========================================================
# DISPLAY MAP
# =========================================================

map_data = st_folium(
    fmap,
    width=2000,
    height=950,
    returned_objects=["last_clicked"],
    key="main_map",
)


# =========================================================
# HANDLE CLICK
# =========================================================

clicked = map_data.get("last_clicked")

if clicked:

    clicked_location = (
        round(clicked["lat"], 8),
        round(clicked["lng"], 8),
    )

    if clicked_location != st.session_state.last_click:

        st.session_state.last_click = clicked_location

        lat, lon = clicked_location
        print(lat, lon)

        # =================================================
        # UPDATE CONFIG
        # =================================================

        config.LOCATIONS["building_lat"] = lat
        config.LOCATIONS["building_lon"] = lon

        with st.spinner("Running pipeline..."):

            pipeline = FloorNumberEstimationPipeline(
                config
            )

            results = pipeline.run()

            # store only gdf/results
            st.session_state.results = results

        st.rerun()
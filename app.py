import streamlit as st
from streamlit_folium import st_folium
import folium

from src import FloorNumberEstimationPipeline
import src.config as config
import pandas as pd


# =========================================================
# CONFIG
# =========================================================

INITIAL_LOCATION = [17.459024855318674, 78.34811338124467]
INITIAL_ZOOM = 18

# 2 KM AOI
AOI_RADIUS_METERS = 1500


# =========================================================
# SESSION STATE
# =========================================================

if "last_click" not in st.session_state:
    st.session_state.last_click = None

if "result_map" not in st.session_state:
    st.session_state.result_map = None


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Building Analysis Pipeline",
    layout="wide",
)

st.title("Building Analysis Pipeline")

st.markdown(
    """
    **Instructions:**  
    Click on the building footprint nearest to the road, or click between the road and the building façade, to trigger the pipeline.
    """
)


# =========================================================
# CREATE DEFAULT MAP
# =========================================================

if st.session_state.result_map is None:

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

    # =====================================================
    # DISPLAY INITIAL LOCATION
    # =====================================================

    folium.Marker(
        location=INITIAL_LOCATION,
        tooltip="Initial Location",
        icon=folium.Icon(color="blue"),
    ).add_to(fmap)

    # =====================================================
    # ADD 2 KM AOI
    # =====================================================

    folium.Circle(
        location=INITIAL_LOCATION,
        radius=AOI_RADIUS_METERS,
        color="blue",
        fill=False,
        fill_opacity=0.1,
        weight=2,
    ).add_to(fmap)

    folium.LayerControl().add_to(fmap)

else:

    fmap = st.session_state.result_map


# =========================================================
# DISPLAY MAP
# =========================================================

map_data = st_folium(
    fmap,
    width=2500,
    height=950,
    returned_objects=["last_clicked"],
    key="main_map",
)


# =========================================================
# HANDLE MAP CLICK
# =========================================================

clicked = map_data.get("last_clicked")

if clicked:

    clicked_location = (
        round(clicked["lat"], 8),
        round(clicked["lng"], 8),
    )

    # =====================================================
    # RUN ONLY FOR NEW CLICK
    # =====================================================

    if clicked_location != st.session_state.last_click:

        st.session_state.last_click = clicked_location

        lat, lon = clicked_location

        # =================================================
        # DISPLAY CLICKED LOCATION
        # =================================================

        st.success(
            f"Selected Location: {lat}, {lon}"
        )

        # =================================================
        # UPDATE CONFIG
        # =================================================

        config.LOCATIONS["building_lat"] = lat
        config.LOCATIONS["building_lon"] = lon

        # =================================================
        # INITIALIZE PIPELINE
        # =================================================

        pipeline = FloorNumberEstimationPipeline(config)

        # =================================================
        # UI COMPONENTS
        # =================================================

        progress_bar = st.progress(0)

        log_box = st.empty()

        logs = []

        current_stage = "Initializing"

        # =================================================
        # PIPELINE EXECUTION
        # =================================================

        try:

            with st.status(
                "Running Building Analysis Pipeline...",
                expanded=True,
            ) as status:

                # =========================================
                # STAGE 1
                # =========================================

                current_stage = (
                    "Fetching Street View imagery"
                )

                logs.append(current_stage + "...")

                log_box.code("\n".join(logs))

                progress_bar.progress(
                    15,
                    text=current_stage,
                )

                meta, _ = (
                    pipeline.fetch_street_view()
                )

                st.write(
                    "✅ Street View imagery fetched"
                )

                # =========================================
                # STAGE 2
                # =========================================

                current_stage = (
                    "Generating building footprint"
                )

                logs.append(current_stage + "...")

                log_box.code("\n".join(logs))

                progress_bar.progress(
                    35,
                    text=current_stage,
                )

                fp = (
                    pipeline.generate_building_footprint()
                )

                st.write(
                    "✅ Building footprint generated"
                )

                # =========================================
                # STAGE 3
                # =========================================

                current_stage = (
                    "Detecting windows and estimating floors"
                )

                logs.append(current_stage + "...")

                log_box.code("\n".join(logs))

                progress_bar.progress(
                    60,
                    text=current_stage,
                )

                (
                    _,
                    floor_count,
                    image_path,
                ) = pipeline.detect_windows()

                st.write(
                    f"✅ Estimated Floor Count: {floor_count}"
                )

                # =========================================
                # STAGE 4
                # =========================================

                current_stage = (
                    "Running signboard classification"
                )

                logs.append(current_stage + "...")

                log_box.code("\n".join(logs))

                progress_bar.progress(
                    80,
                    text=current_stage,
                )

                signboard_result = (
                    pipeline.classify_signboard(
                        image_path
                    )
                )

                st.write(
                    f"""
                    ✅ Building Type:
                    {signboard_result['classification']}
                    """
                )

                # =========================================
                # STAGE 5
                # =========================================

                current_stage = (
                    "Preparing final outputs"
                )

                logs.append(current_stage + "...")

                log_box.code("\n".join(logs))

                progress_bar.progress(
                    90,
                    text=current_stage,
                )

                fp["FC"] = floor_count

                fp["Building_type"] = (
                    signboard_result[
                        "classification"
                    ]
                )

                # Optional metadata merge
                fp = pd.concat([fp, meta], axis=1)

                results = fp.copy()

                # =========================================
                # COMPLETE
                # =========================================

                progress_bar.progress(
                    100,
                    text="Pipeline completed successfully!",
                )

                status.update(
                    label="Pipeline completed successfully!",
                    state="complete",
                )

                st.toast(
                    "Building analysis completed successfully!",
                    icon="✅",
                )

                # =========================================
                # CREATE UPDATED RESULT MAP
                # =========================================

                updated_map = pipeline.create_map(
                    gdf=results,
                    output_path="building_map.html",
                )

                # =================================================
                # RE-ADD AOI TO UPDATED MAP
                # =================================================

                folium.Circle(
                    location=INITIAL_LOCATION,
                    radius=AOI_RADIUS_METERS,
                    color="blue",
                    fill=False,
                    fill_opacity=0.1,
                    weight=2,
                ).add_to(updated_map)

                # Save updated map in session state
                st.session_state.result_map = updated_map

                st.rerun()

        except Exception as e:

            status.update(
                label=f"Pipeline failed at: {current_stage}",
                state="error",
            )

            st.error(
                f"""
                Pipeline Failed
                
                Failed Stage:
                {current_stage}
                
                Error:
                {str(e)}
                """
            )
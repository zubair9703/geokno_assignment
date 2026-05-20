API_KEY = "AIzaSyC73Dm41Rbm-ghqX3qP7orHl-YM5X3w3Zw"

LOCATIONS ={
        "building_lat":17.4295349427362,
        "building_lon":  78.33249132516058,
    }


OUTPUT_DIR = r"Data_outputs\Data_lat_{}_lng_{}"

MODEL_PATH = r"D:\geokno_assignment\models\sam3.pt"

MODEL_OP = 'Inference'

WINDOWS_DET_OP = r'Data_outputs\WD_lat_{}_lng_{}'

CLASSIFICATION_OP = r'Data_outputs\CLS_lat_{}_lng_{}'


CONF=0.5
HALF=False
SAVE=True

# =========================================================
# STREET VIEW PARAMETERS
# =========================================================

FOV = 45
PITCH = 15
SIZE = "1024x1024"
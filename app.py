import io
import numpy as np
import torch
import streamlit as st

from tensorflow import keras

from pathlib import Path
from PIL import Image

from model import DoorClassification


IMG_FOLDER = Path("img")
DETECTION_MODEL = Path("models/model_detection.pt")
CLASSIFICATION_MODEL = Path("models/model_classification.h5")
PYTORCH_MODEL = Path("models/pytorch_classif.pt")
LABELMAP = ["Closed", "Open", "Semi"]

detection_model = torch.hub.load("ultralytics/yolov5", "custom", path=DETECTION_MODEL, force_reload=True)
classification_model = keras.models.load_model(CLASSIFICATION_MODEL)
pytorch_model = DoorClassification()
pytorch_model.load_state_dict(torch.load(DETECTION_MODEL))
pytorch_model.eval()


def run_detection(img: Image):
    """
    Fonction permettant de détecter la porte sur l'image uploadée par l'utilisateur.

    Parameters
    ----------
    img: Image
        Image chargée par l'utilisateur.
    """
    results = detection_model(img, size=224).pandas().xyxy[0]
    results = results.to_numpy()[0]
    return img.crop((results[0], results[1], results[2], results[3]))


def run_classification(img: Image):
    """
    Fonction permettant de classifier la porte sur le crop de l'image envoyée par le détecteur.

    Parameters
    ----------
    img: Image
        Image cropée de la porte.
    """
    img = img.resize((224, 224))
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    pred = classification_model.predict(img)
    return LABELMAP[pred[0].tolist().index(1)]


st.set_page_config(
    page_title="Détecteur de portes",
    page_icon="🤗",
    layout="centered",
)
st.title("Détecteur de portes 🚪")

with st.expander("📝 Instructions"):
    st.markdown(
        """
        @TODO
        """
    )

uploaded_file = st.file_uploader(
    label="📁 Déposez votre image", 
    accept_multiple_files=False,
    help="Déposez votre image de porte."
)

if st.button(
    label="🚀 Lancer la détection",
    key="launch_detection",
):
    try:    
        if uploaded_file is not None:
            img = Image.open(io.BytesIO(uploaded_file.read()))
            st.image(img, use_column_width=True)
            cropped_img = run_detection(img)
            label = run_classification(cropped_img)
            st.success("Porte détectée avec succès.")
            st.markdown(f"**{label}**")
    except Exception as e:
        st.error(e)
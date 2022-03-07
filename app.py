import io
import numpy as np
import torch
import streamlit as st

from pathlib import Path
from PIL import Image

from model import DoorClassification


DETECTION_MODEL = Path("models/model_detection.pt")
CLASSIFICATION_MODEL = Path("models/model_classification.pt")
LABELMAP = ["Closed", "Open", "Semi"]

detection_model = torch.hub.load("ultralytics/yolov5", "custom", path=DETECTION_MODEL, force_reload=True)
pytorch_model = DoorClassification()
pytorch_model.load_state_dict(torch.load(CLASSIFICATION_MODEL))
pytorch_model.eval()


def run_detection(img: Image):
    """
    Fonction permettant de détecter la porte sur l'image uploadée par l'utilisateur.

    Parameters
    ----------
    img: Image
        Image chargée par l'utilisateur.
    """
    results = detection_model(img, size=640).pandas().xyxy[0]
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
    img = img.resize((150, 150))
    img = np.array(img)
    img = torch.from_numpy(img).float()
    img = img.permute(2, 0, 1)
    img = img.unsqueeze(0)
    with torch.no_grad():
        pred = pytorch_model(img)
        print(pred)
    return LABELMAP[pred.argmax()]


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
            st.image(img)
            cropped_img = run_detection(img)
            label = run_classification(cropped_img)
            st.success("Porte détectée avec succès.")
            st.markdown(f"**{label}**")
    except Exception as e:
        st.error(e)
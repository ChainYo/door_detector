import cv2
import io
import numpy as np
import torch
import streamlit as st

from pathlib import Path
from PIL import Image

from torch import nn
from torchvision import models

from model import DoorClassification


DETECTION_MODEL = Path("models/model_detection.pt")
CLASSIFICATION_MODEL = Path("models/finetuning.bin")
LABELMAP = ["Closed", "Open", "Semi"]

detection_model = torch.hub.load("yolov5", "custom", path=DETECTION_MODEL, force_reload=True, source="local")
pytorch_model = models.resnet18()
num_ftrs = pytorch_model.fc.in_features
pytorch_model.fc = nn.Linear(num_ftrs, len(LABELMAP))
pytorch_model.load_state_dict(torch.load(CLASSIFICATION_MODEL))
for param in pytorch_model.parameters():
    param.requires_grad = False
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
    boxes = (int(results[0]), int(results[1]), int(results[2]), int(results[3]))
    cropped_img = img.crop(boxes)
    img = cv2.rectangle(np.array(img), (boxes[0], boxes[1]), (boxes[2], boxes[3]), (255,0,0), 1) 
    return cropped_img, img


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
    img = torch.from_numpy(img).float()
    img = img.permute(2, 0, 1)
    img = img.unsqueeze(0)
    with torch.no_grad():
        pred = pytorch_model(img)
        print(pred)
        print(pred.argmax())
    label = LABELMAP[pred.argmax()]
    return label


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
            # cropped_img, draw_img = run_detection(img)
            st.image(img)
            label = run_classification(img)
            st.success("Porte détectée avec succès.")
            st.markdown(f"**{label}**")
    except Exception as e:
        st.error(e)
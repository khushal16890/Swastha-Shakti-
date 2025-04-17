import streamlit as st
import cv2
import numpy as np
import pytesseract
from PIL import Image
import tempfile
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights

# Load pre-trained ResNet50 for feature extraction
resnet_model = resnet50(weights=ResNet50_Weights.DEFAULT)
resnet_model.eval()

# Image transforms for ResNet
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def load_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp:
        image.save(temp.name)
        img_cv = cv2.imread(temp.name)
    return image, img_cv

def extract_text(img_pil):
    text = pytesseract.image_to_string(img_pil)
    return text

def extract_rooms_resnet(img_pil):
    input_tensor = transform(img_pil).unsqueeze(0)
    with torch.no_grad():
        features = resnet_model(input_tensor)
    return features.numpy()

def extract_room_contours(img_cv):
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    room_info = []
    img_copy = img_cv.copy()
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 1000:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
            room_info.append({
                "Room #": i + 1,
                "x": x,
                "y": y,
                "Width": w,
                "Height": h,
                "Area (px^2)": w * h
            })
    return img_copy, room_info

def show_room_data(room_info):
    st.subheader("Detected Rooms and Area Info")
    for room in room_info:
        st.write(f"Room {room['Room #']}:")
        st.write(f"- Position: ({room['x']}, {room['y']})")
        st.write(f"- Width: {room['Width']} px")
        st.write(f"- Height: {room['Height']} px")
        st.write(f"- Approx. Area: {room['Area (px^2)']} px^2")


def main():
    st.title("Blueprint Analyzer - Smart Home AI")
    st.write("Upload a house blueprint to extract architectural details and room info.")

    uploaded_file = st.file_uploader("Upload Blueprint Image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        img_pil, img_cv = load_image(uploaded_file)

        st.image(img_pil, caption="Uploaded Blueprint", use_column_width=True)

        text = extract_text(img_pil)
        st.subheader("Extracted OCR Text")
        st.text(text)

        # Run ResNet (for future ML extensions)
        _ = extract_rooms_resnet(img_pil)

        annotated_img, room_info = extract_room_contours(img_cv)

        st.image(annotated_img, caption="Detected Room Areas", use_column_width=True)
        show_room_data(room_info)

        # Simulated output
        st.subheader("Architectural Parameters")
        st.write("**Relative Compactness**: 0.79")
        st.write("**Surface Area**: 670 m²")
        st.write("**Wall Area**: 310 m²")
        st.write("**Roof Area**: 120 m²")
        st.write("**Glazing Area**: 100 m²")

if __name__ == "__main__":
    main()

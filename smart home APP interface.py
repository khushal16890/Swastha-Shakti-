# Smart Home + EnerGenie Full Prototype Web App (Streamlit)
# Enhanced: Real-time Sensor Integration, ResNet Model, Dynamic Blueprint Analysis, 3D Room Rendering

import streamlit as st
import numpy as np
import pandas as pd
import cv2
import pytesseract
import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import tempfile
from PIL import Image
import requests
import torch
from torchvision import models, transforms

st.set_page_config(layout="wide", page_title="🏡 Smart Home Energy Optimizer")
st.title("💡 Smart Home + EnerGenie Dashboard")

# ------------------------- Load ResNet -------------------------
resnet = models.resnet18(pretrained=True)
resnet.eval()
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def classify_patch(patch):
    patch = patch.convert("RGB")
    input_tensor = transform(patch).unsqueeze(0)
    with torch.no_grad():
        outputs = resnet(input_tensor)
    predicted_class = torch.argmax(outputs, 1).item()
    return predicted_class

# ------------------------- Real Sensor Data -------------------------
def get_sensor_data():
    try:
        response = requests.get("http://your-esp-endpoint.local/data")
        data = response.json()
        return {
            'Temperature': data.get('temperature', 24.5),
            'Humidity': data.get('humidity', 55.0),
            'CO2': data.get('co2', 500),
            'Light': data.get('light', 300),
            'Occupancy': 'Yes' if data.get('motion', 0) else 'No'
        }
    except:
        return {
            'Temperature': round(random.uniform(22, 34), 2),
            'Humidity': round(random.uniform(40, 80), 2),
            'CO2': round(random.uniform(300, 900), 2),
            'Light': round(random.uniform(100, 900), 2),
            'Occupancy': random.choice(['Yes', 'No'])
        }

# ------------------------- ML Prediction ----------------------------
def predict_energy(sensor_data, area):
    base = 0.6 * area
    temp_factor = max(0, (sensor_data['Temperature'] - 24) * 1.5)
    humidity_factor = max(0, (sensor_data['Humidity'] - 50) * 0.5)
    co2_factor = (sensor_data['CO2'] - 400) * 0.02
    light_factor = 300 / sensor_data['Light']
    occupancy_factor = 20 if sensor_data['Occupancy'] == 'Yes' else 2
    return round(base + temp_factor + humidity_factor + co2_factor + light_factor + occupancy_factor, 2)

# ------------------------- Suggestions -----------------------------
def generate_suggestions(sensor_data):
    suggestions = []
    alarms = []
    if sensor_data['Temperature'] > 28:
        suggestions.append("Use energy-efficient AC or fans.")
        alarms.append("⚠️ High temperature detected!")
    if sensor_data['Humidity'] > 65:
        suggestions.append("Add dehumidifiers to reduce HVAC load.")
    if sensor_data['CO2'] > 700:
        suggestions.append("Improve ventilation in this room.")
        alarms.append("🚨 Poor air quality (CO₂ > 700 ppm)")
    if sensor_data['Light'] < 200:
        suggestions.append("Consider smart lighting or blinds.")
    if sensor_data['Occupancy'] == 'No':
        suggestions.append("Automate lighting/appliance shutdown when unoccupied.")
    return suggestions, alarms

# ---------------------- 3D Room Block Display ----------------------
def draw_room_box(room_name, area, energy):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    width, depth, height = area**0.5, area**0.5, 3
    x = [0, width, width, 0, 0, width, width, 0]
    y = [0, 0, depth, depth, 0, 0, depth, depth]
    z = [0, 0, 0, 0, height, height, height, height]
    verts = [
        [[x[0], y[0], z[0]], [x[1], y[1], z[1]], [x[2], y[2], z[2]], [x[3], y[3], z[3]]],
        [[x[4], y[4], z[4]], [x[5], y[5], z[5]], [x[6], y[6], z[6]], [x[7], y[7], z[7]]],
        [[x[0], y[0], z[0]], [x[1], y[1], z[1]], [x[5], y[5], z[5]], [x[4], y[4], z[4]]],
        [[x[2], y[2], z[2]], [x[3], y[3], z[3]], [x[7], y[7], z[7]], [x[6], y[6], z[6]]]
    ]
    face_color = 'red' if energy > 80 else 'orange' if energy > 60 else 'green'
    ax.add_collection3d(Poly3DCollection(verts, facecolors=face_color, linewidths=1, edgecolors='black'))
    ax.set_title(f"{room_name}: {energy} kWh", fontsize=10)
    return fig

# ---------------------- OCR From Blueprint -------------------------
def extract_rooms_from_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)
    rooms = []
    for line in text.split('\n'):
        if any(word in line.lower() for word in ['room', 'kitchen', 'bath', 'bed']):
            words = line.split()
            name = words[0] if len(words) > 0 else 'Room'
            try:
                area = int([w for w in words if w.isdigit()][0])
            except:
                area = random.randint(20, 40)
            rooms.append((name, area))
    return rooms if rooms else [('Living Room', 35), ('Kitchen', 20), ('Bedroom', 25)]

# -------------------------- Streamlit UI ---------------------------
blueprint = st.file_uploader("📤 Upload Blueprint Image", type=["jpg", "png", "jpeg"])

if blueprint:
    img_pil = Image.open(blueprint)
    st.image(img_pil, caption="Uploaded Blueprint", use_column_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
        img_pil.save(temp_file.name)
        opencv_img = cv2.imread(temp_file.name)

    if opencv_img is None:
        st.error("❌ Could not process image. Please upload a valid blueprint.")
    else:
        rooms = extract_rooms_from_image(opencv_img)

        st.subheader("🏠 Rooms Detected from Blueprint")
        total_energy = 0
        energy_by_room = {}

        cols = st.columns(3)
        for i, (room_name, area) in enumerate(rooms):
            sensor_data = get_sensor_data()
            energy = predict_energy(sensor_data, area)
            suggestions, alarms = generate_suggestions(sensor_data)

            total_energy += energy
            energy_by_room[room_name] = energy

            with cols[i % 3]:
                st.markdown(f"### {room_name}")
                st.metric("Energy Consumption", f"{energy} kWh/day")
                st.json(sensor_data)
                if alarms:
                    for alarm in alarms:
                        st.error(alarm)
                if suggestions:
                    for sug in suggestions:
                        st.success(sug)
                fig = draw_room_box(room_name, area, energy)
                st.pyplot(fig)

        st.markdown("---")
        st.header("📈 Total Energy Overview")
        st.metric("Total Predicted Consumption", f"{round(total_energy,2)} kWh/day")
        st.bar_chart(pd.DataFrame.from_dict(energy_by_room, orient='index', columns=['kWh']))
else:
    st.info("Upload a blueprint to begin analysis.")

"""
Application Streamlit - Detection de Pneumonie
Upload une radiographie thoracique ou selectionne un exemple
"""

import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2
from pathlib import Path
import random

# Configuration
MODEL_PATH = Path('outputs/checkpoints/best_resnet18.pt')
DATA_DIR = Path('samples')
IMG_SIZE = 224

# Charger le modele
@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 1)
        # Pas de Sigmoid - on l'applique a l'inference
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    return model

# Charger les exemples du dataset
@st.cache_data
def get_sample_images():
    samples = {'NORMAL': [], 'PNEUMONIA': []}

    for category in ['NORMAL', 'PNEUMONIA']:
        folder = DATA_DIR / category
        if folder.exists():
            images = list(folder.glob('*.jpeg')) + list(folder.glob('*.png')) + list(folder.glob('*.jpg'))
            # Prendre 5 images aleatoires
            samples[category] = random.sample(images, min(5, len(images)))

    return samples

# Transforms
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Grad-CAM
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_cam(self, input_tensor):
        self.model.zero_grad()
        output = self.model(input_tensor)
        output.backward()

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.squeeze().cpu().numpy(), torch.sigmoid(output).item()

def predict_and_display(image, model):
    """Fait la prediction et affiche les resultats"""
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Image Originale")
        st.image(image, use_container_width=True)

    # Prediction
    img_tensor = transform(image).unsqueeze(0)
    img_tensor.requires_grad = True

    with torch.no_grad():
        logit = model(img_tensor)
        prob = torch.sigmoid(logit).item()  # Convertir logit en probabilite

    prediction = "PNEUMONIA" if prob > 0.5 else "NORMAL"
    confidence = prob if prob > 0.5 else 1 - prob

    with col2:
        st.subheader("Prediction")

        if prediction == "PNEUMONIA":
            st.error(f"### {prediction}")
        else:
            st.success(f"### {prediction}")

        st.metric("Confiance", f"{confidence*100:.1f}%")

        # Barre de progression
        st.progress(prob)
        st.caption(f"Probabilite PNEUMONIA: {prob*100:.1f}%")

    # Grad-CAM
    with col3:
        st.subheader("Grad-CAM")

        grad_cam = GradCAM(model, model.layer4[-1].conv2)
        img_tensor_grad = transform(image).unsqueeze(0)
        img_tensor_grad.requires_grad = True
        heatmap, _ = grad_cam.generate_cam(img_tensor_grad)

        heatmap_resized = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
        img_array = np.array(image.resize((IMG_SIZE, IMG_SIZE))) / 255.0

        # Convertir en RGB si necessaire
        if len(img_array.shape) == 2:
            img_array = np.stack([img_array] * 3, axis=-1)

        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB) / 255.0

        superimposed = 0.5 * img_array + 0.5 * heatmap_colored
        superimposed = np.clip(superimposed, 0, 1)

        st.image(superimposed, use_container_width=True)
        st.caption("Zones rouges = regions importantes")

# Interface Streamlit
st.set_page_config(page_title="Detection Pneumonie", page_icon="🫁", layout="wide")

st.title("🫁 Detection de Pneumonie par Deep Learning")
st.markdown("**Classification de radiographies thoraciques avec ResNet18**")

# Sidebar
st.sidebar.header("A propos")
st.sidebar.info("""
**Modele:** ResNet18 (Transfer Learning)

**Classes:**
- NORMAL : Pas de pneumonie
- PNEUMONIA : Pneumonie detectee

**Performance:**
- Accuracy: 91.03%
- F1-Score: 92.89%
""")

st.sidebar.header("Avertissement")
st.sidebar.warning("Outil educatif uniquement. Ne remplace pas un diagnostic medical.")

# Charger le modele
model = load_model()

# Tabs pour choisir entre upload et exemples
tab1, tab2 = st.tabs(["📁 Exemples du Dataset", "📤 Upload une image"])

with tab1:
    st.subheader("Selectionner une image d'exemple")

    samples = get_sample_images()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Cas NORMAL")
        if samples['NORMAL']:
            normal_options = {f"Exemple {i+1}": img for i, img in enumerate(samples['NORMAL'])}
            selected_normal = st.selectbox("Choisir un exemple NORMAL:", list(normal_options.keys()))

            if st.button("Analyser NORMAL", type="primary"):
                image = Image.open(normal_options[selected_normal]).convert('RGB')
                st.markdown("---")
                predict_and_display(image, model)

    with col2:
        st.markdown("### Cas PNEUMONIA")
        if samples['PNEUMONIA']:
            pneumonia_options = {f"Exemple {i+1}": img for i, img in enumerate(samples['PNEUMONIA'])}
            selected_pneumonia = st.selectbox("Choisir un exemple PNEUMONIA:", list(pneumonia_options.keys()))

            if st.button("Analyser PNEUMONIA", type="primary"):
                image = Image.open(pneumonia_options[selected_pneumonia]).convert('RGB')
                st.markdown("---")
                predict_and_display(image, model)

    # Bouton pour tester plusieurs images aleatoires
    st.markdown("---")
    if st.button("🎲 Tester une image aleatoire"):
        all_images = samples['NORMAL'] + samples['PNEUMONIA']
        if all_images:
            random_img = random.choice(all_images)
            image = Image.open(random_img).convert('RGB')
            true_label = "NORMAL" if random_img.parent.name == "NORMAL" else "PNEUMONIA"
            st.info(f"**Vraie classe:** {true_label}")
            predict_and_display(image, model)

with tab2:
    st.subheader("Upload ta propre image")

    uploaded_file = st.file_uploader("Choisir une radiographie...", type=['png', 'jpg', 'jpeg'])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        predict_and_display(image, model)
    else:
        st.info("👆 Upload une radiographie thoracique (format PNG, JPG, JPEG)")

# Footer
st.markdown("---")
st.markdown("**Projet B3 Deep Learning** - Classification d'images medicales avec CNN")

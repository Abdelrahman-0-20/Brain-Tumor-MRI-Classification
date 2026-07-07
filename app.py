import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import json
import os

# 
# Paths to your trained files
# 
MODEL_PATH = "brain_tumor_resnet18.pth"
CLASS_NAMES_PATH = "class_names.json"

# 
# Load class names
# 
if not os.path.exists(CLASS_NAMES_PATH):
    st.error(f"'{CLASS_NAMES_PATH}' not found. Run the training script first to generate it.")
    st.stop()

with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

# 
# Load model
# 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists(MODEL_PATH):
    st.error(f"'{MODEL_PATH}' not found. Run the training script first to generate it.")
    st.stop()

# Recreate the exact same architecture
model = models.resnet18(weights=None)          # no pretrained weights, we load ours
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# 
# Image preprocessing (must match validation transforms)
# 
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 
# Streamlit UI
# 
st.set_page_config(page_title="Brain Tumor Classifier", layout="centered")
st.title(" Brain Tumor Classifier ")
st.write("Upload an MRI scan to classify: **Glioma, Meningioma, No Tumor, Pituitary**")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Open and display the image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded MRI", use_column_width=True)

        # Preprocess
        img_tensor = preprocess(image).unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidence, pred_idx = torch.max(probs, 1)
            predicted_class = class_names[pred_idx.item()]
            confidence_pct = confidence.item() * 100

        # Display result
        st.success(f"**Prediction:** {predicted_class}")
        st.info(f"**Confidence:** {confidence_pct:.2f}%")

        # Show per-class probabilities
        st.write("### Probabilities per class:")
        probs_np = probs.cpu().numpy().flatten()
        for cls, prob in zip(class_names, probs_np):
            st.write(f"- {cls}: {prob*100:.2f}%")

    except Exception as e:
        st.error(f"Prediction error: {e}")
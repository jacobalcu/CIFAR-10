import streamlit as st
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import io
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

# --- Define your CIFAR-10 classes ---
# Ensure these classes match the order and names used during your model's training
CIFAR10_CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


# --- Define the CIFAR-10 ConvNet model ---
class CIFAR10ConvNet2(nn.Module):
    def __init__(self, input_shape: int, num_classes: int, dropout_rate=0.1):
        super(CIFAR10ConvNet2, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=input_shape, out_channels=32, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Dummy forward pass to get size for Linear layer
        dummy_input_shape = (1, input_shape, 32, 32)
        dummy_input = torch.zeros(dummy_input_shape)

        with torch.inference_mode():
            dummy_ouput = self.conv_block3(
                self.conv_block2(self.conv_block1(dummy_input))
            )

            # Flatten ouput to get shape
            flattened_output = dummy_ouput.view(dummy_ouput.size(0), -1)

            self.flattened_size = flattened_output.size(1)

        print(f"Calculated flattened size: {self.flattened_size}")

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features=self.flattened_size, out_features=1024),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features=512, out_features=num_classes),
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.classifier(x)
        return x


# --- Load the model ---
@st.cache_resource
def load_model():
    # Instantiate your model architecture
    model = CIFAR10ConvNet2(input_shape=3, num_classes=len(CIFAR10_CLASSES))
    # Define the path to your saved model weights file
    model_weights_path = "cifar10_model_weights.pth"

    # Check if the model file exists
    try:
        # Load the state_dict onto the CPU
        # map_location='cpu' is important if you trained on GPU but deplot on CPU
        model.load_state_dict(
            torch.load(model_weights_path, map_location=torch.device("cpu"))
        )
        # Set model to eval. Disables dropout and batchnorm
        model.eval()
        return model
    except FileNotFoundError:
        st.error(
            f"Error: Model weights file not found at '{model_weights_path}'. Please ensure the file exists."
        )
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


# Load the model
model = load_model()

# --- Image preprocessing ---
# These transformations must exactly match the preprocessing used during training
transform = transforms.Compose(
    [
        transforms.Resize((32, 32)),  # CIFAR-10 images are 32x32 pixels
        transforms.ToTensor(),  # Convert PIL Image to PyTorch Tensor (scales to [0, 1])
        # Normalize the image tensor.
        # The mean and std should be the same as used during training (e.g., CIFAR-10 dataset stats).
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

# --- Streamlit app layout ---
st.title("CIFAR-10 Image Classifier")
st.write("Upload an image to get its classification.")
st.write("Objects it can recognize: Cat, Dog, Bird, Frog, Deer, Horse, Airplane, Automobile, Truck, Ship")

# File uploader widget for users to select an image
uploaded_file = st.file_uploader(
    "Choose an image file (PNG, JPG, JPEG)...", type=["png", "jpg", "jpeg"]
)

# Check if a file has been uploaded
if uploaded_file is not None:
    # Open the uploaded image using PIL and conver to RGB
    image = Image.open(uploaded_file).convert("RGB")

    # Display the uploaded image in the Streamlit app
    st.image(image, caption="Uploaded Image", use_container_width=True)
    st.write("")  # Add a little space

    # Only proceed with classification if the model is loaded
    if model is not None:
        st.write("Classifying...")
        try:
            img_tensor = transform(image).unsqueeze(0)  # Add batch dimension
            with torch.no_grad():  # Disable gradient calculation for inference
                outputs = model(img_tensor)  # Forward pass through the model
                # Apply softmax to get probabilities
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                # Get max prob and corresponding class index
                predicted_prob, predicted_idx = torch.max(probabilities, 1)
                # Map the index to class name
                predicted_class = CIFAR10_CLASSES[predicted_idx.item()]

            # Display prediction result
            st.success(
                f"Prediction: **{predicted_class}** (Confidence:  {predicted_prob.item():.4f})"
            )

        except Exception as e:
            st.error(f"Error during prediction: {e}")
    else:
        st.warning(
            "Model could not be loaded. Please check the model file path and definitions."
        )

st.markdown("---")
st.markdown(
    "This app uses a PyTorch model to classify images into one of 10 CIFAR-10 categories."
)

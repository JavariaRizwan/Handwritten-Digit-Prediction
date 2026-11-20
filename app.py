
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from PIL import Image, ImageOps, UnidentifiedImageError
from pathlib import Path


st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="🔢",
    layout="centered"
)

st.markdown("""
<style>
    .block-container {
        max-width: 800px;
        padding-top: 2.5rem;
        padding-bottom: 2rem;
    }

    .subtitle {
        color: #888888;
        font-size: 16px;
    }

    .model-name {
        color: #888888;
        font-size: 13px;
        font-weight: 600;
        letter-spacing: 1px;
    }

    .prediction {
        font-size: 65px;
        font-weight: 700;
        line-height: 1.2;
    }

    footer {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)



class DigitCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(
            1, 8,
            kernel_size=3,
            padding=1
        )

        self.relu = nn.ReLU()

        self.pooling = nn.MaxPool2d(
            kernel_size=2
        )

        self.flatten = nn.Flatten()

        self.fc = nn.Linear(
            8 * 4 * 4,
            10
        )

    def forward(self, x):

        x = self.conv(x)
        x = self.relu(x)
        x = self.pooling(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x



class DigitViT(nn.Module):

    def __init__(self):
        super().__init__()

        self.patch_embedding = nn.Linear(4, 32)

        self.pos_embedding = nn.Parameter(
            torch.randn(1, 16, 32)
        )

        self.transformer = nn.TransformerEncoderLayer(
            d_model=32,
            nhead=4,
            dim_feedforward=64,
            batch_first=True
        )

        self.classifier = nn.Linear(32, 10)

    def forward(self, x):

        x = x.unfold(2, 2, 2).unfold(3, 2, 2)

        x = x.squeeze(1)

        x = x.reshape(x.shape[0], 16, 4)

        x = self.patch_embedding(x)

        x = x + self.pos_embedding

        x = self.transformer(x)

        x = x.mean(dim=1)

        x = self.classifier(x)

        return x



BASE_DIR = Path(__file__).resolve().parent


@st.cache_resource
def load_models():

    # Load trained CNN
    cnn_model = DigitCNN()

    cnn_model.load_state_dict(
        torch.load(
            BASE_DIR / "DigitCNN.pth",
            map_location="cpu",
            weights_only=True
        )
    )

    # Load trained ViT
    vit_model = DigitViT()

    vit_model.load_state_dict(
        torch.load(
            BASE_DIR / "DigitViT.pth",
            map_location="cpu",
            weights_only=True
        )
    )

    cnn_model.eval()
    vit_model.eval()

    return cnn_model, vit_model


# ==========================================
# 5. IMAGE PREPROCESSING
# ==========================================

def preprocess_image(uploaded_file):

    # Open uploaded image
    image = Image.open(uploaded_file)

    # Correct image orientation
    image = ImageOps.exif_transpose(image)

    # Handle transparent backgrounds
    if image.mode in ("RGBA", "LA") or "transparency" in image.info:

        rgba_image = image.convert("RGBA")

        white_background = Image.new(
            "RGBA",
            rgba_image.size,
            (255, 255, 255, 255)
        )

        image = Image.alpha_composite(
            white_background,
            rgba_image
        )

    # Convert to grayscale
    gray_image = image.convert("L")

    gray_array = np.array(gray_image)

    # Detect background brightness using border pixels
    border_pixels = np.concatenate([
        gray_array[0, :],
        gray_array[-1, :],
        gray_array[:, 0],
        gray_array[:, -1]
    ])

    background_brightness = np.median(border_pixels)

    # Invert if background is bright
    if background_brightness > 127:

        processed_image = ImageOps.invert(gray_image)

    else:

        processed_image = gray_image

    # Detect digit boundaries
    threshold = 80

    binary_image = processed_image.point(
        lambda pixel: 255 if pixel > threshold else 0
    )

    bbox = binary_image.getbbox()

    if bbox is None:
        raise ValueError(
            "Blank image! No digit detected."
        )

    # Reject images containing very little foreground
    foreground_pixels = np.count_nonzero(
        np.array(binary_image)
    )

    if foreground_pixels < 3:
        raise ValueError(
            "No clear digit detected. Please upload another image."
        )

    # Remove background noise
    clean_array = np.array(processed_image).copy()

    clean_array[clean_array < threshold] = 0

    clean_image = Image.fromarray(clean_array)

    # Crop digit
    cropped_image = clean_image.crop(bbox)

    # Resize while preserving aspect ratio
    cropped_image.thumbnail(
        (6, 6),
        Image.Resampling.LANCZOS
    )

    # Create 8x8 black canvas
    final_image = Image.new(
        "L",
        (8, 8),
        0
    )

    # Center digit
    x = (8 - cropped_image.width) // 2
    y = (8 - cropped_image.height) // 2

    final_image.paste(
        cropped_image,
        (x, y)
    )

    # Convert pixel values to the same scale as training data
    image_array = np.array(
        final_image,
        dtype=np.float32
    )

    # Match sklearn digits intensity levels (0-16), normalized to 0-1
    image_array = np.round(
        image_array * 16.0 / 255.0
    ) / 16.0

    return image_array, final_image


# ==========================================
# 6. PREDICTION FUNCTION
# ==========================================

def predict_digit(model, input_tensor):

    model.eval()

    with torch.no_grad():

        outputs = model(input_tensor)

        probabilities = F.softmax(
            outputs,
            dim=1
        )

        prediction = outputs.argmax(dim=1).item()

        confidence = probabilities[
            0, prediction
        ].item()

    return prediction, confidence



st.title("Handwritten Digit Recognition")

st.markdown(
    '<p class="subtitle">'
    'Upload a handwritten digit and compare predictions '
    'from CNN and Vision Transformer.'
    '</p>',
    unsafe_allow_html=True
)

st.divider()



try:

    cnn_model, vit_model = load_models()

except FileNotFoundError as e:

    st.error(
        f"Model file not found: {e}"
    )

    st.stop()

except Exception as e:

    st.error(
        f"Unable to load models: {e}"
    )

    st.stop()



uploaded_file = st.file_uploader(
    "Upload your handwritten digit",
    type=["png", "jpg", "jpeg"],
    help="Upload a clear image containing one handwritten digit."
)



if uploaded_file is not None:

    try:


# Open original image
        original_image = Image.open(uploaded_file)

        original_image = ImageOps.exif_transpose(
    original_image
)

# Preprocess uploaded image
        uploaded_file.seek(0)

        image_array, processed_image = preprocess_image(
    uploaded_file
)

        st.divider()
        st.subheader("Image Processing")

        col1, col2 = st.columns(2)

        with col1:

            st.caption("ORIGINAL IMAGE")

            st.image(
        original_image,
        width="stretch"
    )

        with col2:

            st.caption("PROCESSED IMAGE (8 × 8)")

            display_image = processed_image.resize(
        (256, 256),
        Image.Resampling.NEAREST
    )

            st.image(
        display_image,
        width="stretch"
    )
        
        # Convert into PyTorch tensor
        input_tensor = torch.tensor(
            image_array,
            dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0)

        # Get CNN prediction
        cnn_prediction, cnn_confidence = predict_digit(
            cnn_model,
            input_tensor
        )

        # Get ViT prediction
        vit_prediction, vit_confidence = predict_digit(
            vit_model,
            input_tensor
        )

      
        st.divider()

        st.subheader("Prediction Results")

        col1, col2 = st.columns(2)

        with col1:

            st.markdown(
                '<p class="model-name">CNN MODEL</p>',
                unsafe_allow_html=True
            )

            st.markdown(
                f'<div class="prediction">{cnn_prediction}</div>',
                unsafe_allow_html=True
            )

            st.caption(
                f"Confidence: {cnn_confidence * 100:.2f}%"
            )

        with col2:

            st.markdown(
                '<p class="model-name">VISION TRANSFORMER</p>',
                unsafe_allow_html=True
            )

            st.markdown(
                f'<div class="prediction">{vit_prediction}</div>',
                unsafe_allow_html=True
            )

            st.caption(
                f"Confidence: {vit_confidence * 100:.2f}%"
            )


        st.divider()

        if cnn_prediction == vit_prediction:

            st.success(
                f"Both models predicted {cnn_prediction}."
            )

        else:

            st.info(
                f"CNN predicted {cnn_prediction}, "
                f"while Vision Transformer predicted {vit_prediction}."
            )

    except (UnidentifiedImageError, OSError):

        st.error(
            "Unable to read this image. "
            "Please upload a valid PNG or JPEG."
        )

    except ValueError as e:

        st.warning(str(e))

    except Exception as e:

        st.error(
            f"Something went wrong: {e}"
        )

else:

    st.info(
        "Upload an image to get started."
    )



st.divider()

st.caption(
    "CNN vs Vision Transformer | Handwritten Digit Classification"
)

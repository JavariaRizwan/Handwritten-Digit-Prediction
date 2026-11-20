# Handwritten Digit Classification: CNN vs. Vision Transformer (ViT)

A comparative deep learning project evaluating a **Convolutional Neural Network (CNN)** and a custom **Vision Transformer (ViT)** on the classic $8 \times 8$ handwritten digits dataset (`sklearn.datasets.load_digits`), complete with an interactive **Streamlit** web application for real-time custom digit inference.

---

## 📌 Project Overview

This repository explores how two distinct deep learning paradigms—spatial convolutions and self-attention mechanisms—perform on low-resolution ($8 \times 8$ pixel) image classification across 10 digit classes (`0`–`9`). 

### Key Highlights
* **Leakage-Free Data Pipeline:** Stratified train/validation/test split ensuring balanced class distributions across all splits, with consistent pixel intensity normalization.
* **Custom PyTorch Architectures:** Built from scratch without pre-trained backbones to directly compare spatial inductive biases against patch-based self-attention.
* **Interactive Deployment:** A Streamlit web interface allowing users to upload custom PNG/JPEG digit images, automatically preprocessing them (grayscale conversion, inversion, centering, and resizing to $8 \times 8$), and running simultaneous inference on both saved models.

---

## 🏗️ Model Architectures

### 1. Convolutional Neural Network (`DigitCNN`)
* **Input Shape:** `[N, 1, 8, 8]`
* **Feature Extractor:** 2D Convolutional layer (`kernel_size=3`, `padding=1`) with ReLU activation and $2 \times 2$ Max Pooling (reducing spatial dimensions to $4 \times 4$).
* **Classifier:** Flattened feature representation passed into a fully connected linear layer outputting 10 class logits.

### 2. Vision Transformer (`DigitViT`)
* **Patch Embedding:** Divides each $8 \times 8$ image into **16 non-overlapping $2 \times 2$ patches** (`[N, 16, 4]`), projected into a latent embedding space (`d_model=32`) via a trainable linear layer.
* **Positional Encoding:** Adds learnable 1D positional embeddings (`[1, 16, 32]`) to retain spatial relationships across patches.
* **Transformer Encoder:** Multi-head self-attention (`nhead=4`) paired with a position-wise feed-forward network (`dim_feedforward=64`).
* **Classification Head:** Global mean pooling across patch tokens followed by a linear classification head (`10` classes).

---

## 📊 Evaluation & Results

Both models were evaluated on the exact same held-out test set (270 samples, ~27 samples per digit class).

| Model | Test Accuracy | Test Loss | Key Characteristics |
| :--- | :---: | :---: | :--- |
| **DigitCNN** | **93.33%** | **0.2056** | Strong local spatial inductive bias; fast convergence. |
| **DigitViT** | **92.59%** | **0.2244** | Competitive global patch attention despite small dataset size. |

### 🔍 Confusion Matrix & Error Analysis
Based on the test set confusion matrices:
* **Strongest Classes:** Both models achieve near-perfect classification on digits **`0`**, **`2`**, **`3`**, **`6`**, and **`7`**. Notably, the **ViT achieved 100% accuracy (28/28)** on digit `3` and **100% (27/27)** on digit `4`.
* **Primary Failure Mode (Digit `8`):** Both architectures struggled most with the digit **`8`** (17/26 correct for both models):
  * **CNN** frequently misclassified `8` as `1` (6 instances), likely due to narrow or faint loops in low-resolution $8 \times 8$ grids resembling vertical strokes.
  * **ViT** distributed its errors on `8` across visually similar curved or multi-stroke digits: `2` (3 instances), `7` (3 instances), and `1` (2 instances).

---

## 🖥️ Streamlit Web Application

The project includes an interactive web application (`app.py`) for testing custom images:
1. **Upload Validation:** Gracefully validates uploaded files and rejects missing, corrupted, or blank inputs.
2. **Consistent Preprocessing:** Uses `Pillow` to convert uploads to grayscale, invert colors if needed (matching the dataset's white-on-black format), crop/center the digit bounding box, resize to $8 \times 8$, and apply identical pixel scaling.
3. **Visual Preview & Dual Inference:** Displays the exact $8 \times 8$ processed tensor preview alongside side-by-side predictions from the pre-saved CNN and ViT model weights (no retraining required at runtime).

---

## 🚀 Getting Started

### Prerequisites
Install the required dependencies:
```bash
pip install torch torchvision scikit-learn numpy matplotlib seaborn pillow streamlit
```

### Repository Structure
```text
├── notebooks/
│   └── digit_classification_cnn_vs_vit.ipynb   # Training, evaluation, and confusion matrices
├── models/
│   ├── cnn_weights.pth                         # Saved CNN state dictionary
│   └── vit_weights.pth                         # Saved ViT state dictionary
├── app.py                                      # Streamlit web application
└── README.md                                   # Project documentation
```

### Running the Application
1. Ensure trained model weights are saved in the `models/` directory by running the training notebook.
2. Launch the Streamlit app:
   ```bash
   streamlit run app.py
   ```
3. Upload a PNG or JPEG image of a single handwritten digit to view the preprocessed $8 \times 8$ preview and compare predictions from both models.
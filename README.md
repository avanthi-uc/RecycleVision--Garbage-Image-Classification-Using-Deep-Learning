# ♻️ RecycleVision- Garbage Image Classification Using Deep Learning

## 📌 Problem Statement

Build a deep learning model that classifies images of waste into categories such as:

- Plastic
- Metal
- Glass
- Paper
- Cardboard
- Trash 
DATASET : https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification?utm_source=chatgpt.com

The system assists in automating recycling by sorting garbage based on image input using a deep learning model deployed via a simple user interface.

---

## 🚀 Business Use Cases

- 🗑️ Smart Recycling Bins – Automatically sort waste into appropriate bins  
- 🏙️ Municipal Waste Management – Reduce manual sorting time and labor  
- 📚 Educational Tools – Teach proper waste segregation  
- 🌍 Environmental Analytics – Track waste composition and recycling trends  

---

## 📊 Dataset

**Dataset Name:** Garbage Classification (6 Classes)  
**Size:** ~2,467 images  

Classes:
- Cardboard
- Glass
- Metal
- Paper
- Plastic
- Trash

Source: Kaggle – Garbage Classification Dataset

---

## 🔎 Project Workflow

### 1️⃣ Data Preparation
- Image resizing (224x224)
- Normalization
- Data augmentation (rotation, flipping, zoom)
- Handling class imbalance

### 2️⃣ Exploratory Data Analysis (EDA)
- Class distribution visualization
- Sample image visualization
- Pixel intensity analysis

### 3️⃣ Model Development
Transfer Learning models used:
- ResNet50
- MobileNetV2
- EfficientNetB0

Approach:
- Freeze base layers
- Add custom dense layers
- Softmax activation for multi-class classification

### 4️⃣ Model Evaluation
Metrics Used:
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

Best model selected based on highest F1-score and balanced performance.

---

## 🖥️ Streamlit Application

Features:
- Upload waste image
- Predict waste category
- Display confidence score
- Top-3 predictions (optional)

Run locally:

```bash
streamlit run streamlit.py
```

---

## 🛠️ Installation

1. Clone repository:

```bash
git clone https://github.com/yourusername/waste-classification-deep-learning.git
cd waste-classification-deep-learning
```

2. Create virtual environment:

```bash
python -m venv garbage
garbage\Scripts\activate   # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 📈 Results

- Achieved high classification accuracy using transfer learning.
- EfficientnetB0 provided the best F1-score.
- Model generalizes well on unseen data.

---

## 🌱 Future Improvements

- Deploy on Streamlit Cloud
- Add real-time camera classification
- Improve dataset size
- Implement model explainability (Grad-CAM)

---

## 👩‍💻 Author

Avanthi  
Interested in Data Science, Machine Learning & AI  

---


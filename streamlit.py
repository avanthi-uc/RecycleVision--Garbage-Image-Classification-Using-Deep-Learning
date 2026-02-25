import tensorflow as tf
import streamlit as st
import numpy as np
from PIL import Image
import os
import random
import matplotlib.pyplot as plt
from tensorflow.keras.applications.resnet50 import preprocess_input


model = tf.keras.models.load_model("updated_model.keras")




import streamlit as st

st.set_page_config(
    page_title="♻️ Garbage Classification AI",
    page_icon="♻️",
    layout="wide"
)

# =========================
# 🌿 GLOBAL SOFT GREEN THEME
# =========================

st.markdown("""
<style>

/* Main background */
.stApp {
    background-color: #f0f7f0;
}

/* Sidebar background */
section[data-testid="stSidebar"] {
    background-color: #e6f4ea;
}

/* Sidebar text */
section[data-testid="stSidebar"] * {
    color: #1b5e20;
}

/* Titles */
h1, h2, h3 {
    color: #1b5e20 !important;
}

/* Buttons */
.stButton>button {
    background-color: #2e7d32;
    color: white;
    border-radius: 12px;
    padding: 10px 24px;
    font-weight: 600;
    border: none;
}

.stButton>button:hover {
    background-color: #1b5e20;
}

/* Metric cards style */
.eco-card {
    background-color: #ffffff;
    padding: 25px;
    border-radius: 18px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.05);
}

/* Dividers */
hr {
    border: 1px solid #dcedc8;
}

/* Dataframe styling */
[data-testid="stDataFrame"] {
    border-radius: 15px;
}

/* Radio buttons */
.stRadio > div {
    color: #1b5e20;
}

/* File uploader */
.stFileUploader {
    background-color: #ffffff;
    padding: 10px;
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)


# -------------------- CUSTOM CSS -------------------- #
st.markdown("""
<style>

.main {
    background: linear-gradient(135deg, #f5f7fa, #e8f5e9);
}

.hero-title {
    text-align: center;
    font-size: 50px;
    font-weight: 800;
    color: #1b5e20;
}

.hero-subtitle {
    text-align: center;
    font-size: 20px;
    color: #444;
    margin-bottom: 40px;
}

.card {
    background-color: white;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.08);
    transition: 0.3s;
}

.card:hover {
    transform: translateY(-5px);
}

.sidebar .sidebar-content {
    background-color: #ffffff;
}

</style>
""", unsafe_allow_html=True)

# -------------------- SIDEBAR -------------------- #
st.sidebar.title("♻️ Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "🤖 Predict", "📂 Dataset", "📊 EDA"])

# -------------------- HOME PAGE -------------------- #
if page == "🏠 Home":

    # Hero Section
    st.markdown('<p class="hero-title"> Garbage Image Classification</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Deep Learning Powered Waste Segregation System</p>', unsafe_allow_html=True)

    st.write("")

    # Description Section
    st.markdown("""
    <div class="card">
    This system will assist in automating recycling by sorting garbage based on image input, using a deep learning model deployed via a simple user interface.
    This intelligent deep learning system classifies waste images into categories like 
    <b>Plastic</b>, <b>Metal</b>, <b>Glass</b>, <b>Paper</b>, and <b>Organic</b>.  

    It helps automate recycling processes and supports smarter environmental decisions 
    through AI-driven waste segregation.
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    st.write("")

    st.subheader("🌍 Business Use Cases")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="card">
        🗑️ <b>Smart Recycling Bins</b><br><br>
        Smart Recycling Bins: Automatically sort waste into appropriate bins.
        </div>
        """, unsafe_allow_html=True)

        st.write("")

        st.markdown("""
        <div class="card">
        🏢 <b>Municipal Waste Management</b><br><br>
        Municipal Waste Management: Reduce manual sorting time and labor.
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card">
        🎓 <b>Educational Tools</b><br><br>
        Educational Tools: Teach proper segregation through visual tools.

        </div>
        """, unsafe_allow_html=True)

        st.write("")

        st.markdown("""
        <div class="card">
        📊 <b>Environmental Analytics</b><br><br>
        Environmental Analytics: Track waste composition and recycling trends.
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.markdown("---")
    
elif page == "📂 Dataset":

    import os
    import random
    from PIL import Image

    # -------------------- CSS -------------------- #
    st.markdown("""
    <style>
    .dataset-title {
        text-align:center;
        font-size:42px;
        font-weight:800;
        color:#1b5e20;
    }

    .dataset-sub {
        text-align:center;
        color:#555;
        margin-bottom:30px;
    }

    .stat-card {
        background:white;
        padding:20px;
        border-radius:15px;
        text-align:center;
        box-shadow:0 4px 15px rgba(0,0,0,0.08);
    }

    .image-card {
        background:white;
        padding:15px;
        border-radius:15px;
        box-shadow:0 4px 15px rgba(0,0,0,0.08);
        text-align:center;
        margin-bottom:20px;
    }

    .class-badge {
        font-weight:600;
        color:#2e7d32;
        font-size:16px;
    }
    </style>
    """, unsafe_allow_html=True)

    # -------------------- HEADER -------------------- #
    st.markdown('<p class="dataset-title">📂 Dataset Overview</p>', unsafe_allow_html=True)
    st.markdown('<p class="dataset-sub">Understanding the training data behind the AI model</p>', unsafe_allow_html=True)

    # -------------------- DATASET STATS -------------------- #
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="stat-card">
        📦 <h3>6 Classes</h3>
        cardboard, glass, metal, paper, plastic, trash
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="stat-card">
        🖼️ <h3>2,503 Images</h3>
        Total dataset size
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="stat-card">
        📚 <h3>Kaggle</h3>
        Garbage Classification Dataset
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.write("")

    st.subheader("🖼️ Sample Images from Each Class")

    data_dir = r"C:\Users\ADMIN\Desktop\Garbage classification dataset"

    CLASS_NAMES = [
        "cardboard",
        "glass",
        "metal",
        "paper",
        "plastic",
        "trash"
    ]

    cols = st.columns(3)
    col_index = 0

    for class_name in CLASS_NAMES:

        class_path = os.path.join(data_dir, class_name)

        if not os.path.exists(class_path):
            continue

        images = [
            img for img in os.listdir(class_path)
            if img.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        if len(images) == 0:
            continue

        img_name = random.choice(images)
        img_path = os.path.join(class_path, img_name)
        image = Image.open(img_path)

        with cols[col_index]:
            st.markdown('<div class="image-card">', unsafe_allow_html=True)
            st.image(image, width="stretch")
            st.markdown(
                f'<p class="class-badge">♻️ {class_name.capitalize()}</p>',
                unsafe_allow_html=True
            )
            st.markdown('</div>', unsafe_allow_html=True)

        col_index += 1
        if col_index == 3:
            cols = st.columns(3)
            col_index = 0

    st.write("")
    st.markdown("---")
    st.markdown("<center>Dataset Visualization • Deep Learning Pipeline</center>", unsafe_allow_html=True)

elif page == "🤖 Predict":

    CLASS_NAMES = [
        "cardboard",
        "glass",
        "metal",
        "paper",
        "plastic",
        "trash"
    ]

    st.title("🤖 Garbage Classification")

    uploaded_file = st.file_uploader(
        "Upload an image of garbage",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", width=300)

        if st.button("Predict"):
            with st.spinner("Classifying image..."):

                img = image.resize((224, 224))
                img_array = np.array(img)
                img_array = preprocess_input(img_array)
                img_array = np.expand_dims(img_array, axis=0)

                preds = model.predict(img_array)

                prediction = CLASS_NAMES[np.argmax(preds)]
                confidence = float(np.max(preds))
                probs = preds[0]

                st.subheader("Prediction Result")
                st.success(f" **{prediction.upper()}**")
                st.info(f"Confidence: **{confidence:.2%}**")

                st.subheader("Class Probabilities")
                st.bar_chart({
                    CLASS_NAMES[i]: float(probs[i])
                    for i in range(len(CLASS_NAMES))
                })


elif page == "📊 EDA":

    import os
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image
    import streamlit as st
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    data_dir = r"C:\Users\ADMIN\Desktop\Garbage classification dataset"

    st.markdown("## 🌿 Exploratory Data Analysis")

    classes = [cls for cls in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, cls))]

    # KPI CARDS

    total_images = 2527
    total_categories = len(classes)

    col1, col2 = st.columns(2)
    col1.metric("📦 Total Images", total_images)
    col2.metric("♻️ Total Categories", total_categories)

    st.markdown("---")

    # CLASS DISTRIBUTION

    class_counts = {
        cls: len(os.listdir(os.path.join(data_dir, cls)))
        for cls in classes
    }

    st.subheader("📊 Images per Category")

    fig1 = plt.figure()
    plt.bar(class_counts.keys(), class_counts.values(), color="#66bb6a")
    plt.xticks(rotation=45)
    plt.ylabel("Number of Images")
    plt.title("Class Distribution")
    st.pyplot(fig1)

    st.markdown("---")

    # UNIQUE IMAGE SIZE 

    sizes = set()
    for cls in classes:
        folder = os.path.join(data_dir, cls)
        img_name = os.listdir(folder)[0]
        img = Image.open(os.path.join(folder, img_name))
        sizes.add(img.size)

    st.subheader("🖼️ Image Size Analysis")
    st.write(f"Unique image sizes found: {sizes}")

    st.markdown("---")

    # SINGLE IMAGE PIXEL HISTOGRAM 

    st.subheader("🌑 Pixel Intensity Distribution (Single Image)")

    sample_path = os.path.join(data_dir, classes[0], os.listdir(os.path.join(data_dir, classes[0]))[0])
    img = cv2.imread(sample_path)
    img = cv2.resize(img, (128, 128))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    fig2 = plt.figure()
    plt.hist(gray.ravel(), bins=256, range=(0, 256), color="#2e7d32")
    plt.title("Pixel Intensity Distribution")
    plt.xlabel("Intensity (0-255)")
    plt.ylabel("Count")
    st.pyplot(fig2)

    st.markdown("---")

    st.markdown("""
    <div style="
        background-color:#e8f5e9;
        padding:20px;
        border-radius:15px;
        border-left:6px solid #2e7d32;
        font-size:15px;
    ">

    🌑 <b>Brightness Profile:</b>  
    Most pixel intensities lie between <b>130–210</b>, indicating that the image is well-lit and free from extreme dark or overexposed regions.

    📊 <b>Multiple Peaks:</b>  
    The presence of multiple peaks suggests varied regions within the image, such as shadows, textures, and reflective areas.

    </div>
    """, unsafe_allow_html=True)


    # PIXEL INTENSITY PER CLASS

    @st.cache_data
    def compute_pixel_histograms(data_dir, classes):
        results = {}

        for cls in classes:
            folder = os.path.join(data_dir, cls)
            intensities = []

            for img_name in os.listdir(folder)[:8]:  # small sample
                try:
                    img = cv2.imread(os.path.join(folder, img_name))
                    img = cv2.resize(img, (128, 128))
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    intensities.extend(gray.ravel())
                except:
                    pass

            results[cls] = intensities

        return results


    st.subheader("🌑 Pixel Intensity per Class")

    pixel_data = compute_pixel_histograms(data_dir, classes)

    fig3 = plt.figure(figsize=(14, 8))

    for idx, cls in enumerate(classes):
        plt.subplot(2, 3, idx + 1)
        plt.hist(pixel_data[cls], bins=100, range=(0, 256), color="#81c784")
        plt.title(cls)
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()
    st.pyplot(fig3)

    

    st.markdown("""
    <div style="
        background-color:#e8f5e9;
        padding:20px;
        border-radius:15px;
        border-left:6px solid #2e7d32;
        font-size:15px;
    ">

    🌑 <b>Brightness Trend:</b>  
    Most classes show pixel intensity peaks between <b>150–230</b>, indicating that the dataset predominantly contains well-lit images.

    📊 <b>Class Similarity:</b>  
    The distributions across classes appear relatively similar. This suggests that brightness alone is <b>not a strong discriminative feature</b> for classification.

    </div>
    """, unsafe_allow_html=True)


    st.markdown("---")

    # RGB DISTRIBUTION

    @st.cache_data
    def compute_rgb_histograms(data_dir, classes):
        results = {}

        for cls in classes:
            folder = os.path.join(data_dir, cls)
            reds, greens, blues = [], [], []

            for img_name in os.listdir(folder)[:8]:
                try:
                    img = cv2.imread(os.path.join(folder, img_name))
                    img = cv2.resize(img, (128, 128))
                    b, g, r = cv2.split(img)
                    reds.extend(r.ravel())
                    greens.extend(g.ravel())
                    blues.extend(b.ravel())
                except:
                    pass

            results[cls] = (reds, greens, blues)

        return results


    st.subheader("🌈 RGB Distribution per Class")

    rgb_data = compute_rgb_histograms(data_dir, classes)

    fig4 = plt.figure(figsize=(14, 8))

    for idx, cls in enumerate(classes):
        r, g, b = rgb_data[cls]

        plt.subplot(2, 3, idx + 1)
        plt.hist(r, bins=100, color='red', alpha=0.3)
        plt.hist(g, bins=100, color='green', alpha=0.3)
        plt.hist(b, bins=100, color='blue', alpha=0.3)
        plt.title(cls)
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()
    st.pyplot(fig4)

    

    st.markdown("""
    <div style="
        background-color:#e8f5e9;
        padding:20px;
        border-radius:15px;
        border-left:6px solid #2e7d32;
        font-size:15px;
    ">

    🌈 <b>Overall Brightness:</b>  
    Most RGB channels peak in the higher intensity range (150–230), confirming that the dataset consists primarily of well-lit images.

    🎨 <b>Color Overlap:</b>  
    There is significant overlap between Red, Green, and Blue distributions across classes. This suggests that color intensity alone is <b>not sufficient for classification</b>.

    📦 <b>Material-Specific Patterns:</b>  
    <ul>
    <li><b>Cardboard & Paper:</b> Stronger red and green components (brown/yellow tones).</li>
    <li><b>Glass & Metal:</b> Balanced RGB peaks due to reflective surfaces.</li>
    <li><b>Plastic:</b> Wide color spread due to varied object colors.</li>
    <li><b>Trash:</b> More irregular distribution reflecting mixed materials.</li>
    </ul>

    
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # DATA AUGMENTATION

    st.subheader("🔄 Data Augmentation Example")

    train_gen = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True
    )

    class_name = classes[0]
    img_name = os.listdir(os.path.join(data_dir, class_name))[0]
    img_path = os.path.join(data_dir, class_name, img_name)

    original = cv2.imread(img_path)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(original, (224, 224))

    aug_iter = train_gen.flow(np.expand_dims(resized, 0), batch_size=1)
    aug_images = [next(aug_iter)[0].astype(np.uint8) for _ in range(3)]

    fig5 = plt.figure(figsize=(14, 4))

    plt.subplot(1, 4, 1)
    plt.imshow(resized)
    plt.title("Original")
    plt.axis("off")

    for i in range(3):
        plt.subplot(1, 4, i + 2)
        plt.imshow(aug_images[i])
        plt.title(f"Aug {i+1}")
        plt.axis("off")

    plt.tight_layout()
    st.pyplot(fig5)




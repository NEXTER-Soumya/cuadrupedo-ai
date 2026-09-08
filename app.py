import os
from html import escape

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "cat_dog_final.tflite",
)

IMG_SIZE = (160, 160)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Cuadrúpedo AI",
    page_icon="🐾",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.html("""
<style>

:root {
    --bg: #070b12;
    --surface: #111722;
    --surface-2: #171e2b;
    --border: rgba(255,255,255,0.08);
    --text: #f5f7fb;
    --muted: #8c97a9;
    --blue: #5aa9ff;
    --pink: #ff719d;
}

/* ---------- Main page ---------- */

.stApp {
    background:
        radial-gradient(
            circle at 15% 5%,
            rgba(70, 130, 255, 0.15),
            transparent 28%
        ),
        radial-gradient(
            circle at 90% 8%,
            rgba(255, 90, 160, 0.12),
            transparent 25%
        ),
        var(--bg);
}

.block-container {
    max-width: 860px;
    padding: 1.5rem 1rem 3rem 1rem;
}

/* ---------- Hero ---------- */

.hero {
    text-align: center;
    margin: 0.4rem 0 1.5rem 0;
}

.hero-icon {
    font-size: 2.8rem;
    line-height: 1;
    margin-bottom: 0.35rem;
}

.hero-title {
    margin: 0;
    font-size: clamp(2.2rem, 8vw, 4rem);
    line-height: 1.05;
    font-weight: 900;
    letter-spacing: -0.055em;

    background:
        linear-gradient(
            135deg,
            #ffffff 0%,
            #8ec6ff 50%,
            #ff8fb4 100%
        );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    margin-top: 0.65rem;
    color: var(--muted);
    font-size: 0.98rem;
}

/* ---------- Cards ---------- */

.card {
    background: rgba(17, 23, 34, 0.86);
    border: 1px solid var(--border);
    border-radius: 22px;
    padding: 1.15rem;
    margin: 0.9rem 0;
    box-shadow:
        0 18px 45px rgba(0,0,0,0.20);
}

.card-title {
    color: var(--text);
    font-weight: 800;
    font-size: 1.05rem;
    margin-bottom: 0.2rem;
}

.card-subtitle {
    color: var(--muted);
    font-size: 0.88rem;
    line-height: 1.45;
}

/* ---------- Upload ---------- */

.upload-note {
    color: #929daf;
    font-size: 0.83rem;
    margin-top: 0.5rem;
    text-align: center;
}

/* ---------- Result ---------- */

.result-card {
    background:
        linear-gradient(
            145deg,
            rgba(255,255,255,0.055),
            rgba(255,255,255,0.018)
        );

    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 1.15rem;
    margin-top: 0.85rem;
}

.result-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 1rem;
}

.result-label {
    color: #788395;
    font-size: 0.72rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.14em;
}

.result-confidence {
    color: #aeb8c7;
    font-size: 0.82rem;
    white-space: nowrap;
}

.result-animal {
    text-align: center;
    font-size: clamp(2.4rem, 11vw, 4.2rem);
    font-weight: 900;
    letter-spacing: -0.05em;
    line-height: 1;
    margin: 0.55rem 0 0.7rem;
}

.confidence-track {
    width: 100%;
    height: 10px;
    border-radius: 999px;
    background: #202838;
    overflow: hidden;
}

.confidence-fill {
    height: 100%;
    border-radius: 999px;
}

/* ---------- Stats ---------- */

.stats {
    display: flex;
    justify-content: center;
    flex-wrap: wrap;
    gap: 0.55rem;
    margin: 0.8rem 0 1rem;
}

.pill {
    padding: 0.42rem 0.72rem;
    border-radius: 999px;
    background: rgba(255,255,255,0.045);
    border: 1px solid var(--border);
    color: #aab4c3;
    font-size: 0.76rem;
}

/* ---------- Model information ---------- */

.info-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.6rem;
    margin-top: 0.9rem;
}

.info-box {
    text-align: center;
    background: rgba(255,255,255,0.03);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 0.85rem 0.5rem;
}

.info-value {
    color: var(--text);
    font-size: 1.12rem;
    font-weight: 800;
}

.info-label {
    margin-top: 0.2rem;
    color: #788395;
    font-size: 0.68rem;
}

/* ---------- Footer ---------- */

.footer {
    text-align: center;
    color: #667184;
    font-size: 0.74rem;
    line-height: 1.6;
    margin-top: 1.5rem;
}

/* ---------- Mobile ---------- */

@media (max-width: 600px) {

    .block-container {
        padding: 0.9rem 0.7rem 2rem 0.7rem;
    }

    .hero {
        margin-bottom: 1rem;
    }

    .card {
        border-radius: 18px;
        padding: 0.95rem;
    }

    .info-grid {
        grid-template-columns: 1fr;
    }

    .result-card {
        border-radius: 18px;
    }

    .result-top {
        align-items: flex-start;
    }
}

</style>
""")


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(
        model_path=MODEL_PATH
    )

    interpreter.allocate_tensors()

    return interpreter


try:
    interpreter = load_model()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

except Exception as error:
    st.error(f"Could not load the model: {error}")
    st.stop()


# ============================================================
# PREDICTION
# ============================================================

def predict_image(image: Image.Image):
    """
    The TFLite model expects:
        shape = (1, 160, 160, 3)
        dtype = float32

    Output:
        probability of DOG
    """

    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)

    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    interpreter.set_tensor(
        input_details[0]["index"],
        image_array
    )

    interpreter.invoke()

    prediction = float(
        interpreter.get_tensor(
            output_details[0]["index"]
        )[0][0]
    )

    if prediction >= 0.5:
        animal = "DOG"
        confidence = prediction
        emoji = "🐶"
        accent = "#5aa9ff"
    else:
        animal = "CAT"
        confidence = 1.0 - prediction
        emoji = "🐱"
        accent = "#ff719d"

    return animal, confidence, emoji, accent


# ============================================================
# HERO
# ============================================================

st.html("""
<div class="hero">

    <div class="hero-icon">🐾</div>

    <div class="hero-title">
        Cuadrúpedo AI
    </div>

    <div class="hero-subtitle">
        Intelligent Cat & Dog Image Classifier
    </div>

</div>
""")


# ============================================================
# UPLOAD CARD
# ============================================================

st.html("""
<div class="card">

    <div class="card-title">
        Upload your images
    </div>

    <div class="card-subtitle">
        Drag & drop images here or browse your device.
        You can upload one image or several at once.
    </div>

</div>
""")


uploaded_files = st.file_uploader(
    "Select images",
    type=[
        "jpg",
        "jpeg",
        "png",
        "bmp",
        "webp",
    ],
    accept_multiple_files=True,
    label_visibility="collapsed",
)


st.html("""
<div class="upload-note">
    JPG · JPEG · PNG · BMP · WEBP
</div>
""")


# ============================================================
# PREDICTIONS
# ============================================================

if uploaded_files:

    st.html(
        f"""
        <div class="stats">

            <div class="pill">
                📷 {len(uploaded_files)}
                {"image" if len(uploaded_files) == 1 else "images"}
            </div>

            <div class="pill">
                🧠 MobileNetV2
            </div>

            <div class="pill">
                ⚡ TFLite
            </div>

        </div>
        """
    )

    results = []

    for uploaded_file in uploaded_files:

        try:

            image = Image.open(
                uploaded_file
            )

            animal, confidence, emoji, accent = (
                predict_image(image)
            )

            percent = confidence * 100

            results.append({
                "filename": uploaded_file.name,
                "prediction": animal,
                "confidence": confidence,
            })

            # ------------------------------------------------
            # Image
            # ------------------------------------------------

            st.html("""
            <div class="card">
            """)

            st.image(
                image,
                width="stretch"
            )

            st.html(
                f"""
                <div class="upload-note">
                    {escape(uploaded_file.name)}
                </div>
                """
            )


            # ------------------------------------------------
            # Result
            # ------------------------------------------------

            st.html(
                f"""
                <div class="result-card">

                    <div class="result-top">

                        <div class="result-label">
                            Prediction
                        </div>

                        <div class="result-confidence">
                            {percent:.2f}% confidence
                        </div>

                    </div>

                    <div
                        class="result-animal"
                        style="color:{accent};"
                    >
                        {emoji} {animal}
                    </div>

                    <div class="confidence-track">

                        <div
                            class="confidence-fill"
                            style="
                                width:{percent:.2f}%;
                                background:{accent};
                                box-shadow:
                                    0 0 18px
                                    {accent}66;
                            "
                        ></div>

                    </div>

                </div>

                </div>
                """
            )

        except Exception as error:

            st.error(
                f"Could not process "
                f"{uploaded_file.name}: {error}"
            )


    # ========================================================
    # SUMMARY
    # ========================================================

    if len(results) > 1:

        st.html("""
        <div class="card">

            <div class="card-title">
                Prediction Summary
            </div>

            <div class="card-subtitle">
                Results for all uploaded images.
            </div>

        </div>
        """)

        for result in results:

            icon = (
                "🐶"
                if result["prediction"] == "DOG"
                else "🐱"
            )

            filename = escape(
                result["filename"]
            )

            percent = (
                result["confidence"] * 100
            )

            st.html(
                f"""
                <div class="card"
                     style="padding:0.85rem 1rem;">

                    <div style="
                        display:flex;
                        justify-content:space-between;
                        align-items:center;
                        gap:1rem;
                    ">

                        <div style="
                            color:#e9edf4;
                            overflow:hidden;
                            text-overflow:ellipsis;
                            white-space:nowrap;
                        ">
                            {icon} {filename}
                        </div>

                        <div style="
                            color:#aeb8c7;
                            white-space:nowrap;
                        ">
                            {percent:.2f}%
                        </div>

                    </div>

                </div>
                """
            )


# ============================================================
# ABOUT THE MODEL
# ============================================================

st.html("""
<div class="card">

    <div class="card-title">
        About the model
    </div>

    <div class="card-subtitle">
        A fine-tuned MobileNetV2 binary image classifier
        trained to distinguish cats from dogs.
    </div>

    <div class="info-grid">

        <div class="info-box">

            <div class="info-value">
                98.68%
            </div>

            <div class="info-label">
                Validation Accuracy
            </div>

        </div>

        <div class="info-box">

            <div class="info-value">
                24,998
            </div>

            <div class="info-label">
                Images
            </div>

        </div>

        <div class="info-box">

            <div class="info-value">
                160×160
            </div>

            <div class="info-label">
                Input Size
            </div>

        </div>

    </div>

</div>
""")


# ============================================================
# FOOTER
# ============================================================

st.html("""
<div class="footer">
    <b>Cuadrúpedo AI</b><br>
    MobileNetV2 · TensorFlow Lite · Machine Learning
</div>
""")
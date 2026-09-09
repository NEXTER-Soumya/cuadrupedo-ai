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

# Maximum display size for uploaded images.
# This does NOT change the image used for prediction.
PREVIEW_WIDTH = 320


# ============================================================
# PAGE CONFIGURATION
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


/* =========================================================
   STREAMLIT HEADER
   ========================================================= */

header[data-testid="stHeader"] {
    background: transparent !important;
    box-shadow: none !important;
}

header[data-testid="stHeader"]::before {
    background: transparent !important;
}

div[data-testid="stToolbar"] {
    background: transparent !important;
}

div[data-testid="stDecoration"] {
    background: transparent !important;
}


/* =========================================================
   APP BACKGROUND
   ========================================================= */

.stApp {
    min-height: 100vh;

    background:
        radial-gradient(
            circle at 12% 2%,
            rgba(70, 130, 255, 0.18),
            transparent 30%
        ),
        radial-gradient(
            circle at 90% 5%,
            rgba(255, 90, 160, 0.14),
            transparent 27%
        ),
        radial-gradient(
            circle at 50% 100%,
            rgba(50, 100, 220, 0.05),
            transparent 35%
        ),
        #070b12;
}


/* =========================================================
   MAIN CONTENT
   ========================================================= */

.block-container {
    max-width: 860px;
    padding: 3.8rem 1rem 3rem 1rem;
}


/* =========================================================
   HERO
   ========================================================= */

.hero {
    text-align: center;
    padding: 0.4rem 0 1.6rem 0;
}

.hero-icon {
    font-size: 2.7rem;
    line-height: 1;
    margin-bottom: 0.65rem;
}

.hero-title {
    margin: 0;

    font-size: clamp(
        2.15rem,
        8vw,
        4rem
    );

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
    margin-top: 0.7rem;

    color: var(--muted);

    font-size: 0.98rem;

    line-height: 1.5;
}


/* =========================================================
   CARDS
   ========================================================= */

.card {
    background: rgba(17, 23, 34, 0.84);

    border:
        1px solid
        var(--border);

    border-radius: 22px;

    padding: 1.15rem;

    margin: 0.9rem 0;

    box-shadow:
        0 18px 45px
        rgba(0, 0, 0, 0.20);

    backdrop-filter: blur(14px);
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


/* =========================================================
   UPLOADER
   ========================================================= */

[data-testid="stFileUploaderDropzone"] {
    background:
        rgba(255,255,255,0.025);

    border:
        1.5px dashed
        rgba(130,177,255,0.42);

    border-radius: 20px;

    padding: 1.15rem 0.7rem;

    transition:
        background 0.2s ease,
        border-color 0.2s ease;
}

[data-testid="stFileUploaderDropzone"]:hover {
    background:
        rgba(80,130,255,0.055);

    border-color:
        rgba(130,177,255,0.80);
}

.upload-note {
    color: #929daf;

    font-size: 0.82rem;

    margin-top: 0.5rem;

    text-align: center;
}


/* =========================================================
   IMAGE PREVIEW
   ========================================================= */

.image-container {
    width: 100%;
    display: flex;
    justify-content: center;
    align-items: center;
}

.image-container img {
    max-width: 320px !important;
    width: auto !important;
    height: auto !important;
    max-height: 320px !important;
    object-fit: contain !important;
    border-radius: 16px !important;
}


/* =========================================================
   RESULT CARD
   ========================================================= */

.result-card {

    background:
        linear-gradient(
            145deg,
            rgba(255,255,255,0.055),
            rgba(255,255,255,0.018)
        );

    border:
        1px solid
        rgba(255,255,255,0.065);

    border-radius: 20px;

    padding: 1.15rem;

    margin-top: 0.85rem;
}

.result-top {

    display: flex;

    justify-content:
        space-between;

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

    font-size: clamp(
        2.4rem,
        11vw,
        4.2rem
    );

    font-weight: 900;

    letter-spacing: -0.05em;

    line-height: 1;

    margin:
        0.55rem 0
        0.75rem;
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


/* =========================================================
   STAT PILLS
   ========================================================= */

.stats {

    display: flex;

    justify-content: center;

    flex-wrap: wrap;

    gap: 0.55rem;

    margin:
        0.8rem 0
        1rem;
}

.pill {

    padding:
        0.42rem
        0.72rem;

    border-radius: 999px;

    background:
        rgba(255,255,255,0.045);

    border:
        1px solid
        var(--border);

    color: #aab4c3;

    font-size: 0.76rem;
}


/* =========================================================
   MODEL INFORMATION
   ========================================================= */

.info-grid {

    display: grid;

    grid-template-columns:
        repeat(3, 1fr);

    gap: 0.6rem;

    margin-top: 0.9rem;
}

.info-box {

    text-align: center;

    background:
        rgba(255,255,255,0.03);

    border:
        1px solid
        var(--border);

    border-radius: 16px;

    padding:
        0.85rem
        0.5rem;
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


/* =========================================================
   FOOTER
   ========================================================= */

.footer {

    text-align: center;

    color: #667184;

    font-size: 0.74rem;

    line-height: 1.6;

    margin-top: 1.5rem;
}


/* =========================================================
   MOBILE
   ========================================================= */

@media (max-width: 600px) {

    .block-container {
        padding:
            3.2rem
            0.7rem
            2rem
            0.7rem;
    }

    .hero {
        padding-top: 0.2rem;
        margin-bottom: 1rem;
    }

    .hero-icon {
        font-size: 2.4rem;
        margin-bottom: 0.55rem;
    }

    .hero-subtitle {
        font-size: 0.9rem;
    }

    .card {
        border-radius: 18px;
        padding: 0.95rem;
    }

    .result-card {
        border-radius: 18px;
    }

    .info-grid {
        grid-template-columns: 1fr;
    }

    .result-top {
        align-items: flex-start;
    }

    .image-container img {
        max-width: 280px !important;
        max-height: 280px !important;
    }

}


/* =========================================================
   VERY SMALL PHONES
   ========================================================= */

@media (max-width: 380px) {

    .block-container {
        padding:
            3rem
            0.55rem
            1.5rem
            0.55rem;
    }

    .hero-title {
        font-size: 2.2rem;
    }

    .result-animal {
        font-size: 2.6rem;
    }

    .image-container img {
        max-width: 250px !important;
        max-height: 250px !important;
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

    input_details = (
        interpreter.get_input_details()
    )

    output_details = (
        interpreter.get_output_details()
    )

except Exception as error:

    st.error(
        f"Could not load the model: {error}"
    )

    st.stop()


# ============================================================
# PREDICTION
# ============================================================

def predict_image(image: Image.Image):

    # Convert to RGB
    image = image.convert("RGB")

    # Resize for the neural network
    image = image.resize(IMG_SIZE)

    # Convert to float32
    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # Send image to TFLite
    interpreter.set_tensor(
        input_details[0]["index"],
        image_array
    )

    # Run inference
    interpreter.invoke()

    # Read output
    prediction = float(
        interpreter.get_tensor(
            output_details[0]["index"]
        )[0][0]
    )

    # Model output = probability of Dog

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

    return (
        animal,
        confidence,
        emoji,
        accent
    )


# ============================================================
# HERO
# ============================================================

st.html("""
<div class="hero">

    <div class="hero-icon">
        🐾
    </div>

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


# ============================================================
# FILE UPLOADER
# ============================================================

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
                {
                    "image"
                    if len(uploaded_files) == 1
                    else "images"
                }
            </div>

            <div class="pill">
                🧠 MobileNetV2
            </div>

            <div class="pill">
                ⚡ TensorFlow Lite
            </div>

        </div>
        """
    )

    results = []


    # ========================================================
    # EACH IMAGE
    # ========================================================

    for uploaded_file in uploaded_files:

        try:

            image = Image.open(
                uploaded_file
            ).convert("RGB")


            # ------------------------------------------------
            # Prediction uses resized copy internally.
            # Original image remains untouched for display.
            # ------------------------------------------------

            animal, confidence, emoji, accent = (
                predict_image(image)
            )

            percent = confidence * 100


            results.append({
                "filename":
                    uploaded_file.name,

                "prediction":
                    animal,

                "confidence":
                    confidence,
            })


            # ------------------------------------------------
            # Image card
            # ------------------------------------------------

            st.html("""
            <div class="card">

                <div class="image-container">
            """)

            # Display a thumbnail only.
            # The original image is NOT modified for prediction.

            preview = image.copy()
            preview.thumbnail(
                (PREVIEW_WIDTH, PREVIEW_WIDTH),
                Image.Resampling.LANCZOS
            )

            st.image(
                preview,
                width=PREVIEW_WIDTH
            )

            st.html(
                f"""
                    <div class="upload-note">
                        {escape(uploaded_file.name)}
                    </div>
                </div>
                """
            )


            # ------------------------------------------------
            # Result card
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
                        style="
                            color:{accent};
                        "
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
    # BULK SUMMARY
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
                     style="
                        padding:
                            0.85rem
                            1rem;
                     ">

                    <div style="
                        display:flex;
                        justify-content:
                            space-between;
                        align-items:
                            center;
                        gap:1rem;
                    ">

                        <div style="
                            color:#e9edf4;
                            overflow:hidden;
                            text-overflow:
                                ellipsis;
                            white-space:
                                nowrap;
                        ">
                            {icon} {filename}
                        </div>

                        <div style="
                            color:#aeb8c7;
                            white-space:
                                nowrap;
                        ">
                            {percent:.2f}%
                        </div>

                    </div>

                </div>
                """
            )


# ============================================================
# MODEL INFORMATION
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

    <b>Cuadrúpedo AI</b>
    <br>

    MobileNetV2 · TensorFlow Lite · Machine Learning

</div>
""")

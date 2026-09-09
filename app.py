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

# Display size of uploaded image
PREVIEW_SIZE = 320


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
   APPLICATION BACKGROUND
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
    max-width: 900px;
    padding: 3.8rem 1rem 3rem 1rem;
}


/* =========================================================
   HERO
   ========================================================= */

.hero {
    text-align: center;
    padding: 0.4rem 0 1.7rem 0;
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
   GENERAL CARD
   ========================================================= */

.card {
    background:
        rgba(17, 23, 34, 0.84);

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

    padding:
        1.15rem
        0.7rem;

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
   IMAGE + RESULT ROW
   ========================================================= */

.prediction-row {
    display: flex;

    align-items: stretch;

    justify-content: center;

    gap: 1rem;

    margin: 1rem 0;
}


/* =========================================================
   IMAGE PANEL
   ========================================================= */

.image-panel {

    flex: 1;

    min-width: 0;

    min-height: 320px;

    display: flex;

    justify-content: center;

    align-items: center;

    background:
        rgba(17, 23, 34, 0.84);

    border:
        1px solid
        var(--border);

    border-radius: 22px;

    padding: 1rem;

    box-shadow:
        0 18px 45px
        rgba(0,0,0,0.20);
}

.image-panel img {

    width: auto !important;

    height: auto !important;

    max-width: 320px !important;

    max-height: 320px !important;

    object-fit: contain !important;

    border-radius: 16px !important;
}


/* =========================================================
   RESULT PANEL
   ========================================================= */

.result-panel {

    flex: 1;

    min-width: 0;

    min-height: 320px;

    aspect-ratio: 1 / 1;

    max-width: 320px;

    display: flex;

    flex-direction: column;

    justify-content: space-between;

    background:
        linear-gradient(
            145deg,
            rgba(255,255,255,0.055),
            rgba(255,255,255,0.018)
        );

    border:
        1px solid
        rgba(255,255,255,0.07);

    border-radius: 22px;

    padding: 1.35rem;

    box-shadow:
        0 18px 45px
        rgba(0,0,0,0.22);
}


/* =========================================================
   RESULT PANEL CONTENT
   ========================================================= */

.result-header {

    display: flex;

    justify-content:
        space-between;

    align-items:
        center;

    gap: 0.5rem;
}

.result-label {

    color: #788395;

    font-size: 0.72rem;

    font-weight: 800;

    text-transform: uppercase;

    letter-spacing: 0.14em;
}

.result-confidence-small {

    color: #aeb8c7;

    font-size: 0.78rem;

    white-space: nowrap;
}

.result-center {

    display: flex;

    flex-direction: column;

    justify-content: center;

    align-items: center;

    flex: 1;
}

.result-emoji {

    font-size: 4.1rem;

    line-height: 1;

    margin-bottom: 0.5rem;
}

.result-animal {

    font-size: clamp(
        2.4rem,
        8vw,
        3.8rem
    );

    line-height: 1;

    font-weight: 900;

    letter-spacing: -0.05em;
}

.result-description {

    color: #8994a5;

    font-size: 0.83rem;

    margin-top: 0.55rem;

    text-align: center;
}


/* =========================================================
   CONFIDENCE
   ========================================================= */

.confidence-section {

    width: 100%;
}

.confidence-number {

    text-align: center;

    font-size: 0.86rem;

    color: #aeb8c7;

    margin-bottom: 0.55rem;
}

.confidence-track {

    width: 100%;

    height: 11px;

    border-radius: 999px;

    background:
        #202838;

    overflow: hidden;
}

.confidence-fill {

    height: 100%;

    border-radius: 999px;
}


/* =========================================================
   FILE NAME
   ========================================================= */

.filename {

    text-align: center;

    color: #7f8a9b;

    font-size: 0.78rem;

    margin-top: 0.65rem;

    overflow: hidden;

    text-overflow: ellipsis;

    white-space: nowrap;
}


/* =========================================================
   STATS
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
   SECTION TITLE
   ========================================================= */

.section-title {

    font-size: 1.25rem;

    font-weight: 800;

    margin:
        1.6rem 0
        0.5rem 0;
}

.section-description {

    color: #808b9d;

    font-size: 0.88rem;

    margin-bottom: 0.8rem;
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

@media (max-width: 700px) {

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


    /* Stack image and result on phones */

    .prediction-row {

        flex-direction: column;

        align-items: center;

        gap: 0.9rem;
    }


    .image-panel {

        width: 100%;

        min-height: auto;

        padding: 0.8rem;
    }

    .image-panel img {

        max-width: 280px !important;

        max-height: 280px !important;
    }


    .result-panel {

        width: 100%;

        max-width: 320px;

        min-height: 280px;

        aspect-ratio: 1 / 1;
    }


    .card {

        border-radius: 18px;

        padding: 0.95rem;
    }


    .info-grid {

        grid-template-columns:
            1fr;
    }

}


/* =========================================================
   SMALL PHONES
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

    .image-panel img {

        max-width: 245px !important;

        max-height: 245px !important;
    }

    .result-panel {

        max-width: 280px;

        min-height: 280px;
    }

    .result-emoji {

        font-size: 3.5rem;
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
# PREDICTION FUNCTION
# ============================================================

def predict_image(image: Image.Image):
    """
    Prepare the image exactly as required by the
    trained TFLite model.

    Model input:
        1 × 160 × 160 × 3
        float32

    Model output:
        probability of DOG
    """

    image = image.convert("RGB")

    image = image.resize(
        IMG_SIZE
    )

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
    # PROCESS EACH IMAGE
    # ========================================================

    for uploaded_file in uploaded_files:

        try:

            # Load original image
            image = Image.open(
                uploaded_file
            ).convert("RGB")


            # ------------------------------------------------
            # Predict
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
            # Create small preview
            # ------------------------------------------------

            preview = image.copy()

            preview.thumbnail(
                (
                    PREVIEW_SIZE,
                    PREVIEW_SIZE
                ),
                Image.Resampling.LANCZOS
            )


            # =================================================
            # IMAGE + PREDICTION SIDE BY SIDE
            # =================================================

            st.html("""
            <div class="prediction-row">

                <div class="image-panel">
            """)


            # Display image inside left panel

            st.image(
                preview,
                width=PREVIEW_SIZE
            )


            st.html(
                f"""
                </div>

                <div class="result-panel">

                    <div class="result-header">

                        <div class="result-label">
                            Prediction
                        </div>

                        <div
                            class="result-confidence-small"
                        >
                            {percent:.2f}%
                        </div>

                    </div>


                    <div class="result-center">

                        <div class="result-emoji">
                            {emoji}
                        </div>

                        <div
                            class="result-animal"
                            style="
                                color:{accent};
                            "
                        >
                            {animal}
                        </div>

                        <div
                            class="result-description"
                        >
                            The model is
                            {percent:.2f}% confident
                            this is a {animal.lower()}.
                        </div>

                    </div>


                    <div class="confidence-section">

                        <div
                            class="confidence-number"
                        >
                            Confidence
                        </div>

                        <div
                            class="confidence-track"
                        >

                            <div
                                class="confidence-fill"
                                style="
                                    width:{percent:.2f}%;

                                    background:
                                        {accent};

                                    box-shadow:
                                        0 0 18px
                                        {accent}66;
                                "
                            ></div>

                        </div>

                    </div>

                </div>

            </div>

            <div class="filename">
                {escape(uploaded_file.name)}
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
        <div class="section-title">
            Prediction Summary
        </div>

        <div class="section-description">
            Results from all uploaded images.
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
                <div
                    class="card"
                    style="
                        padding:
                            0.85rem
                            1rem;
                    "
                >

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
                            {icon}
                            {filename}
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
# ABOUT THE MODEL
# ============================================================

st.html("""
<div class="section-title">
    About the model
</div>

<div class="section-description">
    A fine-tuned MobileNetV2 binary image classifier
    trained to distinguish cats from dogs.
</div>

<div class="card">

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
                Dataset Images
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

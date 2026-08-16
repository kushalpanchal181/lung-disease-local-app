import os
import json
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

from explainability import (
    make_gradcam_heatmap,
    make_scorecam_heatmap,
    make_layercam_heatmap,
    overlay_heatmap
)


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "model" / "lung_model.keras"
CLASS_NAMES_PATH = BASE_DIR / "model" / "class_names.json"
MODEL_REGISTRY_PATH = BASE_DIR / "model" / "model_registry.json"

RESULTS_DIR = BASE_DIR / "results"
PREDICTION_LOG_PATH = RESULTS_DIR / "prediction_logs.csv"
EVALUATION_METRICS_PATH = RESULTS_DIR / "evaluation_metrics.csv"
CLASSIFICATION_REPORT_PATH = RESULTS_DIR / "classification_report.txt"
VISUAL_COMPARISON_DIR = RESULTS_DIR / "visual_comparisons"
RESULTS_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# Page setup
# --------------------------------------------------

st.set_page_config(
    page_title="AI Lung Disease Detection",
    page_icon="🫁",
    layout="wide"
)

st.title("AI-Based Lung Disease Detection and MLOps Dashboard")
st.write(
    "This prototype demonstrates chest X-ray classification, explainability, "
    "prediction monitoring and MLOps traceability for academic evaluation."
)
st.warning(
    "This is an academic prototype only. It is not a medical diagnosis system "
    "and should not be used for real clinical decisions."
)


# --------------------------------------------------
# Load model, class names and registry
# --------------------------------------------------

@st.cache_resource
def load_trained_model():
    if not MODEL_PATH.exists():
        st.error("Model file not found. Please run train_model.py first.")
        return None

    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data
def load_class_names():
    if not CLASS_NAMES_PATH.exists():
        st.error("class_names.json not found. Please run train_model.py first.")
        return []

    with open(CLASS_NAMES_PATH, "r") as f:
        return json.load(f)


@st.cache_data
def load_model_registry():
    if not MODEL_REGISTRY_PATH.exists():
        return None

    with open(MODEL_REGISTRY_PATH, "r") as f:
        return json.load(f)


def get_git_commit():
    """
    Gets the Git commit hash from the deployment environment.
    Falls back to the local Git repository during development.
    """
    env_commit = os.getenv("GIT_COMMIT")

    if env_commit:
        return env_commit

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=BASE_DIR
        )
        return commit.decode("utf-8").strip()
    except Exception:
        return "Not available"


def apply_temperature_scaling(probabilities, temperature=1.5):
    """
    Prototype confidence calibration using temperature scaling.

    Note:
    This is a simple prototype calculation.
    Proper calibration should tune the temperature using validation data.
    """
    probabilities = np.array(probabilities, dtype=np.float32)
    probabilities = np.clip(probabilities, 1e-8, 1.0)

    logits = np.log(probabilities)
    scaled_logits = logits / temperature

    exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
    calibrated_probs = exp_logits / np.sum(exp_logits)

    return calibrated_probs


def log_prediction(
    predicted_class,
    confidence,
    calibrated_confidence,
    uncertainty,
    top_2_margin
):
    """
    Saves prediction monitoring information for MLOps evidence.
    """
    log_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": "MobileNetV2-CXR",
        "model_version": "v1.0.0",
        "predicted_class": predicted_class,
        "confidence": round(confidence, 2),
        "calibrated_confidence": round(calibrated_confidence, 2),
        "uncertainty": round(uncertainty, 2),
        "top_2_margin": round(top_2_margin, 2),
        "git_commit": get_git_commit()
    }

    log_df = pd.DataFrame([log_data])

    if PREDICTION_LOG_PATH.exists():
        log_df.to_csv(PREDICTION_LOG_PATH, mode="a", header=False, index=False)
    else:
        log_df.to_csv(PREDICTION_LOG_PATH, index=False)


model = load_trained_model()
class_names = load_class_names()


# --------------------------------------------------
# Main tabs
# --------------------------------------------------

prediction_tab, mlops_tab, visual_tab = st.tabs(
    ["Prediction Dashboard", "MLOps Dashboard", "Visual Comparisons"]
)


# --------------------------------------------------
# Prediction Dashboard
# --------------------------------------------------

with prediction_tab:
    st.header("Prediction Dashboard")

    uploaded_file = st.file_uploader(
        "Upload chest X-ray image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        st.subheader("Uploaded X-ray Image")
        st.image(image, caption="Original chest X-ray", width=420)

        if model is not None and class_names:
            # --------------------------------------
            # Preprocess image
            # --------------------------------------
            img = image.resize((224, 224))
            img_array = np.array(img).astype("float32")
            img_array = np.expand_dims(img_array, axis=0)

            # --------------------------------------
            # Prediction
            # --------------------------------------
            probabilities = model.predict(img_array, verbose=0)[0]

            predicted_index = int(np.argmax(probabilities))
            predicted_class = class_names[predicted_index]

            confidence = float(probabilities[predicted_index]) * 100

            calibrated_probabilities = apply_temperature_scaling(
                probabilities,
                temperature=1.5
            )

            calibrated_confidence = (
                float(calibrated_probabilities[predicted_index]) * 100
            )

            sorted_probabilities = np.sort(probabilities)[::-1]

            uncertainty = 100 - confidence

            if len(sorted_probabilities) > 1:
                top_2_margin = (
                    sorted_probabilities[0] - sorted_probabilities[1]
                ) * 100
            else:
                top_2_margin = 0

            log_prediction(
                predicted_class=predicted_class,
                confidence=confidence,
                calibrated_confidence=calibrated_confidence,
                uncertainty=uncertainty,
                top_2_margin=top_2_margin
            )

            # --------------------------------------
            # Dashboard metrics
            # --------------------------------------
            st.subheader("Prediction Result")

            col1, col2, col3, col4 = st.columns(4)

            col1.metric("Predicted Class", predicted_class)
            col2.metric("Confidence", f"{confidence:.2f}%")
            col3.metric("Uncertainty", f"{uncertainty:.2f}%")
            col4.metric("Top-2 Margin", f"{top_2_margin:.2f}%")

            col5, col6 = st.columns(2)

            col5.metric(
                "Calibrated Confidence",
                f"{calibrated_confidence:.2f}%"
            )

            if confidence < 70 or top_2_margin < 15:
                col6.warning("Prediction Status: Review Required")
            else:
                col6.success("Prediction Status: Confident")
                
            st.caption(
                "Calibrated confidence is shown as a prototype temperature-adjusted score. "
                "For final clinical-style calibration, the temperature should be tuned using validation data."
            )

            # --------------------------------------
            # Class probability table
            # --------------------------------------
            st.subheader("Class Probability Breakdown")

            results_df = pd.DataFrame({
                "Class": class_names,
                "Probability (%)": [
                    round(float(p) * 100, 2) for p in probabilities
                ],
                "Calibrated Probability (%)": [
                    round(float(p) * 100, 2) for p in calibrated_probabilities
                ]
            })

            results_df = results_df.sort_values(
                by="Probability (%)",
                ascending=False
            )

            st.dataframe(results_df, use_container_width=True)

            # --------------------------------------
            # Explainability heatmaps
            # --------------------------------------
            st.subheader("Explainability Heatmaps")

            st.caption(
                "These heatmaps show model-focused regions that contributed to the prediction. "
                "Blue indicates lower model attention, while yellow/red indicates stronger model attention. "
                "They are used for AI explainability only and do not confirm the clinically affected area."
            )

            threshold_value = st.slider(
                "Heatmap focus threshold",
                min_value=70,
                max_value=95,
                value=88,
                step=1,
                help="Higher value shows only the strongest attention area."
            )

            scorecam_channels = st.slider(
                "Score-CAM channels",
                min_value=8,
                max_value=48,
                value=24,
                step=4,
                help="Higher value gives more detail but is slower."
            )

            with st.spinner("Generating Grad-CAM and LayerCAM heatmaps..."):
                gradcam_heatmap = make_gradcam_heatmap(
                    img_array=img_array,
                    model=model,
                    predicted_index=predicted_index
                )

                layercam_heatmap = make_layercam_heatmap(
                    img_array=img_array,
                    model=model,
                    predicted_index=predicted_index
                )

                gradcam_image = overlay_heatmap(
                    original_image=image,
                    heatmap=gradcam_heatmap,
                    threshold_percentile=threshold_value
                )

                layercam_image = overlay_heatmap(
                    original_image=image,
                    heatmap=layercam_heatmap,
                    threshold_percentile=threshold_value
                )

            with st.spinner("Generating Score-CAM heatmap. This may take longer..."):
                scorecam_heatmap = make_scorecam_heatmap(
                    img_array=img_array,
                    model=model,
                    predicted_index=predicted_index,
                    top_n=scorecam_channels
                )

                scorecam_image = overlay_heatmap(
                    original_image=image,
                    heatmap=scorecam_heatmap,
                    threshold_percentile=threshold_value
                )

            heatmap_tab1, heatmap_tab2, heatmap_tab3 = st.tabs(
                ["Grad-CAM", "Score-CAM", "LayerCAM"]
            )

            with heatmap_tab1:
                col_a, col_b = st.columns(2)

                with col_a:
                    st.image(
                        image,
                        caption="Original X-ray",
                        use_container_width=True
                    )

                with col_b:
                    st.image(
                        gradcam_image,
                        caption="Grad-CAM attention map",
                        use_container_width=True
                    )

            with heatmap_tab2:
                col_a, col_b = st.columns(2)

                with col_a:
                    st.image(
                        image,
                        caption="Original X-ray",
                        use_container_width=True
                    )

                with col_b:
                    st.image(
                        scorecam_image,
                        caption="Score-CAM attention map",
                        use_container_width=True
                    )

            with heatmap_tab3:
                col_a, col_b = st.columns(2)

                with col_a:
                    st.image(
                        image,
                        caption="Original X-ray",
                        use_container_width=True
                    )

                with col_b:
                    st.image(
                        layercam_image,
                        caption="LayerCAM attention map",
                        use_container_width=True
                    )

            st.info(
                "The model chooses the class with the highest probability. "
                "Low confidence or low top-2 margin means the model is unsure "
                "and may need more training, better balanced data or external validation."
            )


# --------------------------------------------------
# MLOps Dashboard
# --------------------------------------------------

with mlops_tab:
    st.header("MLOps Dashboard")

    st.subheader("Model Registry and Versioning")

    registry = load_model_registry()

    if registry is None:
        st.warning(
            "model/model_registry.json not found. "
            "Create this file to show model registry information."
        )

        default_registry = {
            "model_name": "MobileNetV2-CXR",
            "model_version": "v1.0.0",
            "dataset_version": "CXR-4Class-v1",
            "framework": "TensorFlow/Keras",
            "base_model": "MobileNetV2",
            "deployment_environment": "Local Streamlit Prototype",
            "container_version": "Not containerised yet",
            "approval_status": "Academic prototype - not clinically approved",
            "deployment_date": "2026-06-27",
            "git_commit": get_git_commit(),
            "performance_baseline": {
                "validation_accuracy": "0.90",
                "metrics": [
                    "Accuracy",
                    "Precision",
                    "Recall",
                    "F1 Score",
                    "Confusion Matrix"
                ]
            }
        }

        registry = default_registry

    col1, col2, col3 = st.columns(3)

    col1.metric("Model Name", registry.get("model_name", "Not available"))
    col2.metric("Model Version", registry.get("model_version", "Not available"))
    col3.metric("Dataset Version", registry.get("dataset_version", "Not available"))

    col4, col5, col6 = st.columns(3)

    col4.metric("Framework", registry.get("framework", "Not available"))
    col5.metric("Base Model", registry.get("base_model", "Not available"))
    col6.metric("Environment", registry.get("deployment_environment", "Not available"))

    st.write("### Deployment Information")

    deployment_df = pd.DataFrame({
        "Field": [
            "Git Commit",
            "Container Version",
            "Deployment Date",
            "Approval Status"
        ],
        "Value": [
            get_git_commit(),
            registry.get("container_version", "Not containerised yet"),
            registry.get("deployment_date", "Not available"),
            registry.get("approval_status", "Academic prototype - not clinically approved")
        ]
    })

    st.table(deployment_df)

    st.write("### Training Classes")

    if "training_classes" in registry:
        st.write(", ".join(registry["training_classes"]))
    else:
        st.write(", ".join(class_names) if class_names else "Not available")

    st.write("### Performance Baseline")

    if "performance_baseline" in registry:
        st.json(registry["performance_baseline"])

    if EVALUATION_METRICS_PATH.exists():
        st.write("### Evaluation Metrics")
        metrics_df = pd.read_csv(EVALUATION_METRICS_PATH)
        st.dataframe(metrics_df, use_container_width=True)

    if CLASSIFICATION_REPORT_PATH.exists():
        with st.expander("Classification Report"):
            report_text = CLASSIFICATION_REPORT_PATH.read_text()
            st.text(report_text)

    st.write("### Prediction Monitoring Logs")

    if PREDICTION_LOG_PATH.exists():
        logs_df = pd.read_csv(PREDICTION_LOG_PATH)
        st.dataframe(logs_df.tail(20), use_container_width=True)

        st.caption(
            "This log supports MLOps monitoring by storing prediction time, "
            "model version, predicted class, confidence, uncertainty, "
            "top-2 margin and Git commit."
        )
    else:
        st.info("No prediction logs available yet. Upload an image first.")

    st.info(
        "This dashboard supports MLOps traceability by showing model version, "
        "dataset version, Git commit, deployment status, approval status, "
        "performance baseline and prediction monitoring logs."
    )
    
with visual_tab:
    st.header("Visual Comparison Results")

    st.write(
        "This section presents clear visual comparisons for the image analysis pipeline, "
        "including preprocessing, prediction outputs, baseline comparison, proposed method comparison "
        "and contextual comparison with a published study."
    )

    visual_files = [
        ("Sample Images Before and After Preprocessing", "01_preprocessing_comparison.png"),
        ("Input Images with Prediction Results", "02_input_prediction_comparison.png"),
        ("Evaluation Results: Baseline vs Proposed Method", "03_baseline_vs_proposed_metrics.png"),
        ("Before and After Applying Proposed AI Technique", "04_before_after_ai_technique.png"),
        ("Published Study vs Proposed Method", "05_published_vs_proposed_comparison.png")
    ]

    for title, filename in visual_files:
        st.subheader(title)
        image_path = VISUAL_COMPARISON_DIR / filename

        if image_path.exists():
            st.image(str(image_path), use_container_width=True)
        else:
            st.warning(
                f"{filename} not found. Run python generate_visual_comparisons.py first."
            )

    st.info(
        "All visualisations are for academic evaluation. Published-study comparison should be treated "
        "as contextual if the datasets, disease classes or evaluation settings are different."
    )
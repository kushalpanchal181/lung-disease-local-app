from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image


BASE_DIR = Path(__file__).resolve().parent

DATASET_DIR = BASE_DIR / "dataset"
TEST_DIR = DATASET_DIR / "test"

MODEL_DIR = BASE_DIR / "model"
RESULTS_DIR = BASE_DIR / "results"
VISUAL_DIR = RESULTS_DIR / "visual_comparisons"

PROPOSED_MODEL_PATH = MODEL_DIR / "lung_model.keras"
BASELINE_MODEL_PATH = MODEL_DIR / "baseline_cnn.keras"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"

PROPOSED_METRICS_PATH = RESULTS_DIR / "evaluation_metrics.csv"
BASELINE_METRICS_PATH = RESULTS_DIR / "baseline" / "evaluation_metrics.csv"

PUBLISHED_JSON_PATH = BASE_DIR / "published_study_comparison.json"

VISUAL_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = (224, 224)
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]


def load_class_names():
    with open(CLASS_NAMES_PATH, "r") as f:
        return json.load(f)


def load_model(path):
    if not path.exists():
        return None
    return tf.keras.models.load_model(path)


def get_sample_images(max_images=4):
    samples = []

    for class_folder in sorted(TEST_DIR.iterdir()):
        if not class_folder.is_dir():
            continue

        images = [
            file for file in class_folder.iterdir()
            if file.suffix.lower() in IMAGE_EXTENSIONS
        ]

        if images:
            selected = random.choice(images)
            samples.append((selected, class_folder.name))

        if len(samples) >= max_images:
            break

    return samples


def predict_image(model, image_path, class_names):
    image = Image.open(image_path).convert("RGB")
    resized = image.resize(IMG_SIZE)

    img_array = np.array(resized).astype("float32")
    img_array = np.expand_dims(img_array, axis=0)

    probabilities = model.predict(img_array, verbose=0)[0]

    predicted_index = int(np.argmax(probabilities))
    predicted_class = class_names[predicted_index]
    confidence = float(probabilities[predicted_index]) * 100

    return predicted_class, confidence, probabilities


def save_placeholder_image(output_path, title, message):
    plt.figure(figsize=(10, 5))
    plt.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        fontsize=14,
        wrap=True
    )
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def create_preprocessing_comparison(sample_image_path):
    original = Image.open(sample_image_path).convert("RGB")
    resized = original.resize(IMG_SIZE)

    img_array = np.array(resized).astype("float32")
    preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(
        img_array.copy()
    )

    # Convert MobileNetV2 [-1, 1] range back to [0, 255] for display
    preprocessed_display = ((preprocessed + 1.0) / 2.0) * 255
    preprocessed_display = np.clip(preprocessed_display, 0, 255).astype("uint8")

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(original, cmap="gray")
    plt.title("Before Preprocessing\nOriginal X-ray")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(preprocessed_display)
    plt.title("After Preprocessing\nResized 224x224 + MobileNetV2 Preprocessing")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(VISUAL_DIR / "01_preprocessing_comparison.png")
    plt.close()


def create_input_prediction_comparison(proposed_model, class_names, samples):
    rows = len(samples)

    plt.figure(figsize=(10, 4 * rows))

    for idx, (image_path, actual_class) in enumerate(samples):
        image = Image.open(image_path).convert("RGB")

        predicted_class, confidence, probabilities = predict_image(
            proposed_model,
            image_path,
            class_names
        )

        plt.subplot(rows, 2, (idx * 2) + 1)
        plt.imshow(image, cmap="gray")
        plt.title(f"Input Image\nActual: {actual_class}")
        plt.axis("off")

        probability_text = "\n".join([
            f"{class_names[i]}: {probabilities[i] * 100:.2f}%"
            for i in range(len(class_names))
        ])

        plt.subplot(rows, 2, (idx * 2) + 2)
        plt.text(
            0.05,
            0.5,
            f"Prediction Result\n\n"
            f"Predicted: {predicted_class}\n"
            f"Confidence: {confidence:.2f}%\n\n"
            f"Class Probabilities:\n{probability_text}",
            fontsize=12,
            va="center"
        )
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(VISUAL_DIR / "02_input_prediction_comparison.png")
    plt.close()


def read_metrics(path):
    if not path.exists():
        return None

    df = pd.read_csv(path)
    return dict(zip(df["Metric"], df["Value"]))


def create_baseline_vs_proposed_metrics():
    baseline_metrics = read_metrics(BASELINE_METRICS_PATH)
    proposed_metrics = read_metrics(PROPOSED_METRICS_PATH)

    if baseline_metrics is None:
        save_placeholder_image(
            VISUAL_DIR / "03_baseline_vs_proposed_metrics.png",
            "Baseline vs Proposed Metrics",
            "Baseline metrics not found. Run python train_baseline.py first."
        )
        return

    if proposed_metrics is None:
        save_placeholder_image(
            VISUAL_DIR / "03_baseline_vs_proposed_metrics.png",
            "Baseline vs Proposed Metrics",
            "Proposed metrics not found. Run python train_model.py first."
        )
        return

    metric_names = ["Accuracy", "Precision", "Recall", "F1 Score"]

    baseline_values = [baseline_metrics.get(metric, 0) for metric in metric_names]
    proposed_values = [proposed_metrics.get(metric, 0) for metric in metric_names]

    x = np.arange(len(metric_names))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x - width / 2, baseline_values, width, label="Baseline CNN")
    plt.bar(x + width / 2, proposed_values, width, label="Proposed MobileNetV2")

    plt.xticks(x, metric_names)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Evaluation Results: Baseline CNN vs Proposed MobileNetV2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(VISUAL_DIR / "03_baseline_vs_proposed_metrics.png")
    plt.close()


def create_before_after_predictions(baseline_model, proposed_model, class_names, samples):
    if baseline_model is None:
        save_placeholder_image(
            VISUAL_DIR / "04_before_after_ai_technique.png",
            "Before and After Applying Proposed AI Technique",
            "Baseline model not found. Run python train_baseline.py first."
        )
        return

    rows = len(samples)

    plt.figure(figsize=(12, 4 * rows))

    for idx, (image_path, actual_class) in enumerate(samples):
        image = Image.open(image_path).convert("RGB")

        baseline_pred, baseline_conf, _ = predict_image(
            baseline_model,
            image_path,
            class_names
        )

        proposed_pred, proposed_conf, _ = predict_image(
            proposed_model,
            image_path,
            class_names
        )

        plt.subplot(rows, 3, (idx * 3) + 1)
        plt.imshow(image, cmap="gray")
        plt.title(f"Input X-ray\nActual: {actual_class}")
        plt.axis("off")

        plt.subplot(rows, 3, (idx * 3) + 2)
        plt.text(
            0.05,
            0.5,
            f"Before Proposed Technique\n"
            f"Baseline CNN\n\n"
            f"Prediction: {baseline_pred}\n"
            f"Confidence: {baseline_conf:.2f}%",
            fontsize=12,
            va="center"
        )
        plt.axis("off")

        plt.subplot(rows, 3, (idx * 3) + 3)
        plt.text(
            0.05,
            0.5,
            f"After Proposed Technique\n"
            f"MobileNetV2 Transfer Learning\n\n"
            f"Prediction: {proposed_pred}\n"
            f"Confidence: {proposed_conf:.2f}%",
            fontsize=12,
            va="center"
        )
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(VISUAL_DIR / "04_before_after_ai_technique.png")
    plt.close()


def create_published_vs_proposed_comparison():
    proposed_metrics = read_metrics(PROPOSED_METRICS_PATH)

    if proposed_metrics is None:
        save_placeholder_image(
            VISUAL_DIR / "05_published_vs_proposed_comparison.png",
            "Published Study vs Proposed Method",
            "Proposed metrics not found. Run python train_model.py first."
        )
        return

    if not PUBLISHED_JSON_PATH.exists():
        template = {
            "published_study": "Add published paper name here",
            "published_method": "Add published method name here",
            "published_dataset": "Add dataset name used in paper here",
            "published_metric_name": "Accuracy",
            "published_value": None,
            "proposed_metric_name": "Accuracy",
            "note": "This comparison is contextual because datasets/classes may differ."
        }

        with open(PUBLISHED_JSON_PATH, "w") as f:
            json.dump(template, f, indent=4)

        save_placeholder_image(
            VISUAL_DIR / "05_published_vs_proposed_comparison.png",
            "Published Study vs Proposed Method",
            "published_study_comparison.json created. Add the published study value, then run this script again."
        )
        return

    with open(PUBLISHED_JSON_PATH, "r") as f:
        data = json.load(f)

    published_value = data.get("published_value")

    if published_value is None:
        save_placeholder_image(
            VISUAL_DIR / "05_published_vs_proposed_comparison.png",
            "Published Study vs Proposed Method",
            "Add published_value in published_study_comparison.json, then run this script again."
        )
        return

    proposed_metric_name = data.get("proposed_metric_name", "Accuracy")
    proposed_value = proposed_metrics.get(proposed_metric_name, None)

    if proposed_value is None:
        save_placeholder_image(
            VISUAL_DIR / "05_published_vs_proposed_comparison.png",
            "Published Study vs Proposed Method",
            f"Proposed metric {proposed_metric_name} not found in evaluation_metrics.csv."
        )
        return

    labels = [
        f"Published Study\n{data.get('published_method', '')}",
        "Proposed Method\nMobileNetV2-CXR"
    ]

    values = [published_value, proposed_value]

    plt.figure(figsize=(10, 6))
    plt.bar(labels, values)
    plt.ylim(0, 1)
    plt.ylabel(data.get("published_metric_name", "Metric"))
    plt.title("Published Study vs Proposed Method")

    for i, value in enumerate(values):
        plt.text(i, value + 0.02, f"{value:.2f}", ha="center")

    note = data.get("note", "")
    plt.figtext(0.5, 0.01, note, ha="center", fontsize=9, wrap=True)

    plt.tight_layout()
    plt.savefig(VISUAL_DIR / "05_published_vs_proposed_comparison.png")
    plt.close()


def main():
    class_names = load_class_names()

    proposed_model = load_model(PROPOSED_MODEL_PATH)
    baseline_model = load_model(BASELINE_MODEL_PATH)

    if proposed_model is None:
        raise FileNotFoundError("Proposed model not found. Run python train_model.py first.")

    samples = get_sample_images(max_images=4)

    if not samples:
        raise FileNotFoundError("No test images found in dataset/test class folders.")

    create_preprocessing_comparison(samples[0][0])
    create_input_prediction_comparison(proposed_model, class_names, samples)
    create_baseline_vs_proposed_metrics()
    create_before_after_predictions(baseline_model, proposed_model, class_names, samples)
    create_published_vs_proposed_comparison()

    print("Visual comparisons generated successfully.")
    print(f"Saved in: {VISUAL_DIR}")


if __name__ == "__main__":
    main()
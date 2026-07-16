import cv2
import numpy as np
import tensorflow as tf


# --------------------------------------------------
# Helper functions
# --------------------------------------------------

def get_mobilenet_base_model(model):
    """
    Finds the MobileNetV2 base model inside the saved Keras model.
    """
    for layer in model.layers:
        if "mobilenet" in layer.name.lower():
            return layer

    raise ValueError("MobileNetV2 base model not found inside the saved model.")


def get_classifier_layers(model, base_model):
    """
    Gets all layers after the MobileNetV2 feature extractor.
    """
    base_model_index = model.layers.index(base_model)
    return model.layers[base_model_index + 1:]


def pass_through_classifier(feature_maps, classifier_layers):
    """
    Passes MobileNetV2 feature maps through the custom classification head.
    """
    x = feature_maps

    for layer in classifier_layers:
        try:
            x = layer(x, training=False)
        except TypeError:
            x = layer(x)

    return x


def normalize_heatmap(heatmap):
    """
    Normalises heatmap values between 0 and 1.
    """
    heatmap = np.maximum(heatmap, 0)
    max_value = np.max(heatmap)

    if max_value == 0:
        return heatmap

    return heatmap / max_value


def create_body_mask(original_image):
    """
    Creates a simple body/chest mask to reduce heatmap outside the X-ray body area.
    This is not medical lung segmentation.
    """
    original_image = original_image.convert("RGB")
    original_array = np.array(original_image)

    gray = cv2.cvtColor(original_array, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Remove very dark background
    mask = (gray > 15).astype(np.uint8) * 255

    # Remove tiny artefacts such as small labels in the black background
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    cleaned_mask = np.zeros_like(mask)
    image_area = mask.shape[0] * mask.shape[1]

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]

        # Keep only larger body/chest regions
        if area > image_area * 0.01:
            cleaned_mask[labels == i] = 255

    kernel = np.ones((11, 11), np.uint8)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)
    cleaned_mask = cv2.GaussianBlur(cleaned_mask, (15, 15), 0)

    return cleaned_mask.astype("float32") / 255.0


# --------------------------------------------------
# Grad-CAM
# --------------------------------------------------

def make_gradcam_heatmap(img_array, model, predicted_index):
    """
    Grad-CAM heatmap.
    Shows regions that influenced the final prediction.
    """
    base_model = get_mobilenet_base_model(model)
    classifier_layers = get_classifier_layers(model, base_model)

    preprocessed_img = tf.keras.applications.mobilenet_v2.preprocess_input(
        img_array.copy()
    )

    with tf.GradientTape() as tape:
        feature_maps = base_model(preprocessed_img, training=False)
        tape.watch(feature_maps)

        predictions = pass_through_classifier(feature_maps, classifier_layers)
        class_score = predictions[:, predicted_index]

    gradients = tape.gradient(class_score, feature_maps)

    pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))

    feature_maps = feature_maps[0]
    heatmap = feature_maps @ pooled_gradients[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap).numpy()

    return normalize_heatmap(heatmap)


# --------------------------------------------------
# LayerCAM
# --------------------------------------------------

def make_layercam_heatmap(img_array, model, predicted_index):
    """
    LayerCAM heatmap.
    Uses positive gradients at feature-map level.
    """
    base_model = get_mobilenet_base_model(model)
    classifier_layers = get_classifier_layers(model, base_model)

    preprocessed_img = tf.keras.applications.mobilenet_v2.preprocess_input(
        img_array.copy()
    )

    with tf.GradientTape() as tape:
        feature_maps = base_model(preprocessed_img, training=False)
        tape.watch(feature_maps)

        predictions = pass_through_classifier(feature_maps, classifier_layers)
        class_score = predictions[:, predicted_index]

    gradients = tape.gradient(class_score, feature_maps)

    positive_gradients = tf.nn.relu(gradients)
    weighted_feature_maps = positive_gradients * feature_maps

    heatmap = tf.reduce_sum(weighted_feature_maps, axis=-1)
    heatmap = heatmap[0].numpy()

    return normalize_heatmap(heatmap)


# --------------------------------------------------
# Score-CAM
# --------------------------------------------------

def make_scorecam_heatmap(img_array, model, predicted_index, top_n=24):
    """
    Score-CAM heatmap.
    This is slower because it runs multiple predictions using activation masks.

    top_n controls speed.
    Higher top_n = slower but more detailed.
    Lower top_n = faster but less detailed.
    """
    base_model = get_mobilenet_base_model(model)

    preprocessed_img = tf.keras.applications.mobilenet_v2.preprocess_input(
        img_array.copy()
    )

    feature_maps = base_model(preprocessed_img, training=False)
    feature_maps = feature_maps[0].numpy()

    _, _, channels = feature_maps.shape

    # Use strongest activation channels only to reduce CPU time
    channel_scores = np.mean(feature_maps, axis=(0, 1))
    selected_channels = np.argsort(channel_scores)[-top_n:]

    input_image = img_array[0]
    input_h, input_w = input_image.shape[:2]

    scorecam = np.zeros((input_h, input_w), dtype=np.float32)

    for channel in selected_channels:
        activation_map = feature_maps[:, :, channel]
        activation_map = normalize_heatmap(activation_map)

        if np.max(activation_map) == 0:
            continue

        activation_map = cv2.resize(activation_map, (input_w, input_h))

        masked_image = input_image * activation_map[..., np.newaxis]
        masked_image = np.expand_dims(masked_image, axis=0).astype("float32")

        prediction = model.predict(masked_image, verbose=0)
        class_score = prediction[0][predicted_index]

        scorecam += class_score * activation_map

    return normalize_heatmap(scorecam)


# --------------------------------------------------
# Heatmap overlay
# --------------------------------------------------

def overlay_heatmap(
    original_image,
    heatmap,
    alpha=0.42,
    threshold_percentile=88,
    use_body_mask=True
):
    """
    Soft attention heatmap overlay.

    This version:
    - keeps a blue attention-map style over the X-ray
    - shows strongest model-focused regions in yellow/red
    - keeps the original X-ray visible
    - avoids fake lung-shaped masking

    Important:
    This is not a medically confirmed affected area.
    It only shows model attention.
    """
    original_image = original_image.convert("RGB")
    original_array = np.array(original_image)

    h, w = original_array.shape[:2]

    heatmap = cv2.resize(heatmap, (w, h))
    heatmap = heatmap.astype("float32")
    heatmap = normalize_heatmap(heatmap)

    # Smooth for nicer visual output
    heatmap = cv2.GaussianBlur(heatmap, (31, 31), 0)
    heatmap = normalize_heatmap(heatmap)

    # Optional body/chest mask to reduce colour outside X-ray body area
    if use_body_mask:
        body_mask = create_body_mask(original_image)
        heatmap = heatmap * body_mask
        heatmap = normalize_heatmap(heatmap)
    else:
        body_mask = np.ones((h, w), dtype="float32")

    # Create full colour heatmap: low attention = blue, high attention = red
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # Base blue overlay over the body/chest area
    base_alpha = 0.22 * body_mask[..., np.newaxis]

    # Stronger overlay where heatmap value is high
    attention_alpha = alpha * heatmap[..., np.newaxis]

    # Final alpha mask
    final_alpha = np.clip(base_alpha + attention_alpha, 0, 0.65)

    overlayed_image = (
        original_array.astype("float32") * (1 - final_alpha)
        + heatmap_color.astype("float32") * final_alpha
    )

    overlayed_image = np.clip(overlayed_image, 0, 255).astype("uint8")

    return overlayed_image
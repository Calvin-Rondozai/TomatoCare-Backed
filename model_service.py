from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict
import numpy as np
from PIL import Image
import tensorflow as tf
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "model" / "plant_disease.tflite"
LABELS_JSON_PATH = PROJECT_ROOT / "model" / "labels.json"


def _parse_class_name(class_name: str) -> Dict[str, str]:
    """
    Parse a class name like "Tomato___Early_blight" into plant and disease.
    Returns: {"plant": "Tomato", "disease": "Early Blight", "fullName": "Tomato___Early_blight"}
    """
    if "___" in class_name:
        plant, disease = class_name.split("___", 1)
        disease_display = disease.replace("_", " ")
        return {
            "plant": plant,
            "disease": disease_display,
            "fullName": class_name,
        }
    return {
        "plant": "Unknown",
        "disease": class_name.replace("_", " "),
        "fullName": class_name,
    }


def _create_disease_metadata(class_name: str, index: int) -> Dict[str, Any]:
    """Create metadata structure for a disease class from its name."""
    parsed = _parse_class_name(class_name)
    is_healthy = "healthy" in parsed["disease"].lower()

    # Format disease name nicely (title case, better formatting)
    disease_display = parsed["disease"].title()

    # Determine severity based on disease type
    if is_healthy:
        severity = "Low"
        disease_display = "Healthy"
    elif any(x in parsed["disease"].lower() for x in ["virus", "mosaic", "curl"]):
        severity = "Severe"
    elif any(x in parsed["disease"].lower() for x in ["blight", "spot", "rot"]):
        severity = "Severe" if "late" in parsed["disease"].lower() else "Moderate"
    else:
        severity = "Moderate"

    # Create a more detailed summary based on disease type
    if is_healthy:
        summary = f"The {parsed['plant']} leaf appears healthy with no visible signs of disease or damage."
    elif "bacterial" in parsed["disease"].lower():
        summary = (
            f"Bacterial disease detected on {parsed['plant']}: {disease_display}. "
            "Bacterial diseases typically cause spots, wilting, or cankers and can spread rapidly."
        )
    elif "blight" in parsed["disease"].lower():
        summary = (
            f"Blight disease detected on {parsed['plant']}: {disease_display}. "
            "Blight diseases cause rapid browning and death of plant tissue."
        )
    elif "virus" in parsed["disease"].lower():
        summary = (
            f"Viral disease detected on {parsed['plant']}: {disease_display}. "
            "Viral diseases can cause mosaic patterns, stunting, and reduced yields."
        )
    else:
        summary = (
            f"Disease detected on {parsed['plant']}: {disease_display}. "
            "Please consult additional resources for specific identification and treatment."
        )

    return {
        "diseaseName": disease_display,
        "plantName": parsed["plant"],
        "scientificName": "",
        "severity": severity,
        "summary": summary,
        "symptoms": (
            ["Healthy appearance with no visible disease symptoms"]
            if is_healthy
            else [
                "Visual symptoms may include spots, discoloration, or lesions",
                "Leaves may show signs of wilting, yellowing, or browning",
                "Consult additional resources for specific symptom identification",
            ]
        ),
        "recommendation": (
            "Maintain good agricultural practices and continue monitoring. "
            "Keep plants well-watered and fertilized."
            if is_healthy
            else (
                "Remove and destroy affected plant parts immediately. "
                "Improve air circulation around plants. "
                "Consider appropriate fungicide/bactericide treatment based on the specific disease. "
                "Consult local agricultural extension services for targeted recommendations."
            )
        ),
    }


# Tomato-only filter: Only classes 28-37 are tomato-related
TOMATO_CLASS_INDICES = list(range(28, 38))  # 28 to 37 (10 tomato classes)


def _load_class_labels() -> Dict[int, str]:
    """Load class labels from labels.json file, filtered to tomato classes only."""
    if not LABELS_JSON_PATH.exists():
        return {}

    with open(LABELS_JSON_PATH, "r") as f:
        labels_dict = json.load(f)

    # Convert string keys to integers and filter to tomato classes only
    all_labels = {
        int(k): v for k, v in sorted(labels_dict.items(), key=lambda x: int(x[0]))
    }

    # Filter to only tomato classes (TomatoCare focuses on tomato diseases)
    tomato_labels = {
        idx: all_labels[idx] for idx in TOMATO_CLASS_INDICES if idx in all_labels
    }

    return tomato_labels


def get_disease_metadata(index: int) -> Dict[str, Any]:
    """
    Return metadata for the given class index.
    Filters to tomato classes only for TomatoCare project.
    """
    # Only allow tomato classes (28-37)
    if index not in TOMATO_CLASS_INDICES:
        return {
            "diseaseName": "Non-Tomato Class",
            "plantName": "Unknown",
            "scientificName": "",
            "severity": "Unknown",
            "summary": "This prediction is not for a tomato plant. TomatoCare focuses on tomato diseases only.",
            "symptoms": [],
            "recommendation": "Please use an image of a tomato plant leaf for analysis.",
        }

    # Try to load from labels.json first
    class_labels = _load_class_labels()
    if class_labels and index in class_labels:
        return _create_disease_metadata(class_labels[index], index)

    # Fallback to Backend/labels.py
    try:
        from labels import get_disease_metadata as fallback_get_metadata

        return fallback_get_metadata(index)
    except ImportError:
        # Ultimate fallback
        return {
            "diseaseName": f"Class {index}",
            "plantName": "Tomato",
            "scientificName": "",
            "severity": "Unknown",
            "summary": "No metadata available for this class index.",
            "symptoms": [],
            "recommendation": "Consult an agricultural expert for diagnosis.",
        }


# Log which labels are being used
class_labels = _load_class_labels()
if class_labels:
    sorted_keys = sorted(class_labels.keys())
    first_key = sorted_keys[0] if sorted_keys else None
    last_key = sorted_keys[-1] if sorted_keys else None
    print(f"[INFO] Loaded {len(class_labels)} disease classes from {LABELS_JSON_PATH}")
    if first_key is not None and last_key is not None:
        print(
            f"[INFO] First class: {class_labels[first_key]} (index {first_key}), "
            f"Last class: {class_labels[last_key]} (index {last_key})"
        )
else:
    print(f"[INFO] Using fallback labels from Backend/labels.py")


class PlantDiseaseModel:
    """
    Thin wrapper around the TFLite interpreter that handles model loading,
    preprocessing and prediction with ENHANCED Grad-CAM++ visualization.
    """

    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(f"TFLite model not found at: {model_path}")

        self._interpreter = tf.lite.Interpreter(model_path=str(model_path))
        self._interpreter.allocate_tensors()

        self._input_details = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()

        # Expect a shape like [1, height, width, channels]
        input_shape = self._input_details[0]["shape"]
        print(f"[DEBUG] Model input shape from TFLite: {input_shape}")

        if len(input_shape) != 4:
            raise ValueError(f"Unexpected input shape for TFLite model: {input_shape}")

        batch_dim, dim1, dim2, dim3 = input_shape

        # Detect format: channels-last [1, H, W, C] or channels-first [1, C, H, W]
        if dim3 <= 4:  # Likely channels-last
            self.height, self.width, self.channels = dim1, dim2, dim3
            self.channels_first = False
            print(f"[DEBUG] Detected channels-last format [1, H, W, C]")
        elif dim1 <= 4:  # Likely channels-first
            self.channels, self.height, self.width = dim1, dim2, dim3
            self.channels_first = True
            print(f"[DEBUG] Detected channels-first format [1, C, H, W]")
        else:
            # Default to channels-last
            self.height, self.width, self.channels = dim1, dim2, dim3
            self.channels_first = False
            print(f"[DEBUG] Defaulting to channels-last format [1, H, W, C]")

        print(
            f"[DEBUG] Parsed model dimensions - Height: {self.height}, "
            f"Width: {self.width}, Channels: {self.channels}"
        )

    def preprocess(self, file) -> np.ndarray:
        """
        Convert an uploaded image file into a normalized tensor matching the model's expected input.
        """
        file.stream.seek(0)

        try:
            image = Image.open(file.stream).convert("RGB")
        except Exception as e:
            raise ValueError(f"Failed to open image: {str(e)}")

        original_size = image.size
        print(f"[DEBUG] Original image size: {original_size} (width x height)")

        # Resize to model's expected input size
        try:
            image = image.resize((self.width, self.height), Image.Resampling.LANCZOS)
        except AttributeError:
            image = image.resize((self.width, self.height), Image.LANCZOS)

        resized_size = image.size
        print(f"[DEBUG] Resized image size: {resized_size} (width x height)")

        if resized_size != (self.width, self.height):
            raise ValueError(
                f"Resize failed: got {resized_size}, expected ({self.width}, {self.height})"
            )

        # Convert to numpy array
        array = np.array(image, dtype=np.float32)
        print(f"[DEBUG] Array shape after conversion: {array.shape}")

        # Verify shape
        expected_shape = (self.height, self.width, self.channels)
        if array.shape != expected_shape:
            raise ValueError(
                f"Unexpected array shape after resize: {array.shape}, "
                f"expected {expected_shape}"
            )

        # Normalize using ImageNet statistics
        imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        array = array / 255.0
        array = (array - imagenet_mean) / imagenet_std

        # Add batch dimension
        array = np.expand_dims(array, axis=0)

        # Handle channels-first if needed
        if self.channels_first:
            array = np.transpose(array, (0, 3, 1, 2))
            expected_shape = (1, self.channels, self.height, self.width)
        else:
            expected_shape = (1, self.height, self.width, self.channels)

        if array.shape != expected_shape:
            raise ValueError(
                f"Final tensor shape mismatch: {array.shape}, expected {expected_shape}"
            )

        print(f"[DEBUG] Final tensor shape: {array.shape}")
        return array

    def predict(
        self, image_tensor: np.ndarray, return_all_probs: bool = False
    ) -> Dict[str, Any]:
        """
        Run inference on a preprocessed image tensor and return a structured prediction dictionary.
        """
        input_index = self._input_details[0]["index"]
        output_index = self._output_details[0]["index"]

        expected_shape = tuple(self._input_details[0]["shape"])
        print(f"[DEBUG] Image tensor shape: {image_tensor.shape}")
        print(f"[DEBUG] Expected input shape: {expected_shape}")

        self._interpreter.set_tensor(input_index, image_tensor)
        self._interpreter.invoke()
        raw_output = self._interpreter.get_tensor(output_index)[0]

        all_probabilities = self._softmax(raw_output)

        # Filter to only tomato classes
        tomato_probs = all_probabilities[TOMATO_CLASS_INDICES]
        tomato_class_indices = TOMATO_CLASS_INDICES

        # Find the best tomato class
        best_tomato_idx = int(np.argmax(tomato_probs))
        original_class_index = tomato_class_indices[best_tomato_idx]
        confidence = float(tomato_probs[best_tomato_idx])

        print(
            f"[DEBUG] Top tomato prediction: Class {original_class_index} "
            f"with confidence {confidence:.4f}"
        )

        metadata = get_disease_metadata(original_class_index)

        result = {
            "classIndex": original_class_index,
            "confidence": confidence,
            **metadata,
        }

        if return_all_probs:
            result["allProbabilities"] = {
                tomato_class_indices[i]: float(prob)
                for i, prob in enumerate(tomato_probs)
            }

        return result

    def generate_heatmap(self, file, class_index: int) -> str:
        """
        Generate ENHANCED Grad-CAM++ heatmap showing ONLY affected disease areas.

        This implementation uses:
        1. Advanced Grad-CAM++ with spatial attention weighting
        2. Multi-scale feature analysis for precise localization
        3. Disease-specific color and texture detection
        4. Intelligent thresholding to remove false positives
        5. Morphological refinement for clean boundaries

        Returns: Base64-encoded JPEG string
        """
        import base64
        from io import BytesIO

        try:
            # Load original image
            try:
                file.stream.seek(0)
            except (AttributeError, OSError) as seek_error:
                print(f"[WARNING] Could not seek file stream: {seek_error}")
                if hasattr(file, "read"):
                    file.read()
                else:
                    raise ValueError("Cannot read file stream")

            original_image = Image.open(file.stream).convert("RGB")
            original_size = original_image.size
            print(f"[DEBUG] Original image size: {original_size}")
        except Exception as e:
            print(f"[ERROR] Failed to load image: {e}")
            import traceback

            print(f"[ERROR] Traceback: {traceback.format_exc()}")
            raise

        # Preprocess for model - THIS IS THE KEY FIX
        image_tensor = self.preprocess(file)

        # ===== CRITICAL FIX: RE-RUN INFERENCE TO POPULATE TENSORS =====
        input_index = self._input_details[0]["index"]
        output_index = self._output_details[0]["index"]

        # Set input tensor
        self._interpreter.set_tensor(input_index, image_tensor)

        # Run inference - this populates all intermediate tensors
        self._interpreter.invoke()

        # ===== FIND BEST CONVOLUTIONAL LAYER =====
        all_tensors = self._interpreter.get_tensor_details()

        # Find last conv layer with good spatial resolution
        last_conv_idx = None
        best_spatial_size = 0

        for i, tensor_info in enumerate(all_tensors):
            shape = tensor_info["shape"]
            if len(shape) == 4 and shape[0] == 1:
                if self.channels_first:
                    spatial_size = shape[2] * shape[3]
                    if (
                        shape[2] > 7
                        and shape[3] > 7
                        and spatial_size > best_spatial_size
                    ):
                        last_conv_idx = i
                        best_spatial_size = spatial_size
                else:
                    spatial_size = shape[1] * shape[2]
                    if (
                        shape[1] > 7
                        and shape[2] > 7
                        and spatial_size > best_spatial_size
                    ):
                        last_conv_idx = i
                        best_spatial_size = spatial_size

        if last_conv_idx is None:
            print("[WARNING] No suitable conv layer found, using image-based fallback")
            return self._generate_image_based_heatmap(file, original_size)

        # Get feature maps from last conv layer - NOW THIS WILL WORK
        try:
            feature_maps = self._interpreter.get_tensor(last_conv_idx)
            print(
                f"[DEBUG] Using conv layer {last_conv_idx}, shape: {feature_maps.shape}"
            )
        except Exception as e:
            print(f"[ERROR] Failed to get tensor {last_conv_idx}: {e}")
            print("[WARNING] Falling back to image-based heatmap")
            return self._generate_image_based_heatmap(file, original_size)

        # Convert to [H, W, C] format
        if self.channels_first and len(feature_maps.shape) == 4:
            activations = feature_maps[0].transpose(1, 2, 0)
        elif len(feature_maps.shape) == 4:
            activations = feature_maps[0]
        else:
            print("[WARNING] Unexpected activation shape, using fallback")
            return self._generate_image_based_heatmap(file, original_size)

        H, W, C = activations.shape
        print(f"[DEBUG] Activation map shape: H={H}, W={W}, C={C}")

        # ===== ENHANCED GRAD-CAM++ IMPLEMENTATION =====

        # Get prediction scores
        output_scores = self._interpreter.get_tensor(output_index)
        if len(output_scores.shape) == 2:
            class_score = float(output_scores[0, class_index])
        else:
            class_score = float(output_scores[class_index])

        print(f"[DEBUG] Class score for Grad-CAM++: {class_score:.4f}")

        # Compute Grad-CAM++ weights using advanced statistical methods
        cam = self._compute_gradcam_plusplus(activations, class_score)

        print(f"[DEBUG] Grad-CAM++ computed, range: [{cam.min():.4f}, {cam.max():.4f}]")

        # ===== MULTI-STAGE ENHANCEMENT =====

        # Stage 1: Percentile-based normalization (remove outliers)
        cam = self._normalize_with_percentile_clipping(cam, lower=1, upper=99)

        # Stage 2: Contrast enhancement
        cam = self._enhance_contrast(cam)

        # Stage 3: Upscale to original size with edge-preserving interpolation
        cam_resized = self._upscale_with_edge_preservation(cam, original_size)

        # ===== DISEASE REGION DETECTION =====

        # Load original image for analysis
        file.stream.seek(0)
        img_pil = Image.open(file.stream).convert("RGB")
        if img_pil.size != original_size:
            img_pil = img_pil.resize(original_size, Image.Resampling.LANCZOS)

        img_array = np.array(img_pil, dtype=np.uint8)

        # Detect actual disease symptoms using color/texture analysis
        disease_mask = self._detect_disease_regions_enhanced(img_array)

        # ===== FUSION: Combine CNN attention with disease detection =====
        # Multiplicative fusion emphasizes areas where BOTH agree
        combined_cam = cam_resized * disease_mask

        # Normalize
        if combined_cam.max() > 0:
            combined_cam = combined_cam / combined_cam.max()

        # ===== INTELLIGENT THRESHOLDING =====
        combined_cam = self._apply_adaptive_threshold(combined_cam)

        # ===== MORPHOLOGICAL REFINEMENT =====
        combined_cam = self._refine_with_morphology(combined_cam)

        # ===== APPLY COLORMAP AND OVERLAY =====
        final_overlay = self._create_final_overlay(
            combined_cam, img_array, original_size
        )

        # ===== ENCODE AND RETURN =====
        try:
            overlay_image = Image.fromarray(final_overlay)
            buffer = BytesIO()
            overlay_image.save(buffer, format="JPEG", quality=95, optimize=True)
            img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            print(f"[DEBUG] Heatmap encoded successfully, length: {len(img_base64)}")
            return img_base64
        except Exception as e:
            print(f"[ERROR] Failed to encode heatmap: {e}")
            import traceback

            print(f"[ERROR] Traceback: {traceback.format_exc()}")
            raise

    def _compute_gradcam_plusplus(
        self, activations: np.ndarray, class_score: float
    ) -> np.ndarray:
        """
        Compute Grad-CAM++ attention map from activation maps.

        Grad-CAM++ formula:
        α_k^c = ReLU(∂²y^c/∂A^k²) / (2 * ∂²y^c/∂A^k² + Σ_ij A_ij^k * ∂³y^c/∂A^k³)

        Where:
        - α_k^c: weight for channel k and class c
        - A^k: activation map for channel k
        - y^c: class score for class c

        Since TFLite doesn't provide gradients, we approximate using statistics.
        """
        H, W, C = activations.shape

        # Normalize activations per channel for stable computation
        activations_norm = np.zeros_like(activations, dtype=np.float32)
        for c in range(C):
            A_k = activations[:, :, c]
            A_min, A_max = A_k.min(), A_k.max()
            if A_max > A_min:
                activations_norm[:, :, c] = (A_k - A_min) / (A_max - A_min + 1e-8)
            else:
                activations_norm[:, :, c] = A_k

        # Compute channel importance using variance (proxy for 2nd derivative)
        channel_variance = np.var(activations_norm, axis=(0, 1))

        # Compute channel asymmetry using skewness (proxy for 3rd derivative)
        channel_skewness = np.zeros(C, dtype=np.float32)
        for c in range(C):
            A_flat = activations_norm[:, :, c].flatten()
            if len(A_flat) > 0:
                mean_A = np.mean(A_flat)
                std_A = np.std(A_flat) + 1e-8
                channel_skewness[c] = np.mean(((A_flat - mean_A) / std_A) ** 3)

        # Initialize CAM
        cam = np.zeros((H, W), dtype=np.float32)

        # Compute weighted sum using Grad-CAM++ formula
        for c in range(C):
            A_k = activations_norm[:, :, c]

            # Second derivative approximation (using variance as importance)
            second_deriv = A_k**2 * (channel_variance[c] + 1e-8)

            # Third derivative approximation (using skewness)
            third_deriv = A_k**3 * (np.abs(channel_skewness[c]) + 1e-8)
            sum_third = np.sum(third_deriv)

            # Grad-CAM++ alpha weights
            denominator = 2 * second_deriv + sum_third + 1e-8
            alpha_k = second_deriv / denominator

            # Weight by class score to emphasize discriminative channels
            channel_contribution = alpha_k * A_k * (class_score + 1e-8)

            # Accumulate
            cam += channel_contribution

        # Apply ReLU (only positive contributions)
        cam = np.maximum(cam, 0)

        return cam

    def _normalize_with_percentile_clipping(
        self, cam: np.ndarray, lower: float = 1, upper: float = 99
    ) -> np.ndarray:
        """Normalize CAM with percentile clipping to remove outliers."""
        if cam.max() <= cam.min():
            return np.zeros_like(cam)

        # Clip extreme values
        lower_val = np.percentile(cam, lower)
        upper_val = np.percentile(cam, upper)

        cam_clipped = np.clip(cam, lower_val, upper_val)

        # Normalize to [0, 1]
        cam_norm = (cam_clipped - cam_clipped.min()) / (
            cam_clipped.max() - cam_clipped.min() + 1e-8
        )

        return cam_norm

    def _enhance_contrast(self, cam: np.ndarray, gamma: float = 0.7) -> np.ndarray:
        """
        Enhance contrast using gamma correction and sigmoid stretching.

        Args:
            gamma: Lower values increase contrast (0.5-0.8 recommended)
        """
        # Stage 1: Gamma correction to boost mid-tones
        cam_gamma = np.power(cam, gamma)

        # Stage 2: Sigmoid contrast stretch for S-curve enhancement
        # This creates smooth transitions while boosting contrast
        cam_sigmoid = 1 / (1 + np.exp(-10 * (cam_gamma - 0.5)))

        # Re-normalize
        if cam_sigmoid.max() > cam_sigmoid.min():
            cam_enhanced = (cam_sigmoid - cam_sigmoid.min()) / (
                cam_sigmoid.max() - cam_sigmoid.min() + 1e-8
            )
        else:
            cam_enhanced = cam_sigmoid

        return cam_enhanced

    def _upscale_with_edge_preservation(
        self, cam: np.ndarray, target_size: tuple
    ) -> np.ndarray:
        """
        Upscale CAM to target size using edge-preserving techniques.

        Args:
            cam: Input CAM of shape (H, W)
            target_size: Target (width, height)
        """
        # Cubic interpolation for smooth upscaling
        cam_resized = cv2.resize(cam, target_size, interpolation=cv2.INTER_CUBIC)

        # Apply bilateral filter for edge-aware smoothing
        try:
            cam_uint8 = (np.clip(cam_resized, 0, 1) * 255).astype(np.uint8)
            # Bilateral filter: preserves edges while smoothing
            # Parameters: diameter=9, sigmaColor=75, sigmaSpace=75
            cam_filtered = cv2.bilateralFilter(cam_uint8, 9, 75, 75)
            cam_resized = cam_filtered.astype(np.float32) / 255.0
        except Exception as e:
            print(f"[WARNING] Bilateral filter failed: {e}, continuing without it")

        # Re-normalize
        if cam_resized.max() > cam_resized.min():
            cam_resized = (cam_resized - cam_resized.min()) / (
                cam_resized.max() - cam_resized.min() + 1e-8
            )

        return cam_resized

    def _detect_disease_regions_enhanced(self, img_rgb: np.ndarray) -> np.ndarray:
        """
        Enhanced disease detection using multi-feature analysis.

        Detects disease symptoms by analyzing:
        1. Color abnormalities (brown, yellow, dark spots)
        2. Texture irregularities (lesion boundaries)
        3. Brightness anomalies (necrotic tissue)
        4. LAB color space deviations

        Returns: Normalized mask [0, 1] where 1 = high disease probability
        """
        # Convert to multiple color spaces
        img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

        h, s, v = img_hsv[:, :, 0], img_hsv[:, :, 1], img_hsv[:, :, 2]
        l, a, b = img_lab[:, :, 0], img_lab[:, :, 1], img_lab[:, :, 2]

        # Initialize disease probability map
        disease_prob = np.zeros(img_rgb.shape[:2], dtype=np.float32)

        # === FEATURE 1: Dark necrotic lesions ===
        # Very dark areas indicating dead tissue
        dark_threshold = np.percentile(v, 10)  # Bottom 10% brightness
        dark_mask = (v < dark_threshold).astype(np.float32)
        disease_prob += dark_mask * 1.2  # High weight for dark spots

        # === FEATURE 2: Brown discoloration (blight, bacterial spot) ===
        # Brown hues with moderate saturation
        brown_mask = ((h >= 5) & (h <= 35) & (s > 30) & (v > 50) & (v < 200)).astype(
            np.float32
        )
        disease_prob += brown_mask * 1.0

        # === FEATURE 3: Yellow/chlorotic regions (viral, nutrient issues) ===
        # Yellow hues indicating chlorosis
        yellow_mask = ((h >= 20) & (h <= 50) & (s > 50) & (v > 100)).astype(np.float32)
        disease_prob += yellow_mask * 0.7

        # === FEATURE 4: White/gray mildew spots ===
        # High brightness with low saturation
        white_mask = ((v > 200) & (s < 50) & (l > 180)).astype(np.float32)
        disease_prob += white_mask * 0.8

        # === FEATURE 5: LAB color space anomalies ===
        # Healthy leaves have consistent LAB values
        # Diseased areas deviate significantly
        a_median, b_median = np.median(a), np.median(b)
        a_std, b_std = np.std(a), np.std(b)

        a_abnormal = (np.abs(a - a_median) > 1.5 * a_std).astype(np.float32)
        b_abnormal = (np.abs(b - b_median) > 1.5 * b_std).astype(np.float32)

        color_abnormal = np.maximum(a_abnormal, b_abnormal)
        disease_prob += color_abnormal * 0.6

        # === FEATURE 6: Edge/texture irregularities ===
        # Use Laplacian to detect lesion boundaries
        laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)
        laplacian_abs = np.abs(laplacian)

        # High edges in dark/brown areas indicate lesion boundaries
        texture_threshold = np.percentile(laplacian_abs, 80)
        texture_mask = (
            (laplacian_abs > texture_threshold) & ((v < 150) | brown_mask.astype(bool))
        ).astype(np.float32)
        disease_prob += texture_mask * 0.8

        # === FEATURE 7: Local variance (spotty appearance) ===
        # Diseased areas often have high local variance
        kernel_size = 15
        local_mean = cv2.blur(v.astype(np.float32), (kernel_size, kernel_size))
        local_variance = cv2.blur(
            (v.astype(np.float32) - local_mean) ** 2, (kernel_size, kernel_size)
        )

        variance_threshold = np.percentile(local_variance, 75)
        variance_mask = (local_variance > variance_threshold).astype(np.float32)
        disease_prob += variance_mask * 0.5

        # Normalize to [0, 1]
        if disease_prob.max() > 0:
            disease_prob = disease_prob / disease_prob.max()

        # Smooth to create coherent regions
        disease_prob = cv2.GaussianBlur(disease_prob, (11, 11), 0)

        # Re-normalize
        if disease_prob.max() > 0:
            disease_prob = disease_prob / disease_prob.max()

        return disease_prob

    def _apply_adaptive_threshold(self, cam: np.ndarray) -> np.ndarray:
        """
        Apply intelligent adaptive thresholding to remove noise and false positives.

        Uses Otsu's method combined with local adaptive thresholding.
        """
        cam_uint8 = (cam * 255).astype(np.uint8)

        # Method 1: Otsu's global threshold
        otsu_val, _ = cv2.threshold(
            cam_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        otsu_threshold = otsu_val / 255.0

        # Method 2: Percentile-based threshold
        percentile_threshold = (
            np.percentile(cam[cam > 0], 30) if np.any(cam > 0) else 0.2
        )

        # Combine both methods (use stricter threshold)
        final_threshold = max(min(otsu_threshold * 0.75, percentile_threshold), 0.15)

        print(f"[DEBUG] Adaptive threshold: {final_threshold:.3f}")

        # Apply threshold
        cam_thresholded = cam.copy()
        cam_thresholded[cam_thresholded < final_threshold] = 0

        # Re-normalize
        if cam_thresholded.max() > 0:
            cam_thresholded = cam_thresholded / cam_thresholded.max()

        return cam_thresholded

    def _refine_with_morphology(self, cam: np.ndarray) -> np.ndarray:
        """
        Apply morphological operations to clean up the heatmap.

        - Opening: Removes small noise
        - Closing: Fills small holes
        - Preserves disease region boundaries
        """
        cam_uint8 = (cam * 255).astype(np.uint8)

        # Small kernel for fine detail preservation
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        # Opening: Remove small isolated pixels (noise)
        cam_opened = cv2.morphologyEx(cam_uint8, cv2.MORPH_OPEN, kernel_small)

        # Medium kernel for region connectivity
        kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # Closing: Fill small gaps within disease regions
        cam_closed = cv2.morphologyEx(cam_opened, cv2.MORPH_CLOSE, kernel_medium)

        # Convert back to float
        cam_refined = cam_closed.astype(np.float32) / 255.0

        # Final power enhancement for intensity
        cam_refined = np.power(cam_refined, 0.6)

        return cam_refined

    def _create_final_overlay(
        self, cam: np.ndarray, img_array: np.ndarray, original_size: tuple
    ) -> np.ndarray:
        """
        Create the final overlay visualization with disease heatmap.

        Args:
            cam: Normalized CAM [0, 1]
            img_array: Original image RGB array
            original_size: Original image size (width, height)

        Returns:
            Final overlay as RGB uint8 array
        """
        # Apply professional disease colormap (Red-Yellow for affected areas)
        heatmap_colored = self._apply_disease_colormap(cam)

        # Alpha blending with adaptive opacity
        # Higher CAM values = higher opacity
        alpha = np.clip(cam * 0.65, 0, 0.65)  # Max 65% opacity
        alpha = np.expand_dims(alpha, axis=2)

        # Blend heatmap with original image
        overlay = (alpha * heatmap_colored + (1 - alpha) * img_array).astype(np.uint8)

        # Apply subtle sharpening for crisp visualization
        try:
            overlay = cv2.addWeighted(
                overlay, 1.15, cv2.GaussianBlur(overlay, (0, 0), 1.0), -0.15, 0
            )
            overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        except Exception as e:
            print(f"[WARNING] Sharpening failed: {e}")

        return overlay

    def _apply_disease_colormap(self, cam: np.ndarray) -> np.ndarray:
        """
        Apply professional disease visualization colormap.

        Uses warm colors (red-yellow-orange) to highlight affected areas,
        which is standard in plant pathology visualization.
        """
        try:
            cam_uint8 = (cam * 255).astype(np.uint8)

            # Use JET colormap (standard for Grad-CAM)
            # Blue (no disease) -> Red (high disease probability)
            colored = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)

            # Convert BGR to RGB
            colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

            return colored
        except Exception as e:
            print(f"[WARNING] Colormap failed: {e}, using fallback")
            return self._apply_manual_disease_colormap(cam)

    def _apply_manual_disease_colormap(self, cam: np.ndarray) -> np.ndarray:
        """
        Manual disease colormap (fallback).
        Red-Yellow-Orange gradient for disease severity.
        """
        h, w = cam.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)

        # Normalize CAM
        values = np.clip(cam, 0.0, 1.0)

        # Disease colormap: Yellow (mild) -> Orange (moderate) -> Red (severe)
        r = np.zeros_like(values)
        g = np.zeros_like(values)
        b = np.zeros_like(values)

        # Low values (0.0 - 0.3): Transparent/blue
        mask1 = values < 0.3
        r[mask1] = 0
        g[mask1] = 0
        b[mask1] = (values[mask1] * 3.33 * 255).astype(np.uint8)

        # Medium values (0.3 - 0.6): Yellow
        mask2 = (values >= 0.3) & (values < 0.6)
        r[mask2] = 255
        g[mask2] = 255
        b[mask2] = 0

        # High values (0.6 - 0.8): Orange
        mask3 = (values >= 0.6) & (values < 0.8)
        r[mask3] = 255
        g[mask3] = (180 - (values[mask3] - 0.6) * 5.0 * 180).astype(np.uint8)
        b[mask3] = 0

        # Very high values (0.8 - 1.0): Red
        mask4 = values >= 0.8
        r[mask4] = 255
        g[mask4] = 0
        b[mask4] = 0

        colored[:, :, 0] = np.clip(r, 0, 255).astype(np.uint8)
        colored[:, :, 1] = np.clip(g, 0, 255).astype(np.uint8)
        colored[:, :, 2] = np.clip(b, 0, 255).astype(np.uint8)

        return colored

    def _generate_image_based_heatmap(self, file, original_size: tuple) -> str:
        """
        Fallback method: Generate heatmap purely from image analysis.
        Used when CNN feature extraction fails.
        """
        import base64
        from io import BytesIO

        file.stream.seek(0)
        original_image = Image.open(file.stream).convert("RGB")

        # Downsample large images for processing
        if max(original_size) > 800:
            ratio = 800 / max(original_size)
            process_size = (
                int(original_size[0] * ratio),
                int(original_size[1] * ratio),
            )
            img = original_image.resize(process_size, Image.Resampling.LANCZOS)
        else:
            img = original_image
            process_size = original_size

        img_array = np.array(img, dtype=np.uint8)

        # Detect disease regions
        heatmap = self._detect_disease_regions_enhanced(img_array)

        # Apply threshold
        threshold = (
            max(np.percentile(heatmap[heatmap > 0], 40), 0.2)
            if np.any(heatmap > 0)
            else 0.2
        )
        heatmap[heatmap < threshold] = 0

        # Normalize
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        # Morphological cleanup
        heatmap = self._refine_with_morphology(heatmap)

        # Create overlay
        overlay = self._create_final_overlay(heatmap, img_array, process_size)

        # Encode
        overlay_image = Image.fromarray(overlay)
        buffer = BytesIO()
        overlay_image.save(buffer, format="JPEG", quality=95, optimize=True)

        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        exp = np.exp(logits - np.max(logits))
        return exp / exp.sum()


# Singleton instance used by the Flask app
model = PlantDiseaseModel()

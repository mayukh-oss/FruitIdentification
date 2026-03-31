import cv2
import numpy as np
from tensorflow.keras.models import load_model # type: ignore
import os
import sys


# ── Class Names (must match training folder names exactly) ────
CLASS_NAMES = [
    "Apple",
    "Banana",
    "Blackberry",
    "Cherry",
    "Mango",
    "Raspberry",
    "Strawberry"
]


def load_fruit_model(model_path="../models/fruit_model.h5"):
    """
    Loads and returns the trained fruit classification model.
    """
    print("\n" + "=" * 60)
    print("🔍 FRUIT CLASSIFICATION — PREDICTION MODULE")
    print("=" * 60)

    print(f"\n📦 Step 1 — Loading trained model from '{model_path}'...")

    if not os.path.exists(model_path):
        print(f"   ❌ ERROR: Model file not found at '{model_path}'")
        print("   💡 Please run training first to generate the model.")
        return None

    model = load_model(model_path)
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"   ✅ Model loaded successfully! ({size_mb:.1f} MB)")
    print(f"   • Input shape  : {model.input_shape}")
    print(f"   • Output shape : {model.output_shape}  ({len(CLASS_NAMES)} classes)")

    return model


def pick_image_via_dialog():
    """
    Opens a native file-chooser window and returns the selected image path.
    Falls back to a text prompt if tkinter is unavailable.
    """
    print("\n🖼️  Step 2 — Selecting Image...")
    print("   Opening file chooser window — please select a fruit image.")

    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()          # Hide the main window
        root.attributes('-topmost', True)   # Bring dialog to front

        img_path = filedialog.askopenfilename(
            title="Select a Fruit Image",
            filetypes=[
                ("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("All Files", "*.*")
            ]
        )
        root.destroy()

        if not img_path:
            print("   ⚠️  No file selected. Exiting prediction.")
            return None

        print(f"   ✅ File selected: {img_path}")
        return img_path

    except Exception as e:
        print(f"   ⚠️  GUI dialog unavailable ({e}). Falling back to text input.")
        img_path = input("   👉 Enter full image path manually: ").strip()
        return img_path if img_path else None


def preprocess_image(img_path):
    """
    Reads, converts to RGB, resizes to 224x224, normalizes, and
    returns both the display-ready RGB image and the model-ready batch.
    """
    print(f"\n⚙️  Step 3 — Preprocessing Image...")
    print(f"   Reading: {img_path}")

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"   ❌ ERROR: Could not read image. File may be corrupt or unsupported.")
        return None, None

    h, w = img_bgr.shape[:2]
    print(f"   • Original size : {w}x{h} pixels")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    print("   • Color space   : BGR → RGB ✅")

    img_resized = cv2.resize(img_rgb, (224, 224))
    print("   • Resized to    : 224x224 pixels ✅")

    img_normalized = img_resized / 255.0
    print("   • Normalized    : pixel values scaled to [0.0, 1.0] ✅")

    img_batch = np.expand_dims(img_normalized, axis=0)
    print(f"   • Batch shape   : {img_batch.shape}  (1 image, 224x224, 3 channels) ✅")

    return img_rgb, img_batch


def predict_fruit(model, img_batch, class_names=None):
    """
    Runs inference and returns predicted class name, confidence, and full probability array.
    """
    if class_names is None:
        class_names = CLASS_NAMES

    print(f"\n🤖 Step 4 — Running Model Inference...")
    print("   Passing preprocessed image through MobileNetV2 network...")

    predictions = model.predict(img_batch, verbose=0)
    all_probs   = predictions[0]        # Shape: (num_classes,)

    predicted_idx   = np.argmax(all_probs)
    predicted_class = class_names[predicted_idx]
    confidence      = all_probs[predicted_idx] * 100

    print(f"\n   📊 Raw Probabilities:")
    for i, (cls, prob) in enumerate(zip(class_names, all_probs)):
        marker = " ◀ PREDICTED" if i == predicted_idx else ""
        bar    = "█" * int(prob * 30)
        print(f"      {cls:<15} {bar:<32} {prob * 100:5.1f}%{marker}")

    print(f"\n{'=' * 60}")
    print(f"  🍎 Predicted Fruit : {predicted_class}")
    print(f"  📊 Confidence      : {confidence:.2f}%")
    if confidence >= 80:
        print(f"  🟢 Confidence Level: HIGH — very likely correct")
    elif confidence >= 50:
        print(f"  🟡 Confidence Level: MODERATE — probably correct")
    else:
        print(f"  🔴 Confidence Level: LOW — model is uncertain")
    print(f"{'=' * 60}")

    return predicted_class, confidence, all_probs


def run_prediction(model_path="../models/fruit_model.h5", class_names=None):
    """
    Full prediction pipeline:
      1. Load model
      2. Pick image via GUI dialog
      3. Preprocess
      4. Predict
      5. Return results for visualization
    """
    if class_names is None:
        class_names = CLASS_NAMES

    # Load model
    model = load_fruit_model(model_path)
    if model is None:
        return None, None, None, None, None

    # Select image
    img_path = pick_image_via_dialog()
    if img_path is None:
        return None, None, None, None, None

    if not os.path.exists(img_path):
        print(f"\n❌ ERROR: Image file not found at '{img_path}'")
        return None, None, None, None, None

    # Preprocess
    img_rgb, img_batch = preprocess_image(img_path)
    if img_rgb is None:
        return None, None, None, None, None

    # Predict
    predicted_class, confidence, all_probs = predict_fruit(model, img_batch, class_names)

    return img_rgb, predicted_class, confidence, all_probs, class_names


# ── Standalone execution ───────────────────────────────────────
if __name__ == "__main__":
    from utils import plot_prediction

    img_rgb, predicted_class, confidence, all_probs, class_names = run_prediction()

    if img_rgb is not None:
        plot_prediction(img_rgb, predicted_class, confidence, all_probs, class_names)

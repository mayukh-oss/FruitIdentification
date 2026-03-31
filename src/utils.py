import matplotlib.pyplot as plt
import numpy as np


def plot_history(history):
    """
    Plots training & validation accuracy and loss curves side by side.
    """
    print("\n📊 Generating Training History Plots...")

    epochs = range(1, len(history.history['accuracy']) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Model Training History', fontsize=16, fontweight='bold')

    # ── Accuracy Plot ─────────────────────────────────────────
    axes[0].plot(epochs, history.history['accuracy'],     'b-o', label='Train Accuracy',      linewidth=2, markersize=5)
    axes[0].plot(epochs, history.history['val_accuracy'], 'r-o', label='Validation Accuracy', linewidth=2, markersize=5)
    axes[0].set_title('Accuracy over Epochs')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].set_ylim([0, 1.05])

    # ── Loss Plot ─────────────────────────────────────────────
    axes[1].plot(epochs, history.history['loss'],     'b-o', label='Train Loss',      linewidth=2, markersize=5)
    axes[1].plot(epochs, history.history['val_loss'], 'r-o', label='Validation Loss', linewidth=2, markersize=5)
    axes[1].set_title('Loss over Epochs')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

    # ── Summary Stats ─────────────────────────────────────────
    best_epoch     = np.argmax(history.history['val_accuracy']) + 1
    best_val_acc   = max(history.history['val_accuracy']) * 100
    best_val_loss  = min(history.history['val_loss'])

    print(f"\n📈 Training Summary:")
    print(f"   • Best Epoch          : {best_epoch}")
    print(f"   • Best Val Accuracy   : {best_val_acc:.2f}%")
    print(f"   • Best Val Loss       : {best_val_loss:.4f}")
    print(f"   • Total Epochs Ran    : {len(epochs)}")


def plot_prediction(img_rgb, predicted_class, confidence, all_probs, class_names):
    """
    Shows the input image alongside a bar chart of all class probabilities.
    """
    print("\n🖼️  Generating Prediction Visualization...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Fruit Classification Result', fontsize=16, fontweight='bold')

    # ── Image ─────────────────────────────────────────────────
    axes[0].imshow(img_rgb)
    axes[0].set_title(
        f"Predicted: {predicted_class}\nConfidence: {confidence:.2f}%",
        fontsize=13, color='green', fontweight='bold'
    )
    axes[0].axis('off')

    # ── Probability Bar Chart ─────────────────────────────────
    colors = ['#2ecc71' if c == predicted_class else '#3498db' for c in class_names]
    bars = axes[1].barh(class_names, all_probs * 100, color=colors)
    axes[1].set_xlabel('Confidence (%)')
    axes[1].set_title('Class Probabilities')
    axes[1].set_xlim([0, 110])
    axes[1].grid(True, axis='x', linestyle='--', alpha=0.5)

    # Add percentage labels on bars
    for bar, prob in zip(bars, all_probs * 100):
        axes[1].text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                     f'{prob:.1f}%', va='center', fontsize=9)

    plt.tight_layout()
    plt.show()

    print("   ✅ Visualization displayed.")

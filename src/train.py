import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.applications import MobileNetV2 # type: ignore
from tensorflow.keras import layers, models # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore

def train_model(data_path="../data/train", model_path="../models/fruit_model.h5"):
    """
    Trains a MobileNetV2-based fruit classification model.
    Returns the trained model and training history.
    """

    print("=" * 60)
    print("🚀 FRUIT CLASSIFICATION MODEL - TRAINING STARTED")
    print("=" * 60)

    # ── Settings ──────────────────────────────────────────────
    IMG_SIZE   = (224, 224)
    BATCH_SIZE = 32
    EPOCHS     = 10

    print(f"\n📋 Training Configuration:")
    print(f"   • Image Size  : {IMG_SIZE[0]}x{IMG_SIZE[1]} pixels")
    print(f"   • Batch Size  : {BATCH_SIZE}")
    print(f"   • Max Epochs  : {EPOCHS}")
    print(f"   • Data Path   : {data_path}")
    print(f"   • Model Output: {model_path}")

    # ── Data Augmentation & Generators ───────────────────────
    print("\n📂 Step 1 — Setting up Image Data Generators...")
    print("   Applying augmentations: rescale, rotation(20°), zoom(20%), horizontal flip")
    print("   80% data → Training  |  20% data → Validation")

    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )

    print("\n   Loading TRAINING data from disk...")
    train_data = datagen.flow_from_directory(
        data_path,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    print("\n   Loading VALIDATION data from disk...")
    val_data = datagen.flow_from_directory(
        data_path,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    num_classes = train_data.num_classes
    class_names = list(train_data.class_indices.keys())

    print(f"\n✅ Data loaded successfully!")
    print(f"   • Classes found ({num_classes}): {class_names}")
    print(f"   • Training samples   : {train_data.samples}")
    print(f"   • Validation samples : {val_data.samples}")
    print(f"   • Training batches   : {len(train_data)}")
    print(f"   • Validation batches : {len(val_data)}")

    # ── Base Model ────────────────────────────────────────────
    print("\n🧠 Step 2 — Loading MobileNetV2 Base Model (pretrained on ImageNet)...")
    print("   Downloading weights if not cached (this may take a moment)...")

    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False   # Freeze base layers

    print(f"   ✅ MobileNetV2 loaded — {len(base_model.layers)} layers (all frozen)")
    print("   💡 We only train the new classification head on top.")

    # ── Custom Classification Head ────────────────────────────
    print("\n🔧 Step 3 — Building Custom Classification Head...")

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    print(f"   Architecture:")
    print(f"     MobileNetV2 (frozen) → GlobalAveragePooling2D → Dense(128, ReLU) → Dense({num_classes}, Softmax)")

    # ── Compile ───────────────────────────────────────────────
    print("\n⚙️  Step 4 — Compiling Model...")
    print("   Optimizer: Adam  |  Loss: Categorical Crossentropy  |  Metric: Accuracy")

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    total_params     = model.count_params()
    trainable_params = sum(tf.size(w).numpy() for w in model.trainable_weights)
    print(f"   Total parameters     : {total_params:,}")
    print(f"   Trainable parameters : {trainable_params:,}")

    # ── Training ──────────────────────────────────────────────
    print("\n🏋️  Step 5 — Training the Model...")
    print("   Early stopping enabled: patience=3 (monitors val_loss)")
    print("   Training will stop early if validation loss stops improving.\n")
    print("-" * 60)

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=EPOCHS,
        callbacks=[early_stop],
        verbose=1
    )

    epochs_trained = len(history.history['accuracy'])
    final_train_acc = history.history['accuracy'][-1] * 100
    final_val_acc   = history.history['val_accuracy'][-1] * 100
    best_val_loss   = min(history.history['val_loss'])

    print("-" * 60)
    print(f"\n🎉 Training Complete!")
    print(f"   • Epochs trained      : {epochs_trained}/{EPOCHS}")
    print(f"   • Final Train Acc     : {final_train_acc:.2f}%")
    print(f"   • Final Val Acc       : {final_val_acc:.2f}%")
    print(f"   • Best Val Loss       : {best_val_loss:.4f}")

    # ── Save Model ────────────────────────────────────────────
    print(f"\n💾 Step 6 — Saving Model to '{model_path}'...")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"   ✅ Model saved! File size: {size_mb:.1f} MB")

    print("\n" + "=" * 60)
    print("✅ TRAINING PIPELINE COMPLETE")
    print("=" * 60)

    return model, history, class_names

# AsanaAI — Truly Final Project Plan
### Explainable Deep Learning System for Yoga Pose Recognition
**Course:** CELBC608 — Data Science Laboratory | Semester VI
**Fr. C. Rodrigues Institute of Technology, Vashi, Navi-Mumbai**
**Practicals:** 4 & 5 | Team of 2 | Duration: 8 hours

---

## Complete Changelog — All Review Rounds

| # | Refinement | Source | Decision |
|---|---|---|---|
| 1 | Use ImageDataGenerator instead of NumPy arrays | ChatGPT R1 | ✅ Accepted |
| 2 | Robust Grad-CAM layer detection loop | ChatGPT R1 | ✅ Accepted |
| 3 | Session state guards to prevent retraining | ChatGPT R1 | ✅ Accepted |
| 4 | ROC curves as optional bonus, not core | ChatGPT R1 | ✅ Accepted |
| 5 | Augmentation must not apply to validation | ChatGPT R1 | ✅ Accepted |
| 6 | Two generators + same seed=42 explanation correction | Claude | ✅ Corrected |
| 7 | Manual train/val folder split — no runtime split | ChatGPT R2 | ✅ Accepted |
| 8 | ZIP cleanup with shutil.rmtree + temp_dir in session_state | ChatGPT R2 | ✅ Accepted |
| 9 | Remove st.rerun() — use conditional rendering | ChatGPT R2 | ✅ Accepted |
| 10 | Confusion matrix: separate xticks/yticks rotation | ChatGPT R2 | ✅ Accepted |
| 11 | val_gen.reset() before every metric call | ChatGPT R2 | ✅ Confirmed |
| 12 | shuffle=False on val_gen | Claude | ✅ Confirmed |
| 13 | +1e-8 in Grad-CAM normalization | Claude | ✅ Confirmed |
| 14 | Reset session state on new dataset upload | ChatGPT R3 | ✅ Accepted |
| 15 | Add steps_per_epoch and validation_steps | ChatGPT R3 | ✅ Accepted (clarity) |
| 16 | Remove 'Conv' in layer.name — isinstance check is sufficient | ChatGPT R3 | ✅ Accepted — simpler |
| 17 | Increase epochs from 7 to 10 | ChatGPT R3 | ✅ Accepted |
| 18 | Add EarlyStopping(patience=3, restore_best_weights=True) | ChatGPT R3 | ✅ Accepted |

---

## 🎯 Project Identity

**Title:**
AsanaAI: An Explainable Deep Learning System for Yoga Pose Recognition
using Transfer Learning and Grad-CAM

**Problem Statement:**
Millions of people practice yoga without access to a trained instructor,
leading to incorrect form and potential injury. AsanaAI uses computer vision
to identify yoga poses from a single image and visually explains which body
regions influenced the model's decision — making AI-assisted fitness
guidance accessible to everyone.

**Viva disclaimer:**
*"This is an AI-assisted pose classification and visual feedback tool
intended to support beginner practitioners, not replace certified instructors."*

**Tech Stack:**
Python 3.10 | TensorFlow 2.x | Streamlit | MobileNetV2 |
Grad-CAM | OpenCV | Scikit-learn | Matplotlib | Seaborn

---

## 📁 Final Project Structure

```
AsanaAI/
├── app.py
├── requirements.txt
└── yoga_dataset.zip
    └── yoga_dataset/
        ├── train/
        │   ├── downward_dog/           ← 96 images
        │   ├── warrior/                ← 96 images
        │   ├── tree/                   ← 96 images
        │   ├── cobra/                  ← 96 images
        │   ├── plank/                  ← 96 images
        │   ├── triangle/               ← 96 images
        │   ├── child_pose/             ← 96 images
        │   └── seated_forward_bend/    ← 96 images
        └── val/
            ├── downward_dog/           ← 24 images
            ├── warrior/                ← 24 images
            ├── tree/                   ← 24 images
            ├── cobra/                  ← 24 images
            ├── plank/                  ← 24 images
            ├── triangle/               ← 24 images
            ├── child_pose/             ← 24 images
            └── seated_forward_bend/    ← 24 images
```

**Total: 768 train + 192 val = 960 images. Within 1000 image constraint.**

---

## 🏠 Phase 0 — Pre-Lab Preparation (Night Before, At Home)

Complete every step here before the lab session.
This saves approximately 2 hours of lab time.

---

### Step 1 — Download Dataset
Download the tr1gg3rtrash Yoga Posture Dataset from Kaggle:
https://www.kaggle.com/datasets/tr1gg3rtrash/yoga-posture-dataset
It contains 47 class folders.

---

### Step 2 — Select 8 Classes

| Dataset Folder | Rename To | Why |
|---|---|---|
| Downward-Facing-Dog | downward_dog | Inverted V — unmistakable |
| Warrior-I-Pose | warrior | Wide lunge — distinct |
| Tree-Pose | tree | One-leg balance — unique |
| Cobra-Pose | cobra | Arched back — clear |
| Plank-Pose | plank | Horizontal body — unmistakable |
| Triangle-Pose | triangle | Side stretch — distinct |
| Childs-Pose | child_pose | Curled rest — unique |
| Seated-Forward-Bend | seated_forward_bend | Seated fold — clear |

---

### Step 3 — Manual 80/20 Split

For each class: take exactly 120 images.
- First 96 files alphabetically → `train/classname/`
- Last 24 files → `val/classname/`

**Why this beats validation_split:**
Split is physically visible in the filesystem. Fully deterministic.
Zero hidden logic. One-sentence viva explanation.
Zero risk of any leakage or ordering bug.

---

### Step 4 — ZIP the Dataset

Zip `yoga_dataset/` (containing `train/` and `val/`) as `yoga_dataset.zip`.
Test-unzip once to verify the structure is exactly correct before the lab.

---

### Step 5 — Install Dependencies

```bash
pip install streamlit tensorflow numpy pandas matplotlib seaborn \
            pillow opencv-python scikit-learn
```

Verify GPU:
```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
# Should show: [PhysicalDevice(name='/physical_device:GPU:0', ...)]
```

---

### Step 6 — Prepare Reference Dictionaries At Home

```python
sanskrit_names = {
    'downward_dog': 'Adho Mukha Svanasana',
    'warrior': 'Virabhadrasana I',
    'tree': 'Vrikshasana',
    'cobra': 'Bhujangasana',
    'plank': 'Phalakasana',
    'triangle': 'Trikonasana',
    'child_pose': 'Balasana',
    'seated_forward_bend': 'Paschimottanasana'
}

pose_info = {
    'downward_dog': {
        'benefits': 'Stretches hamstrings, calves, and spine. Strengthens arms and legs. Relieves back pain.',
        'cues': 'Press heels toward floor. Keep spine long. Relax neck between arms.',
        'difficulty': 'Beginner'
    },
    'warrior': {
        'benefits': 'Builds strength in legs and core. Improves balance and focus. Opens hips and chest.',
        'cues': 'Front knee directly over ankle. Back foot at 45 degrees. Arms parallel to floor.',
        'difficulty': 'Beginner'
    },
    'tree': {
        'benefits': 'Improves balance and concentration. Strengthens ankles and calves. Opens hip of raised leg.',
        'cues': 'Fix gaze on a still point. Press foot into inner thigh. Avoid locking standing knee.',
        'difficulty': 'Beginner'
    },
    'cobra': {
        'benefits': 'Strengthens spine and back muscles. Opens chest and shoulders. Stimulates abdominal organs.',
        'cues': 'Keep elbows close to body. Do not crunch neck. Lift chest using back muscles not arms.',
        'difficulty': 'Beginner'
    },
    'plank': {
        'benefits': 'Builds core strength. Tones arms, wrists, and spine. Improves posture.',
        'cues': 'Body forms straight line head to heel. Engage core. Do not let hips sag or rise.',
        'difficulty': 'Beginner'
    },
    'triangle': {
        'benefits': 'Stretches legs, hips, and spine. Opens chest. Improves digestion and relieves stress.',
        'cues': 'Keep both legs straight. Stack shoulders vertically. Look up at raised hand.',
        'difficulty': 'Beginner'
    },
    'child_pose': {
        'benefits': 'Gently stretches hips, thighs, and ankles. Calms the mind. Relieves back pain.',
        'cues': 'Sink hips toward heels. Extend arms forward or alongside body. Breathe deeply.',
        'difficulty': 'Beginner'
    },
    'seated_forward_bend': {
        'benefits': 'Stretches spine, hamstrings, and shoulders. Calms nervous system. Improves digestion.',
        'cues': 'Hinge from hips not waist. Keep spine long. Hold feet or shins, not toes.',
        'difficulty': 'Beginner–Intermediate'
    }
}
```

---

### Step 7 — Create app.py Skeleton At Home

```python
# ── IMPORTS ──────────────────────────────────────────────────────────────
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os
import zipfile
import tempfile
import shutil
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from PIL import Image

# ── GPU SAFETY ────────────────────────────────────────────────────────────
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

# ── PAGE CONFIG ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AsanaAI",
    page_icon="🧘",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── SESSION STATE DEFAULTS ────────────────────────────────────────────────
defaults = {
    'model': None,
    'trained': False,
    'class_names': [],
    'history': None,
    'dataset_path': None,
    'val_gen': None,
    'temp_dir': None,
    'dataset_loaded': False
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ── PASTE REFERENCE DICTS HERE ────────────────────────────────────────────
# sanskrit_names = { ... }
# pose_info = { ... }
```

---

## 🔷 PRACTICAL 4 — Design & Implementation

**Duration: ~4 hours | CO-3 Apply**
**Goal: Working app — dataset upload, training, learning curves**

---

### P4-1: CSS & UI Shell (45 min) — Student A

Inject dark theme CSS via `st.markdown(css, unsafe_allow_html=True)`.

**Design tokens:**
```
Background:    #0a0a0f
Card bg:       rgba(255,255,255,0.04)
Border:        rgba(255,255,255,0.08)
Accent blue:   #667eea
Accent purple: #764ba2
Accent cyan:   #06b6d4
Text primary:  #f1f5f9
Text secondary:#94a3b8
Success:       #10b981
Warning:       #f59e0b
```

**CSS classes to define:**
- `.hero` — large gradient title with subtitle
- `.glass-card` — frosted glass panel, backdrop-filter blur
- `.metric-card` — accuracy/loss value card
- `.section-header` — gradient underline section title
- `.pose-badge` — colored pill for predicted class
- `.info-card` — pose benefits display
- Button: gradient + glow on hover

**App sections — all defined at top, conditionally shown via session_state:**
```
SECTION 1: Dataset Upload          ← always visible
SECTION 2: Dataset Summary         ← if dataset_loaded
SECTION 3: Model Training          ← if dataset_loaded and not trained
SECTION 4: Training Results        ← if trained
SECTION 5: Model Evaluation  [P5]  ← if trained
SECTION 6: Prediction + Grad-CAM [P5] ← if trained
```

**Key principle:** No `st.rerun()` anywhere. Sections render automatically
because they check `st.session_state.trained` on every Streamlit script run.

---

### P4-2: Dataset Upload & ZIP Processing (40 min) — Student B

```python
uploaded_file = st.file_uploader(
    "Upload yoga_dataset.zip",
    type=['zip'],
    help="Must contain yoga_dataset/train/ and yoga_dataset/val/ folders"
)

def process_uploaded_zip(uploaded_file):
    # STEP 1: Clean up old temp directory (refinement #8)
    if st.session_state.temp_dir:
        shutil.rmtree(st.session_state.temp_dir, ignore_errors=True)

    # STEP 2: Reset all training state on new upload (refinement #14)
    st.session_state.trained = False
    st.session_state.model = None
    st.session_state.history = None
    st.session_state.class_names = []
    st.session_state.val_gen = None

    # STEP 3: Write and extract ZIP
    tmp_dir = tempfile.mkdtemp()
    st.session_state.temp_dir = tmp_dir

    zip_path = os.path.join(tmp_dir, "dataset.zip")
    with open(zip_path, 'wb') as f:
        f.write(uploaded_file.read())

    extract_path = os.path.join(tmp_dir, "extracted")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

    # STEP 4: Find root folder containing train/ and val/
    dataset_root = None
    for root, dirs, _ in os.walk(extract_path):
        if 'train' in dirs and 'val' in dirs:
            dataset_root = root
            break

    if dataset_root is None:
        st.error("Invalid ZIP structure. Must contain train/ and val/ folders.")
        return None

    return dataset_root
```

**Dataset summary display after processing:**
- Count images per class via `os.listdir`
- Horizontal bar chart (matplotlib) — class distribution
- 2-row grid of 8 sample images using `st.columns(4)`
- 4 metric cards: Total Classes | Total Images | Train | Val
- Set `st.session_state.dataset_loaded = True`

---

### P4-3: Data Generators (20 min) — Student B

```python
def create_generators(dataset_path):
    # Training: WITH augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        zoom_range=0.1,
        horizontal_flip=True
        # No validation_split — using physical folder split
    )

    # Validation: ONLY rescaling — no augmentation
    val_datagen = ImageDataGenerator(
        rescale=1./255
    )

    train_gen = train_datagen.flow_from_directory(
        os.path.join(dataset_path, 'train'),
        target_size=(128, 128),
        batch_size=16,
        class_mode='categorical',
        shuffle=True,
        seed=42
    )

    val_gen = val_datagen.flow_from_directory(
        os.path.join(dataset_path, 'val'),
        target_size=(128, 128),
        batch_size=16,
        class_mode='categorical',
        shuffle=False   # Essential — keeps label order matching predictions
    )

    return train_gen, val_gen
```

**Why two generators:** One handles augmented training, one handles clean
evaluation. Both point to separate physical directories — zero overlap risk.

**Why shuffle=False on val_gen:** val_gen.classes returns labels in filesystem
order. Shuffling would desync label order from prediction order, making all
metrics — confusion matrix, classification report, ROC — incorrect.

---

### P4-4: Model Architecture (25 min) — Student B

```python
def build_model(num_classes):
    base_model = MobileNetV2(
        input_shape=(128, 128, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False    # Freeze all 154 base layers

    x = base_model.output
    x = GlobalAveragePooling2D()(x)   # Reduces 4×4×1280 → 1280 values
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)               # Prevents overfitting on small dataset
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
```

**Architecture summary:**
```
Input (128×128×3)
    ↓
MobileNetV2 Base [FROZEN — 154 layers, ImageNet weights]
Depthwise separable convolutions + inverted residuals
    ↓
GlobalAveragePooling2D  →  1280 values
    ↓
Dense(128, ReLU)
    ↓
Dropout(0.3)
    ↓
Dense(8, Softmax)  →  8 yoga pose classes
```

---

### P4-5: Training with Live Progress (45 min) — Student A

**Custom Streamlit callback:**
```python
class StreamlitProgressCallback(tf.keras.callbacks.Callback):
    def __init__(self, total_epochs, progress_bar, status_text):
        super().__init__()
        self.total_epochs = total_epochs
        self.progress_bar = progress_bar
        self.status_text = status_text

    def on_epoch_end(self, epoch, logs=None):
        progress = (epoch + 1) / self.total_epochs
        self.progress_bar.progress(progress)
        self.status_text.markdown(
            f"**Epoch {epoch+1}/{self.total_epochs}** — "
            f"Train Acc: `{logs['accuracy']:.4f}` | "
            f"Val Acc: `{logs['val_accuracy']:.4f}` | "
            f"Val Loss: `{logs['val_loss']:.4f}`"
        )
```

**Training trigger — session state guard + EarlyStopping:**
```python
if st.session_state.dataset_loaded and not st.session_state.trained:
    if st.button("🚀 Begin Training", type="primary"):

        progress_bar = st.progress(0)
        status_text = st.empty()

        train_gen, val_gen = create_generators(st.session_state.dataset_path)
        num_classes = len(train_gen.class_indices)
        model = build_model(num_classes)

        # Explicit steps for academic clarity (refinement #15)
        steps_per_epoch = train_gen.samples // train_gen.batch_size
        validation_steps = val_gen.samples // val_gen.batch_size

        EPOCHS = 10   # Increased from 7 (refinement #17)

        callbacks = [
            StreamlitProgressCallback(EPOCHS, progress_bar, status_text),
            # EarlyStopping prevents overfitting, restores best weights
            # (refinement #18)
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True,
                verbose=0
            )
        ]

        history = model.fit(
            train_gen,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_gen,
            validation_steps=validation_steps,
            epochs=EPOCHS,
            callbacks=callbacks
        )

        # Persist in session state
        st.session_state.model = model
        st.session_state.trained = True
        st.session_state.history = history.history
        st.session_state.class_names = list(train_gen.class_indices.keys())
        st.session_state.val_gen = val_gen
        # No st.rerun() — sections below auto-render (refinement #9)

elif st.session_state.trained:
    st.success("✅ Model trained. Scroll down for results and evaluation.")
```

**Training results display — Section 4 (visible when trained):**

4 metric cards in a row:
```python
col1, col2, col3, col4 = st.columns(4)
col1.metric("🎯 Train Accuracy", f"{history['accuracy'][-1]:.2%}")
col2.metric("✅ Val Accuracy",   f"{history['val_accuracy'][-1]:.2%}")
col3.metric("📉 Train Loss",     f"{history['loss'][-1]:.4f}")
col4.metric("📊 Val Loss",       f"{history['val_loss'][-1]:.4f}")
```

Accuracy + Loss curves side by side (matplotlib, dark background).
Train line: solid `#667eea`. Val line: dashed `#06b6d4` / `#f59e0b`.

---

## 🔷 PRACTICAL 5 — Evaluation + Grad-CAM + Prediction

**Duration: ~4 hours | CO-3 Analyze**
**Goal: Full metrics + complete prediction pipeline with Grad-CAM**

---

### P5-1: Confusion Matrix (30 min) — Student A

```python
def plot_confusion_matrix(model, val_gen, class_names):
    val_gen.reset()   # Always reset before full-pass (refinement #11)
    y_pred_probs = model.predict(val_gen, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = val_gen.classes

    cm = confusion_matrix(y_true, y_pred)
    labels = [c.replace('_', ' ').title() for c in class_names]

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor('#0a0a0f')
    ax.set_facecolor('#111827')

    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=labels, yticklabels=labels,
        ax=ax, linewidths=0.5, linecolor='#1f2937'
    )

    ax.set_title('Confusion Matrix — Validation Set',
                 color='white', fontsize=14, pad=20)
    ax.set_xlabel('Predicted Label', color='#94a3b8', labelpad=10)
    ax.set_ylabel('True Label', color='#94a3b8', labelpad=10)

    # Separate rotation for x and y (refinement #11)
    plt.xticks(rotation=45, ha='right', color='#94a3b8')
    plt.yticks(rotation=0, color='#94a3b8')

    plt.tight_layout()
    return fig
```

---

### P5-2: Classification Report (20 min) — Student B

```python
def display_classification_report(model, val_gen, class_names):
    val_gen.reset()
    y_pred = np.argmax(model.predict(val_gen, verbose=0), axis=1)
    y_true = val_gen.classes

    labels = [c.replace('_', ' ').title() for c in class_names]
    report_dict = classification_report(
        y_true, y_pred,
        target_names=labels,
        output_dict=True
    )

    df = pd.DataFrame(report_dict).transpose()
    df = df.drop(['accuracy'], errors='ignore')
    df = df.round(3)

    st.dataframe(
        df.style
          .background_gradient(
              cmap='Blues',
              subset=['precision', 'recall', 'f1-score'])
          .format({
              'precision': '{:.3f}',
              'recall': '{:.3f}',
              'f1-score': '{:.3f}',
              'support': '{:.0f}'
          }),
        use_container_width=True
    )
```

---

### P5-3: Grad-CAM Implementation (45 min) — Student B

**Layer detection — simplified (refinement #16):**
```python
def get_last_conv_layer(model):
    # isinstance check is sufficient — classification head has no Conv2D layers
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found in model.")
```

**Why simplify:** The custom head (Dense, Dropout, GlobalAveragePooling2D)
contains no Conv2D layers. So `isinstance` alone correctly finds the last
conv layer in the MobileNetV2 base. The extra name check was unnecessary
complexity for this architecture.

**Grad-CAM computation — complete:**
```python
def generate_gradcam(model, img_array, class_idx):
    last_conv_name = get_last_conv_layer(model)

    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(last_conv_name).output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_idx]

    # Gradients of class score w.r.t. conv feature maps
    grads = tape.gradient(loss, conv_outputs)

    # Pool gradients across spatial dimensions
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight feature maps by gradient importance
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # ReLU — retain only positive influence
    heatmap = tf.maximum(heatmap, 0)

    # Normalize to [0,1] — +1e-8 prevents division by zero (refinement #13)
    heatmap = heatmap / (tf.math.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()

    # Resize to image dimensions
    heatmap_resized = cv2.resize(heatmap, (128, 128))

    # Apply JET colormap: blue=low importance, red=high importance
    heatmap_colored = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Blend heatmap with original image
    original = np.uint8(img_array[0] * 255)
    overlay = cv2.addWeighted(original, 0.6, heatmap_colored, 0.4, 0)

    return heatmap_colored, overlay
```

---

### P5-4: Prediction UI (35 min) — Student A

```python
if st.session_state.trained:
    pred_image = st.file_uploader(
        "Upload a yoga pose image to classify",
        type=['jpg', 'jpeg', 'png'],
        key='prediction_upload'
    )

    if pred_image:
        # Preprocess
        img = Image.open(pred_image).convert('RGB')
        img_resized = img.resize((128, 128))
        img_array = np.expand_dims(
            np.array(img_resized) / 255.0, axis=0
        )

        # Predict
        predictions = st.session_state.model.predict(img_array, verbose=0)
        class_idx = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][class_idx])
        predicted_class = st.session_state.class_names[class_idx]
        display_name = predicted_class.replace('_', ' ').title()
        sanskrit = sanskrit_names.get(predicted_class, '')

        # Grad-CAM
        heatmap, overlay = generate_gradcam(
            st.session_state.model, img_array, class_idx
        )

        # Prediction badge
        st.markdown(f"""
        <div class="pose-badge">
            🧘 {display_name} &nbsp;|&nbsp;
            <em>{sanskrit}</em> &nbsp;|&nbsp;
            Confidence: {confidence:.1%}
        </div>
        """, unsafe_allow_html=True)

        # Three-column image display
        c1, c2, c3 = st.columns(3)
        c1.image(img_resized,   caption="📷 Original",          use_column_width=True)
        c2.image(heatmap,       caption="🔥 Grad-CAM Heatmap",  use_column_width=True)
        c3.image(overlay,       caption="🔍 Overlay",           use_column_width=True)

        # Probability bar chart
        prob_df = pd.DataFrame({
            'Pose': [c.replace('_', ' ').title()
                     for c in st.session_state.class_names],
            'Confidence': predictions[0]
        }).sort_values('Confidence', ascending=True)

        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor('#0a0a0f')
        ax.set_facecolor('#111827')
        bar_colors = [
            '#10b981' if p == display_name else '#667eea'
            for p in prob_df['Pose']
        ]
        ax.barh(prob_df['Pose'], prob_df['Confidence'],
                color=bar_colors, edgecolor='none')
        ax.set_xlabel('Confidence', color='#94a3b8')
        ax.set_title('Class Probability Distribution',
                     color='white', fontsize=12)
        ax.tick_params(colors='#94a3b8')
        ax.grid(True, axis='x', alpha=0.15)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Pose info card
        info = pose_info.get(predicted_class, {})
        if info:
            st.markdown(f"""
            <div class="info-card">
                <h4>📋 {display_name} ({sanskrit})</h4>
                <p><strong>Benefits:</strong> {info['benefits']}</p>
                <p><strong>Form Cues:</strong> {info['cues']}</p>
                <p><strong>Difficulty:</strong> {info['difficulty']}</p>
                <hr style="opacity:0.2">
                <p style="opacity:0.5; font-size:0.75em;">
                ⚠️ This tool supports beginner practitioners.
                It does not replace certified yoga instruction.
                </p>
            </div>
            """, unsafe_allow_html=True)
```

---

### P5-5: ROC Curves (25 min — OPTIONAL, implement last)

```python
def plot_roc_curves(model, val_gen, class_names):
    val_gen.reset()
    y_pred_probs = model.predict(val_gen, verbose=0)
    y_true = val_gen.classes
    num_classes = len(class_names)

    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor('#0a0a0f')
    ax.set_facecolor('#111827')

    palette = ['#667eea','#764ba2','#06b6d4','#10b981',
               '#f59e0b','#ef4444','#8b5cf6','#ec4899']

    for i, (name, color) in enumerate(zip(class_names, palette)):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
        roc_auc = auc(fpr, tpr)
        label = f"{name.replace('_',' ').title()} (AUC = {roc_auc:.2f})"
        ax.plot(fpr, tpr, color=color, linewidth=2, label=label)

    ax.plot([0,1],[0,1], 'white', linestyle='--', alpha=0.3, linewidth=1)
    ax.set_xlabel('False Positive Rate', color='#94a3b8')
    ax.set_ylabel('True Positive Rate', color='#94a3b8')
    ax.set_title('ROC Curves — One-vs-Rest', color='white', fontsize=14)
    ax.legend(loc='lower right', fontsize=8,
              facecolor='#1f2937', labelcolor='white', framealpha=0.8)
    ax.tick_params(colors='#94a3b8')
    ax.grid(True, alpha=0.1)

    plt.tight_layout()
    return fig
```

**Time rule:** If fewer than 25 minutes remain when you reach this step,
skip it. Confusion matrix + classification report is fully sufficient for
Practical 5. ROC is a bonus mark only.

---

## 📦 requirements.txt

```
streamlit>=1.28.0
tensorflow>=2.12.0
numpy>=1.23.0
pandas>=1.5.0
matplotlib>=3.6.0
seaborn>=0.12.0
pillow>=9.0.0
opencv-python>=4.7.0
scikit-learn>=1.2.0
```

---

## 🎨 Complete User Flow

```
User opens app
    ↓
Dark hero header — "🧘 AsanaAI"
    ↓
Uploads yoga_dataset.zip
    ↓
App:
  • Cleans old temp directory
  • Resets all training state ← NEW
  • Extracts ZIP
  • Finds train/ and val/
  • Shows dataset summary:
      8 classes | 960 images | 768 train | 192 val
      Class distribution bar chart
      8 sample images grid
    ↓
Clicks "🚀 Begin Training"
    ↓
Progress bar fills — 10 epochs max
Status updates per epoch:
    "Epoch 5/10 — Train Acc: 0.8234 | Val Acc: 0.7917 | Val Loss: 0.5421"
EarlyStopping may halt early if val_loss stops improving ← NEW
    ↓
Training complete — sections auto-render (no st.rerun)
    ↓
4 metric cards: Train Acc | Val Acc | Train Loss | Val Loss
Accuracy curve + Loss curve side by side
    ↓
Confusion matrix (8×8 heatmap)
Classification report (styled table)
[Optional] ROC curves
    ↓
User uploads yoga pose photo
App predicts:
    "🧘 Tree Pose — Vrikshasana | Confidence: 89.3%"
Three images: Original | Heatmap | Overlay
Probability bar chart (highlighted predicted class)
Pose info card: Benefits | Form Cues | Difficulty | Disclaimer
```

---

## ⏱️ 8-Hour Lab Budget

| Task | Duration | Owner |
|---|---|---|
| GPU verify + environment check | 10 min | Both |
| CSS + UI skeleton (P4-1) | 45 min | Student A |
| ZIP upload + processing (P4-2) | 40 min | Student B |
| Data generators (P4-3) | 20 min | Student B |
| Model architecture (P4-4) | 25 min | Student B |
| Training callback + results UI (P4-5) | 45 min | Student A |
| Confusion matrix (P5-1) | 30 min | Student A |
| Classification report (P5-2) | 20 min | Student B |
| Grad-CAM function (P5-3) | 45 min | Student B |
| Prediction UI (P5-4) | 35 min | Student A |
| Integration + bug fixes | 40 min | Both |
| ROC curves if time permits (P5-5) | 25 min | Either |
| Demo rehearsal + viva prep | 20 min | Both |
| **Total** | **~8 hrs** | |

**If running behind:** Reduce CSS complexity first.
Never compromise the ML logic or Grad-CAM for UI polish.

---

## ✅ Pre-Lab Checklist

### Night Before (At Home):
- [ ] Kaggle dataset downloaded
- [ ] 8 class folders selected and renamed correctly
- [ ] Manual 80/20 split: 96 train + 24 val per class
- [ ] yoga_dataset.zip created and test-unzipped to verify structure
- [ ] sanskrit_names dict written in app.py skeleton
- [ ] pose_info dict written in app.py skeleton
- [ ] Dependencies installed and verified
- [ ] GPU check: `tf.config.list_physical_devices('GPU')` returns device
- [ ] app.py skeleton created with imports + session state defaults

### Start of Lab:
- [ ] Both teammates have yoga_dataset.zip accessible
- [ ] GPU confirmed available
- [ ] Task division agreed: A = UI, Training UI, Prediction Display; B = Model, Generators, Evaluation
- [ ] This plan open for reference

---

## 💬 Complete Viva Q&A

**Q: Why MobileNetV2?**
MobileNetV2 uses depthwise separable convolutions and inverted residual
blocks — reducing computation by 8–9× versus standard convolutions while
maintaining accuracy. Designed for mobile and edge deployment, ideal for
our 4GB GPU constraint.

**Q: Why freeze the base layers?**
ImageNet pretraining encodes strong feature detectors: edges, textures,
shapes. Human poses are well-represented by these. Freezing prevents
catastrophic forgetting on our 768-image training set and lets the custom
head learn pose-specific mappings without corrupting pretrained weights.

**Q: Why 128×128 not 224×224?**
Reduces memory by ~75% and training time proportionally while retaining
sufficient detail for gross body pose classification. A deliberate and
justified constraint for 4GB VRAM.

**Q: Why EarlyStopping?**
To prevent overfitting on a small dataset and to automatically restore the
best weights if validation loss starts increasing — giving us the best
checkpoint without manual epoch tuning.

**Q: Why manual train/val split?**
Split is physically visible in the filesystem. Fully deterministic.
No hidden runtime logic. Zero risk of data leakage. Easiest possible
viva explanation: "We split 80/20 manually before training."

**Q: Why no augmentation on validation?**
Validation measures real-world performance. Augmenting it introduces
artificial variations not present in deployment, making metrics unreliable.
We augment training to improve generalization; evaluate on clean data
to measure true performance.

**Q: What does Grad-CAM show?**
Grad-CAM computes gradients of the predicted class score with respect to
the last conv layer's feature maps. These gradients show which spatial
locations contributed most to the prediction. For Tree Pose it highlights
the raised leg and arm extension — exactly what distinguishes it visually.

**Q: What are the limitations?**
Static image only — no video. Performance degrades on unusual angles, heavy
clothing, poor lighting. 24 validation images per class is small. Not tested
outside the dataset's visual distribution.

**Q: Is this an injury prevention tool?**
No. This is an AI-assisted pose classification and visual feedback tool
to support beginner practitioners. It does not replace certified instruction.

---

## 📄 Report Outline

| Section | Content | Length |
|---|---|---|
| Abstract | Problem, approach, results summary | 150 words |
| Introduction | Motivation, problem statement, objectives | 1 page |
| Literature Review | 3 cited papers on pose estimation / yoga AI | 1 page |
| Dataset | Source, 8 classes, manual split rationale, preprocessing | 0.5 page |
| System Architecture | MobileNetV2 layer diagram, Grad-CAM explanation | 1.5 pages |
| Results | Accuracy/loss curves, confusion matrix, classification report, ROC | 1.5 pages |
| Discussion | Accuracy analysis, per-class performance, limitations | 0.5 page |
| Conclusion & Future Scope | Summary + video/real-time as future work | 0.5 page |

---

## 🏆 What This Demonstrates to Examiners

- Supervised learning pipeline
- Transfer learning with frozen backbone
- Correct augmentation strategy (train only)
- Deterministic dataset handling
- GPU memory management
- Multi-class evaluation: accuracy, precision, recall, F1, AUC
- Explainable AI via Grad-CAM
- Early stopping and best weight restoration
- Production-grade session state management
- Web application deployment via Streamlit

This exceeds standard undergraduate lab expectations.

---
*This plan is final. All known technical risks have been identified,
reviewed across 3 rounds, and resolved. No further architectural
changes are expected before implementation.*

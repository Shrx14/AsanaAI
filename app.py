# ═══════════════════════════════════════════════════════════════════════════
# AsanaAI — Practicals 4 & 5
# Explainable Deep Learning System for Yoga Pose Recognition
# ═══════════════════════════════════════════════════════════════════════════

import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import json
import os
import zipfile
import tempfile
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
from PIL import Image

# ── CONSTANTS ────────────────────────────────────────────────────────────────
SAVE_DIR     = "asanai_saved"
MODEL_PATH   = os.path.join(SAVE_DIR, "model_weights.pt")
HISTORY_PATH = os.path.join(SAVE_DIR, "training_history.json")
CLASSES_PATH = os.path.join(SAVE_DIR, "class_names.json")
IMG_SIZE     = 160
BATCH_SIZE   = 12
EPOCHS       = 24
PATIENCE     = 6
LR_HEAD      = 5e-4
LR_FINE_TUNE = 2e-5
WEIGHT_DECAY = 1e-4
LABEL_SMOOTH = 0.05
GRAD_CLIP    = 1.0
FINE_TUNE_AT_EPOCH = 10
UNFREEZE_FROM_BLOCK = 16
CLASS_WEIGHT_IMBALANCE_THRESHOLD = 1.5
MIXUP_ALPHA = 0.2
MIXUP_EPOCHS = 10

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

os.makedirs(SAVE_DIR, exist_ok=True)

# ── GPU SETUP ────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GPU_AVAILABLE = torch.cuda.is_available()

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AsanaAI",
    page_icon="🧘",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── SESSION STATE DEFAULTS ───────────────────────────────────────────────────
_defaults = {
    "model":          None,
    "trained":        False,
    "class_names":    [],
    "history":        None,
    "temperature":    1.0,
    "dataset_path":   None,
    "val_loader":     None,
    "dataset_loaded": False,
    "temp_dir":       None,
    "model_source":   None,   # "file" | "trained"
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


def load_image_rgb(image_source):
    """Open an image and safely convert to RGB across PNG palette edge-cases."""
    with Image.open(image_source) as img:
        # Pillow warns on some palette PNGs unless they pass through RGBA first.
        if (img.mode == "P" and
            "transparency" in img.info and
            isinstance(img.info["transparency"], bytes)):
            img = img.convert("RGBA")
        return img.convert("RGB")

# ═══════════════════════════════════════════════════════════════════════════
# REFERENCE DATA
# ═══════════════════════════════════════════════════════════════════════════

sanskrit_names = {
    "downward_dog":        "Adho Mukha Svanasana",
    "warrior":             "Virabhadrasana I",
    "tree":                "Vrikshasana",
    "cobra":               "Bhujangasana",
    "plank":               "Phalakasana",
    "triangle":            "Trikonasana",
    "child_pose":          "Balasana",
    "seated_forward_bend": "Paschimottanasana",
}

pose_info = {
    "downward_dog": {
        "benefits":   "Stretches hamstrings, calves, and spine. Strengthens arms and legs. Relieves back pain.",
        "cues":       "Press heels toward the floor. Keep spine long. Relax neck between arms.",
        "difficulty": "Beginner",
        "color":      "#667eea",
    },
    "warrior": {
        "benefits":   "Builds strength in legs and core. Improves balance and focus. Opens hips and chest.",
        "cues":       "Front knee directly over ankle. Back foot at 45°. Arms parallel to floor.",
        "difficulty": "Beginner",
        "color":      "#f59e0b",
    },
    "tree": {
        "benefits":   "Improves balance and concentration. Strengthens ankles and calves. Opens hip of raised leg.",
        "cues":       "Fix gaze on a still point. Press foot into inner thigh. Avoid locking standing knee.",
        "difficulty": "Beginner",
        "color":      "#10b981",
    },
    "cobra": {
        "benefits":   "Strengthens spine and back muscles. Opens chest and shoulders. Stimulates abdominal organs.",
        "cues":       "Keep elbows close to body. Do not crunch neck. Lift chest using back muscles, not arms.",
        "difficulty": "Beginner",
        "color":      "#ef4444",
    },
    "plank": {
        "benefits":   "Builds core strength. Tones arms, wrists, and spine. Improves posture.",
        "cues":       "Body forms a straight line head to heel. Engage core. Do not let hips sag or rise.",
        "difficulty": "Beginner",
        "color":      "#8b5cf6",
    },
    "triangle": {
        "benefits":   "Stretches legs, hips, and spine. Opens chest. Improves digestion and relieves stress.",
        "cues":       "Keep both legs straight. Stack shoulders vertically. Look up at raised hand.",
        "difficulty": "Beginner",
        "color":      "#06b6d4",
    },
    "child_pose": {
        "benefits":   "Gently stretches hips, thighs, and ankles. Calms the mind. Relieves back pain.",
        "cues":       "Sink hips toward heels. Extend arms forward or alongside body. Breathe deeply.",
        "difficulty": "Beginner",
        "color":      "#ec4899",
    },
    "seated_forward_bend": {
        "benefits":   "Stretches spine, hamstrings, and shoulders. Calms nervous system. Improves digestion.",
        "cues":       "Hinge from hips not waist. Keep spine long. Hold feet or shins, not toes.",
        "difficulty": "Beginner–Intermediate",
        "color":      "#f97316",
    },
}

# ═══════════════════════════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
  body, .main { background: #0a0a0f !important; color: #f1f5f9; }

  .hero {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border: 1px solid rgba(102,126,234,0.3);
    padding: 2.5rem 2rem 2rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
  }
  .hero h1 { font-size: 2.8rem; font-weight: 700; color: #fff; margin: 0 0 .4rem; }
  .hero p  { color: #94a3b8; font-size: 1.05rem; margin: 0; }
  .hero .badge {
    display: inline-block;
    background: rgba(102,126,234,.15);
    border: 1px solid rgba(102,126,234,.4);
    color: #a5b4fc;
    padding: .2rem .7rem;
    border-radius: 20px;
    font-size: .78rem;
    margin-right: .4rem;
    margin-top: .6rem;
  }

  .status-bar {
    display: flex; align-items: center; gap: 1rem;
    background: rgba(16,185,129,.06);
    border: 1px solid rgba(16,185,129,.25);
    border-radius: 10px; padding: .75rem 1.25rem;
    margin-bottom: 1.5rem;
  }
  .status-bar.warn {
    background: rgba(245,158,11,.06);
    border-color: rgba(245,158,11,.25);
  }
  .status-dot {
    width: 10px; height: 10px; border-radius: 50%;
    background: #10b981; flex-shrink: 0;
    box-shadow: 0 0 6px #10b981;
  }
  .status-dot.warn { background: #f59e0b; box-shadow: 0 0 6px #f59e0b; }
  .status-text { color: #e2e8f0; font-size: .9rem; }

  .section-header {
    font-size: 1.3rem; font-weight: 600; color: #f1f5f9;
    margin: 1.8rem 0 1rem;
    padding-bottom: .45rem;
    border-bottom: 1px solid rgba(255,255,255,.07);
    letter-spacing: .01em;
  }
  .section-header span { color: #667eea; }

  .glass-card {
    background: rgba(255,255,255,.03);
    border: 1px solid rgba(255,255,255,.07);
    border-radius: 12px; padding: 1.5rem;
    margin: 1rem 0;
  }

  .metric-row { display: flex; gap: .75rem; flex-wrap: wrap; margin: .75rem 0; }
  .metric-box {
    flex: 1; min-width: 110px;
    background: rgba(102,126,234,.08);
    border: 1px solid rgba(102,126,234,.2);
    border-radius: 10px; padding: 1rem;
    text-align: center;
  }
  .metric-box .mval { font-size: 1.6rem; font-weight: 700; color: #a5b4fc; }
  .metric-box .mlbl { font-size: .78rem; color: #64748b; margin-top: .2rem; }

  /* Prediction result */
  .pred-card {
    background: linear-gradient(135deg, rgba(102,126,234,.12), rgba(118,75,162,.08));
    border: 1px solid rgba(102,126,234,.3);
    border-radius: 14px; padding: 1.5rem 2rem;
    margin: 1rem 0;
  }
  .pred-title { font-size: 1.5rem; font-weight: 700; color: #fff; margin: 0 0 .25rem; }
  .pred-sub { color: #06b6d4; font-style: italic; font-size: 1rem; margin: 0; }
  .conf-bar-wrap {
    background: rgba(255,255,255,.06); border-radius: 20px;
    height: 8px; margin: .75rem 0 .25rem; overflow: hidden;
  }
  .conf-bar { height: 100%; border-radius: 20px;
    background: linear-gradient(90deg, #667eea, #06b6d4); }
  .conf-label { color: #94a3b8; font-size: .82rem; }

  .top3-row {
    display: flex; align-items: center; gap: .7rem;
    padding: .4rem 0;
    border-bottom: 1px solid rgba(255,255,255,.04);
  }
  .top3-label { flex: 1; font-size: .85rem; color: #cbd5e1; }
  .top3-pct { font-size: .85rem; font-weight: 600; color: #a5b4fc;
    min-width: 42px; text-align: right; }
  .top3-bar-bg { width: 100px; background: rgba(255,255,255,.06);
    border-radius: 4px; height: 5px; }
  .top3-bar { height: 5px; border-radius: 4px; background: #667eea; }

  .info-card {
    background: rgba(16,185,129,.04);
    border-left: 3px solid #10b981;
    border-radius: 0 10px 10px 0;
    padding: 1.25rem 1.5rem; margin-top: 1rem;
    line-height: 1.8;
  }
  .info-card h4 { color: #10b981; margin: 0 0 .75rem; font-size: 1rem; }
  .info-card p  { color: #cbd5e1; margin: .3rem 0; font-size: .9rem; }
  .info-card strong { color: #f1f5f9; }
  .disclaimer { color: #475569; font-size: .75rem; margin-top: .75rem; }

  .stButton > button {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    color: #fff !important; border: none !important;
    padding: .65rem 1.8rem !important; border-radius: 8px !important;
    font-weight: 600 !important; font-size: .95rem !important;
    transition: all .25s ease !important;
    box-shadow: 0 3px 12px rgba(102,126,234,.3) !important;
  }
  .stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(102,126,234,.5) !important;
  }

  .aug-grid { display: grid; grid-template-columns: repeat(3,1fr); gap: .5rem; }

  footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# PYTORCH CLASSES
# ═══════════════════════════════════════════════════════════════════════════

class ImageDataset(Dataset):
    """Loads images from class-named subdirectories."""

    def __init__(self, root_dir, transform=None):
        self.transform   = transform
        self.images      = []
        self.labels      = []
        self.class_names = sorted(
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        )
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}

        for cls in self.class_names:
            cls_dir = os.path.join(root_dir, cls)
            for fname in sorted(os.listdir(cls_dir)):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.images.append(os.path.join(cls_dir, fname))
                    self.labels.append(self.class_to_idx[cls])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = load_image_rgb(self.images[idx])
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


class YogaPoseModel(nn.Module):
    """MobileNetV2 backbone + lightweight classification head."""

    def __init__(self, num_classes):
        super().__init__()
        self.base_model = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.DEFAULT
        )
        for p in self.base_model.parameters():
            p.requires_grad = False

        in_features = self.base_model.classifier[1].in_features
        self.base_model.classifier = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.base_model(x)


# ═══════════════════════════════════════════════════════════════════════════
# TRANSFORMS
# ═══════════════════════════════════════════════════════════════════════════

TRAIN_TRANSFORMS = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.85, 1.0), ratio=(0.85, 1.15)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=10, translate=(0.04, 0.04), scale=(0.95, 1.05)),
    transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08, hue=0.02),
    transforms.RandomAdjustSharpness(sharpness_factor=1.3, p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

VAL_TRANSFORMS = transforms.Compose([
    transforms.Resize(int(IMG_SIZE * 1.15)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

PRED_TRANSFORM = transforms.Compose([
    transforms.Resize(int(IMG_SIZE * 1.15)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# ═══════════════════════════════════════════════════════════════════════════
# PERSISTENCE HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def save_model(model, history, class_names):
    """Persist model weights, training history, and class order to disk."""
    torch.save(model.state_dict(), MODEL_PATH)
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f)
    with open(CLASSES_PATH, "w") as f:
        json.dump(class_names, f)


def load_model_from_disk():
    """
    Load saved model + metadata if all three checkpoint files exist.
    Returns (model, history, class_names) or (None, None, None).
    """
    if not (os.path.exists(MODEL_PATH) and
            os.path.exists(HISTORY_PATH) and
            os.path.exists(CLASSES_PATH)):
        return None, None, None

    with open(CLASSES_PATH) as f:
        class_names = json.load(f)
    with open(HISTORY_PATH) as f:
        history = json.load(f)

    model = YogaPoseModel(num_classes=len(class_names)).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model, history, class_names


# ── Auto-load on first run ───────────────────────────────────────────────────
if not st.session_state.trained:
    _model, _hist, _cls = load_model_from_disk()
    if _model is not None:
        st.session_state.model        = _model
        st.session_state.history      = _hist
        st.session_state.class_names  = _cls
        st.session_state.temperature  = float(_hist.get("temperature", 1.0))
        st.session_state.trained      = True
        st.session_state.model_source = "file"


# ═══════════════════════════════════════════════════════════════════════════
# DATASET HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def process_uploaded_zip(uploaded_file):
    """Extract ZIP, validate train/val structure, return dataset root path."""
    if st.session_state.temp_dir:
        shutil.rmtree(st.session_state.temp_dir, ignore_errors=True)

    # Reset training-related state (but NOT the loaded model)
    st.session_state.val_loader     = None
    st.session_state.dataset_loaded = False

    tmp_dir  = tempfile.mkdtemp()
    st.session_state.temp_dir = tmp_dir

    zip_path = os.path.join(tmp_dir, "dataset.zip")
    with open(zip_path, "wb") as f:
        f.write(uploaded_file.read())

    extract_path = os.path.join(tmp_dir, "extracted")
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_path)
    except Exception as e:
        st.error(f"Failed to extract ZIP: {e}")
        return None

    dataset_root = None
    for root, dirs, _ in os.walk(extract_path):
        if "train" in dirs and "val" in dirs:
            dataset_root = root
            break

    if dataset_root is None:
        st.error("Invalid ZIP structure — must contain `train/` and `val/` folders.")
        return None

    return dataset_root


def create_data_loaders(dataset_path, use_weighted_sampler=False):
    train_ds = ImageDataset(os.path.join(dataset_path, "train"), TRAIN_TRANSFORMS)
    val_ds   = ImageDataset(os.path.join(dataset_path, "val"),   VAL_TRANSFORMS)

    if use_weighted_sampler:
        counts = np.bincount(train_ds.labels, minlength=len(train_ds.class_names))
        per_class_weights = 1.0 / np.maximum(counts, 1)
        sample_weights = torch.tensor(
            [per_class_weights[label] for label in train_ds.labels],
            dtype=torch.double,
        )
        sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            sampler=sampler,
            num_workers=0,
        )
    else:
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)

    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    return train_loader, val_loader, train_ds.class_names


# ═══════════════════════════════════════════════════════════════════════════
# EDA HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def count_images(split_path):
    counts = {}
    if not os.path.exists(split_path):
        return counts
    for cls in sorted(os.listdir(split_path)):
        d = os.path.join(split_path, cls)
        if os.path.isdir(d):
            counts[cls] = len([f for f in os.listdir(d)
                                if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    return counts


def show_dataset_summary(dataset_path):
    train_counts = count_images(os.path.join(dataset_path, "train"))
    val_counts   = count_images(os.path.join(dataset_path, "val"))

    total_train = sum(train_counts.values())
    total_val   = sum(val_counts.values())

    # ── Metric pills ───────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="metric-row">
      <div class="metric-box"><div class="mval">{len(train_counts)}</div><div class="mlbl">Classes</div></div>
      <div class="metric-box"><div class="mval">{total_train + total_val}</div><div class="mlbl">Total images</div></div>
      <div class="metric-box"><div class="mval">{total_train}</div><div class="mlbl">Train</div></div>
      <div class="metric-box"><div class="mval">{total_val}</div><div class="mlbl">Validation</div></div>
      <div class="metric-box"><div class="mval">{round(total_train/(total_train+total_val)*100) if (total_train+total_val)>0 else 0}%</div><div class="mlbl">Train ratio</div></div>
    </div>
    """, unsafe_allow_html=True)

    col_chart, col_samples = st.columns([1, 1])

    with col_chart:
        st.write("**Class distribution**")
        classes     = list(train_counts.keys())
        train_vals  = [train_counts[c] for c in classes]
        val_vals    = [val_counts.get(c, 0) for c in classes]
        labels      = [c.replace("_", " ").title() for c in classes]

        fig, ax = plt.subplots(figsize=(5, max(3, len(classes) * 0.55)))
        fig.patch.set_facecolor("#0a0a0f")
        ax.set_facecolor("#111827")
        y = np.arange(len(labels))
        w = 0.38
        ax.barh(y - w/2, train_vals, w, color="#667eea", label="Train")
        ax.barh(y + w/2, val_vals,   w, color="#06b6d4", label="Val")
        ax.set_yticks(y); ax.set_yticklabels(labels)
        ax.set_xlabel("Images", color="#94a3b8")
        ax.set_title("Images per class", color="white", fontsize=11, pad=10)
        ax.legend(facecolor="#1f2937", labelcolor="white", fontsize=9)
        ax.tick_params(colors="#94a3b8"); ax.grid(axis="x", alpha=0.1)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col_samples:
        st.write("**Sample images (one per class)**")
        train_path = os.path.join(dataset_path, "train")
        cols       = st.columns(4)
        for i, cls in enumerate(classes[:8]):
            cls_dir = os.path.join(train_path, cls)
            imgs    = sorted(f for f in os.listdir(cls_dir)
                             if f.lower().endswith((".jpg", ".jpeg", ".png")))
            if imgs:
                img = load_image_rgb(os.path.join(cls_dir, imgs[0]))
                cols[i % 4].image(img, caption=cls.replace("_", " ").title(),
                                  width="stretch")


def show_augmentation_preview(dataset_path):
    """Show the same image augmented 6 times to demonstrate train transforms."""
    train_path = os.path.join(dataset_path, "train")
    classes    = sorted(os.listdir(train_path))
    if not classes:
        return

    cls_dir = os.path.join(train_path, classes[0])
    imgs    = sorted(f for f in os.listdir(cls_dir)
                     if f.lower().endswith((".jpg", ".jpeg", ".png")))
    if not imgs:
        return

    base_img = load_image_rgb(os.path.join(cls_dir, imgs[0]))
    st.write(f"**Showing 6 random augmentations of one `{classes[0].replace('_',' ').title()}` image**")
    st.caption("Each is unique — rotation, flip, zoom, brightness vary randomly. "
               "Validation images are never augmented.")

    cols = st.columns(6)
    for col in cols:
        aug = TRAIN_TRANSFORMS(base_img)
        # Denormalize for display
        mean = torch.tensor(IMAGENET_MEAN).view(3,1,1)
        std  = torch.tensor(IMAGENET_STD).view(3,1,1)
        disp = torch.clamp(aug * std + mean, 0, 1)
        disp = (disp.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        col.image(disp, width="stretch")


# ═══════════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════════

def run_training(train_loader, val_loader, class_names,
                 progress_bar, status_text, epoch_log,
                 class_weight_mode="auto"):
    num_classes = len(class_names)
    model       = YogaPoseModel(num_classes).to(DEVICE)

    labels = np.array(train_loader.dataset.labels)
    counts = np.bincount(labels, minlength=num_classes)
    imbalance_ratio = counts.max() / max(counts.min(), 1)

    class_weights = None
    enable_class_weights = (
        class_weight_mode == "force" or
        (class_weight_mode == "auto" and imbalance_ratio >= CLASS_WEIGHT_IMBALANCE_THRESHOLD)
    )
    if enable_class_weights:
        class_weights_np = counts.sum() / np.maximum(counts, 1)
        class_weights_np = class_weights_np / class_weights_np.mean()
        class_weights = torch.tensor(
            class_weights_np,
            dtype=torch.float32,
            device=DEVICE,
        )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=LABEL_SMOOTH,
    )

    optimizer = optim.AdamW(
        model.base_model.classifier.parameters(),
        lr=LR_HEAD,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=1,
        min_lr=1e-6,
    )

    best_val_loss     = float("inf")
    best_val_acc      = 0.0
    best_epoch        = 1
    patience_counter  = 0
    best_state        = copy.deepcopy(model.state_dict())   # FIX: deep copy
    stopped_early_at  = EPOCHS
    fine_tune_enabled = False

    history = {
        "loss": [],
        "accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "lr": [],
    }

    for epoch in range(EPOCHS):
        if (not fine_tune_enabled) and (epoch + 1 >= FINE_TUNE_AT_EPOCH):
            for p in model.base_model.parameters():
                p.requires_grad = False
            for idx, block in enumerate(model.base_model.features):
                if idx >= UNFREEZE_FROM_BLOCK:
                    for p in block.parameters():
                        p.requires_grad = True

            optimizer = optim.AdamW(
                (p for p in model.parameters() if p.requires_grad),
                lr=LR_FINE_TUNE,
                weight_decay=WEIGHT_DECAY,
            )
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=1,
                min_lr=1e-6,
            )
            fine_tune_enabled = True

        mixup_this_epoch = MIXUP_ALPHA > 0 and (epoch + 1) <= MIXUP_EPOCHS

        # ── Training phase ─────────────────────────────────────────────────
        model.train()
        t_loss, t_correct, t_total = 0.0, 0.0, 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)

            if mixup_this_epoch:
                lam = np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA)
                shuffle_idx = torch.randperm(images.size(0), device=images.device)
                mixed_images = lam * images + (1.0 - lam) * images[shuffle_idx]
                labels_a, labels_b = labels, labels[shuffle_idx]

                out = model(mixed_images)
                loss = lam * criterion(out, labels_a) + (1.0 - lam) * criterion(out, labels_b)
            else:
                out = model(images)
                loss = criterion(out, labels)

            loss.backward()

            if GRAD_CLIP > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    GRAD_CLIP,
                )

            optimizer.step()
            t_loss    += loss.item()
            _, pred    = torch.max(out, 1)
            t_total   += labels.size(0)
            if mixup_this_epoch:
                t_correct += lam * (pred == labels_a).sum().item()
                t_correct += (1.0 - lam) * (pred == labels_b).sum().item()
            else:
                t_correct += (pred == labels).sum().item()

        t_loss /= len(train_loader)
        t_acc   = t_correct / t_total

        # ── Validation phase ────────────────────────────────────────────────
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                out   = model(images)
                loss  = criterion(out, labels)
                v_loss    += loss.item()
                _, pred    = torch.max(out, 1)
                v_total   += labels.size(0)
                v_correct += (pred == labels).sum().item()

        v_loss /= len(val_loader)
        v_acc   = v_correct / v_total

        scheduler.step(v_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        history["loss"].append(round(t_loss, 5))
        history["accuracy"].append(round(t_acc, 5))
        history["val_loss"].append(round(v_loss, 5))
        history["val_accuracy"].append(round(v_acc, 5))
        history["lr"].append(float(current_lr))

        progress_bar.progress((epoch + 1) / EPOCHS)
        phase = "head" if not fine_tune_enabled else "fine-tune"
        aug_mode = "mixup" if mixup_this_epoch else "standard"
        status_text.markdown(
            f"**Epoch {epoch+1}/{EPOCHS}** — "
            f"Train Acc: `{t_acc:.4f}` | Val Acc: `{v_acc:.4f}` | "
            f"Val Loss: `{v_loss:.4f}` | LR: `{current_lr:.2e}` | "
            f"Phase: `{phase}` | Aug: `{aug_mode}`"
        )
        epoch_log.append(
            {"epoch": epoch + 1, "train_acc": round(t_acc, 4),
             "val_acc": round(v_acc, 4), "val_loss": round(v_loss, 4),
             "lr": float(current_lr), "phase": phase, "aug": aug_mode}
        )

        # ── Early stopping ──────────────────────────────────────────────────
        improved = (
            (v_acc > best_val_acc + 1e-6) or
            (abs(v_acc - best_val_acc) <= 1e-6 and v_loss < best_val_loss)
        )

        if improved:
            best_val_acc     = v_acc
            best_val_loss    = v_loss
            best_epoch       = epoch + 1
            patience_counter = 0
            best_state       = copy.deepcopy(model.state_dict())   # FIX: deep copy
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            stopped_early_at = epoch + 1
            break

    model.load_state_dict(best_state)
    model.eval()

    history["best_epoch"] = best_epoch
    history["best_val_accuracy"] = round(float(best_val_acc), 5)
    history["best_val_loss"] = round(float(best_val_loss), 5)
    history["class_weighting"] = (
        "enabled" if class_weights is not None else "disabled"
    )
    history["class_weight_mode"] = class_weight_mode
    history["imbalance_ratio"] = round(float(imbalance_ratio), 3)
    history["temperature"] = round(float(fit_temperature(model, val_loader)), 4)

    return model, history, stopped_early_at


def fit_temperature(model, val_loader):
    """Estimate a post-hoc temperature on validation logits for confidence calibration."""
    model.eval()
    logits_list, labels_list = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            out = model(images.to(DEVICE))
            logits_list.append(out.detach().cpu())
            labels_list.append(labels.cpu())

    if not logits_list:
        return 1.0

    logits = torch.cat(logits_list, dim=0)
    labels = torch.cat(labels_list, dim=0)

    # Lightweight grid search is stable on small datasets and avoids optimizer edge-cases.
    temperature_grid = np.linspace(0.5, 2.5, 41)
    best_temp = 1.0
    best_nll = float("inf")

    for temp in temperature_grid:
        nll = nn.functional.cross_entropy(logits / float(temp), labels).item()
        if nll < best_nll:
            best_nll = nll
            best_temp = float(temp)

    return best_temp


def get_temperature():
    """Return safe calibration temperature for probability scaling."""
    temp = float(st.session_state.get("temperature", 1.0))
    return max(0.5, min(2.5, temp))


def ensure_temperature_calibrated():
    """Calibrate confidence once when history lacks a temperature value."""
    if not st.session_state.trained:
        return
    if st.session_state.model is None or st.session_state.val_loader is None:
        return
    if st.session_state.history is None:
        return
    if "temperature" in st.session_state.history:
        return

    temp = fit_temperature(st.session_state.model, st.session_state.val_loader)
    st.session_state.temperature = float(temp)
    st.session_state.history["temperature"] = round(float(temp), 4)


# ═══════════════════════════════════════════════════════════════════════════
# EVALUATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def _get_predictions_cached(_model_id, _val_loader_id, _temperature):
    """
    Cache predictions so confusion matrix / report / ROC don't each
    run a full forward pass.  _model_id / _val_loader_id are just
    hashable keys (id of the objects); real objects are in session state.
    """
    model      = st.session_state.model
    val_loader = st.session_state.val_loader
    model.eval()
    preds, probs, labels = [], [], []
    with torch.no_grad():
        for imgs, lbls in val_loader:
            out  = model(imgs.to(DEVICE))
            calibrated_out = out / float(_temperature)
            p = torch.softmax(calibrated_out, 1)
            _, q = torch.max(calibrated_out, 1)
            preds.extend(q.cpu().numpy())
            probs.extend(p.cpu().numpy())
            labels.extend(lbls.numpy())
    return np.array(preds), np.array(probs), np.array(labels)


def plot_learning_curves(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for ax in (ax1, ax2):
        ax.set_facecolor("#111827"); fig.patch.set_facecolor("#0a0a0f")

    ep = range(1, len(history["accuracy"]) + 1)
    ax1.plot(ep, history["accuracy"],     color="#667eea", lw=2.5, label="Train")
    ax1.plot(ep, history["val_accuracy"], color="#06b6d4", lw=2.5, ls="--", label="Val")
    ax1.set_title("Accuracy", color="white", fontsize=12)
    ax1.set_xlabel("Epoch", color="#94a3b8"); ax1.set_ylabel("Accuracy", color="#94a3b8")
    ax1.legend(facecolor="#1f2937", labelcolor="white"); ax1.tick_params(colors="#94a3b8")
    ax1.grid(alpha=0.1); ax1.set_xticks(list(ep))

    ax2.plot(ep, history["loss"],     color="#667eea", lw=2.5, label="Train")
    ax2.plot(ep, history["val_loss"], color="#f59e0b", lw=2.5, ls="--", label="Val")
    ax2.set_title("Loss", color="white", fontsize=12)
    ax2.set_xlabel("Epoch", color="#94a3b8"); ax2.set_ylabel("Loss", color="#94a3b8")
    ax2.legend(facecolor="#1f2937", labelcolor="white"); ax2.tick_params(colors="#94a3b8")
    ax2.grid(alpha=0.1); ax2.set_xticks(list(ep))

    plt.tight_layout()
    st.pyplot(fig); plt.close()


def plot_confusion_matrix_fig(y_pred, y_true, class_names):
    cm     = confusion_matrix(y_true, y_pred)
    labels = [c.replace("_", " ").title() for c in class_names]

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor("#0a0a0f"); ax.set_facecolor("#111827")
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax,
                linewidths=0.4, linecolor="#1f2937")
    ax.set_title("Confusion Matrix — Validation Set", color="white", fontsize=13, pad=16)
    ax.set_xlabel("Predicted", color="#94a3b8", labelpad=8)
    ax.set_ylabel("True",      color="#94a3b8", labelpad=8)
    plt.xticks(rotation=45, ha="right", color="#94a3b8")
    plt.yticks(rotation=0,  color="#94a3b8")
    plt.tight_layout()
    return fig


def plot_per_class_accuracy(y_pred, y_true, class_names):
    labels = [c.replace("_", " ").title() for c in class_names]
    accs   = []
    for i in range(len(class_names)):
        mask = y_true == i
        acc  = (y_pred[mask] == i).sum() / mask.sum() if mask.sum() > 0 else 0
        accs.append(round(float(acc), 3))

    fig, ax = plt.subplots(figsize=(8, max(3, len(labels) * 0.55)))
    fig.patch.set_facecolor("#0a0a0f"); ax.set_facecolor("#111827")
    colors = ["#10b981" if a >= 0.8 else "#f59e0b" if a >= 0.6 else "#ef4444"
              for a in accs]
    y = range(len(labels))
    ax.barh(list(y), accs, color=colors)
    ax.set_yticks(list(y)); ax.set_yticklabels(labels)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Accuracy", color="#94a3b8")
    ax.set_title("Per-class accuracy (green ≥ 0.8, orange ≥ 0.6, red < 0.6)",
                 color="white", fontsize=10, pad=10)
    ax.tick_params(colors="#94a3b8"); ax.grid(axis="x", alpha=0.1)
    for i, a in enumerate(accs):
        ax.text(a + 0.01, i, f"{a:.0%}", va="center", color="#94a3b8", fontsize=9)
    plt.tight_layout()
    return fig


def show_classification_report(y_pred, y_true, class_names):
    labels  = [c.replace("_", " ").title() for c in class_names]
    report  = classification_report(y_true, y_pred,
                                    target_names=labels, output_dict=True)
    df = pd.DataFrame(report).T.drop("accuracy", errors="ignore").round(3)
    st.dataframe(
        df.style
          .background_gradient(cmap="Blues", subset=["precision", "recall", "f1-score"])
          .format({"precision": "{:.3f}", "recall": "{:.3f}",
                   "f1-score": "{:.3f}", "support": "{:.0f}"}),
        width="stretch",
    )


def plot_roc_curves(y_probs, y_true, class_names):
    n     = len(class_names)
    y_bin = label_binarize(y_true, classes=list(range(n)))

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor("#0a0a0f"); ax.set_facecolor("#111827")

    palette = ["#667eea", "#764ba2", "#06b6d4", "#10b981",
               "#f59e0b", "#ef4444", "#8b5cf6", "#ec4899"]

    for i, (name, color) in enumerate(zip(class_names, palette)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{name.replace('_',' ').title()} (AUC={roc_auc:.2f})")

    ax.plot([0,1],[0,1], "white", ls="--", alpha=0.25, lw=1)
    ax.set_xlabel("False Positive Rate", color="#94a3b8")
    ax.set_ylabel("True Positive Rate",  color="#94a3b8")
    ax.set_title("ROC Curves — One-vs-Rest", color="white", fontsize=13)
    ax.legend(loc="lower right", fontsize=8,
              facecolor="#1f2937", labelcolor="white", framealpha=0.8)
    ax.tick_params(colors="#94a3b8"); ax.grid(alpha=0.08)
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# GRAD-CAM
# ═══════════════════════════════════════════════════════════════════════════

def get_last_conv_layer(model):
    last_conv = None
    for module in model.base_model.features.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    if last_conv is None:
        raise ValueError("No Conv2d layer found in model backbone.")
    return last_conv


def generate_gradcam(model, img_tensor, class_idx):
    """Return (heatmap_rgb, overlay_rgb) as uint8 numpy arrays."""
    model.eval()
    activations, gradients = [], []

    target_layer = get_last_conv_layer(model)

    fwd = target_layer.register_forward_hook(
        lambda m, i, o: activations.append(o.detach()))
    bwd = target_layer.register_full_backward_hook(
        lambda m, gi, go: gradients.append(go[0].detach()))

    inp  = img_tensor.clone().requires_grad_(True)
    out  = model(inp)
    model.zero_grad()
    out[0, class_idx].backward()

    fwd.remove(); bwd.remove()

    act  = activations[0][0]          # (C, H, W)
    grad = gradients[0][0]            # (C, H, W)

    weights = torch.mean(grad, dim=(1, 2))  # (C,)
    cam     = torch.zeros(act.shape[1:])
    for i, w in enumerate(weights):
        cam += w * act[i]

    cam = torch.relu(cam)
    cam = cam / (cam.max() + 1e-8)    # FIX: +1e-8 guard
    cam = cam.cpu().numpy()

    cam_resized  = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
    heatmap_u8   = np.uint8(255 * cam_resized)
    heatmap_bgr  = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
    heatmap_rgb  = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    # Denormalize original for blending
    mean   = torch.tensor(IMAGENET_MEAN).view(3,1,1)
    std    = torch.tensor(IMAGENET_STD).view(3,1,1)
    orig   = torch.clamp(img_tensor[0].cpu() * std + mean, 0, 1)
    orig   = (orig.permute(1,2,0).numpy() * 255).astype(np.uint8)

    overlay = cv2.addWeighted(orig, 0.6, heatmap_rgb, 0.4, 0)
    return heatmap_rgb, overlay


# ═══════════════════════════════════════════════════════════════════════════
# PREDICTION HELPER
# ═══════════════════════════════════════════════════════════════════════════

def predict_image(uploaded_img):
    """
    Run full prediction pipeline on an uploaded image file.
    Returns dict with class, confidence, top3, heatmap, overlay, img_resized.
    """
    model       = st.session_state.model
    class_names = st.session_state.class_names

    img         = load_image_rgb(uploaded_img)
    img_resized = img.resize((IMG_SIZE, IMG_SIZE))
    img_tensor  = PRED_TRANSFORM(img).unsqueeze(0).to(DEVICE)

    model.eval()
    with torch.no_grad():
        out = model(img_tensor)
        calibrated_out = out / get_temperature()
        probs = torch.softmax(calibrated_out, 1)[0].cpu().numpy()

    top3_idx    = np.argsort(probs)[::-1][:3]
    class_idx   = int(top3_idx[0])
    confidence  = float(probs[class_idx])

    heatmap, overlay = generate_gradcam(model, img_tensor, class_idx)

    return {
        "class_key":    class_names[class_idx],
        "display_name": class_names[class_idx].replace("_", " ").title(),
        "sanskrit":     sanskrit_names.get(class_names[class_idx], ""),
        "confidence":   confidence,
        "top3":         [(class_names[i].replace("_"," ").title(),
                          float(probs[i])) for i in top3_idx],
        "all_probs":    probs,
        "heatmap":      heatmap,
        "overlay":      overlay,
        "img_resized":  img_resized,
        "class_names":  class_names,
    }


# ═══════════════════════════════════════════════════════════════════════════
# ── UI ──────────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

# ── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🧘 AsanaAI</h1>
  <p>Explainable Deep Learning · Transfer Learning · Grad-CAM Visualisation</p>
  <span class="badge">MobileNetV2</span>
  <span class="badge">PyTorch</span>
  <span class="badge">Grad-CAM</span>
  <span class="badge">CELBC608 — DSL Sem VI</span>
</div>
""", unsafe_allow_html=True)

if not GPU_AVAILABLE:
    st.warning("⚠️ GPU not detected — running on CPU. Training will be slower.")

# ── Status bar ───────────────────────────────────────────────────────────────
if st.session_state.trained:
    src = st.session_state.model_source
    src_txt = "loaded from saved checkpoint" if src == "file" else "trained this session"
    n_cls = len(st.session_state.class_names)
    st.markdown(f"""
    <div class="status-bar">
      <div class="status-dot"></div>
      <div class="status-text">
        ✅ Model ready — {n_cls} classes, {src_txt}.
        Scroll down to <strong>Predict a Pose</strong> or review evaluation metrics.
      </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="status-bar warn">
      <div class="status-dot warn"></div>
      <div class="status-text">
        ⚠️ No trained model found. Upload your dataset ZIP and click <strong>Begin Training</strong>.
      </div>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1 — PREDICT A POSE  (shown at top when model is ready)
# ═══════════════════════════════════════════════════════════════════════════

if st.session_state.trained:
    st.markdown('<div class="section-header">🔮 <span>Predict a Pose</span></div>',
                unsafe_allow_html=True)

    st.markdown(
        "Upload any yoga pose image — the model will classify it, show confidence, "
        "and highlight the body regions that drove the decision using **Grad-CAM**.",
        unsafe_allow_html=False,
    )

    pred_file = st.file_uploader(
        "Upload yoga pose image (JPG / PNG)",
        type=["jpg", "jpeg", "png"],
        key="pred_upload",
    )

    if pred_file is not None:
        with st.spinner("Classifying…"):
            result = predict_image(pred_file)

        cls_key   = result["class_key"]
        disp      = result["display_name"]
        sanskrit  = result["sanskrit"]
        conf      = result["confidence"]
        accent    = pose_info.get(cls_key, {}).get("color", "#667eea")

        # ── Prediction card ─────────────────────────────────────────────────
        conf_pct = round(conf * 100, 1)
        st.markdown(f"""
        <div class="pred-card">
          <p class="pred-title">{disp}</p>
          <p class="pred-sub">{sanskrit}</p>
          <div class="conf-bar-wrap">
            <div class="conf-bar" style="width:{conf_pct}%;
              background:linear-gradient(90deg,{accent},{accent}aa)"></div>
          </div>
          <p class="conf-label">Confidence: <strong style="color:{accent}">{conf_pct}%</strong></p>
        </div>
        """, unsafe_allow_html=True)

        # ── Images ──────────────────────────────────────────────────────────
        c1, c2, c3 = st.columns(3)
        c1.image(result["img_resized"], caption="📷 Original",         width="stretch")
        c2.image(result["heatmap"],     caption="🔥 Grad-CAM Heatmap",  width="stretch")
        c3.image(result["overlay"],     caption="🔍 Overlay",           width="stretch")

        st.caption(
            "Grad-CAM: red = strongest influence on prediction, blue = weakest. "
            "Good predictions highlight relevant body regions (limbs, torso alignment)."
        )

        # ── Top-3 + full probability bar chart ──────────────────────────────
        col_t3, col_bar = st.columns([1, 2])

        with col_t3:
            st.write("**Top-3 predictions**")
            for label, prob in result["top3"]:
                bar_w = round(prob * 100)
                color = accent if label == disp else "#667eea"
                st.markdown(f"""
                <div class="top3-row">
                  <span class="top3-label">{label}</span>
                  <div class="top3-bar-bg">
                    <div class="top3-bar" style="width:{bar_w}%;background:{color}"></div>
                  </div>
                  <span class="top3-pct">{bar_w}%</span>
                </div>
                """, unsafe_allow_html=True)

        with col_bar:
            st.write("**All class probabilities**")
            all_p   = result["all_probs"]
            cnames  = result["class_names"]
            prob_df = pd.DataFrame({
                "Pose":       [c.replace("_"," ").title() for c in cnames],
                "Confidence": all_p,
            }).sort_values("Confidence", ascending=True)

            fig, ax = plt.subplots(figsize=(6, max(3, len(cnames)*0.55)))
            fig.patch.set_facecolor("#0a0a0f"); ax.set_facecolor("#111827")
            bar_colors = [accent if p == disp else "#334155"
                          for p in prob_df["Pose"]]
            ax.barh(prob_df["Pose"], prob_df["Confidence"],
                    color=bar_colors, edgecolor="none")
            ax.set_xlim(0, 1)
            ax.set_xlabel("Confidence", color="#94a3b8")
            ax.tick_params(colors="#94a3b8"); ax.grid(axis="x", alpha=0.1)
            plt.tight_layout(); st.pyplot(fig); plt.close()

        # ── Pose info card ───────────────────────────────────────────────────
        info = pose_info.get(cls_key, {})
        if info:
            st.markdown(f"""
            <div class="info-card">
              <h4>📋 {disp} — {sanskrit}</h4>
              <p><strong>Benefits:</strong> {info['benefits']}</p>
              <p><strong>Form cues:</strong> {info['cues']}</p>
              <p><strong>Difficulty:</strong> {info['difficulty']}</p>
              <p class="disclaimer">
                ⚠️ AsanaAI is an educational tool for beginner practitioners.
                It does not replace certified yoga instruction.
              </p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2 — DATASET UPLOAD & EDA
# ═══════════════════════════════════════════════════════════════════════════

with st.expander(
    "📁  Dataset — Upload & Explore"
    + (" (dataset loaded)" if st.session_state.dataset_loaded else ""),
    expanded=not st.session_state.dataset_loaded,
):
    st.markdown(
        "Upload `yoga_dataset.zip` — must contain `train/` and `val/` subfolders, "
        "each with one folder per class. **Required only for training.**"
    )

    uploaded_zip = st.file_uploader(
        "yoga_dataset.zip",
        type=["zip"],
        key="dataset_zip",
        label_visibility="collapsed",
    )

    # FIX: only process on a fresh upload, not on every rerun
    if uploaded_zip is not None and not st.session_state.dataset_loaded:
        with st.spinner("Extracting ZIP…"):
            dp = process_uploaded_zip(uploaded_zip)
        if dp:
            st.session_state.dataset_path   = dp
            st.session_state.dataset_loaded = True
            st.success("✅ Dataset extracted successfully.")

    if st.session_state.dataset_loaded:
        st.markdown('<div class="section-header">📊 <span>EDA — Dataset Overview</span></div>',
                    unsafe_allow_html=True)
        show_dataset_summary(st.session_state.dataset_path)

        st.markdown('<div class="section-header">🎨 <span>Augmentation Preview</span></div>',
                    unsafe_allow_html=True)
        show_augmentation_preview(st.session_state.dataset_path)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3 — TRAINING
# ═══════════════════════════════════════════════════════════════════════════

train_header = (
    "🚀  Model Training"
    + (" ✅" if st.session_state.trained else " — required")
)

with st.expander(train_header, expanded=not st.session_state.trained):

    if st.session_state.trained:
        src = st.session_state.model_source
        if src == "file":
            st.info(
                "✅ Model loaded from saved checkpoint (`asanai_saved/`). "
                "Retrain below only if you have uploaded a new dataset."
            )
        else:
            st.success("✅ Model was trained this session.")

    if st.session_state.dataset_loaded:
        class_weight_choice = st.selectbox(
            "Class weighting mode",
            options=["Auto (recommended)", "Off", "Force on"],
            index=0,
            help=(
                "Auto enables class weights only when imbalance is high. "
                "Use Off for an ablation run."
            ),
        )
        use_weighted_sampler = st.checkbox(
            "Use weighted sampler (experimental)",
            value=False,
            help="Oversamples underrepresented classes in training batches.",
        )
        st.caption("Tip: run one experiment with class weighting Off to compare class precision/recall trade-offs.")

        class_weight_mode_map = {
            "Auto (recommended)": "auto",
            "Off": "off",
            "Force on": "force",
        }
        class_weight_mode = class_weight_mode_map[class_weight_choice]

        btn_label = "🔄 Retrain model" if st.session_state.trained else "🚀 Begin Training"
        if st.button(btn_label, type="primary"):
            prog     = st.progress(0)
            stat     = st.empty()
            log_data = []

            with st.spinner("Training… this may take a few minutes."):
                tl, vl, cnames = create_data_loaders(
                    st.session_state.dataset_path,
                    use_weighted_sampler=use_weighted_sampler,
                )
                model, history, stopped_at = run_training(
                    tl, vl, cnames, prog, stat, log_data,
                    class_weight_mode=class_weight_mode,
                )
                history["sampler"] = "weighted" if use_weighted_sampler else "standard"

            # Persist everything
            save_model(model, history, cnames)

            st.session_state.model        = model
            st.session_state.history      = history
            st.session_state.class_names  = cnames
            st.session_state.val_loader   = vl
            st.session_state.temperature  = float(history.get("temperature", 1.0))
            st.session_state.trained      = True
            st.session_state.model_source = "trained"

            if stopped_at < EPOCHS:
                st.success(
                    f"✅ Training complete — early stopping fired at epoch "
                    f"**{stopped_at}/{EPOCHS}** (patience={PATIENCE}). "
                    f"Best weights restored automatically."
                )
            else:
                st.success(f"✅ Training complete — ran all {EPOCHS} epochs.")

            # Epoch log table
            st.write("**Epoch log**")
            st.dataframe(pd.DataFrame(log_data).set_index("epoch"),
                         width="stretch")

    else:
        st.info("Upload the dataset ZIP above to enable training.")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4 — TRAINING RESULTS (learning curves)
# ═══════════════════════════════════════════════════════════════════════════

if st.session_state.trained and st.session_state.history:
    with st.expander("📈  Training Results — Learning Curves", expanded=True):
        h = st.session_state.history
        best_epoch = int(h.get("best_epoch", np.argmax(h["val_accuracy"]) + 1))
        best_idx = max(0, min(best_epoch - 1, len(h["val_accuracy"]) - 1))
        best_val_acc = h["val_accuracy"][best_idx]
        best_val_loss = h["val_loss"][best_idx]
        class_weighting = h.get("class_weighting", "not recorded")
        class_weight_mode = h.get("class_weight_mode", "not recorded")
        sampler_mode = h.get("sampler", "standard")
        calibration_temp = float(h.get("temperature", st.session_state.temperature))

        st.markdown(f"""
        <div class="metric-row">
          <div class="metric-box"><div class="mval">{h['accuracy'][-1]:.2%}</div><div class="mlbl">Final train acc</div></div>
          <div class="metric-box"><div class="mval">{best_val_acc:.2%}</div><div class="mlbl">Best val acc</div></div>
          <div class="metric-box"><div class="mval">{h['loss'][-1]:.4f}</div><div class="mlbl">Final train loss</div></div>
          <div class="metric-box"><div class="mval">{best_val_loss:.4f}</div><div class="mlbl">Val loss @ best acc</div></div>
          <div class="metric-box"><div class="mval">{len(h['accuracy'])}</div><div class="mlbl">Epochs run</div></div>
        </div>
        """, unsafe_allow_html=True)

        st.caption(
            f"Best validation epoch: {best_epoch}. "
            f"Class weighting: {class_weighting} ({class_weight_mode}). "
            f"Sampler: {sampler_mode}. "
            f"Confidence temperature: {calibration_temp:.2f}."
        )

        plot_learning_curves(h)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5 — EVALUATION METRICS
# ═══════════════════════════════════════════════════════════════════════════

if st.session_state.trained:

    # val_loader may not exist if model was loaded from file without uploading dataset
    if st.session_state.val_loader is None and st.session_state.dataset_loaded:
        _, vl, _ = create_data_loaders(st.session_state.dataset_path)
        st.session_state.val_loader = vl

    if st.session_state.val_loader is not None:
        if "temperature" not in (st.session_state.history or {}):
            with st.spinner("Calibrating confidence on validation set…"):
                ensure_temperature_calibrated()

        with st.expander("📊  Model Evaluation — Full Metrics", expanded=True):
            with st.spinner("Running inference on validation set…"):
                y_pred, y_probs, y_true = _get_predictions_cached(
                    id(st.session_state.model),
                    id(st.session_state.val_loader),
                    round(get_temperature(), 4),
                )

            class_names = st.session_state.class_names
            overall_acc = (y_pred == y_true).mean()
            st.markdown(f"""
            <div class="metric-row">
              <div class="metric-box"><div class="mval">{overall_acc:.2%}</div><div class="mlbl">Overall val accuracy</div></div>
              <div class="metric-box"><div class="mval">{len(y_true)}</div><div class="mlbl">Val images</div></div>
            </div>
            """, unsafe_allow_html=True)

            tab_cm, tab_cls, tab_pca, tab_roc = st.tabs([
                "Confusion Matrix",
                "Classification Report",
                "Per-class Accuracy",
                "ROC Curves",
            ])

            with tab_cm:
                fig = plot_confusion_matrix_fig(y_pred, y_true, class_names)
                st.pyplot(fig); plt.close()

            with tab_cls:
                show_classification_report(y_pred, y_true, class_names)

            with tab_pca:
                fig = plot_per_class_accuracy(y_pred, y_true, class_names)
                st.pyplot(fig); plt.close()
                st.caption("Green ≥ 80%, orange ≥ 60%, red < 60%")

            with tab_roc:
                fig = plot_roc_curves(y_probs, y_true, class_names)
                st.pyplot(fig); plt.close()

    else:
        st.info(
            "ℹ️ Evaluation metrics require the validation dataset. "
            "Upload the dataset ZIP in **Dataset** section above — "
            "the val loader will be created automatically."
        )


# ═══════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#334155;font-size:.8rem;padding:1rem 0 2rem">
  AsanaAI — CELBC608 Data Science Laboratory · Semester VI ·
  MobileNetV2 + Grad-CAM · PyTorch
</div>
""", unsafe_allow_html=True)
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
import time
import zipfile
import tempfile
import shutil
import hashlib
from io import BytesIO

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import mediapipe as mp

from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
from PIL import Image, UnidentifiedImageError

try:
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
    import av
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False

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
    initial_sidebar_state="expanded",
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
    "webcam_active": False,
    "webcam_result": None,
    "webcam_frame_count": 0,
    "webcam_smoothed_probs": None,
    "webcam_auto_refresh": False,
    "webcam_refresh_interval": 1.0,
    "webcam_last_frame_hash": None,
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


def make_fig(w=10, h=4, ncols=1, nrows=1):
    """Create a consistently styled figure matching the app theme."""
    fig, axes = plt.subplots(nrows, ncols, figsize=(w, h))
    fig.patch.set_facecolor("#0a0a0f")

    if isinstance(axes, np.ndarray):
        ax_list = axes.flatten().tolist()
    else:
        ax_list = [axes]

    for ax in ax_list:
        ax.set_facecolor("#111827")
        ax.tick_params(colors="#94a3b8")
        ax.grid(alpha=0.1)

    return fig, axes

# ═══════════════════════════════════════════════════════════════════════════
# REFERENCE DATA
# ═══════════════════════════════════════════════════════════════════════════

def normalize_class_key(name):
    """Normalize class labels so metadata matching is resilient to separators/case."""
    return str(name).strip().lower().replace("-", "_").replace(" ", "_")

sanskrit_names = {
    "camel_pose": "Ustrasana",
    "crane_pose": "Bakasana",
    "eight_angle_pose": "Astavakrasana",
    "full_boat_pose": "Paripurna Navasana",
    "king_dancer_pose": "Natarajasana",
    "plow_pose": "Halasana",
    "sitting_half_spinal_twist": "Ardha Matsyendrasana",
    "cow_pose": "Bitilasana",
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
    "camel_pose": {
        "benefits": "Opens chest and shoulders, strengthens back body, and improves spine mobility.",
        "cues": "Press shins down, lift sternum upward, and keep neck long while reaching for heels.",
        "difficulty": "Intermediate",
        "color": "#f97316",
    },
    "crane_pose": {
        "benefits": "Builds wrist, arm, and core strength while sharpening balance and focus.",
        "cues": "Stack knees high on upper arms, look forward, and shift weight gradually into hands.",
        "difficulty": "Advanced",
        "color": "#0ea5e9",
    },
    "eight_angle_pose": {
        "benefits": "Strengthens core and arms, improves hip mobility, and develops coordinated control.",
        "cues": "Hook one leg over shoulder, cross ankles, and engage lower belly to stay lifted.",
        "difficulty": "Advanced",
        "color": "#a855f7",
    },
    "full_boat_pose": {
        "benefits": "Strengthens abdominals and hip flexors while improving posture and stability.",
        "cues": "Lift chest, keep spine long, and balance on sit bones with steady breath.",
        "difficulty": "Beginner-Intermediate",
        "color": "#22c55e",
    },
    "king_dancer_pose": {
        "benefits": "Improves balance, opens chest and shoulders, and stretches quadriceps and hip flexors.",
        "cues": "Fix your gaze, press lifted foot into hand, and keep standing leg active and stable.",
        "difficulty": "Intermediate-Advanced",
        "color": "#14b8a6",
    },
    "plow_pose": {
        "benefits": "Lengthens the spine, stretches shoulders, and can calm the nervous system.",
        "cues": "Support the back, keep neck neutral, and avoid forcing feet to the floor.",
        "difficulty": "Intermediate",
        "color": "#e11d48",
    },
    "sitting_half_spinal_twist": {
        "benefits": "Improves spinal mobility, aids digestion, and releases lower-back tension.",
        "cues": "Lengthen through crown first, then rotate from ribs while grounding both sit bones.",
        "difficulty": "Beginner-Intermediate",
        "color": "#84cc16",
    },
    "cow_pose": {
        "benefits": "Mobilizes the spine, opens chest, and helps coordinate breath with movement.",
        "cues": "Inhale to lift tailbone and sternum while keeping shoulders relaxed away from ears.",
        "difficulty": "Beginner",
        "color": "#f43f5e",
    },
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
        "difficulty": "Beginner-Intermediate",
        "color":      "#f97316",
    },
}


def get_sanskrit_name(class_name):
    return sanskrit_names.get(normalize_class_key(class_name), "")


def get_pose_details(class_name):
    return pose_info.get(normalize_class_key(class_name), {})

# ═══════════════════════════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
  body, .main { background: #0a0a0f !important; color: #f1f5f9; }

    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes glowPulse {
        0%, 100% { box-shadow: 0 0 6px currentColor; }
        50% { box-shadow: 0 0 14px currentColor; }
    }

  .hero {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border: 1px solid rgba(102,126,234,0.3);
    padding: 2.5rem 2rem 2rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
        animation: fadeUp .6s ease-out both;
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
        animation: fadeUp .55s ease-out both;
  }
  .status-bar.warn {
    background: rgba(245,158,11,.06);
    border-color: rgba(245,158,11,.25);
  }
  .status-dot {
    width: 0.65rem; height: 0.65rem; border-radius: 50%;
    background: #10b981; flex-shrink: 0;
    box-shadow: 0 0 6px #10b981;
        animation: glowPulse 2.2s ease-in-out infinite;
  }
  .status-dot.warn { background: #f59e0b; box-shadow: 0 0 6px #f59e0b; }
  .status-text { color: #e2e8f0; font-size: .9rem; }

  .section-header {
    font-size: 1.3rem; font-weight: 600; color: #f1f5f9;
    margin: 1.8rem 0 1rem;
    padding-bottom: .45rem;
    border-bottom: 1px solid rgba(255,255,255,.07);
    letter-spacing: .01em;
        animation: fadeUp .45s ease-out both;
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
        animation: fadeUp .4s ease-out both;
  }
  .metric-box .mval { font-size: 1.6rem; font-weight: 700; color: #a5b4fc; }
  .metric-box .mlbl { font-size: .78rem; color: #64748b; margin-top: .2rem; }

  /* Prediction result */
  .pred-card {
    background: linear-gradient(135deg, rgba(102,126,234,.12), rgba(118,75,162,.08));
    border: 1px solid rgba(102,126,234,.3);
    border-radius: 14px; padding: 1.5rem 2rem;
    margin: 1rem 0;
        animation: fadeUp .42s ease-out both;
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

    .confidence-band-wrap {
        display: flex;
        align-items: center;
        gap: .85rem;
        margin: .4rem 0 .95rem;
        animation: fadeUp .35s ease-out both;
    }
    .confidence-ring {
        width: 3.75rem;
        height: 3.75rem;
        border-radius: 50%;
        display: grid;
        place-items: center;
        position: relative;
        background: conic-gradient(var(--ring-color) calc(var(--ring-value) * 1%), rgba(148,163,184,.2) 0);
    }
    .confidence-ring::before {
        content: "";
        position: absolute;
        inset: 6px;
        border-radius: 50%;
        background: #0f172a;
        border: 1px solid rgba(255,255,255,.09);
    }
    .confidence-ring span {
        position: relative;
        font-size: .72rem;
        font-weight: 700;
        color: #e2e8f0;
    }
    .confidence-band-label {
        margin: 0;
        font-size: .87rem;
        font-weight: 700;
    }
    .confidence-band-sub {
        margin: .1rem 0 0;
        color: #64748b;
        font-size: .74rem;
    }

  .top3-row {
    display: flex; align-items: center; gap: .7rem;
    padding: .4rem 0;
    border-bottom: 1px solid rgba(255,255,255,.04);
  }
  .top3-label { flex: 1; font-size: .85rem; color: #cbd5e1; }
  .top3-pct { font-size: .85rem; font-weight: 600; color: #a5b4fc;
    min-width: 2.6rem; text-align: right; }
  .top3-bar-bg { flex: 0 0 30%; background: rgba(255,255,255,.06);
    border-radius: 4px; height: .32rem; }
  .top3-bar { height: .32rem; border-radius: 4px; background: #667eea; }

  .info-card {
    background: rgba(16,185,129,.04);
    border-left: 3px solid #10b981;
    border-radius: 0 10px 10px 0;
    padding: 1.25rem 1.5rem; margin-top: 1rem;
    line-height: 1.8;
        animation: fadeUp .45s ease-out both;
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

  /* ── Viewport-fit: scale everything with zoom ──────────────────────── */
  .main .block-container {
    max-width: 100% !important;
    padding-top: 1rem !important;
    padding-bottom: 1rem !important;
  }

  /* All matplotlib / pyplot charts scale to container */
  .stPlotlyChart, .stPyplot {
    width: 100% !important;
  }
  .stPyplot img, .stPyplot > div > img {
    width: 100% !important;
    height: auto !important;
    max-height: 70vh !important;
    object-fit: contain !important;
  }

  /* Dataframes fill their container */
  .stDataFrame { width: 100% !important; }

  /* WebRTC video element – scale to fit viewport */
  iframe[title="streamlit_webrtc.component"],
  video {
    max-height: 60vh !important;
    width: 100% !important;
    object-fit: contain !important;
  }

  /* Streamlit camera input – same constraint */
  .stCameraInput video,
  .stCameraInput img {
    max-height: 55vh !important;
    object-fit: contain !important;
  }

  /* Streamlit image elements scale to container */
  .stImage img {
    max-width: 100% !important;
    height: auto !important;
    max-height: 55vh !important;
    object-fit: contain !important;
  }

  /* Metric boxes scale */
  [data-testid="stMetric"] {
    width: 100% !important;
  }

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
        self.skipped_images = []
        self.class_names = sorted(
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        )
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}

        for cls in self.class_names:
            cls_dir = os.path.join(root_dir, cls)
            for fname in sorted(os.listdir(cls_dir)):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    image_path = os.path.join(cls_dir, fname)
                    try:
                        load_image_rgb(image_path)
                    except (OSError, UnidentifiedImageError, ValueError) as exc:
                        self.skipped_images.append({"path": image_path, "error": str(exc)})
                        continue
                    self.images.append(image_path)
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

TTA_TRANSFORMS = [
    PRED_TRANSFORM,
    transforms.Compose([
        transforms.Resize(int(IMG_SIZE * 1.15)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]),
    transforms.Compose([
        transforms.Resize(int(IMG_SIZE * 1.25)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]),
    transforms.Compose([
        transforms.Resize(int(IMG_SIZE * 1.15)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]),
]

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

    skipped_summary = {
        "train": train_ds.skipped_images,
        "val": val_ds.skipped_images,
    }

    return train_loader, val_loader, train_ds.class_names, skipped_summary


def show_skipped_images_warning(skipped_summary):
    train_skipped = skipped_summary.get("train", [])
    val_skipped = skipped_summary.get("val", [])
    total_skipped = len(train_skipped) + len(val_skipped)
    if total_skipped == 0:
        return

    preview_paths = [
        os.path.basename(item["path"])
        for item in (train_skipped + val_skipped)[:5]
    ]
    preview = ", ".join(preview_paths)
    st.warning(
        "Skipped unreadable images while building the dataset "
        f"(train: {len(train_skipped)}, val: {len(val_skipped)}). "
        f"Examples: {preview}"
    )


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
    total_images = total_train + total_val

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Classes", len(train_counts))
    m2.metric("Train Images", total_train)
    m3.metric("Val Images", total_val)
    train_ratio = round(total_train / total_images * 100) if total_images > 0 else 0
    m4.metric("Train Ratio", f"{train_ratio}%")

    st.caption(f"Total images: {total_images}")

    col_chart, col_samples = st.columns([1, 1])

    with col_chart:
        st.write("**Class distribution**")
        classes     = list(train_counts.keys())
        train_vals  = [train_counts[c] for c in classes]
        val_vals    = [val_counts.get(c, 0) for c in classes]
        labels      = [c.replace("_", " ").title() for c in classes]

        fig, ax = make_fig(w=5, h=max(3, len(classes) * 0.55))
        y = np.arange(len(labels))
        w = 0.38
        ax.barh(y - w/2, train_vals, w, color="#667eea", label="Train")
        ax.barh(y + w/2, val_vals,   w, color="#06b6d4", label="Val")
        ax.set_yticks(y); ax.set_yticklabels(labels)
        ax.set_xlabel("Images", color="#94a3b8")
        ax.set_title("Images per class", color="white", fontsize=11, pad=10)
        ax.legend(facecolor="#1f2937", labelcolor="white", fontsize=9)
        ax.tick_params(colors="#94a3b8"); ax.grid(axis="x", alpha=0.1)
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    with col_samples:
        st.write("**Sample images (one per class)**")
        train_path = os.path.join(dataset_path, "train")
        num_cols   = 4
        cols       = st.columns(num_cols)
        for i, cls in enumerate(classes):
            cls_dir = os.path.join(train_path, cls)
            imgs    = sorted(f for f in os.listdir(cls_dir)
                             if f.lower().endswith((".jpg", ".jpeg", ".png")))
            if imgs:
                img = load_image_rgb(os.path.join(cls_dir, imgs[0]))
                cols[i % num_cols].image(
                    img,
                    caption=cls.replace("_", " ").title(),
                    use_container_width=True,
                )


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
        col.image(disp, use_container_width=True)


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
    """Calibrate confidence when stored temperature is missing or default."""
    if not st.session_state.trained:
        return
    if st.session_state.model is None or st.session_state.val_loader is None:
        return
    if st.session_state.history is None:
        return

    temp_from_history = st.session_state.history.get("temperature")
    try:
        needs_calibration = (not temp_from_history) or float(temp_from_history) == 1.0
    except (TypeError, ValueError):
        needs_calibration = True

    if not needs_calibration:
        st.session_state.temperature = float(temp_from_history)
        return

    temp = fit_temperature(st.session_state.model, st.session_state.val_loader)
    st.session_state.temperature = float(temp)
    st.session_state.history["temperature"] = round(float(temp), 4)


CONFIDENCE_COLORS = {
    "high": ("#10b981", "High confidence ✅"),
    "medium": ("#f59e0b", "Medium confidence ⚠️"),
    "low": ("#ef4444", "Low confidence ❌ - try a clearer image"),
}


def confidence_band(conf):
    if conf >= 0.75:
        return CONFIDENCE_COLORS["high"]
    if conf >= 0.45:
        return CONFIDENCE_COLORS["medium"]
    return CONFIDENCE_COLORS["low"]


def render_confidence_ring(conf, band_color, band_label):
        conf_pct = int(round(float(conf) * 100))
        conf_pct = max(0, min(100, conf_pct))
        return f"""
        <div class="confidence-band-wrap">
            <div class="confidence-ring" style="--ring-color:{band_color}; --ring-value:{conf_pct};">
                <span>{conf_pct}%</span>
            </div>
            <div>
                <p class="confidence-band-label" style="color:{band_color}">{band_label}</p>
                <p class="confidence-band-sub">Calibrated confidence band</p>
            </div>
        </div>
        """


def prediction_entropy(probs):
    """Shannon entropy normalized to [0, 1], where higher means less certain."""
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-9, 1.0)
    entropy = -np.sum(p * np.log(p))
    entropy_max = np.log(len(p))
    if entropy_max <= 0:
        return 0.0
    return float(entropy / entropy_max)


def five_crop_ensemble(img, model, temperature):
    """Average predictions from center+corner crops for borderline cases."""
    crop_norm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    five_crop_tfm = transforms.Compose([
        transforms.Resize(int(IMG_SIZE * 1.2)),
        transforms.FiveCrop(IMG_SIZE),
        transforms.Lambda(lambda crops: torch.stack([crop_norm(c) for c in crops])),
    ])

    crops_tensor = five_crop_tfm(img).to(DEVICE)
    with torch.no_grad():
        out = model(crops_tensor)
        probs = torch.softmax(out / float(temperature), 1).cpu().numpy()
    return probs.mean(axis=0)


SIMILAR_POSES = {
    frozenset(["triangle", "warrior"]): "Both involve wide-leg standing alignment patterns.",
    frozenset(["cobra", "plank"]): "Both are floor-based prone poses with strong shoulder engagement.",
    frozenset(["tree", "king_dancer_pose"]): "Both are single-leg balance poses requiring hip stability.",
    frozenset(["child_pose", "seated_forward_bend"]): "Both involve forward folding with spine length awareness.",
}

def get_similarity_hint(cls_a, cls_b):
    key = frozenset([normalize_class_key(cls_a), normalize_class_key(cls_b)])
    return SIMILAR_POSES.get(key, "")


# ── MediaPipe Pose setup (Tasks API) ─────────────────────────────────────────
from mediapipe.tasks.python import BaseOptions as _MPBaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker as _PoseLandmarker,
    PoseLandmarkerOptions as _PoseLandmarkerOptions,
    PoseLandmarksConnections as _PoseConns,
    RunningMode as _RunningMode,
)

_POSE_MODEL_PATH = os.path.join(SAVE_DIR, "pose_landmarker_lite.task")
_POSE_CONNECTIONS = _PoseConns.POSE_LANDMARKS


def _ensure_pose_model():
    """Download the Pose Landmarker model if not present."""
    if os.path.exists(_POSE_MODEL_PATH):
        return
    import urllib.request
    url = (
        "https://storage.googleapis.com/mediapipe-models/"
        "pose_landmarker/pose_landmarker_lite/float16/latest/"
        "pose_landmarker_lite.task"
    )
    urllib.request.urlretrieve(url, _POSE_MODEL_PATH)


def draw_pose_landmarks(bgr_frame, min_detection_confidence=0.5):
    """Run MediaPipe PoseLandmarker on a BGR frame and draw skeleton.
    Returns the annotated frame and the detection result."""
    _ensure_pose_model()

    options = _PoseLandmarkerOptions(
        base_options=_MPBaseOptions(model_asset_path=_POSE_MODEL_PATH),
        running_mode=_RunningMode.IMAGE,
        min_pose_detection_confidence=min_detection_confidence,
        min_tracking_confidence=0.5,
    )

    with _PoseLandmarker.create_from_options(options) as landmarker:
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)

        if result.pose_landmarks:
            h, w = bgr_frame.shape[:2]
            for pose_lms in result.pose_landmarks:
                # Draw connections
                for conn in _POSE_CONNECTIONS:
                    lm_a = pose_lms[conn.start]
                    lm_b = pose_lms[conn.end]
                    ax, ay = int(lm_a.x * w), int(lm_a.y * h)
                    bx, by = int(lm_b.x * w), int(lm_b.y * h)
                    min_vis = min(lm_a.visibility, lm_b.visibility)
                    if min_vis > 0.3:
                        color = (180, 230, 180) if min_vis > 0.6 else (100, 180, 180)
                        cv2.line(bgr_frame, (ax, ay), (bx, by), color, 2, cv2.LINE_AA)

                # Draw landmarks with color by visibility
                for lm in pose_lms:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    vis = lm.visibility
                    if vis > 0.7:
                        color, r = (0, 230, 118), 5
                    elif vis > 0.4:
                        color, r = (0, 220, 255), 4
                    else:
                        color, r = (80, 80, 255), 3
                    cv2.circle(bgr_frame, (cx, cy), r, color, -1)
                    cv2.circle(bgr_frame, (cx, cy), r + 1, (255, 255, 255), 1)

    return bgr_frame, result


if WEBRTC_AVAILABLE:
    class YogaVideoProcessor(VideoProcessorBase):
        def __init__(self, model=None, class_names=None, temperature=1.0):
            self.model = model
            self.class_names = class_names or []
            self.temperature = max(0.5, min(2.5, float(temperature)))
            if self.model is not None:
                self.model.eval()

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            if self.model is None or not self.class_names:
                return av.VideoFrame.from_ndarray(img, format="bgr24")

            # Draw pose skeleton
            img, _ = draw_pose_landmarks(img, min_detection_confidence=0.5)

            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            tensor = PRED_TRANSFORM(pil_img).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                out = self.model(tensor)
                probs = torch.softmax(out / self.temperature, 1)[0].cpu().numpy()

            cls_idx = int(np.argmax(probs))
            cls_key = self.class_names[cls_idx]
            cls_name = cls_key.replace("_", " ").title()
            conf = float(probs[cls_idx])
            label = f"{cls_name} {conf:.0%}"

            # Pose label background
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            cv2.rectangle(img, (5, 8), (15 + tw, 44 + th), (0, 0, 0), -1)
            cv2.putText(img, label, (10, 36), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 255, 120), 2, cv2.LINE_AA)

            # Show form cue from pose_info
            info = pose_info.get(normalize_class_key(cls_key), {})
            cue = info.get("cues", "")
            if cue and conf > 0.45:
                h_img = img.shape[0]
                # Wrap cue text to fit
                cue_short = cue[:80] + ("..." if len(cue) > 80 else "")
                (cw, ch), _ = cv2.getTextSize(cue_short, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(img, (5, h_img - 35 - ch), (15 + cw, h_img - 5), (0, 0, 0), -1)
                cv2.putText(img, cue_short, (10, h_img - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 230, 255), 1, cv2.LINE_AA)

            return av.VideoFrame.from_ndarray(img, format="bgr24")


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
    fig, (ax1, ax2) = make_fig(w=11, h=4, ncols=2)

    ep = list(range(1, len(history["accuracy"]) + 1))
    best_ep = int(history.get("best_epoch", 1))
    tick_step = max(1, len(ep) // 6) if ep else 1
    if ep:
        best_ep = max(ep[0], min(best_ep, ep[-1]))

    ax1.plot(ep, history["accuracy"],     color="#667eea", lw=2.5, label="Train")
    ax1.plot(ep, history["val_accuracy"], color="#06b6d4", lw=2.5, ls="--", label="Val")
    ax1.axvline(x=best_ep, color="#10b981", ls=":", lw=1.5, label=f"Best (ep {best_ep})")
    ax1.set_title("Accuracy", color="white", fontsize=12)
    ax1.set_xlabel("Epoch", color="#94a3b8"); ax1.set_ylabel("Accuracy", color="#94a3b8")
    ax1.legend(facecolor="#1f2937", labelcolor="white")
    ax1.set_xticks(ep[::tick_step])

    ax2.plot(ep, history["loss"],     color="#667eea", lw=2.5, label="Train")
    ax2.plot(ep, history["val_loss"], color="#f59e0b", lw=2.5, ls="--", label="Val")
    ax2.axvline(x=best_ep, color="#10b981", ls=":", lw=1.5)
    ax2.set_title("Loss", color="white", fontsize=12)
    ax2.set_xlabel("Epoch", color="#94a3b8"); ax2.set_ylabel("Loss", color="#94a3b8")
    ax2.legend(facecolor="#1f2937", labelcolor="white")
    ax2.set_xticks(ep[::tick_step])

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close()


def plot_confusion_matrix_fig(y_pred, y_true, class_names):
    label_ids = list(range(len(class_names)))
    cm     = confusion_matrix(y_true, y_pred, labels=label_ids)
    labels = [c.replace("_", " ").title() for c in class_names]
    n = len(class_names)
    fig_size = max(7, n * 0.55)
    tick_size = max(6, 9 - n // 4)

    fig, ax = make_fig(w=fig_size, h=fig_size * 0.8)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax,
                linewidths=0.4, linecolor="#1f2937")
    ax.set_title("Confusion Matrix — Validation Set", color="white", fontsize=13, pad=16)
    ax.set_xlabel("Predicted", color="#94a3b8", labelpad=8)
    ax.set_ylabel("True",      color="#94a3b8", labelpad=8)
    plt.xticks(rotation=45, ha="right", fontsize=tick_size, color="#94a3b8")
    plt.yticks(rotation=0, fontsize=tick_size, color="#94a3b8")
    plt.tight_layout()
    return fig


def plot_per_class_accuracy(y_pred, y_true, class_names):
    labels = [c.replace("_", " ").title() for c in class_names]
    accs   = []
    for i in range(len(class_names)):
        mask = y_true == i
        acc  = (y_pred[mask] == i).sum() / mask.sum() if mask.sum() > 0 else 0
        accs.append(round(float(acc), 3))

    fig, ax = make_fig(w=8, h=max(3, len(labels) * 0.55))
    colors = ["#10b981" if a >= 0.8 else "#f59e0b" if a >= 0.6 else "#ef4444"
              for a in accs]
    y = range(len(labels))
    ax.barh(list(y), accs, color=colors)
    ax.set_yticks(list(y)); ax.set_yticklabels(labels)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Accuracy", color="#94a3b8")
    ax.set_title("Per-class accuracy (green ≥ 0.8, orange ≥ 0.6, red < 0.6)",
                 color="white", fontsize=10, pad=10)
    for i, a in enumerate(accs):
        ax.text(a + 0.01, i, f"{a:.0%}", va="center", color="#94a3b8", fontsize=9)
    plt.tight_layout()
    return fig


def show_classification_report(y_pred, y_true, class_names):
    display_names = [c.replace("_", " ").title() for c in class_names]
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=display_names,
        output_dict=True,
        zero_division=0,
    )
    df = pd.DataFrame(report).T.drop("accuracy", errors="ignore").round(3)
    st.dataframe(
        df.style
          .background_gradient(cmap="Blues", subset=["precision", "recall", "f1-score"])
          .format({"precision": "{:.3f}", "recall": "{:.3f}",
                   "f1-score": "{:.3f}", "support": "{:.0f}"}),
        use_container_width=True,
    )


def plot_roc_curves(y_probs, y_true, class_names):
    n     = len(class_names)
    y_bin = label_binarize(y_true, classes=list(range(n)))

    fig, ax = make_fig(w=9, h=7)
    colors = [
        "#667eea", "#764ba2", "#06b6d4", "#10b981",
        "#f59e0b", "#ef4444", "#8b5cf6", "#ec4899",
        "#22c55e", "#14b8a6", "#f97316", "#e11d48",
        "#0ea5e9", "#84cc16", "#f43f5e", "#a855f7",
    ]

    plotted = 0
    skipped = []
    for i, name in enumerate(class_names):
        col = y_bin[:, i]
        # roc_curve needs both positive and negative samples for a class.
        if np.unique(col).size < 2:
            skipped.append(name.replace("_", " ").title())
            continue

        fpr, tpr, _ = roc_curve(col, y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        color = colors[i % len(colors)]
        ax.plot(
            fpr,
            tpr,
            color=color,
            lw=2,
            label=f"{name.replace('_', ' ').title()} (AUC={roc_auc:.2f})",
        )
        plotted += 1

    ax.plot([0,1],[0,1], "white", ls="--", alpha=0.25, lw=1)
    ax.set_xlabel("False Positive Rate", color="#94a3b8")
    ax.set_ylabel("True Positive Rate",  color="#94a3b8")
    ax.set_title("ROC Curves — One-vs-Rest", color="white", fontsize=13)
    if plotted > 0:
        ax.legend(
            loc="lower right",
            fontsize=7,
            facecolor="#1f2937",
            labelcolor="white",
            framealpha=0.8,
            ncol=2,
        )
    else:
        ax.text(
            0.5,
            0.5,
            "ROC unavailable for current labels",
            color="#94a3b8",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    if skipped:
        skipped_txt = ", ".join(skipped[:4])
        if len(skipped) > 4:
            skipped_txt += ", ..."
        ax.text(
            0.99,
            0.01,
            f"Skipped classes lacking positives/negatives: {skipped_txt}",
            color="#64748b",
            fontsize=7,
            ha="right",
            va="bottom",
            transform=ax.transAxes,
        )

    ax.grid(alpha=0.08)
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

def predict_image(uploaded_img, use_tta=True, show_gradcam_spinner=False):
    """
    Run full prediction pipeline on an uploaded image file.
    Returns dict with class, confidence, top3, heatmap, overlay, img_resized.
    """
    model       = st.session_state.model
    class_names = st.session_state.class_names

    img = load_image_rgb(uploaded_img)
    img_resized = img.resize((IMG_SIZE, IMG_SIZE))
    temperature = get_temperature()

    model.eval()
    if use_tta:
        all_probs = []
        for tfm in TTA_TRANSFORMS:
            t = tfm(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                out = model(t)
                p = torch.softmax(out / temperature, 1)[0].cpu().numpy()
            all_probs.append(p)

        probs = np.mean(all_probs, axis=0)
        if float(np.max(probs)) < 0.6:
            probs = (probs + five_crop_ensemble(img, model, temperature)) / 2.0

        img_tensor = PRED_TRANSFORM(img).unsqueeze(0).to(DEVICE)
    else:
        img_tensor = PRED_TRANSFORM(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = model(img_tensor)
            probs = torch.softmax(out / temperature, 1)[0].cpu().numpy()

    top3_idx    = np.argsort(probs)[::-1][:3]
    class_idx   = int(top3_idx[0])
    confidence  = float(probs[class_idx])

    if show_gradcam_spinner:
        with st.spinner("Generating Grad-CAM..."):
            heatmap, overlay = generate_gradcam(model, img_tensor, class_idx)
    else:
        heatmap, overlay = generate_gradcam(model, img_tensor, class_idx)

    return {
        "class_key":    class_names[class_idx],
        "display_name": class_names[class_idx].replace("_", " ").title(),
        "sanskrit":     get_sanskrit_name(class_names[class_idx]),
        "confidence":   confidence,
        "top3":         [(class_names[i].replace("_"," ").title(),
                          float(probs[i])) for i in top3_idx],
        "top3_keys":    [class_names[i] for i in top3_idx],
        "all_probs":    probs,
        "heatmap":      heatmap,
        "overlay":      overlay,
        "img_resized":  img_resized,
        "class_names":  class_names,
    }


# ═══════════════════════════════════════════════════════════════════════════
# ── UI ──────────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🧘 AsanaAI")
    st.markdown("---")
    nav = st.radio(
        "Navigate",
        [
            "🔮 Predict (Image)",
            "📷 Live Webcam",
            "📁 Dataset & Train",
            "📈 Results",
            "📊 Evaluation",
        ],
        index=0,
    )
    st.markdown("---")

    if st.session_state.trained:
        h = st.session_state.history or {}
        st.metric("Best Val Acc", f"{h.get('best_val_accuracy', 0):.1%}")
        st.metric("Classes", len(st.session_state.class_names))
        st.metric("Epochs Run", len(h.get("accuracy", [])))
    else:
        st.caption("No trained model loaded yet.")

    with st.expander("Model classes"):
        if st.session_state.class_names:
            for i, c in enumerate(st.session_state.class_names, start=1):
                st.write(f"{i}. {c.replace('_', ' ').title()}")
        else:
            st.caption("Classes will appear after loading/training a model.")

    st.markdown("---")
    st.caption("CELBC608 - DSL Sem VI\nMobileNetV2 + Grad-CAM")

show_predict = nav == "🔮 Predict (Image)"
show_webcam = nav == "📷 Live Webcam"
show_dataset_train = nav == "📁 Dataset & Train"
show_results = nav == "📈 Results"
show_evaluation = nav == "📊 Evaluation"

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
                Use the sidebar to open prediction, webcam, results, or evaluation views.
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

if show_predict:
    if st.session_state.trained:
        st.markdown('<div class="section-header">🔮 <span>Predict a Pose</span></div>',
                    unsafe_allow_html=True)

        st.markdown(
            "Upload any yoga pose image - the model will classify it, show confidence, "
            "and highlight the body regions that drove the decision using **Grad-CAM**.",
            unsafe_allow_html=False,
        )

        use_tta = st.checkbox(
            "Use Test-Time Augmentation (TTA) - slightly slower, more stable",
            value=True,
            key="predict_use_tta",
        )

        pred_file = st.file_uploader(
            "Upload yoga pose image (JPG / PNG)",
            type=["jpg", "jpeg", "png"],
            key="pred_upload",
        )

        if pred_file is not None:
            with st.spinner("Classifying..."):
                result = predict_image(
                    pred_file,
                    use_tta=use_tta,
                    show_gradcam_spinner=True,
                )

            cls_key = result["class_key"]
            disp = result["display_name"]
            sanskrit = result["sanskrit"]
            conf = result["confidence"]
            accent = get_pose_details(cls_key).get("color", "#667eea")

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

            band_color, band_label = confidence_band(conf)
            st.markdown(
                render_confidence_ring(conf, band_color, band_label),
                unsafe_allow_html=True,
            )

            entropy = prediction_entropy(result["all_probs"])
            entropy_label = "Low" if entropy < 0.3 else "Medium" if entropy < 0.6 else "High"
            st.caption(
                f"Prediction uncertainty: **{entropy_label}** ({entropy:.2f}) - lower is better"
            )

            if conf < 0.6 and len(result.get("top3_keys", [])) >= 2:
                top2_keys = result["top3_keys"][:2]
                hint = get_similarity_hint(top2_keys[0], top2_keys[1])
                if hint:
                    first = top2_keys[0].replace("_", " ").title()
                    second = top2_keys[1].replace("_", " ").title()
                    st.info(f"Possible overlap: **{first}** and **{second}**. {hint}")

            c1, c2, c3 = st.columns(3)
            c1.image(result["img_resized"], caption="📷 Original", use_container_width=True)
            c2.image(result["heatmap"], caption="🔥 Grad-CAM Heatmap", use_container_width=True)
            c3.image(result["overlay"], caption="🔍 Overlay", use_container_width=True)

            buf = BytesIO()
            Image.fromarray(result["overlay"]).save(buf, format="PNG")
            st.download_button(
                "⬇ Download Grad-CAM overlay",
                data=buf.getvalue(),
                file_name=f"gradcam_{result['display_name'].replace(' ', '_').lower()}.png",
                mime="image/png",
                key="download_gradcam_predict",
            )

            st.caption(
                "Grad-CAM: red = strongest influence on prediction, blue = weakest. "
                "Good predictions highlight relevant body regions (limbs, torso alignment)."
            )

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
                all_p = result["all_probs"]
                cnames = result["class_names"]
                prob_df = pd.DataFrame({
                    "PoseFull": [c.replace("_", " ").title() for c in cnames],
                    "Confidence": all_p,
                })
                prob_df["IsTop"] = prob_df["PoseFull"] == disp
                prob_df["Pose"] = prob_df["PoseFull"].apply(
                    lambda x: x[:22] + "..." if len(x) > 22 else x
                )
                prob_df = prob_df.sort_values("Confidence", ascending=True)

                fig, ax = make_fig(w=6, h=max(3, len(cnames) * 0.55))
                bar_colors = [accent if is_top else "#334155" for is_top in prob_df["IsTop"]]
                ax.barh(prob_df["Pose"], prob_df["Confidence"], color=bar_colors, edgecolor="none")
                ax.set_xlim(0, 1)
                ax.set_xlabel("Confidence", color="#94a3b8")
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()

            info = get_pose_details(cls_key)
            if info:
                st.markdown(f"""
                <div class="info-card">
                  <h4>📋 {disp} - {sanskrit}</h4>
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
    else:
        st.info("Train or load a model first from the Dataset & Train section.")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1B — LIVE WEBCAM MODE
# ═══════════════════════════════════════════════════════════════════════════

if show_webcam:
    st.markdown('<div class="section-header">📷 <span>Live Webcam Mode</span></div>',
                unsafe_allow_html=True)

    if not st.session_state.trained:
        st.info("Train or load a model first to use webcam assistance.")
    else:
        webcam_mode = st.radio(
            "Webcam mode",
            ["Snapshot mode (built-in)", "Real-time stream (WebRTC)"],
            horizontal=True,
            key="webcam_mode",
        )

        if webcam_mode == "Real-time stream (WebRTC)":
            # Keep snapshot camera controls independent from WebRTC streaming.
            st.session_state.webcam_active = False
            if WEBRTC_AVAILABLE:
                st.caption("Real-time streaming is active. Allow browser camera access when prompted.")
                # Capture into local vars so the lambda (which runs in a
                # background thread without ScriptRunContext) doesn't need
                # to access st.session_state.
                _model = st.session_state.model
                _class_names = st.session_state.class_names
                _temperature = get_temperature()
                webrtc_streamer(
                    key="yoga_stream",
                    video_processor_factory=lambda: YogaVideoProcessor(
                        _model,
                        _class_names,
                        _temperature,
                    ),
                    media_stream_constraints={"video": True, "audio": False},
                    async_processing=True,
                )
            else:
                st.warning(
                    "Real-time mode requires optional packages. Install `streamlit-webrtc` and `aiortc`, "
                    "or use Snapshot mode."
                )
        else:
            st.markdown(
                "Click **Start Camera** and hold your yoga pose. Snapshot predictions are smoothed "
                "across captures for stability."
            )

            control_col, speed_col = st.columns([1, 2])
            with control_col:
                camera_btn = "⏹ Stop Camera" if st.session_state.webcam_active else "▶️ Start Camera"
                if st.button(camera_btn, key="toggle_snapshot_camera"):
                    st.session_state.webcam_active = not st.session_state.webcam_active
                    if not st.session_state.webcam_active:
                        st.session_state.webcam_last_frame_hash = None
                    st.rerun()

            with speed_col:
                st.toggle(
                    "Auto-refresh after each capture",
                    key="webcam_auto_refresh",
                    disabled=not st.session_state.webcam_active,
                    help=(
                        "After each captured frame, wait briefly then rerun so the camera input "
                        "is immediately ready for the next capture."
                    ),
                )
                st.slider(
                    "Refresh delay (seconds)",
                    min_value=0.5,
                    max_value=2.5,
                    step=0.1,
                    key="webcam_refresh_interval",
                    disabled=not st.session_state.webcam_active,
                )

            if not st.session_state.webcam_active:
                st.info("Start Camera to begin live-assisted snapshot predictions.")
            else:
                st.caption("Capture repeatedly while holding pose transitions for smoother feedback.")

                SMOOTHING_ALPHA = 0.35
                CONFIDENCE_THRESHOLD = 0.45
                col_cam, col_result = st.columns([1, 1])

                with col_cam:
                    cam_frame = st.camera_input("📸 Hold your pose and capture", key="webcam_input")

                if cam_frame is not None:
                    frame_hash = hashlib.sha1(cam_frame.getvalue()).hexdigest()
                    is_new_frame = frame_hash != st.session_state.webcam_last_frame_hash

                    if is_new_frame or st.session_state.webcam_result is None:
                        with st.spinner("Analysing..."):
                            result = predict_image(cam_frame, use_tta=True, show_gradcam_spinner=False)

                        st.session_state.webcam_last_frame_hash = frame_hash
                        new_probs = np.asarray(result["all_probs"], dtype=np.float64)

                        if st.session_state.webcam_smoothed_probs is None:
                            st.session_state.webcam_smoothed_probs = new_probs
                        else:
                            prev = st.session_state.webcam_smoothed_probs
                            st.session_state.webcam_smoothed_probs = (
                                SMOOTHING_ALPHA * new_probs + (1.0 - SMOOTHING_ALPHA) * prev
                            )

                        st.session_state.webcam_frame_count += 1
                        st.session_state.webcam_result = result
                    else:
                        result = st.session_state.webcam_result

                    smooth_probs = st.session_state.webcam_smoothed_probs
                    if smooth_probs is None:
                        smooth_probs = np.asarray(result["all_probs"], dtype=np.float64)
                        st.session_state.webcam_smoothed_probs = smooth_probs

                    smooth_idx = int(np.argmax(smooth_probs))
                    smooth_conf = float(smooth_probs[smooth_idx])
                    smooth_cls = st.session_state.class_names[smooth_idx]
                    smooth_disp = smooth_cls.replace("_", " ").title()
                    smooth_skt = get_sanskrit_name(smooth_cls)
                    accent = get_pose_details(smooth_cls).get("color", "#667eea")

                    with col_result:
                        if smooth_conf < CONFIDENCE_THRESHOLD:
                            st.warning(
                                f"⚠️ Low confidence ({smooth_conf:.0%}) - try a clearer pose or better lighting."
                            )
                        else:
                            conf_pct = round(smooth_conf * 100, 1)
                            st.markdown(f"""
                            <div class="pred-card">
                              <p class="pred-title">{smooth_disp}</p>
                              <p class="pred-sub">{smooth_skt}</p>
                              <div class="conf-bar-wrap">
                                <div class="conf-bar" style="width:{conf_pct}%;
                                  background:linear-gradient(90deg,{accent},{accent}aa)"></div>
                              </div>
                              <p class="conf-label">
                                Smoothed confidence: <strong style="color:{accent}">{conf_pct}%</strong>
                                &nbsp;·&nbsp; Frame #{st.session_state.webcam_frame_count}
                              </p>
                            </div>
                            """, unsafe_allow_html=True)

                        cam_band_color, cam_band_label = confidence_band(smooth_conf)
                        st.markdown(
                            render_confidence_ring(smooth_conf, cam_band_color, cam_band_label),
                            unsafe_allow_html=True,
                        )

                        smooth_top3 = np.argsort(smooth_probs)[::-1][:3]
                        for i in smooth_top3:
                            lbl = st.session_state.class_names[i].replace("_", " ").title()
                            pct = round(float(smooth_probs[i]) * 100, 1)
                            color = accent if i == smooth_idx else "#667eea"
                            bar_w = round(float(smooth_probs[i]) * 100)
                            st.markdown(f"""
                            <div class="top3-row">
                              <span class="top3-label">{lbl}</span>
                              <div class="top3-bar-bg">
                                <div class="top3-bar" style="width:{bar_w}%;background:{color}"></div>
                              </div>
                              <span class="top3-pct">{pct}%</span>
                            </div>
                            """, unsafe_allow_html=True)

                        # Generate pose landmark overlay for snapshot
                        cam_img_bgr = cv2.cvtColor(np.array(result["img_resized"]), cv2.COLOR_RGB2BGR)
                        pose_img_bgr, _ = draw_pose_landmarks(cam_img_bgr.copy(), min_detection_confidence=0.4)
                        pose_img_rgb = cv2.cvtColor(pose_img_bgr, cv2.COLOR_BGR2RGB)

                        c1, c2, c3 = st.columns(3)
                        c1.image(result["img_resized"], caption="📷 Captured", use_container_width=True)
                        c2.image(pose_img_rgb, caption="🦴 Pose Guide", use_container_width=True)
                        c3.image(result["overlay"], caption="🔥 Grad-CAM", use_container_width=True)

                        info = get_pose_details(smooth_cls)
                        if info and smooth_conf >= CONFIDENCE_THRESHOLD:
                            st.markdown(f"""
                            <div class="info-card">
                              <h4>📋 {smooth_disp}</h4>
                              <p><strong>Benefits:</strong> {info['benefits']}</p>
                              <p><strong>Form cues:</strong> {info['cues']}</p>
                              <p><strong>Difficulty:</strong> {info['difficulty']}</p>
                            </div>
                            """, unsafe_allow_html=True)

                    if st.session_state.webcam_auto_refresh and is_new_frame:
                        time.sleep(float(st.session_state.webcam_refresh_interval))
                        st.rerun()

            if st.session_state.webcam_frame_count > 0:
                if st.button("🔄 Reset smoothing", key="reset_smooth"):
                    st.session_state.webcam_smoothed_probs = None
                    st.session_state.webcam_frame_count = 0
                    st.session_state.webcam_result = None
                    st.session_state.webcam_last_frame_hash = None
                    st.rerun()

    st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2 — DATASET UPLOAD & EDA
# ═══════════════════════════════════════════════════════════════════════════

if show_dataset_train:
    with st.expander(
        "📁  Dataset — Upload & Explore"
        + (" (dataset loaded)" if st.session_state.dataset_loaded else ""),
        expanded=not st.session_state.dataset_loaded,
    ):
        st.markdown(
            "Upload yoga_dataset.zip - must contain train/ and val/ subfolders, "
            "each with one folder per class. **Required only for training.**"
        )

        uploaded_zip = st.file_uploader(
            "yoga_dataset.zip",
            type=["zip"],
            key="dataset_zip",
            label_visibility="collapsed",
        )

        if uploaded_zip is not None and not st.session_state.dataset_loaded:
            with st.spinner("Extracting ZIP..."):
                dp = process_uploaded_zip(uploaded_zip)
            if dp:
                st.session_state.dataset_path = dp
                st.session_state.dataset_loaded = True
                st.success("✅ Dataset extracted successfully.")

        if st.session_state.dataset_loaded:
            st.markdown('<div class="section-header">📊 <span>EDA — Dataset Overview</span></div>',
                        unsafe_allow_html=True)
            show_dataset_summary(st.session_state.dataset_path)

            st.markdown('<div class="section-header">🎨 <span>Augmentation Preview</span></div>',
                        unsafe_allow_html=True)
            show_augmentation_preview(st.session_state.dataset_path)

    train_header = (
        "🚀  Model Training"
        + (" ✅" if st.session_state.trained else " - required")
    )

    with st.expander(train_header, expanded=not st.session_state.trained):

        if st.session_state.trained:
            src = st.session_state.model_source
            if src == "file":
                st.info(
                    "✅ Model loaded from saved checkpoint (asanai_saved/). "
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
                prog = st.progress(0)
                stat = st.empty()
                log_data = []

                with st.spinner("Training... this may take a few minutes."):
                    tl, vl, cnames, skipped = create_data_loaders(
                        st.session_state.dataset_path,
                        use_weighted_sampler=use_weighted_sampler,
                    )
                    show_skipped_images_warning(skipped)
                    model, history, stopped_at = run_training(
                        tl, vl, cnames, prog, stat, log_data,
                        class_weight_mode=class_weight_mode,
                    )
                    history["sampler"] = "weighted" if use_weighted_sampler else "standard"

                save_model(model, history, cnames)
                _get_predictions_cached.clear()

                st.session_state.model = model
                st.session_state.history = history
                st.session_state.class_names = cnames
                st.session_state.val_loader = vl
                st.session_state.temperature = float(history.get("temperature", 1.0))
                st.session_state.trained = True
                st.session_state.model_source = "trained"
                st.session_state.webcam_smoothed_probs = None
                st.session_state.webcam_frame_count = 0
                st.session_state.webcam_result = None

                if stopped_at < EPOCHS:
                    st.success(
                        f"✅ Training complete - early stopping fired at epoch "
                        f"**{stopped_at}/{EPOCHS}** (patience={PATIENCE}). "
                        f"Best weights restored automatically."
                    )
                else:
                    st.success(f"✅ Training complete - ran all {EPOCHS} epochs.")

                st.write("**Epoch log**")
                st.dataframe(pd.DataFrame(log_data).set_index("epoch"),
                             use_container_width=True)

        else:
            st.info("Upload the dataset ZIP above to enable training.")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4 — TRAINING RESULTS (learning curves)
# ═══════════════════════════════════════════════════════════════════════════

if show_results:
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

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Final Train Acc", f"{h['accuracy'][-1]:.2%}")
            m2.metric("Best Val Acc", f"{best_val_acc:.2%}")
            m3.metric("Final Train Loss", f"{h['loss'][-1]:.4f}")
            m4.metric("Val Loss @ Best", f"{best_val_loss:.4f}")
            m5.metric("Epochs Run", f"{len(h['accuracy'])}")

            st.caption(
                f"Best validation epoch: {best_epoch}. "
                f"Class weighting: {class_weighting} ({class_weight_mode}). "
                f"Sampler: {sampler_mode}. "
                f"Confidence temperature: {calibration_temp:.2f}."
            )

            plot_learning_curves(h)
    else:
        st.info("Training results will appear after you train or load a model.")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5 — EVALUATION METRICS
# ═══════════════════════════════════════════════════════════════════════════

if show_evaluation:
    if st.session_state.trained:

        if st.session_state.val_loader is None and st.session_state.dataset_loaded:
            _, vl, _, skipped = create_data_loaders(st.session_state.dataset_path)
            show_skipped_images_warning(skipped)
            st.session_state.val_loader = vl

        if st.session_state.val_loader is not None:
            history_temp = (st.session_state.history or {}).get("temperature")
            try:
                needs_calibration = (not history_temp) or float(history_temp) == 1.0
            except (TypeError, ValueError):
                needs_calibration = True

            if needs_calibration:
                with st.spinner("Calibrating confidence on validation set..."):
                    ensure_temperature_calibrated()
            else:
                st.session_state.temperature = float(history_temp)

            with st.expander("📊  Model Evaluation — Full Metrics", expanded=True):
                with st.spinner("Running inference on validation set..."):
                    y_pred, y_probs, y_true = _get_predictions_cached(
                        id(st.session_state.model),
                        id(st.session_state.val_loader),
                        round(get_temperature(), 4),
                    )

                class_names = st.session_state.class_names
                overall_acc = (y_pred == y_true).mean()
                ev1, ev2 = st.columns(2)
                ev1.metric("Overall Val Accuracy", f"{overall_acc:.2%}")
                ev2.metric("Validation Images", f"{len(y_true)}")

                tab_cm, tab_cls, tab_pca, tab_roc = st.tabs([
                    "Confusion Matrix",
                    "Classification Report",
                    "Per-class Accuracy",
                    "ROC Curves",
                ])

                with tab_cm:
                    fig = plot_confusion_matrix_fig(y_pred, y_true, class_names)
                    st.pyplot(fig, use_container_width=True); plt.close()

                with tab_cls:
                    show_classification_report(y_pred, y_true, class_names)

                with tab_pca:
                    fig = plot_per_class_accuracy(y_pred, y_true, class_names)
                    st.pyplot(fig, use_container_width=True); plt.close()
                    st.caption("Green ≥ 80%, orange ≥ 60%, red < 60%")

                with tab_roc:
                    fig = plot_roc_curves(y_probs, y_true, class_names)
                    st.pyplot(fig, use_container_width=True); plt.close()

        else:
            st.info(
                "ℹ️ Evaluation metrics require the validation dataset. "
                "Upload the dataset ZIP in Dataset & Train - "
                "the val loader will be created automatically."
            )
    else:
        st.info("Train or load a model first to access evaluation metrics.")


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
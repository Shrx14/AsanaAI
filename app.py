# ═══════════════════════════════════════════════════════════════════════════
# AsanaAI — Practical 4 & 5
# Explainable Deep Learning System for Yoga Pose Recognition
# ═══════════════════════════════════════════════════════════════════════════

# ── IMPORTS ──────────────────────────────────────────────────────────────────
import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os
import zipfile
import tempfile
import shutil
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from PIL import Image

# ── GPU SETUP ────────────────────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GPU_AVAILABLE = torch.cuda.is_available()

if GPU_AVAILABLE:
    print(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA Version: {torch.version.cuda}")
    print(f"   Device: {DEVICE}")
else:
    print("⚠️ GPU not available, using CPU")

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AsanaAI",
    page_icon="🧘",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── SESSION STATE DEFAULTS ───────────────────────────────────────────────────
defaults = {
    'model': None,
    'trained': False,
    'class_names': [],
    'history': None,
    'dataset_path': None,
    'temp_dir': None,
    'val_loader': None,
    'dataset_loaded': False
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ── REFERENCE DICTIONARIES ───────────────────────────────────────────────────
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

# ── CSS STYLING ──────────────────────────────────────────────────────────────
css = """
<style>
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    body {
        background: #0a0a0f;
        color: #f1f5f9;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
    }
    
    .main {
        background-color: #0a0a0f;
    }
    
    /* Hero Section */
    .hero {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    
    .hero h1 {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        color: white;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
    }
    
    .hero p {
        font-size: 1.1rem;
        color: rgba(255, 255, 255, 0.9);
        opacity: 95%;
    }
    
    /* Glass Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.04);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    /* Section Headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #f1f5f9;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid;
        border-image: linear-gradient(90deg, #667eea, #764ba2, #06b6d4) 1;
    }
    
    /* Metric Cards */
    .metric-card {
        background: rgba(102, 126, 234, 0.1);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 8px;
        padding: 1.5rem;
        text-align: center;
        margin: 1rem 0;
    }
    
    .metric-card .metric-label {
        color: #94a3b8;
        font-size: 0.85rem;
        margin-bottom: 0.5rem;
    }
    
    .metric-card .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
    }
    
    /* Pose Badge */
    .pose-badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 1rem 2rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 1.1rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .pose-badge em {
        color: #06b6d4;
        font-style: italic;
        margin: 0 0.5rem;
    }
    
    /* Info Card */
    .info-card {
        background: rgba(16, 185, 129, 0.05);
        border-left: 4px solid #10b981;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        line-height: 1.8;
    }
    
    .info-card h4 {
        color: #10b981;
        margin-bottom: 1rem;
        font-size: 1.2rem;
    }
    
    .info-card p {
        color: #cbd5e1;
        margin: 0.5rem 0;
    }
    
    .info-card strong {
        color: #f1f5f9;
    }
    
    .info-card hr {
        border: none;
        height: 1px;
        background: rgba(255, 255, 255, 0.1);
        margin: 1rem 0;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: white !important;
        border: none !important;
        padding: 0.75rem 2rem !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.5) !important;
    }
    
    /* Success, Warning, Error */
    .stSuccess {
        background: rgba(16, 185, 129, 0.1) !important;
        border: 1px solid #10b981 !important;
        color: #10b981 !important;
    }
    
    .stWarning {
        background: rgba(245, 158, 11, 0.1) !important;
        border: 1px solid #f59e0b !important;
        color: #fcd34d !important;
    }
    
    .stError {
        background: rgba(239, 68, 68, 0.1) !important;
        border: 1px solid #ef4444 !important;
        color: #fca5a5 !important;
    }
    
    /* Slider */
    .stSlider {
        padding: 1rem 0;
    }
    
    /* Dataframe */
    .stDataFrame {
        background: rgba(255, 255, 255, 0.02) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
    }
</style>
"""

st.markdown(css, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def process_uploaded_zip(uploaded_file):
    """
    Extract ZIP, validate structure, return dataset path.
    Also resets training state on new upload (refinement #14).
    """
    # Clean up old temp directory
    if st.session_state.temp_dir:
        shutil.rmtree(st.session_state.temp_dir, ignore_errors=True)
    
    # Reset training state on new upload
    st.session_state.trained = False
    st.session_state.model = None
    st.session_state.history = None
    st.session_state.class_names = []
    st.session_state.val_loader = None
    
    # Create temp directory and extract ZIP
    tmp_dir = tempfile.mkdtemp()
    st.session_state.temp_dir = tmp_dir
    
    zip_path = os.path.join(tmp_dir, "dataset.zip")
    with open(zip_path, 'wb') as f:
        f.write(uploaded_file.read())
    
    extract_path = os.path.join(tmp_dir, "extracted")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
    except Exception as e:
        st.error(f"Failed to extract ZIP: {e}")
        return None
    
    # Find root folder containing train/ and val/
    dataset_root = None
    for root, dirs, _ in os.walk(extract_path):
        if 'train' in dirs and 'val' in dirs:
            dataset_root = root
            break
    
    if dataset_root is None:
        st.error("Invalid ZIP structure. Must contain `train/` and `val/` folders.")
        return None
    
    return dataset_root


class ImageDataset(Dataset):
    """Custom PyTorch Dataset for loading images from directory structure."""
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_names = []
        
        # Get sorted class names (alphabetical)
        self.class_names = sorted([d for d in os.listdir(root_dir) 
                                   if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.class_names)}
        
        # Collect all image paths and labels
        for class_name in self.class_names:
            class_dir = os.path.join(root_dir, class_name)
            for img_name in sorted(os.listdir(class_dir)):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(class_dir, img_name)
                    self.images.append(img_path)
                    self.labels.append(self.class_to_idx[class_name])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path)
        # Convert palette images to RGB to prevent transparency warnings
        if image.mode in ('P', 'PA'):
            image = image.convert('RGB')
        elif image.mode != 'RGB':
            image = image.convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def create_data_loaders(dataset_path, batch_size=16):
    """
    Create training and validation DataLoaders with appropriate transforms.
    Train: WITH augmentation | Val: WITHOUT augmentation (refinement #5)
    """
    # Training transforms WITH augmentation
    train_transforms = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, scale=(0.9, 1.1)),  # Zoom equivalent
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet stats
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Validation transforms WITHOUT augmentation
    val_transforms = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = ImageDataset(os.path.join(dataset_path, 'train'), 
                                 transform=train_transforms)
    val_dataset = ImageDataset(os.path.join(dataset_path, 'val'), 
                               transform=val_transforms)
    
    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                           shuffle=False, num_workers=0)
    
    return train_loader, val_loader, train_dataset.class_names


class YogaPoseModel(nn.Module):
    """MobileNetV2 + custom head for yoga pose classification."""
    
    def __init__(self, num_classes):
        super(YogaPoseModel, self).__init__()
        
        # Load pre-trained MobileNetV2
        self.base_model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        
        # Freeze all base layers
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Replace classifier with custom head
        in_features = self.base_model.classifier[1].in_features
        self.base_model.classifier = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.base_model(x)


def update_progress(epoch, total_epochs, progress_bar, status_text, 
                   train_loss, train_acc, val_loss, val_acc):
    """Update Streamlit progress bar and status during training."""
    progress = (epoch + 1) / total_epochs
    progress_bar.progress(progress)
    status_text.markdown(
        f"**Epoch {epoch+1}/{total_epochs}** — "
        f"Train Loss: `{train_loss:.4f}` | "
        f"Train Acc: `{train_acc:.4f}` | "
        f"Val Acc: `{val_acc:.4f}` | "
        f"Val Loss: `{val_loss:.4f}`"
    )


def display_dataset_summary(dataset_path):
    """Display dataset summary: metrics, class distribution, sample images."""
    train_path = os.path.join(dataset_path, 'train')
    val_path = os.path.join(dataset_path, 'val')
    
    # Count images per class
    class_counts_train = {}
    class_counts_val = {}
    
    for class_name in os.listdir(train_path):
        class_dir = os.path.join(train_path, class_name)
        if os.path.isdir(class_dir):
            class_counts_train[class_name] = len([f for f in os.listdir(class_dir) 
                                                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    for class_name in os.listdir(val_path):
        class_dir = os.path.join(val_path, class_name)
        if os.path.isdir(class_dir):
            class_counts_val[class_name] = len([f for f in os.listdir(class_dir) 
                                                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    total_train = sum(class_counts_train.values())
    total_val = sum(class_counts_val.values())
    total_images = total_train + total_val
    
    # Metric cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📚 Total Classes", len(class_counts_train))
    col2.metric("📷 Total Images", total_images)
    col3.metric("🏋️ Train", total_train)
    col4.metric("✅ Validation", total_val)
    
    # Class distribution bar chart
    st.write("**Class Distribution**")
    class_df = pd.DataFrame({
        'Class': list(class_counts_train.keys()),
        'Train': list(class_counts_train.values()),
        'Val': [class_counts_val.get(c, 0) for c in class_counts_train.keys()]
    })
    
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor('#0a0a0f')
    ax.set_facecolor('#111827')
    
    class_names = [c.replace('_', ' ').title() for c in class_df['Class']]
    x = np.arange(len(class_names))
    width = 0.35
    
    ax.barh(x - width/2, class_df['Train'], width, label='Train', color='#667eea')
    ax.barh(x + width/2, class_df['Val'], width, label='Val', color='#06b6d4')
    ax.set_yticks(x)
    ax.set_yticklabels(class_names)
    ax.set_xlabel('Images', color='#94a3b8')
    ax.set_title('Images per Class', color='white', fontsize=12, pad=15)
    ax.legend(facecolor='#1f2937', labelcolor='white')
    ax.tick_params(colors='#94a3b8')
    ax.grid(True, axis='x', alpha=0.1)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    # Sample images grid
    st.write("**Sample Images**")
    cols = st.columns(4)
    
    class_list = list(class_counts_train.keys())
    for idx, class_name in enumerate(class_list[:8]):
        col = cols[idx % 4]
        class_dir = os.path.join(train_path, class_name)
        images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if images:
            img_path = os.path.join(class_dir, images[0])
            img = Image.open(img_path).convert('RGB')
            col.image(img, caption=class_name.replace('_', '\n').title(), width=200)


# ═══════════════════════════════════════════════════════════════════════════
# P5 — EVALUATION & GRAD-CAM FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def get_predictions(model, val_loader):
    """Run inference on validation set, return predictions, probabilities, and true labels."""
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())

    return np.array(all_preds), np.array(all_probs), np.array(all_labels)


def plot_confusion_matrix(model, val_loader, class_names):
    """Generate confusion matrix heatmap with dark theme."""
    y_pred, _, y_true = get_predictions(model, val_loader)

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

    ax.set_title('Confusion Matrix \u2014 Validation Set',
                 color='white', fontsize=14, pad=20)
    ax.set_xlabel('Predicted Label', color='#94a3b8', labelpad=10)
    ax.set_ylabel('True Label', color='#94a3b8', labelpad=10)

    plt.xticks(rotation=45, ha='right', color='#94a3b8')
    plt.yticks(rotation=0, color='#94a3b8')

    plt.tight_layout()
    return fig


def display_classification_report(model, val_loader, class_names):
    """Display classification report as styled DataFrame."""
    y_pred, _, y_true = get_predictions(model, val_loader)

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


def plot_roc_curves(model, val_loader, class_names):
    """Generate ROC curves (One-vs-Rest) with AUC scores."""
    _, y_pred_probs, y_true = get_predictions(model, val_loader)
    num_classes = len(class_names)

    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor('#0a0a0f')
    ax.set_facecolor('#111827')

    palette = ['#667eea', '#764ba2', '#06b6d4', '#10b981',
               '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899']

    for i, (name, color) in enumerate(zip(class_names, palette)):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
        roc_auc = auc(fpr, tpr)
        label = f"{name.replace('_', ' ').title()} (AUC = {roc_auc:.2f})"
        ax.plot(fpr, tpr, color=color, linewidth=2, label=label)

    ax.plot([0, 1], [0, 1], 'white', linestyle='--', alpha=0.3, linewidth=1)
    ax.set_xlabel('False Positive Rate', color='#94a3b8')
    ax.set_ylabel('True Positive Rate', color='#94a3b8')
    ax.set_title('ROC Curves \u2014 One-vs-Rest', color='white', fontsize=14)
    ax.legend(loc='lower right', fontsize=8,
              facecolor='#1f2937', labelcolor='white', framealpha=0.8)
    ax.tick_params(colors='#94a3b8')
    ax.grid(True, alpha=0.1)

    plt.tight_layout()
    return fig


def get_last_conv_layer(model):
    """Find the last Conv2d layer in MobileNetV2 features backbone."""
    last_conv = None
    for module in model.base_model.features.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    if last_conv is None:
        raise ValueError("No Conv2d layer found in model.")
    return last_conv


def generate_gradcam(model, img_tensor, class_idx):
    """
    Generate Grad-CAM heatmap and overlay using PyTorch hooks.
    Returns (heatmap_colored, overlay) as numpy uint8 arrays.
    """
    model.eval()

    activations = []
    gradients = []

    target_layer = get_last_conv_layer(model)

    # Hooks to capture activations and gradients
    def forward_hook(module, input, output):
        activations.append(output.detach())

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    fwd_handle = target_layer.register_forward_hook(forward_hook)
    bwd_handle = target_layer.register_full_backward_hook(backward_hook)

    # Forward pass — need requires_grad on input so gradients flow through frozen layers
    img_input = img_tensor.clone().requires_grad_(True)
    output = model(img_input)

    # Backward pass for target class
    model.zero_grad()
    target_score = output[0, class_idx]
    target_score.backward()

    # Remove hooks
    fwd_handle.remove()
    bwd_handle.remove()

    # Compute Grad-CAM
    activation = activations[0][0]  # (C, H, W)
    gradient = gradients[0][0]      # (C, H, W)

    # Pool gradients across spatial dimensions
    pooled_grads = torch.mean(gradient, dim=(1, 2))  # (C,)

    # Weight each feature map by its gradient importance
    for i in range(activation.shape[0]):
        activation[i] *= pooled_grads[i]

    heatmap = torch.mean(activation, dim=0)  # (H, W)
    heatmap = torch.relu(heatmap)  # ReLU — retain only positive influence
    heatmap = heatmap / (torch.max(heatmap) + 1e-8)  # Normalize (refinement #13)
    heatmap = heatmap.cpu().numpy()

    # Resize to image dimensions
    heatmap_resized = cv2.resize(heatmap, (128, 128))

    # Apply JET colormap: blue=low importance, red=high importance
    heatmap_colored = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Denormalize original image (undo ImageNet normalization)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_denorm = img_tensor[0].cpu() * std + mean
    img_denorm = torch.clamp(img_denorm, 0, 1)
    original = np.uint8(img_denorm.permute(1, 2, 0).numpy() * 255)

    # Blend heatmap with original image
    overlay = cv2.addWeighted(original, 0.6, heatmap_colored, 0.4, 0)

    return heatmap_colored, overlay


# ═══════════════════════════════════════════════════════════════════════════
# PAGE SECTIONS
# ═══════════════════════════════════════════════════════════════════════════

# Hero Header
st.markdown("""
<div class="hero">
    <h1>🧘 AsanaAI</h1>
    <p>Explainable Deep Learning System for Yoga Pose Recognition</p>
</div>
""", unsafe_allow_html=True)

# GPU Warning if not available
if not GPU_AVAILABLE:
    st.warning(
        "⚠️ **GPU not detected** — Using CPU mode. Training will be slower. "
        "For GPU support, ensure CUDA/cuDNN are properly installed."
    )

st.markdown("---")

# ── SECTION 1: DATASET UPLOAD ────────────────────────────────────────────────
st.markdown('<h2 class="section-header">📁 Upload Dataset</h2>', 
            unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload yoga_dataset.zip",
    type=['zip'],
    help="Must contain yoga_dataset/train/ and yoga_dataset/val/ folders"
)

if uploaded_file:
    with st.spinner("Processing ZIP file..."):
        dataset_path = process_uploaded_zip(uploaded_file)
        
        if dataset_path:
            st.session_state.dataset_path = dataset_path
            st.session_state.dataset_loaded = True
            st.success("✅ Dataset loaded successfully!")
        else:
            st.session_state.dataset_loaded = False

st.markdown("---")

# ── SECTION 2: DATASET SUMMARY ───────────────────────────────────────────────
if st.session_state.dataset_loaded:
    st.markdown('<h2 class="section-header">📊 Dataset Summary</h2>', 
                unsafe_allow_html=True)
    display_dataset_summary(st.session_state.dataset_path)
    st.markdown("---")

# ── SECTION 3: MODEL TRAINING ────────────────────────────────────────────────
if st.session_state.dataset_loaded and not st.session_state.trained:
    st.markdown('<h2 class="section-header">🚀 Model Training</h2>', 
                unsafe_allow_html=True)
    
    if st.button("🚀 Begin Training", type="primary", use_container_width=True):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        with st.spinner("Training model..."):
            # Create data loaders
            train_loader, val_loader, class_names = create_data_loaders(
                st.session_state.dataset_path, 
                batch_size=16
            )
            num_classes = len(class_names)
            
            # Build model and move to device
            model = YogaPoseModel(num_classes).to(DEVICE)
            
            # Loss and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            EPOCHS = 10
            patience = 3
            best_val_loss = float('inf')
            patience_counter = 0
            
            # History tracking
            history = {
                'loss': [],
                'accuracy': [],
                'val_loss': [],
                'val_accuracy': []
            }
            
            # Training loop
            for epoch in range(EPOCHS):
                # Training phase
                model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0
                
                for batch_idx, (images, labels) in enumerate(train_loader):
                    images, labels = images.to(DEVICE), labels.to(DEVICE)
                    
                    # Forward pass
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    # Track metrics
                    train_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    train_total += labels.size(0)
                    train_correct += (predicted == labels).sum().item()
                
                train_loss /= len(train_loader)
                train_acc = train_correct / train_total
                
                # Validation phase
                model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for images, labels in val_loader:
                        images, labels = images.to(DEVICE), labels.to(DEVICE)
                        
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                        
                        val_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        val_total += labels.size(0)
                        val_correct += (predicted == labels).sum().item()
                
                val_loss /= len(val_loader)
                val_acc = val_correct / val_total
                
                # Store history
                history['loss'].append(train_loss)
                history['accuracy'].append(train_acc)
                history['val_loss'].append(val_loss)
                history['val_accuracy'].append(val_acc)
                
                # Update progress
                update_progress(epoch, EPOCHS, progress_bar, status_text, 
                               train_loss, train_acc, val_loss, val_acc)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = model.state_dict().copy()
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    # Restore best weights
                    model.load_state_dict(best_model_state)
                    break
            
            # Persist to session state
            st.session_state.model = model
            st.session_state.trained = True
            st.session_state.history = history
            st.session_state.class_names = class_names
            st.session_state.val_loader = val_loader
            
            st.success("✅ Training complete!")

elif st.session_state.trained:
    st.info("✅ Model trained. Scroll down for results.")
    st.markdown("---")

# ── SECTION 4: TRAINING RESULTS ──────────────────────────────────────────────
if st.session_state.trained and st.session_state.history:
    st.markdown('<h2 class="section-header">📈 Training Results</h2>', 
                unsafe_allow_html=True)
    
    history = st.session_state.history
    
    # Metric cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🎯 Train Accuracy", f"{history['accuracy'][-1]:.2%}")
    col2.metric("✅ Val Accuracy", f"{history['val_accuracy'][-1]:.2%}")
    col3.metric("📉 Train Loss", f"{history['loss'][-1]:.4f}")
    col4.metric("📊 Val Loss", f"{history['val_loss'][-1]:.4f}")
    
    # Learning curves
    col_acc, col_loss = st.columns(2)
    
    with col_acc:
        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor('#0a0a0f')
        ax.set_facecolor('#111827')
        
        ax.plot(history['accuracy'], label='Train Accuracy', 
               color='#667eea', linewidth=2.5)
        ax.plot(history['val_accuracy'], label='Val Accuracy', 
               color='#06b6d4', linewidth=2.5, linestyle='--')
        
        ax.set_xlabel('Epoch', color='#94a3b8')
        ax.set_ylabel('Accuracy', color='#94a3b8')
        ax.set_title('Model Accuracy', color='white', fontsize=12)
        ax.legend(facecolor='#1f2937', labelcolor='white')
        ax.tick_params(colors='#94a3b8')
        ax.grid(True, alpha=0.1)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    with col_loss:
        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor('#0a0a0f')
        ax.set_facecolor('#111827')
        
        ax.plot(history['loss'], label='Train Loss', 
               color='#667eea', linewidth=2.5)
        ax.plot(history['val_loss'], label='Val Loss', 
               color='#f59e0b', linewidth=2.5, linestyle='--')
        
        ax.set_xlabel('Epoch', color='#94a3b8')
        ax.set_ylabel('Loss', color='#94a3b8')
        ax.set_title('Model Loss', color='white', fontsize=12)
        ax.legend(facecolor='#1f2937', labelcolor='white')
        ax.tick_params(colors='#94a3b8')
        ax.grid(True, alpha=0.1)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    st.markdown("---")

# ── SECTION 5: MODEL EVALUATION ──────────────────────────────────────────────
if st.session_state.trained and st.session_state.val_loader is not None:
    st.markdown('<h2 class="section-header">📊 Model Evaluation</h2>',
                unsafe_allow_html=True)

    _eval_model = st.session_state.model
    _eval_val_loader = st.session_state.val_loader
    _eval_class_names = st.session_state.class_names

    # ── Confusion Matrix ──
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.write("**Confusion Matrix**")
    fig_cm = plot_confusion_matrix(_eval_model, _eval_val_loader, _eval_class_names)
    st.pyplot(fig_cm)
    plt.close()
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Classification Report ──
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.write("**Classification Report**")
    display_classification_report(_eval_model, _eval_val_loader, _eval_class_names)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── ROC Curves ──
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.write("**ROC Curves (One-vs-Rest)**")
    fig_roc = plot_roc_curves(_eval_model, _eval_val_loader, _eval_class_names)
    st.pyplot(fig_roc)
    plt.close()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

# ── SECTION 6: PREDICTION + GRAD-CAM ─────────────────────────────────────────
if st.session_state.trained:
    st.markdown('<h2 class="section-header">🔮 Pose Prediction & Grad-CAM</h2>',
                unsafe_allow_html=True)

    pred_image = st.file_uploader(
        "Upload a yoga pose image to classify",
        type=['jpg', 'jpeg', 'png'],
        key='prediction_upload'
    )

    if pred_image:
        _pred_model = st.session_state.model
        _pred_class_names = st.session_state.class_names

        # Load and prepare image
        img = Image.open(pred_image).convert('RGB')
        img_resized = img.resize((128, 128))

        # Prepare tensor with validation transforms
        pred_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        img_tensor = pred_transform(img).unsqueeze(0).to(DEVICE)

        # Predict
        _pred_model.eval()
        with torch.no_grad():
            outputs = _pred_model(img_tensor)
            probs = torch.softmax(outputs, dim=1)

        class_idx = int(torch.argmax(probs[0]).item())
        confidence = float(probs[0][class_idx].item())
        predicted_class = _pred_class_names[class_idx]
        display_name = predicted_class.replace('_', ' ').title()
        sanskrit = sanskrit_names.get(predicted_class, '')

        # Generate Grad-CAM
        heatmap, overlay = generate_gradcam(_pred_model, img_tensor, class_idx)

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
        c1.image(img_resized,   caption="📷 Original",          use_container_width=True)
        c2.image(heatmap,       caption="🔥 Grad-CAM Heatmap",  use_container_width=True)
        c3.image(overlay,       caption="🔍 Overlay",           use_container_width=True)

        # Probability bar chart
        predictions_np = probs[0].cpu().numpy()
        prob_df = pd.DataFrame({
            'Pose': [c.replace('_', ' ').title() for c in _pred_class_names],
            'Confidence': predictions_np
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

    st.markdown("---")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #94a3b8; font-size: 0.85rem; margin-top: 2rem;">
    <p>AsanaAI is a learning tool. It does not replace certified yoga instruction.</p>
    <p>Built for CELBC608 — Data Science Laboratory | Semester VI</p>
</div>
""", unsafe_allow_html=True)

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

# ── SECTION 5: MODEL EVALUATION (P5 Placeholder) ─────────────────────────────
if st.session_state.trained:
    st.markdown('<h2 class="section-header">📊 Model Evaluation (P5)</h2>', 
                unsafe_allow_html=True)
    st.info("Evaluation metrics will be displayed here in Practical 5 (Confusion Matrix, Classification Report, etc.)")
    st.markdown("---")

# ── SECTION 6: PREDICTION + GRAD-CAM (P5 Placeholder) ────────────────────────
if st.session_state.trained:
    st.markdown('<h2 class="section-header">🔮 Pose Prediction (P5)</h2>', 
                unsafe_allow_html=True)
    st.info("Prediction UI with Grad-CAM visualization will be available in Practical 5")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #94a3b8; font-size: 0.85rem; margin-top: 2rem;">
    <p>AsanaAI is a learning tool. It does not replace certified yoga instruction.</p>
    <p>Built for CELBC608 — Data Science Laboratory | Semester VI</p>
</div>
""", unsafe_allow_html=True)

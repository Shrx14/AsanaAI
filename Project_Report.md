# AsanaAI — Yoga Pose Recognition (Up to 10 Pages)

> **Flow reference (chalkboard):** Application selection → Dataset selection → EDA & visualization → Pre-processing → Training neural network.  
> **Goal:** Explainable image-classification pipeline for eight yoga poses with Grad-CAM overlays.

## 1) Problem Statement & Description
- **Problem:** Many beginners practice yoga without instructors, increasing risk of poor form and injury. We need an accessible tool that classifies a pose from a single image and highlights the body regions driving the decision.
- **Solution (AsanaAI):** Streamlit app that trains a MobileNetV2 transfer-learning model on eight poses and provides Grad-CAM visual explanations plus practice cues (benefits, posture tips).
- **Scope:** Single-image classification (eight classes) with explainability; not a replacement for certified instruction.

## 2) Dataset
- **Source:** [Kaggle — tr1gg3rtrash Yoga Posture Dataset](https://www.kaggle.com/datasets/tr1gg3rtrash/yoga-posture-dataset)
- **Classes used (8):** downward_dog, warrior, tree, cobra, plank, triangle, child_pose, seated_forward_bend.
- **Curation:** Manually picked 120 images per class → 96 train / 24 val (80/20) sorted deterministically; zipped as `yoga_dataset.zip`.
- **Repository copy:** `yoga_dataset.zip` at repo root, structured as:
  ```
  yoga_dataset/
    train/<class>/*.jpg|png   # 96 per class
    val/<class>/*.jpg|png     # 24 per class
  ```

## 3) Data Loading, Pre-processing, Augmentation
- **Transforms (train):** Resize 128×128 → RandomRotation(15°) → RandomAffine scale 0.9–1.1 (zoom) → RandomHorizontalFlip → ToTensor → Normalize (ImageNet mean/std).
- **Transforms (val):** Resize 128×128 → ToTensor → Normalize (ImageNet mean/std).
- **Batches:** 16 images; `shuffle=True` train / `shuffle=False` val.
- **Safety:** Palette images converted to RGB; only `.jpg/.jpeg/.png` ingested.

```python
# app.py:create_data_loaders (high-level)
train_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, scale=(0.9, 1.1)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
val_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
train_loader, val_loader, class_names = create_data_loaders(path, batch_size=16)
```

## 4) Model Architecture
- **Backbone:** `torchvision.models.mobilenet_v2` with ImageNet weights; all base layers frozen.
- **Custom head:** `Linear(in_features→128) → ReLU → Dropout(0.3) → Linear(128→num_classes)`.
- **Device:** Auto CUDA when available; CPU fallback with warning.

```python
# app.py:YogaPoseModel
self.base_model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
for p in self.base_model.parameters():
    p.requires_grad = False
self.base_model.classifier = nn.Sequential(
    nn.Linear(in_features, 128),
    nn.ReLU(inplace=True),
    nn.Dropout(0.3),
    nn.Linear(128, num_classes),
)
```

## 5) Training Pipeline
- **Loss / Optimizer:** CrossEntropyLoss + Adam (lr=0.001).
- **Schedule:** 10 epochs max, batch 16, early stopping patience=3 (restore best weights).
- **Tracking:** Train/val loss & accuracy stored in `history`.
- **Progress UI:** Streamlit progress bar and per-epoch status text.

```python
# app.py (training loop excerpt)
for epoch in range(EPOCHS):
    model.train()
    for images, labels in train_loader:
        outputs = model(images.to(DEVICE))
        loss = criterion(outputs, labels.to(DEVICE))
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images.to(DEVICE))
            val_loss += criterion(outputs, labels.to(DEVICE)).item()
    if val_loss < best_val_loss: best_model_state = model.state_dict()
    else: patience_counter += 1
    if patience_counter >= patience: model.load_state_dict(best_model_state); break
```

## 6) Explainability (Grad-CAM)
- **Target layer:** Last convolutional block of MobileNetV2 (auto-selected).
- **Steps:** Forward pass → register hooks → backward on predicted class → weight activations by pooled gradients → ReLU & normalize (+1e-8 guard) → resize to 128×128 → blend with denormalized input.
- **Outputs:** Heatmap and overlaid RGB image shown alongside predictions.

## 7) Visualizations (insert screenshots)
- **Dataset summary:** Class distribution bars; train/val counts; total classes/images.
- **Training curves:** Loss and accuracy plots over epochs.
- **Confusion matrix:** Normalized heatmap with class labels.
- **Classification report:** Precision/recall/F1 per class.
- **ROC curves:** One-vs-rest ROC & AUC per class (optional in app).
- **Grad-CAM:** Original vs. heatmap overlay for uploaded image.

> _Leave space here for screenshots from Streamlit UI (dataset summary, curves, confusion matrix, Grad-CAM, sample predictions)._

## 8) Evaluation & Metrics
- **Primary:** Validation accuracy (per-epoch) with early stopping.
- **Secondary:** Precision, recall, F1, support (classification report).
- **Per-class AUC:** One-vs-rest ROC where computed.
- **Confusion matrix:** To inspect misclassifications.
- **Qualitative:** Grad-CAM focus alignment with relevant joints/limbs.

## 9) Application & Usage
- **User flow:** Upload `yoga_dataset.zip` → Review dataset summary → Click “Begin Training” → Inspect metrics/curves → Upload single pose image → View predicted class, Sanskrit name, benefits/cues, and Grad-CAM overlay.
- **Intended setting:** Educational/demo tool for beginner practice support; not a medical or coaching substitute.
- **Deployment:** `streamlit run app.py` (GPU recommended; CPU supported but slower).

## 10) Reproduction Checklist
1. Create/activate virtualenv; `pip install -r requirements.txt`.
2. Ensure `yoga_dataset.zip` follows `train/` & `val/` split (80/20, eight classes).
3. Run `streamlit run app.py`; upload dataset; start training.
4. After training, capture:
   - Loss/accuracy curves,
   - Confusion matrix,
   - Classification report,
   - ROC curves (if generated),
   - Grad-CAM overlays for sample predictions.

## 11) Future Improvements (optional)
- Unfreeze last MobileNetV2 blocks for fine-tuning once baseline converges.
- Add data-quality checks (blur/pose detection) before training.
- Export best checkpoint for reuse without retraining each session.
- Introduce lightweight pose estimation overlay for richer cues.

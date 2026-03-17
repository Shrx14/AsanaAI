# AsanaAI — Explainable Yoga Pose Recognition

Streamlit application that trains a MobileNetV2-based classifier on eight yoga poses and explains predictions with Grad-CAM overlays. Built for CELBC608 — Data Science Laboratory.

## Features
- Upload a zipped dataset (`yoga_dataset.zip`) with `train/` and `val/` splits
- Transfer-learning pipeline (PyTorch) with early stopping and accuracy/loss tracking
- Metrics dashboard with confusion matrix, classification report, and ROC curves
- Pose prediction with Grad-CAM visual explanations and practice cues
- GPU auto-detection (falls back to CPU if unavailable)

## Repository layout
- `app.py` — Streamlit UI, training loop, evaluation visuals, and Grad-CAM
- `yoga_dataset.zip` — ready-to-use sample dataset (8 classes, 80/20 split)
- `requirements.txt` — runtime dependencies
- `verify_pytorch_conversion.py` — optional environment sanity check
- `AsanaAI_Final_Plan.md` — background plan and rationale

## Quickstart
1. **Install dependencies**
   ```bash
   cd AsanaAI
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. **Launch the app**
   ```bash
   streamlit run app.py
   ```
3. **Load data**
   - Use the bundled `yoga_dataset.zip`, or supply your own ZIP with structure:
     ```
     yoga_dataset/
       train/<class_name>/*.jpg|png
       val/<class_name>/*.jpg|png
     ```
   - The bundled dataset covers eight poses (downward_dog, warrior, tree, cobra, plank, triangle, child_pose, seated_forward_bend) with 96 train and 24 val images per class.
4. **Train**
   - Click **“Begin Training”** in the UI. Training history and early stopping are handled automatically.
5. **Evaluate & explain**
   - Review accuracy/loss curves, confusion matrix, classification report, and ROC curves.
   - Upload a pose image to see predictions with Grad-CAM heatmaps and overlaid explanations.

## Optional: quick environment verification
Run the smoke test to confirm PyTorch, data pipeline, and model wiring:
```bash
python verify_pytorch_conversion.py
```

## Notes
- GPU is recommended for faster training; CPU mode also works (slower).
- The app does not bundle pre-trained weights; training occurs after you upload a dataset ZIP.

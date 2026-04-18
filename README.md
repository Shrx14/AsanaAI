# AsanaAI - Explainable Yoga Pose Recognition

AsanaAI is a Streamlit application that classifies yoga poses from images (and live webcam) using a PyTorch MobileNetV2 model, explains predictions with Grad-CAM overlays, and gives pose guidance cards.

## What the app does

- Loads an existing trained checkpoint automatically (if available)
- Uploads a ZIP dataset (`train/` and `val/`) for training or retraining
- Trains a transfer learning model with two-stage fine-tuning and early stopping
- Shows learning curves and full evaluation metrics
- Predicts a pose from a single uploaded image with:
  - Top-3 probabilities with optional Test-Time Augmentation (TTA)
  - Full class probability chart (16 bars)
  - Grad-CAM heatmap and overlay
  - Pose guidance card (Sanskrit name, benefits, cues, difficulty)
- **Live Webcam Mode** — captures frames from your webcam, runs inference, and shows EMA-smoothed predictions with a live Grad-CAM overlay

## Repository structure

- `app.py` — Streamlit app, model pipeline, Grad-CAM, webcam inference, and UI
- `requirements.txt` — Python dependencies
- `verify_pytorch_conversion.py` — Optional environment/model sanity check
- `yoga_dataset.zip` — Sample curated dataset (16 classes)
- `AsanaAI_Final_Plan.md` — Final implementation plan and execution mapping
- `Project_Report.md` — Project report for Practicals 4 and 5

## Tech stack

- Python 3.10+
- Streamlit ≥ 1.35.0
- PyTorch + Torchvision (MobileNetV2)
- NumPy, Pandas, Matplotlib, Seaborn
- OpenCV
- scikit-learn

## Setup

1. Create and activate a virtual environment.

Windows PowerShell:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:
```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies.
```bash
pip install -r requirements.txt
```

3. Launch the app.
```bash
streamlit run app.py
```

## Dataset format

Upload a ZIP that contains a root folder with this structure:

```text
yoga_dataset/
  train/
    class_name_1/*.jpg|jpeg|png
    class_name_2/*.jpg|jpeg|png
    ...
  val/
    class_name_1/*.jpg|jpeg|png
    class_name_2/*.jpg|jpeg|png
    ...
```

The bundled dataset uses 16 classes:

| Class | Sanskrit Name |
|---|---|
| Camel Pose | Ustrasana |
| Crane Pose | Bakasana |
| Eight-Angle Pose | Astavakrasana |
| Full Boat Pose | Navasana |
| King Dancer Pose | Natarajasana |
| Plow Pose | Halasana |
| Sitting Half Spinal Twist | Ardha Matsyendrasana |
| child_pose | Balasana |
| cobra | Bhujangasana |
| cow pose | Bitilasana |
| downward_dog | Adho Mukha Svanasana |
| plank | Phalakasana |
| seated_forward_bend | Paschimottanasana |
| tree | Vrikshasana |
| triangle | Trikonasana |
| warrior | Virabhadrasana I |

## Runtime behavior

- If `asanai_saved/model_weights.pt`, `asanai_saved/training_history.json`, and `asanai_saved/class_names.json` exist, the app auto-loads the model.
- The **Predict a Pose** and **Live Webcam** sections are available immediately when a model is loaded.
- Evaluation charts require a validation loader — upload the dataset ZIP when using a loaded checkpoint.

## Live Webcam Mode

Click the camera button in the **Live Webcam** section to capture a frame. The model:
1. Runs inference with Test-Time Augmentation across 4 augmented versions of the frame.
2. Applies Exponential Moving Average (EMA, α = 0.35) smoothing across consecutive frames for stable predictions.
3. Shows a low-confidence warning (< 45%) instead of a class name when the model is uncertain.
4. Displays a live Grad-CAM overlay highlighting which body regions drove the prediction.

Click **Reset smoothing** to clear the EMA history when switching poses.

## Saved artifacts

Training writes these files under `asanai_saved/`:

- `model_weights.pt`
- `training_history.json`
- `class_names.json`

## Training defaults

- Image size: 160 × 160
- Batch size: 12
- Epochs: 24
- Early stopping patience: 6
- Optimizer: AdamW (two-stage — head training then partial backbone fine-tuning from epoch 10)
- Learning rates: 5e-4 (head), 2e-5 (fine-tune)
- Label smoothing: 0.05
- Weight decay: 1e-4
- MixUp: alpha 0.2 (first 10 epochs only)
- Class weighting: auto-enabled when imbalance ratio ≥ 1.5
- Post-hoc temperature calibration via grid search on validation logits

## Accuracy features

| Feature | What it does |
|---|---|
| Test-Time Augmentation (TTA) | Averages predictions across 4 augmented versions of the input |
| Temperature calibration | Post-hoc confidence scaling fitted on validation set |
| EMA webcam smoothing | Reduces per-frame flicker in live mode |
| Confidence threshold | Shows uncertainty warning below 45% confidence |

## Optional verification

```bash
python verify_pytorch_conversion.py
```

## Disclaimer

AsanaAI is an educational aid for beginner yoga practitioners. It does not replace certified yoga instruction.

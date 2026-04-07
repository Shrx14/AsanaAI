# AsanaAI - Explainable Yoga Pose Recognition

AsanaAI is a Streamlit application that classifies yoga poses from images using a PyTorch MobileNetV2 model and explains predictions with Grad-CAM overlays.

## What the app does

- Loads an existing trained checkpoint automatically (if available)
- Uploads a ZIP dataset (`train/` and `val/`) for training/retraining
- Trains a transfer learning model with early stopping
- Shows learning curves and full evaluation metrics
- Predicts a pose from a single image with:
  - Top-3 probabilities
  - Full class probability chart
  - Grad-CAM heatmap and overlay
  - Pose guidance (Sanskrit name, benefits, cues, difficulty)

## Repository structure

- `app.py` - Streamlit app, model pipeline, Grad-CAM, and UI
- `requirements.txt` - Python dependencies
- `verify_pytorch_conversion.py` - Optional environment/model sanity check
- `yoga_dataset.zip` - Sample curated dataset (8 classes)
- `AsanaAI_Final_Plan.md` - Final implementation plan and execution mapping
- `Project_Report.md` - Project report for Practicals 4 and 5

## Tech stack

- Python 3.10+
- Streamlit
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

The bundled dataset uses 8 classes:

- downward_dog
- warrior
- tree
- cobra
- plank
- triangle
- child_pose
- seated_forward_bend

## Runtime behavior

- If `asanai_saved/model_weights.pt`, `asanai_saved/training_history.json`, and `asanai_saved/class_names.json` exist, the app auto-loads the model.
- Prediction section is available immediately when a model is loaded.
- Evaluation charts require a validation loader, so upload the dataset ZIP when using a loaded checkpoint.

## Saved artifacts

Training writes these files under `asanai_saved/`:

- `model_weights.pt`
- `training_history.json`
- `class_names.json`

## Training defaults (from app constants)

- Image size: 160 x 160
- Batch size: 12
- Epochs: 24
- Early stopping patience: 6
- Optimizer: AdamW (two-stage: head training then partial backbone fine-tuning)
- Learning rates: 5e-4 (head), 2e-5 (fine-tune)
- Label smoothing: 0.05
- Weight decay: 1e-4
- MixUp: alpha 0.2 (first 10 epochs)

## Optional verification

Run the helper script:

```bash
python verify_pytorch_conversion.py
```

## Disclaimer

AsanaAI is an educational aid for beginner yoga practitioners. It does not replace certified yoga instruction.

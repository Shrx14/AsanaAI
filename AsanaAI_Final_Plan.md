# AsanaAI - Final Project Plan (Detailed)
## Explainable Deep Learning System for Yoga Pose Recognition

Course: CELBC608 - Data Science Laboratory (Semester VI)
Institution: Fr. C. Rodrigues Institute of Technology, Vashi
Practicals Covered: 4 and 5

---

## Executive Summary

This project builds an end-to-end, explainable image-classification system for yoga poses.
The implementation in app.py is based on Streamlit + PyTorch MobileNetV2 + Grad-CAM,
with model checkpoint persistence and full evaluation reporting.

Core goals:

- classify 8 yoga poses from a single image
- explain model decisions visually using Grad-CAM
- provide metrics suitable for practical evaluation and viva
- keep workflow reproducible through saved checkpoints

---

## Complete Changelog (Consolidated)

The following refinements are retained from iterative review and mapped to the final PyTorch app:

| # | Refinement | Status in current app.py |
|---|---|---|
| 1 | Transfer-learning workflow for small dataset | Implemented |
| 2 | Robust Grad-CAM target layer selection | Implemented (`get_last_conv_layer`) |
| 3 | Session-state guards to prevent accidental retraining | Implemented |
| 4 | ROC curves included as part of evaluation | Implemented |
| 5 | No augmentation in validation pipeline | Implemented |
| 6 | Deterministic class ordering from folder structure | Implemented |
| 7 | Physical train/val split from dataset folders | Implemented |
| 8 | ZIP cleanup via temp directory replacement | Implemented |
| 9 | Conditional rendering instead of forced rerun logic | Implemented |
| 10 | Readable confusion matrix ticks/labels | Implemented |
| 11 | Single cached inference pass for eval views | Implemented (`@st.cache_data`) |
| 12 | Validation loader is non-shuffled for metrics integrity | Implemented |
| 13 | Grad-CAM normalization guard `+1e-8` | Implemented |
| 14 | Dataset re-upload resets dataset-dependent state safely | Implemented |
| 15 | Explicit, trackable epoch logs in UI | Implemented |
| 16 | Lightweight backbone + custom head architecture | Implemented |
| 17 | Epoch count set to 10 with early stopping | Implemented |
| 18 | Best-weights restoration via deep-copied state | Implemented |

---

## Project Identity

Title:
AsanaAI: Explainable Yoga Pose Recognition Using Transfer Learning and Grad-CAM

Problem Statement:
Beginners often practice yoga without direct instructor supervision. The system predicts a pose
from an uploaded image and highlights key visual regions used by the model to support interpretability.

Academic Disclaimer:
This is an educational AI-assisted pose classification and visual feedback tool. It does not replace
certified yoga instruction.

---

## Final Tech Stack (Actual Implementation)

- Python 3.10+
- Streamlit
- PyTorch (`torch`, `torchvision`)
- MobileNetV2 pretrained backbone
- scikit-learn (confusion matrix, classification report, ROC/AUC)
- NumPy, Pandas, Matplotlib, Seaborn
- OpenCV and Pillow

---

## Final Project Structure

```text
AsanaAI-1/
	app.py
	requirements.txt
	verify_pytorch_conversion.py
	yoga_dataset.zip
	README.md
	AsanaAI_Final_Plan.md
	Project_Report.md
	asanai_saved/
		model_weights.pt
		training_history.json
		class_names.json
```

Dataset folder expected inside uploaded ZIP:

```text
yoga_dataset/
	train/
		downward_dog/
		warrior/
		tree/
		cobra/
		plank/
		triangle/
		child_pose/
		seated_forward_bend/
	val/
		downward_dog/
		warrior/
		tree/
		cobra/
		plank/
		triangle/
		child_pose/
		seated_forward_bend/
```

Target curation count typically used: 96 train + 24 val per class.

---

## Phase 0 - Pre-Lab Preparation (Night Before)

1. Download dataset from Kaggle.
2. Select the final 8 classes.
3. Create deterministic 80/20 split into physical `train` and `val` folders.
4. Zip folder as `yoga_dataset.zip` and test unzip once.
5. Install dependencies from `requirements.txt`.
6. Run optional verification script: `python verify_pytorch_conversion.py`.
7. Keep dataset ZIP ready for lab machine.

---

## Practical 4 - Design and Implementation Plan

### P4-1 UI shell and sectioning

Deliverables:

- Hero section + status bar
- Expanders for Dataset, Training, Training Results, Evaluation
- Prediction section visible when model is available
- No forced rerun mechanism

### P4-2 Dataset upload and validation

Deliverables:

- ZIP upload handling
- extraction to temporary path
- root detection containing `train` and `val`
- error messaging for invalid ZIP structure

### P4-3 Dataset summary and augmentation preview

Deliverables:

- class-wise image counts (train/val)
- class distribution bar chart
- sample images grid
- six random augmentation previews from train transforms

### P4-4 Model and training loop

Backbone:

- `models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)`
- feature extractor frozen

Head:

- `Linear -> ReLU -> Dropout(0.3) -> Linear(num_classes)`

Training defaults:

- image size: 128 x 128
- batch size: 16
- max epochs: 10
- patience: 3
- optimizer: Adam (lr=1e-3)
- loss: CrossEntropyLoss

Checkpoint behavior:

- save best restored model + history + class names under `asanai_saved/`
- auto-load checkpoint on next app launch if all files exist

---

## Practical 5 - Evaluation and Explainability Plan

### P5-1 Evaluation metrics panel

Metrics and visuals:

- overall validation accuracy
- confusion matrix
- classification report
- per-class accuracy chart
- one-vs-rest ROC curves

Implementation note:

- inference over validation loader is cached once and reused across tabs

### P5-2 Prediction and Grad-CAM section

User output components:

- predicted class and Sanskrit name
- confidence bar and top-3 predictions
- all-class probability chart
- original image + heatmap + overlay
- pose info card: benefits, form cues, difficulty

Grad-CAM strategy:

- choose last Conv2d in MobileNetV2 features
- use hooks to capture activations/gradients
- weighted sum -> ReLU -> normalize with `+1e-8`
- overlay heatmap with denormalized image

---

## Full Execution Flow (Current App)

1. App starts and sets session defaults.
2. CUDA is detected; warning shown for CPU mode.
3. Existing checkpoint is auto-loaded if present.
4. If model exists, prediction section is immediately usable.
5. User optionally uploads dataset ZIP for training/retraining.
6. App shows EDA summary and augmentation preview.
7. Training runs with progress bar and epoch logs.
8. Best model is restored and saved to disk.
9. Training results panel shows curve metrics.
10. Evaluation tabs show confusion matrix, report, per-class accuracy, ROC.
11. Prediction flow provides explainability and pose guidance.

---

## 8-Hour Lab Budget

| Task | Duration | Owner |
|---|---|---|
| Environment and dependency check | 10 min | Both |
| UI shell and section flow | 45 min | Student A |
| ZIP upload and dataset validation | 40 min | Student B |
| EDA and augmentation preview | 30 min | Student A |
| Model and training integration | 50 min | Student B |
| Training results and logging | 30 min | Student A |
| Evaluation tabs and metrics | 50 min | Student B |
| Grad-CAM + prediction UI | 45 min | Student A |
| Testing and bug fixes | 35 min | Both |
| Demo rehearsal and viva prep | 20 min | Both |
| Total | ~8 hours | Team |

---

## Validation Checklist

Pre-run:

- dependencies install correctly
- Streamlit app launches without import errors
- dataset ZIP has valid `train` and `val` structure

Post-training:

- checkpoint files are generated in `asanai_saved/`
- training history fields contain loss and accuracy arrays
- evaluation tabs render without recomputing full inference repeatedly

Prediction:

- model returns class, confidence, top-3
- Grad-CAM heatmap and overlay render correctly
- pose info card displays aligned class metadata

---

## Risk Register and Mitigations

1. Risk: Invalid ZIP structure
Mitigation: strict folder validation + user-facing error message.

2. Risk: Overfitting on small dataset
Mitigation: frozen pretrained backbone + dropout + early stopping.

3. Risk: Long CPU training time
Mitigation: CUDA detection and warning + lightweight backbone.

4. Risk: Misleading metrics from shuffled validation data
Mitigation: validation loader uses `shuffle=False`.

5. Risk: Gradient map instability
Mitigation: Grad-CAM normalization epsilon `+1e-8`.

---

## Viva Q and A (Updated to PyTorch)

Q1. Why MobileNetV2?
A. It balances accuracy and efficiency using depthwise separable convolutions and is suitable for constrained hardware.

Q2. Why freeze backbone layers?
A. With limited data, freezing pretrained layers improves stability and reduces overfitting risk.

Q3. Why keep validation transform clean?
A. Validation should represent real performance; augmentation is for training robustness only.

Q4. How is early stopping implemented?
A. Validation loss is monitored each epoch and best weights are restored after patience is exceeded.

Q5. How does Grad-CAM help?
A. It visualizes spatial regions contributing most to the predicted class, improving interpretability.

Q6. What is saved for reproducibility?
A. model weights, training history, and class-name order are saved and auto-loaded in later sessions.

Q7. Can the model evaluate without re-uploading dataset?
A. Prediction can run from checkpoint directly, but evaluation charts require validation loader from dataset upload.

---

## Report Outline (Detailed Submission Guidance)

1. Abstract
2. Problem statement and motivation
3. Dataset and preprocessing
4. Architecture and transfer-learning design
5. Training pipeline and checkpointing
6. Evaluation metrics and interpretation
7. Explainability (Grad-CAM)
8. Results and screenshots
9. Limitations
10. Conclusion and future scope

---

## Future Enhancements

- selective unfreezing for fine-tuning after baseline convergence
- stronger augmentation policy tuning by class performance
- model export flow for lightweight deployment
- optional pose-keypoint based correction feedback

---

This detailed plan is aligned to the current app.py implementation and is ready for practical execution, reporting, and viva discussion.

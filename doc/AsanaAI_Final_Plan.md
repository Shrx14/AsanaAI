# AsanaAI — Full Enhancement Implementation Plan

## Overview

This plan covers four enhancement areas:
1. **Live Webcam Yoga Pose Assistance**
2. **Improved UI / UX**
3. **Better Graph & Image Alignment**
4. **Increased Accuracy Logic**

All changes are drop-in additions to the existing `app.py`. Nothing in the core training loop needs to change.

---

## 1. Live Webcam Yoga Pose Assistance

### Approach — Two Tiers

**Tier 1 (Recommended for lab — zero extra packages):**
Use Streamlit's built-in `st.camera_input`. It captures a single still photo from the webcam. Pair it with an auto-refresh loop using `st.rerun()` with a small delay to simulate "live" mode. This works out of the box with your existing `requirements.txt`.

**Tier 2 (True streaming — one extra package):**
Use `streamlit-webrtc` for real-time frame-by-frame inference. Requires `pip install streamlit-webrtc aiortc`. Better for a live demo but more complex to set up.

The plan below implements **Tier 1** with an upgrade path to Tier 2.

---

### 1a. Session State Additions

Add these to the `_defaults` dict:

```python
"webcam_active":     False,
"webcam_result":     None,
"webcam_frame_count": 0,
"webcam_smoothed_probs": None,   # exponential moving average across frames
```

---

### 1b. Webcam Section in app.py

Place this section right after the existing **Predict a Pose** section (still inside `if st.session_state.trained:`):

```python
# ═══════════════════════════════════════════════════════════════
# SECTION 1B — LIVE WEBCAM MODE
# ═══════════════════════════════════════════════════════════════

st.markdown('<div class="section-header">📷 <span>Live Webcam Mode</span></div>',
            unsafe_allow_html=True)

st.markdown(
    "Click **Start Camera** and hold your yoga pose. The model will classify "
    "each captured frame. Results are smoothed across frames for stability."
)

SMOOTHING_ALPHA = 0.35    # EMA weight for new frame (lower = more smoothing)
CONFIDENCE_THRESHOLD = 0.45   # show "uncertain" below this

col_cam, col_result = st.columns([1, 1])

with col_cam:
    cam_frame = st.camera_input("📸 Hold your pose and capture", key="webcam_input")

if cam_frame is not None:
    with st.spinner("Analysing…"):
        result = predict_image(cam_frame)

    new_probs = result["all_probs"]

    # Exponential moving average smoothing across frames
    if st.session_state.webcam_smoothed_probs is None:
        st.session_state.webcam_smoothed_probs = new_probs
    else:
        prev = st.session_state.webcam_smoothed_probs
        st.session_state.webcam_smoothed_probs = (
            SMOOTHING_ALPHA * new_probs + (1.0 - SMOOTHING_ALPHA) * prev
        )

    smooth_probs  = st.session_state.webcam_smoothed_probs
    smooth_idx    = int(np.argmax(smooth_probs))
    smooth_conf   = float(smooth_probs[smooth_idx])
    smooth_cls    = st.session_state.class_names[smooth_idx]
    smooth_disp   = smooth_cls.replace("_", " ").title()
    smooth_skt    = sanskrit_names.get(smooth_cls, "")
    accent        = pose_info.get(smooth_cls, {}).get("color", "#667eea")

    st.session_state.webcam_frame_count += 1
    st.session_state.webcam_result = result

    with col_result:
        if smooth_conf < CONFIDENCE_THRESHOLD:
            st.warning(
                f"⚠️ Low confidence ({smooth_conf:.0%}) — "
                "try a clearer pose or better lighting."
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

            # Live top-3
            smooth_top3 = np.argsort(smooth_probs)[::-1][:3]
            for i in smooth_top3:
                lbl = st.session_state.class_names[i].replace("_"," ").title()
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

            # Grad-CAM from the raw (un-smoothed) frame
            c1, c2 = st.columns(2)
            c1.image(result["img_resized"], caption="Captured Frame", use_container_width=True)
            c2.image(result["overlay"],     caption="Grad-CAM Overlay", use_container_width=True)

        # Pose info card
        info = pose_info.get(smooth_cls, {})
        if info and smooth_conf >= CONFIDENCE_THRESHOLD:
            st.markdown(f"""
            <div class="info-card">
              <h4>📋 {smooth_disp}</h4>
              <p><strong>Benefits:</strong> {info['benefits']}</p>
              <p><strong>Form cues:</strong> {info['cues']}</p>
              <p><strong>Difficulty:</strong> {info['difficulty']}</p>
            </div>
            """, unsafe_allow_html=True)

    # Reset smoothing button
    if st.button("🔄 Reset smoothing", key="reset_smooth"):
        st.session_state.webcam_smoothed_probs = None
        st.session_state.webcam_frame_count = 0
        st.rerun()

st.markdown("---")
```

---

### 1c. Add to `requirements.txt`

No new packages needed for Tier 1. For Tier 2 streaming, add:

```
streamlit-webrtc>=0.47.0
aiortc>=1.6.0
```

---

### 1d. Tier 2 Upgrade (streamlit-webrtc)

If you want real-time inference without clicking, replace the `st.camera_input` block with this WebRTC callback pattern. Add after imports:

```python
# Optional real-time mode
try:
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
    import av
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False


class YogaVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.result_text = ""

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        tensor = PRED_TRANSFORM(pil_img).unsqueeze(0).to(DEVICE)

        model = st.session_state.model
        if model is not None:
            with torch.no_grad():
                out = model(tensor)
                probs = torch.softmax(out / get_temperature(), 1)[0].cpu().numpy()
            cls_idx = int(np.argmax(probs))
            cls_name = st.session_state.class_names[cls_idx]
            conf = float(probs[cls_idx])
            label = f"{cls_name.replace('_',' ').title()}  {conf:.0%}"
            cv2.putText(img, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 255, 120), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")
```

Then in the UI section:
```python
if WEBRTC_AVAILABLE:
    webrtc_streamer(key="yoga_stream", video_processor_factory=YogaVideoProcessor)
```

---

## 2. Improved UI / UX

### 2a. Sidebar Navigation

Add this right after `st.set_page_config(...)`:

```python
with st.sidebar:
    st.markdown("## 🧘 AsanaAI")
    st.markdown("---")
    nav = st.radio(
        "Navigate",
        ["🔮 Predict (Image)", "📷 Live Webcam", "📁 Dataset & Train",
         "📈 Results", "📊 Evaluation"],
        index=0,
    )
    st.markdown("---")
    if st.session_state.trained:
        h = st.session_state.history
        st.metric("Best Val Acc", f"{h.get('best_val_accuracy', 0):.1%}")
        st.metric("Classes", len(st.session_state.class_names))
        st.metric("Epochs Run", len(h.get("accuracy", [])))
    st.markdown("---")
    st.caption("CELBC608 — DSL Sem VI\nMobileNetV2 + Grad-CAM")
```

Use `nav` to conditionally show/hide sections:
```python
show_predict = nav == "🔮 Predict (Image)"
show_webcam  = nav == "📷 Live Webcam"
show_dataset = nav == "📁 Dataset & Train"
# etc — wrap each section in: if show_predict: ...
```

---

### 2b. Confidence Threshold Indicator

In `predict_image()` output, add a colour-coded ring:

```python
CONFIDENCE_COLORS = {
    "high":   ("#10b981", "High confidence ✅"),
    "medium": ("#f59e0b", "Medium confidence ⚠️"),
    "low":    ("#ef4444", "Low confidence ❌ — try a clearer image"),
}

def confidence_band(conf):
    if conf >= 0.75: return CONFIDENCE_COLORS["high"]
    if conf >= 0.45: return CONFIDENCE_COLORS["medium"]
    return CONFIDENCE_COLORS["low"]
```

Display alongside the confidence bar:
```python
band_color, band_label = confidence_band(conf)
st.markdown(f'<span style="color:{band_color};font-weight:600">{band_label}</span>',
            unsafe_allow_html=True)
```

---

### 2c. Compact Metric Row for Sidebar

The sidebar already shows key metrics via `st.metric`. For the main area, replace the raw HTML metric boxes with Streamlit's native:

```python
m1, m2, m3, m4 = st.columns(4)
m1.metric("Classes", len(class_names))
m2.metric("Train Images", total_train)
m3.metric("Val Images", total_val)
m4.metric("Train Ratio", f"{round(total_train/(total_train+total_val)*100)}%")
```

---

## 3. Better Graph & Image Alignment

### 3a. Consistent Figure Factory

Replace all scattered `figsize` values with a single helper. Add near the top of app.py after constants:

```python
def make_fig(w=10, h=4, ncols=1, nrows=1):
    """Create a figure with consistent dark background styling."""
    fig, axes = plt.subplots(nrows, ncols, figsize=(w, h))
    fig.patch.set_facecolor("#0a0a0f")
    for ax in (axes if hasattr(axes, '__iter__') else [axes]):
        ax.set_facecolor("#111827")
        ax.tick_params(colors="#94a3b8")
        ax.grid(alpha=0.1)
    return fig, axes
```

Use in all plot functions:
```python
# learning curves — was plt.subplots(1, 2, figsize=(11, 4))
fig, (ax1, ax2) = make_fig(w=11, h=4, ncols=2)
```

---

### 3b. Learning Curve — Epoch Marker for Best Epoch

```python
best_ep = history.get("best_epoch", 1)
ax1.axvline(x=best_ep, color="#10b981", ls=":", lw=1.5, label=f"Best (ep {best_ep})")
ax2.axvline(x=best_ep, color="#10b981", ls=":", lw=1.5)
```

---

### 3c. Confusion Matrix — Dynamic Figure Height

```python
def plot_confusion_matrix_fig(y_pred, y_true, class_names):
    n = len(class_names)
    fig_size = max(7, n * 0.55)          # scales with number of classes
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.8))
    ...
    plt.xticks(rotation=45, ha="right", fontsize=max(6, 9 - n//4), color="#94a3b8")
    plt.yticks(fontsize=max(6, 9 - n//4), color="#94a3b8")
```

---

### 3d. Image Display — Replace `width="stretch"` with `use_container_width`

Streamlit deprecated `width="stretch"`. Replace every instance:
```python
# OLD
col.image(img, caption="...", width="stretch")
# NEW
col.image(img, caption="...", use_container_width=True)
```

---

### 3e. Probability Bar Chart — Sort + Truncate Long Labels

```python
# Truncate long class names for readability
prob_df["Pose"] = prob_df["Pose"].apply(lambda x: x[:22] + "…" if len(x) > 22 else x)
```

---

## 4. Accuracy Improvements

### 4a. Test-Time Augmentation (TTA)

TTA runs multiple augmented versions of the same image through the model and averages the predictions. This typically improves accuracy by 1–3% for borderline cases with no additional training.

Add after `PRED_TRANSFORM`:
```python
TTA_TRANSFORMS = [
    PRED_TRANSFORM,   # original
    transforms.Compose([
        transforms.Resize(int(IMG_SIZE * 1.15)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(p=1.0),   # always flip
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]),
    transforms.Compose([
        transforms.Resize(int(IMG_SIZE * 1.25)),   # slightly larger crop
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
```

Modify `predict_image()` to use TTA:
```python
def predict_image(uploaded_img, use_tta=True):
    model       = st.session_state.model
    class_names = st.session_state.class_names

    img         = load_image_rgb(uploaded_img)
    img_resized = img.resize((IMG_SIZE, IMG_SIZE))

    model.eval()

    if use_tta:
        all_probs = []
        for tfm in TTA_TRANSFORMS:
            t = tfm(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                out = model(t)
                p   = torch.softmax(out / get_temperature(), 1)[0].cpu().numpy()
            all_probs.append(p)
        probs = np.mean(all_probs, axis=0)   # average across augmentations
        img_tensor = PRED_TRANSFORM(img).unsqueeze(0).to(DEVICE)
    else:
        img_tensor = PRED_TRANSFORM(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out   = model(img_tensor)
            probs = torch.softmax(out / get_temperature(), 1)[0].cpu().numpy()

    top3_idx   = np.argsort(probs)[::-1][:3]
    class_idx  = int(top3_idx[0])
    confidence = float(probs[class_idx])

    heatmap, overlay = generate_gradcam(model, img_tensor, class_idx)
    ...
```

Add a UI toggle in the predict section:
```python
use_tta = st.checkbox("Use Test-Time Augmentation (TTA) — slightly slower, more stable", value=True)
result = predict_image(pred_file, use_tta=use_tta)
```

---

### 4b. Entropy-Based Uncertainty Display

High entropy = model is uncertain between many classes. Show this alongside confidence:

```python
def prediction_entropy(probs):
    """Shannon entropy normalised to [0, 1]. Higher = more uncertain."""
    p = np.clip(probs, 1e-9, 1.0)
    H = -np.sum(p * np.log(p))
    H_max = np.log(len(probs))
    return float(H / H_max)
```

In the prediction display:
```python
entropy = prediction_entropy(result["all_probs"])
entropy_label = "Low" if entropy < 0.3 else "Medium" if entropy < 0.6 else "High"
st.caption(f"Prediction uncertainty: **{entropy_label}** ({entropy:.2f}) "
           "— lower is better")
```

---

### 4c. Multi-Crop Ensemble for Borderline Cases

When TTA is enabled and the top class confidence is below 0.6, add a 5-crop ensemble automatically (4 corners + center):

```python
def five_crop_ensemble(img, model, temperature):
    """Returns averaged probabilities from 5 crops."""
    five_crop_tfm = transforms.Compose([
        transforms.Resize(int(IMG_SIZE * 1.2)),
        transforms.FiveCrop(IMG_SIZE),
        transforms.Lambda(lambda crops: torch.stack([
            transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])(c) for c in crops
        ])),
    ])
    crops_tensor = five_crop_tfm(img).to(DEVICE)   # (5, C, H, W)
    with torch.no_grad():
        out   = model(crops_tensor)
        probs = torch.softmax(out / temperature, 1).cpu().numpy()
    return probs.mean(axis=0)
```

---

### 4d. Pose Similarity Hint

When confidence is below 0.6, show the top-2 classes and note their visual similarity:

```python
SIMILAR_POSES = {
    frozenset(["triangle", "warrior"]):    "Both involve wide leg stances.",
    frozenset(["cobra", "plank"]):          "Both are floor-based prone poses.",
    frozenset(["tree", "King Dancer Pose"]):"Both are single-leg balance poses.",
    frozenset(["child_pose", "seated_forward_bend"]): "Both involve forward folding.",
}

def get_similarity_hint(cls_a, cls_b):
    key = frozenset([cls_a, cls_b])
    return SIMILAR_POSES.get(key, "")
```

---

## 5. Additional Optimizations

### 5a. `use_container_width=True` (Streamlit deprecation fix)

All `st.image(..., width="stretch")` → `st.image(..., use_container_width=True)`

### 5b. Spinner on Grad-CAM

Grad-CAM involves a backward pass and can lag on CPU. Wrap it:
```python
with st.spinner("Generating Grad-CAM…"):
    heatmap, overlay = generate_gradcam(model, img_tensor, class_idx)
```

### 5c. Model Info Card in Sidebar

Show which classes the model knows when loaded from checkpoint:
```python
with st.sidebar.expander("Model classes"):
    for i, c in enumerate(st.session_state.class_names):
        st.write(f"{i+1}. {c.replace('_',' ').title()}")
```

### 5d. Download Grad-CAM Button

```python
from io import BytesIO
buf = BytesIO()
Image.fromarray(result["overlay"]).save(buf, format="PNG")
st.download_button("⬇ Download Grad-CAM overlay",
                   data=buf.getvalue(),
                   file_name=f"gradcam_{result['display_name']}.png",
                   mime="image/png")
```

### 5e. Clear Cache on Retrain

After `save_model(...)` in the training button callback:
```python
_get_predictions_cached.clear()   # force fresh eval after retraining
```

---

## Requirements Update

```
streamlit>=1.35.0          # was 1.28.0 — use_container_width fix
torch>=2.0.0
torchvision>=0.15.0
torchaudio>=2.0.0
numpy>=1.23.0
pandas>=1.5.0
matplotlib>=3.6.0
seaborn>=0.12.0
pillow>=9.0.0
opencv-python>=4.7.0
scikit-learn>=1.2.0
kagglehub>=0.2.0
# Optional for real-time webcam streaming:
# streamlit-webrtc>=0.47.0
# aiortc>=1.6.0
```

---

## Summary Table

| Feature | File Change | Effort |
|---|---|---|
| Webcam (`st.camera_input`) | app.py — new section | Low |
| EMA frame smoothing | app.py — session state | Low |
| Sidebar navigation | app.py — `st.sidebar` block | Low |
| Confidence band indicator | app.py — helper + display | Low |
| `use_container_width` fix | app.py — find & replace | Trivial |
| `make_fig()` factory | app.py — replace all plt.subplots | Medium |
| Best-epoch vertical line | app.py — `plot_learning_curves` | Trivial |
| TTA (4 augmentations) | app.py — `predict_image` | Medium |
| Entropy uncertainty | app.py — display block | Low |
| 5-crop ensemble fallback | app.py — `five_crop_ensemble` | Medium |
| Grad-CAM download button | app.py — after overlay display | Trivial |
| Cache clear on retrain | app.py — training callback | Trivial |
| Streamlit-webrtc (real-time) | app.py + requirements.txt | High |

# =====================================================================
# WASTEAI — Gradio App  (Responsive + Attractive + Full Calibration)
# DINOv2 ViT-B/14 · FAISS · Temperature Scaling · MC Dropout
# =====================================================================

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import faiss
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

# ── CONFIG ─────────────────────────────────────────────────────────────
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
FEAT_DIM    = 768
NUM_CLASSES = 9

FINAL_CLASSES = [
    "plastic", "paper", "metal", "clothes", "shoes",
    "biological", "white-glass", "cardboard", "battery"
]

RECYCLABLE = {"plastic", "paper", "metal", "white-glass", "cardboard", "battery"}

CLASS_ICONS = {
    "plastic":    "🧴", "paper":     "📄", "metal":     "🥫",
    "clothes":    "👕", "shoes":     "👟", "biological": "🌿",
    "white-glass":"🪟", "cardboard": "📦", "battery":   "🔋",
}

# ── PATHS ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FEATS_PATH       = os.path.join(BASE_DIR, "train_feats_dinov2_9cls.npy")
LABELS_PATH      = os.path.join(BASE_DIR, "train_labels_dinov2_9cls.npy")
BEST_MODEL_PATH  = os.path.join(BASE_DIR, "best_model_dinov2_9cls.pth")
TEMP_SCALER_PATH = os.path.join(BASE_DIR, "temp_scaler.pth")

# ── AUTO-DOWNLOAD FROM GOOGLE DRIVE IF FILES NOT PRESENT ───────────────
import requests

DRIVE_FILES = {
    FEATS_PATH:       "1E0Kb4E8rSMHmYofmuOSHcu58KCsx_Li2",
    LABELS_PATH:      "1Q36SYAd25vwvWrVi_ffxlEFb2DX-ArGp",
    BEST_MODEL_PATH:  "1uZ6vliXqnivv8mCZLfDf2wk0fOGG2lXh",
    TEMP_SCALER_PATH: "16CT2sOPxCjIgV1Iikd-A5OL3EeuOh1-i",
}

def download_from_drive(file_id, dest_path):
    print(f"⬇️  Downloading {os.path.basename(dest_path)}...")
    URL = "https://drive.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={"id": file_id}, stream=True)
    # Handle large file confirmation token
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            response = session.get(URL, params={"id": file_id, "confirm": value}, stream=True)
            break
    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)
    print(f"✅ {os.path.basename(dest_path)} ready")

for path, file_id in DRIVE_FILES.items():
    if not os.path.exists(path):
        download_from_drive(file_id, path)

# ── MODEL DEFINITIONS (must match training code exactly) ───────────────
class DINOv2Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        for p in self.backbone.parameters():
            p.requires_grad = False
    def forward(self, x):
        with torch.no_grad():
            return self.backbone(x)

class AttentionFusion(nn.Module):
    def __init__(self, dim=FEAT_DIM):
        super().__init__()
        self.scale = dim ** 0.5
    def forward(self, fq, fs):
        attn = torch.softmax((fq * fs) / self.scale, dim=1)
        return torch.cat([fq, attn * fs], dim=1)

class DropConnectLinear(nn.Linear):
    def __init__(self, in_f, out_f, p=0.2):
        super().__init__(in_f, out_f)
        self.p = p
    def forward(self, x):
        if self.training:
            mask   = torch.bernoulli(torch.ones_like(self.weight) * (1 - self.p))
            weight = self.weight * mask
        else:
            weight = self.weight
        return F.linear(x, weight, self.bias)

class Classifier(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, input_dim=FEAT_DIM * 2):
        super().__init__()
        dims, layers = [input_dim, 1024, 512, 256, 128], []
        for i in range(len(dims) - 1):
            layers += [DropConnectLinear(dims[i], dims[i+1], p=0.1),
                       nn.BatchNorm1d(dims[i+1]), nn.ReLU(inplace=True), nn.Dropout(0.2)]
        self.fc  = nn.Sequential(*layers)
        self.out = nn.Linear(128, num_classes)
    def forward(self, x):
        return self.out(self.fc(x))

class FastGarbageModel(nn.Module):
    def __init__(self, feat_dim=FEAT_DIM):
        super().__init__()
        self.attn = AttentionFusion(dim=feat_dim)
        self.cls  = Classifier(num_classes=NUM_CLASSES, input_dim=feat_dim * 2)
    def forward(self, fq, fs):
        return self.cls(self.attn(fq, fs))

class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.T = nn.Parameter(torch.ones(1) * 1.5)
    def forward(self, logits):
        return logits / self.T.clamp(min=0.05)

# ── TRANSFORM ───────────────────────────────────────────────────────────
val_tfms = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ── LOAD ALL MODELS ─────────────────────────────────────────────────────
print("⏳ Loading models…")
encoder = DINOv2Encoder().to(DEVICE).eval()

model = FastGarbageModel().to(DEVICE)
model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE, weights_only=True))
model.eval()

temp_scaler = TemperatureScaler().to(DEVICE)
temp_scaler.load_state_dict(torch.load(TEMP_SCALER_PATH, map_location=DEVICE, weights_only=True))
temp_scaler.eval()

train_feats      = np.load(FEATS_PATH).astype("float32")
train_labels_arr = np.load(LABELS_PATH)
faiss_index      = faiss.IndexFlatL2(FEAT_DIM)
faiss_index.add(train_feats)
print(f"✅ Ready  |  Device: {DEVICE}  |  FAISS: {faiss_index.ntotal} vectors  |  T = {temp_scaler.T.item():.4f}")

# ── HELPERS ─────────────────────────────────────────────────────────────
def enable_mc_dropout(m):
    if isinstance(m, nn.Dropout):
        m.train()

def mc_dropout_predict(img_tensor, passes=40):
    model.apply(enable_mc_dropout)
    with torch.no_grad():
        fq    = encoder(img_tensor)
        fq_np = fq.cpu().numpy().astype("float32")
        _, idx = faiss_index.search(fq_np, 1)
        fs    = torch.tensor(train_feats[idx[:, 0]], dtype=torch.float32).to(DEVICE)
        preds = [F.softmax(temp_scaler(model(fq, fs)), dim=1).cpu().numpy() for _ in range(passes)]
    model.eval()
    preds = np.stack(preds)
    return preds.mean(axis=0), preds.std(axis=0)

def compute_ece(probs, labels, n_bins=15):
    confidences = probs.max(axis=1)
    preds       = probs.argmax(axis=1)
    accs        = (preds == labels).astype(float)
    bins        = np.linspace(0, 1, n_bins + 1)
    ece         = 0.0
    for i in range(n_bins):
        mask = (confidences > bins[i]) & (confidences <= bins[i+1])
        if mask.sum() > 0:
            ece += (mask.sum() / len(labels)) * abs(accs[mask].mean() - confidences[mask].mean())
    return ece

# ── CALIBRATION RELIABILITY DIAGRAM ─────────────────────────────────────
def make_calibration_plot():
    model.eval()
    all_probs, all_labs = [], []
    batch = 256
    for start in range(0, min(len(train_feats), 2000), batch):
        fq_np = train_feats[start:start+batch]
        fq    = torch.tensor(fq_np).to(DEVICE)
        _, idx = faiss_index.search(fq_np, 1)
        fs    = torch.tensor(train_feats[idx[:, 0]], dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            probs = F.softmax(temp_scaler(model(fq, fs)), dim=1).cpu().numpy()
        all_probs.append(probs)
        all_labs.extend(train_labels_arr[start:start+batch].tolist())

    all_probs = np.vstack(all_probs)
    all_labs  = np.array(all_labs[:len(all_probs)])
    ece       = compute_ece(all_probs, all_labs)

    confidences = all_probs.max(axis=1)
    preds       = all_probs.argmax(axis=1)
    accs        = (preds == all_labs).astype(float)

    n_bins = 15
    bins   = np.linspace(0, 1, n_bins + 1)
    bin_acc, bin_conf, bin_cnt = [], [], []
    for i in range(n_bins):
        mask = (confidences > bins[i]) & (confidences <= bins[i+1])
        if mask.sum() > 0:
            bin_acc.append(accs[mask].mean())
            bin_conf.append(confidences[mask].mean())
            bin_cnt.append(int(mask.sum()))
        else:
            bin_acc.append(0)
            bin_conf.append((bins[i] + bins[i+1]) / 2)
            bin_cnt.append(0)

    bin_centers = [(bins[i] + bins[i+1]) / 2 for i in range(n_bins)]
    width       = 1.0 / n_bins * 0.82

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("#0d1117")

    ax = axes[0]
    ax.set_facecolor("#161b22")
    gap_colors = ["#f85149" if a < c else "#3fb950" for a, c in zip(bin_acc, bin_conf)]
    ax.bar(bin_centers, bin_acc,   width=width, color="#58a6ff", alpha=0.9, label="Accuracy",  zorder=3)
    ax.bar(bin_centers, bin_conf,  width=width, color=gap_colors, alpha=0.28, label="Gap",     zorder=2)
    ax.plot([0, 1], [0, 1], "w--", linewidth=1.6, alpha=0.55, label="Perfect calibration")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    ax.set_xlabel("Confidence", color="#8b949e", fontsize=11)
    ax.set_ylabel("Accuracy",   color="#8b949e", fontsize=11)
    ax.set_title(f"Reliability Diagram", color="#e6edf3", fontsize=12, fontweight="bold", pad=12)
    ax.tick_params(colors="#8b949e")
    for spine in ax.spines.values(): spine.set_edgecolor("#30363d")
    ax.legend(facecolor="#161b22", labelcolor="#e6edf3", fontsize=9, framealpha=0.85)
    ax.grid(axis="y", color="#30363d", linewidth=0.6, zorder=0)
    ece_color = "#3fb950" if ece < 0.03 else ("#d29922" if ece < 0.07 else "#f85149")
    ax.text(0.97, 0.04, f"ECE = {ece:.4f}", transform=ax.transAxes,
            ha="right", va="bottom", color=ece_color, fontsize=12, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#21262d", edgecolor="#444c56", linewidth=1.2))

    ax2 = axes[1]
    ax2.set_facecolor("#161b22")
    ax2.bar(bin_centers, bin_cnt, width=width, color="#bc8cff", alpha=0.85, zorder=3)
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("Confidence",    color="#8b949e", fontsize=11)
    ax2.set_ylabel("Sample Count",  color="#8b949e", fontsize=11)
    ax2.set_title("Confidence Distribution", color="#e6edf3", fontsize=12, fontweight="bold", pad=12)
    ax2.tick_params(colors="#8b949e")
    for spine in ax2.spines.values(): spine.set_edgecolor("#30363d")
    ax2.grid(axis="y", color="#30363d", linewidth=0.6, zorder=0)

    plt.tight_layout(pad=2.5)
    return fig, ece

# ── PROBABILITY CHART ───────────────────────────────────────────────────
def make_prob_chart(probs, mc_std_sq, pred_idx):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    colors          = ["#3fb950" if c in RECYCLABLE else "#f85149" for c in FINAL_CLASSES]
    colors[pred_idx] = "#f0e130"
    y_pos            = np.arange(NUM_CLASSES)

    ax.barh(y_pos, probs * 100, color=colors, alpha=0.88, height=0.62, zorder=3)
    ax.errorbar(probs * 100, y_pos,
                xerr=mc_std_sq * 100,
                fmt="none", color="white", alpha=0.5, capsize=4, linewidth=1.2, zorder=4)

    ax.set_xlim(0, 115)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{CLASS_ICONS.get(c,'●')} {c.capitalize()}" for c in FINAL_CLASSES],
                       color="#e6edf3", fontsize=10)
    ax.set_xlabel("Calibrated Probability (%)", color="#8b949e", fontsize=10)
    ax.set_title("Class Probabilities  (error bars = MC Dropout uncertainty)",
                 color="#e6edf3", fontsize=11, fontweight="bold", pad=10)
    ax.tick_params(axis="x", colors="#8b949e")
    for spine in ax.spines.values(): spine.set_edgecolor("#30363d")
    ax.grid(axis="x", color="#30363d", linewidth=0.5, zorder=0)

    for i, (bar_val, p) in enumerate(zip(y_pos, probs * 100)):
        if p > 0.5:
            ax.text(p + 1.5, bar_val, f"{p:.1f}%", va="center", color="#e6edf3", fontsize=9)

    ax.legend(handles=[
        mpatches.Patch(color="#3fb950", label="Recyclable"),
        mpatches.Patch(color="#f85149", label="Non-Recyclable"),
        mpatches.Patch(color="#f0e130", label="Predicted"),
    ], facecolor="#161b22", labelcolor="#e6edf3", fontsize=9, framealpha=0.9, loc="lower right")

    plt.tight_layout()
    return fig

# ── MAIN INFERENCE FUNCTION ─────────────────────────────────────────────
def classify_waste(image):
    if image is None:
        empty = plt.figure(figsize=(9, 4))
        plt.close()
        placeholder = "<div class='result-placeholder'>Upload an image and click Analyze</div>"
        return placeholder, empty, empty, "", ""

    img_tensor = val_tfms(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        fq     = encoder(img_tensor)
        fq_np  = fq.cpu().numpy().astype("float32")
        _, idx = faiss_index.search(fq_np, 1)
        fs     = torch.tensor(train_feats[idx[:, 0]], dtype=torch.float32).to(DEVICE)
        logits = temp_scaler(model(fq, fs))
        probs  = F.softmax(logits, dim=1).cpu().numpy()[0]

    mc_mean, mc_std = mc_dropout_predict(img_tensor, passes=40)
    mc_std_sq       = mc_std.squeeze()

    pred_idx    = int(probs.argmax())
    pred_class  = FINAL_CLASSES[pred_idx]
    confidence  = probs[pred_idx] * 100
    uncertainty = mc_std_sq[pred_idx]
    is_rec      = pred_class in RECYCLABLE

    rec_color  = "#3fb950" if is_rec else "#f85149"
    rec_label  = "♻️ Recyclable" if is_rec else "🚫 Non-Recyclable"
    conf_color = "#3fb950" if confidence > 75 else ("#d29922" if confidence > 50 else "#f85149")
    if uncertainty < 0.02:   unc_level, unc_color = "Very Low",  "#3fb950"
    elif uncertainty < 0.05: unc_level, unc_color = "Low",       "#58a6ff"
    elif uncertainty < 0.10: unc_level, unc_color = "Moderate",  "#d29922"
    else:                     unc_level, unc_color = "High",      "#f85149"

    pred_html = f"""
    <div class="pred-card">
      <div class="pred-top">
        <span class="pred-icon">{CLASS_ICONS.get(pred_class, '●')}</span>
        <div class="pred-info">
          <div class="pred-class">{pred_class.upper()}</div>
          <div class="pred-rec" style="color:{rec_color}">{rec_label}</div>
        </div>
        <div class="pred-conf-big" style="color:{conf_color}">{confidence:.1f}%</div>
      </div>
      <div class="conf-label">Confidence</div>
      <div class="conf-bar-track">
        <div class="conf-bar-fill" style="width:{min(confidence,100)}%;background:{conf_color}"></div>
      </div>
      <div class="meta-row">
        <span class="meta-label">MC Uncertainty</span>
        <span class="meta-value" style="color:{unc_color}">±{uncertainty:.4f} &nbsp;·&nbsp; {unc_level}</span>
      </div>
      <div class="meta-row">
        <span class="meta-label">Temperature (T)</span>
        <span class="meta-value" style="color:#bc8cff">{temp_scaler.T.item():.4f}</span>
      </div>
    </div>"""

    top3_idx = np.argsort(probs)[::-1][:3]
    medals   = ["🥇", "🥈", "🥉"]
    rows     = ""
    for rank, i in enumerate(top3_idx):
        rc   = "#3fb950" if FINAL_CLASSES[i] in RECYCLABLE else "#f85149"
        rows += f"""<tr>
          <td>{medals[rank]}</td>
          <td>{CLASS_ICONS.get(FINAL_CLASSES[i],'●')} <b>{FINAL_CLASSES[i].capitalize()}</b></td>
          <td style="color:{rc}">{'Recyclable' if FINAL_CLASSES[i] in RECYCLABLE else 'Non-Recyclable'}</td>
          <td><b>{probs[i]*100:.2f}%</b></td>
          <td style="color:#8b949e">±{mc_std_sq[i]:.4f}</td>
        </tr>"""

    top3_html = f"""
    <div class="top3-wrap">
      <div class="section-title">🏅 Top-3 Predictions</div>
      <table class="top3-table">
        <thead><tr><th></th><th>Class</th><th>Type</th><th>Confidence</th><th>Uncertainty</th></tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </div>"""

    prob_fig       = make_prob_chart(probs, mc_std_sq, pred_idx)
    cal_fig, ece   = make_calibration_plot()

    ece_color = "#3fb950" if ece < 0.03 else ("#d29922" if ece < 0.07 else "#f85149")
    ece_grade = "Excellent 🟢" if ece < 0.03 else ("Good 🟡" if ece < 0.07 else "Needs work 🔴")
    t_val     = temp_scaler.T.item()
    t_interp  = "Sharpening ↑" if t_val < 1 else "Smoothing ↓"

    cal_html = f"""
    <div class="cal-wrap">
      <div class="section-title">📐 Calibration Metrics</div>
      <div class="cal-grid">
        <div class="cal-card">
          <div class="cal-val" style="color:{ece_color}">{ece:.4f}</div>
          <div class="cal-name">ECE</div>
          <div class="cal-desc">Expected Calibration Error<br>(lower is better)</div>
          <div class="cal-grade" style="color:{ece_color}">{ece_grade}</div>
        </div>
        <div class="cal-card">
          <div class="cal-val" style="color:#bc8cff">{t_val:.4f}</div>
          <div class="cal-name">Temperature T</div>
          <div class="cal-desc">LBFGS-fitted on val set</div>
          <div class="cal-grade" style="color:#8b949e">{t_interp}</div>
        </div>
        <div class="cal-card">
          <div class="cal-val" style="color:#58a6ff">40</div>
          <div class="cal-name">MC Passes</div>
          <div class="cal-desc">Stochastic forward passes<br>with Dropout active</div>
          <div class="cal-grade" style="color:#8b949e">±{mc_std_sq.mean():.4f} avg</div>
        </div>
        <div class="cal-card">
          <div class="cal-val" style="color:{conf_color}">{confidence:.1f}%</div>
          <div class="cal-name">This Prediction</div>
          <div class="cal-desc">Calibrated softmax<br>confidence score</div>
          <div class="cal-grade" style="color:{unc_color}">{unc_level} uncertainty</div>
        </div>
      </div>
    </div>"""

    return pred_html, prob_fig, cal_fig, top3_html, cal_html


# ── CSS ─────────────────────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
  --bg:     #0d1117;
  --surf:   #161b22;
  --surf2:  #21262d;
  --border: #30363d;
  --green:  #3fb950;
  --red:    #f85149;
  --blue:   #58a6ff;
  --purple: #bc8cff;
  --yellow: #f0e130;
  --text:   #e6edf3;
  --muted:  #8b949e;
  --r:      14px;
  --font:   'Outfit', sans-serif;
  --mono:   'JetBrains Mono', monospace;
}
*, *::before, *::after { box-sizing: border-box; }

body, .gradio-container, .main, .wrap {
  background: var(--bg) !important;
  color: var(--text) !important;
  font-family: var(--font) !important;
}

.app-header {
  text-align: center;
  padding: 2.8rem 1rem 1.6rem;
  background: radial-gradient(ellipse 80% 50% at 50% 0%, rgba(63,185,80,0.07) 0%, transparent 70%);
}
.header-badge {
  display: inline-block;
  background: var(--surf2);
  border: 1px solid var(--border);
  border-radius: 20px;
  padding: 4px 16px;
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 1.5px;
  text-transform: uppercase;
  color: var(--green);
  margin-bottom: 14px;
}
.header-title {
  font-size: clamp(2rem, 5vw, 3.2rem);
  font-weight: 800;
  letter-spacing: -1.5px;
  background: linear-gradient(135deg, #3fb950 0%, #58a6ff 55%, #bc8cff 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  line-height: 1.1;
  margin-bottom: 10px;
}
.header-sub {
  color: var(--muted);
  font-size: clamp(0.9rem, 2vw, 1rem);
  max-width: 580px;
  margin: 0 auto 16px;
  line-height: 1.6;
}
.header-pills { display: flex; flex-wrap: wrap; gap: 8px; justify-content: center; }
.pill {
  background: var(--surf2);
  border: 1px solid var(--border);
  border-radius: 20px;
  padding: 4px 12px;
  font-size: 0.73rem;
  color: var(--muted);
  font-weight: 500;
}

.sec-hdr {
  font-size: 0.75rem;
  text-transform: uppercase;
  letter-spacing: 1.2px;
  color: var(--muted);
  font-weight: 700;
  margin: 18px 0 8px;
}
.divider { border: none; border-top: 1px solid var(--border); margin: 24px 0; }

.upload-zone { border: 2px dashed var(--border) !important; border-radius: 10px !important; background: var(--bg) !important; transition: border-color 0.25s; min-height: 270px; }
.upload-zone:hover { border-color: var(--green) !important; }

.analyze-btn {
  width: 100%; height: 52px !important;
  background: linear-gradient(135deg, #238636, #2ea043) !important;
  border: none !important; border-radius: 10px !important;
  color: #fff !important; font-family: var(--font) !important;
  font-size: 1.05rem !important; font-weight: 700 !important;
  box-shadow: 0 4px 16px rgba(46,160,67,0.28) !important;
  transition: opacity 0.2s, transform 0.15s !important;
}
.analyze-btn:hover { opacity: 0.9 !important; transform: translateY(-2px) !important; }
.analyze-btn:active { transform: translateY(0) !important; }

.result-placeholder { color: var(--muted); font-size: 0.95rem; padding: 1.8rem; text-align: center; }
.pred-card { background: var(--surf); border: 1px solid var(--border); border-radius: var(--r); padding: 1.4rem 1.6rem; }
.pred-top { display: flex; align-items: center; gap: 14px; margin-bottom: 16px; flex-wrap: wrap; }
.pred-icon { font-size: 2.8rem; line-height: 1; }
.pred-info { flex: 1; min-width: 120px; }
.pred-class { font-size: clamp(1.3rem, 3vw, 1.7rem); font-weight: 800; letter-spacing: -0.5px; }
.pred-rec { font-size: 0.9rem; font-weight: 600; margin-top: 3px; }
.pred-conf-big { font-size: clamp(1.8rem, 4vw, 2.4rem); font-weight: 800; font-family: var(--mono); letter-spacing: -1px; }
.conf-label { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 1px; color: var(--muted); font-weight: 600; margin-bottom: 6px; }
.conf-bar-track { background: var(--surf2); border-radius: 6px; height: 10px; overflow: hidden; margin-bottom: 14px; border: 1px solid var(--border); }
.conf-bar-fill { height: 100%; border-radius: 6px; transition: width 0.6s cubic-bezier(.4,0,.2,1); }
.meta-row { display: flex; justify-content: space-between; align-items: center; padding: 8px 0; border-top: 1px solid var(--border); flex-wrap: wrap; gap: 4px; }
.meta-label { font-size: 0.82rem; color: var(--muted); font-weight: 500; }
.meta-value { font-family: var(--mono); font-size: 0.85rem; font-weight: 600; }

.top3-wrap { background: var(--surf); border: 1px solid var(--border); border-radius: var(--r); padding: 1.2rem 1.4rem; margin-top: 14px; }
.section-title { font-size: 0.78rem; text-transform: uppercase; letter-spacing: 1.2px; color: var(--muted); font-weight: 700; margin-bottom: 12px; }
.top3-table { width: 100%; border-collapse: collapse; font-size: 0.86rem; }
.top3-table th { text-align: left; padding: 6px 10px; color: var(--muted); font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.8px; border-bottom: 1px solid var(--border); }
.top3-table td { padding: 9px 10px; border-bottom: 1px solid var(--surf2); }
.top3-table tr:last-child td { border-bottom: none; }

.cal-wrap { background: var(--surf); border: 1px solid var(--border); border-radius: var(--r); padding: 1.2rem 1.4rem; }
.cal-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 12px; margin-top: 10px; }
.cal-card { background: var(--surf2); border: 1px solid var(--border); border-radius: 10px; padding: 14px 10px; text-align: center; }
.cal-val  { font-family: var(--mono); font-size: 1.5rem; font-weight: 700; letter-spacing: -0.5px; }
.cal-name { font-size: 0.8rem; font-weight: 700; color: var(--text); margin: 4px 0 2px; }
.cal-desc { font-size: 0.7rem; color: var(--muted); line-height: 1.4; }
.cal-grade { font-size: 0.76rem; font-weight: 600; margin-top: 6px; }

.app-footer { text-align: center; color: var(--muted); font-size: 0.76rem; padding: 2rem 0 2.5rem; border-top: 1px solid var(--border); margin-top: 30px; line-height: 1.8; }

.block, .form { background: transparent !important; border: none !important; box-shadow: none !important; padding: 0 !important; }
label > span, .label-wrap span { color: var(--muted) !important; font-size: 0.72rem !important; text-transform: uppercase !important; letter-spacing: 0.8px !important; font-weight: 600 !important; }

@media (max-width: 640px) {
  .pred-top { flex-direction: column; align-items: flex-start; }
  .pred-conf-big { align-self: flex-end; }
  .cal-grid { grid-template-columns: 1fr 1fr; }
  .top3-table { font-size: 0.78rem; }
}
"""

# ── GRADIO LAYOUT ────────────────────────────────────────────────────────
with gr.Blocks(css=CSS, theme=gr.themes.Base(), title="WasteAI Classifier") as demo:

    gr.HTML("""
    <div class="app-header">
      <div class="header-badge">AI · DINOv2 · 9 Classes · Calibrated</div>
      <div class="header-title">♻️ WasteAI Classifier</div>
      <div class="header-sub">
        Upload any waste photo — get instant classification, calibrated confidence,
        MC Dropout uncertainty, and full calibration diagnostics.
      </div>
      <div class="header-pills">
        <span class="pill">DINOv2 ViT-B/14</span>
        <span class="pill">FAISS Retrieval</span>
        <span class="pill">Temperature Scaling</span>
        <span class="pill">MC Dropout · 40 passes</span>
        <span class="pill">ECE Calibration</span>
        <span class="pill">Reliability Diagram</span>
      </div>
    </div>
    """)

    with gr.Row(equal_height=False):
        with gr.Column(scale=1, min_width=300):
            gr.HTML('<div class="sec-hdr">📤 Upload Waste Image</div>')
            image_input = gr.Image(type="pil", label="", show_label=False,
                                   elem_classes=["upload-zone"], height=270)
            analyze_btn = gr.Button("🔍  Analyze Waste",
                                    elem_classes=["analyze-btn"], variant="primary")
            gr.HTML('<div style="color:var(--muted);font-size:0.74rem;text-align:center;margin-top:8px;">JPG · PNG · WEBP · Max 10 MB</div>')

        with gr.Column(scale=1, min_width=340):
            gr.HTML('<div class="sec-hdr">🎯 Prediction Result</div>')
            pred_out = gr.HTML('<div class="result-placeholder">Upload an image and click Analyze</div>')
            top3_out = gr.HTML()

    gr.HTML('<hr class="divider">')

    with gr.Row(equal_height=False):
        with gr.Column(scale=3, min_width=380):
            gr.HTML('<div class="sec-hdr">📊 Class Probability Distribution</div>')
            prob_chart = gr.Plot(label="", show_label=False)
        with gr.Column(scale=2, min_width=280):
            gr.HTML('<div class="sec-hdr">📐 Calibration Metrics</div>')
            cal_metrics_out = gr.HTML()

    gr.HTML('<hr class="divider">')

    gr.HTML('<div class="sec-hdr">🔬 Reliability Diagram &amp; Confidence Distribution</div>')
    cal_chart = gr.Plot(label="", show_label=False)

    gr.HTML("""
    <div class="app-footer">
      WasteAI &nbsp;·&nbsp; DINOv2 ViT-B/14 &nbsp;·&nbsp; FAISS &nbsp;·&nbsp;
      Temperature Scaling &nbsp;·&nbsp; MC Dropout &nbsp;·&nbsp; 9 Waste Classes<br>
      Label Smoothing · DropConnect · Weighted Sampling · AttentionFusion
    </div>
    """)

    outputs = [pred_out, prob_chart, cal_chart, top3_out, cal_metrics_out]
    analyze_btn.click(fn=classify_waste, inputs=[image_input], outputs=outputs)
    image_input.change(fn=classify_waste, inputs=[image_input], outputs=outputs)


# ── LAUNCH ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch()

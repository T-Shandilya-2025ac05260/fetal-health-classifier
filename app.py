import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, matthews_corrcoef, confusion_matrix
)

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Fetal Health Classifier",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Brand palette
PURPLE      = '#A855F7'
TEAL        = '#14B8A6'
DARK_BG     = '#0E1117'
CARD_BG     = '#1A1F2E'
BORDER      = '#2D3441'
TEXT_LIGHT  = '#E4E6EB'
TEXT_MUTED  = '#9CA3AF'
AMBER       = '#F59E0B'
RED         = '#EF4444'

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown(f"""
<style>
    .stApp {{
        background: linear-gradient(135deg, {DARK_BG} 0%, #151922 100%);
    }}
    h1 {{
        background: linear-gradient(90deg, {PURPLE} 0%, {TEAL} 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 700;
        font-size: 2.5rem !important;
        padding-top: 0.5rem;
    }}
    h2, h3 {{ color: {TEXT_LIGHT}; font-weight: 600; }}

    div[data-testid="metric-container"] {{
        background: linear-gradient(135deg, rgba(168,85,247,0.05) 0%, rgba(20,184,166,0.05) 100%);
        border: 1px solid {BORDER};
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
        transition: all 0.2s ease;
    }}
    div[data-testid="metric-container"]:hover {{
        border-color: {PURPLE};
        box-shadow: 0 4px 20px rgba(168,85,247,0.2);
    }}
    div[data-testid="metric-container"] label {{
        color: {TEXT_MUTED} !important;
        font-size: 0.8rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {{
        color: {TEAL} !important;
        font-size: 1.6rem !important;
        font-weight: 700;
    }}

    [data-testid="stFileUploader"] {{
        background: rgba(168,85,247,0.05);
        border: 2px dashed {PURPLE};
        border-radius: 12px;
        padding: 20px;
    }}

    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #151922 0%, {DARK_BG} 100%);
        border-right: 1px solid {BORDER};
    }}

    .stTabs [data-baseweb="tab-list"] {{ gap: 8px; }}
    .stTabs [data-baseweb="tab"] {{
        background: {CARD_BG};
        border-radius: 8px 8px 0 0;
        padding: 8px 20px;
        color: {TEXT_MUTED};
    }}
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(90deg, rgba(168,85,247,0.2) 0%, rgba(20,184,166,0.2) 100%);
        color: {TEXT_LIGHT};
        border-bottom: 2px solid {PURPLE};
    }}

    hr {{ border-color: {BORDER}; margin: 2rem 0; }}
</style>
""", unsafe_allow_html=True)

plt.style.use('dark_background')

# ============================================================
# LOAD ARTIFACTS (cached)
# ============================================================
@st.cache_resource
def load_artifacts():
    with open('model/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    files = {
        'Logistic Regression': 'model/logistic_regression.pkl',
        'Decision Tree':       'model/decision_tree.pkl',
        'K-Nearest Neighbors': 'model/knn.pkl',
        'Naive Bayes':         'model/naive_bayes.pkl',
        'Random Forest':       'model/random_forest.pkl',
    }
    models = {}
    for name, path in files.items():
        with open(path, 'rb') as f:
            models[name] = pickle.load(f)
    return scaler, models

scaler, models = load_artifacts()

# ============================================================
# HEADER
# ============================================================
col_t, col_b = st.columns([3, 1])
with col_t:
    st.title("🩺 Fetal Health Classifier")
    st.markdown(
        f"<p style='color:{TEXT_MUTED};font-size:1.1rem;margin-top:-10px;'>"
        f"Multi-model classifier for cardiotocogram (CTG) analysis · 3 classes · 5 models</p>",
        unsafe_allow_html=True
    )
with col_b:
    st.markdown(
        f"<div style='text-align:right;padding-top:20px;'>"
        f"<span style='background:rgba(20,184,166,0.2);color:{TEAL};"
        f"padding:6px 14px;border-radius:20px;font-size:0.85rem;font-weight:600;'>● LIVE</span></div>",
        unsafe_allow_html=True
    )

st.divider()

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown(f"<h2 style='color:{PURPLE};'>⚙️ Controls</h2>", unsafe_allow_html=True)
    selected_model_name = st.selectbox(
        "Model", list(models.keys()), index=4,
        help="Random Forest is typically the top performer."
    )

    st.markdown("---")
    st.markdown(f"<h3 style='color:{TEAL};font-size:1rem;'>📊 Dataset</h3>", unsafe_allow_html=True)
    st.markdown(
        f"<p style='color:{TEXT_MUTED};font-size:0.85rem;'><b>Fetal Health Classification</b> (Kaggle)<br>2,126 CTG records × 21 features<br>3 classes: Normal / Suspect / Pathological</p>",
        unsafe_allow_html=True
    )

    st.markdown("---")
    st.markdown(f"<h3 style='color:{TEAL};font-size:1rem;'>🎯 Class Legend</h3>", unsafe_allow_html=True)
    for cls, color in [("1 · Normal", TEAL), ("2 · Suspect", AMBER), ("3 · Pathological", RED)]:
        st.markdown(
            f"<div style='padding:4px 0;'>"
            f"<span style='background:{color};width:12px;height:12px;border-radius:50%;"
            f"display:inline-block;margin-right:8px;'></span>"
            f"<span style='color:{TEXT_LIGHT};font-size:0.9rem;'>{cls}</span></div>",
            unsafe_allow_html=True
        )
    st.markdown("---")
    st.caption("BITS Pilani M.Tech AIML · ML Assignment 2")
    st.caption("BITS ID: 2025AC05260")

# ============================================================
# UPLOAD
# ============================================================
st.markdown(f"<h2>📁 Upload Test Data</h2>", unsafe_allow_html=True)
uploaded = st.file_uploader(
    "Drag and drop your CSV or click to browse", type="csv",
    help="Expected: 21 feature columns + 1 target column 'fetal_health'"
)

if uploaded is None:
    st.info("👆 Upload the `test_data.csv` file to begin analysis.")
    st.stop()

try:
    df = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Could not read the CSV: {e}")
    st.stop()

if 'fetal_health' not in df.columns:
    st.error("Missing required column `fetal_health` (target labels 1/2/3).")
    st.stop()

# ============================================================
# DATA OVERVIEW
# ============================================================
n1 = (df['fetal_health']==1).sum()
n2 = (df['fetal_health']==2).sum()
n3 = (df['fetal_health']==3).sum()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Records", f"{len(df):,}")
c2.metric("Normal", f"{n1}", f"{n1/len(df)*100:.1f}%")
c3.metric("Suspect", f"{n2}", f"{n2/len(df)*100:.1f}%")
c4.metric("Pathological", f"{n3}", f"{n3/len(df)*100:.1f}%")

with st.expander("🔍 Preview first 10 rows"):
    st.dataframe(df.head(10), use_container_width=True)

st.divider()

# ============================================================
# PREPROCESS + PREDICT
# ============================================================
X = df.drop('fetal_health', axis=1)
y = df['fetal_health'].astype(int)
X_scaled = scaler.transform(X)

model = models[selected_model_name]
y_pred = model.predict(X_scaled)
y_proba = model.predict_proba(X_scaled)

# ============================================================
# KPI CARDS
# ============================================================
st.markdown(
    f"<h2>📈 Performance Metrics "
    f"<span style='color:{PURPLE};font-size:1.2rem;'>· {selected_model_name}</span></h2>",
    unsafe_allow_html=True
)
metrics = {
    'Accuracy':  accuracy_score(y, y_pred),
    'AUC':       roc_auc_score(y, y_proba, multi_class='ovr', average='macro'),
    'Precision': precision_score(y, y_pred, average='macro', zero_division=0),
    'Recall':    recall_score(y, y_pred, average='macro', zero_division=0),
    'F1':        f1_score(y, y_pred, average='macro', zero_division=0),
    'MCC':       matthews_corrcoef(y, y_pred),
}
cols = st.columns(6)
for col, (name, val) in zip(cols, metrics.items()):
    col.metric(name, f"{val:.4f}")

st.divider()

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3 = st.tabs(["🎯 Confusion Matrix", "🔬 Prediction Explorer", "⚡ Feature Importance"])
class_labels = ['Normal', 'Suspect', 'Pathological']

# --- Tab 1: Confusion Matrix ---
with tab1:
    st.markdown(
        f"<p style='color:{TEXT_MUTED};'>Diagonal = correct predictions. "
        f"Off-diagonal cells reveal error patterns.</p>",
        unsafe_allow_html=True
    )
    cm = confusion_matrix(y, y_pred)
    cmap = LinearSegmentedColormap.from_list('purple_teal', [DARK_BG, PURPLE, TEAL], N=256)

    fig, ax = plt.subplots(figsize=(7, 5.5), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)
    sns.heatmap(
        cm, annot=True, fmt='d', cmap=cmap,
        xticklabels=class_labels, yticklabels=class_labels,
        ax=ax, cbar=True, linewidths=1, linecolor=DARK_BG,
        annot_kws={'color': 'white', 'fontweight': 'bold', 'fontsize': 14}
    )
    ax.set_xlabel('Predicted', color=TEXT_LIGHT, fontsize=12, fontweight='bold')
    ax.set_ylabel('Actual',   color=TEXT_LIGHT, fontsize=12, fontweight='bold')
    ax.tick_params(colors=TEXT_LIGHT)
    for spine in ax.spines.values():
        spine.set_edgecolor(TEXT_MUTED)
    st.pyplot(fig, use_container_width=False)

    st.markdown(f"<h4 style='color:{TEAL};'>Error Analysis</h4>", unsafe_allow_html=True)
    e1, e2, e3 = st.columns(3)
    correct = np.trace(cm)
    errors = cm.sum() - correct
    critical = cm[2, 0] + cm[2, 1]  # Pathological misclassified
    e1.metric("Total Correct", correct, f"{correct/cm.sum()*100:.1f}%")
    e2.metric("Total Errors",  errors,  f"-{errors/cm.sum()*100:.1f}%", delta_color="inverse")
    e3.metric("Critical Misses", critical,
              help="Pathological cases predicted as Normal or Suspect — clinically dangerous")

# --- Tab 2: Prediction Explorer ---
with tab2:
    st.markdown(
        f"<p style='color:{TEXT_MUTED};'>Pick any row from your uploaded data "
        f"to inspect how the model interpreted it.</p>",
        unsafe_allow_html=True
    )
    row_idx = st.slider("Row index", 0, len(df)-1, 0)

    cL, cR = st.columns([1, 1.2])
    with cL:
        st.markdown(f"<h4 style='color:{PURPLE};'>Selected Sample</h4>", unsafe_allow_html=True)
        actual = int(y.iloc[row_idx])
        predicted = int(y_pred[row_idx])
        match_color = PURPLE if actual == predicted else RED
        st.markdown(
            f"<div style='background:{CARD_BG};padding:20px;border-radius:12px;border:1px solid {BORDER};'>"
            f"<p style='color:{TEXT_MUTED};margin:0;font-size:0.85rem;'>ROW INDEX</p>"
            f"<h2 style='color:{TEXT_LIGHT};margin:4px 0;'>#{row_idx}</h2>"
            f"<hr style='margin:12px 0;border-color:{BORDER};'>"
            f"<p style='color:{TEXT_MUTED};margin:8px 0 4px 0;font-size:0.85rem;'>ACTUAL LABEL</p>"
            f"<p style='color:{TEAL};font-size:1.2rem;font-weight:600;margin:0;'>{actual} · {class_labels[actual-1]}</p>"
            f"<p style='color:{TEXT_MUTED};margin:12px 0 4px 0;font-size:0.85rem;'>PREDICTED</p>"
            f"<p style='color:{match_color};font-size:1.2rem;font-weight:600;margin:0;'>"
            f"{predicted} · {class_labels[predicted-1]} {'✓' if actual==predicted else '✗'}</p></div>",
            unsafe_allow_html=True
        )
        with st.expander("View feature values"):
            st.dataframe(X.iloc[row_idx].to_frame(name='value'), use_container_width=True)

    with cR:
        st.markdown(f"<h4 style='color:{PURPLE};'>Predicted Probabilities</h4>", unsafe_allow_html=True)
        probs = y_proba[row_idx]
        fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=DARK_BG)
        ax.set_facecolor(DARK_BG)
        bars = ax.barh(class_labels, probs, color=[TEAL, AMBER, RED], edgecolor='none', height=0.6)
        for bar, p in zip(bars, probs):
            ax.text(p + 0.02, bar.get_y() + bar.get_height()/2,
                    f'{p*100:.1f}%', color=TEXT_LIGHT, va='center',
                    fontweight='bold', fontsize=11)
        ax.set_xlim(0, 1.15)
        ax.set_xlabel('Probability', color=TEXT_LIGHT, fontsize=11)
        ax.tick_params(colors=TEXT_LIGHT, labelsize=9)
        for s in ['top','right']:
            ax.spines[s].set_visible(False)
        for s in ['left','bottom']:
            ax.spines[s].set_edgecolor(TEXT_MUTED)
        ax.grid(axis='x', color=BORDER, alpha=0.5)
        ax.set_axisbelow(True)
        st.pyplot(fig, use_container_width=True)

# --- Tab 3: Feature Importance ---
with tab3:
    st.markdown(
        f"<p style='color:{TEXT_MUTED};'>Which features drive this model's predictions most?</p>",
        unsafe_allow_html=True
    )
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        method = "Native feature importance (from tree splits)"
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_).mean(axis=0)
        method = "Average absolute coefficient magnitude across classes"
    else:
        importances = None

    if importances is None:
        st.info(
            f"**{selected_model_name}** doesn't expose a natural feature-importance measure. "
            f"kNN uses distance-based voting and Naive Bayes uses class-conditional distributions — "
            f"neither ranks features intrinsically. Switch to Random Forest, Decision Tree, or Logistic Regression."
        )
    else:
        st.markdown(
            f"<p style='color:{TEXT_MUTED};font-size:0.85rem;font-style:italic;'>Method: {method}</p>",
            unsafe_allow_html=True
        )
        feat_imp = pd.DataFrame({'feature': X.columns, 'importance': importances}) \
                   .sort_values('importance', ascending=True).tail(15)

        n = len(feat_imp)
        grad = [(0.078+(0.66-0.078)*i/n, 0.72-(0.72-0.33)*i/n, 0.65+(0.97-0.65)*i/n) for i in range(n)]

        fig, ax = plt.subplots(figsize=(7, 5.5), facecolor=DARK_BG)
        ax.set_facecolor(DARK_BG)
        ax.barh(feat_imp['feature'], feat_imp['importance'], color=grad, edgecolor='none')
        ax.set_xlabel('Importance', color=TEXT_LIGHT, fontsize=10)
        ax.tick_params(colors=TEXT_LIGHT, labelsize=9)
        for s in ['top','right']:
            ax.spines[s].set_visible(False)
        for s in ['left','bottom']:
            ax.spines[s].set_edgecolor(TEXT_MUTED)
        ax.grid(axis='x', color=BORDER, alpha=0.5)
        ax.set_axisbelow(True)
        ax.set_title('Top 15 Most Important Features', color=TEXT_LIGHT, fontsize=12, pad=12, loc='left')

        fi_left, fi_center, fi_right = st.columns([1, 4, 1])

        with fi_center:

            st.pyplot(fig, use_container_width=True)

st.divider()

# ============================================================
# COMPARE ALL MODELS
# ============================================================
st.markdown(f"<h2>🏆 Compare All Models</h2>", unsafe_allow_html=True)
st.markdown(
    f"<p style='color:{TEXT_MUTED};'>Best score per column highlighted in teal.</p>",
    unsafe_allow_html=True
)

rows = []
for name, m in models.items():
    yp  = m.predict(X_scaled)
    ypp = m.predict_proba(X_scaled)
    rows.append({
        'Model': name,
        'Accuracy':  accuracy_score(y, yp),
        'AUC':       roc_auc_score(y, ypp, multi_class='ovr', average='macro'),
        'Precision': precision_score(y, yp, average='macro', zero_division=0),
        'Recall':    recall_score(y, yp, average='macro', zero_division=0),
        'F1':        f1_score(y, yp, average='macro', zero_division=0),
        'MCC':       matthews_corrcoef(y, yp),
    })
comparison_df = pd.DataFrame(rows).set_index('Model')
st.dataframe(
    comparison_df.style
        .highlight_max(axis=0, props=f'background-color:{TEAL};color:white;font-weight:bold;')
        .format('{:.4f}'),
    use_container_width=True
)

winner = comparison_df['MCC'].idxmax()
st.markdown(
    f"<div style='background:linear-gradient(90deg,rgba(168,85,247,0.1) 0%,rgba(20,184,166,0.1) 100%);"
    f"padding:16px 20px;border-radius:12px;border-left:4px solid {TEAL};margin-top:16px;'>"
    f"<p style='color:{TEXT_MUTED};margin:0;font-size:0.85rem;'>🏆 OVERALL WINNER (BY MCC)</p>"
    f"<h3 style='color:{TEAL};margin:4px 0 0 0;'>{winner}</h3></div>",
    unsafe_allow_html=True
)

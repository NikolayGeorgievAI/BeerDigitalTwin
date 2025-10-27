import os
import re
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from io import BytesIO
from openai import OpenAI

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# ------------------------------------------------------------------------------------
# TEXT NORMALIZATION / MATCHING HELPERS
# ------------------------------------------------------------------------------------

def _clean_name(name: str) -> str:
    """Lowercase, strip special marks, keep alnum only."""
    if name is None:
        return ""
    s = str(name).lower()
    s = s.replace("®", "").replace("™", "")
    s = re.sub(r"[^a-z0-9]", "", s)
    return s


def _best_feature_match(user_name: str, feature_cols: list, prefix: str):
    """
    Fuzzy-ish match of user input to model feature column.
    We only consider columns that start with the given prefix.
    """
    cleaned_user = _clean_name(user_name)
    best_match = None
    best_score = -1

    for col in feature_cols:
        if not col.startswith(prefix):
            continue

        raw_label = col[len(prefix):]  # e.g. 'Citra®'
        cleaned_label = _clean_name(raw_label)

        # quick gate
        if len(cleaned_user) >= 3 and cleaned_user[:3] not in cleaned_label:
            continue

        # naive overlap score
        common = set(cleaned_user) & set(cleaned_label)
        score = len(common)
        if score > best_score:
            best_score = score
            best_match = col

    return best_match


def _choices_from_features(feature_cols, preferred_prefix=None):
    """
    Build human-friendly dropdowns from model's feature columns.
    1. Try limiting to columns matching preferred_prefix ('hop_', 'malt_', 'yeast_').
    2. Fallback to everything.
    3. Cleanup prefixes, remove ®/™, replace '_' with space.
    """

    def prettify(label: str) -> str:
        label = label.replace("®", "").replace("™", "")
        label = label.replace("_", " ").strip()
        return label

    subset = []

    # Pass 1: just preferred prefix
    if preferred_prefix:
        for col in feature_cols:
            if col.startswith(preferred_prefix):
                raw_label = col[len(preferred_prefix):]
                subset.append(prettify(raw_label))

    # Pass 2: fallback (strip common prefixes)
    if not subset:
        for col in feature_cols:
            cand = col
            for p in ["hop_", "malt_", "grain_", "base_", "yeast_", "strain_", "y_", "m_"]:
                if cand.startswith(p):
                    cand = cand[len(p):]
            subset.append(prettify(cand))

    # dedupe and sort
    cleaned = []
    for name in subset:
        if name and name not in cleaned:
            cleaned.append(name)

    cleaned = sorted(cleaned, key=lambda s: s.lower())
    return cleaned


# ------------------------------------------------------------------------------------
# LOAD MODELS
# ------------------------------------------------------------------------------------

ROOT_DIR = os.path.dirname(__file__)

# --- Hop aroma model bundle ---
HOP_MODEL_PATH = os.path.join(ROOT_DIR, "hop_aroma_model.joblib")
hop_bundle = joblib.load(HOP_MODEL_PATH)
hop_model = hop_bundle["model"]
hop_feature_cols = hop_bundle["feature_cols"]
hop_dims = [
    a for a in hop_bundle["aroma_dims"]
    if str(a).lower() not in ("nan", "", "none")
]

# --- Malt model bundle ---
MALT_MODEL_PATH = os.path.join(ROOT_DIR, "malt_sensory_model.joblib")
malt_bundle = joblib.load(MALT_MODEL_PATH)
malt_model = malt_bundle["model"]
malt_feature_cols = malt_bundle["feature_cols"]
malt_dims = malt_bundle["flavor_cols"]

# --- Yeast model bundle ---
YEAST_MODEL_PATH = os.path.join(ROOT_DIR, "yeast_sensory_model.joblib")
yeast_bundle = joblib.load(YEAST_MODEL_PATH)
yeast_model = yeast_bundle["model"]
yeast_feature_cols = yeast_bundle["feature_cols"]
yeast_dims = yeast_bundle["flavor_cols"]  # ex: ['Temp_avg_F', 'Flocculation_num', 'Attenuation_num', ...]

# Build dropdown lists
HOP_CHOICES = _choices_from_features(hop_feature_cols, preferred_prefix="hop_")
MALT_CHOICES = _choices_from_features(malt_feature_cols, preferred_prefix="malt_")
YEAST_CHOICES = _choices_from_features(yeast_feature_cols, preferred_prefix="yeast_")


# ------------------------------------------------------------------------------------
# FEATURE BUILDERS & PREDICTORS
# ------------------------------------------------------------------------------------

def build_hop_features(user_hops):
    """
    user_hops: [ {"name": "Citra", "amt": 50}, ...]
    amt in grams.
    Returns 1 x n_features DataFrame aligned to hop_feature_cols.
    """
    totals = {c: 0.0 for c in hop_feature_cols}

    for entry in user_hops:
        nm = entry.get("name", "")
        amt = float(entry.get("amt", 0.0))
        if amt <= 0 or not nm or str(nm).strip() in ["", "-"]:
            continue

        match = _best_feature_match(nm, hop_feature_cols, prefix="hop_")
        if match:
            totals[match] += amt

    return pd.DataFrame([totals], columns=hop_feature_cols)


def predict_hop_profile(user_hops):
    """
    Return dict { hop_dimension -> score }
    """
    X = build_hop_features(user_hops)
    y_pred = hop_model.predict(X)[0]
    return {dim: float(val) for dim, val in zip(hop_dims, y_pred)}


def build_malt_features(user_malts):
    """
    user_malts: [ {"name": "Maris Otter", "pct": 70}, ...]
    pct is % of grist.
    Returns 1 x n_features DataFrame aligned to malt_feature_cols.
    """
    totals = {c: 0.0 for c in malt_feature_cols}

    for entry in user_malts:
        nm = entry.get("name", "")
        pct = float(entry.get("pct", 0.0))
        if pct <= 0 or not nm or str(nm).strip() in ["", "-"]:
            continue

        # try "malt_" first
        match = _best_feature_match(nm, malt_feature_cols, prefix="malt_")
        if match is None:
            # fallback prefixes
            for pfx in ["grain_", "base_", "m_", "malt_"]:
                m2 = _best_feature_match(nm, malt_feature_cols, prefix=pfx)
                if m2:
                    match = m2
                    break

        if match:
            totals[match] += pct

    return pd.DataFrame([totals], columns=malt_feature_cols)


def predict_malt_profile(user_malts):
    """
    Return dict { malt_dimension -> score }.
    ex {'sweetness': 12, 'body_full': 0.3, 'color_intensity': 5.0, ...}
    """
    X = build_malt_features(user_malts)
    y_pred = malt_model.predict(X)[0]
    return {dim: float(val) for dim, val in zip(malt_dims, y_pred)}


def build_yeast_features(user_yeast):
    """
    user_yeast:
      {"strain": "London Ale III", "ferm_temp_c": 20.0 }
    We'll fuzzy match strain to yeast_feature_cols, set that to 1.0.
    """
    totals = {c: 0.0 for c in yeast_feature_cols}

    strain = user_yeast.get("strain", "")
    # first pass: "yeast_"
    match = _best_feature_match(strain, yeast_feature_cols, prefix="yeast_")
    if match is None:
        for pfx in ["strain_", "y_", "yeast_", ""]:
            m2 = _best_feature_match(strain, yeast_feature_cols, prefix=pfx) if pfx else None
            if m2:
                match = m2
                break
    if match:
        totals[match] = 1.0

    # Note: we are not yet encoding temp into numeric features for yeast_model.
    return pd.DataFrame([totals], columns=yeast_feature_cols)


def predict_yeast_profile(user_yeast):
    """
    Return dict { yeast_dim -> score }.
    ex {'Temp_avg_F': 68.5, 'Flocculation_num': 0.6, 'Attenuation_num': 0.7}
    """
    X = build_yeast_features(user_yeast)
    y_pred = yeast_model.predict(X)[0]
    return {dim: float(val) for dim, val in zip(yeast_dims, y_pred)}


# ------------------------------------------------------------------------------------
# RADAR PLOTS (SPIDER CHARTS)
# ------------------------------------------------------------------------------------

def _radar_subplot(ax, labels, values, title=None, color="#1f77b4"):
    """
    Draws a filled radar plot on an existing polar axis.
    - Hides radial tick labels (numbers)
    - Shows category labels around the shape
    """
    n = len(labels)
    # angles around circle
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)

    # close shape
    vals2 = list(values) + [values[0]]
    th2 = list(theta) + [theta[0]]

    # main polygon
    ax.plot(th2, vals2, color=color, linewidth=2)
    ax.fill(th2, vals2, color=color, alpha=0.15)

    # put spoke labels on axes
    ax.set_xticks(theta)
    ax.set_xticklabels(labels, fontsize=10)

    # hide radial labels (numbers)
    ax.set_yticks([])
    # lighten the radial grid so we keep shape but not emphasize values
    ax.grid(True, color="#cccccc", alpha=0.4)

    if title:
        ax.set_title(title, fontsize=14, pad=20, fontweight="600")


def make_radar_row(hop_profile, malt_profile, yeast_profile):
    """
    Build a single figure with 3 side-by-side radar plots (Hops, Malt, Yeast).
    We'll select representative dims from each profile for readability.
    """
    # Pick hop aroma top dims
    # We'll sort by descending intensity and take up to 8
    hop_items = sorted(hop_profile.items(), key=lambda kv: kv[1], reverse=True)
    hop_items = hop_items[:8] if hop_items else []
    hop_labels = [k for k, v in hop_items]
    hop_vals = [v for k, v in hop_items]

    # Malt: choose e.g. sweetness, body_full, color_intensity if present
    malt_pick_order = ["sweetness", "body_full", "color_intensity"]
    malt_labels = []
    malt_vals = []
    for dim in malt_pick_order:
        if dim in malt_profile:
            malt_labels.append(dim)
            malt_vals.append(malt_profile[dim])
    # fallback if none found
    if not malt_labels and malt_profile:
        # just take top 3 by value
        malt_items = sorted(malt_profile.items(), key=lambda kv: kv[1], reverse=True)[:3]
        malt_labels = [k for k, v in malt_items]
        malt_vals = [v for k, v in malt_items]

    # Yeast: choose flocculation, attenuation, temp if present
    yeast_pick_order = ["Flocculation_num", "Attenuation_num", "Temp_avg_F", "Temp_avg_F_", "Temp_avg_F_c"]
    yeast_labels = []
    yeast_vals = []
    for dim in yeast_pick_order:
        if dim in yeast_profile:
            yeast_labels.append(dim)
            yeast_vals.append(yeast_profile[dim])
    if not yeast_labels and yeast_profile:
        y_items = sorted(yeast_profile.items(), key=lambda kv: kv[1], reverse=True)[:3]
        yeast_labels = [k for k, v in y_items]
        yeast_vals = [v for k, v in y_items]

    # If any list is empty, create a dummy so the subplot doesn't explode
    if not hop_labels:
        hop_labels = ["aroma"]
        hop_vals = [0.01]
    if not malt_labels:
        malt_labels = ["malt"]
        malt_vals = [0.01]
    if not yeast_labels:
        yeast_labels = ["fermentation"]
        yeast_vals = [0.01]

    # We'll not rescale everything to a single 0-1 range because absolute
    # numeric scales differ, but we *hide* numbers anyway. We just want shape.

    fig = plt.figure(figsize=(10, 3))
    # create 3 subplots horizontally
    ax1 = fig.add_subplot(1, 3, 1, polar=True)
    ax2 = fig.add_subplot(1, 3, 2, polar=True)
    ax3 = fig.add_subplot(1, 3, 3, polar=True)

    _radar_subplot(ax1, hop_labels, hop_vals, title="Hops / Aroma", color="#1f77b4")
    _radar_subplot(ax2, malt_labels, malt_vals, title="Malt / Body-Sweetness", color="#2ca02c")
    _radar_subplot(ax3, yeast_labels, yeast_vals, title="Yeast / Fermentation", color="#d62728")

    fig.tight_layout()
    return fig


# ------------------------------------------------------------------------------------
# AZURE OPENAI BREWMASTER ADVISOR
# ------------------------------------------------------------------------------------

AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "").strip()
AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY", "").strip()
AZURE_OPENAI_DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "").strip()

def call_azure_brewmaster_notes(
    brewer_goal_text: str,
    hop_vec: dict,
    malt_vec: dict,
    yeast_vec: dict,
):
    """
    Ask Azure OpenAI for guidance using your deployed model,
    or fall back to a built-in prototype if it fails.
    """

    def short_block(vec: dict, topn=5):
        if not vec:
            return "N/A"
        items = sorted(vec.items(), key=lambda kv: kv[1], reverse=True)
        out_lines = []
        for k, v in items[:topn]:
            out_lines.append(f"{k}: {v:.2f}")
        return "; ".join(out_lines)

    hop_desc = short_block(hop_vec, topn=5)
    malt_desc = short_block(malt_vec, topn=5)
    yeast_desc = short_block(yeast_vec, topn=5)

    system_msg = (
        "You are a professional brewmaster. "
        "Given hop aroma, malt/body, and fermentation character predictions, "
        "suggest concrete recipe/process adjustments to move the beer toward "
        "the brewer's stated style goal. Focus on late hopping, bitterness, "
        "grist tweaks for body/sweetness, and fermentation/yeast guidance. "
        "Answer in 3-4 short numbered sections."
    )

    user_msg = (
        f"Brewer's style/goal:\n{brewer_goal_text}\n\n"
        f"Hop aroma profile (top notes ~0..1+): {hop_desc}\n"
        f"Malt/body profile (sweetness, fullness, color): {malt_desc}\n"
        f"Yeast/fermentation traits: {yeast_desc}\n\n"
        "Please respond in 3-4 short numbered sections:\n"
        "1. Hop adjustments (varieties, timing, bitterness)\n"
        "2. Malt/grist tweaks (body, sweetness, color)\n"
        "3. Fermentation guidance (yeast choice, temp)\n"
        "4. A one-sentence summary of what to change.\n"
        "Be concise and practical."
    )

    ai_text = ""

    if AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY and AZURE_OPENAI_DEPLOYMENT:
        try:
            # Azure-style OpenAI client
            # - base_url: <endpoint>/openai/deployments/<deployment_name>
            # - default_query: { api-version: "2024-02-15-preview" }  (or a valid version for your deployment)
            client = OpenAI(
                api_key=AZURE_OPENAI_API_KEY,
                base_url=f"{AZURE_OPENAI_ENDPOINT}openai/deployments/{AZURE_OPENAI_DEPLOYMENT}",
                default_query={"api-version": "2024-02-15-preview"},
            )

            completion = client.chat.completions.create(
                model=AZURE_OPENAI_DEPLOYMENT,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.4,
                max_tokens=450,
            )

            if completion and completion.choices:
                ai_text = completion.choices[0].message.get("content", "").strip()

        except Exception:
            ai_text = ""

    if not ai_text:
        # Fallback if Azure fails
        ai_text = (
            "Brewmaster Notes (prototype)\n"
            "• Hop toward tropical/stone-fruit character using high-oil late/dry hop "
            "additions; keep bitterness low to maintain softness.\n"
            "• Add oats or wheat for pillowy mouthfeel and stable haze.\n"
            "• Avoid overly toasty/crystal malts if you want a pale, juicy profile.\n"
            "• Use a moderately ester-friendly yeast and keep fermentation temps "
            "in a softer range (around 18–20°C) to avoid harsh fusels.\n"
            "• Goal: juicy, saturated fruit aroma and plush mouthfeel without sharp bitterness."
        )

    return ai_text


# ------------------------------------------------------------------------------------
# STREAMLIT UI
# ------------------------------------------------------------------------------------

st.set_page_config(
    page_title="Beer Recipe Digital Twin",
    page_icon="🍺",
    layout="wide"
)

st.title("🍺 Beer Recipe Digital Twin")
st.markdown(
    """
Your AI brew assistant:

1. Build a hop bill, grain bill, and fermentation plan.  
2. Predict aroma, body, color, esters, mouthfeel — together.  
3. Get brewmaster-style guidance based on your style goal.
"""
)

st.markdown("---")

# ------------------------------------------------------------------------------------
# HOPS INPUT
# ------------------------------------------------------------------------------------
st.header("🌿 Hops (late/aroma additions)")

col_h1, col_h2 = st.columns([1,1])
with col_h1:
    hop1 = st.selectbox(
        "Main Hop Variety",
        HOP_CHOICES,
        index=HOP_CHOICES.index("Citra") if "Citra" in HOP_CHOICES else 0,
        key="hop1_select",
    ) if HOP_CHOICES else ""

    hop2 = st.selectbox(
        "Secondary Hop Variety",
        HOP_CHOICES,
        index=HOP_CHOICES.index("Mosaic") if "Mosaic" in HOP_CHOICES else 0,
        key="hop2_select",
    ) if HOP_CHOICES else ""

with col_h2:
    hop1_amt = st.number_input(
        "Hop 1 amount (g)",
        min_value=0.0,
        max_value=500.0,
        value=30.0,
        step=5.0,
    )
    hop2_amt = st.number_input(
        "Hop 2 amount (g)",
        min_value=0.0,
        max_value=500.0,
        value=20.0,
        step=5.0,
    )

user_hops = []
if hop1 and hop1_amt > 0:
    user_hops.append({"name": hop1, "amt": hop1_amt})
if hop2 and hop2_amt > 0:
    user_hops.append({"name": hop2, "amt": hop2_amt})

st.markdown("---")

# ------------------------------------------------------------------------------------
# MALT INPUT
# ------------------------------------------------------------------------------------
st.header("🌾 Malt / Grain Bill")

col_m1, col_m2 = st.columns([1,1])

with col_m1:
    malt1 = st.selectbox(
        "Base / primary malt",
        MALT_CHOICES,
        index=MALT_CHOICES.index("EXTRA PALE MALT") if "EXTRA PALE MALT" in MALT_CHOICES else 0,
        key="malt1_select",
    ) if MALT_CHOICES else ""

    malt2 = st.selectbox(
        "Specialty / character malt",
        MALT_CHOICES,
        index=MALT_CHOICES.index("HANÁ MALT") if "HANÁ MALT" in MALT_CHOICES else 0,
        key="malt2_select",
    ) if MALT_CHOICES else ""

with col_m2:
    malt1_pct = st.number_input(
        "Malt 1 (% grist)",
        min_value=0.0,
        max_value=100.0,
        value=70.0,
        step=1.0,
    )
    malt2_pct = st.number_input(
        "Malt 2 (% grist)",
        min_value=0.0,
        max_value=100.0,
        value=8.0,
        step=1.0,
    )

user_malts = []
if malt1 and malt1_pct > 0:
    user_malts.append({"name": malt1, "pct": malt1_pct})
if malt2 and malt2_pct > 0:
    user_malts.append({"name": malt2, "pct": malt2_pct})

st.markdown("---")

# ------------------------------------------------------------------------------------
# YEAST INPUT
# ------------------------------------------------------------------------------------
st.header("🧫 Yeast & Fermentation")

col_y1, col_y2 = st.columns([1,1])
with col_y1:
    yeast_strain = st.selectbox(
        "Yeast strain",
        YEAST_CHOICES,
        index=YEAST_CHOICES.index("Nottingham Ale Yeast") if "Nottingham Ale Yeast" in YEAST_CHOICES else 0,
        key="yeast_select",
    ) if YEAST_CHOICES else ""

with col_y2:
    # store temp in °C for UI, but the model's predicted output is in F etc.
    ferm_temp_c = st.number_input(
        "Fermentation temp (°C)",
        min_value=15.0,
        max_value=25.0,
        value=20.0,
        step=0.5,
    )

user_yeast = {
    "strain": yeast_strain,
    "ferm_temp_c": ferm_temp_c,
}

st.markdown("---")

# ------------------------------------------------------------------------------------
# PREDICT FLAVOR & BALANCE
# ------------------------------------------------------------------------------------
st.header("🍻 Predict Beer Flavor & Balance")

st.caption("Fill hops, malt, and yeast above — then click **'Predict Beer Flavor & Balance'** to simulate aroma, body, esters, color, etc.")

predict_clicked = st.button("🔬 Predict Beer Flavor & Balance")

hop_profile = {}
malt_profile = {}
yeast_profile = {}

if predict_clicked:
    if user_hops:
        hop_profile = predict_hop_profile(user_hops)
    if user_malts:
        malt_profile = predict_malt_profile(user_malts)
    if yeast_strain:
        yeast_profile = predict_yeast_profile(user_yeast)

    # ----------------  TEXT SNAPSHOT SECTION
    st.subheader("📊 Predicted Flavor Snapshot")

    # Hops
    st.markdown("**Hop aroma / character:**")
    if hop_profile:
        for k, v in hop_profile.items():
            st.markdown(f"- {k}: {v:.2f}")
    else:
        st.markdown("- (no hop data)")

    # Malt
    st.markdown("**Malt body / sweetness / color:**")
    if malt_profile:
        for k, v in malt_profile.items():
            st.markdown(f"- {k}: {v:.2f}")
    else:
        st.markdown("- (no malt data)")

    # Yeast
    st.markdown("**Yeast / fermentation profile:**")
    if yeast_profile:
        for k, v in yeast_profile.items():
            st.markdown(f"- {k}: {v:.2f}")
    else:
        st.markdown("- (no yeast data)")

    st.markdown("---")

    # ----------------  RADAR OVERVIEW
    st.subheader("🕸 Radar Overview")
    st.caption("Relative shape only. Axes are labeled by trait, numeric ticks/values are hidden.")

    try:
        fig = make_radar_row(hop_profile, malt_profile, yeast_profile)
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Radar plot unavailable: {e}")

st.markdown("---")

# ------------------------------------------------------------------------------------
# BREWMASTER AI ADVISOR (AZURE)
# ------------------------------------------------------------------------------------
st.header("🧪 AI Brewmaster Guidance")

brewer_goal = st.text_area(
    "What's your intent for this beer? (e.g. 'Soft hazy IPA with saturated stone fruit and pineapple, low bitterness, pillowy mouthfeel')",
    "i want to increase mango aroma without increasing bitterness.",
)

gen_notes_clicked = st.button("🧪 Generate Brewmaster Notes")

if gen_notes_clicked:
    # We'll call Azure with whatever the *latest predicted* profiles are
    # If they haven't hit predict yet, these dicts might be empty
    ai_md = call_azure_brewmaster_notes(
        brewer_goal_text=brewer_goal,
        hop_vec=hop_profile,
        malt_vec=malt_profile,
        yeast_vec=yeast_profile,
    )

    st.subheader("Brewmaster Notes")
    st.write(ai_md)

    st.caption(
        "Prototype — not production brewing advice. "
        "Always match your yeast strain's process window."
    )

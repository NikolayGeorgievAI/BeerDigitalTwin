import os
import re
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # <<< NEW (for radar charts)
import streamlit as st
import openai  # Azure OpenAI client

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# ======================================================
# ENV / AZURE OPENAI SETUP  (Chat Completions)
# ======================================================

AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "")  # e.g. beer-brewmaster-gpt-4.1-mini
AZURE_OPENAI_API_VERSION = os.environ.get(
    "AZURE_OPENAI_API_VERSION",
    "2024-02-01"  # adjust if portal says a different version for your model
)

client = openai.AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION,
)

# ======================================================
# TEXT NORMALIZATION
# ======================================================

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
    Fuzzy match something the user typed (e.g. "Citra")
    to a model feature column with a given prefix (e.g. "hop_Citra®").
    """
    cleaned_user = _clean_name(user_name)
    best_match = None
    best_score = -1

    for col in feature_cols:
        if not col.startswith(prefix):
            continue

        raw_label = col[len(prefix):]
        cleaned_label = _clean_name(raw_label)

        # quick gate
        if len(cleaned_user) >= 3 and cleaned_user[:3] not in cleaned_label:
            continue

        # naive overlap score
        score = len(set(cleaned_user) & set(cleaned_label))
        if score > best_score:
            best_score = score
            best_match = col

    return best_match


def _choices_from_features(feature_cols, preferred_prefix=None):
    """
    Create nice dropdown options from model feature columns.

    preferred_prefix='malt_', 'yeast_', etc.:
      we'll try to use only those first,
      then fall back to everything.
    """
    def prettify(label: str) -> str:
        label = label.replace("®", "").replace("™", "")
        label = label.replace("_", " ").strip()
        return label

    subset = []

    # pass 1: just preferred prefix
    if preferred_prefix:
        for col in feature_cols:
            if col.startswith(preferred_prefix):
                raw_label = col[len(preferred_prefix):]
                subset.append(prettify(raw_label))

    # pass 2: fallback to all if that was empty
    if not subset:
        for col in feature_cols:
            cand = col
            for p in ["hop_", "malt_", "grain_", "base_", "yeast_", "strain_", "y_", "m_"]:
                if cand.startswith(p):
                    cand = cand[len(p):]
            subset.append(prettify(cand))

    cleaned = []
    for name in subset:
        if name and name not in cleaned:
            cleaned.append(name)

    cleaned = sorted(cleaned, key=lambda s: s.lower())
    return cleaned


# ======================================================
# LOAD MODELS
# ======================================================

ROOT_DIR = os.path.dirname(__file__)

# --- Hop model
HOP_MODEL_PATH = os.path.join(ROOT_DIR, "hop_aroma_model.joblib")
hop_bundle = joblib.load(HOP_MODEL_PATH)
hop_model = hop_bundle["model"]
hop_feature_cols = hop_bundle["feature_cols"]
hop_dims = [
    a for a in hop_bundle.get("aroma_dims", [])
    if str(a).lower() not in ("nan", "", "none")
]

# --- Malt model (retrained)
MALT_MODEL_PATH = os.path.join(ROOT_DIR, "malt_sensory_model.joblib")
malt_bundle = joblib.load(MALT_MODEL_PATH)
malt_model = malt_bundle["model"]
malt_feature_cols = malt_bundle["feature_cols"]  # e.g. ["MOISTURE_MAX", ...]
malt_dims = malt_bundle["flavor_cols"]          # e.g. ["bready","caramel",...]

# --- Yeast model (retrained)
YEAST_MODEL_PATH = os.path.join(ROOT_DIR, "yeast_sensory_model.joblib")
yeast_bundle = joblib.load(YEAST_MODEL_PATH)
yeast_model = yeast_bundle["model"]
yeast_feature_cols = yeast_bundle["feature_cols"]  # "Name - Nottingham Ale Yeast", etc.
yeast_dims = yeast_bundle["flavor_cols"]           # e.g. ["fruity_esters","phenolic_spicy",...]

# Build dropdowns for UI
HOP_CHOICES = _choices_from_features(hop_feature_cols, preferred_prefix="hop_")
MALT_CHOICES = _choices_from_features(malt_feature_cols, preferred_prefix="malt_")
YEAST_CHOICES = _choices_from_features(yeast_feature_cols, preferred_prefix="yeast_")


# ======================================================
# FEATURE BUILDERS
# ======================================================

def build_hop_features(user_hops):
    """
    user_hops: [ {"name":"Citra","amt":50}, ... ]
    Return DataFrame (1 x hop_feature_cols)
    """
    totals = {c: 0.0 for c in hop_feature_cols}
    for entry in user_hops:
        nm = entry.get("name", "")
        amt = float(entry.get("amt", 0.0))
        if amt <= 0 or not nm.strip():
            continue

        match = _best_feature_match(nm, hop_feature_cols, prefix="hop_")
        if match:
            totals[match] += amt

    return pd.DataFrame([totals], columns=hop_feature_cols)


def predict_hop_profile(user_hops):
    """
    Return dict {hop_dim -> score}
    """
    if not user_hops:
        return {}
    X = build_hop_features(user_hops)
    y_pred = hop_model.predict(X)[0]
    return {dim: float(val) for dim, val in zip(hop_dims, y_pred)}


def build_malt_features(user_malts):
    """
    user_malts: [ {"name":"Maris Otter","pct":70}, ... ]
    Return DataFrame (1 x malt_feature_cols)
    """
    totals = {c: 0.0 for c in malt_feature_cols}

    for entry in user_malts:
        nm = entry.get("name", "")
        pct = float(entry.get("pct", 0.0))
        if pct <= 0 or not nm.strip():
            continue

        # try preferred 'malt_' first
        match = _best_feature_match(nm, malt_feature_cols, prefix="malt_")
        if match is None:
            # fallback to other prefixes
            for pfx in ["grain_", "base_", "malt_", "m_"]:
                match = _best_feature_match(nm, malt_feature_cols, prefix=pfx)
                if match:
                    break

        if match:
            totals[match] += pct

    return pd.DataFrame([totals], columns=malt_feature_cols)


def predict_malt_profile(user_malts):
    """
    Dict {malt_dim -> val}
    """
    if not user_malts:
        return {}
    X = build_malt_features(user_malts)
    y_pred = malt_model.predict(X)[0]
    return {dim: float(val) for dim, val in zip(malt_dims, y_pred)}


def build_yeast_features(user_yeast):
    """
    user_yeast:
      {
        "strain": "...",
        "ferm_temp_c": 20.0  # We keep C in UI now
      }
    We'll approximate by marking the chosen strain with 1.0,
    ignoring temp for now (simple first pass).
    """
    totals = {c: 0.0 for c in yeast_feature_cols}

    strain = user_yeast.get("strain", "")
    match = _best_feature_match(strain, yeast_feature_cols, prefix="yeast_")
    if match is None:
        for pfx in ["strain_", "y_", "yeast_", ""]:
            if not pfx:
                continue
            m2 = _best_feature_match(strain, yeast_feature_cols, prefix=pfx)
            if m2:
                match = m2
                break

    if match:
        totals[match] = 1.0

    return pd.DataFrame([totals], columns=yeast_feature_cols)


def predict_yeast_profile(user_yeast):
    """
    Dict {yeast_dim -> val}
    """
    if not user_yeast or not user_yeast.get("strain"):
        return {}
    X = build_yeast_features(user_yeast)
    y_pred = yeast_model.predict(X)[0]
    return {dim: float(val) for dim, val in zip(yeast_dims, y_pred)}


# ======================================================
# RADAR CHART HELPER
# ======================================================

import numpy as np

def plot_radar(profile_dict, title="Profile"):
    """
    Create a radar/spider chart for a dict {dimension: value}.
    We'll autoscale the radial max to the largest value in this dict
    so each plot is independently legible.
    """
    if not profile_dict:
        fig = plt.figure(figsize=(4,4))
        ax = plt.subplot(111)
        ax.text(0.5, 0.5, "no data", ha="center", va="center")
        ax.set_axis_off()
        return fig

    dims = list(profile_dict.keys())
    vals = [float(profile_dict[d]) for d in dims]

    # close polygon
    dims.append(dims[0])
    vals.append(vals[0])

    angles = np.linspace(0, 2 * np.pi, len(dims), endpoint=False)

    fig = plt.figure(figsize=(4,4))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, vals, marker="o")
    ax.fill(angles, vals, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dims[:-1], fontsize=8)
    ax.set_title(title)

    radial_max = max(vals) if max(vals) > 0 else 1.0
    ax.set_ylim(0, radial_max)

    plt.tight_layout()
    return fig


# ======================================================
# AI BREWMASTER CALL (Azure Chat Completions)
# ======================================================

def call_azure_brewmaster_notes(goal_text, beer_summary_md,
                                hop_prof, malt_prof, yeast_prof):
    """
    Ask Azure OpenAI (chat completions) for brew guidance.
    """

    if not beer_summary_md:
        beer_summary_md = "(No predicted beer summary yet.)"

    system_msg = (
        "You are an expert brewmaster. You analyze hop aroma, "
        "malt body/sweetness/color, and yeast fermentation character. "
        "You give concise, practical tuning advice for small-batch brewers."
    )

    user_msg = f"""
User's stated style / goal:
{goal_text or '(no stated goal)'}

Predicted beer (hops/malt/yeast summary):
{beer_summary_md}

Please provide:
1. Overall sensory read in ~2 short sentences.
2. Hop tuning advice (late additions, aroma shifts).
3. Malt/grist advice (body, sweetness, color).
4. Fermentation / yeast advice (strain, temp, ester goals).
5. A one-sentence final summary for a pro brewer.

Keep bullet points tight, <200 words total.
"""

    completion = client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        max_tokens=400,
        temperature=0.5,
    )

    ai_text = completion.choices[0].message["content"]

    answer_md = "### 🧪 AI Brewmaster Guidance\n\n" + ai_text.strip()
    return answer_md


# ======================================================
# STREAMLIT PAGE SETUP
# ======================================================

st.set_page_config(
    page_title="Beer Recipe Digital Twin",
    page_icon="🍺",
    layout="centered"
)

st.title("🍺 Beer Recipe Digital Twin")

st.markdown(
    """
**Your AI brew assistant.**
1. Pick hops, malt bill, and yeast.
2. Click **Predict Beer Flavor & Balance**.
3. Get an AI brewmaster readout.
4. Iterate.
    """
)
st.markdown("---")


# ------------------------------------------------------
# Session state for predictions / notes
# ------------------------------------------------------
if "hop_profile" not in st.session_state:
    st.session_state.hop_profile = {}
if "malt_profile" not in st.session_state:
    st.session_state.malt_profile = {}
if "yeast_profile" not in st.session_state:
    st.session_state.yeast_profile = {}
if "beer_summary" not in st.session_state:
    st.session_state.beer_summary = ""
if "ai_brewmaster_md" not in st.session_state:
    st.session_state.ai_brewmaster_md = ""


# ======================================================
# HOPS SECTION
# ======================================================

st.header("🌿 Hop Selection")

c1, c2 = st.columns([1,1])
with c1:
    hop1 = st.selectbox(
        "Hop 1 variety",
        HOP_CHOICES,
        index=HOP_CHOICES.index("Citra") if "Citra" in HOP_CHOICES else 0,
        key="hop1_select",
    ) if HOP_CHOICES else ""
    hop2 = st.selectbox(
        "Hop 2 variety",
        HOP_CHOICES,
        index=HOP_CHOICES.index("Mosaic") if "Mosaic" in HOP_CHOICES else 0,
        key="hop2_select",
    ) if HOP_CHOICES else ""

with c2:
    hop1_amt = st.number_input(
        "Hop 1 (g)",
        min_value=0.0,
        max_value=500.0,
        value=50.0,
        step=5.0
    )
    hop2_amt = st.number_input(
        "Hop 2 (g)",
        min_value=0.0,
        max_value=500.0,
        value=30.0,
        step=5.0
    )

user_hops = []
if hop1 and hop1_amt > 0:
    user_hops.append({"name": hop1, "amt": hop1_amt})
if hop2 and hop2_amt > 0:
    user_hops.append({"name": hop2, "amt": hop2_amt})

st.caption("We'll assess these hops' predicted aroma contribution (citrus, stone fruit, tropical, etc.).")

st.markdown("---")


# ======================================================
# MALT SECTION
# ======================================================

st.header("🌾 Malt / Grain Bill")

m1, m2 = st.columns([1,1])

with m1:
    malt1 = st.selectbox(
        "Malt 1 name",
        MALT_CHOICES,
        key="malt1_select",
    ) if MALT_CHOICES else ""
    malt1_pct = st.number_input(
        "Malt 1 (% grist)",
        min_value=0.0,
        max_value=100.0,
        value=70.0,
        step=1.0
    )

with m2:
    malt2 = st.selectbox(
        "Malt 2 name",
        MALT_CHOICES,
        key="malt2_select",
    ) if MALT_CHOICES else ""
    malt2_pct = st.number_input(
        "Malt 2 (% grist)",
        min_value=0.0,
        max_value=100.0,
        value=8.0,
        step=1.0
    )

user_malts = []
if malt1 and malt1_pct > 0:
    user_malts.append({"name": malt1, "pct": malt1_pct})
if malt2 and malt2_pct > 0:
    user_malts.append({"name": malt2, "pct": malt2_pct})

st.caption("We'll estimate body, sweetness, color, etc. from this grist.")
st.markdown("---")


# ======================================================
# YEAST / FERMENTATION SECTION
# (Now just shown expanded, Celsius in UI)
# ======================================================

st.header("🧫 Yeast & Fermentation")

y1, y2 = st.columns([1,1])

with y1:
    yeast_strain = st.selectbox(
        "Yeast strain",
        YEAST_CHOICES,
        key="yeast_select",
    ) if YEAST_CHOICES else ""

with y2:
    ferm_temp_c = st.number_input(
        "Fermentation temp (°C)",
        min_value=15.0,
        max_value=30.0,
        value=20.0,
        step=0.5
    )

user_yeast = {
    "strain": yeast_strain,
    "ferm_temp_c": ferm_temp_c,
}

st.caption("We'll infer ester profile / 'clean vs fruity', and mouthfeel from your chosen strain + temp.")
st.markdown("---")


# ======================================================
# PREDICT BEER FLAVOR & BALANCE
# ======================================================

st.subheader("🍻 Predict Beer Flavor & Balance")
st.caption(
    "Fill hops, malt, and yeast above — then click to simulate aroma, body, color, esters, etc."
)

if st.button("🍺 Predict Beer Flavor & Balance"):
    # compute hop/malt/yeast predictions
    hop_prof = predict_hop_profile(user_hops)
    malt_prof = predict_malt_profile(user_malts)
    yeast_prof = predict_yeast_profile(user_yeast)

    st.session_state.hop_profile = hop_prof
    st.session_state.malt_profile = malt_prof
    st.session_state.yeast_profile = yeast_prof

    # Build a short "beer summary" to feed into AI and show user
    # We'll do a friendly text summary
    lines = []

    # hop lines
    if hop_prof:
        lines.append("**Hop aroma / character**:")
        for k, v in hop_prof.items():
            lines.append(f"- {k}: {v:.2f}")
        lines.append("")

    # malt lines
    if malt_prof:
        lines.append("**Malt body / sweetness / color**:")
        for k, v in malt_prof.items():
            lines.append(f"- {k}: {v:.2f}")
        lines.append("")

    # yeast lines
    if yeast_prof:
        lines.append("**Yeast & fermentation character**:")
        for k, v in yeast_prof.items():
            lines.append(f"- {k}: {v:.2f}")
        lines.append("")

    beer_summary_md = "\n".join(lines).strip()
    st.session_state.beer_summary = beer_summary_md

    st.success("Beer flavor & balance predicted ✅")


# If we have a predicted summary, show it (and keep showing it)
if st.session_state.beer_summary:
    st.markdown("### 📊 Predicted Flavor Snapshot")
    st.markdown(st.session_state.beer_summary)

    # --- Radar charts section
    st.markdown("#### Visual Flavor Shape")
    col_r1, col_r2, col_r3 = st.columns(3)

    # Hops radar
    with col_r1:
        st.caption("Hops")
        fig_hops = plot_radar(st.session_state.hop_profile or {}, "Hops")
        st.pyplot(fig_hops)

    # Malt radar (normalize so high sweetness doesn't dominate)
    with col_r2:
        st.caption("Malt / Body / Sweetness / Color")
        malt_prof_plot = st.session_state.malt_profile or {}
        if malt_prof_plot:
            mvals = list(malt_prof_plot.values())
            mmax = max(mvals) if max(mvals) > 0 else 1
            malt_prof_norm = {k: (v / mmax) for k, v in malt_prof_plot.items()}
        else:
            malt_prof_norm = {}
        fig_malt = plot_radar(malt_prof_norm, "Malt (scaled)")
        st.pyplot(fig_malt)

    # Yeast radar
    with col_r3:
        st.caption("Yeast / Fermentation")
        fig_yeast = plot_radar(st.session_state.yeast_profile or {}, "Yeast")
        st.pyplot(fig_yeast)

st.markdown("---")


# ======================================================
# AI BREWMASTER GUIDANCE
# ======================================================

st.header("👨‍🔬 AI Brewmaster Guidance")

goal_text = st.text_area(
    "What's your intent for this beer? (e.g. 'Soft hazy IPA with saturated stone fruit and pineapple, low bitterness, pillowy mouthfeel')",
    value="Soft hazy IPA with saturated stone fruit and pineapple, low bitterness, pillowy mouthfeel",
    height=80,
)

if st.button("🧪 Generate Brewmaster Notes"):
    # call Azure chat completions with the last predicted summary
    if not st.session_state.beer_summary:
        st.warning("Please Predict Beer Flavor & Balance first.")
    else:
        ai_md = call_azure_brewmaster_notes(
            goal_text,
            st.session_state.beer_summary,
            st.session_state.hop_profile,
            st.session_state.malt_profile,
            st.session_state.yeast_profile,
        )
        st.session_state.ai_brewmaster_md = ai_md

# Keep showing whatever the last AI result was
if st.session_state.ai_brewmaster_md:
    st.markdown(st.session_state.ai_brewmaster_md)

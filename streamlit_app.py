import os
import re
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# -------------------------------------------------
# UTILITIES: fuzzy name matching
# -------------------------------------------------

def _clean_name(name: str) -> str:
    if name is None:
        return ""
    s = str(name).lower()
    s = s.replace("®", "").replace("™", "")
    s = re.sub(r"[^a-z0-9]", "", s)
    return s

def _best_feature_match(user_name: str, feature_cols: list, prefix: str):
    """
    prefix is 'hop_', 'malt_', 'yeast_' etc.
    Returns the best matching feature column name from the model.
    """
    cleaned_user = _clean_name(user_name)
    best_match = None
    best_score = -1

    for col in feature_cols:
        if not col.startswith(prefix):
            continue
        raw_label = col[len(prefix):]  # e.g. "Citra®" from "hop_Citra®"
        cleaned_label = _clean_name(raw_label)

        if len(cleaned_user) >= 3 and cleaned_user[:3] not in cleaned_label:
            continue

        common = set(cleaned_user) & set(cleaned_label)
        score = len(common)

        if score > best_score:
            best_score = score
            best_match = col

    return best_match

# -------------------------------------------------
# LOAD MODELS
# -------------------------------------------------

ROOT_DIR = os.path.dirname(__file__)

# --- Hop model bundle ---
HOP_MODEL_PATH = os.path.join(ROOT_DIR, "hop_aroma_model.joblib")
hop_bundle = joblib.load(HOP_MODEL_PATH)
hop_model = hop_bundle["model"]
hop_feature_cols = hop_bundle["feature_cols"]
hop_dims = [a for a in hop_bundle["aroma_dims"] if str(a).lower() not in ("nan", "", "none")]

# --- Malt model bundle ---
# Assumptions:
#   - file: malt_sensory_model.joblib
#   - same structure: { "model": ..., "feature_cols": [...], "aroma_dims": [...] }
MALT_MODEL_PATH = os.path.join(ROOT_DIR, "malt_sensory_model.joblib")
malt_bundle = joblib.load(MALT_MODEL_PATH)
malt_model = malt_bundle["model"]
malt_feature_cols = malt_bundle["feature_cols"]
# We'll call these malt dimensions (sweetness, caramel, toast, body, color, etc.)
malt_dims = [a for a in malt_bundle["aroma_dims"] if str(a).lower() not in ("nan", "", "none")]

# --- Yeast model bundle ---
# Assumptions:
#   - file: yeast_sensory_model.joblib
#   - same structure: { "model": ..., "feature_cols": [...], "aroma_dims": [...] }
YEAST_MODEL_PATH = os.path.join(ROOT_DIR, "yeast_sensory_model.joblib")
yeast_bundle = joblib.load(YEAST_MODEL_PATH)
yeast_model = yeast_bundle["model"]
yeast_feature_cols = yeast_bundle["feature_cols"]
yeast_dims = [a for a in yeast_bundle["aroma_dims"] if str(a).lower() not in ("nan", "", "none")]

# -------------------------------------------------
# HOP FUNCTIONS
# -------------------------------------------------

def build_hop_features(user_hops):
    """
    user_hops: [ {"name": "Citra", "amt": 50}, {"name": "Mosaic", "amt": 30}, ... ]
    amt assumed in grams.
    Returns 1-row DataFrame aligned to hop_feature_cols.
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
    Returns dict {hop_dim -> score}
    """
    X = build_hop_features(user_hops)
    y_pred = hop_model.predict(X)[0]
    return {dim: float(val) for dim, val in zip(hop_dims, y_pred)}

def advise_hops(user_hops, target_dim, trial_amt=20.0):
    """
    Brute force: try adding trial_amt grams of each known hop.
    Pick hop that increases target_dim the most.
    """
    base_vec = predict_hop_profile(user_hops)
    base_score = base_vec.get(target_dim, 0.0)

    best_choice = None
    best_delta = -999.0
    best_new_profile = None

    for col in hop_feature_cols:
        if not col.startswith("hop_"):
            continue
        candidate_label = col[len("hop_"):]
        trial_bill = user_hops + [{"name": candidate_label, "amt": trial_amt}]
        trial_vec = predict_hop_profile(trial_bill)
        trial_score = trial_vec.get(target_dim, 0.0)
        delta = trial_score - base_score
        if delta > best_delta:
            best_delta = delta
            best_choice = candidate_label
            best_new_profile = trial_vec

    return {
        "target_dim": target_dim,
        "addition_grams": trial_amt,
        "recommended_hop": best_choice,
        "expected_improvement": best_delta,
        "new_profile": best_new_profile,
        "current_score": base_score,
    }

# -------------------------------------------------
# MALT FUNCTIONS
# -------------------------------------------------

def build_malt_features(user_malts):
    """
    user_malts: [ {"name": "Maris Otter", "pct": 70}, {"name": "Caramunich III", "pct": 8}, ... ]
    We assume pct is percent of grist (0-100).
    Returns 1-row DataFrame aligned to malt_feature_cols.
    """
    totals = {c: 0.0 for c in malt_feature_cols}
    for entry in user_malts:
        nm = entry.get("name", "")
        pct = float(entry.get("pct", 0.0))
        if pct <= 0 or not nm or str(nm).strip() in ["", "-"]:
            continue
        match = _best_feature_match(nm, malt_feature_cols, prefix="malt_")
        if match:
            totals[match] += pct
    return pd.DataFrame([totals], columns=malt_feature_cols)

def predict_malt_profile(user_malts):
    """
    Returns dict {malt_dim -> score}
    e.g. {'body': 0.78, 'caramel': 0.62, 'toast': 0.44, 'color_srm': 14.2, ...}
    """
    X = build_malt_features(user_malts)
    y_pred = malt_model.predict(X)[0]
    return {dim: float(val) for dim, val in zip(malt_dims, y_pred)}

def advise_malt(user_malts, target_dim, trial_pct=2.0):
    """
    Try bumping each malt by trial_pct% of grist.
    Return the malt that best improves target_dim (body, caramel, etc.)
    """
    base_vec = predict_malt_profile(user_malts)
    base_score = base_vec.get(target_dim, 0.0)

    best_choice = None
    best_delta = -999.0
    best_new_profile = None

    for col in malt_feature_cols:
        if not col.startswith("malt_"):
            continue
        candidate_label = col[len("malt_"):]

        trial_bill = user_malts + [{"name": candidate_label, "pct": trial_pct}]
        trial_vec = predict_malt_profile(trial_bill)
        trial_score = trial_vec.get(target_dim, 0.0)
        delta = trial_score - base_score

        if delta > best_delta:
            best_delta = delta
            best_choice = candidate_label
            best_new_profile = trial_vec

    return {
        "target_dim": target_dim,
        "addition_pct": trial_pct,
        "recommended_malt": best_choice,
        "expected_improvement": delta if best_choice else 0.0,
        "new_profile": best_new_profile,
        "current_score": base_score,
    }

# -------------------------------------------------
# YEAST / FERMENTATION FUNCTIONS
# -------------------------------------------------

def build_yeast_features(user_yeast):
    """
    user_yeast: dict like
      {
        "strain": "London Ale III",
        "ferm_temp_f": 68
      }

    We'll do two things:
    - one-hot strain to yeast_feature_cols (prefix 'yeast_')
    - optionally bin fermentation temp into a feature if your model expects temp.
    For now we just match the strain by fuzzy name and set that column = 1.
    """
    totals = {c: 0.0 for c in yeast_feature_cols}

    strain = user_yeast.get("strain", "")
    match = _best_feature_match(strain, yeast_feature_cols, prefix="yeast_")
    if match:
        totals[match] = 1.0

    # If your yeast model expects numeric features like temp, you'd add them here.
    # For example, if yeast_feature_cols includes "fermtemp_68f", we could fuzzy-match temp buckets.
    # We'll keep it simple for now.

    return pd.DataFrame([totals], columns=yeast_feature_cols)

def predict_yeast_profile(user_yeast):
    """
    Returns dict {yeast_dim -> score}
    e.g. {'stone_fruit_ester': 0.8, 'attenuation_dryness': 0.6, 'phenolic': 0.1, 'haze_stability': 0.7, ...}
    """
    X = build_yeast_features(user_yeast)
    y_pred = yeast_model.predict(X)[0]
    return {dim: float(val) for dim, val in zip(yeast_dims, y_pred)}

def advise_yeast(user_yeast, target_dim):
    """
    Try swapping strain for every strain we know (i.e. every yeast_... feature),
    choose the strain that best improves target_dim.
    """
    base_vec = predict_yeast_profile(user_yeast)
    base_score = base_vec.get(target_dim, 0.0)

    best_choice = None
    best_delta = -999.0
    best_new_profile = None

    # brute force each yeast strain:
    for col in yeast_feature_cols:
        if not col.startswith("yeast_"):
            continue
        candidate_label = col[len("yeast_"):]

        trial_ferm = {
            "strain": candidate_label,
            # could copy ferm temp from user_yeast if you like
            "ferm_temp_f": user_yeast.get("ferm_temp_f", 68),
        }
        trial_vec = predict_yeast_profile(trial_ferm)
        trial_score = trial_vec.get(target_dim, 0.0)
        delta = trial_score - base_score

        if delta > best_delta:
            best_delta = delta
            best_choice = candidate_label
            best_new_profile = trial_vec

    return {
        "target_dim": target_dim,
        "recommended_strain": best_choice,
        "expected_improvement": best_delta,
        "new_profile": best_new_profile,
        "current_score": base_score,
    }

# -------------------------------------------------
# VIS
# -------------------------------------------------

def plot_radar(aroma_profile, title="Profile"):
    dims = list(aroma_profile.keys())
    vals = [aroma_profile[d] for d in dims]

    dims.append(dims[0])
    vals.append(vals[0])

    angles = np.linspace(0, 2 * np.pi, len(dims), endpoint=False)

    fig = plt.figure(figsize=(5, 5))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, vals, marker="o")
    ax.fill(angles, vals, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dims[:-1], fontsize=8)
    ax.set_title(title)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    return fig

# -------------------------------------------------
# GENERATIVE ADVISOR (Azure placeholder)
# -------------------------------------------------

def generate_brewmaster_notes(hop_prof, malt_prof, yeast_prof, brewer_goal):
    """
    This is the slot where Azure OpenAI will go.
    For now we just stitch together a narrative.
    """
    lines = []
    lines.append("Brewmaster Notes (Prototype)")
    lines.append("")
    lines.append("Your stated goal:")
    lines.append(f"- {brewer_goal}")
    lines.append("")
    lines.append("Current Hop Expression:")
    for k, v in hop_prof.items():
        lines.append(f"  {k}: {v:.2f}")
    lines.append("")
    lines.append("Current Malt / Body / Color:")
    for k, v in malt_prof.items():
        lines.append(f"  {k}: {v:.2f}")
    lines.append("")
    lines.append("Current Yeast / Fermentation Character:")
    for k, v in yeast_prof.items():
        lines.append(f"  {k}: {v:.2f}")
    lines.append("")
    lines.append("High-level guidance:")
    lines.append("- Increase late-addition hops that boost your target ester/aroma.")
    lines.append("- Tune malt % to adjust mid-palate sweetness and perceived body without overshooting color.")
    lines.append("- Pick a yeast strain / temp that reinforces the fruit profile you actually want.")
    lines.append("")
    lines.append("This section will be AI-generated by Azure (process tweaks, timing, etc.).")

    return "\n".join(lines)

# -------------------------------------------------
# STREAMLIT APP
# -------------------------------------------------

st.set_page_config(page_title="Beer Recipe Digital Twin", page_icon="🍺", layout="centered")

st.title("🍺 Beer Recipe Digital Twin")
st.markdown("""
1. Enter your hop bill (variety + grams).
2. The model predicts flavor dimensions like citrus, pine, tropical fruit.
3. Pick what you want more of.
4. It recommends a single hop addition (and grams) to steer the beer.
""")
st.markdown("---")

# ---------------------------
# HOPS SECTION
# ---------------------------
st.header("🌿 Hops: Aroma + Hop Addition Advisor")

c1, c2, c3 = st.columns([1,1,1])
with c1:
    hop1 = st.text_input("Hop 1 name", "Citra")
    hop2 = st.text_input("Hop 2 name", "Mosaic")
with c2:
    hop1_amt = st.number_input("Hop 1 (g)", min_value=0.0, max_value=500.0, value=50.0, step=5.0)
    hop2_amt = st.number_input("Hop 2 (g)", min_value=0.0, max_value=500.0, value=30.0, step=5.0)
with c3:
    st.write("We'll predict the aroma balance of this hop bill, then tell you how to push it.")

user_hops = []
if hop1 and hop1_amt > 0:
    user_hops.append({"name": hop1, "amt": hop1_amt})
if hop2 and hop2_amt > 0:
    user_hops.append({"name": hop2, "amt": hop2_amt})

st.markdown("")
hop_predict_clicked = st.button("🔍 Predict Hop Aroma")

hop_profile = {}
hop_advice = None

if hop_predict_clicked and user_hops:
    hop_profile = predict_hop_profile(user_hops)
    st.subheader("Predicted Hop Aroma Profile")
    st.json(hop_profile)

    fig_hops = plot_radar(hop_profile, title="Current Hop Bill")
    st.pyplot(fig_hops)

    st.markdown("### 🎯 Hop Adjustment Advisor")
    hop_target = st.selectbox("Which hop aroma do you want more of?", hop_dims)
    trial_amt = st.slider("Simulate late-addition / whirlpool hop (g):", 5, 60, 20, 5)

    hop_advise_clicked = st.button("🧠 Advise Hop Addition")

    if hop_advise_clicked:
        hop_advice = advise_hops(user_hops, target_dim=hop_target, trial_amt=trial_amt)
        st.success(
            f"To boost **{hop_advice['target_dim']}**, "
            f"add {hop_advice['addition_grams']} g of **{hop_advice['recommended_hop']}**.\n\n"
            f"Expected improvement: +{hop_advice['expected_improvement']:.3f}"
        )
        st.subheader("New projected hop aroma after that change")
        st.json(hop_advice["new_profile"])

        fig_hops_new = plot_radar(hop_advice["new_profile"], title="Revised Hop Bill")
        st.pyplot(fig_hops_new)

st.markdown("---")

# ---------------------------
# MALT SECTION
# ---------------------------
with st.expander("🌾 Malt / Grain Bill: Body, Sweetness, Color Advisor", expanded=False):
    m1, m2 = st.columns([1,1])
    with m1:
        malt1 = st.text_input("Malt 1 name", "Maris Otter")
        malt2 = st.text_input("Malt 2 name", "Caramunich III")
    with m2:
        malt1_pct = st.number_input("Malt 1 (% grist)", min_value=0.0, max_value=100.0, value=70.0, step=1.0)
        malt2_pct = st.number_input("Malt 2 (% grist)", min_value=0.0, max_value=100.0, value=8.0, step=1.0)

    user_malts = []
    if malt1 and malt1_pct > 0:
        user_malts.append({"name": malt1, "pct": malt1_pct})
    if malt2 and malt2_pct > 0:
        user_malts.append({"name": malt2, "pct": malt2_pct})

    malt_predict_clicked = st.button("🔍 Predict Malt Profile")

    malt_profile = {}
    malt_advice = None

    if malt_predict_clicked and user_malts:
        malt_profile = predict_malt_profile(user_malts)
        st.subheader("Predicted Malt Profile / Body / Color")
        st.json(malt_profile)

        fig_malt = plot_radar(malt_profile, title="Malt Body / Sweetness / Color")
        st.pyplot(fig_malt)

        st.markdown("### 🍞 Malt Adjustment Advisor")
        malt_target = st.selectbox(
            "What malt dimension do you want to increase? (body, caramel, sweetness, etc.)",
            malt_dims,
        )
        trial_pct = st.slider("Simulate adding (+% of grist):", 1, 10, 2, 1)

        malt_advise_clicked = st.button("🧠 Advise Malt Change")

        if malt_advise_clicked:
            malt_advice = advise_malt(user_malts, target_dim=malt_target, trial_pct=trial_pct)
            st.success(
                f"To boost **{malt_advice['target_dim']}**, "
                f"add about {malt_advice['addition_pct']}% of **{malt_advice['recommended_malt']}** "
                f"to the grist. Expected change: +{malt_advice['expected_improvement']:.3f}"
            )
            st.subheader("New projected malt profile after that change")
            st.json(malt_advice["new_profile"])

            fig_malt_new = plot_radar(malt_advice["new_profile"], title="Revised Malt Bill")
            st.pyplot(fig_malt_new)

st.markdown("---")

# ---------------------------
# YEAST / FERMENTATION SECTION
# ---------------------------

with st.expander("🧫 Yeast & Fermentation: Ester / Mouthfeel Advisor", expanded=False):
    y1, y2 = st.columns([1,1])
    with y1:
        yeast_strain = st.text_input("Yeast strain", "London Ale III")
    with y2:
        ferm_temp = st.number_input("Fermentation temp (°F)", min_value=60, max_value=80, value=68, step=1)

    user_yeast = {
        "strain": yeast_strain,
        "ferm_temp_f": ferm_temp,
    }

    yeast_predict_clicked = st.button("🔍 Predict Fermentation Profile")

    yeast_profile = {}
    yeast_advice = None

    if yeast_predict_clicked and yeast_strain:
        yeast_profile = predict_yeast_profile(user_yeast)
        st.subheader("Predicted Fermentation / Yeast Profile")
        st.json(yeast_profile)

        fig_yeast = plot_radar(yeast_profile, title="Yeast-Driven Sensory / Mouthfeel")
        st.pyplot(fig_yeast)

        st.markdown("### 🧪 Yeast Adjustment Advisor")
        yeast_target = st.selectbox(
            "Which direction do you want to push? (more stone fruit esters? cleaner? drier?)",
            yeast_dims,
        )

        yeast_advise_clicked = st.button("🧠 Advise Fermentation Change")

        if yeast_advise_clicked:
            yeast_advice = advise_yeast(user_yeast, target_dim=yeast_target)
            st.success(
                f"To boost **{yeast_advice['target_dim']}**, "
                f"switch to **{yeast_advice['recommended_strain']}**. "
                f"Expected improvement: +{yeast_advice['expected_improvement']:.3f}"
            )

            st.subheader("New projected fermentation profile after that change")
            st.json(yeast_advice["new_profile"])

            fig_yeast_new = plot_radar(yeast_advice["new_profile"], title="Revised Fermentation Plan")
            st.pyplot(fig_yeast_new)

st.markdown("---")

# ---------------------------
# AZURE "HEAD BREWER" ADVISOR (stub for now)
# ---------------------------

st.header("👨‍🔬 Brewmaster Notes (AI Co-Brewer)")
brewer_goal = st.text_area(
    "What's your intent for this beer? (e.g. 'Soft hazy IPA with saturated stone fruit and pineapple, low bitterness, pillowy mouthfeel')",
    "",
)

generate_notes_clicked = st.button("🗣 Generate Brewmaster Notes")

if generate_notes_clicked:
    # if user hasn't run sections above, we'll just feed empty dicts
    hop_prof_for_notes = hop_profile if hop_profile else {}
    malt_prof_for_notes = malt_profile if 'malt_profile' in locals() and malt_profile else {}
    yeast_prof_for_notes = yeast_profile if 'yeast_profile' in locals() and yeast_profile else {}

    notes = generate_brewmaster_notes(
        hop_prof_for_notes,
        malt_prof_for_notes,
        yeast_prof_for_notes,
        brewer_goal,
    )

    st.subheader("Prototype Brewmaster Guidance")
    st.code(notes, language="text")

    st.info("This block will soon come from Azure OpenAI, using your hop/malt/yeast predictions + your target style.")

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
# TEXT NORMALIZATION / MATCHING HELPERS
# -------------------------------------------------

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
    Given e.g. "Citra", find best model column like "hop_Citra®".
    Only consider columns that start with that prefix.
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
    Generate nice human-readable dropdown options from model feature columns.

    - If preferred_prefix is provided (e.g. 'malt_', 'yeast_'), we FIRST try to
      include only columns starting with that prefix.
    - If that yields nothing, we fall back to using all columns.

    We also clean/truncate prefixes like 'malt_', 'grain_', 'yeast_', etc.,
    remove ™/®, and replace '_' with space.
    """

    def prettify(label: str) -> str:
        label = label.replace("®", "").replace("™", "")
        label = label.replace("_", " ").strip()
        return label

    subset = []

    # Pass 1: try preferred_prefix
    if preferred_prefix:
        for col in feature_cols:
            if col.startswith(preferred_prefix):
                raw_label = col[len(preferred_prefix):]
                subset.append(prettify(raw_label))

    # Pass 2: if nothing with preferred_prefix, fall back to everything
    if not subset:
        for col in feature_cols:
            cand = col
            # strip common prefixes if present
            for p in ["hop_", "malt_", "grain_", "base_", "yeast_", "strain_", "y_", "m_"]:
                if cand.startswith(p):
                    cand = cand[len(p):]
            subset.append(prettify(cand))

    # Unique + sorted
    cleaned = []
    for name in subset:
        if name and name not in cleaned:
            cleaned.append(name)

    cleaned = sorted(cleaned, key=lambda s: s.lower())
    return cleaned


# -------------------------------------------------
# LOAD MODELS
# -------------------------------------------------

ROOT_DIR = os.path.dirname(__file__)

# --- Hop model bundle ---
HOP_MODEL_PATH = os.path.join(ROOT_DIR, "hop_aroma_model.joblib")
hop_bundle = joblib.load(HOP_MODEL_PATH)
hop_model = hop_bundle["model"]
hop_feature_cols = hop_bundle["feature_cols"]          # e.g. ['hop_Citra®', 'hop_Mosaic', ...]
hop_dims = [
    a for a in hop_bundle["aroma_dims"]
    if str(a).lower() not in ("nan", "", "none")
]

# --- Malt model bundle ---
# keys present: "model", "feature_cols", "flavor_cols"
MALT_MODEL_PATH = os.path.join(ROOT_DIR, "malt_sensory_model.joblib")
malt_bundle = joblib.load(MALT_MODEL_PATH)
malt_model = malt_bundle["model"]
malt_feature_cols = malt_bundle["feature_cols"]        # e.g. ['malt_Maris Otter', 'grain_Caramunich III', ...]
malt_dims = malt_bundle["flavor_cols"]                 # e.g. ['body', 'sweetness', 'color_srm', ...]

# --- Yeast model bundle ---
# keys present: "model", "feature_cols", "flavor_cols"
YEAST_MODEL_PATH = os.path.join(ROOT_DIR, "yeast_sensory_model.joblib")
yeast_bundle = joblib.load(YEAST_MODEL_PATH)
yeast_model = yeast_bundle["model"]
yeast_feature_cols = yeast_bundle["feature_cols"]      # e.g. ['yeast_London Ale III', ...]
yeast_dims = yeast_bundle["flavor_cols"]               # e.g. ['stone_fruit_ester', 'attenuation', ...]

# Build dropdown lists from each model's known features,
# using preferred prefixes when we can.
HOP_CHOICES = _choices_from_features(hop_feature_cols, preferred_prefix="hop_")
MALT_CHOICES = _choices_from_features(malt_feature_cols, preferred_prefix="malt_")
YEAST_CHOICES = _choices_from_features(yeast_feature_cols, preferred_prefix="yeast_")

# Debug info in sidebar so we can inspect what's really in the joblibs on Cloud
with st.sidebar:
    st.header("🔬 Debug model vocab")
    st.write("malt_feature_cols[:10] =", malt_feature_cols[:10])
    st.write("yeast_feature_cols[:10] =", yeast_feature_cols[:10])
    st.write("MALT_CHOICES[:10] =", MALT_CHOICES[:10])
    st.write("YEAST_CHOICES[:10] =", YEAST_CHOICES[:10])


# -------------------------------------------------
# HOP FUNCTIONS
# -------------------------------------------------

def build_hop_features(user_hops):
    """
    user_hops: [ {"name": "Citra", "amt": 50}, {"name": "Mosaic", "amt": 30}, ... ]
    amt in grams.
    Returns a (1 x n_features) DataFrame aligned to hop_feature_cols.
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


def advise_hops(user_hops, target_dim, trial_amt=20.0):
    """
    Try adding `trial_amt` grams of each possible hop and see which one
    improves `target_dim` the most.
    """
    base_vec = predict_hop_profile(user_hops)
    base_score = base_vec.get(target_dim, 0.0)

    best_choice = None
    best_delta = -999.0
    best_new_profile = None

    for col in hop_feature_cols:
        if not col.startswith("hop_"):
            continue
        candidate_label = col[len("hop_"):]  # "Citra®"

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
    pct is % of grist (0..100).
    Returns (1 x n_features) DataFrame aligned to malt_feature_cols.
    """
    totals = {c: 0.0 for c in malt_feature_cols}

    for entry in user_malts:
        nm = entry.get("name", "")
        pct = float(entry.get("pct", 0.0))
        if pct <= 0 or not nm or str(nm).strip() in ["", "-"]:
            continue

        # We'll *try* to match with prefix "malt_".
        # If no match, loosely fall back to first best match in ANY column.
        match = _best_feature_match(nm, malt_feature_cols, prefix="malt_")
        if match is None:
            # fallback pass: try with "grain_" prefix, etc.
            for pfx in ["grain_", "base_", "malt_", "m_"]:
                match = _best_feature_match(nm, malt_feature_cols, prefix=pfx)
                if match:
                    break

        if match:
            totals[match] += pct

    return pd.DataFrame([totals], columns=malt_feature_cols)


def predict_malt_profile(user_malts):
    """
    Return dict { malt_dimension -> score }.
    e.g. {'body': 0.78, 'sweetness': 0.62, 'color_srm': 14.2, ...}
    """
    X = build_malt_features(user_malts)
    y_pred = malt_model.predict(X)[0]
    return {dim: float(val) for dim, val in zip(malt_dims, y_pred)}


def advise_malt(user_malts, target_dim, trial_pct=2.0):
    """
    Try adding trial_pct% of each malt and see which improves target_dim most.
    """
    base_vec = predict_malt_profile(user_malts)
    base_score = base_vec.get(target_dim, 0.0)

    best_choice = None
    best_delta = -999.0
    best_new_profile = None

    for col in malt_feature_cols:
        # best label to show user:
        cand_label = col
        for p in ["malt_", "grain_", "base_", "m_"]:
            if cand_label.startswith(p):
                cand_label = cand_label[len(p):]
        cand_label = cand_label.replace("®", "").replace("™", "").replace("_", " ").strip()

        trial_bill = user_malts + [{"name": cand_label, "pct": trial_pct}]
        trial_vec = predict_malt_profile(trial_bill)
        trial_score = trial_vec.get(target_dim, 0.0)
        delta = trial_score - base_score

        if delta > best_delta:
            best_delta = delta
            best_choice = cand_label
            best_new_profile = trial_vec

    return {
        "target_dim": target_dim,
        "addition_pct": trial_pct,
        "recommended_malt": best_choice,
        "expected_improvement": best_delta,
        "new_profile": best_new_profile,
        "current_score": base_score,
    }


# -------------------------------------------------
# YEAST / FERMENTATION FUNCTIONS
# -------------------------------------------------

def build_yeast_features(user_yeast):
    """
    user_yeast:
      {
        "strain": "London Ale III",
        "ferm_temp_f": 68
      }

    We'll fuzzy match the strain to yeast_feature_cols.
    We'll try 'yeast_' prefix first, then fallback prefixes.
    We'll set that matching col = 1.0.
    (We are NOT yet encoding fermentation temp in features here.)
    """
    totals = {c: 0.0 for c in yeast_feature_cols}

    strain = user_yeast.get("strain", "")
    match = _best_feature_match(strain, yeast_feature_cols, prefix="yeast_")
    if match is None:
        for pfx in ["strain_", "y_", "yeast_", ""]:
            m2 = _best_feature_match(strain, yeast_feature_cols, prefix=pfx) if pfx else None
            if m2:
                match = m2
                break

    if match:
        totals[match] = 1.0

    return pd.DataFrame([totals], columns=yeast_feature_cols)


def predict_yeast_profile(user_yeast):
    """
    Return dict { yeast_dim -> score }.
    e.g. {'stone_fruit_ester': 0.8, 'attenuation': 0.6, 'phenolic': 0.1, ...}
    """
    X = build_yeast_features(user_yeast)
    y_pred = yeast_model.predict(X)[0]
    return {dim: float(val) for dim, val in zip(yeast_dims, y_pred)}


def advise_yeast(user_yeast, target_dim):
    """
    Try swapping to each possible yeast strain column, pick the one
    that best improves target_dim.
    """
    base_vec = predict_yeast_profile(user_yeast)
    base_score = base_vec.get(target_dim, 0.0)

    best_choice = None
    best_delta = -999.0
    best_new_profile = None

    for col in yeast_feature_cols:
        cand_label = col
        for p in ["yeast_", "strain_", "y_"]:
            if cand_label.startswith(p):
                cand_label = cand_label[len(p):]
        cand_label = cand_label.replace("®", "").replace("™", "").replace("_", " ").strip()

        trial_ferm = {
            "strain": cand_label,
            "ferm_temp_f": user_yeast.get("ferm_temp_f", 68),
        }
        trial_vec = predict_yeast_profile(trial_ferm)
        trial_score = trial_vec.get(target_dim, 0.0)
        delta = trial_score - base_score

        if delta > best_delta:
            best_delta = delta
            best_choice = cand_label
            best_new_profile = trial_vec

    return {
        "target_dim": target_dim,
        "recommended_strain": best_choice,
        "expected_improvement": best_delta,
        "new_profile": best_new_profile,
        "current_score": base_score,
    }


# -------------------------------------------------
# RADAR PLOT
# -------------------------------------------------

def plot_radar(profile_dict, title="Profile"):
    """
    Radar plot for a {dimension: value} mapping.
    Assumes values are ~0..1-ish for most dims.
    """
    if not profile_dict:
        fig = plt.figure(figsize=(4,4))
        ax = plt.subplot(111)
        ax.text(0.5, 0.5, "no data", ha="center", va="center")
        ax.set_axis_off()
        return fig

    dims = list(profile_dict.keys())
    vals = [profile_dict[d] for d in dims]

    # close polygon
    dims.append(dims[0])
    vals.append(vals[0])

    angles = np.linspace(0, 2 * np.pi, len(dims), endpoint=False)

    fig = plt.figure(figsize=(5,5))
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
# BREWMASTER / AZURE STUB
# -------------------------------------------------

def generate_brewmaster_notes(hop_prof, malt_prof, yeast_prof, brewer_goal):
    """
    Placeholder for Azure OpenAI integration.
    We stitch together predictions + your stated goal.
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
    lines.append("- Tune late-addition hops to push target aromatics.")
    lines.append("- Adjust grist % for sweetness / body / color without overshooting style.")
    lines.append("- Pick a yeast strain & temp schedule that reinforces the ester profile you want.")
    lines.append("")
    lines.append("Soon this will become Azure-generated process/timing advice.")
    return "\n".join(lines)


# -------------------------------------------------
# STREAMLIT APP LAYOUT
# -------------------------------------------------

st.set_page_config(page_title="Beer Recipe Digital Twin", page_icon="🍺", layout="centered")

st.title("🍺 Beer Recipe Digital Twin")
st.markdown("""
Your AI brew assistant.
1. Predict hop aroma.
2. Predict malt body / sweetness / color.
3. Predict fermentation profile (esters, mouthfeel).
4. Get targeted change suggestions.
5. (Soon) Get full brewmaster guidance from Azure.
""")
st.markdown("---")

# ---------------------------
# HOPS SECTION
# ---------------------------

st.header("🌿 Hops: Aroma + Hop Addition Advisor")

c1, c2, c3 = st.columns([1,1,1])

with c1:
    default_hop1 = HOP_CHOICES[0] if HOP_CHOICES else ""
    default_hop2 = HOP_CHOICES[1] if len(HOP_CHOICES) > 1 else default_hop1

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
    hop1_amt = st.number_input("Hop 1 (g)", min_value=0.0, max_value=500.0, value=50.0, step=5.0)
    hop2_amt = st.number_input("Hop 2 (g)", min_value=0.0, max_value=500.0, value=30.0, step=5.0)

with c3:
    st.write("We'll predict the aroma balance of this hop bill, then tell you how to push it.")

user_hops = []
if hop1 and hop1_amt > 0:
    user_hops.append({"name": hop1, "amt": hop1_amt})
if hop2 and hop2_amt > 0:
    user_hops.append({"name": hop2, "amt": hop2_amt})

hop_profile = {}
hop_advice = None

hop_predict_clicked = st.button("🔍 Predict Hop Aroma")
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
        malt1 = st.selectbox(
            "Malt 1 name",
            MALT_CHOICES,
            index=MALT_CHOICES.index("Maris Otter") if "Maris Otter" in MALT_CHOICES else 0,
            key="malt1_select",
        ) if MALT_CHOICES else ""

        malt2 = st.selectbox(
            "Malt 2 name",
            MALT_CHOICES,
            index=MALT_CHOICES.index("Caramunich III") if "Caramunich III" in MALT_CHOICES else 0,
            key="malt2_select",
        ) if MALT_CHOICES else ""

    with m2:
        malt1_pct = st.number_input("Malt 1 (% grist)", min_value=0.0, max_value=100.0, value=70.0, step=1.0)
        malt2_pct = st.number_input("Malt 2 (% grist)", min_value=0.0, max_value=100.0, value=8.0, step=1.0)

    user_malts = []
    if malt1 and malt1_pct > 0:
        user_malts.append({"name": malt1, "pct": malt1_pct})
    if malt2 and malt2_pct > 0:
        user_malts.append({"name": malt2, "pct": malt2_pct})

    malt_profile = {}
    malt_advice = None

    malt_predict_clicked = st.button("🔍 Predict Malt Profile")
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
# YEAST SECTION
# ---------------------------

with st.expander("🧫 Yeast & Fermentation: Ester / Mouthfeel Advisor", expanded=False):
    y1, y2 = st.columns([1,1])

    with y1:
        yeast_strain = st.selectbox(
            "Yeast strain",
            YEAST_CHOICES,
            index=YEAST_CHOICES.index("London Ale III") if "London Ale III" in YEAST_CHOICES else 0,
            key="yeast_select",
        ) if YEAST_CHOICES else ""

    with y2:
        ferm_temp = st.number_input("Fermentation temp (°F)", min_value=60, max_value=80, value=68, step=1)

    user_yeast = {
        "strain": yeast_strain,
        "ferm_temp_f": ferm_temp,
    }

    yeast_profile = {}
    yeast_advice = None

    yeast_predict_clicked = st.button("🔍 Predict Fermentation Profile")
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
# BREWMASTER NOTES (AZURE STUB)
# ---------------------------

st.header("👨‍🔬 Brewmaster Notes (AI Co-Brewer)")
brewer_goal = st.text_area(
    "What's your intent for this beer? (e.g. 'Soft hazy IPA with saturated stone fruit and pineapple, low bitterness, pillowy mouthfeel')",
    "",
)

generate_notes_clicked = st.button("🗣 Generate Brewmaster Notes")
if generate_notes_clicked:
    hop_prof_for_notes = hop_profile if hop_profile else {}
    malt_prof_for_notes = locals().get("malt_profile", {}) or {}
    yeast_prof_for_notes = locals().get("yeast_profile", {}) or {}

    notes = generate_brewmaster_notes(
        hop_prof_for_notes,
        malt_prof_for_notes,
        yeast_prof_for_notes,
        brewer_goal,
    )

    st.subheader("Prototype Brewmaster Guidance")
    st.code(notes, language="text")

    st.info("Soon this will be Azure OpenAI output using your hop/malt/yeast predictions + style target.")

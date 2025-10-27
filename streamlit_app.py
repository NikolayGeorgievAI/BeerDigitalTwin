import os
import re
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from openai import OpenAI

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# =========================================================
# Utility: text cleanup / fuzzy-ish matching
# =========================================================

def _clean_name(name: str) -> str:
    """Lowercase, strip special chars, keep alnum only."""
    if name is None:
        return ""
    s = str(name).lower()
    s = s.replace("®", "").replace("™", "")
    s = re.sub(r"[^a-z0-9]", "", s)
    return s


def _best_feature_match(user_name: str, feature_cols: list, prefix: str):
    """
    Given user_name (e.g. 'Citra'), find best col in feature_cols starting with the prefix
    (e.g. 'hop_').
    We'll compare by character overlap after cleaning.
    """
    cleaned_user = _clean_name(user_name)
    best_match = None
    best_score = -1

    for col in feature_cols:
        if not col.startswith(prefix):
            continue

        # remove prefix visually to compare core name
        raw_label = col[len(prefix):]
        cleaned_label = _clean_name(raw_label)

        # quick gate: if no overlap at all, skip
        if len(cleaned_user) >= 3 and cleaned_user[:3] not in cleaned_label:
            # still let partial fuzzy chance below, but mostly skip
            pass

        common = set(cleaned_user) & set(cleaned_label)
        score = len(common)
        if score > best_score:
            best_score = score
            best_match = col

    return best_match


def _choices_from_features(feature_cols, preferred_prefix=None):
    """
    Build nice dropdown labels from model feature columns.
    We try first to keep only columns starting with preferred_prefix (like 'hop_').
    If that's empty, we fall back to all columns.
    We strip common prefixes and cleanup.
    """

    def prettify(label: str) -> str:
        label = label.replace("®", "").replace("™", "")
        label = label.replace("_", " ").strip()
        return label

    subset = []

    # Pass 1: prefix
    if preferred_prefix:
        for col in feature_cols:
            if col.startswith(preferred_prefix):
                raw = col[len(preferred_prefix):]
                subset.append(prettify(raw))

    # Pass 2: fallback all
    if not subset:
        for col in feature_cols:
            cand = col
            for p in ["hop_", "malt_", "grain_", "base_", "yeast_", "strain_", "y_", "m_"]:
                if cand.startswith(p):
                    cand = cand[len(p):]
            subset.append(prettify(cand))

    # dedupe, sort
    cleaned = []
    for nm in subset:
        if nm and nm not in cleaned:
            cleaned.append(nm)

    cleaned = sorted(cleaned, key=lambda s: s.lower())
    return cleaned


# =========================================================
# Load models (expect .joblib files to be in same dir)
# =========================================================

ROOT_DIR = os.path.dirname(__file__)

# --- hop model
HOP_MODEL_PATH = os.path.join(ROOT_DIR, "hop_aroma_model.joblib")
hop_bundle = joblib.load(HOP_MODEL_PATH)
hop_model = hop_bundle["model"]
hop_feature_cols = hop_bundle["feature_cols"]        # e.g. ['hop_Citra®', 'hop_Mosaic', ...]
hop_dims = [
    a for a in hop_bundle.get("aroma_dims", [])
    if str(a).strip().lower() not in ("", "nan", "none")
]

# --- malt model
MALT_MODEL_PATH = os.path.join(ROOT_DIR, "malt_sensory_model.joblib")
malt_bundle = joblib.load(MALT_MODEL_PATH)
malt_model = malt_bundle["model"]
malt_feature_cols = malt_bundle["feature_cols"]
malt_dims = malt_bundle["flavor_cols"]  # e.g. ['bready','caramel','nutty',...]

# --- yeast model
YEAST_MODEL_PATH = os.path.join(ROOT_DIR, "yeast_sensory_model.joblib")
yeast_bundle = joblib.load(YEAST_MODEL_PATH)
yeast_model = yeast_bundle["model"]
yeast_feature_cols = yeast_bundle["feature_cols"]
yeast_dims = yeast_bundle["flavor_cols"]


# Build the dropdown vocab for each
HOP_CHOICES = _choices_from_features(hop_feature_cols, preferred_prefix="hop_")
MALT_CHOICES = _choices_from_features(malt_feature_cols, preferred_prefix="malt_")
YEAST_CHOICES = _choices_from_features(yeast_feature_cols, preferred_prefix="yeast_")


# =========================================================
# Feature builders + predictors
# =========================================================

def build_hop_features(user_hops):
    """
    user_hops: [ {"name":"Citra", "amt":50}, ... ]
    Return a DataFrame with same columns hop_feature_cols
    Values = grams per hop col (0 if not used)
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
    Return dict {hop_dim -> score}
    """
    X = build_hop_features(user_hops)
    y_pred = hop_model.predict(X)[0]
    return {dim: float(val) for dim, val in zip(hop_dims, y_pred)}


def build_malt_features(user_malts):
    """
    user_malts: [{"name":"Maris Otter","pct":70}, ...]
    We'll do fuzzy match across multiple prefixes typical in training.
    """
    totals = {c: 0.0 for c in malt_feature_cols}

    for entry in user_malts:
        nm = entry.get("name","")
        pct = float(entry.get("pct",0.0))
        if pct <= 0 or not nm or str(nm).strip() in ["","-"]:
            continue

        # attempt prefix matches in order
        prefixes = ["malt_", "grain_", "base_", "m_"]
        match = None
        for pfx in prefixes:
            match = _best_feature_match(nm, malt_feature_cols, prefix=pfx)
            if match:
                break

        if match:
            totals[match] += pct

    return pd.DataFrame([totals], columns=malt_feature_cols)


def predict_malt_profile(user_malts):
    """
    Return dict { malt_dimension -> score }
    """
    X = build_malt_features(user_malts)
    y_pred = malt_model.predict(X)[0]
    return {dim: float(val) for dim, val in zip(malt_dims, y_pred)}


def build_yeast_features(user_yeast):
    """
    user_yeast: {"strain":"London Ale III","ferm_temp_c":20}
    We match strain to yeast_feature_cols.
    We'll only encode strain as a 1.0 in its column.
    We do NOT currently feed temperature as a model input.
    """
    totals = {c: 0.0 for c in yeast_feature_cols}

    strain = user_yeast.get("strain", "")
    match = _best_feature_match(strain, yeast_feature_cols, prefix="yeast_")
    if match is None:
        for pfx in ["strain_", "y_", "yeast_", ""]:
            if pfx == "":
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
    Return dict { yeast_dim -> score }
    """
    X = build_yeast_features(user_yeast)
    y_pred = yeast_model.predict(X)[0]
    return {dim: float(val) for dim, val in zip(yeast_dims, y_pred)}


# =========================================================
# Radar plot utilities (shared scale, minimal look)
# =========================================================

def normalize_profile_dict(profile_dict, dim_list, global_min=0.0, global_max=1.0):
    """
    Take a dict of {dim:value} and:
    - ensure all dims in dim_list are present (missing -> 0.0)
    - rescale each value to [0,1] based on global_min/global_max
    We'll clip to [0,1].
    Returns a list of floats in the same order as dim_list.
    """
    vals = []
    for d in dim_list:
        raw = float(profile_dict.get(d, 0.0))
        if global_max == global_min:
            scaled = 0.0
        else:
            scaled = (raw - global_min) / (global_max - global_min)
        scaled = max(0.0, min(1.0, scaled))
        vals.append(scaled)
    return vals


def plot_radar_sharedscale(dim_list, values_0to1, title="Profile"):
    """
    Draw a radar/spider chart on a 0..1 shared scale.
    We hide radial ticks, radial labels, numeric gridlines;
    we only show the polygon + fill + axis labels + title.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # close shape
    vals_closed = list(values_0to1) + [values_0to1[0]]
    angles = np.linspace(0, 2*np.pi, len(dim_list), endpoint=False)
    angles_closed = list(angles) + [angles[0]]

    fig = plt.figure(figsize=(4,4))
    ax = plt.subplot(111, polar=True)

    # polygon
    ax.plot(angles_closed, vals_closed, marker="o", linewidth=1.4, color="#1f77b4")
    ax.fill(angles_closed, vals_closed, color="#1f77b4", alpha=0.25)

    # label each axis
    ax.set_xticks(angles)
    ax.set_xticklabels(dim_list, fontsize=8, color="#333")

    # hide radial ticks / labels
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.grid(False)
    ax.spines["polar"].set_visible(False)

    ax.set_title(title, fontsize=11, pad=10)
    plt.tight_layout()
    return fig


# =========================================================
# Azure OpenAI Brewmaster Notes
# =========================================================

def get_azure_client():
    """
    Create an OpenAI client for Azure using env vars set as Streamlit secrets.
    We'll read from st.secrets to avoid printing keys.
    """
    endpoint = st.secrets.get("AZURE_OPENAI_ENDPOINT", "")
    api_key = st.secrets.get("AZURE_OPENAI_API_KEY", "")
    deployment = st.secrets.get("AZURE_OPENAI_DEPLOYMENT", "")

    if not endpoint or not api_key or not deployment:
        return None, None

    # Create a client configured for Azure
    client = OpenAI(
        api_key=api_key,
        base_url=f"{endpoint}/openai/deployments/{deployment}",
        default_headers={"api-key": api_key},
    )
    return client, deployment


def call_azure_brewmaster_notes(
    hop_vec,
    malt_vec,
    yeast_vec,
    brewer_goal,
):
    """
    Ask Azure OpenAI to act like a brewmaster and return bullet guidance.
    We prompt with the predicted profiles + the user's desired target outcome.
    """
    client, deployment = get_azure_client()
    if client is None:
        return (
            "Brewmaster Notes\n"
            "- (Azure OpenAI not configured; showing placeholder.)\n"
            "• Tune hops toward juicy, tropical, stone-fruit varieties.\n"
            "• Add oats / wheat for pillowy mouthfeel.\n"
            "• Choose a yeast that enhances esters without going solventy.\n"
        )

    # We'll summarize numeric vectors into short bullet text
    def summarize_block(name, vec):
        lines = [f"{k}: {round(v,3)}" for k, v in vec.items()]
        joined = "; ".join(lines)
        return f"{name}: {joined}"

    hop_block = summarize_block("Hop profile", hop_vec)
    malt_block = summarize_block("Malt/body profile", malt_vec)
    yeast_block = summarize_block("Yeast/fermentation profile", yeast_vec)

    system_msg = (
        "You are an experienced craft brewmaster. "
        "You give practical, concise brewing guidance to adjust aroma, body, color, and mouthfeel. "
        "Focus on sensory results, not lab safety or liability."
    )

    user_msg = (
        f"My goal for this beer is:\n"
        f"  {brewer_goal}\n\n"
        f"Here are the predicted sensory profiles:\n"
        f"- {hop_block}\n"
        f"- {malt_block}\n"
        f"- {yeast_block}\n\n"
        "Please provide:\n"
        "1. Overall read (1–2 sentences)\n"
        "2. Hop adjustments (bullet points)\n"
        "3. Malt/grist tweaks (bullet points)\n"
        "4. Fermentation guidance (bullet points)\n"
        "5. A final short summary for a pro brewer.\n"
        "Keep it direct. Avoid disclaimers or legal language. "
        "No extra commentary about being AI."
    )

    try:
        completion = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.5,
            max_tokens=600,
        )
    except Exception as e:
        return (
            "Brewmaster Notes\n"
            "• (Azure OpenAI request failed. Here's a fallback idea.)\n"
            f"• Hop toward tropical/stone-fruit; keep bitterness low.\n"
            f"• Add oats/wheat for pillowy mouthfeel.\n"
            f"• Choose a moderately attenuating ester-friendly yeast.\n"
            f"(Error was: {e})"
        )

    # Safely parse response
    ai_text = ""
    if hasattr(completion, "choices") and len(completion.choices) > 0:
        choice0 = completion.choices[0]
        if hasattr(choice0, "message") and "content" in choice0.message:
            ai_text = choice0.message["content"]

    if not ai_text:
        ai_text = (
            "Brewmaster Notes\n"
            "• Hop: lean into juicy tropical fruit. "
            "• Malt: add oats/wheat for softness.\n"
            "• Yeast: select soft ester strain; keep temp moderate."
        )

    return ai_text


# =========================================================
# Streamlit App Layout
# =========================================================

st.set_page_config(
    page_title="Beer Recipe Digital Twin",
    page_icon="🍺",
    layout="centered"
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

# ---------------------------
# HOPS SECTION
# ---------------------------
st.subheader("🌿 Hops (late/aroma additions)")

col_h1, col_h2 = st.columns([1,1])
with col_h1:
    hop1_choice = st.selectbox(
        "Main Hop Variety",
        HOP_CHOICES,
        index=HOP_CHOICES.index("Mosaic") if "Mosaic" in HOP_CHOICES else 0,
        key="hop1_choice",
    )
    hop2_choice = st.selectbox(
        "Secondary Hop Variety",
        HOP_CHOICES,
        index=HOP_CHOICES.index("Citra") if "Citra" in HOP_CHOICES else 0,
        key="hop2_choice",
    )

with col_h2:
    hop1_amt = st.number_input(
        "Hop 1 amount (g)",
        min_value=0.0, max_value=500.0,
        value=30.0, step=5.0
    )
    hop2_amt = st.number_input(
        "Hop 2 amount (g)",
        min_value=0.0, max_value=500.0,
        value=20.0, step=5.0
    )

user_hops = []
if hop1_choice and hop1_amt>0:
    user_hops.append({"name":hop1_choice,"amt":hop1_amt})
if hop2_choice and hop2_amt>0:
    user_hops.append({"name":hop2_choice,"amt":hop2_amt})

st.markdown("---")

# ---------------------------
# MALT SECTION
# ---------------------------
st.subheader("🌾 Malt / Grain Bill")

col_m1, col_m2 = st.columns([1,1])
with col_m1:
    malt1_choice = st.selectbox(
        "Base / primary malt",
        MALT_CHOICES,
        index=0,
        key="malt1_choice",
    )
    malt2_choice = st.selectbox(
        "Specialty / character malt",
        MALT_CHOICES,
        index=1 if len(MALT_CHOICES)>1 else 0,
        key="malt2_choice",
    )

with col_m2:
    malt1_pct = st.number_input(
        "Malt 1 (% grist)",
        min_value=0.0,max_value=100.0,
        value=70.0, step=1.0
    )
    malt2_pct = st.number_input(
        "Malt 2 (% grist)",
        min_value=0.0,max_value=100.0,
        value=8.0, step=1.0
    )

user_malts = []
if malt1_choice and malt1_pct>0:
    user_malts.append({"name":malt1_choice,"pct":malt1_pct})
if malt2_choice and malt2_pct>0:
    user_malts.append({"name":malt2_choice,"pct":malt2_pct})

st.markdown("---")

# ---------------------------
# YEAST SECTION
# ---------------------------
st.subheader("🧫 Yeast & Fermentation")

col_y1, col_y2 = st.columns([1,1])
with col_y1:
    yeast_choice = st.selectbox(
        "Yeast strain",
        YEAST_CHOICES,
        index=0,
        key="yeast_choice",
    )
with col_y2:
    ferm_temp_c = st.number_input(
        "Fermentation temp (°C)",
        min_value=10.0,max_value=30.0,
        value=20.0, step=0.5,
        help="Target average ferment temp in Celsius"
    )

user_yeast = {
    "strain": yeast_choice,
    "ferm_temp_c": ferm_temp_c,
}

st.markdown("---")

# ---------------------------
# PREDICT BUTTON
# ---------------------------

st.subheader("🍻 Predict Beer Flavor & Balance")
st.caption(
    "Fill hops, malt, and yeast above — then click "
    "'Predict Beer Flavor & Balance' to simulate aroma, body, esters, color, etc."
)

predict_clicked = st.button("🍺 Predict Beer Flavor & Balance")

predicted_hops = {}
predicted_malt = {}
predicted_yeast = {}
radar_figs = []

if predict_clicked:
    # 1) run the sub-models
    if user_hops:
        predicted_hops = predict_hop_profile(user_hops)
    if user_malts:
        predicted_malt = predict_malt_profile(user_malts)
    if yeast_choice:
        predicted_yeast = predict_yeast_profile(user_yeast)

    # 2) Show textual summary (quick numeric snapshot)
    st.subheader("📊 Predicted Flavor Snapshot")

    # Hop aroma block
    if predicted_hops:
        st.markdown("**Hop aroma / character:**")
        hop_listed = [
            f"- {k}: {round(v,2)}"
            for k,v in predicted_hops.items()
        ]
        st.markdown("\n".join(hop_listed))
        st.markdown("")

    # Malt body/sweetness/color block
    if predicted_malt:
        st.markdown("**Malt body / sweetness / color:**")
        malt_listed = [
            f"- {k}: {round(v,2)}"
            for k,v in predicted_malt.items()
        ]
        st.markdown("\n".join(malt_listed))
        st.markdown("")

    # Yeast / fermentation block
    if predicted_yeast:
        st.markdown("**Yeast / fermentation profile:**")
        # We'll also include the chosen fermentation temp in F just for reference
        temp_f = ferm_temp_c * 9.0/5.0 + 32.0
        # We'll tack that onto predicted_yeast for text readability
        yeast_for_text = dict(predicted_yeast)
        yeast_for_text["Temp_avg_F"] = temp_f
        yeast_listed = [
            f"- {k}: {round(v,2)}"
            for k,v in yeast_for_text.items()
        ]
        st.markdown("\n".join(yeast_listed))
        st.markdown("")

    # 3) Build / display radars with shared 0..1 scale
    #    We'll do 3 radars:
    #    Hops (all hop_dims in predicted_hops),
    #    Malt (malt_dims),
    #    Yeast (yeast_dims)
    #    We'll gather all raw values across all three dicts
    #    and compute a global min/max for scaling.

    all_vals = []
    for dct in [predicted_hops, predicted_malt, predicted_yeast]:
        all_vals.extend(list(dct.values()))
    if not all_vals:
        all_vals = [0.0]
    global_min = min(all_vals)
    global_max = max(all_vals)
    if global_min == global_max:
        # avoid divide by zero -> expand
        global_min = 0.0
        global_max = 1.0

    radar_col1, radar_col2, radar_col3 = st.columns(3)

    # Hops radar
    if predicted_hops:
        hop_axes = list(predicted_hops.keys())
        hop_vals_norm = normalize_profile_dict(predicted_hops, hop_axes,
                                               global_min, global_max)
        fig_hops = plot_radar_sharedscale(hop_axes, hop_vals_norm,
                                          title="Hops / Aroma")
        with radar_col1:
            st.pyplot(fig_hops)

    # Malt radar
    if predicted_malt:
        malt_axes = list(predicted_malt.keys())
        malt_vals_norm = normalize_profile_dict(predicted_malt, malt_axes,
                                                global_min, global_max)
        fig_malt = plot_radar_sharedscale(malt_axes, malt_vals_norm,
                                          title="Malt / Body-Sweetness")
        with radar_col2:
            st.pyplot(fig_malt)

    # Yeast radar (we only include the model outputs, not the temperature)
    if predicted_yeast:
        yeast_axes = list(predicted_yeast.keys())
        yeast_vals_norm = normalize_profile_dict(predicted_yeast, yeast_axes,
                                                 global_min, global_max)
        fig_yeast = plot_radar_sharedscale(yeast_axes, yeast_vals_norm,
                                           title="Yeast / Fermentation")
        with radar_col3:
            st.pyplot(fig_yeast)

st.markdown("---")

# ---------------------------
# BREWMASTER (Azure OpenAI) SECTION
# ---------------------------

st.subheader("🧪 AI Brewmaster Guidance")

brewer_goal = st.text_area(
    "What's your intent for this beer? "
    "(e.g. 'Soft hazy IPA with saturated stone fruit and pineapple, low bitterness, pillowy mouthfeel')",
    "Soft hazy IPA with saturated stone fruit and pineapple, low bitterness, pillowy mouthfeel",
)

advise_clicked = st.button("🍻 Generate Brewmaster Notes")

if advise_clicked:
    # We'll recompute predictions if user didn't click Predict,
    # so your advisor always sees something.
    if not predicted_hops and user_hops:
        predicted_hops = predict_hop_profile(user_hops)
    if not predicted_malt and user_malts:
        predicted_malt = predict_malt_profile(user_malts)
    if not predicted_yeast and yeast_choice:
        predicted_yeast = predict_yeast_profile(user_yeast)

    # Build final dicts to feed to Azure
    hop_vec = predicted_hops if predicted_hops else {}
    malt_vec = predicted_malt if predicted_malt else {}
    # Add temperature info to yeast_vec context, but not radar scaling
    yeast_vec_for_ai = dict(predicted_yeast) if predicted_yeast else {}
    yeast_vec_for_ai["ferm_temp_c"] = ferm_temp_c

    ai_md = call_azure_brewmaster_notes(
        hop_vec,
        malt_vec,
        yeast_vec_for_ai,
        brewer_goal,
    )

    st.markdown("#### Brewmaster Notes")
    st.markdown(ai_md)
    st.caption(
        "Prototype — not production brewing advice. "
        "Always match your yeast strain's process window."
    )

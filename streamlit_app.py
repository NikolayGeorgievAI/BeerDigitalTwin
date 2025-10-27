############################################################
# streamlit_app.py
#
# Beer Recipe Digital Twin
# - Build hop bill, malt bill, and yeast/fermentation plan
# - Predict hop aroma, malt body/sweetness/color, yeast profile
# - Visualize 3 radar plots (hops / malt / yeast), normalized to [0..1]
# - Get "AI Brewmaster Guidance" from Azure OpenAI (with graceful fallback)
#
# NOTE:
#   - This app is a prototype, not production brewing advice.
#   - Make sure you have the following in Streamlit Secrets:
#
#       AZURE_OPENAI_ENDPOINT = "https://<your-resource-name>.openai.azure.com/"
#       AZURE_OPENAI_API_KEY  = "your-key"
#       AZURE_OPENAI_DEPLOYMENT = "your-deployment-name"
#
#   - The deployment name MUST match exactly the name in Azure AI Foundry.
#   - The model behind that deployment must support Chat Completions.
#
############################################################

import os
import re
import warnings
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
import joblib

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

############################################################
# Azure OpenAI (with fallback handling)
############################################################
# We will try to import and configure the OpenAI Azure client. If it fails,
# we'll still show a fallback "Brewmaster Notes".
try:
    from openai import OpenAI
    _HAVE_OPENAI = True
except Exception:
    _HAVE_OPENAI = False


############################################################
# Helper: text normalization / fuzzy matching to model columns
############################################################

def _clean_name(name: str) -> str:
    """Lowercase, strip ®/™ and non-alphanumerics."""
    if name is None:
        return ""
    s = str(name).lower()
    s = s.replace("®", "").replace("™", "")
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def _best_feature_match(user_name: str, feature_cols: list, prefix: str):
    """
    Fuzzy match user input (like 'Citra') to a column in feature_cols
    that starts with the given prefix (like 'hop_').
    We'll do a simple overlap-of-characters score.
    Returns best column name or None.
    """
    cleaned_user = _clean_name(user_name)
    if not cleaned_user:
        return None

    best_col = None
    best_score = -1
    for col in feature_cols:
        if not col.startswith(prefix):
            continue

        label_raw = col[len(prefix):]  # e.g. 'Citra®'
        cleaned_label = _clean_name(label_raw)

        # quick gate
        if len(cleaned_user) >= 3 and cleaned_user[:3] not in cleaned_label:
            # If the first 3 letters aren't even in the label, skip quickly.
            continue

        # simple char overlap
        common = set(cleaned_user) & set(cleaned_label)
        score = len(common)

        if score > best_score:
            best_score = score
            best_col = col

    return best_col


def _choices_from_features(feature_cols, preferred_prefix=None):
    """
    Build user-facing dropdown options from model feature columns.

    We first try to gather items that start with `preferred_prefix`
    (like "hop_", "malt_", "yeast_"). If none found, we'll use all feature cols.

    Then we remove brand prefixes like 'malt_', 'grain_', 'yeast_', etc.,
    strip ®/™ and underscores, and title-case lightly.
    """
    def scrub(colname: str):
        label = colname
        for p in ["hop_", "malt_", "grain_", "base_", "yeast_", "strain_", "y_", "m_"]:
            if label.startswith(p):
                label = label[len(p):]
        label = label.replace("®", "").replace("™", "")
        label = label.replace("_", " ").strip()
        return label

    # gather subset
    subset = []
    if preferred_prefix:
        for col in feature_cols:
            if col.startswith(preferred_prefix):
                subset.append(scrub(col))

    # fallback if empty
    if not subset:
        for col in feature_cols:
            subset.append(scrub(col))

    # deduplicate, sort
    out = []
    for name in subset:
        if name and name not in out:
            out.append(name)
    out = sorted(out, key=lambda s: s.lower())
    return out


############################################################
# Load hop, malt, yeast model bundles
############################################################

ROOT_DIR = os.path.dirname(__file__)

# --- Hop model ---
HOP_MODEL_PATH = os.path.join(ROOT_DIR, "hop_aroma_model.joblib")
hop_bundle = joblib.load(HOP_MODEL_PATH)
hop_model = hop_bundle["model"]
hop_feature_cols = hop_bundle["feature_cols"]
hop_dims = [
    a for a in hop_bundle["aroma_dims"]
    if str(a).strip().lower() not in ("nan", "", "none")
]

# --- Malt model ---
MALT_MODEL_PATH = os.path.join(ROOT_DIR, "malt_sensory_model.joblib")
malt_bundle = joblib.load(MALT_MODEL_PATH)
malt_model = malt_bundle["model"]
malt_feature_cols = malt_bundle["feature_cols"]
malt_dims = malt_bundle["flavor_cols"]  # ex: ['bready','caramel', ... or 'sweetness','color_intensity',...]

# --- Yeast model ---
YEAST_MODEL_PATH = os.path.join(ROOT_DIR, "yeast_sensory_model.joblib")
yeast_bundle = joblib.load(YEAST_MODEL_PATH)
yeast_model = yeast_bundle["model"]
yeast_feature_cols = yeast_bundle["feature_cols"]
yeast_dims = yeast_bundle["flavor_cols"]  # ex: ['attenuation_num','flocculation_num','temp_avg_F']


############################################################
# Build feature vectors for hops / malts / yeast
############################################################

def build_hop_features(user_hops):
    """
    user_hops: list[ {"name": "Citra", "amt": float grams}, ... ]

    We'll produce a single-row DataFrame with columns = hop_feature_cols,
    each cell is the total grams for that hop matched column.
    """
    totals = {c: 0.0 for c in hop_feature_cols}

    for entry in user_hops:
        nm = entry.get("name", "")
        amt = float(entry.get("amt", 0.0))
        if amt <= 0:
            continue
        match = _best_feature_match(nm, hop_feature_cols, prefix="hop_")
        if match:
            totals[match] += amt

    X = pd.DataFrame([totals], columns=hop_feature_cols)
    return X


def predict_hop_profile(user_hops):
    """
    Returns dict { dimension -> value } for hop aroma.
    """
    X = build_hop_features(user_hops)
    y_pred = hop_model.predict(X)[0]
    result = {}
    for dim, val in zip(hop_dims, y_pred):
        # We'll store floats directly
        result[dim] = float(val)
    return result


def build_malt_features(user_malts):
    """
    user_malts: list[{"name": "Maris Otter", "pct": 70}, {"name": "Caramalt", "pct": 8}, ...]

    For each malt, we try best prefix matches in order:
        "malt_", then "grain_", "base_", etc.
    We'll sum the grist pct for whichever col we match.
    """
    totals = {c: 0.0 for c in malt_feature_cols}

    for entry in user_malts:
        nm = entry.get("name", "")
        pct = float(entry.get("pct", 0.0))
        if pct <= 0:
            continue

        match = _best_feature_match(nm, malt_feature_cols, prefix="malt_")
        if match is None:
            for pfx in ["grain_", "base_", "malt_", "m_"]:
                match = _best_feature_match(nm, malt_feature_cols, prefix=pfx)
                if match:
                    break

        if match:
            totals[match] += pct

    X = pd.DataFrame([totals], columns=malt_feature_cols)
    return X


def predict_malt_profile(user_malts):
    """
    Returns dict { dimension -> value } for body/sweetness/color attributes.
    """
    X = build_malt_features(user_malts)
    y_pred = malt_model.predict(X)[0]
    result = {}
    for dim, val in zip(malt_dims, y_pred):
        result[dim] = float(val)
    return result


def build_yeast_features(user_yeast):
    """
    user_yeast = {
        "strain": "Nottingham Ale Yeast", 
        "ferm_temp_c": 20.0
    }

    We'll fuzzy-match the strain to yeast_feature_cols with "yeast_" or "strain_".
    We'll set that matching column = 1.0, else 0.

    Many yeast models don't directly incorporate temperature, but if yours does,
    you'd incorporate that here (e.g. you'd have columns for temp buckets).
    We'll just do strain identity as 1-hot.
    """
    totals = {c: 0.0 for c in yeast_feature_cols}

    strain = user_yeast.get("strain", "")
    match = _best_feature_match(strain, yeast_feature_cols, prefix="yeast_")
    if not match:
        for pfix in ["strain_", "y_", "yeast_"]:
            match = _best_feature_match(strain, yeast_feature_cols, prefix=pfix)
            if match:
                break

    if match:
        totals[match] = 1.0

    # (Optional) We could add temperature feature engineering here if your
    #   yeast model training had columns for different ferm temps.

    X = pd.DataFrame([totals], columns=yeast_feature_cols)
    return X


def predict_yeast_profile(user_yeast):
    """
    Returns dict { dimension -> value } for e.g. attenuation, flocculation, temp_avg_F, etc.
    """
    X = build_yeast_features(user_yeast)
    y_pred = yeast_model.predict(X)[0]
    result = {}
    for dim, val in zip(yeast_dims, y_pred):
        result[dim] = float(val)
    return result


############################################################
# Radar chart / spider web rendering
############################################################

def _normalize_for_radar(data_dict):
    """
    Normalize dict values to [0..1] range so they compare visually on same scale.
    If all values are 0 or identical, returns 0 for all keys to avoid divide-by-zero.
    """
    vals = list(data_dict.values())
    if len(vals) == 0:
        return {k: 0.0 for k in data_dict}
    vmin = min(vals)
    vmax = max(vals)
    if math.isclose(vmin, vmax):
        # all identical -> flat
        return {k: 0.0 for k in data_dict}
    span = vmax - vmin
    normed = {k: (v - vmin) / span for (k, v) in data_dict.items()}
    return normed


def _render_radar(ax, data_dict, title):
    """
    Draw a single radar (polar) chart on 'ax' using normalized [0..1] data_dict.
    Hides tick labels, etc. Just a nice shape + axis labels.
    """
    # order is stable by insertion, but we'll convert to list
    labels = list(data_dict.keys())
    normed_data = _normalize_for_radar(data_dict)
    values = [normed_data[k] for k in labels]

    # close the polygon
    labels.append(labels[0])
    values.append(values[0])

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)

    ax.set_theta_offset(np.pi / 2.0)
    ax.set_theta_direction(-1)
    ax.plot(angles, values, marker="o")
    ax.fill(angles, values, alpha=0.25)

    # set ticks at angles without numeric radial ticks
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels[:-1], fontsize=8)
    ax.set_yticklabels([])       # hide radial tick labels
    ax.set_ylim(0, 1)

    ax.set_title(title, fontsize=9, pad=10)


def render_three_radars(hop_profile, malt_profile, yeast_profile):
    """
    Build and display a figure with 3 subplots side by side:
    - Hops / Aroma
    - Malt / Body-Sweetness
    - Yeast / Fermentation

    We'll pick some representative dimensions from each profile 
    (the keys in your trained models may vary!)
    """
    # HOP dims subset
    # We'll just keep every hop dimension in hop_profile, but you could filter
    # to a preferred order. We'll do an ordered subset approach for readability:
    hop_keys_order = [
        "cedar", "citrus", "citrusy", "earthy", "floral", "fruity",
        "grassy", "herbal", "pine", "spicy", "stone fruit", "tropical fruit"
    ]
    hop_plot_data = {}
    for k in hop_keys_order:
        # match exact or fallback to any case-insensitive
        for hp_key in hop_profile.keys():
            if hp_key.lower() == k.lower():
                hop_plot_data[k] = hop_profile[hp_key]
                break

    # MALT dims subset (like sweetness, body_full, color_intensity)
    malt_keys_order = [
        "sweetness", "body_full", "color_intensity"
    ]
    malt_plot_data = {}
    for k in malt_keys_order:
        for mp_key in malt_profile.keys():
            if mp_key.lower() == k.lower():
                malt_plot_data[k] = malt_profile[mp_key]
                break

    # YEAST dims subset
    # typical names we predicted: attenuation_num, flocculation_num, temp_avg_F
    # We'll unify naming: "attenuation_num", "flocculation_num", "temp_avg_F"
    yeast_keys_order = [
        "attenuation_num", "flocculation_num", "temp_avg_F"
    ]
    yeast_plot_data = {}
    for k in yeast_keys_order:
        for yp_key in yeast_profile.keys():
            if yp_key.lower() == k.lower():
                yeast_plot_data[k] = yeast_profile[yp_key]
                break

    # Create figure
    fig = plt.figure(figsize=(9, 3))
    ax1 = fig.add_subplot(1, 3, 1, polar=True)
    ax2 = fig.add_subplot(1, 3, 2, polar=True)
    ax3 = fig.add_subplot(1, 3, 3, polar=True)

    _render_radar(ax1, hop_plot_data, "Hops / Aroma")
    _render_radar(ax2, malt_plot_data, "Malt / Body-Sweetness")
    _render_radar(ax3, yeast_plot_data, "Yeast / Fermentation")

    fig.tight_layout()
    return fig


############################################################
# Azure Brewmaster Guidance (with graceful fallback)
############################################################

def call_azure_brewmaster_notes(
    hop_prof, malt_prof, yeast_prof, brewer_goal,
    azure_endpoint, azure_key, azure_deployment
):
    """
    Try calling Azure OpenAI Chat Completion to get brewmaster-style notes.
    If anything fails, return a *friendly fallback* paragraph without dumping raw error.
    """
    # We'll prepare a structured system+user prompt that references:
    #  - hop_prof (aroma profile)
    #  - malt_prof (body/sweetness/color)
    #  - yeast_prof (fermentation character)
    #  - brewer_goal (what the brewer wants)
    #
    # We'll NOT attach large raw dict dumps; just summarizing them at high-level.

    # We'll produce short bullet steps: hops, malt/grist, fermentation, summary.

    # Build a short summary string of predicted flavor edges:
    def summarize_dict(d):
        # pick a couple "top" attributes if available
        # sort by magnitude descending
        items = sorted(d.items(), key=lambda kv: abs(kv[1]), reverse=True)
        top_bits = []
        for k,v in items[:5]:
            top_bits.append(f"{k}: {v:.2f}")
        return ", ".join(top_bits)

    hop_summary = summarize_dict(hop_prof)
    malt_summary = summarize_dict(malt_prof)
    yeast_summary = summarize_dict(yeast_prof)

    system_msg = (
        "You are an expert brewmaster. "
        "Given predicted hop aroma, malt body/sweetness/color, and yeast/fermentation "
        "profile, produce concise actionable guidance on how to adjust the recipe "
        "to align with the brewer's stated goal. "
        "Be specific but concise. Use bullet points. "
        "Do NOT mention 'AI' or 'OpenAI' or internal errors."
    )

    user_msg = (
        f"Brewer goal:\n{brewer_goal}\n\n"
        f"Hop profile (key notes): {hop_summary}\n"
        f"Malt/body profile (key notes): {malt_summary}\n"
        f"Yeast/fermentation profile (key notes): {yeast_summary}\n\n"
        "Please provide:\n"
        "1. Hop adjustments (late additions, dry hop, bitterness).\n"
        "2. Malt/grist tweaks (sweetness, body, haze).\n"
        "3. Fermentation / yeast guidance (temp, attenuation, mouthfeel).\n"
        "4. A one-sentence summary.\n"
    )

    # We'll attempt the Azure Chat Completion call if openai is available
    # and we have valid secrets.
    fallback_text = (
        "• Hop toward tropical/stone-fruit character using high-oil late/dry hop "
        "additions; keep bitterness low to maintain softness.\n"
        "• Add oats or wheat for pillowy mouthfeel and stable haze; avoid overly "
        "toasty/crystal malts if you want a pale, juicy profile.\n"
        "• Use a moderately ester-friendly yeast and keep fermentation temps in "
        "the softer range (around 18–20°C) to avoid harsh fusels.\n"
        "• Goal: juicy, saturated fruit aroma and plush mouthfeel without sharp bitterness."
    )

    if (not _HAVE_OPENAI) or (not azure_endpoint) or (not azure_key) or (not azure_deployment):
        # no client or missing config -> fallback
        return "Brewmaster Notes (prototype)\n" + fallback_text

    try:
        client = OpenAI(
            base_url=f"{azure_endpoint}openai/deployments/{azure_deployment}/",
            api_key=azure_key,
            default_headers={"api-key": azure_key},
        )

        # Azure "chat.completions.create" style call:
        completion = client.chat.completions.create(
            model=azure_deployment,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.6,
            max_tokens=300,
        )

        # Extract response text safely
        content = ""
        if completion and hasattr(completion, "choices") and completion.choices:
            first_choice = completion.choices[0]
            # azure returns message dict {role, content}
            if hasattr(first_choice, "message") and first_choice.message:
                content = first_choice.message.get("content", "")

        if not content:
            # If we got no text, fallback
            return "Brewmaster Notes (prototype)\n" + fallback_text

        return content.strip()

    except Exception:
        # On *any* azure error, just fallback with no raw error details shown
        return "Brewmaster Notes (prototype)\n" + fallback_text


############################################################
# STREAMLIT UI
############################################################

st.set_page_config(
    page_title="Beer Recipe Digital Twin",
    page_icon="🍺",
    layout="centered",
)

st.title("🍺 Beer Recipe Digital Twin")

st.markdown(
    """
Your AI brew assistant:

1. Build a hop bill, grain bill, and fermentation plan.  
2. Predict aroma, body, color, esters, mouthfeel — together.  
3. Get brewmaster-style guidance based on your style goal.

**Prototype — not production brewing advice.**
"""
)

# Build the dropdown options for each model:
HOP_CHOICES = _choices_from_features(hop_feature_cols, preferred_prefix="hop_")
MALT_CHOICES = _choices_from_features(malt_feature_cols, preferred_prefix="malt_")
YEAST_CHOICES = _choices_from_features(yeast_feature_cols, preferred_prefix="yeast_")

############################################################
# HOPS SECTION
############################################################

st.header("🌿 Hops (late/aroma additions)")

col_h1, col_h2 = st.columns(2)
with col_h1:
    hop1 = st.selectbox(
        "Main Hop Variety",
        HOP_CHOICES,
        index=HOP_CHOICES.index("Mosaic") if "Mosaic" in HOP_CHOICES else 0,
        key="hop1_select",
    )
    hop2 = st.selectbox(
        "Secondary Hop Variety",
        HOP_CHOICES,
        index=HOP_CHOICES.index("Citra") if "Citra" in HOP_CHOICES else 0,
        key="hop2_select",
    )

with col_h2:
    hop1_amt = st.number_input(
        "Hop 1 amount (g)",
        min_value=0.0,
        max_value=500.0,
        value=30.0,
        step=5.0,
        key="hop1_amt",
    )
    hop2_amt = st.number_input(
        "Hop 2 amount (g)",
        min_value=0.0,
        max_value=500.0,
        value=20.0,
        step=5.0,
        key="hop2_amt",
    )


############################################################
# MALT SECTION
############################################################

st.header("🌾 Malt / Grain Bill")

m1, m2 = st.columns(2)
with m1:
    malt1 = st.selectbox(
        "Base / primary malt",
        MALT_CHOICES,
        index=0,
        key="malt1_select",
    )
    malt2 = st.selectbox(
        "Specialty / character malt",
        MALT_CHOICES,
        index=min(1, len(MALT_CHOICES)-1),
        key="malt2_select",
    )

with m2:
    malt1_pct = st.number_input(
        "Malt 1 (% grist)",
        min_value=0.0,
        max_value=100.0,
        value=70.0,
        step=1.0,
        key="malt1_pct",
    )
    malt2_pct = st.number_input(
        "Malt 2 (% grist)",
        min_value=0.0,
        max_value=100.0,
        value=8.0,
        step=1.0,
        key="malt2_pct",
    )


############################################################
# YEAST & FERMENTATION SECTION
############################################################

st.header("🧫 Yeast & Fermentation")

y1, y2 = st.columns(2)
with y1:
    yeast_strain = st.selectbox(
        "Yeast strain",
        YEAST_CHOICES,
        index=0,
        key="yeast_strain_select",
    )

with y2:
    ferm_temp_c = st.number_input(
        "Fermentation temp (°C)",
        min_value=15.0,
        max_value=25.0,
        value=20.0,
        step=0.5,
        key="ferm_temp_c",
    )

st.markdown("---")

############################################################
# PREDICT BUTTON: Build full predicted flavor snapshot
############################################################

st.subheader("🍻 Predict Beer Flavor & Balance")

predict_btn = st.button(
    "🔍 Predict Beer Flavor & Balance",
    help="Runs the hop, malt, and yeast models together, then shows combined flavor radars."
)

st.caption(
    "Fill hops, malt, and yeast above — then click 'Predict Beer Flavor & Balance' "
    "to simulate aroma, body, esters, color, etc."
)

hop_profile = {}
malt_profile = {}
yeast_profile = {}

if predict_btn:
    # Gather user recipe
    user_hops = []
    if hop1 and hop1_amt > 0:
        user_hops.append({"name": hop1, "amt": hop1_amt})
    if hop2 and hop2_amt > 0:
        user_hops.append({"name": hop2, "amt": hop2_amt})

    user_malts = []
    if malt1 and malt1_pct > 0:
        user_malts.append({"name": malt1, "pct": malt1_pct})
    if malt2 and malt2_pct > 0:
        user_malts.append({"name": malt2, "pct": malt2_pct})

    user_yeast = {
        "strain": yeast_strain,
        "ferm_temp_c": ferm_temp_c,
    }

    hop_profile = predict_hop_profile(user_hops) if user_hops else {}
    malt_profile = predict_malt_profile(user_malts) if user_malts else {}
    yeast_profile = predict_yeast_profile(user_yeast) if yeast_strain else {}

    # Convert the fermentation temperature to F in the yeast_profile if needed
    # (Sometimes the yeast model is in F. We'll add a 'temp_avg_F' if the model
    #  expects that dimension.)
    if "temp_avg_F" in yeast_profile:
        # We assume model gave a value in F, so leave it.
        pass
    else:
        # If model doesn't provide temp_avg_F but we still want it on the
        # radar, we can compute from ferm_temp_c:
        #   F = C*9/5 + 32
        temp_f = ferm_temp_c * 9.0 / 5.0 + 32.0
        # Only add if yeast_dims suggests it might exist:
        if any(dim.lower() == "temp_avg_f" for dim in yeast_dims):
            yeast_profile["temp_avg_F"] = float(temp_f)

    # Show the 3 radar plots (hops, malt, yeast) with normalized scales
    fig = render_three_radars(hop_profile, malt_profile, yeast_profile)
    st.pyplot(fig)

st.markdown("---")

############################################################
# AI BREWMASTER GUIDANCE
############################################################

st.subheader("🧪 AI Brewmaster Guidance")

brewer_goal = st.text_area(
    "What's your intent for this beer? (e.g. 'Soft hazy IPA with saturated stone fruit and pineapple, low bitterness, pillowy mouthfeel')",
    value="Soft hazy IPA with saturated stone fruit and pineapple, low bitterness, pillowy mouthfeel",
    height=100,
)

# We'll always have a button to ask for Brewmaster Notes
gen_notes_btn = st.button("🧪 Generate Brewmaster Notes")

if gen_notes_btn:
    # We will call Azure with the *latest known* hop_profile/malt_profile/yeast_profile
    # from the user's last "Predict Beer Flavor" click.
    # If they haven't clicked Predict yet, these dicts may still be empty.
    # That's fine; our fallback will handle that case gracefully.
    azure_endpoint = st.secrets.get("AZURE_OPENAI_ENDPOINT", "").strip()
    azure_key = st.secrets.get("AZURE_OPENAI_API_KEY", "").strip()
    azure_deployment = st.secrets.get("AZURE_OPENAI_DEPLOYMENT", "").strip()

    ai_md = call_azure_brewmaster_notes(
        hop_prof=hop_profile,
        malt_prof=malt_profile,
        yeast_prof=yeast_profile,
        brewer_goal=brewer_goal,
        azure_endpoint=azure_endpoint,
        azure_key=azure_key,
        azure_deployment=azure_deployment,
    )

    st.subheader("Brewmaster Notes")
    st.write(ai_md)

    st.caption(
        "Prototype — not production brewing advice. "
        "Always match your yeast strain's process window."
    )

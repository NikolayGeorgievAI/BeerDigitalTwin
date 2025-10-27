# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from matplotlib.projections.polar import PolarAxes
from math import pi

# --------------------------
# 1. PAGE CONFIG
# --------------------------
st.set_page_config(
    page_title="Beer Recipe Digital Twin",
    page_icon="🍺",
    layout="wide"
)

# --------------------------
# 2. LOAD DATA AND MODELS
# --------------------------

@st.cache_data(show_spinner=False)
def load_reference_data():
    # These pickles are expected to exist in repo root
    try:
        clean_malt_df = pd.read_pickle("clean_malt_df.pkl")
    except Exception:
        clean_malt_df = pd.DataFrame(columns=["malt_name", "descriptor"])

    try:
        clean_yeast_df = pd.read_pickle("clean_yeast_df.pkl")
    except Exception:
        clean_yeast_df = pd.DataFrame(columns=["name", "style", "notes"])

    return clean_malt_df, clean_yeast_df

@st.cache_resource(show_spinner=False)
def load_models():
    # These joblibs are expected to exist in repo root
    try:
        hop_aroma_model = joblib.load("hop_aroma_model.joblib")
    except Exception:
        hop_aroma_model = None

    try:
        malt_sensory_model = joblib.load("malt_sensory_model.joblib")
    except Exception:
        malt_sensory_model = None

    try:
        yeast_sensory_model = joblib.load("yeast_sensory_model.joblib")
    except Exception:
        yeast_sensory_model = None

    return hop_aroma_model, malt_sensory_model, yeast_sensory_model


clean_malt_df, clean_yeast_df = load_reference_data()
hop_aroma_model, malt_sensory_model, yeast_sensory_model = load_models()

# --------------------------
# 3. CONSTANTS YOU MUST KEEP IN SYNC WITH TRAINING
# --------------------------
# This MUST be the list of hops your model actually knows.
# Fill this with the EXACT hop names used in training, e.g.:
# ["Simcoe", "Amarillo", "Citra", "Mosaic", "Galaxy", ...]
AVAILABLE_HOPS = [
    "Simcoe",
    "Amarillo",
    "Citra",
    "Mosaic",
    "Galaxy",
    "Nelson Sauvin",
    "Centennial",
    "Cascade",
    "Azacca",
    "Ekuanot",
    "Sabro",
    "El Dorado",
    "Chinook",
    "Columbus",
    "Warrior",
    "Astra",
    "Eclipse",
    "Ella",
    "Enigma",
    "Adeeena",  # <-- check spelling vs training set!
    # etc...
]

# This MUST be the exact column order that hop_aroma_model expects.
# Typically: ["hop_Simcoe","hop_Amarillo",...,"total_hop_g","ibu_proxy","late_add_pct",...]
# For now we assume simple one-hot by hop name, plus total weight.
# You MUST replace this with the real training columns in the correct order.
HOP_MODEL_FEATURE_ORDER = [
    # one-hot columns for each hop in training
    "hop_Simcoe",
    "hop_Amarillo",
    "hop_Citra",
    "hop_Mosaic",
    "hop_Galaxy",
    "hop_Nelson Sauvin",
    "hop_Centennial",
    "hop_Cascade",
    "hop_Azacca",
    "hop_Ekuanot",
    "hop_Sabro",
    "hop_El Dorado",
    "hop_Chinook",
    "hop_Columbus",
    "hop_Warrior",
    "hop_Astra",
    "hop_Eclipse",
    "hop_Ella",
    "hop_Enigma",
    "hop_Adeeena",  # again, spelling must match training
    # optional engineered features from training:
    "total_hop_g",
]

# Radar plot axes (model output aroma notes).
# The hop_aroma_model should output these in *this* order.
AROMA_AXES = [
    "fruity",
    "citrus",
    "tropical",
    "earthy",
    "spicy",
    "herbal",
    "floral",
    "resinous",
]

# --------------------------
# 4. FEATURE BUILDERS
# --------------------------

def build_hop_feature_vector(hop_bill):
    """
    hop_bill = list of (hop_name, grams) for up to 4 hops.
    Example:
        [("Simcoe", 60.0),
         ("Amarillo", 30.0),
         ("-", 0.0),
         ("-", 0.0)]

    Returns:
        row_df: pandas.DataFrame shaped (1, len(HOP_MODEL_FEATURE_ORDER))
        debug_df: short DataFrame showing internal one-hot representation
    """
    # initialize all zeros
    feat = {col: 0.0 for col in HOP_MODEL_FEATURE_ORDER}

    # sum total grams and also assign to each hop column
    total = 0.0
    for hop_name, g in hop_bill:
        if hop_name is None or hop_name == "-" or g is None:
            continue
        try:
            g_float = float(g)
        except Exception:
            g_float = 0.0
        total += g_float
        # put grams for that hop in matching column if exists
        hop_col = f"hop_{hop_name}"
        if hop_col in feat:
            feat[hop_col] += g_float

    # store total grams
    if "total_hop_g" in feat:
        feat["total_hop_g"] = total

    # make row df
    row_df = pd.DataFrame([feat], columns=HOP_MODEL_FEATURE_ORDER)

    # debug df so user can see what model actually received
    debug_df = row_df.copy()
    return row_df, debug_df


def predict_hop_aroma(hop_bill):
    """
    Returns:
        aroma_scores: dict of {axis: value} for AROMA_AXES,
                      or zeros if model missing / fails
        debug_df: hop feature input row for inspection
    """
    X_row, debug_df = build_hop_feature_vector(hop_bill)

    # default zeros
    aroma_scores = {ax: 0.0 for ax in AROMA_AXES}

    if hop_aroma_model is None:
        return aroma_scores, debug_df

    try:
        y_pred = hop_aroma_model.predict(X_row)
        # We assume y_pred is shape (1, len(AROMA_AXES))
        if hasattr(y_pred, "shape") and y_pred.shape[1] == len(AROMA_AXES):
            for ax, val in zip(AROMA_AXES, y_pred[0]):
                aroma_scores[ax] = float(val)
        else:
            # length mismatch -> keep zeros
            pass
    except Exception:
        # prediction failed -> keep zeros
        pass

    return aroma_scores, debug_df


def predict_malt_character(malt_inputs):
    """
    Very simple placeholder using malt_sensory_model if available.
    malt_inputs = [(malt1_name, pct1), (malt2_name, pct2), (malt3_name, pct3)]

    You should replace with your real feature builder for malt.

    We'll just return some fake-ish text now, or a fallback phrase.
    """
    if malt_sensory_model is None:
        return "bready"  # fallback if model missing

    # TODO: engineer real malt features and model.predict
    # For now, dummy:
    return "bready"


def predict_yeast_character(yeast_choice):
    """
    Yeast character / fermentation notes.

    You should replace with real yeast feature builder,
    or matching to yeast_sensory_model.
    """
    if yeast_sensory_model is None:
        # fallback
        if yeast_choice and yeast_choice != "-":
            return "fruity_esters, clean_neutral"
        else:
            return "clean_neutral"

    # TODO real model
    return "fruity_esters, clean_neutral"


def predict_style_direction(yeast_choice, malt_desc, aroma_scores):
    """
    Rough style guess based on yeast + malt + hop aroma.
    This is currently heuristic / placeholder.
    """
    # Example logic:
    if "clean" in malt_desc or "pils" in malt_desc.lower():
        style = "Clean / Neutral Ale direction"
        emoji = "🧪"
    else:
        style = "Experimental / Hybrid"
        emoji = "🧪"

    return emoji, style


def get_top_hop_notes(aroma_scores, top_n=2):
    """
    Take aroma_scores dict, pick best N, return list of (note, val).
    """
    items = list(aroma_scores.items())
    # sort descending by value
    items.sort(key=lambda x: x[1], reverse=True)
    # top_n
    return items[:top_n]

# --------------------------
# 5. PLOTTING
# --------------------------

def plot_radar(aroma_scores):
    """
    Draw radar chart with the 8 axes from AROMA_AXES.
    """
    categories = AROMA_AXES
    N = len(categories)

    values = [aroma_scores[cat] for cat in categories]
    # close the loop for radar
    values += values[:1]

    # angles for each axis
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(
        figsize=(6, 6),
        subplot_kw=dict(polar=True),
    )

    # draw one axis per variable
    plt.xticks(angles[:-1], categories)

    # draw radial labels (y labels)
    ax.set_rlabel_position(0)
    # auto range based on data
    max_val = max(values) if max(values) > 0 else 1.0
    # pick some "nice" grid, e.g. 5 rings
    tick_vals = np.linspace(0, max_val, 6)
    tick_vals = np.round(tick_vals, 2)
    plt.yticks(tick_vals[:-1], [str(v) for v in tick_vals[:-1]], color="gray", size=8)
    plt.ylim(0, tick_vals[-1])

    # plot data
    ax.plot(angles, values, linewidth=2, linestyle='solid')
    ax.fill(angles, values, alpha=0.25)

    return fig

# --------------------------
# 6. SIDEBAR INPUTS
# --------------------------

def sidebar_inputs():
    """
    Renders the entire left sidebar (hops, malts, yeast)
    and returns a tuple of structured inputs + Predict button state.
    """

    st.sidebar.header("🧪 Model Inputs")

    st.sidebar.subheader("Hop Bill (g)")

    # Hop 1
    hop1_name = st.sidebar.selectbox(
        "Hop 1",
        ["-"] + AVAILABLE_HOPS,
        index=0,
        key="hop1_name"
    )
    hop1_amt = st.sidebar.number_input(
        f"{hop1_name} (g)",
        min_value=0.0,
        max_value=500.0,
        value=0.0,
        step=5.0,
        key="hop1_amt"
    )

    # Hop 2
    hop2_name = st.sidebar.selectbox(
        "Hop 2",
        ["-"] + AVAILABLE_HOPS,
        index=0,
        key="hop2_name"
    )
    hop2_amt = st.sidebar.number_input(
        f"{hop2_name} (g)",
        min_value=0.0,
        max_value=500.0,
        value=0.0,
        step=5.0,
        key="hop2_amt"
    )

    # Hop 3
    hop3_name = st.sidebar.selectbox(
        "Hop 3",
        ["-"] + AVAILABLE_HOPS,
        index=0,
        key="hop3_name"
    )
    hop3_amt = st.sidebar.number_input(
        f"{hop3_name} (g)",
        min_value=0.0,
        max_value=500.0,
        value=0.0,
        step=5.0,
        key="hop3_amt"
    )

    # Hop 4
    hop4_name = st.sidebar.selectbox(
        "Hop 4",
        ["-"] + AVAILABLE_HOPS,
        index=0,
        key="hop4_name"
    )
    hop4_amt = st.sidebar.number_input(
        f"{hop4_name} (g)",
        min_value=0.0,
        max_value=500.0,
        value=0.0,
        step=5.0,
        key="hop4_amt"
    )

    st.sidebar.subheader("Malt Bill")

    malt1_type = st.sidebar.text_input(
        "Malt 1",
        "AMBER MALT",
        key="malt1_type"
    )
    malt1_pct = st.sidebar.number_input(
        "Malt 1 %",
        min_value=0.0,
        max_value=100.0,
        value=70.0,
        step=1.0,
        key="malt1_pct"
    )

    malt2_type = st.sidebar.text_input(
        "Malt 2",
        "BEST ALE MALT",
        key="malt2_type"
    )
    malt2_pct = st.sidebar.number_input(
        "Malt 2 %",
        min_value=0.0,
        max_value=100.0,
        value=20.0,
        step=1.0,
        key="malt2_pct"
    )

    malt3_type = st.sidebar.text_input(
        "Malt 3",
        "BLACK MALT",
        key="malt3_type"
    )
    malt3_pct = st.sidebar.number_input(
        "Malt 3 %",
        min_value=0.0,
        max_value=100.0,
        value=10.0,
        step=1.0,
        key="malt3_pct"
    )

    st.sidebar.subheader("Yeast Strain")

    # Yeast options from clean_yeast_df if it exists
    if "name" in clean_yeast_df.columns:
        yeast_options = ["-"] + sorted(clean_yeast_df["name"].unique().tolist())
    else:
        yeast_options = ["-"]

    yeast_choice = st.sidebar.selectbox(
        "Select yeast",
        yeast_options,
        index=0,
        key="yeast_choice"
    )

    run_button = st.sidebar.button("Predict Flavor 🧪", key="run_button")

    hop_inputs = [
        (hop1_name, hop1_amt),
        (hop2_name, hop2_amt),
        (hop3_name, hop3_amt),
        (hop4_name, hop4_amt),
    ]

    malt_inputs = [
        (malt1_type, malt1_pct),
        (malt2_type, malt2_pct),
        (malt3_type, malt3_pct),
    ]

    return hop_inputs, malt_inputs, yeast_choice, run_button

# --------------------------
# 7. MAIN LAYOUT
# --------------------------

def main():

    hop_inputs, malt_inputs, yeast_choice, run_button = sidebar_inputs()

    # MAIN TITLE AREA
    st.markdown(
        """
        <div style="display:flex;align-items:center;gap:0.5rem;">
            <div style="font-size:2rem;">🍺</div>
            <div>
                <div style="font-size:2rem; font-weight:600; line-height:2rem;">
                    Beer Recipe Digital Twin
                </div>
                <div style="font-size:0.9rem; color:#666; margin-top:0.5rem;">
                    Predict hop aroma, malt character, and fermentation profile using trained ML models.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Only run predictions if user clicks Predict Flavor
    if run_button:
        # Hop aroma prediction
        aroma_scores, debug_df = predict_hop_aroma(hop_inputs)

        # Malt & Yeast characterization
        malt_desc = predict_malt_character(malt_inputs)
        yeast_desc = predict_yeast_character(yeast_choice)
        style_emoji, style_text = predict_style_direction(yeast_choice, malt_desc, aroma_scores)

        # Get top hop notes
        top_notes = get_top_hop_notes(aroma_scores, top_n=2)

        # -----------------------
        # LAYOUT: 2 columns wide
        # -----------------------
        col_left, col_right = st.columns([2,1], gap="large")

        with col_left:
            st.subheader("Hop Aroma Radar")
            fig = plot_radar(aroma_scores)
            st.pyplot(fig, use_container_width=True)

            # Debug expander so we can verify model input
            with st.expander("🔬 Debug: hop model input (what the model actually sees)"):
                st.write(debug_df)

        with col_right:
            st.subheader("Top hop notes:")
            if len(top_notes) == 0:
                st.write("No recognizable hop input (or total was 0).")
            else:
                for note, val in top_notes:
                    st.write(f"- {note} ({round(val, 2)})")

            st.subheader("Malt character:")
            st.write(malt_desc)

            st.subheader("Yeast character:")
            st.write(yeast_desc)

            st.subheader("Style direction:")
            st.write(f"{style_emoji} {style_text}")

            # show which hops we fed in (non-zero)
            used_hops = [f"{name} ({amt}g)" for (name, amt) in hop_inputs if name != "-" and amt and amt > 0]
            st.subheader("Hops used by the model:")
            if used_hops:
                st.write(", ".join(used_hops))
            else:
                st.write("None")

    else:
        # BEFORE the user presses Predict Flavor:
        hint = """
        🧪 Build your hop bill (up to 4 hops, with nonzero grams),
        set malt bill (% grist), choose yeast,
        then click **Predict Flavor 🧪** in the sidebar.
        """
        st.info(hint)


# --------------------------
# 8. ENTRY POINT
# --------------------------
if __name__ == "__main__":
    main()

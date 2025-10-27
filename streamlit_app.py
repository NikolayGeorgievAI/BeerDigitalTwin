import streamlit as st
import numpy as np
import pandas as pd
import joblib
import math
import matplotlib.pyplot as plt

###############################################################################
# 1. CONFIG / CONSTANTS
###############################################################################

st.set_page_config(
    page_title="Beer Recipe Digital Twin",
    layout="wide",
    initial_sidebar_state="expanded",
)

# These should match exactly the hop names from your training data
# (the ones that correspond to columns in the hop model input).
# We'll map user selections (like "Simcoe") into these one-hot/weighted columns.
AVAILABLE_HOPS = [
    "Astra", "Eclipse", "Ella", "Enigma", "Feux-Coeur Francais", "Galaxy",
    "Helga", "HPA-016", "Melba", "Pride of Ringwood", "Summer", "Super Pride",
    "Sylva", "Topaz", "Vic Secret", "Vienna Gold", "Canadian Redvine",
    "Simcoe", "Amarillo", "Citra", "Mosaic", "AnecdotalExample"  # etc.
    # ^^^ Add the rest of your hops here, exactly how they were named in training
]

# This MUST match the exact column order (and names) the hop aroma model expects.
# The hop model likely expects a DataFrame with columns like:
#   ["hop_Astra", "hop_Eclipse", ..., "hop_Simcoe", ...]
# or maybe just ["Astra", "Eclipse", ...] depending on how you trained it.
#
# >>> ACTION REQUIRED BY YOU <<<
# Put the real list of feature columns from training here, in order.
HOP_MODEL_FEATURE_ORDER = [
    # EXAMPLE PLACEHOLDER (you must replace these with real model columns):
    # "Astra", "Eclipse", "Ella", "Enigma", "Feux-Coeur Francais", "Galaxy",
    # "Helga", "HPA-016", "Melba", "Pride of Ringwood", "Summer", "Super Pride",
    # "Sylva", "Topaz", "Vic Secret", "Vienna Gold", "Canadian Redvine",
    # "Simcoe", "Amarillo", "Citra", "Mosaic"
]

# Radar/spider chart aroma axes
AROMA_AXES = [
    "fruity", "citrus", "tropical", "earthy", "spicy", "herbal", "floral", "resinous"
]
# We will plot them in a loop around a circle.


###############################################################################
# 2. LOAD MODELS / DATA
###############################################################################

@st.cache_resource
def load_models_and_data():
    # hop aroma model (scikit-learn regressor-like that outputs aroma intensities)
    hop_model = joblib.load("hop_aroma_model.joblib")

    # malt data / yeast data you saved (we won't retrain anything, just maybe
    # look up possible values or use them for display)
    clean_malt_df = pd.read_pickle("clean_malt_df.pkl")
    clean_yeast_df = pd.read_pickle("clean_yeast_df.pkl")

    return hop_model, clean_malt_df, clean_yeast_df


hop_model, clean_malt_df, clean_yeast_df = load_models_and_data()


###############################################################################
# 3. HELPERS: BUILD HOP FEATURE VECTOR
###############################################################################

def build_hop_feature_vector(hop_selections):
    """
    hop_selections = list of tuples:
        [ (hop_name, grams), (hop_name, grams), ...]
    We'll convert this into the feature vector the hop model expects.

    Strategy:
    - Sum total grams
    - Compute normalized weight for each hop in AVAILABLE_HOPS
      weight(h) = grams_h / total_grams
    - Put those weights into the correct feature columns (HOP_MODEL_FEATURE_ORDER).
    - Return 1-row DataFrame matching hop_model input expectations.
    """

    # Turn into dict: {hop_name: grams}
    grams_by_hop = {}
    for hop_name, grams in hop_selections:
        # filter empties / zero
        if hop_name and hop_name != "-" and grams and grams > 0:
            grams_by_hop[hop_name] = grams_by_hop.get(hop_name, 0) + grams

    total = sum(grams_by_hop.values())
    # Avoid divide-by-zero
    if total <= 0:
        total = 1e-9

    # Normalized weights by hop
    weights_by_hop = {h: (g / total) for h, g in grams_by_hop.items()}

    # Now we build a row for the model
    # Initialize all columns in the model's expected order to 0
    row_values = []
    for col in HOP_MODEL_FEATURE_ORDER:
        # There are two common naming conventions:
        #   A) model columns exactly match hop names, e.g. "Simcoe"
        #   B) model columns prefixed, e.g. "hop_Simcoe"
        #
        # We'll try exact match first, then try stripping known prefixes.
        base_name = col
        if base_name.startswith("hop_"):
            base_name = base_name[len("hop_"):]
        val = weights_by_hop.get(base_name, 0.0)
        row_values.append(val)

    hop_feature_df = pd.DataFrame([row_values], columns=HOP_MODEL_FEATURE_ORDER)
    return hop_feature_df


###############################################################################
# 4. PREDICT AROMA USING HOP MODEL
###############################################################################

def predict_hop_aroma(hop_feature_df):
    """
    Run hop_model on the feature df. We assume hop_model produces aroma intensities
    for AROMA_AXES (order doesn't strictly matter, but we will assume the model
    returns them in that order or we adapt here).
    """

    # hop_model.predict(...) should return shape (1, N).
    # We'll assume N == len(AROMA_AXES).
    y_pred = hop_model.predict(hop_feature_df)

    # Handle if model returns a 2D array
    if hasattr(y_pred, "shape") and len(y_pred.shape) == 2:
        y_pred = y_pred[0]

    # Safety: if the model does NOT have same length as AROMA_AXES yet,
    # we just clip/pad.
    aroma_scores = {}
    for i, axis in enumerate(AROMA_AXES):
        if i < len(y_pred):
            aroma_scores[axis] = float(y_pred[i])
        else:
            aroma_scores[axis] = 0.0

    return aroma_scores


###############################################################################
# 5. PLOT RADAR / SPIDER CHART
###############################################################################

def plot_radar(aroma_scores):
    """
    aroma_scores is dict {axis_name: value}
    We'll build a spider chart in polar coords using matplotlib.
    """

    labels = AROMA_AXES
    values = [aroma_scores[a] for a in labels]

    # close the loop for polar polygon
    values += [values[0]]
    angles = np.linspace(0, 2 * math.pi, len(labels), endpoint=False).tolist()
    angles += [angles[0]]

    fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
    ax.plot(angles, values, color="#1f77b4", linewidth=2)
    ax.fill(angles, values, color="#1f77b4", alpha=0.2)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)

    # radial labels
    ax.set_yticklabels([])

    # annotate each point
    for angle, val, label in zip(angles[:-1], values[:-1], labels):
        ax.annotate(
            f"{val:.2f}",
            xy=(angle, val),
            xytext=(5,5),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#1f77b4", alpha=0.6)
        )

    ax.set_title("Hop Aroma Radar", fontsize=18, fontweight="bold", pad=20)
    return fig


###############################################################################
# 6. SIDEBAR INPUTS
###############################################################################

def sidebar_inputs():
    st.sidebar.header("🧪 Model Inputs")

    st.sidebar.subheader("Hop Bill (g)")

    hop1_name = st.sidebar.selectbox("Hop 1", ["-"] + AVAILABLE_HOPS, index=0)
    hop1_amt  = st.sidebar.number_input(f"{hop1_name} (g)", min_value=0.0, max_value=500.0, value=0.0, step=5.0)

    hop2_name = st.sidebar.selectbox("Hop 2", ["-"] + AVAILABLE_HOPS, index=0)
    hop2_amt  = st.sidebar.number_input(f"{hop2_name} (g)", min_value=0.0, max_value=500.0, value=0.0, step=5.0)

    hop3_name = st.sidebar.selectbox("Hop 3", ["-"] + AVAILABLE_HOPS, index=0)
    hop3_amt  = st.sidebar.number_input(f"{hop3_name} (g)", min_value=0.0, max_value=500.0, value=0.0, step=5.0)

    hop4_name = st.sidebar.selectbox("Hop 4", ["-"] + AVAILABLE_HOPS, index=0)
    hop4_amt  = st.sidebar.number_input(f"{hop4_name} (g)", min_value=0.0, max_value=500.0, value=0.0, step=5.0)

    st.sidebar.subheader("Malt Bill")
    # We'll just collect them but won't model malt yet.
    malt1_type = st.sidebar.text_input("Malt 1", "AMBER MALT")
    malt1_pct  = st.sidebar.number_input("Malt 1 %", min_value=0.0, max_value=100.0, value=70.0, step=1.0)

    malt2_type = st.sidebar.text_input("Malt 2", "BEST ALE MALT")
    malt2_pct  = st.sidebar.number_input("Malt 2 %", min_value=0.0, max_value=100.0, value=20.0, step=1.0)

    malt3_type = st.sidebar.text_input("Malt 3", "BLACK MALT")
    malt3_pct  = st.sidebar.number_input("Malt 3 %", min_value=0.0, max_value=100.0, value=10.0, step=1.0)

    st.sidebar.subheader("Yeast Strain")
    # We can load yeast strain names from clean_yeast_df if there's a column like 'name'
    if "name" in clean_yeast_df.columns:
        yeast_options = ["-"] + sorted(clean_yeast_df["name"].unique().tolist())
    else:
        yeast_options = ["-"]
    yeast_choice = st.sidebar.selectbox("Select yeast", yeast_options, index=0)

    run_button = st.sidebar.button("Predict Flavor 🧪")

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


###############################################################################
# 7. APP BODY
###############################################################################

def main():
    st.title("🍺 Beer Recipe Digital Twin")
    st.write(
        "Predict hop aroma, malt character, and fermentation profile using trained ML models."
    )

    hop_inputs, malt_inputs, yeast_choice, run_button = sidebar_inputs()

    # Layout columns for chart + narrative
    col_chart, col_notes = st.columns([2, 1])

    if run_button:
        # 1. Build hop model input vector
        hop_feature_df = build_hop_feature_vector(hop_inputs)

        # 2. DEBUG display
        st.subheader("🔬 Debug: hop model input (what the model actually sees)")
        st.write("If this table is all zeros, or missing your chosen hops, we know why the radar looks flat.")
        st.dataframe(hop_feature_df)

        # 3. Predict aroma
        aroma_scores = predict_hop_aroma(hop_feature_df)

        # 4. Plot the radar
        with col_chart:
            fig = plot_radar(aroma_scores)
            st.pyplot(fig, clear_figure=True)

        # 5. Narrative blocks
        with col_notes:
            st.markdown("### Top hop notes:")
            # We'll just sort the predicted intensities (descending)
            # and show the top 2-3 descriptors.
            sorted_aromas = sorted(
                aroma_scores.items(), key=lambda x: x[1], reverse=True
            )
            for axis_name, val in sorted_aromas[:2]:
                st.write(f"- **{axis_name} ({val:.2f})**")

            st.markdown("### Malt character:")
            # crude malt guess: if darkest malt > ~10%, call it 'roasty/bready'
            # else 'bready', 'sweet_malt', etc (you can customize)
            malt_desc = "bready"
            # small quick heuristic:
            malt3_name, malt3_pct = malt_inputs[2]
            if malt3_pct and malt3_pct >= 10:
                malt_desc = "roasty / dark malt"
            st.write(malt_desc)

            st.markdown("### Yeast character:")
            # simple text mapping from yeast name
            yeast_desc = "clean_neutral"
            if isinstance(yeast_choice, str):
                lname = yeast_choice.lower()
                if "cooper" in lname or "ale" in lname:
                    yeast_desc = "fruity_esters, clean_neutral"
                elif "lager" in lname:
                    yeast_desc = "clean_neutral, low esters"
            st.write(yeast_desc)

            st.markdown("### Style direction:")
            # quick heuristic:
            style_desc = "🧪 Experimental / Hybrid"
            if "lager" in yeast_desc:
                style_desc = "🍺 Clean / Neutral Ale direction"
            st.write(style_desc)

    else:
        # Before clicking "Predict Flavor"
        col_chart.markdown("#### Hop Aroma Radar")
        col_chart.write(
            "Pick hops, grams, malt bill %, and yeast in the sidebar, then click **Predict Flavor 🧪**."
        )
        col_notes.markdown("#### Notes will appear here")


###############################################################################
# 8. RUN
###############################################################################

if __name__ == "__main__":
    main()

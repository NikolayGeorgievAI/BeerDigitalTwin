import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(
    page_title="Beer Recipe Digital Twin",
    layout="wide"
)

# -------------------------------------------------
# Load datasets / models
# -------------------------------------------------
@st.cache_resource
def load_hop_model():
    """
    Load hop_aroma_model.joblib and return a tuple:
      (pipeline_estimator, expected_feature_names)

    The .joblib may be:
      - a dict with {"model": pipeline, "feature_names": [...]}, or
      - a bare Pipeline, in which case we try to introspect the feature names.
    """
    raw = joblib.load("hop_aroma_model.joblib")

    # Step 1: unwrap dicts
    if isinstance(raw, dict):
        model_obj = raw.get("model", None)
        feature_names = raw.get("feature_names", None)
    else:
        model_obj = raw
        feature_names = getattr(raw, "feature_names_in_", None)

    if model_obj is None:
        return None, None

    # Step 2: if we still don't have feature_names, try to infer them
    if feature_names is None:
        # Try attribute on the pipeline itself
        if hasattr(model_obj, "feature_names_in_"):
            feature_names = list(model_obj.feature_names_in_)
        # Try top-level get_feature_names_out()
        elif hasattr(model_obj, "get_feature_names_out"):
            try:
                feature_names = list(model_obj.get_feature_names_out())
            except Exception:
                feature_names = None
        # Try first step of pipeline (often a ColumnTransformer or vectorizer)
        if feature_names is None and hasattr(model_obj, "named_steps"):
            for step_name, step_est in model_obj.named_steps.items():
                if hasattr(step_est, "get_feature_names_out"):
                    try:
                        feature_names = list(step_est.get_feature_names_out())
                        break
                    except Exception:
                        pass
                if hasattr(step_est, "feature_names_in_"):
                    try:
                        feature_names = list(step_est.feature_names_in_)
                        break
                    except Exception:
                        pass

    # As a last resort, leave feature_names as None
    return model_obj, feature_names


@st.cache_resource
def load_yeast_df():
    return pd.read_pickle("clean_yeast_df.pkl")

@st.cache_resource
def load_malt_df():
    return pd.read_pickle("clean_malt_df.pkl")


# -------------------------------------------------
# Helper: Make hop feature DF for prediction
# -------------------------------------------------
def build_hop_feature_df(hop_inputs, feature_names):
    """
    hop_inputs: list of (hop_name, grams)
        e.g. [("Simcoe", 100.0), ("Amarillo", 50.0), ...]
    feature_names: list[str] the model expects, e.g. 
        ["hop_Adeena", "hop_Admiral", "hop_African Queen", ...]

    Return:
      aligned_df: DataFrame with exactly those columns in that order
      debug_df:   DataFrame with user's nonzero grams per hop
    """
    # 1. Aggregate grams by hop name from user input
    agg = {}
    for hop_name, g in hop_inputs:
        if hop_name and g > 0:
            agg[hop_name] = agg.get(hop_name, 0.0) + float(g)

    # debug_df for display
    debug_df = pd.DataFrame([agg]) if agg else pd.DataFrame([{}])

    # 2. Build the row we'll feed the model
    row = {}

    if feature_names:
        # The model told us exactly which columns it wants.
        # We'll fill them all in order, defaulting to 0.0
        for feat in feature_names:
            # trained features look like hop_<Variety Name>
            if feat.startswith("hop_"):
                base_name = feat.replace("hop_", "")
                row[feat] = agg.get(base_name, 0.0)
            else:
                # if there are any non-hop features, just set 0 for now
                row[feat] = 0.0
        aligned_df = pd.DataFrame([row], columns=feature_names)
    else:
        # We don't know the feature_names -> fall back to whatever we have
        # (This WILL still error with the pipeline, but we keep it for debug.)
        for k, v in agg.items():
            row[f"hop_{k}"] = v
        aligned_df = pd.DataFrame([row])

    return aligned_df, debug_df



# -------------------------------------------------
# Hop aroma prediction wrapper
# -------------------------------------------------
def predict_hop_aroma(hop_model, hop_feature_names, user_hops_dict):
    """
    hop_model           : the loaded scikit-learn Pipeline
    hop_feature_names   : FULL list of feature names the model was trained on
                          e.g. ["hop_Adeena", "hop_Admiral", ...]
    user_hops_dict      : {"Amarillo": grams, "Astra": grams, ...}  (only nonzero hops)
    returns:
        aroma_vector (list or np.array of length 8) or zeros if fail
        debug_info (dict for debugging UI)
    """

    debug_info = {}

    if hop_model is None:
        debug_info["error"] = "hop_model is None"
        return [0]*8, debug_info

    # 1. create all-zero row with full training columns
    aligned_df = pd.DataFrame([[0.0]*len(hop_feature_names)],
                              columns=hop_feature_names)

    # 2. set the grams for each hop the user actually chose
    for hop_name, grams in user_hops_dict.items():
        col_name = f"hop_{hop_name}"
        if col_name in aligned_df.columns:
            aligned_df.loc[0, col_name] = grams

    debug_info["aligned_df"] = aligned_df.copy()

    try:
        pred = hop_model.predict(aligned_df)
        # pred shape: (1, 8) maybe -> tropical, citrus, fruity, resinous, etc
        aroma_vector = pred[0].tolist()
        debug_info["prediction_raw"] = aroma_vector
    except Exception as e:
        aroma_vector = [0]*8
        debug_info["exception"] = f"Predict exception: {e}"

    return aroma_vector, debug_info



# -------------------------------------------------
# Radar plot helper
# -------------------------------------------------
def plot_radar(aroma_scores):
    """
    aroma_scores: dict {axis_name: value}
      e.g. {"fruity":0.3, "citrus":0.1, ...}
    Renders a matplotlib polar (radar) chart and returns fig.
    """
    categories = list(aroma_scores.keys())
    values = [aroma_scores[c] for c in categories]

    # close the loop
    categories += [categories[0]]
    values += [values[0]]

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)

    fig, ax = plt.subplots(
        figsize=(6,6),
        subplot_kw=dict(polar=True)
    )

    ax.plot(angles, values, color="#2a4b8d", linewidth=2)
    ax.fill(angles, values, color="#2a4b8d", alpha=0.25)

    ax.set_xticks(angles)
    ax.set_xticklabels(categories, fontsize=11)

    # radial labels styling
    ax.set_yticklabels([])
    ax.spines["polar"].set_color("#666666")
    ax.spines["polar"].set_linewidth(1)

    # show numeric annotation in the middle for debugging (optional)
    # You can comment this out if you don't want a box in the center.
    center_val = np.mean(values[:-1]) if len(values)>1 else 0.0
    ax.text(
        0.0, 0.0,
        f"{center_val:.2f}",
        ha="center", va="center",
        bbox=dict(boxstyle="round", fc="white", ec="#2a4b8d")
    )

    return fig


# -------------------------------------------------
# Sidebar input builder
# -------------------------------------------------
def sidebar_inputs(yeast_df):
    st.sidebar.header("Hop Bill (g)")

    # up to 4 hops
    hop_names = [
        "Astra","Amarillo","Citra","Simcoe",
        "Mosaic","Galaxy","Nelson Sauvin","Vic Secret"
    ]

    hop_inputs = []
    for i in range(1,5):
        hop_name = st.sidebar.selectbox(
            f"Hop {i}",
            options=[""] + hop_names,
            index=0,
            key=f"hop{i}_name"
        )
        hop_g = st.sidebar.number_input(
            f"{hop_name or ' '} (g)",
            min_value=0.0, max_value=500.0,
            value=0.0, step=5.0,
            key=f"hop{i}_grams"
        )
        hop_inputs.append((hop_name, hop_g))

    st.sidebar.subheader("Malt Bill")
    # Basic 3-malt bill
    malt1 = st.sidebar.selectbox("Malt 1", ["AMBER MALT","BLACK MALT","BEST ALE MALT"], index=0)
    malt1_perc = st.sidebar.number_input("Malt 1 %", 0.0, 100.0, 70.0, 1.0)
    malt2 = st.sidebar.selectbox("Malt 2", ["AMBER MALT","BLACK MALT","BEST ALE MALT"], index=1)
    malt2_perc = st.sidebar.number_input("Malt 2 %", 0.0, 100.0, 20.0, 1.0)
    malt3 = st.sidebar.selectbox("Malt 3", ["AMBER MALT","BLACK MALT","BEST ALE MALT"], index=2)
    malt3_perc = st.sidebar.number_input("Malt 3 %", 0.0, 100.0, 10.0, 1.0)

    st.sidebar.subheader("Yeast Strain")

    # build yeast options from yeast_df["Name"]
    yeast_options = ["-"] + sorted(yeast_df["Name"].unique().tolist())
    yeast_choice = st.sidebar.selectbox(
        "Select yeast",
        options=yeast_options,
        index=0
    )

    run_button = st.sidebar.button("Predict Flavor 🧪")

    malt_inputs = {
        "malt1": (malt1, malt1_perc),
        "malt2": (malt2, malt2_perc),
        "malt3": (malt3, malt3_perc),
    }

    return hop_inputs, malt_inputs, yeast_choice, run_button


# -------------------------------------------------
# Main layout
# -------------------------------------------------
def main():
    hop_model, hop_feature_names = load_hop_model()
    yeast_df = load_yeast_df()
    # malt_df = load_malt_df()  # not deeply used yet in this snippet

    hop_inputs, malt_inputs, yeast_choice, run_button = sidebar_inputs(yeast_df)

    st.title("🍺 Beer Recipe Digital Twin")
    st.write(
        "Predict hop aroma, malt character, and fermentation profile using trained ML models "
        "(work in progress)."
    )

    # Build features for hop aroma
    aligned_df, raw_debug_df = build_hop_feature_df(hop_inputs, hop_feature_names)

    aroma_scores = {ax:0.0 for ax in
        ["fruity","citrus","tropical","earthy","spicy","herbal","floral","resinous"]
    }
    model_error = None

    if run_button:
        aroma_scores, model_error = predict_hop_aroma(hop_model, aligned_df)

    # --- LAYOUT ---
    col1, col2 = st.columns([2,1])

    with col1:
        st.subheader("Hop Aroma Radar")

        fig = plot_radar(aroma_scores)
        st.pyplot(fig, use_container_width=True)

    with col2:
        st.subheader("Top hop notes:")
        st.write(f"• tropical ({aroma_scores['tropical']:.2f})")
        st.write(f"• citrus ({aroma_scores['citrus']:.2f})")
        st.markdown("---")

        st.subheader("Malt character:")
        # placeholder example based on malt bill
        st.write("bready")
        st.markdown("---")

        st.subheader("Yeast character:")
        # placeholder example: just show some descriptors from columns
        # if yeast_choice != "-" we can look it up:
        if yeast_choice != "-":
            row = yeast_df[yeast_df["Name"] == yeast_choice]
            # e.g. show these boolean-ish columns
            notes = []
            if not row.empty:
                r0 = row.iloc[0]
                if "fruity_esters" in r0 and r0["fruity_esters"] == 1:
                    notes.append("fruity_esters")
                if "clean_neutral" in r0 and r0["clean_neutral"] == 1:
                    notes.append("clean_neutral")
                if "phenolic_spicy" in r0 and r0["phenolic_spicy"] == 1:
                    notes.append("phenolic_spicy")
            if notes:
                st.write(", ".join(notes))
            else:
                st.write("clean / neutral")
        else:
            st.write("clean neutral")

        st.markdown("---")

        st.subheader("Style direction:")
        st.write("🧪 Experimental / Hybrid")

    # --- DEBUG PANELS ---
    with st.expander("🧪 Debug: hop model input / prediction"):
        st.write("hop_model is None?", hop_model is None)
        st.write("hop_feature_names:", hop_feature_names)

        if model_error:
            st.error(f"Model predict() error: {model_error}")

        st.write("aligned_df passed to model:")
        st.dataframe(aligned_df, use_container_width=True)

        st.write("User aggregate hop grams by hop variety:")
        st.dataframe(raw_debug_df, use_container_width=True)

        if hop_model is not None:
            st.write("type(hop_model):", type(hop_model))
            st.write("dir(hop_model)[:20]:", dir(hop_model)[:20])

    with st.expander("🧬 Debug: yeast dataset"):
        st.write("Columns:", list(yeast_df.columns))
        st.dataframe(yeast_df.head(20), use_container_width=True)


if __name__ == "__main__":
    main()

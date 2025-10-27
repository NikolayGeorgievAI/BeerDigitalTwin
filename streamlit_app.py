#########################
# Beer Recipe Digital Twin
# streamlit_app.py
#
# What this app does (current version):
# - Sidebar: choose hop bill (up to 4 hops), malt bill (3 malts), yeast strain
# - "Predict Flavor" runs a hop aroma model (if available) to estimate aroma radar
# - Displays predicted hop aroma on a spider/radar chart
# - Shows simple malt / yeast / style text summaries
# - Shows debug info with all inputs + internal state
#
# NOTE: Hop model predictions will only be meaningful if hop_aroma_model.joblib
# actually contains a trained estimator + feature names. If not, you’ll just
# get 0.00s for now (which is expected).
#########################

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import joblib
from typing import List, Dict, Any, Tuple, Optional


#########################
# --- Streamlit page setup
#########################

st.set_page_config(
    page_title="Beer Recipe Digital Twin",
    page_icon="🍺",
    layout="wide",
)


#########################
# --- Helper classes / utils
#########################


class ModelWrapper:
    """
    A thin wrapper around the hop aroma model so we can keep:
      - model: the actual estimator/pipeline (or dict if not exported correctly)
      - feature_names: list of columns like ["hop_Simcoe", "hop_Citra", ...]
    """

    def __init__(self, raw_obj: Any):
        """
        raw_obj can be:
          - a dict with keys {"model": <sklearn Pipeline>, "feature_names": [...]}
          - just an sklearn Pipeline
          - None
        """
        self.model = None
        self.feature_names = None

        if raw_obj is None:
            return

        # case 1: dict exported like {"model": pipeline, "feature_names": [...]}
        if isinstance(raw_obj, dict):
            self.model = raw_obj.get("model", None)
            self.feature_names = raw_obj.get("feature_names", None)

        else:
            # maybe joblib was just a pipeline
            self.model = raw_obj
            # no feature names discovered
            self.feature_names = None


def safe_zero_aroma() -> Dict[str, float]:
    """
    If the model can't predict, or we don't have valid inputs,
    return zero for all aroma axes so the radar won't crash.
    """
    return {
        "fruity": 0.0,
        "citrus": 0.0,
        "tropical": 0.0,
        "earthy": 0.0,
        "spicy": 0.0,
        "herbal": 0.0,
        "floral": 0.0,
        "resinous": 0.0,
    }


def build_user_hop_df(hop_entries: List[Dict[str, Any]],
                      feature_names: List[str]) -> pd.DataFrame:
    """
    Turn the user's hop bill into a single-row DataFrame with columns
    exactly matching model feature names like ["hop_Simcoe","hop_Citra",...].

    hop_entries is a list of dicts like:
      [{"name": "Simcoe", "amt": 50.0}, {"name": "Citra", "amt": 20.0}, ...]
    We'll sum grams by hop name and place them in the matching "hop_<name>" column.
    """
    # Sum grams per hop variety
    grams_by_name = {}
    for h in hop_entries:
        nm = h.get("name", "-")
        amt = float(h.get("amt", 0.0) or 0.0)
        if nm and nm != "-" and amt > 0:
            grams_by_name[nm] = grams_by_name.get(nm, 0.0) + amt

    # Build aligned row with any missing cols filled as 0
    row_data = {}
    for feat in feature_names:
        if feat.startswith("hop_"):
            hop_name = feat[4:]
        else:
            hop_name = feat
        row_data[feat] = grams_by_name.get(hop_name, 0.0)

    df = pd.DataFrame([row_data])
    return df


def predict_hop_aroma(wrapper: ModelWrapper,
                      hop_entries: List[Dict[str, Any]]) -> Tuple[Dict[str, float], str]:
    """
    Uses the hop model (if available) to predict an aroma vector for:
      fruity, citrus, tropical, earthy, spicy, herbal, floral, resinous

    Returns:
      (scores_dict, model_error_message)

    scores_dict is safe_zero_aroma() if model can't run.
    model_error_message is "" if all good, else a short string we can show in debug.
    """
    # default to silence
    model_error = ""
    if wrapper is None or wrapper.model is None:
        return safe_zero_aroma(), "No hop model loaded."

    # We need a list of features so we can align user hops
    feature_names = wrapper.feature_names
    if not feature_names or not isinstance(feature_names, (list, tuple)):
        # We can't align columns -> bail out with zeros
        return safe_zero_aroma(), "Model has no usable feature_names."

    # Build aligned DF from user hops
    aligned_df = build_user_hop_df(hop_entries, list(feature_names))

    # If that row is empty or all zeros, we can still attempt predict,
    # but let's guard in case shape mismatch triggers errors.
    try:
        # The "wrapper.model" should be a fitted estimator or pipeline with .predict
        if not hasattr(wrapper.model, "predict"):
            return safe_zero_aroma(), "Loaded object doesn't implement .predict()."

        y_pred = wrapper.model.predict(aligned_df)

        # y_pred could be:
        #  - shape (1, 8) numeric array [fruity,citrus,tropical,earthy,spicy,herbal,floral,resinous]
        #  - shape (1,) dict, etc...
        # We'll handle the (1,8) numeric array case first.
        aroma_axes = [
            "fruity",
            "citrus",
            "tropical",
            "earthy",
            "spicy",
            "herbal",
            "floral",
            "resinous",
        ]
        scores = safe_zero_aroma()

        if isinstance(y_pred, (list, np.ndarray)):
            arr = np.array(y_pred)
            # If it's 2D and width is 8, map them directly
            if arr.ndim == 2 and arr.shape[1] == len(aroma_axes):
                for i, axis in enumerate(aroma_axes):
                    scores[axis] = float(arr[0, i])
            # If it's 1D of length 8
            elif arr.ndim == 1 and arr.shape[0] == len(aroma_axes):
                for i, axis in enumerate(aroma_axes):
                    scores[axis] = float(arr[i])
            else:
                model_error = (
                    f"Unexpected predict() shape {arr.shape}, using zeros."
                )
        else:
            # fallback for weird return types
            model_error = "Model predict() returned non-array, using zeros."

        return scores, model_error

    except Exception as e:
        # If anything blows up, we still want the app to run
        model_error = f"Predict exception: {e}"
        return safe_zero_aroma(), model_error


def make_radar(aroma_scores: Dict[str, float]) -> plt.Figure:
    """
    Build a nice radar / spider chart. The labels are fixed in a
    consistent order around the circle.
    """
    labels = [
        "fruity",
        "citrus",
        "tropical",
        "earthy",
        "spicy",
        "herbal",
        "floral",
        "resinous",
    ]

    values = [aroma_scores.get(lbl, 0.0) for lbl in labels]

    # close the loop
    labels_loop = labels + [labels[0]]
    values_loop = values + [values[0]]

    # angles for each axis
    angles = np.linspace(0, 2 * np.pi, len(labels_loop), endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])  # ensure closure

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2.0)      # start at top
    ax.set_theta_direction(-1)            # clockwise

    # gridlines/axes
    ax.set_xticks(np.linspace(0, 2 * np.pi, len(labels), endpoint=False))
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_yticklabels([])

    # radial ticks just for context
    ax.set_rgrids([0.2, 0.4, 0.6, 0.8, 1.0], angle=90, fontsize=10, color="#555")
    ax.set_ylim(0, 1.0)

    # fill polygon
    ax.plot(angles, values_loop, color="#1f2a44", linewidth=2)
    ax.fill(angles, values_loop, color="#1f2a44", alpha=0.15)

    # big center label = average intensity
    avg_val = float(np.mean(values)) if len(values) else 0.0
    ax.text(
        0.0,
        0.0,
        f"{avg_val:.2f}",
        ha="center",
        va="center",
        fontsize=20,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", fc="#e9eefc", ec="#1f2a44"),
    )

    ax.set_title("Hop Aroma Radar", fontsize=22, pad=20, fontweight="bold")
    fig.tight_layout()
    return fig


#########################
# --- Data/model loading
#########################

@st.cache_resource(show_spinner=False)
def load_datasets_and_models():
    """
    Load:
      - clean_yeast_df.pkl
      - clean_malt_df.pkl
      - hop_aroma_model.joblib

    We wrap the hop model in ModelWrapper so we always return something
    predictable. If the joblib file doesn't contain usable metadata
    (like feature_names), we inject a fallback hop variety list so the
    dropdowns aren't empty.
    """
    # yeast
    try:
        yeast_df = pd.read_pickle("clean_yeast_df.pkl")
    except Exception:
        yeast_df = pd.DataFrame()

    # malt
    try:
        malt_df = pd.read_pickle("clean_malt_df.pkl")
    except Exception:
        malt_df = pd.DataFrame()

    # hop model
    try:
        raw_model = joblib.load("hop_aroma_model.joblib")
        hop_wrapper = ModelWrapper(raw_model)
    except Exception:
        hop_wrapper = ModelWrapper(None)

    # Fallback hop feature names if model didn't expose any
    # This controls both:
    #  1) hop dropdown contents
    #  2) columns we build for prediction
    if not hop_wrapper.feature_names:
        FALLBACK_HOP_COLUMNS = [
            "hop_Simcoe",
            "hop_Amarillo",
            "hop_Citra",
            "hop_Mosaic",
            "hop_Centennial",
            "hop_Cascade",
            "hop_Galaxy",
            "hop_Nelson",
        ]
        hop_wrapper.feature_names = FALLBACK_HOP_COLUMNS

    return yeast_df, malt_df, hop_wrapper


#########################
# --- Sidebar input builder
#########################

def sidebar_inputs(yeast_df: pd.DataFrame,
                   malt_df: pd.DataFrame,
                   hop_wrapper: ModelWrapper):
    """
    Renders the Streamlit sidebar controls and returns:
      hop_entries: list of {name, amt}
      malt_entries: list of {name, pct}
      yeast_choice: string
      run_button: bool
    """

    st.sidebar.header("Model Inputs")

    ###################
    # Hops
    ###################
    st.sidebar.subheader("Hop Bill (g)")

    # Choices for the hop dropdown:
    hop_choices = ["-"]
    hop_varieties = []
    if hop_wrapper and hop_wrapper.feature_names:
        for feat in hop_wrapper.feature_names:
            if feat.startswith("hop_"):
                hop_varieties.append(feat[4:])
            else:
                # just in case it's already 'Simcoe' instead of 'hop_Simcoe'
                hop_varieties.append(feat)
    hop_varieties = sorted(set(hop_varieties))
    if hop_varieties:
        hop_choices = ["-"] + hop_varieties

    hop1_name = st.sidebar.selectbox("Hop 1", hop_choices, index=0, key="hop1_name")
    hop1_amt  = st.sidebar.number_input(
        "(g)",
        min_value=0.0,
        max_value=500.0,
        value=0.0,
        step=5.0,
        key="hop1_amt",
    )

    hop2_name = st.sidebar.selectbox("Hop 2", hop_choices, index=0, key="hop2_name")
    hop2_amt  = st.sidebar.number_input(
        "(g)",
        min_value=0.0,
        max_value=500.0,
        value=0.0,
        step=5.0,
        key="hop2_amt",
    )

    hop3_name = st.sidebar.selectbox("Hop 3", hop_choices, index=0, key="hop3_name")
    hop3_amt  = st.sidebar.number_input(
        "(g)",
        min_value=0.0,
        max_value=500.0,
        value=0.0,
        step=5.0,
        key="hop3_amt",
    )

    hop4_name = st.sidebar.selectbox("Hop 4", hop_choices, index=0, key="hop4_name")
    hop4_amt  = st.sidebar.number_input(
        "(g)",
        min_value=0.0,
        max_value=500.0,
        value=0.0,
        step=5.0,
        key="hop4_amt",
    )

    hop_entries = [
        {"name": hop1_name, "amt": hop1_amt},
        {"name": hop2_name, "amt": hop2_amt},
        {"name": hop3_name, "amt": hop3_amt},
        {"name": hop4_name, "amt": hop4_amt},
    ]

    ###################
    # Malt
    ###################
    st.sidebar.subheader("Malt Bill (%)")

    # We build a malt list from the malt_df if present; otherwise fallback.
    # Expect malt_df to have a column with names. Let's guess there's "Name"
    # If not, we just fallback to a couple example malts.
    if not malt_df.empty and "Name" in malt_df.columns:
        malt_names = sorted(set(malt_df["Name"].astype(str).tolist()))
    else:
        malt_names = [
            "BEST ALE MALT",
            "BLACK MALT",
            "CARA GOLD MALT",
            "PILSNER MALT",
            "MUNICH MALT",
            "-",
        ]
    # Guarantee "-" is first for "none"
    malt_names = ["-"] + [m for m in malt_names if m != "-"]

    malt1_name = st.sidebar.selectbox("Malt 1 name", malt_names, index=0, key="malt1_name")
    malt1_pct  = st.sidebar.number_input(
        "Malt 1 %",
        min_value=0.0,
        max_value=100.0,
        value=0.0,
        step=1.0,
        key="malt1_pct",
    )

    malt2_name = st.sidebar.selectbox("Malt 2 name", malt_names, index=0, key="malt2_name")
    malt2_pct  = st.sidebar.number_input(
        "Malt 2 %",
        min_value=0.0,
        max_value=100.0,
        value=0.0,
        step=1.0,
        key="malt2_pct",
    )

    malt3_name = st.sidebar.selectbox("Malt 3 name", malt_names, index=0, key="malt3_name")
    malt3_pct  = st.sidebar.number_input(
        "Malt 3 %",
        min_value=0.0,
        max_value=100.0,
        value=0.0,
        step=1.0,
        key="malt3_pct",
    )

    malt_entries = [
        {"name": malt1_name, "pct": malt1_pct},
        {"name": malt2_name, "pct": malt2_pct},
        {"name": malt3_name, "pct": malt3_pct},
    ]

    ###################
    # Yeast
    ###################
    st.sidebar.subheader("Yeast Strain")

    # We try to build a nice yeast dropdown from yeast_df["Name"]
    # If that's missing, we'll fallback to "-"
    if not yeast_df.empty and "Name" in yeast_df.columns:
        yeast_list = yeast_df["Name"].astype(str).tolist()
        yeast_list = ["-"] + yeast_list
    else:
        yeast_list = [
            "-",
            "Cooper Ale",
            "Edme Ale Yeast",
            "Nottingham Ale Yeast",
            "WLP001 California Ale",
            "WLP002 English Ale",
        ]

    yeast_choice = st.sidebar.selectbox("Select yeast", yeast_list, index=0, key="yeast_choice")

    ###################
    # Run button
    ###################
    run_button = st.sidebar.button("Predict Flavor 🧪")

    return hop_entries, malt_entries, yeast_choice, run_button


#########################
# --- Display helpers
#########################

def summarize_malt(malt_entries: List[Dict[str, Any]]) -> str:
    """
    Placeholder. Right now we just say 'bready' no matter what.
    Eventually you could look up malt flavor tags from malt_df and
    build a description.
    """
    # TODO: incorporate actual malt_df flavor columns
    return "bready"


def summarize_yeast(yeast_choice: str, yeast_df: pd.DataFrame) -> str:
    """
    Very simplified yeast character summary. We try to match the chosen
    yeast row in yeast_df and derive general trait tags like
    fruity_esters / clean_neutral / phenolic_spicy, etc.
    For now, fallback to "clean / neutral".
    """
    if yeast_choice and yeast_choice != "-" and not yeast_df.empty and "Name" in yeast_df.columns:
        row = yeast_df[yeast_df["Name"] == yeast_choice]
        if not row.empty:
            # crude logic: check columns like 'clean_neutral', 'fruity_esters'
            # and build a tag string.
            tags = []
            if "clean_neutral" in row.columns:
                if row["clean_neutral"].iloc[0] == 1 or str(row["clean_neutral"].iloc[0]).lower() in ["1", "true", "yes"]:
                    tags.append("clean / neutral")
            if "fruity_esters" in row.columns:
                val = row["fruity_esters"].iloc[0]
                # if numeric non-zero, we assume 'fruity esters'
                try:
                    if float(val) > 0:
                        tags.append("fruity esters")
                except:
                    pass
            if "phenolic_spicy" in row.columns:
                val = row["phenolic_spicy"].iloc[0]
                try:
                    if float(val) > 0:
                        tags.append("phenolic / spicy")
                except:
                    pass
            if tags:
                return ", ".join(sorted(set(tags)))

    # fallback
    return "clean / neutral"


def guess_style_direction(yeast_desc: str, malt_desc: str) -> str:
    """
    Another placeholder. You could implement real style guidance
    (e.g. "Experimental / Hybrid", "Clean / Neutral Ale", etc)
    based on yeast_desc + malt_desc.
    """
    if "clean" in yeast_desc and "bready" in malt_desc:
        return "🍻 Experimental / Hybrid"
    elif "phenolic" in yeast_desc:
        return "Belgian / Phenolic"
    else:
        return "Clean / Neutral Ale direction"


def best_two_aromas(aroma_scores: Dict[str, float]) -> List[Tuple[str, float]]:
    """
    Take the radar dict (fruity,citrus,...) and return the top two notes
    sorted by descending score: [("tropical",0.91), ("citrus",0.83)]
    """
    items = list(aroma_scores.items())
    items.sort(key=lambda kv: kv[1], reverse=True)
    return items[:2]


#########################
# --- Main app layout
#########################

def main():
    st.title("🍺 Beer Recipe Digital Twin")
    st.caption(
        "Predict hop aroma, malt character, and fermentation profile "
        "using trained ML models (work in progress)."
    )

    # 1. Load data + model
    yeast_df, malt_df, hop_wrapper = load_datasets_and_models()

    # 2. Get user inputs from sidebar
    hop_entries, malt_entries, yeast_choice, run_button = sidebar_inputs(
        yeast_df, malt_df, hop_wrapper
    )

    # 3. Only run model when button is clicked OR first load we can still do a "silent" run
    # We'll do the run unconditionally so you always see something, then you're free
    # to tweak after pressing the button.
    aroma_scores, model_error = predict_hop_aroma(hop_wrapper, hop_entries)

    # 4. Build text summaries
    malt_desc = summarize_malt(malt_entries)
    yeast_desc = summarize_yeast(yeast_choice, yeast_df)
    style_guess = guess_style_direction(yeast_desc, malt_desc)

    # 5. Layout columns for radar + textual summaries
    col_radar, col_text = st.columns([2, 1])

    with col_radar:
        st.subheader("Hop Aroma Radar")
        fig = make_radar(aroma_scores)
        st.pyplot(fig, use_container_width=True)

    with col_text:
        st.subheader("Top hop notes:")
        top2 = best_two_aromas(aroma_scores)
        if not top2:
            st.write("• (no hop aroma detected)")
        else:
            for note, val in top2:
                st.write(f"• {note} ({val:.2f})")

        st.markdown("---")

        st.subheader("Malt character:")
        st.write(malt_desc)

        st.subheader("Yeast character:")
        st.write(yeast_desc)

        st.subheader("Style direction:")
        st.write(style_guess)

        st.subheader("Hops used by the model:")
        used_hops_str = ", ".join(
            [f"{h['name']} ({h['amt']:.0f}g)" for h in hop_entries if h["name"] != "-" and h["amt"] > 0]
        )
        st.write(used_hops_str if used_hops_str else "—")

    # 6. Debug panel
    with st.expander("🔬 Debug info"):
        st.markdown("**User hop entries:**")
        st.write(hop_entries)

        st.markdown("**User malt entries:**")
        st.write(malt_entries)

        st.markdown("**Selected yeast:**")
        st.write(yeast_choice)

        st.markdown("**Aroma scores dict:**")
        st.write(aroma_scores)

        st.markdown("**Model error (if any):**")
        st.write(model_error)

        st.markdown("**Wrapper feature_names:**")
        st.write(hop_wrapper.feature_names if hop_wrapper else None)

        st.markdown("**Yeast dataset columns:**")
        st.write(list(yeast_df.columns) if not yeast_df.empty else "(no yeast_df)")


if __name__ == "__main__":
    main()

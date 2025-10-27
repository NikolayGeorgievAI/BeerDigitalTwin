############################
# streamlit_app.py
############################

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any, Union

# ---------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Beer Recipe Digital Twin",
    layout="wide",
)

# ---------------------------------------------------------------------
# Helper classes / functions
# ---------------------------------------------------------------------

class ModelWrapper:
    """
    Wraps a trained hop aroma model (or a placeholder dict) so we have
    a consistent interface:
        - .model: the actual sklearn pipeline (or None / dict)
        - .feature_names: list[str] we expect as columns for predict()
    """
    def __init__(self, raw_obj: Any):
        # Case 1: We actually got a dict like
        #   {"model": sklearn_pipeline, "feature_names": [...]}
        if isinstance(raw_obj, dict):
            self.model = raw_obj.get("model", None)
            self.feature_names = raw_obj.get("feature_names", None)
        else:
            # Case 2: maybe we loaded an sklearn Pipeline directly
            self.model = raw_obj
            # try to read a built-in attribute for columns:
            if raw_obj is not None and hasattr(raw_obj, "feature_names_in_"):
                self.feature_names = list(raw_obj.feature_names_in_)
            else:
                self.feature_names = None

    def predict(self, user_hops_df: pd.DataFrame) -> Optional[np.ndarray]:
        """
        1) Align user_hops_df columns to self.feature_names.
        2) Run self.model.predict() if possible.
        Returns np.ndarray or None if predict fails.
        """
        if self.model is None:
            return None

        # If we don't know the feature names, we can't align -> None
        if not self.feature_names:
            return None

        # Build aligned df with exactly these columns
        aligned = pd.DataFrame(
            {feat: user_hops_df[feat] if feat in user_hops_df.columns else 0.0
             for feat in self.feature_names}
        )

        try:
            preds = self.model.predict(aligned)
        except Exception:
            return None

        return preds


def safe_zero_aroma() -> Dict[str, float]:
    """
    Default 8-dimension aroma dict. This prevents the radar plot
    from crashing if the model can't produce valid output yet.
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


def make_radar(ax, aroma_scores: Union[Dict[str,float], List[float], np.ndarray]):
    """
    Draws a radar / spider plot of hop aroma.

    aroma_scores:
      - dict {"fruity":..., "citrus":..., ...} with exactly these 8 keys
      - OR list-like of length >= 8 (we'll trim/pad to 8)

    We always return a valid spiderweb so the plot won't crash.
    """

    categories = [
        "fruity",
        "citrus",
        "tropical",
        "earthy",
        "spicy",
        "herbal",
        "floral",
        "resinous",
    ]
    num_vars = len(categories)

    # Normalize aroma_scores into a list of length num_vars
    if aroma_scores is None:
        vals = [0.0] * num_vars
    elif isinstance(aroma_scores, dict):
        vals = [float(aroma_scores.get(cat, 0.0)) for cat in categories]
    else:
        # assume list-like / array
        raw_list = list(aroma_scores)
        # pad / trim
        if len(raw_list) < num_vars:
            raw_list = raw_list + [0.0]*(num_vars - len(raw_list))
        elif len(raw_list) > num_vars:
            raw_list = raw_list[:num_vars]
        vals = [float(x) for x in raw_list]

    # Close the polygon
    vals_closed = vals + [vals[0]]

    # Angles
    angles = np.linspace(0, 2*np.pi, num_vars, endpoint=False).tolist()
    angles_closed = angles + [angles[0]]

    # Clear axis in case this is re-used
    ax.clear()

    # Polar setup
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Grid / ticks
    ax.set_rlabel_position(0)
    ax.set_ylim(0, 1.0)
    ax.set_xticks(angles)
    ax.set_xticklabels(categories)

    r_levels = [0.2, 0.4, 0.6, 0.8, 1.0]
    ax.set_yticks(r_levels)
    ax.set_yticklabels([f"{r:.1f}" for r in r_levels])

    # Style
    ax.grid(color="#999999", linestyle="--", linewidth=0.8, alpha=0.6)

    for spine in ax.spines.values():
        spine.set_color("#1f2a44")
        spine.set_linewidth(2)

    # Plot fill
    ax.plot(angles_closed, vals_closed, color="#1f2a44", linewidth=2)
    ax.fill(angles_closed, vals_closed, color="#1f2a44", alpha=0.25)

    # Middle label = average intensity
    avg_val = float(np.mean(vals)) if len(vals) else 0.0
    ax.text(
        0.5,
        0.5,
        f"{avg_val:.2f}",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.4",
            fc="#e9edf8",
            ec="#1f2a44",
            lw=2,
        ),
    )


def build_user_hop_df(hop_entries: List[Dict[str, Union[str,float]]],
                      feature_names: Optional[List[str]]) -> pd.DataFrame:
    """
    Turn user's hop bill into a single-row dataframe of {hop_<variety>: grams}.
    We'll normalize hop names to columns like hop_Simcoe, hop_Amarillo, etc.
    The returned dataframe is 1 row.

    feature_names is the list of columns the model expects.
    If it's None, we just build from what the user gave us.
    """

    # aggregate grams by hop name
    grams_by_hop = {}
    for h in hop_entries:
        name = h["name"]
        amt  = h["amt"]
        if not name or name == "-":
            continue
        g = grams_by_hop.get(name, 0.0)
        grams_by_hop[name] = g + float(amt)

    # convert to columns "hop_<name>": grams
    row_dict = {}
    for hop_name, grams in grams_by_hop.items():
        col = f"hop_{hop_name}"
        row_dict[col] = grams

    raw_df = pd.DataFrame([row_dict])

    # If we have a known feature list, align to it;
    # else just return raw_df
    if feature_names:
        aligned = {
            feat: raw_df[feat] if feat in raw_df.columns else 0.0
            for feat in feature_names
        }
        return pd.DataFrame(aligned)
    else:
        return raw_df


def predict_hop_aroma(wrapper: ModelWrapper,
                      hop_entries: List[Dict[str, Union[str,float]]]) -> Dict[str,float]:
    """
    Calls wrapper.predict() on user hop data, then tries to turn
    the model output into {flavor: value} for 8 spider axes.

    If we can't get predictions, return safe_zero_aroma().
    """
    # Build user hop dataframe aligned to model features
    user_df = build_user_hop_df(hop_entries, wrapper.feature_names)

    preds = wrapper.predict(user_df)
    if preds is None:
        return safe_zero_aroma()

    # We don't know exact shape of your model output. We'll assume either:
    #  - a single row of 8 floats (one per flavor axis), OR
    #  - something 1D we can flatten
    arr = np.array(preds)
    arr = arr.flatten()

    # Force length 8
    categories = [
        "fruity",
        "citrus",
        "tropical",
        "earthy",
        "spicy",
        "herbal",
        "floral",
        "resinous",
    ]
    if len(arr) < len(categories):
        arr = np.concatenate([arr, np.zeros(len(categories)-len(arr))])
    elif len(arr) > len(categories):
        arr = arr[:len(categories)]

    return {cat: float(val) for cat, val in zip(categories, arr)}


# ---------------------------------------------------------------------
# Load data and models
# ---------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_datasets_and_models():
    """
    Loads:
      - clean_yeast_df.pkl
      - clean_malt_df.pkl
      - hop_aroma_model.joblib (wrapped)

    We handle missing files gracefully.
    """
    yeast_df = None
    malt_df = None
    hop_wrapper = ModelWrapper(None)

    # Yeast data
    try:
        yeast_df = pd.read_pickle("clean_yeast_df.pkl")
    except Exception:
        yeast_df = pd.DataFrame()

    # Malt data
    try:
        malt_df = pd.read_pickle("clean_malt_df.pkl")
    except Exception:
        malt_df = pd.DataFrame()

    # Hop model
    try:
        raw_model = joblib.load("hop_aroma_model.joblib")
        hop_wrapper = ModelWrapper(raw_model)
    except Exception:
        hop_wrapper = ModelWrapper(None)

    return yeast_df, malt_df, hop_wrapper


# ---------------------------------------------------------------------
# Sidebar inputs
# ---------------------------------------------------------------------

def sidebar_inputs(yeast_df: pd.DataFrame, malt_df: pd.DataFrame):
    """
    Build all user inputs (hops, malts, yeast, predict button)
    and return them to main().
    """

    st.sidebar.header("Model Inputs")

    # -----------------------
    # Hop inputs
    # -----------------------
    st.sidebar.subheader("Hop Bill (g)")

    # Build hop choices (for now: from model feature names 'hop_xxx')
    # or fallback to an empty list => user sees just "-"
    hop_choices = ["-"]
    # we don't know your full hop list; we try from hop model's features
    _, _, hop_wrapper = load_datasets_and_models()
    if hop_wrapper.feature_names:
        # extract each feature name WITHOUT the 'hop_' prefix
        hop_varieties = []
        for feat in hop_wrapper.feature_names:
            if feat.startswith("hop_"):
                hop_varieties.append(feat[4:])
        hop_varieties = sorted(set(hop_varieties))
        if hop_varieties:
            hop_choices = ["-"] + hop_varieties

    hop1_name = st.sidebar.selectbox("Hop 1", hop_choices, index=0, key="hop1_name")
    hop1_amt  = st.sidebar.number_input(
        f"{hop1_name} (g)",
        min_value=0.0,
        max_value=500.0,
        value=0.0,
        step=5.0,
        key="hop1_amt",
    )

    hop2_name = st.sidebar.selectbox("Hop 2", hop_choices, index=0, key="hop2_name")
    hop2_amt  = st.sidebar.number_input(
        f"{hop2_name} (g)",
        min_value=0.0,
        max_value=500.0,
        value=0.0,
        step=5.0,
        key="hop2_amt",
    )

    hop3_name = st.sidebar.selectbox("Hop 3", hop_choices, index=0, key="hop3_name")
    hop3_amt  = st.sidebar.number_input(
        f"{hop3_name} (g)",
        min_value=0.0,
        max_value=500.0,
        value=0.0,
        step=5.0,
        key="hop3_amt",
    )

    hop4_name = st.sidebar.selectbox("Hop 4", hop_choices, index=0, key="hop4_name")
    hop4_amt  = st.sidebar.number_input(
        f"{hop4_name} (g)",
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

    # -----------------------
    # Malt inputs
    # -----------------------
    st.sidebar.subheader("Malt Bill (%)")

    # Build malt choices from malt_df
    # We'll look for a column that looks like malt name. We'll guess first column.
    if malt_df is not None and not malt_df.empty:
        # guess first col as "malt name"
        malt_name_col = malt_df.columns[0]
        unique_malts = (
            malt_df[malt_name_col].dropna().astype(str).unique().tolist()
        )
        unique_malts = sorted(unique_malts)
        malt_choices = ["-"] + unique_malts
    else:
        malt_choices = ["-"]

    malt1_name = st.sidebar.selectbox("Malt 1 name", malt_choices, index=0, key="malt1_name")
    malt1_pct  = st.sidebar.number_input(
        f"{malt1_name} %",
        min_value=0.0,
        max_value=100.0,
        value=0.0,
        step=1.0,
        key="malt1_pct",
    )

    malt2_name = st.sidebar.selectbox("Malt 2 name", malt_choices, index=0, key="malt2_name")
    malt2_pct  = st.sidebar.number_input(
        f"{malt2_name} %",
        min_value=0.0,
        max_value=100.0,
        value=0.0,
        step=1.0,
        key="malt2_pct",
    )

    malt3_name = st.sidebar.selectbox("Malt 3 name", malt_choices, index=0, key="malt3_name")
    malt3_pct  = st.sidebar.number_input(
        f"{malt3_name} %",
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

    # -----------------------
    # Yeast input
    # -----------------------
    st.sidebar.subheader("Yeast Strain")

    yeast_choices = ["-"]
    if yeast_df is not None and not yeast_df.empty:
        # guess first column is "Name"
        yeast_name_col = yeast_df.columns[0]
        yeast_list = yeast_df[yeast_name_col].dropna().astype(str).unique().tolist()
        yeast_list = sorted(yeast_list)
        if yeast_list:
            yeast_choices = ["-"] + yeast_list

    yeast_choice = st.sidebar.selectbox(
        "Select yeast",
        yeast_choices,
        index=0,
        key="yeast_choice",
    )

    # -----------------------
    # Predict button
    # -----------------------
    run_button = st.sidebar.button("Predict Flavor 🧪", key="run_button")

    return hop_entries, malt_entries, yeast_choice, run_button


# ---------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------

def main():
    yeast_df, malt_df, hop_wrapper = load_datasets_and_models()

    # Title / header
    st.title("🍺 Beer Recipe Digital Twin")
    st.write(
        "Predict hop aroma, malt character, and fermentation profile "
        "using trained ML models (work in progress)."
    )

    # Sidebar UI
    hop_entries, malt_entries, yeast_choice, run_button = sidebar_inputs(
        yeast_df, malt_df
    )

    # --- When user clicks Predict, generate hop aroma scores ---
    if run_button:
        aroma_scores = predict_hop_aroma(hop_wrapper, hop_entries)
    else:
        # no click yet → default zeros (so spider still shows)
        aroma_scores = safe_zero_aroma()

    # Layout: radar chart left, textual right
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("Hop Aroma Radar")

        # Make radar figure safely
        fig = plt.figure(figsize=(5,5), dpi=150)
        ax = plt.subplot(111, polar=True)
        make_radar(ax, aroma_scores)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col_right:
        st.subheader("Top hop notes:")
        # naive summary: show a couple highest aroma dimensions
        # sort aroma_scores by value desc
        items_sorted = sorted(aroma_scores.items(), key=lambda x: x[1], reverse=True)
        for note, val in items_sorted[:2]:
            st.write(f"• {note} ({val:.2f})")

        st.markdown("---")
        st.subheader("Malt character:")
        # toy malt descriptor:
        st.write("bready")

        st.markdown("---")
        st.subheader("Yeast character:")
        # naive placeholder: fruitiness / neutral
        st.write("clean / neutral")

        st.markdown("---")
        st.subheader("Style direction:")
        st.write("🍻 Experimental / Hybrid")

        st.markdown("---")
        st.subheader("Hops used by the model:")
        # list hop entries that are not "-"
        hop_used = [h["name"] for h in hop_entries if h["name"] and h["name"] != "-"]
        if hop_used:
            st.write(", ".join(hop_used))
        else:
            st.write("—")

    # Debug / dev info (optional)
    with st.expander("🔬 Debug info"):
        st.write("User hop entries:", hop_entries)
        st.write("User malt entries:", malt_entries)
        st.write("Selected yeast:", yeast_choice)
        st.write("Aroma scores dict:", aroma_scores)
        st.write("Wrapper feature_names:", hop_wrapper.feature_names)
        st.write("Yeast dataset columns:", list(yeast_df.columns) if not yeast_df.empty else "[]")
        st.dataframe(yeast_df.head() if not yeast_df.empty else pd.DataFrame())


if __name__ == "__main__":
    main()

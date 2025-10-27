import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any, Union

# --------------------------
# PAGE CONFIG
# --------------------------
st.set_page_config(
    page_title="Beer Recipe Digital Twin",
    page_icon="🍺",
    layout="wide",
)

# --------------------------
# CONSTANTS / DISPLAY LABELS
# --------------------------
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

DEFAULT_AROMA_SCORES = [0.0] * len(AROMA_AXES)

# --------------------------
# DATA LOADING HELPERS
# --------------------------
@st.cache_resource
def load_pickle_df(pkl_path: str) -> pd.DataFrame:
    p = Path(pkl_path)
    if not p.exists():
        st.warning(f"Missing data file: {pkl_path}")
        return pd.DataFrame()
    return pd.read_pickle(p)

@st.cache_resource
def load_joblib_model(jl_path: str):
    p = Path(jl_path)
    if not p.exists():
        st.warning(f"Missing model file: {jl_path}")
        return None
    try:
        return joblib.load(p)
    except Exception as e:
        st.error(f"Could not load model from {jl_path}: {e}")
        return None

# --------------------------
# MODEL WRAPPER
# --------------------------
class HopAromaModelWrapper:
    """
    Holds:
      - model: the sklearn Pipeline we loaded
      - feature_names: list of input feature names in the order model expects
    """

    def __init__(self, model_obj: Any):
        self.model = model_obj
        self.feature_names: List[str] = []

        # Try to recover real feature names from this pipeline
        self.feature_names = self._extract_feature_names(model_obj)

    def _extract_feature_names(self, model_obj: Any) -> List[str]:
        """
        We attempt multiple strategies to infer the pipeline's expected input
        column names, in the correct order. We'll try common scikit-learn attrs.
        """
        feats: List[str] = []

        if model_obj is None:
            return feats

        # Strategy 1: direct attribute on pipeline (sklearn >=1.0 sometimes adds this)
        # e.g. pipeline.feature_names_in_
        if hasattr(model_obj, "feature_names_in_"):
            attr = getattr(model_obj, "feature_names_in_")
            if isinstance(attr, (list, np.ndarray, pd.Index)):
                feats = list(attr)
                if feats:
                    return feats

        # Strategy 2: first step in pipeline might expose feature_names_in_
        if hasattr(model_obj, "named_steps"):
            for step_name, step_obj in model_obj.named_steps.items():
                if hasattr(step_obj, "feature_names_in_"):
                    attr = getattr(step_obj, "feature_names_in_")
                    if isinstance(attr, (list, np.ndarray, pd.Index)):
                        feats = list(attr)
                        if feats:
                            return feats

        # Strategy 3: ColumnTransformer w/ get_feature_names_out()
        # e.g. pipeline.named_steps["preprocess"].get_feature_names_out()
        if hasattr(model_obj, "named_steps"):
            for step_name, step_obj in model_obj.named_steps.items():
                if hasattr(step_obj, "get_feature_names_out"):
                    try:
                        out = step_obj.get_feature_names_out()
                        if isinstance(out, (list, np.ndarray, pd.Index)):
                            feats = list(out)
                            if feats:
                                return feats
                    except Exception:
                        pass

        # Strategy 4: If there's a "preprocessor" or "vectorizer" style step
        common_keys = ["preprocessor", "prep", "process", "vect", "vectorizer"]
        if hasattr(model_obj, "named_steps"):
            for key in common_keys:
                if key in model_obj.named_steps:
                    step_obj = model_obj.named_steps[key]
                    # try again inside that step
                    if hasattr(step_obj, "feature_names_in_"):
                        attr = getattr(step_obj, "feature_names_in_", [])
                        if isinstance(attr, (list, np.ndarray, pd.Index)):
                            feats = list(attr)
                            if feats:
                                return feats
                    if hasattr(step_obj, "get_feature_names_out"):
                        try:
                            out = step_obj.get_feature_names_out()
                            if isinstance(out, (list, np.ndarray, pd.Index)):
                                feats = list(out)
                                if feats:
                                    return feats
                        except Exception:
                            pass

        # If we still didn't find anything, we'll leave feats empty.
        # We'll handle this downstream, but it means we can't match columns yet.
        return feats


def build_aligned_feature_df(
    user_hop_grams: Dict[str, float],
    required_feature_names: List[str],
) -> pd.DataFrame:
    """
    We take the user hop bill (like {'Simcoe': 30, 'Amarillo': 20})
    and assemble a single-row DataFrame with columns in `required_feature_names`.

    We assume each feature is named something like "hop_Simcoe" in training.
    If your model was trained on different naming, adjust here.
    """

    if not required_feature_names:
        # we can't align columns, so just make an empty frame
        return pd.DataFrame()

    row = {}
    for feat in required_feature_names:
        # Heuristic:
        # If columns looked like "hop_Simcoe", we parse after 'hop_'.
        # Otherwise just try to match raw hop name as-is.
        if feat.startswith("hop_"):
            hop_name = feat.replace("hop_", "")
        else:
            hop_name = feat

        grams = user_hop_grams.get(hop_name, 0.0)
        row[feat] = float(grams)

    df_out = pd.DataFrame([row], columns=required_feature_names)
    return df_out


def predict_aroma_scores(
    wrapper: HopAromaModelWrapper,
    aligned_df: pd.DataFrame,
) -> Tuple[List[float], Optional[str]]:
    """
    Actually run wrapper.model.predict(aligned_df).
    Return (scores, error_msg). We'll map the model output into the
    8 aroma axes in this demo.

    NOTE:
    - If your model is truly multi-output with the 8 aroma axes,
      we want y_pred.shape == (1, 8).
    - If it's something else (single score, classification, etc.),
      adapt the mapping logic below.
    """

    if wrapper.model is None:
        return DEFAULT_AROMA_SCORES, "No model loaded."

    try:
        if aligned_df.empty:
            return DEFAULT_AROMA_SCORES, (
                "Aligned DF is empty (feature names unknown or mismatch)."
            )

        y_pred = wrapper.model.predict(aligned_df)

        # y_pred could be shape (1, 8) or (1,) etc.
        y_pred = np.array(y_pred)

        if y_pred.ndim == 1:
            # ex: (8,) or (1,)
            if y_pred.shape[0] == len(AROMA_AXES):
                scores = y_pred.tolist()
            else:
                # single value or mismatch
                scores = [float(y_pred[0])] * len(AROMA_AXES)
        elif y_pred.ndim == 2:
            # ex: (1,8)
            if y_pred.shape[1] == len(AROMA_AXES):
                scores = y_pred[0, :].tolist()
            else:
                # unexpected width: duplicate or pad
                base_val = float(y_pred[0, 0])
                scores = [base_val] * len(AROMA_AXES)
        else:
            scores = DEFAULT_AROMA_SCORES

        # clip to >=0 for nicer display
        scores = [max(0.0, float(x)) for x in scores]
        return scores, None

    except Exception as e:
        return DEFAULT_AROMA_SCORES, f"Predict exception: {e}"

# --------------------------
# PLOTTING: RADAR / SPIDER
# --------------------------
def plot_aroma_radar(scores: List[float]) -> plt.Figure:
    """
    Radar/spider chart of the hop aroma axes.
    """
    n_axes = len(AROMA_AXES)
    if len(scores) != n_axes:
        scores = DEFAULT_AROMA_SCORES

    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
    angles += angles[:1]
    vals = scores + scores[:1]

    fig, ax = plt.subplots(
        figsize=(6, 6),
        subplot_kw=dict(polar=True)
    )
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # draw one axis per variable + add labels
    ax.set_thetagrids(
        np.degrees(angles[:-1]),
        AROMA_AXES,
    )

    # radial grid
    ax.set_rlabel_position(0)
    ax.set_ylim(0, max(1.0, max(vals) if vals else 1.0))
    ax.yaxis.grid(True, color="gray", linestyle="dashed", linewidth=0.8)
    ax.xaxis.grid(True, color="gray", linestyle="dashed", linewidth=0.8)

    line = ax.plot(
        angles,
        vals,
        color="#2F3A57",
        linewidth=2,
    )[0]
    ax.fill(
        angles,
        vals,
        color="#2F3A57",
        alpha=0.25,
    )

    # annotate center score = just show avg
    if scores:
        avg_score = np.mean(scores)
    else:
        avg_score = 0.0
    ax.text(
        0.0,
        0.0,
        f"{avg_score:.2f}",
        ha="center",
        va="center",
        fontsize=14,
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="#E8ECF5",
            edgecolor="#2F3A57",
            linewidth=1,
        ),
    )

    ax.set_title("Hop Aroma Radar", fontsize=18, pad=20)
    return fig

# --------------------------
# SIDEBAR INPUTS
# --------------------------
def sidebar_inputs(
    hops_master_list: List[str],
    malts_master_list: List[str],
    yeast_master_list: List[str],
) -> Tuple[Dict[str, float], Dict[str, float], str, bool]:
    """
    Left sidebar:
    - Up to 4 hops + grams
    - Up to 3 malts + %
    - Yeast strain
    - Predict button
    Returns:
      hop_bill_dict: {hop_name: grams, ...}
      malt_bill_dict: {malt_name: pct, ...}
      yeast_choice: string
      run_button: bool
    """
    st.sidebar.header("Model Inputs", anchor=False)

    # ---- Hop Bill
    st.sidebar.subheader("Hop Bill (g)")
    hop_bill_dict: Dict[str, float] = {}
    for i in range(1, 5):
        col1, col2 = st.sidebar.columns([2, 1])
        hop_sel = col1.selectbox(
            f"Hop {i}",
            ["-"] + hops_master_list,
            index=0,
            key=f"hop_sel_{i}",
        )
        grams = col2.number_input(
            f"{hop_sel or 'Hop'} (g)",
            min_value=0.0,
            max_value=500.0,
            step=5.0,
            value=0.0,
            key=f"hop_g_{i}",
        )
        if hop_sel != "-" and grams > 0:
            hop_bill_dict[hop_sel] = hop_bill_dict.get(hop_sel, 0) + grams

    # ---- Malt Bill
    st.sidebar.subheader("Malt Bill (%)")
    malt_bill_dict: Dict[str, float] = {}
    for j in range(1, 4):
        c1, c2 = st.sidebar.columns([2, 1])
        malt_sel = c1.selectbox(
            f"Malt {j}",
            ["-"] + malts_master_list,
            index=0,
            key=f"malt_sel_{j}",
        )
        pct_val = c2.number_input(
            f"{malt_sel or 'Malt'} %",
            min_value=0.0,
            max_value=100.0,
            step=1.0,
            value=0.0,
            key=f"malt_pct_{j}",
        )
        if malt_sel != "-" and pct_val > 0:
            malt_bill_dict[malt_sel] = malt_bill_dict.get(malt_sel, 0) + pct_val

    # ---- Yeast
    st.sidebar.subheader("Yeast Strain")
    yeast_choice = st.sidebar.selectbox(
        "Select yeast",
        ["-"] + yeast_master_list,
        index=0,
        key="yeast_sel_main",
    )

    run_button = st.sidebar.button("Predict Flavor 🍾")

    return hop_bill_dict, malt_bill_dict, yeast_choice, run_button

# --------------------------
# MAIN APP
# --------------------------
def main():
    # --- Load reference data
    clean_malt_df = load_pickle_df("clean_malt_df.pkl")
    clean_yeast_df = load_pickle_df("clean_yeast_df.pkl")

    # Fallback seeds for lists
    hops_master_list = [
        "Simcoe", "Amarillo", "Citra", "Mosaic", "Astra", "Adeena", "Ella"
    ]
    malts_master_list = list(clean_malt_df.get("malt_name", pd.Series(["BEST ALE MALT", "BLACK MALT"])).unique())
    yeast_master_list = list(clean_yeast_df.get("Name", pd.Series(["Cooper Ale", "Manchester"])).unique())

    # --- Load hop aroma model (wrapper)
    raw_model = load_joblib_model("hop_aroma_model.joblib")
    wrapper = HopAromaModelWrapper(raw_model)

    # --- Gather user recipe inputs
    hop_bill_dict, malt_bill_dict, yeast_choice, run_button = sidebar_inputs(
        hops_master_list,
        malts_master_list,
        yeast_master_list,
    )

    # MAIN LAYOUT
    col_plot, col_info = st.columns([2, 1])

    with col_plot:
        st.title("🍺 Beer Recipe Digital Twin")
        st.write(
            "Predict hop aroma, malt character, and fermentation profile "
            "using trained ML models (work in progress)."
        )

    # if user clicks the button => attempt predict
    if run_button:
        # 1) Align user hop bill to model feature vector
        aligned_df = build_aligned_feature_df(
            user_hop_grams=hop_bill_dict,
            required_feature_names=wrapper.feature_names,
        )

        # 2) Predict via pipeline
        aroma_scores, err_msg = predict_aroma_scores(wrapper, aligned_df)

        # 3) Plot radar
        fig = plot_aroma_radar(aroma_scores)
        with col_plot:
            st.pyplot(fig, use_container_width=True)

        # 4) Right column info cards
        with col_info:
            st.subheader("Top hop notes:")
            # For demo: pick 2 highest from aroma_scores
            if aroma_scores and max(aroma_scores) > 0:
                arr = np.array(aroma_scores)
                top_idx = arr.argsort()[::-1][:2]
                for idx in top_idx:
                    st.write(f"- {AROMA_AXES[idx]} ({aroma_scores[idx]:.2f})")
            else:
                st.write("- tropical (0.00)")
                st.write("- citrus (0.00)")

            st.markdown("---")
            st.subheader("Malt character:")
            st.write("bready")  # placeholder: you can map malt bill -> descriptor

            st.markdown("---")
            st.subheader("Yeast character:")
            if yeast_choice and yeast_choice != "-":
                # Show some yeast trait heuristics from clean_yeast_df if possible
                row_match = clean_yeast_df[clean_yeast_df["Name"] == yeast_choice]
                if not row_match.empty:
                    # try "clean_neutral", "fruity_esters", etc.
                    traits = []
                    if "clean_neutral" in row_match.columns:
                        if row_match["clean_neutral"].iloc[0] == 1:
                            traits.append("clean / neutral")
                    if "fruity_esters" in row_match.columns:
                        if row_match["fruity_esters"].iloc[0] == 1:
                            traits.append("fruity esters")
                    st.write(", ".join(traits) if traits else "house character")
                else:
                    st.write("house character")
            else:
                st.write("clean / neutral")

            st.markdown("---")
            st.subheader("Style direction:")
            st.write("🍻 Experimental / Hybrid")

        # --- DEBUG ZONE
        with st.expander("🧪 Debug: hop model input / prediction", expanded=False):
            st.write("wrapper.model is None?", wrapper.model is None)
            st.write("wrapper.feature_names:", wrapper.feature_names[:10] if wrapper.feature_names else "(None)")
            if err_msg:
                st.error(f"Model predict() error: {err_msg}")
            st.write("aligned_df passed to model:")
            st.dataframe(aligned_df)

            st.write("User aggregate hop grams by hop variety:")
            if hop_bill_dict:
                st.dataframe(pd.DataFrame([hop_bill_dict]))
            else:
                st.write("(no hop bill)")

            st.write("type(wrapper.model):")
            st.write(type(wrapper.model))
            st.write(dir(wrapper.model)[:20])

        # more debug info for yeast
        with st.expander("🧬 Debug: yeast dataset", expanded=False):
            st.write("Columns:")
            st.write(list(clean_yeast_df.columns))
            # show first ~10
            st.dataframe(clean_yeast_df.head(10))

    else:
        # no prediction yet => just show an empty radar + placeholders
        fig = plot_aroma_radar(DEFAULT_AROMA_SCORES)
        with col_plot:
            st.pyplot(fig, use_container_width=True)

        with col_info:
            st.subheader("Top hop notes:")
            st.write("- tropical (0.00)")
            st.write("- citrus (0.00)")
            st.markdown("---")

            st.subheader("Malt character:")
            st.write("bready")
            st.markdown("---")

            st.subheader("Yeast character:")
            st.write("clean / neutral")
            st.markdown("---")

            st.subheader("Style direction:")
            st.write("🍻 Experimental / Hybrid")

        # Also surface model feature names right away so you can inspect before hitting Predict
        with st.expander("🧪 Model feature names detected"):
            st.write("These are the columns we think the hop model expects.")
            st.write(
                "(We'll line up your hop grams to these col names "
                "before calling .predict())"
            )
            if wrapper.feature_names:
                st.dataframe(pd.DataFrame({"feature_name": wrapper.feature_names}))
            else:
                st.warning(
                    "No feature_names detected. The model may not expose "
                    "feature_names_in_ / get_feature_names_out(), or this "
                    "isn't the final trained pipeline."
                )

        with st.expander("🧬 Yeast dataset preview"):
            st.dataframe(clean_yeast_df.head(10))


# run
if __name__ == "__main__":
    main()

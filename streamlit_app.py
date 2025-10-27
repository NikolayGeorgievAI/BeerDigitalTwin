import math
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# -------------------------
# STREAMLIT PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="Beer Recipe Digital Twin",
    layout="wide",
)

# Matplotlib style tweaks for radar
plt.rcParams.update({
    "font.size": 12,
    "axes.edgecolor": "#333333",
    "text.color": "#333333",
    "axes.labelcolor": "#333333",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
})


# -------------------------
# UTIL: LOAD DATA AND MODEL
# -------------------------

@st.cache_resource(show_spinner=False)
def load_clean_dfs():
    """
    Load the cleaned malt and yeast dataframes from pickle.
    """
    malt_df = joblib.load("clean_malt_df.pkl")
    yeast_df = joblib.load("clean_yeast_df.pkl")

    # Make sure column names are consistent / nice
    # (your pickle already seemed good, just return as-is)
    return malt_df, yeast_df


class HopModelWrapper:
    """
    Wrap the hop aroma model so we always have:
      - .model : something with .predict(...)
      - .feature_names : list[str] of features columns the model expects

    We'll aggressively try to unwrap whatever was saved in hop_aroma_model.joblib.
    """

    def __init__(self, raw_obj):
        self.raw_obj = raw_obj
        self.model = None
        self.feature_names = None
        self._unwrap()

    def _unwrap(self):
        """
        Heuristic:
         1. If raw_obj has .predict already -> that's our model.
         2. If raw_obj is dict:
            - look for keys that look like model objects ("model", "pipeline", etc.)
            - look for keys that look like feature names ("feature_names", "features", etc.)
        """
        candidate = None
        features = None

        if hasattr(self.raw_obj, "predict"):
            candidate = self.raw_obj

        elif isinstance(self.raw_obj, dict):
            # try common keys for model steps
            model_keys = ["model", "pipeline", "estimator", "clf", "regressor"]
            feat_keys = ["feature_names", "features", "feature_list", "cols"]

            # find model
            for mk in model_keys:
                if mk in self.raw_obj and hasattr(self.raw_obj[mk], "predict"):
                    candidate = self.raw_obj[mk]
                    break

            # find features
            for fk in feat_keys:
                if fk in self.raw_obj:
                    # must be list-like of strings
                    maybe_feats = self.raw_obj[fk]
                    if isinstance(maybe_feats, (list, tuple)):
                        features = list(maybe_feats)
                    # else ignore

        # last fallback: maybe there's only one value in dict that's a predictor
        if candidate is None and isinstance(self.raw_obj, dict):
            for v in self.raw_obj.values():
                if hasattr(v, "predict"):
                    candidate = v
                    break

        # store
        self.model = candidate
        self.feature_names = features

    def is_ready(self):
        """
        Do we have a real model we can call .predict() on?
        """
        return (self.model is not None) and hasattr(self.model, "predict")

    def get_feature_names(self):
        """
        best guess at feature names:
        1. if self.feature_names (unwrapped from file)
        2. if model has .feature_names_in_ or .n_features_in_ and it's array-like
        """
        if self.feature_names is not None:
            return list(self.feature_names)

        # sometimes sklearn pipelines expose feature_names_in_
        if self.is_ready():
            if hasattr(self.model, "feature_names_in_"):
                return list(self.model.feature_names_in_)
            if hasattr(self.model, "named_steps"):
                # try first step for feature names
                for step_name, step_est in self.model.named_steps.items():
                    if hasattr(step_est, "feature_names_in_"):
                        return list(step_est.feature_names_in_)
        return None


@st.cache_resource(show_spinner=False)
def load_hop_model():
    """
    Load hop_aroma_model.joblib and wrap it.
    Returns HopModelWrapper.
    """
    raw_obj = joblib.load("hop_aroma_model.joblib")
    wrapper = HopModelWrapper(raw_obj)
    return wrapper


# -------------------------
# FEATURE ENGINEERING
# -------------------------

def build_user_hop_vector(hop_inputs, feature_cols):
    """
    hop_inputs: dict { "Amarillo": grams, "Simcoe": grams, ... }
    feature_cols: e.g. ["hop_Amarillo","hop_Simcoe", ...]

    Returns aligned_df: 1-row DataFrame with exactly feature_cols as columns.
    Any missing hops default to 0.
    """
    if feature_cols is None or len(feature_cols) == 0:
        # no knowledge of which columns to build
        return pd.DataFrame()

    row_data = {}
    for col in feature_cols:
        # We assume training columns looked like "hop_<variety>"
        # strip prefix so we can match to user hop_inputs
        # Example:
        #   feature "hop_Amarillo" -> base "Amarillo"
        base = col
        if base.startswith("hop_"):
            base = base[4:]

        grams = hop_inputs.get(base, 0.0)
        row_data[col] = float(grams)

    aligned_df = pd.DataFrame([row_data], columns=feature_cols)
    return aligned_df


def predict_hop_aroma(wrapper, aligned_df):
    """
    Predict aroma with wrapper.model if possible.
    Return tuple: (scores_dict, err_msg)

    scores_dict is { "fruity": val, "citrus": val, "tropical": val, ... }
    We'll assume the model outputs an array of intensities for 8 axes:
    fruity, citrus, tropical, earthy, spicy, herbal, floral, resinous
    in that exact order. If your model differs we can update mapping here.
    """
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

    if not wrapper.is_ready():
        return {a: 0.0 for a in aroma_axes}, "Model not ready (no .predict)."

    if aligned_df.empty:
        return {a: 0.0 for a in aroma_axes}, "Aligned DF is empty (feature mismatch)."

    try:
        y_pred = wrapper.model.predict(aligned_df)
        # y_pred shape should be (1, n_axes)
        if hasattr(y_pred, "tolist"):
            y_pred = y_pred.tolist()
        if isinstance(y_pred, list) and len(y_pred) > 0:
            first_row = y_pred[0]
        else:
            first_row = [0.0]*len(aroma_axes)

        # map them
        out = {}
        for i, axis in enumerate(aroma_axes):
            if i < len(first_row):
                out[axis] = float(first_row[i])
            else:
                out[axis] = 0.0
        return out, None

    except Exception as e:
        return {a: 0.0 for a in aroma_axes}, f"Predict exception: {e}"


# -------------------------
# PLOTTING: RADAR
# -------------------------

def make_radar(ax, values_dict):
    """
    Draw spider/radar plot of aroma axes.
    values_dict: {"fruity":0.1,"citrus":0.2,...} length = 8
    """

    labels = list(values_dict.keys())
    vals = list(values_dict.values())

    # close the loop
    labels += [labels[0]]
    vals   += [vals[0]]

    # angles
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    # ensure we close
    angles += [angles[0]]

    # Draw
    ax.set_theta_offset(np.pi / 2.0)
    ax.set_theta_direction(-1)

    ax.set_rlabel_position(180/len(labels))  # put radial labels nicely

    # grid / style
    ax.set_facecolor("white")
    ax.spines["polar"].set_color("#333333")
    ax.grid(color="#999999", linestyle="--", linewidth=1, alpha=0.6)

    # Plot outline
    ax.plot(angles, vals, color="#1f2a44", linewidth=2)
    ax.fill(angles, vals, color="#1f2a44", alpha=0.2)

    # Ticks
    ax.set_yticks([0.2,0.4,0.6,0.8,1.0])
    ax.set_yticklabels(["0.2","0.4","0.6","0.8","1.0"], color="#333333")
    ax.set_ylim(0,1.0)

    # Axes labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels[:-1], color="#333333", fontsize=12)

    # center label
    center_val = np.mean(vals[:-1]) if len(vals) > 1 else 0.0
    ax.text(
        0.5, 0.5,
        f"{center_val:0.2f}",
        transform=ax.transAxes,
        ha="center", va="center",
        fontsize=20,
        bbox=dict(boxstyle="round,pad=0.4", fc="#E8ECF7", ec="#1f2a44", lw=2)
    )


# -------------------------
# SIDEBAR: USER INPUTS
# -------------------------

def sidebar_inputs(malt_df, yeast_df):
    """
    Render all user controls in the sidebar,
    return:
      hop_inputs (dict hop->grams),
      malt_bill (list of dicts),
      yeast_choice (str)
    """

    st.sidebar.header("Hop Bill (g)")
    # We'll allow up to 4 hop additions:
    hop_inputs = {}
    for i in range(1,5):
        st.sidebar.write(f"Hop {i}")
        hop_name = st.sidebar.selectbox(
            f"Hop {i} name",
            options=["-"] + sorted(list(set(["Amarillo","Simcoe","Astra","Citra","Mosaic","Galaxy","Cascade","Centennial"]))),
            key=f"hop_name_{i}"
        )
        hop_grams = st.sidebar.number_input(
            f"{hop_name} (g)",
            min_value=0.0,
            max_value=500.0,
            value=0.0,
            step=5.0,
            key=f"hop_grams_{i}"
        )
        if hop_name != "-" and hop_grams > 0:
            hop_inputs[hop_name] = hop_inputs.get(hop_name, 0.0) + hop_grams

    st.sidebar.markdown("---")

    st.sidebar.header("Malt Bill (%)")
    # up to 3 malts from malt_df['malt_name'] if present,
    # or fallback to unique names in that pickle.
    malt_options = ["-"]
    malt_name_col = None
    for cand in ["Malt","malt","malt_name","MaltName","name","Name"]:
        if cand in malt_df.columns:
            malt_name_col = cand
            break
    if malt_name_col:
        malt_options += sorted(malt_df[malt_name_col].astype(str).unique().tolist())
    else:
        malt_options += ["BEST ALE MALT","BLACK MALT"]

    malt_bill = []
    for j in range(1,4):
        st.sidebar.write(f"Malt {j}")
        m_name = st.sidebar.selectbox(
            f"Malt {j} name",
            malt_options,
            key=f"malt_name_{j}"
        )
        m_pct = st.sidebar.number_input(
            f"{m_name} %",
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            step=1.0,
            key=f"malt_pct_{j}"
        )
        if m_name != "-" and m_pct > 0:
            malt_bill.append({"name": m_name, "pct": m_pct})

    st.sidebar.markdown("---")

    st.sidebar.header("Yeast Strain")
    # Yeast dropdown from yeast_df["Name"] if exists
    yeast_name_col = "Name" if "Name" in yeast_df.columns else yeast_df.columns[0]
    yeast_options = ["-"] + yeast_df[yeast_name_col].astype(str).unique().tolist()
    yeast_choice = st.sidebar.selectbox(
        "Select yeast",
        yeast_options,
        key="yeast_choice"
    )

    st.sidebar.markdown("---")
    run_button = st.sidebar.button("Predict Flavor 🧪")

    return hop_inputs, malt_bill, yeast_choice, run_button


# -------------------------
# MAIN PAGE LAYOUT / LOGIC
# -------------------------

def main():
    # Title/header
    col_logo, col_title = st.columns([1,8])
    with col_logo:
        st.markdown("🍺")
    with col_title:
        st.title("Beer Recipe Digital Twin")
        st.caption(
            "Predict hop aroma, malt character, and fermentation profile "
            "using trained ML models (work in progress)."
        )

    # Load base data + model
    malt_df, yeast_df = load_clean_dfs()
    hop_wrapper = load_hop_model()

    # Sidebar inputs
    hop_inputs, malt_bill, yeast_choice, run_button = sidebar_inputs(malt_df, yeast_df)

    # Build hop feature row for prediction
    hop_feature_cols = hop_wrapper.get_feature_names()
    aligned_df = build_user_hop_vector(hop_inputs, hop_feature_cols)
    aroma_scores, model_error = predict_hop_aroma(hop_wrapper, aligned_df)

    # -----------------
    # LAYOUT: radar + text panels
    # -----------------
    col_left, col_right = st.columns([2,1], vertical_alignment="top")

    with col_left:
        st.subheader("Hop Aroma Radar")
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111, polar=True)

        make_radar(ax, aroma_scores)
        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)

    with col_right:
        st.subheader("Top hop notes:")
        st.write(f"• tropical ({aroma_scores.get('tropical',0):.2f})")
        st.write(f"• citrus ({aroma_scores.get('citrus',0):.2f})")

        st.markdown("---")
        st.subheader("Malt character:")
        # naive placeholder from first malt in bill, fallback text
        malt_desc = "bready"
        if malt_bill:
            # could refine using malt_df lookups later
            malt_desc = " / ".join([m['name'] for m in malt_bill])
        st.write(malt_desc if malt_desc else "n/a")

        st.markdown("---")
        st.subheader("Yeast character:")
        # approximate from yeast_df row for chosen yeast
        yeast_profile = "clean / neutral"
        if yeast_choice and yeast_choice != "-" and "Name" in yeast_df.columns:
            row = yeast_df[yeast_df["Name"]==yeast_choice]
            if not row.empty:
                # Compose a short string from those phenotype flags
                tags = []
                if "fruity_esters" in row.columns and row["fruity_esters"].iloc[0] == 1:
                    tags.append("fruity esters")
                if "clean_neutral" in row.columns and row["clean_neutral"].iloc[0] == 1:
                    tags.append("clean / neutral")
                if "phenolic_spicy" in row.columns and row["phenolic_spicy"].iloc[0] == 1:
                    tags.append("phenolic / spicy")
                if tags:
                    yeast_profile = ", ".join(tags)
        st.write(yeast_profile)

        st.markdown("---")
        st.subheader("Style direction:")
        # toy / placeholder logic:
        style_hint = "Experimental / Hybrid"
        if yeast_profile.startswith("clean"):
            style_hint = "Clean / Neutral Ale direction"
        st.write(style_hint)

        st.markdown("---")
        st.subheader("Hops used by the model:")
        if hop_inputs:
            hop_list_str = ", ".join([f"{name} ({grams:.0f} g)" for name, grams in hop_inputs.items()])
        else:
            hop_list_str = "n/a"
        st.write(hop_list_str)

    # -----------------
    # DEBUG / DEV PANEL
    # -----------------
    with st.expander("🔬 Debug: model + feature alignment", expanded=False):
        st.write("wrapper.model is None?", hop_wrapper.model is None)
        st.write("wrapper.is_ready()?", hop_wrapper.is_ready())
        st.write("wrapper.feature_names:", hop_wrapper.feature_names)
        st.write("wrapper.get_feature_names():", hop_feature_cols)

        st.write("model_error:", model_error)
        st.write("aligned_df passed to model:")
        st.dataframe(aligned_df)

        st.write("Raw hop_inputs dict:")
        st.write(hop_inputs)

        st.write("type(wrapper.model):", str(type(hop_wrapper.model)))
        st.write("Available attributes on wrapper.model (first 25):")
        if hop_wrapper.model is not None:
            st.write(dir(hop_wrapper.model)[:25])

    with st.expander("🧬 Debug: yeast dataset", expanded=False):
        st.write("Columns:", list(yeast_df.columns))
        st.dataframe(yeast_df.head(25))


if __name__ == "__main__":
    main()

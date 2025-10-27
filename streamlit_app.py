import math
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Page / debug mode
# ------------------------------------------------------------
st.set_page_config(
    page_title="Beer Recipe Digital Twin",
    layout="wide",
)

DEBUG_MODE = True  # flip to False later if you want a clean UI


# ------------------------------------------------------------
# Helpers: safe loaders
# ------------------------------------------------------------
@st.cache_resource
def load_pickle(path: str):
    try:
        return joblib.load(path)
    except Exception as e:
        st.warning(f"[load_pickle] Failed to load {path}: {e}")
        return None


clean_malt_df = load_pickle("clean_malt_df.pkl")
clean_yeast_df = load_pickle("clean_yeast_df.pkl")

_raw_hop_obj = load_pickle("hop_aroma_model.joblib")
malt_model = load_pickle("malt_sensory_model.joblib")
yeast_model = load_pickle("yeast_sensory_model.joblib")


# ------------------------------------------------------------
# Hop model wrapper
# ------------------------------------------------------------
class HopModelWrapper:
    """
    This hides the fact that hop_aroma_model.joblib might be:
    - a dict like {"model": pipeline, "feature_names": [...], ...}, OR
    - a direct sklearn pipeline.

    We expose:
      .model            -> object with .predict()
      .feature_names    -> list of columns the model expects
    """

    def __init__(self, raw_obj):
        self.model = None
        self.feature_names = []

        if raw_obj is None:
            return

        # Case A: already a sklearn pipeline with .predict
        if hasattr(raw_obj, "predict"):
            self.model = raw_obj

            # Try to detect feature names
            for attr in ["feature_names_in_", "features_in_", "feature_names_", "columns_"]:
                if hasattr(raw_obj, attr):
                    possible = getattr(raw_obj, attr)
                    self.feature_names = _ensure_list(possible)
                    break

        # Case B: it's a dict (typical when you joblib.dump({...}))
        elif isinstance(raw_obj, dict):
            # Best guess keys:
            # 'model' should be pipeline, 'feature_names' or 'hop_feature_names' the columns
            if "model" in raw_obj and hasattr(raw_obj["model"], "predict"):
                self.model = raw_obj["model"]

            # try known keys for features
            for key in ["feature_names", "hop_feature_names", "columns", "feature_names_in_"]:
                if key in raw_obj:
                    self.feature_names = _ensure_list(raw_obj[key])
                    break

            # Also check pipeline attributes if not found above
            if self.model is not None and not self.feature_names:
                for attr in ["feature_names_in_", "features_in_", "feature_names_", "columns_"]:
                    if hasattr(self.model, attr):
                        self.feature_names = _ensure_list(getattr(self.model, attr))
                        break


def _ensure_list(x):
    if x is None:
        return []
    if isinstance(x, (list, tuple, np.ndarray, pd.Index)):
        return list(x)
    return [x]


hop_wrapper = HopModelWrapper(_raw_hop_obj)


# ------------------------------------------------------------
# Utility fns
# ------------------------------------------------------------
def unique_sorted_hops_from_wrapper(wrapper: HopModelWrapper):
    """
    Build a nice list of hop names to show in the sidebar dropdown.
    We'll parse wrapper.feature_names, which might look like ["hop_Simcoe", "hop_Amarillo", ...]
    We'll strip "hop_" and dedupe.
    """
    fallback = ["Simcoe", "Amarillo", "Astra", "Citra", "Mosaic"]

    feats = _ensure_list(wrapper.feature_names)
    cleaned = []
    for f in feats:
        s = str(f)
        if s.startswith("hop_"):
            s = s.replace("hop_", "", 1)
        if s and s not in cleaned:
            cleaned.append(s)

    if not cleaned:
        return fallback

    # keep first ~30 just to avoid a monster dropdown
    return cleaned[:30]


def aggregate_user_hops(hop_inputs):
    """
    hop_inputs = list of (hop_name, grams)
    Combine duplicates and drop zeros.
    Returns dict like {"Simcoe": 100, "Amarillo": 50}
    """
    agg = {}
    for name, grams in hop_inputs:
        if not name or name.strip() == "-":
            continue
        if grams <= 0:
            continue
        agg[name] = agg.get(name, 0.0) + grams
    return agg


def build_aligned_df(agg_hops: dict, wrapper: HopModelWrapper):
    """
    Build a single-row DataFrame whose columns match wrapper.feature_names.
    We'll fill 0 if user didn't include that hop.
    If wrapper has no feature names, fallback to just user's hops.
    """

    feats = _ensure_list(wrapper.feature_names)

    if feats:
        # e.g. feats = ["hop_Simcoe","hop_Amarillo",...]
        data = {}
        for col in feats:
            hop_var = col
            if hop_var.startswith("hop_"):
                hop_var = hop_var.replace("hop_", "", 1)
            grams = agg_hops.get(hop_var, 0.0)
            data[col] = [grams]
        return pd.DataFrame(data)
    else:
        # fallback columns from user
        if len(agg_hops) == 0:
            return pd.DataFrame([0.0], index=["0"], columns=["total_hops"])
        return pd.DataFrame([agg_hops])


def safe_hop_predict(wrapper: HopModelWrapper, X_df: pd.DataFrame):
    """
    Try to call wrapper.model.predict(X_df). Return (np_array or None, error_message or None)
    """
    if wrapper.model is None:
        return None, "No usable hop model inside joblib (no .predict)."

    try:
        preds = wrapper.model.predict(X_df)
        return preds, None
    except Exception as e:
        return None, f"Predict exception: {e}"


def summarize_hop_aroma_vector(values):
    """
    values is the numeric spider vector [tropical, citrus, fruity, resinous, floral,
                                       herbal, spicy, earthy]
    We'll pick the top 2.
    """
    labels = [
        "tropical", "citrus", "fruity", "resinous",
        "floral", "herbal", "spicy", "earthy",
    ]

    if not values:
        return [("tropical", 0.0), ("citrus", 0.0)]

    pairs = [(labels[i], float(values[i])) for i in range(min(len(labels), len(values)))]
    pairs_sorted = sorted(pairs, key=lambda x: x[1], reverse=True)
    return pairs_sorted[:2] if pairs_sorted else [("tropical", 0.0), ("citrus", 0.0)]


def descriptive_malt_profile(malt_model_obj, malt_inputs_dict):
    """
    Placeholder malt logic. We'll just guess 'bready' / 'roasty' etc.
    """
    # find top malt %
    top_pct = 0.0
    top_name = "-"
    for key, val in malt_inputs_dict.items():
        if key.endswith("_pct"):
            if val > top_pct:
                prefix = key.split("_pct")[0]  # e.g. 'malt1'
                name_key = prefix + "_name"
                nm = malt_inputs_dict.get(name_key, "-")
                top_pct = val
                top_name = nm

    if "BLACK" in str(top_name).upper():
        return "roasty / burnt"
    if "AMBER" in str(top_name).upper():
        return "bready"
    if "BEST" in str(top_name).upper() or "PALE" in str(top_name).upper():
        return "bready"

    return "bready"


def descriptive_yeast_profile(yeast_model_obj, yeast_choice, clean_yeast_df):
    """
    Pull a short descriptor from columns like fruity_esters / clean_neutral.
    Also guess a 'style direction'.
    """
    if not yeast_choice or yeast_choice == "-":
        return ("clean / neutral", "Experimental / Hybrid")

    if clean_yeast_df is None or "Name" not in clean_yeast_df.columns:
        return ("clean / neutral", "Experimental / Hybrid")

    row = clean_yeast_df[clean_yeast_df["Name"] == yeast_choice]
    if row.empty:
        return ("clean / neutral", "Experimental / Hybrid")

    row = row.iloc[0]
    notes = []

    if "clean_neutral" in row and row["clean_neutral"] == 1:
        notes.append("clean / neutral")
    if "fruity_esters" in row and row["fruity_esters"] == 1:
        notes.append("fruity esters")
    if "phenolic_spicy" in row and row["phenolic_spicy"] == 1:
        notes.append("spicy phenolics")
    if not notes:
        notes.append("clean / neutral")

    style_guess = "Experimental / Hybrid"
    if "Notes" in row and isinstance(row["Notes"], str):
        txt = row["Notes"].lower()
        if "lager" in txt:
            style_guess = "Clean / Neutral Ale direction"
        elif "ester" in txt:
            style_guess = "Fruity / English style"

    return (", ".join(notes), style_guess)


def draw_radar(scores_dict):
    """
    scores_dict = Ordered mapping of labels -> value (floats)
    We'll draw a spider plot with those values.
    """
    labels = list(scores_dict.keys())
    vals = [float(scores_dict[k]) for k in labels]

    # close the loop
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    vals += vals[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(
        subplot_kw=dict(polar=True),
        figsize=(6, 6),
        dpi=150
    )

    ax.plot(angles, vals, color="#2F3A56", linewidth=2)
    ax.fill(angles, vals, color="#2F3A56", alpha=0.2)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)

    vmax = max(1.0, max(vals))
    yticks = np.linspace(0, vmax, 5)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{y:.1f}" for y in yticks])
    ax.set_ylim(0, vmax)

    center_val = np.mean(vals[:-1]) if len(vals) > 1 else 0.0
    ax.text(
        0,
        0,
        f"{center_val:.2f}",
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#2F3A56", lw=2),
        fontsize=14,
        color="#2F3A56",
    )

    return fig


# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------
def sidebar_inputs(hop_variety_list):
    st.sidebar.header("Model Inputs")
    st.sidebar.subheader("Hop Bill (g)")

    hop_inputs = []
    for i in range(3):
        c1, c2 = st.sidebar.columns([2, 1])
        with c1:
            hop_name = st.selectbox(
                f"Hop {i+1}",
                options=["-"] + hop_variety_list,
                index=0,
                key=f"hop_name_{i}",
            )
        with c2:
            hop_amt = st.number_input(
                f"{hop_name} (g)",
                min_value=0.0,
                max_value=500.0,
                value=0.0,
                step=5.0,
                key=f"hop_amt_{i}",
            )
        hop_inputs.append((hop_name, hop_amt))

    st.sidebar.subheader("Malt Bill")
    malt_inputs = {}
    malt_inputs["malt1_name"] = st.sidebar.selectbox(
        "Malt 1",
        options=["BEST ALE MALT", "AMBER MALT", "BLACK MALT", "PALE MALT"],
        index=0,
        key="malt1_name_key",
    )
    malt_inputs["malt1_pct"] = st.sidebar.number_input(
        "Malt 1 %",
        min_value=0.0,
        max_value=100.0,
        value=70.0,
        step=1.0,
        key="malt1_pct_key",
    )

    malt_inputs["malt2_name"] = st.sidebar.selectbox(
        "Malt 2",
        options=["BEST ALE MALT", "AMBER MALT", "BLACK MALT", "PALE MALT"],
        index=1,
        key="malt2_name_key",
    )
    malt_inputs["malt2_pct"] = st.sidebar.number_input(
        "Malt 2 %",
        min_value=0.0,
        max_value=100.0,
        value=20.0,
        step=1.0,
        key="malt2_pct_key",
    )

    malt_inputs["malt3_name"] = st.sidebar.selectbox(
        "Malt 3",
        options=["BEST ALE MALT", "AMBER MALT", "BLACK MALT", "PALE MALT"],
        index=2,
        key="malt3_name_key",
    )
    malt_inputs["malt3_pct"] = st.sidebar.number_input(
        "Malt 3 %",
        min_value=0.0,
        max_value=100.0,
        value=10.0,
        step=1.0,
        key="malt3_pct_key",
    )

    st.sidebar.subheader("Yeast Strain")
    if clean_yeast_df is not None and "Name" in clean_yeast_df.columns:
        yeast_list = ["-"] + sorted(clean_yeast_df["Name"].astype(str).unique().tolist())
    else:
        yeast_list = ["-"]

    yeast_choice = st.sidebar.selectbox(
        "Select yeast",
        options=yeast_list,
        index=0,
        key="yeast_choice_key",
    )

    run_button = st.sidebar.button("Predict Flavor 🧪", key="predict_button_key")

    return hop_inputs, malt_inputs, yeast_choice, run_button


# ------------------------------------------------------------
# Main app
# ------------------------------------------------------------
def main():
    st.title("🍺 Beer Recipe Digital Twin")
    st.write(
        "Predict hop aroma, malt character, and fermentation profile using trained ML models (work in progress)."
    )

    hop_variety_list = unique_sorted_hops_from_wrapper(hop_wrapper)

    hop_inputs, malt_inputs, yeast_choice, run_button = sidebar_inputs(hop_variety_list)

    # aggregate hop bill
    agg_hops = aggregate_user_hops(hop_inputs)

    # build input DF for hop model
    aligned_df = build_aligned_df(agg_hops, hop_wrapper)

    aroma_scores = None
    model_error = None

    if run_button:
        aroma_scores, model_error = safe_hop_predict(hop_wrapper, aligned_df)

    # Build radar spider data
    label_order = [
        "tropical", "citrus", "fruity", "resinous",
        "floral", "herbal", "spicy", "earthy",
    ]

    radar_values = [0.0]*len(label_order)
    if aroma_scores is not None:
        # flatten first row
        arr = np.array(aroma_scores).flatten().tolist()
        for i in range(min(len(arr), len(radar_values))):
            radar_values[i] = float(arr[i])

    radar_dict = {lbl: radar_values[i] for i, lbl in enumerate(label_order)}

    # Top hop notes
    top2_notes = summarize_hop_aroma_vector(radar_values)

    # Malt
    malt_desc = descriptive_malt_profile(malt_model, malt_inputs)

    # Yeast
    yeast_desc, style_dir = descriptive_yeast_profile(
        yeast_model, yeast_choice, clean_yeast_df
    )

    # Layout display
    col_plot, col_text = st.columns([2, 1])

    with col_plot:
        st.subheader("Hop Aroma Radar")
        fig = draw_radar(radar_dict)
        st.pyplot(fig, use_container_width=True)

    with col_text:
        st.markdown("### Top hop notes:")
        for (note, val) in top2_notes:
            st.write(f"• {note} ({val:.2f})")

        st.markdown("---")
        st.markdown("### Malt character:")
        st.write(malt_desc)

        st.markdown("---")
        st.markdown("### Yeast character:")
        st.write(yeast_desc)

        st.markdown("---")
        st.markdown("### Style direction:")
        st.write(style_dir)

        st.markdown("---")
        st.markdown("### Hops used by the model:")
        if len(agg_hops) == 0:
            st.write("(no hops added)")
        else:
            hop_list_lines = [f"{hop}: {grams:.1f} g" for hop, grams in agg_hops.items()]
            st.write(", ".join(hop_list_lines))

    # Debug panel
    if DEBUG_MODE:
        st.markdown("---")
        st.subheader("🧪 Debug: hop model input / prediction")

        st.write("wrapper.model is None? ", hop_wrapper.model is None)
        st.write("wrapper.feature_names:")
        st.write(hop_wrapper.feature_names[:200] if hop_wrapper.feature_names else "(None)")

        if model_error:
            st.error(f"Model predict() error: {model_error}")

        st.write("aligned_df passed to model:")
        st.dataframe(aligned_df, use_container_width=True)

        st.write("User aggregate hop grams by hop variety:")
        st.dataframe(pd.DataFrame([agg_hops]), use_container_width=True)

        st.write("type(wrapper.model):")
        st.write(type(hop_wrapper.model))
        if hop_wrapper.model is not None:
            st.write(dir(hop_wrapper.model)[:20])

        st.markdown("---")
        st.subheader("🧬 Debug: yeast dataset")
        if clean_yeast_df is not None:
            st.write("Columns:")
            st.write(list(clean_yeast_df.columns))
            st.dataframe(clean_yeast_df.head(10), use_container_width=True)
        else:
            st.write("clean_yeast_df is None")


if __name__ == "__main__":
    main()

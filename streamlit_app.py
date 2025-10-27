import math
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
st.set_page_config(
    page_title="Beer Recipe Digital Twin",
    layout="wide",
)

DEBUG_MODE = True  # set False to hide debug sections


# ------------------------------------------------------------
# Safe data/model loaders
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

# hop_aroma_model can sometimes be a dict in your repo,
# or a sklearn Pipeline, etc. we'll load "as-is"
hop_aroma_model = load_pickle("hop_aroma_model.joblib")
malt_model = load_pickle("malt_sensory_model.joblib")
yeast_model = load_pickle("yeast_sensory_model.joblib")


# ------------------------------------------------------------
# Small helpers
# ------------------------------------------------------------
def ensure_listlike(x):
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    try:
        return list(x)
    except Exception:
        return [x]


def unique_sorted_hops_from_model(hop_model_obj) -> list:
    """
    Tries very hard to guess the hop variety names / feature columns
    that the hop model expects. We'll use those to drive the hop name dropdowns.
    If we can't figure it out from the model, fall back to some placeholder list.
    """
    fallback = ["Simcoe", "Amarillo", "Astra", "Citra", "Mosaic"]

    if hop_model_obj is None:
        return fallback

    # case 1: pipeline with feature_names_in_
    feats = []
    for attr in ["feature_names_in_", "features_in_", "feature_names_", "columns_", "expected_features"]:
        if hasattr(hop_model_obj, attr):
            cand = getattr(hop_model_obj, attr)
            feats.extend(ensure_listlike(cand))

    # case 2: pipeline.named_steps
    # sometimes hop_aroma_model["feature_names"] or similar if user stored dict
    if isinstance(hop_model_obj, dict):
        # if they've stashed variety names or columns in the dict:
        for k in ["feature_names", "hop_feature_names", "columns"]:
            if k in hop_model_obj:
                feats.extend(ensure_listlike(hop_model_obj[k]))

    # Clean up
    feats = [str(f) for f in feats if isinstance(f, (str, int, float))]
    feats = list(dict.fromkeys(feats))  # dedupe preserve order

    # Some models store columns like hop_Simcoe, hop_Amarillo -> strip 'hop_'
    cleaned = []
    for col in feats:
        c = str(col)
        if c.startswith("hop_"):
            c = c.replace("hop_", "")
        cleaned.append(c)

    cleaned = [c for c in cleaned if c not in ["", "None", "nan"]]
    if len(cleaned) == 0:
        return fallback

    # limit to a sane subset
    return cleaned[:30]


def aggregate_user_hops(hop_inputs):
    """
    hop_inputs = list of (hop_name, grams).
    We'll collapse duplicates and sum grams.
    Returns dict { "Simcoe": total_grams, "Citra": total_grams, ... }
    """
    agg = {}
    for (name, grams) in hop_inputs:
        if not name or name.strip() == "-":
            continue
        if grams <= 0:
            continue
        agg[name] = agg.get(name, 0.0) + grams
    return agg


def build_model_input_df(agg_hops: dict, model_feature_names: list):
    """
    Create a single-row DF whose columns match the model's expected
    hop columns (like hop_Simcoe, hop_Citra, etc). We'll fill grams for any
    variety the user used, 0 otherwise.
    If we can't figure out the model's feature names, we fallback to the
    user's keys directly (still 1-row).
    """
    if model_feature_names and len(model_feature_names) > 0:
        cols = []
        for feat in model_feature_names:
            # unify 'hop_Simcoe' style
            feat_str = str(feat)
            hop_var = feat_str
            if feat_str.startswith("hop_"):
                hop_var = feat_str.replace("hop_", "")
            cols.append(feat_str)

        data = {}
        for feat_str in cols:
            hop_var = feat_str
            if hop_var.startswith("hop_"):
                hop_var = hop_var.replace("hop_", "")

            grams = agg_hops.get(hop_var, 0.0)
            data[feat_str] = [grams]
        return pd.DataFrame(data)
    else:
        # fallback: just build columns from agg_hops
        if len(agg_hops) == 0:
            # user added nothing, produce a single row of zero
            return pd.DataFrame([0.0], index=["0"], columns=["total_hops"])
        else:
            return pd.DataFrame([agg_hops])


def safe_pipeline_predict(pipeline_obj, X_df: pd.DataFrame):
    """
    Try to call pipeline_obj.predict(X_df) and return np.array of scores.
    If it fails (like feature mismatch), return None and the error text.
    """
    if pipeline_obj is None:
        return None, "hop_model is None (not loaded)"
    try:
        preds = pipeline_obj.predict(X_df)
        return preds, None
    except Exception as e:
        return None, f"Predict exception: {e}"


def radial_plot(scores_dict, ax=None, annotate_center_val=None):
    """
    scores_dict: {axis_label: value, ...} in circular order
    We'll draw a radar / spider.
    If values missing or None -> treat as 0.
    """
    labels = list(scores_dict.keys())
    values = [max(0.0, float(scores_dict[k] or 0.0)) for k in labels]

    N = len(labels)
    if N == 0:
        labels = ["tropical", "citrus", "fruity", "resinous", "floral",
                  "herbal", "spicy", "earthy"]
        values = [0]*len(labels)
        N = len(labels)

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    if ax is None:
        fig, ax = plt.subplots(subplot_kw=dict(polar=True))
    else:
        fig = ax.figure
        ax.set_theta_offset(0)

    ax.plot(angles, values, color="#2F3A56", linewidth=2)
    ax.fill(angles, values, color="#2F3A56", alpha=0.2)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_yticks(np.linspace(0, max(1.0, max(values)), 5))
    ax.set_yticklabels([f"{y:.1f}" for y in np.linspace(0, max(1.0, max(values)), 5)])
    ax.set_ylim(0, max(1.0, max(values)))

    # Show center annotation (like "0.00")
    if annotate_center_val is None:
        center_val = np.mean(values[:-1]) if len(values) > 1 else 0.0
    else:
        center_val = annotate_center_val
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

    return fig, ax


def summarize_hop_aroma_vector(raw_scores):
    """
    Takes a numeric vector of hop aroma intensities [tropical, citrus, fruity, ...]
    Returns top 2 note names + (value).
    We'll just return placeholders if we can't parse.
    """
    if raw_scores is None or len(raw_scores) == 0:
        return [("tropical", 0.0), ("citrus", 0.0)]

    # define label order for final plot
    label_order = [
        "tropical", "citrus", "fruity", "resinous",
        "floral", "herbal", "spicy", "earthy",
    ]
    n = min(len(label_order), len(raw_scores))
    pairs = []
    for i in range(n):
        pairs.append((label_order[i], float(raw_scores[i])))

    # sort descending
    pairs_sorted = sorted(pairs, key=lambda x: x[1], reverse=True)
    top2 = pairs_sorted[:2]
    return top2 if len(top2) > 0 else [("tropical", 0.0), ("citrus", 0.0)]


def descriptive_malt_profile(malt_model_obj, malt_inputs_dict):
    """
    Basic placeholder malt text. You can hook into malt_model if you
    later train it. For now we guess by which malt % is highest.
    malt_inputs_dict = { 'malt1_name':..., 'malt1_pct':..., etc. }
    """
    # trivial heuristic
    top_name = "-"
    top_pct = 0.0
    for k, v in malt_inputs_dict.items():
        if k.endswith("_pct"):
            if v > top_pct:
                # figure out which malt name goes with this pct
                malt_idx = k.split("_")[0]  # 'malt1'
                nm_key = malt_idx + "_name"
                nm_val = malt_inputs_dict.get(nm_key, "-")
                top_pct = v
                top_name = nm_val

    # very naive mapping -> text
    if "BLACK" in str(top_name).upper():
        return "roasty / burnt"
    if "AMBER" in str(top_name).upper():
        return "bready"
    if "BEST" in str(top_name).upper() or "PALE" in str(top_name).upper():
        return "bready"
    return "bready"


def descriptive_yeast_profile(yeast_model_obj, yeast_choice: str, clean_yeast_df: pd.DataFrame):
    """
    We'll look up the chosen yeast in clean_yeast_df and grab 1-2
    flavor descriptors from columns 'fruity_esters', 'clean_neutral', etc.
    We return a tuple (yeast_flavor_text, style_direction_text).
    """
    if not yeast_choice or yeast_choice.strip() == "-":
        return ("clean / neutral", "Experimental / Hybrid")

    if clean_yeast_df is None or "Name" not in clean_yeast_df.columns:
        return ("clean / neutral", "Experimental / Hybrid")

    row = clean_yeast_df[clean_yeast_df["Name"] == yeast_choice]
    if row.empty:
        return ("clean / neutral", "Experimental / Hybrid")

    row = row.iloc[0]
    # Build a short descriptor. We'll check these columns if they exist:
    char_bits = []
    if "clean_neutral" in row and row["clean_neutral"] == 1:
        char_bits.append("clean / neutral")
    if "fruity_esters" in row and row["fruity_esters"] == 1:
        char_bits.append("fruity esters")
    if "phenolic_spicy" in row and row["phenolic_spicy"] == 1:
        char_bits.append("spicy phenolics")
    if len(char_bits) == 0:
        char_bits.append("clean / neutral")

    # style direction guess:
    style_dir = "Experimental / Hybrid"
    if "Notes" in row and isinstance(row["Notes"], str):
        # extremely naive guess
        if "lager" in row["Notes"].lower():
            style_dir = "Clean / Neutral Ale direction"
        elif "ester" in row["Notes"].lower():
            style_dir = "Fruity / English style"
    return (", ".join(char_bits), style_dir)


# ------------------------------------------------------------
# Sidebar UI
# ------------------------------------------------------------
def sidebar_inputs(hop_variety_list):
    st.sidebar.header("Model Inputs", anchor=None)
    st.sidebar.subheader("Hop Bill (g)")

    # We'll allow up to 3 hop slots
    hop_inputs = []
    for i in range(3):
        col1, col2 = st.sidebar.columns([2, 1])
        with col1:
            hop_name = st.selectbox(
                f"Hop {i+1}",
                options=["-"] + hop_variety_list,
                index=0,
                key=f"hop_name_{i}",
            )
        with col2:
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
    # malt1
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

    # malt2
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

    # malt3
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
    yeast_choice = "-"
    if clean_yeast_df is not None and "Name" in clean_yeast_df.columns:
        yeast_list = ["-"] + sorted(clean_yeast_df["Name"].astype(str).unique().tolist())
        yeast_choice = st.sidebar.selectbox(
            "Select yeast",
            options=yeast_list,
            index=0,
            key="yeast_choice_key",
        )
    else:
        yeast_choice = st.sidebar.selectbox(
            "Select yeast",
            options=["-"],
            index=0,
            key="yeast_choice_key_fallback",
        )

    run_button = st.sidebar.button("Predict Flavor 🧪", key="predict_button_key")

    return hop_inputs, malt_inputs, yeast_choice, run_button


# ------------------------------------------------------------
# Main App
# ------------------------------------------------------------
def main():
    st.title("🍺 Beer Recipe Digital Twin")
    st.write(
        "Predict hop aroma, malt character, and fermentation profile using trained ML models (work in progress)."
    )

    # figure out hop variety list from model
    hop_variety_list = unique_sorted_hops_from_model(hop_aroma_model)

    hop_inputs, malt_inputs, yeast_choice, run_button = sidebar_inputs(hop_variety_list)

    # prep data for hop model
    agg_hops = aggregate_user_hops(hop_inputs)

    # figure out the model's feature columns
    hop_feature_names = []
    if hasattr(hop_aroma_model, "feature_names_in_"):
        hop_feature_names = list(hop_aroma_model.feature_names_in_)
    elif hasattr(hop_aroma_model, "features_in_"):
        hop_feature_names = list(hop_aroma_model.features_in_)
    elif isinstance(hop_aroma_model, dict) and "feature_names" in hop_aroma_model:
        hop_feature_names = list(hop_aroma_model["feature_names"])

    aligned_df = build_model_input_df(agg_hops, hop_feature_names)

    aroma_scores = None
    model_error = None

    if run_button:
        if hop_aroma_model is not None:
            aroma_scores, model_error = safe_pipeline_predict(hop_aroma_model, aligned_df)
            # aroma_scores could be None, or could be array-like
        else:
            model_error = "No hop_aroma_model loaded."

    # Now build the radar data.
    # We'll parse aroma_scores (if any) and create a dict of {label: score}
    label_order = [
        "tropical", "citrus", "fruity", "resinous",
        "floral", "herbal", "spicy", "earthy",
    ]

    radar_dict = {}
    if aroma_scores is not None and not isinstance(aroma_scores, str):
        # assume shape (1, N) or (N,)
        try:
            arr = np.array(aroma_scores).flatten().tolist()
        except Exception:
            arr = []
        for i, lbl in enumerate(label_order):
            val = arr[i] if i < len(arr) else 0.0
            radar_dict[lbl] = float(val)
    else:
        # fallback 0 spider
        for lbl in label_order:
            radar_dict[lbl] = 0.0

    # top hop notes
    top2_notes = summarize_hop_aroma_vector(
        [radar_dict[lbl] for lbl in label_order]
    )
    # e.g. [("tropical", 0.12), ("citrus", 0.07)]

    # Malt text
    malt_desc = descriptive_malt_profile(malt_model, malt_inputs)

    # Yeast text
    yeast_desc, style_dir = descriptive_yeast_profile(yeast_model, yeast_choice, clean_yeast_df)

    # Layout for radar + side text
    col_plot, col_text = st.columns([2, 1])

    with col_plot:
        st.subheader("Hop Aroma Radar")
        fig, ax = plt.subplots(
            subplot_kw=dict(polar=True),
            figsize=(6, 6),
            dpi=150
        )
        # We'll reorder radar_dict to [tropical,citrus,fruity,resinous,floral,herbal,spicy,earthy]
        ordered_radar = {k: radar_dict[k] for k in label_order}
        radial_plot(
            ordered_radar,
            ax=ax,
            annotate_center_val=np.mean(list(ordered_radar.values()))
        )
        st.pyplot(fig, use_container_width=True)

    with col_text:
        st.markdown("### Top hop notes:")
        for name, val in top2_notes:
            st.write(f"• {name} ({val:.2f})")

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
        # show each hop & grams user used
        if len(agg_hops) == 0:
            st.write("(no hops added)")
        else:
            hop_list_lines = [f"{hop}: {grams:.1f} g" for hop, grams in agg_hops.items()]
            st.write(", ".join(hop_list_lines))

    # Debug panel
    if DEBUG_MODE:
        st.markdown("---")
        st.subheader("🧪 Debug: hop model input / prediction")

        st.write("hop_model is None?", hop_aroma_model is None)

        # Show guessed hop_feature_names
        st.write("hop_feature_names:")
        st.write(hop_feature_names[:200] if hop_feature_names else "(None)")

        if model_error:
            st.error(f"Model predict() error: {model_error}")

        st.write("aligned_df passed to model:")
        st.dataframe(aligned_df, use_container_width=True)

        st.write("User aggregate hop grams by hop variety:")
        st.dataframe(pd.DataFrame([agg_hops]), use_container_width=True)

        st.write("type(hop_model):")
        st.write(type(hop_aroma_model))
        st.write(dir(hop_aroma_model)[:20])

        st.markdown("---")
        st.subheader("🧬 Debug: yeast dataset")
        if clean_yeast_df is not None:
            st.write("Columns:")
            st.write(list(clean_yeast_df.columns))
            # show partial table
            st.dataframe(clean_yeast_df.head(10), use_container_width=True)
        else:
            st.write("clean_yeast_df is None")


# ------------------------------------------------------------
# run
# ------------------------------------------------------------
if __name__ == "__main__":
    main()

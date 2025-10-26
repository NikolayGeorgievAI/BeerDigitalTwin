import streamlit as st
import numpy as np
import pandas as pd
import joblib
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# -----------------------------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Beer Recipe Digital Twin",
    page_icon="🍺",
    layout="wide"
)

# -----------------------------------------------------------------------------
# LOAD MODELS + DATA (CACHED)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_models_and_data():
    """
    Loads:
      - hop aroma model bundle (model, feature_cols, aroma_dims)
      - malt sensory model bundle (model, feature_cols, flavor_cols)
      - yeast sensory model bundle (model, feature_cols, flavor_cols)
      - clean malt dataframe (for malt chemistry features)
      - clean yeast dataframe (for yeast metadata)
    """
    hop_bundle   = joblib.load("hop_aroma_model.joblib")
    malt_bundle  = joblib.load("malt_sensory_model.joblib")
    yeast_bundle = joblib.load("yeast_sensory_model.joblib")

    hop_model      = hop_bundle["model"]
    hop_features   = hop_bundle["feature_cols"]     # e.g. ["hop_Citra","hop_Mosaic",...]
    hop_dims       = hop_bundle["aroma_dims"]       # e.g. ["tropical","citrus","floral",...]

    malt_model     = malt_bundle["model"]
    malt_features  = malt_bundle["feature_cols"]    # numeric malt chemistry columns
    malt_dims      = malt_bundle["flavor_cols"]     # e.g. ["bready","caramel","roasty",...]

    yeast_model    = yeast_bundle["model"]
    yeast_features = yeast_bundle["feature_cols"]   # e.g. ["Temp_avg_C","Flocculation_num","Attenuation_pct"]
    yeast_dims     = yeast_bundle["flavor_cols"]    # e.g. ["clean_neutral","fruity_esters",...]

    malt_df  = pd.read_pickle("clean_malt_df.pkl")
    yeast_df = pd.read_pickle("clean_yeast_df.pkl")

    return (
        hop_model, hop_features, hop_dims,
        malt_model, malt_features, malt_dims,
        yeast_model, yeast_features, yeast_dims,
        malt_df, yeast_df
    )


(
    hop_model, hop_features, hop_dims,
    malt_model, malt_features, malt_dims,
    yeast_model, yeast_features, yeast_dims,
    malt_df, yeast_df
) = load_models_and_data()

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------

# ---- HOPS ----
def get_all_hop_names(hop_features):
    """
    Each hop feature column is like "hop_Citra". We strip "hop_" off.
    """
    return [c.replace("hop_", "") for c in hop_features]

def build_hop_feature_vector(hop_bill_dict, hop_features):
    """
    hop_bill_dict looks like: {"Citra": 40, "Mosaic": 20, ... grams}
    Return a 2D array (1 x n_features) aligned to hop_features.
    If a hop isn't in hop_bill_dict, intensity=0.
    We also normalize the bill so total grams sum to 1.0-ish,
    because most models trained on relative %.
    """
    # total grams
    total = sum(hop_bill_dict.values())
    row = []
    for col in hop_features:
        hop_name = col.replace("hop_", "")
        grams = hop_bill_dict.get(hop_name, 0.0)
        if total > 0:
            row.append(grams / total)
        else:
            row.append(0.0)
    x = np.array(row).reshape(1, -1)
    return x, total

def predict_hop_profile(hop_bill_dict, hop_model, hop_features, hop_dims):
    """
    Returns dict of aroma intensities {dim: value}
    """
    x, total = build_hop_feature_vector(hop_bill_dict, hop_features)
    y_pred = hop_model.predict(x)[0]  # numeric intensities per dim
    hop_out = dict(zip(hop_dims, y_pred))

    return hop_out, total


# ---- MALTS ----
def get_weighted_malt_vector(malt_selections, malt_df, malt_features):
    """
    malt_selections = [
        {"name": "Maris Otter", "pct": 70.0},
        {"name": "Crystal 60L", "pct": 20.0},
        {"name": "Flaked Oats", "pct": 10.0}
    ]
    We'll compute a weighted average of these malts' chemistry (malt_features).
    """
    blend_vec = np.zeros(len(malt_features), dtype=float)
    total_pct = sum(float(m["pct"]) for m in malt_selections)

    if total_pct <= 0:
        return blend_vec.reshape(1, -1)

    for item in malt_selections:
        malt_name = item["name"]
        pct       = float(item["pct"])

        row = malt_df[malt_df["PRODUCT NAME"] == malt_name].head(1)
        if row.empty:
            continue

        vec = np.array([row.iloc[0][feat] for feat in malt_features], dtype=float)
        weight = pct / total_pct
        blend_vec += vec * weight

    return blend_vec.reshape(1, -1)

def predict_malt_profile(malt_selections, malt_model, malt_df, malt_features, malt_dims):
    """
    Returns dict {flavor_dim: 0/1 or score}
    """
    x = get_weighted_malt_vector(malt_selections, malt_df, malt_features)
    y_pred = malt_model.predict(x)[0]
    out = dict(zip(malt_dims, y_pred))
    return out


# ---- YEAST ----
def get_yeast_feature_vector(yeast_name, yeast_df, yeast_features):
    """
    Build single-row model input for chosen yeast strain.
    We assume yeast_df has columns used in yeast_features.
    """
    row = yeast_df[yeast_df["Name"] == yeast_name].head(1)
    if row.empty:
        # fallback zeros
        return np.zeros(len(yeast_features)).reshape(1, -1)

    # we trust yeast_features order matches columns in df
    vec = [row.iloc[0][feat] for feat in yeast_features]
    return np.array(vec).reshape(1, -1)

def predict_yeast_profile(yeast_name, yeast_model, yeast_df, yeast_features, yeast_dims):
    """
    Returns dict {flavor_dim: 0/1 or score}
    """
    x = get_yeast_feature_vector(yeast_name, yeast_df, yeast_features)
    y_pred = yeast_model.predict(x)[0]
    out = dict(zip(yeast_dims, y_pred))
    return out


# ---- STYLE HEURISTIC ----
def guess_style(hop_out, malt_out, yeast_out):
    """
    Dumb but fun heuristic to label 'style direction'.
    """
    style_guess = "Experimental / Hybrid"

    # Yeast-driven buckets
    if yeast_out.get("clean_neutral", 0) == 1:
        # if it's clean + neutral
        if malt_out.get("sweet_malt", 0) == 1 or malt_out.get("caramel", 0) == 1:
            style_guess = "Clean / Neutral Ale direction"
        else:
            style_guess = "Crisp / Neutral Fermentation direction"

    if yeast_out.get("fruity_esters", 0) == 1:
        # hazy-ish direction if tropical is strong
        if hop_out.get("tropical", 0) > 0.6:
            style_guess = "Juicy / Hazy Ale direction"
        else:
            style_guess = "Fruity / Ester-forward direction"

    if yeast_out.get("phenolic_spicy", 0) == 1:
        style_guess = "Belgian / Saison direction"

    if malt_out.get("roasty", 0) == 1 or malt_out.get("dark_roast", 0) == 1:
        style_guess = "Roasty / Dark Ale direction"

    return style_guess


# ---- TOP HOP NOTES ----
def get_top_hop_notes(hop_out, n=2):
    """
    Sort aroma dims by intensity desc and return label + value
    """
    items = sorted(hop_out.items(), key=lambda x: x[1], reverse=True)
    return items[:n]


# -----------------------------------------------------------------------------
# RADAR / SPIDER PLOT
# -----------------------------------------------------------------------------
def make_spider_plot(hop_out_dict):
    """
    Draw a filled radar/spider chart of hop aroma intensities.

    We'll use a consistent axis order:
    tropical, citrus, fruity, resinous, floral, herbal, spicy, earthy
    """

    axes_order = [
        "tropical",
        "citrus",
        "fruity",
        "resinous",
        "floral",
        "herbal",
        "spicy",
        "earthy",
    ]
    vals = [float(hop_out_dict.get(dim, 0.0)) for dim in axes_order]

    # close polygon
    vals_closed = vals + [vals[0]]
    angles = np.linspace(0, 2 * np.pi, len(axes_order), endpoint=False)
    angles_closed = np.concatenate([angles, [angles[0]]])

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_facecolor("#fafafa")

    # polygon line + fill
    ax.plot(
        angles_closed,
        vals_closed,
        color="#1f77b4",
        linewidth=2,
    )
    ax.fill(
        angles_closed,
        vals_closed,
        color="#1f77b4",
        alpha=0.25,
    )

    # numeric labels at each vertex
    for ang, val in zip(angles, vals):
        ax.text(
            ang,
            val,
            f"{val:.2f}",
            ha="center",
            va="center",
            fontsize=10,
            bbox=dict(
                boxstyle="round,pad=0.2",
                fc="white",
                ec="#1f77b4",
                lw=1
            ),
        )

    # category labels
    ax.set_xticks(angles)
    ax.set_xticklabels(axes_order, fontsize=12)

    # radial web styling
    ax.set_rlabel_position(0)
    ax.yaxis.grid(color="gray", linestyle="--", alpha=0.4)
    ax.xaxis.grid(color="gray", linestyle="--", alpha=0.4)

    vmax = max(max(vals), 0.5)
    ax.set_ylim(0, vmax * 1.2)

    ax.set_title("Hop Aroma Radar", fontsize=24, fontweight="bold", pad=20)
    fig.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# SIDEBAR UI
# -----------------------------------------------------------------------------
st.sidebar.markdown("## 🧪 Model Inputs")
st.sidebar.markdown("### Hop Bill (g)")

all_hops_list = get_all_hop_names(hop_features)

def hop_input(slot_idx: int, default_idx: int):
    # pick default_idx if possible, else 0
    safe_idx = default_idx if default_idx < len(all_hops_list) else 0
    hop_name = st.sidebar.selectbox(
        f"Hop {slot_idx}",
        all_hops_list,
        index=safe_idx,
        key=f"hop_select_{slot_idx}"
    )
    hop_amt = st.sidebar.number_input(
        f"{hop_name} (g)",
        min_value=0.0,
        max_value=500.0,
        value=0.0,
        step=5.0,
        key=f"hop_amount_{slot_idx}"
    )
    return hop_name, hop_amt

hop1_name, hop1_amt = hop_input(1, 0)
hop2_name, hop2_amt = hop_input(2, 1)
hop3_name, hop3_amt = hop_input(3, 2)
hop4_name, hop4_amt = hop_input(4, 3)

hop_bill = {
    hop1_name: hop1_amt,
    hop2_name: hop2_amt,
    hop3_name: hop3_amt,
    hop4_name: hop4_amt,
}

# MALTS
st.sidebar.markdown("### Malt Bill")
malt_options = sorted(malt_df["PRODUCT NAME"].unique().tolist())

def malt_picker(slot_label: str, default_idx: int, default_pct: float):
    safe_idx = default_idx if default_idx < len(malt_options) else 0
    m_name = st.sidebar.selectbox(
        f"{slot_label}",
        malt_options,
        index=safe_idx,
        key=f"{slot_label}_name"
    )
    m_pct = st.sidebar.number_input(
        f"{slot_label} %",
        min_value=0.0,
        max_value=100.0,
        value=default_pct,
        step=1.0,
        key=f"{slot_label}_pct"
    )
    return {"name": m_name, "pct": m_pct}

malt1 = malt_picker("Malt 1", 0, 70.0)
malt2 = malt_picker("Malt 2", 1, 20.0)
malt3 = malt_picker("Malt 3", 2, 10.0)

malt_selections = [malt1, malt2, malt3]

# YEAST
st.sidebar.markdown("### Yeast Strain")
yeast_options = sorted(yeast_df["Name"].dropna().unique().tolist())
default_yeast_idx = 0 if len(yeast_options) > 0 else 0
chosen_yeast = st.sidebar.selectbox(
    "Select yeast",
    yeast_options,
    index=default_yeast_idx,
    key="yeast_choice"
)

# RUN BUTTON
run_button = st.sidebar.button("Predict Flavor 🧪")

# -----------------------------------------------------------------------------
# MAIN LAYOUT
# -----------------------------------------------------------------------------
title_col_spacer, title_col_main = st.columns([0.02, 0.98])
with title_col_main:
    st.markdown(
        "## 🍺 Beer Recipe Digital Twin",
    )
    st.caption(
        "Predict hop aroma, malt character, and fermentation profile using trained ML models."
    )


if not run_button:
    # BEFORE running prediction
    st.info(
        "👉 Build your hop bill (up to 4 hops, with nonzero grams), "
        "set your malt bill (% grist), choose yeast, then click **Predict Flavor 🧪** "
        "in the sidebar."
    )

else:
    # AFTER running prediction
    hop_out, total_hop_grams = predict_hop_profile(
        hop_bill,
        hop_model,
        hop_features,
        hop_dims
    )

    malt_out = predict_malt_profile(
        malt_selections,
        malt_model,
        malt_df,
        malt_features,
        malt_dims
    )

    yeast_out = predict_yeast_profile(
        chosen_yeast,
        yeast_model,
        yeast_df,
        yeast_features,
        yeast_dims
    )

    style_guess = guess_style(hop_out, malt_out, yeast_out)
    top_notes = get_top_hop_notes(hop_out, n=2)

    # --- LAYOUT: radar left / text right
    left_col, right_col = st.columns([0.6, 0.4], vertical_alignment="top")

    with left_col:
        fig = make_spider_plot(hop_out)
        st.pyplot(fig, use_container_width=True)

    with right_col:
        st.markdown("### Top hop notes:")
        if top_notes:
            for note, val in top_notes:
                st.write(f"- **{note}** ({val:.2f})")
        else:
            st.write("_No dominant hop note_")

        st.markdown("### Malt character:")
        # just show which malt dims are 'on' or highest
        malt_active = [dim for dim, v in malt_out.items() if v == 1 or v > 0.5]
        if len(malt_active) == 0:
            # fallback to top "scorey" dimension
            malt_sorted = sorted(malt_out.items(), key=lambda x: x[1], reverse=True)
            if malt_sorted:
                malt_active = [malt_sorted[0][0]]
        if malt_active:
            st.write(", ".join(malt_active))
        else:
            st.write("_None_")

        st.markdown("### Yeast character:")
        yeast_active = [dim for dim, v in yeast_out.items() if v == 1 or v > 0.5]
        if len(yeast_active) == 0:
            yeast_sorted = sorted(yeast_out.items(), key=lambda x: x[1], reverse=True)
            if yeast_sorted:
                yeast_active = [yeast_sorted[0][0]]
        if yeast_active:
            st.write(", ".join(yeast_active))
        else:
            st.write("_None_")

        st.markdown("### Style direction:")
        st.write(f"🧭 {style_guess}")

        st.markdown("### Hops used by the model:")
        used_hops = [name for name, amt in hop_bill.items() if amt > 0]
        if used_hops:
            st.write(", ".join(used_hops))
        else:
            st.write("_No hops in bill (or all 0g)_")

    # If total hops = 0, warn why radar might look flat
    if total_hop_grams <= 0:
        st.warning(
            "⚠ No recognizable hops (or total was 0 after normalization), "
            "so hop aroma prediction is basically flat. "
            "Add some grams of real hops and click Predict Flavor again."
        )

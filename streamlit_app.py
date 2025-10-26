import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Beer Recipe Digital Twin",
    page_icon="🍺",
    layout="wide",
)

# -----------------------------------------------------------------------------
# LOAD MODELS + DATA (CACHED)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_models_and_data():
    """
    Load all trained models + reference data once, cache them.
    We assume these joblib/pkl files are present in the repo root.
    """
    hop_bundle   = joblib.load("hop_aroma_model.joblib")
    malt_bundle  = joblib.load("malt_sensory_model.joblib")
    yeast_bundle = joblib.load("yeast_sensory_model.joblib")

    hop_model      = hop_bundle["model"]
    hop_features   = hop_bundle["feature_cols"]   # e.g. ["hop_Adeena","hop_Admiral",...]
    hop_dims       = hop_bundle["aroma_dims"]     # e.g. ["tropical","citrus","fruity",...]

    malt_model     = malt_bundle["model"]
    malt_features  = malt_bundle["feature_cols"]
    malt_dims      = malt_bundle["flavor_cols"]

    yeast_model    = yeast_bundle["model"]
    yeast_features = yeast_bundle["feature_cols"]
    yeast_dims     = yeast_bundle["flavor_cols"]

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
# HELPERS: HOPS
# -----------------------------------------------------------------------------
def all_hop_names_from_features(hop_features):
    # hop_features like "hop_Adeena", "hop_Admiral"
    # -> ["Adeena","Admiral",...]
    return [c.replace("hop_", "") for c in hop_features]

def build_hop_feature_row(hop_bill_dict, hop_features):
    """
    hop_bill_dict: {"Adeena": 40, "Admiral": 20, ...}
    We must return a row (1 x n_features) matching hop_features order.
    """
    row_vals = []
    for col in hop_features:
        hopname = col.replace("hop_", "")
        row_vals.append(float(hop_bill_dict.get(hopname, 0.0)))
    return np.array(row_vals).reshape(1, -1)

def predict_hop_profile(hop_bill_dict, hop_model, hop_features, hop_dims):
    x = build_hop_feature_row(hop_bill_dict, hop_features)
    y_pred = hop_model.predict(x)[0]  # numeric intensities
    return dict(zip(hop_dims, y_pred))

# -----------------------------------------------------------------------------
# HELPERS: MALTS
# -----------------------------------------------------------------------------
def malt_blend_vector(malt_selections, malt_df, malt_features):
    """
    malt_selections:
    [
      {"name": "AMBER MALT", "pct": 70},
      {"name": "BEST ALE MALT", "pct": 20},
      {"name": "BLACK MALT", "pct": 10},
    ]
    We'll build a weighted average of the chemistry columns in malt_features.
    """
    blend = np.zeros(len(malt_features), dtype=float)

    for item in malt_selections:
        malt_name = item["name"]
        pct       = float(item["pct"])
        row = malt_df[malt_df["PRODUCT NAME"] == malt_name].head(1)

        if row.empty:
            continue

        vec = np.array([row.iloc[0][feat] for feat in malt_features], dtype=float)
        blend += vec * (pct / 100.0)

    return blend.reshape(1, -1)

def predict_malt_profile_from_blend(malt_selections, malt_model, malt_df, malt_features, malt_dims):
    x = malt_blend_vector(malt_selections, malt_df, malt_features)
    y_pred = malt_model.predict(x)[0]  # often 0/1 classification for each flavor flag
    return dict(zip(malt_dims, y_pred))

# -----------------------------------------------------------------------------
# HELPERS: YEAST
# -----------------------------------------------------------------------------
def yeast_feature_row(yeast_name, yeast_df, yeast_features):
    """
    yeast_df should have columns like:
    Name, Temp_avg_C, Flocculation_num, Attenuation_pct ...
    We'll build [Temp_avg_C, Flocculation_num, Attenuation_pct]
    """
    row = yeast_df[yeast_df["Name"] == yeast_name].head(1)
    if row.empty:
        return np.zeros(len(yeast_features)).reshape(1, -1)

    vec = []
    for f in yeast_features:
        vec.append(float(row.iloc[0][f]))
    return np.array(vec).reshape(1, -1)

def predict_yeast_profile(yeast_name, yeast_model, yeast_df, yeast_features, yeast_dims):
    x = yeast_feature_row(yeast_name, yeast_df, yeast_features)
    y_pred = yeast_model.predict(x)[0]  # often 0/1 flags
    return dict(zip(yeast_dims, y_pred))

# -----------------------------------------------------------------------------
# SUMMARY LOGIC
# -----------------------------------------------------------------------------
def summarize_beer(
    hop_bill_dict,
    malt_selections,
    yeast_name,
    hop_model, hop_features, hop_dims,
    malt_model, malt_df, malt_features, malt_dims,
    yeast_model, yeast_df, yeast_features, yeast_dims,
):
    hop_out   = predict_hop_profile(hop_bill_dict, hop_model, hop_features, hop_dims)
    malt_out  = predict_malt_profile_from_blend(malt_selections, malt_model, malt_df, malt_features, malt_dims)
    yeast_out = predict_yeast_profile(yeast_name, yeast_model, yeast_df, yeast_features, yeast_dims)

    # "top hop notes": pick two highest intensities
    hop_sorted = sorted(hop_out.items(), key=lambda kv: kv[1], reverse=True)
    top_hop_notes = [f"{k} ({round(v, 2)})" for k, v in hop_sorted[:2]]

    # Malt / Yeast trait lists (any '1' or True style)
    malt_traits  = [k for k,v in malt_out.items() if v == 1]
    yeast_traits = [k for k,v in yeast_out.items() if v == 1]

    # Guess style from a couple heuristics:
    style_guess = "Experimental / Hybrid"
    if ("clean_neutral" in yeast_out and yeast_out["clean_neutral"] == 1
        and "dry_finish" in yeast_out and yeast_out["dry_finish"] == 1):
        style_guess = "West Coast IPA / Modern IPA"

    if ("fruity_esters" in yeast_out and yeast_out["fruity_esters"] == 1) and \
       ("tropical" in hop_out and hop_out["tropical"] > 0.6):
        style_guess = "Hazy / NEIPA leaning"

    if ("phenolic_spicy" in yeast_out and yeast_out["phenolic_spicy"] == 1):
        style_guess = "Belgian / Saison leaning"

    if ("caramel" in malt_out and malt_out["caramel"] == 1):
        style_guess = "English / Malt-forward Ale"

    return {
        "hop_out": hop_out,
        "hop_top_notes": top_hop_notes,
        "malt_traits": malt_traits,
        "yeast_traits": yeast_traits,
        "style_guess": style_guess
    }

# -----------------------------------------------------------------------------
# SPIDER-WEB RADAR PLOT
# -----------------------------------------------------------------------------
def plot_hop_radar(hop_profile, title="Hop Aroma Radar"):
    """
    Draw a proper spider-web radar:
    - polygon rings
    - radial spokes
    - filled aroma polygon
    """

    # Fallback if hop_profile is empty
    if not hop_profile:
        # Give 7 standard-ish dims so chart doesn't explode
        hop_profile = {
            "tropical": 0.0,
            "citrus": 0.0,
            "fruity": 0.0,
            "resinous": 0.0,
            "floral": 0.0,
            "herbal": 0.0,
            "earthy": 0.0
        }

    labels = list(hop_profile.keys())
    values = [float(hop_profile[k]) for k in labels]

    # Close polygon
    values_closed = values + [values[0]]

    num_axes = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_axes, endpoint=False)
    angles_closed = list(angles) + [angles[0]]

    # Nice radial max
    vmax_candidate = max(values) if values else 1.0
    vmax = max(1.0, vmax_candidate * 1.2)

    fig, ax = plt.subplots(
        figsize=(4,4),
        subplot_kw=dict(polar=True)
    )

    # -- spider web rings --
    num_rings = 5
    ring_radii = np.linspace(0, vmax, num_rings+1)[1:]  # skip 0

    for r in ring_radii:
        ax.plot(angles_closed, [r]*len(angles_closed),
                color="gray", linewidth=0.8, linestyle="--", alpha=0.5)

    # -- spokes --
    for ang in angles:
        ax.plot([ang, ang], [0, vmax],
                color="gray", linewidth=0.8, linestyle="--", alpha=0.5)

    # -- aroma polygon --
    ax.plot(angles_closed, values_closed,
            linewidth=2, color="#1f77b4")
    ax.fill(angles_closed, values_closed,
            color="#1f77b4", alpha=0.25)

    # numeric labels at each vertex
    for ang, val in zip(angles, values):
        ax.text(
            ang,
            val,
            f"{val:.2f}",
            ha="center",
            va="center",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2",
                      fc="white",
                      ec="#1f77b4",
                      lw=0.8),
        )

    # category names
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=10)

    # hide radial tick labels (we draw custom web instead)
    ax.set_yticklabels([])
    ax.set_ylim(0, vmax)

    # turn off default polar grid
    ax.grid(False)

    # some older matplotlib backends might complain about spines["polar"]
    # so we do it safely:
    if hasattr(ax, "spines") and "polar" in ax.spines:
        ax.spines["polar"].set_visible(False)

    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    fig.tight_layout()

    return fig

# -----------------------------------------------------------------------------
# SIDEBAR UI
# -----------------------------------------------------------------------------
st.sidebar.header("🍃 Hop Bill")

all_hops_sorted = sorted(all_hop_names_from_features(hop_features))

# We'll keep 4 hop slots (hop1..hop4)
# We'll store in session_state so we don't crash on rerun order
def _safe_get_session(key, default):
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]

# init session keys:
_safe_get_session("hop1_name", all_hops_sorted[0] if all_hops_sorted else "")
_safe_get_session("hop2_name", all_hops_sorted[1] if len(all_hops_sorted)>1 else (all_hops_sorted[0] if all_hops_sorted else ""))
_safe_get_session("hop3_name", all_hops_sorted[2] if len(all_hops_sorted)>2 else (all_hops_sorted[0] if all_hops_sorted else ""))
_safe_get_session("hop4_name", all_hops_sorted[3] if len(all_hops_sorted)>3 else (all_hops_sorted[0] if all_hops_sorted else ""))

with st.sidebar.expander("Hop 1", expanded=True):
    st.session_state.hop1_name = st.selectbox(
        "Hop 1",
        all_hops_sorted,
        index=all_hops_sorted.index(st.session_state.hop1_name)
              if st.session_state.hop1_name in all_hops_sorted else 0,
        key="hop1_select"
    )
    hop1_g = st.number_input(
        f"{st.session_state.hop1_name} (g)",
        min_value=0.0,
        max_value=200.0,
        value=40.0,
        step=5.0,
        key="hop1_g"
    )

with st.sidebar.expander("Hop 2", expanded=False):
    st.session_state.hop2_name = st.selectbox(
        "Hop 2",
        all_hops_sorted,
        index=all_hops_sorted.index(st.session_state.hop2_name)
              if st.session_state.hop2_name in all_hops_sorted else 0,
        key="hop2_select"
    )
    hop2_g = st.number_input(
        f"{st.session_state.hop2_name} (g)",
        min_value=0.0,
        max_value=200.0,
        value=40.0,
        step=5.0,
        key="hop2_g"
    )

with st.sidebar.expander("Hop 3", expanded=False):
    st.session_state.hop3_name = st.selectbox(
        "Hop 3",
        all_hops_sorted,
        index=all_hops_sorted.index(st.session_state.hop3_name)
              if st.session_state.hop3_name in all_hops_sorted else 0,
        key="hop3_select"
    )
    hop3_g = st.number_input(
        f"{st.session_state.hop3_name} (g)",
        min_value=0.0,
        max_value=200.0,
        value=0.0,
        step=5.0,
        key="hop3_g"
    )

with st.sidebar.expander("Hop 4", expanded=False):
    st.session_state.hop4_name = st.selectbox(
        "Hop 4",
        all_hops_sorted,
        index=all_hops_sorted.index(st.session_state.hop4_name)
              if st.session_state.hop4_name in all_hops_sorted else 0,
        key="hop4_select"
    )
    hop4_g = st.number_input(
        f"{st.session_state.hop4_name} (g)",
        min_value=0.0,
        max_value=200.0,
        value=0.0,
        step=5.0,
        key="hop4_g"
    )

# Compose the hop bill dict
hop_bill = {
    st.session_state.hop1_name: hop1_g,
    st.session_state.hop2_name: hop2_g,
    st.session_state.hop3_name: hop3_g,
    st.session_state.hop4_name: hop4_g,
}

# MALT BILL
st.sidebar.header("🌵 Malt Bill")
malt_choices = sorted(malt_df["PRODUCT NAME"].dropna().unique().tolist())

with st.sidebar.expander("Malt 1", expanded=True):
    malt1_name = st.selectbox("Malt 1", malt_choices, key="malt1_name")
    malt1_pct  = st.number_input("Malt 1 %", min_value=0.0, max_value=100.0,
                                 value=70.0, step=1.0, key="malt1_pct")
with st.sidebar.expander("Malt 2", expanded=False):
    malt2_name = st.selectbox("Malt 2", malt_choices, key="malt2_name")
    malt2_pct  = st.number_input("Malt 2 %", min_value=0.0, max_value=100.0,
                                 value=20.0, step=1.0, key="malt2_pct")
with st.sidebar.expander("Malt 3", expanded=False):
    malt3_name = st.selectbox("Malt 3", malt_choices, key="malt3_name")
    malt3_pct  = st.number_input("Malt 3 %", min_value=0.0, max_value=100.0,
                                 value=10.0, step=1.0, key="malt3_pct")

malt_bill_list = [
    {"name": malt1_name, "pct": malt1_pct},
    {"name": malt2_name, "pct": malt2_pct},
    {"name": malt3_name, "pct": malt3_pct},
]

# YEAST
st.sidebar.header("🧪 Yeast")
yeast_choices = sorted(yeast_df["Name"].dropna().unique().tolist())
chosen_yeast  = st.selectbox("Yeast Strain", yeast_choices, key="chosen_yeast")

run_button = st.sidebar.button("Predict Flavor 🧪")

# -----------------------------------------------------------------------------
# MAIN PAGE LAYOUT
# -----------------------------------------------------------------------------
st.title("🍺 Beer Recipe Digital Twin")
st.caption("Predict hop aroma, malt character, and fermentation profile using trained ML models.")

left_col, right_col = st.columns([1.2, 0.8], vertical_alignment="top")

if run_button:
    summary = summarize_beer(
        hop_bill,
        malt_bill_list,
        chosen_yeast,
        hop_model, hop_features, hop_dims,
        malt_model, malt_df, malt_features, malt_dims,
        yeast_model, yeast_df, yeast_features, yeast_dims
    )

    hop_profile  = summary["hop_out"]
    hop_notes    = summary["hop_top_notes"]
    malt_traits  = summary["malt_traits"]
    yeast_traits = summary["yeast_traits"]
    style_guess  = summary["style_guess"]

    with left_col:
        st.subheader("Hop Aroma Radar")
        fig = plot_hop_radar(hop_profile, title="Hop Aroma Radar")
        st.pyplot(fig, use_container_width=False)

    with right_col:
        st.subheader("Top hop notes:")
        if hop_notes:
            for n in hop_notes:
                st.write(f"- {n}")
        else:
            st.write("None")

        st.subheader("Malt character:")
        if malt_traits:
            st.write(", ".join(malt_traits))
        else:
            st.write("None")

        st.subheader("Yeast character:")
        if yeast_traits:
            st.write(", ".join(yeast_traits))
        else:
            st.write("None")

        st.subheader("Style direction:")
        st.write(f"🧭 {style_guess}")

        st.subheader("Hops used by the model:")
        used_hops = [name for name, grams in hop_bill.items() if grams and grams > 0]
        st.write(", ".join(used_hops) if used_hops else "None")

else:
    st.info(
        "👉 Build your hop bill (up to 4 hops, with nonzero grams), "
        "set malt bill (% grist), choose yeast, then click **Predict Flavor 🧪** in the sidebar."
    )

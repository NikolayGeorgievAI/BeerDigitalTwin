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

st.markdown(
    """
    <style>
    /* tighten sidebar widgets spacing */
    section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] {
        gap: 0.4rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# CACHE: LOAD MODELS AND DATA
# -----------------------------------------------------------------------------
@st.cache_resource
def load_models_and_data():
    """
    Load all trained models + reference data once, cache them for the session.
    We assume these files exist in the repo root.
    """

    hop_bundle   = joblib.load("hop_aroma_model.joblib")
    malt_bundle  = joblib.load("malt_sensory_model.joblib")
    yeast_bundle = joblib.load("yeast_sensory_model.joblib")

    hop_model      = hop_bundle["model"]
    hop_features   = hop_bundle["feature_cols"]   # e.g. ["hop_Citra", ...]
    hop_dims       = hop_bundle["aroma_dims"]     # e.g. ["tropical","citrus",...]

    malt_model     = malt_bundle["model"]
    malt_features  = malt_bundle["feature_cols"]  # e.g. malt chemistry columns
    malt_dims      = malt_bundle["flavor_cols"]   # predicted malt traits

    yeast_model    = yeast_bundle["model"]
    yeast_features = yeast_bundle["feature_cols"] # e.g. Temp_avg_C, etc.
    yeast_dims     = yeast_bundle["flavor_cols"]  # predicted yeast traits

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

# Precompute hop names the model actually knows
ALL_MODEL_HOPS = [c.replace("hop_", "") for c in hop_features]


# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------

# ---- HOPS ----
def build_hop_features(hop_bill_dict, hop_features):
    """
    hop_bill_dict = {"Citra": 40, "Mosaic": 20, ...}
    Return:
      x -> (1, n_features) array aligned with hop_features order.
      matched -> list of hop names that successfully mapped into x.
    We also L1-normalize by total grams so model input is proportion not mass.
    """
    row_vals = []
    matched = []
    total = sum(hop_bill_dict.values()) if len(hop_bill_dict) > 0 else 0.0

    for col in hop_features:
        hop_name = col.replace("hop_", "")
        grams = hop_bill_dict.get(hop_name, 0.0)

        # proportion of total
        val = grams / total if total > 0 else 0.0
        if grams > 0:
            matched.append(hop_name)

        row_vals.append(val)

    x = np.array(row_vals, dtype=float).reshape(1, -1)
    return x, matched


def predict_hop_profile(hop_bill_dict, hop_model, hop_features, hop_dims):
    """
    Predict aroma intensity from the hop model and return:
      profile  -> {dimension: normalized_intensity}
      matched  -> hops that matched feature columns
    Adds diagnostics; rescales predictions 0-1 for nicer radar visibility.
    """
    x, matched = build_hop_features(hop_bill_dict, hop_features)

    # If nothing matched or all proportions are 0, warn and bail gracefully
    if np.sum(x) == 0 or len(matched) == 0:
        # This triggers when:
        # - user chose hops not in training set
        # - or they entered 0g for all hops (total==0)
        profile = {dim: 0.0 for dim in hop_dims}
        return profile, matched, x

    # Model prediction
    y_pred = hop_model.predict(x)[0]  # raw intensities (can be any scale / sign)

    # Clean up negatives (cosmetic)
    y_pred = np.maximum(0, y_pred)

    # Normalize purely for plotting so spider-web isn't flat
    max_val = y_pred.max() if y_pred.size > 0 else 1.0
    if max_val > 0:
        y_pred = y_pred / max_val

    profile = dict(zip(hop_dims, y_pred))

    return profile, matched, x


# ---- MALTS ----
def get_weighted_malt_vector(malt_selections, malt_df, malt_features):
    """
    malt_selections looks like:
    [
      {"name": "Maris Otter", "pct": 70},
      {"name": "Crystal 60L", "pct": 20},
      {"name": "Flaked Oats", "pct": 10}
    ]

    We build a weighted blend of the malt_features based on pct of grist.
    """
    blend_vec = np.zeros(len(malt_features), dtype=float)

    for item in malt_selections:
        malt_name = item["name"]
        pct       = float(item["pct"])

        row = malt_df[malt_df["PRODUCT NAME"] == malt_name].head(1)
        if row.empty:
            continue

        vec = np.array([row.iloc[0][feat] for feat in malt_features], dtype=float)
        blend_vec += vec * (pct / 100.0)

    return blend_vec.reshape(1, -1)


def predict_malt_profile_from_blend(malt_selections, malt_model, malt_df, malt_features, malt_dims):
    x = get_weighted_malt_vector(malt_selections, malt_df, malt_features)
    y_pred = malt_model.predict(x)[0]  # typically 0/1 flags for traits
    return dict(zip(malt_dims, y_pred))


# ---- YEAST ----
def get_yeast_feature_vector(yeast_name, yeast_df, yeast_features):
    """
    Build single-row model input for chosen yeast strain.
    Assumes yeast_df has these columns:
      - Temp_avg_C
      - Flocculation_num
      - Attenuation_pct
    """
    row = yeast_df[yeast_df["Name"] == yeast_name].head(1)
    if row.empty:
        return np.zeros(len(yeast_features)).reshape(1, -1)

    vec = [
        row.iloc[0]["Temp_avg_C"],
        row.iloc[0]["Flocculation_num"],
        row.iloc[0]["Attenuation_pct"]
    ]
    return np.array(vec).reshape(1, -1)


def predict_yeast_profile(yeast_name, yeast_model, yeast_df, yeast_features, yeast_dims):
    x = get_yeast_feature_vector(yeast_name, yeast_df, yeast_features)
    y_pred = yeast_model.predict(x)[0]  # typically 0/1 flags for traits
    return dict(zip(yeast_dims, y_pred))


# ---- SUMMARY ----
def summarize_beer(
    hop_bill_dict,
    malt_selections,
    yeast_name,
    hop_model, hop_features, hop_dims,
    malt_model, malt_df, malt_features, malt_dims,
    yeast_model, yeast_df, yeast_features, yeast_dims,
):
    hop_out, hop_matched, hop_x = predict_hop_profile(
        hop_bill_dict, hop_model, hop_features, hop_dims
    )
    malt_out  = predict_malt_profile_from_blend(
        malt_selections, malt_model, malt_df, malt_features, malt_dims
    )
    yeast_out = predict_yeast_profile(
        yeast_name, yeast_model, yeast_df, yeast_features, yeast_dims
    )

    # Top hop notes (sorted desc)
    hop_sorted = sorted(hop_out.items(), key=lambda kv: kv[1], reverse=True)
    top_hops   = [f"{k} ({round(v, 2)})" for k, v in hop_sorted[:2]]

    # Malt traits that fired
    malt_active  = [k for k,v in malt_out.items() if v == 1]

    # Yeast traits that fired
    yeast_active = [k for k,v in yeast_out.items() if v == 1]

    # Style guess heuristic
    style_guess = "Experimental / Hybrid"

    if ("clean_neutral" in yeast_out and yeast_out["clean_neutral"] == 1
        and "dry_finish" in yeast_out and yeast_out["dry_finish"] == 1):
        if any("citrus" in n[0] or "resin" in n[0] for n in hop_sorted[:2]):
            style_guess = "West Coast IPA / Modern IPA"
        else:
            style_guess = "Clean, dry ale"

    if ("fruity_esters" in yeast_out and yeast_out["fruity_esters"] == 1) and \
       ("tropical" in hop_out and hop_out["tropical"] > 0.6):
        style_guess = "Hazy / NEIPA leaning"

    if ("phenolic_spicy" in yeast_out and yeast_out["phenolic_spicy"] == 1):
        style_guess = "Belgian / Saison leaning"

    if ("caramel" in malt_out and malt_out["caramel"] == 1):
        style_guess = "English / Malt-forward Ale"

    return {
        "hop_out": hop_out,
        "hop_top_notes": top_hops,
        "malt_traits": malt_active,
        "yeast_traits": yeast_active,
        "style_guess": style_guess,
        "hop_matched": hop_matched,
        "hop_x": hop_x,
    }


# -----------------------------------------------------------------------------
# RADAR PLOT
# -----------------------------------------------------------------------------
def plot_hop_radar(hop_profile, title="Hop Aroma Radar"):
    """
    Radar plot where ticks and labels match exactly.
    hop_profile is a dict like:
      {"tropical":0.8,"citrus":0.7,...}
    Already normalized to [0,1] in predict_hop_profile()
    """

    if not hop_profile:
        hop_profile = {
            "tropical": 0.0,
            "citrus": 0.0,
            "fruity": 0.0,
            "resinous": 0.0,
            "floral": 0.0,
            "herbal": 0.0,
        }

    labels = list(hop_profile.keys())
    values = list(hop_profile.values())
    values_arr = np.array(values, dtype=float)

    # close polygon for fill
    closed_values = np.concatenate([values_arr, values_arr[:1]])

    n = len(labels)
    base_angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    closed_angles = np.concatenate([base_angles, base_angles[:1]])

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    ax.plot(closed_angles, closed_values, linewidth=2)
    ax.fill(closed_angles, closed_values, alpha=0.25)

    # numeric label at each vertex
    for ang, val in zip(base_angles, values_arr):
        ax.text(
            ang,
            val,
            f"{val:.4f}",
            color="black",
            ha="center",
            va="center",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#1f77b4", lw=1)
        )

    # category ticks (unique angles, unique labels)
    ax.set_xticks(base_angles)
    ax.set_xticklabels(labels, fontsize=12)

    # radial grid styling
    ax.set_rlabel_position(0)
    ax.yaxis.grid(color="gray", linestyle="--", alpha=0.4)
    ax.xaxis.grid(color="gray", linestyle="--", alpha=0.4)

    # force radial range to 0-1 because we normalized to [0,1]
    ax.set_ylim(0, 1.0)

    ax.set_title(title, fontsize=28, fontweight="bold", pad=20)

    fig.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# SIDEBAR UI
# -----------------------------------------------------------------------------
# We keep hop choices in session_state so Streamlit Cloud reloads don't crash.
if "hop1_name" not in st.session_state:
    st.session_state.hop1_name = ALL_MODEL_HOPS[0] if len(ALL_MODEL_HOPS) > 0 else ""
if "hop2_name" not in st.session_state:
    st.session_state.hop2_name = ALL_MODEL_HOPS[1] if len(ALL_MODEL_HOPS) > 1 else st.session_state.hop1_name
if "hop3_name" not in st.session_state:
    st.session_state.hop3_name = ALL_MODEL_HOPS[2] if len(ALL_MODEL_HOPS) > 2 else st.session_state.hop1_name
if "hop4_name" not in st.session_state:
    st.session_state.hop4_name = ALL_MODEL_HOPS[3] if len(ALL_MODEL_HOPS) > 3 else st.session_state.hop1_name

st.sidebar.header("🌿 Hop Bill")

# Hop 1
st.session_state.hop1_name = st.sidebar.selectbox(
    "Hop 1", ALL_MODEL_HOPS, index=ALL_MODEL_HOPS.index(st.session_state.hop1_name) if st.session_state.hop1_name in ALL_MODEL_HOPS else 0,
    key="hop1_name"
)
hop1_amt  = st.sidebar.number_input(f"{st.session_state.hop1_name} (g)", min_value=0.0, max_value=200.0, value=40.0, step=5.0, key="hop1_amt")

# Hop 2
st.session_state.hop2_name = st.sidebar.selectbox(
    "Hop 2", ALL_MODEL_HOPS, index=ALL_MODEL_HOPS.index(st.session_state.hop2_name) if st.session_state.hop2_name in ALL_MODEL_HOPS else 0,
    key="hop2_name"
)
hop2_amt  = st.sidebar.number_input(f"{st.session_state.hop2_name} (g)", min_value=0.0, max_value=200.0, value=40.0, step=5.0, key="hop2_amt")

# Hop 3
st.session_state.hop3_name = st.sidebar.selectbox(
    "Hop 3", ALL_MODEL_HOPS, index=ALL_MODEL_HOPS.index(st.session_state.hop3_name) if st.session_state.hop3_name in ALL_MODEL_HOPS else 0,
    key="hop3_name"
)
hop3_amt  = st.sidebar.number_input(f"{st.session_state.hop3_name} (g)", min_value=0.0, max_value=200.0, value=0.0, step=5.0, key="hop3_amt")

# Hop 4
st.session_state.hop4_name = st.sidebar.selectbox(
    "Hop 4", ALL_MODEL_HOPS, index=ALL_MODEL_HOPS.index(st.session_state.hop4_name) if st.session_state.hop4_name in ALL_MODEL_HOPS else 0,
    key="hop4_name"
)
hop4_amt  = st.sidebar.number_input(f"{st.session_state.hop4_name} (g)", min_value=0.0, max_value=200.0, value=0.0, step=5.0, key="hop4_amt")

hop_bill = {
    st.session_state.hop1_name: hop1_amt,
    st.session_state.hop2_name: hop2_amt,
    st.session_state.hop3_name: hop3_amt,
    st.session_state.hop4_name: hop4_amt,
}

# Malt Bill
st.sidebar.header("🌾 Malt Bill")
malt_options = sorted(malt_df["PRODUCT NAME"].unique().tolist())

malt1_name = st.sidebar.selectbox("Malt 1", malt_options, key="malt1_name")
malt1_pct  = st.sidebar.number_input("Malt 1 %", min_value=0.0, max_value=100.0,
                                     value=70.0, step=1.0, key="malt1_pct")

malt2_name = st.sidebar.selectbox("Malt 2", malt_options, key="malt2_name")
malt2_pct  = st.sidebar.number_input("Malt 2 %", min_value=0.0, max_value=100.0,
                                     value=20.0, step=1.0, key="malt2_pct")

malt3_name = st.sidebar.selectbox("Malt 3", malt_options, key="malt3_name")
malt3_pct  = st.sidebar.number_input("Malt 3 %", min_value=0.0, max_value=100.0,
                                     value=10.0, step=1.0, key="malt3_pct")

malt_selections = [
    {"name": malt1_name, "pct": malt1_pct},
    {"name": malt2_name, "pct": malt2_pct},
    {"name": malt3_name, "pct": malt3_pct},
]

# Yeast
st.sidebar.header("🧬 Yeast")
yeast_options = sorted(yeast_df["Name"].dropna().unique().tolist())
chosen_yeast  = st.sidebar.selectbox("Yeast Strain", yeast_options)

run_button = st.sidebar.button("Predict Flavor 🧪", type="primary")

with st.sidebar.expander("💡 Model trained on hops:"):
    st.caption(", ".join(sorted(ALL_MODEL_HOPS)))


# -----------------------------------------------------------------------------
# MAIN APP LAYOUT
# -----------------------------------------------------------------------------
st.title("🍺 Beer Recipe Digital Twin")
st.caption("Predict hop aroma, malt character, and fermentation profile using trained ML models.")

if run_button:
    summary = summarize_beer(
        hop_bill,
        malt_selections,
        chosen_yeast,
        hop_model, hop_features, hop_dims,
        malt_model, malt_df, malt_features, malt_dims,
        yeast_model, yeast_df, yeast_features, yeast_dims
    )

    hop_profile   = summary["hop_out"]          # dict aroma_dim -> normalized intensity
    hop_notes     = summary["hop_top_notes"]    # list of strings
    malt_traits   = summary["malt_traits"]      # list of malt descriptors triggered
    yeast_traits  = summary["yeast_traits"]     # list of yeast descriptors triggered
    style_guess   = summary["style_guess"]
    hop_matched   = summary["hop_matched"]      # hops actually mapped into model
    hop_x         = summary["hop_x"]            # numeric model input row (1 x n_features)

    # LEFT big radar, RIGHT text summary
    col_left, col_right = st.columns([2, 1], vertical_alignment="top")

    with col_left:
        fig = plot_hop_radar(hop_profile, title="Hop Aroma Radar")
        st.pyplot(fig, use_container_width=True)

    with col_right:
        st.markdown("### Top hop notes:")
        if hop_notes:
            for n in hop_notes:
                st.write(f"- {n}")
        else:
            st.write("_No dominant hop note_")

        st.markdown("### Malt character:")
        if malt_traits:
            st.write(", ".join(malt_traits))
        else:
            st.write("None")

        st.markdown("### Yeast character:")
        if yeast_traits:
            st.write(", ".join(yeast_traits))
        else:
            st.write("None")

        st.markdown("### Style direction:")
        st.write(f"🧭 {style_guess}")

        st.markdown("### Hops used by the model:")
        if hop_matched:
            st.write(", ".join(hop_matched))
        else:
            st.write("_No recognizable hops (model saw zeros)_")

    # DEBUG EXPANDER
    with st.expander("🧪 Debug: hop model input (what the model actually sees)"):
        debug_df = pd.DataFrame(hop_x, columns=hop_features)
        st.dataframe(debug_df)

else:
    st.info(
        "👈 Build your hop bill (4 hops with grams), malt bill (3 malts with %), "
        "choose yeast, then click **Predict Flavor 🧪**.\n\n"
        "Tip: use hops from the '💡 Model trained on hops:' list so the aroma radar isn't flat."
    )

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
# CACHE: LOAD MODELS AND DATA
# -----------------------------------------------------------------------------
@st.cache_resource
def load_models_and_data():
    """
    Load all trained models + reference data once, cache them for the session.
    Assumes these files live in the repo root:
      - hop_aroma_model.joblib
      - malt_sensory_model.joblib
      - yeast_sensory_model.joblib
      - clean_malt_df.pkl
      - clean_yeast_df.pkl
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


# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------

# ---- HOPS ----
def get_all_hop_names(hop_features):
    """
    Convert ['hop_Citra','hop_Mosaic', ...] -> ['Citra','Mosaic', ...]
    """
    return [c.replace("hop_", "") for c in hop_features]


def build_hop_features(hop_bill_dict, hop_features):
    """
    Convert user's hop bill dict into the model input vector.

    hop_bill_dict = {"Citra": 40, "Mosaic": 20, ...}  (grams or whatever)
    The model is usually trained on RELATIVE proportions, not raw grams.

    Returns:
      x : 1 x n array aligned with hop_features
      matched_hops : list of hop names actually recognized (non-zero in row)
    """

    # Keep only >0 values
    positive = {k: float(v) for k, v in hop_bill_dict.items()
                if v is not None and float(v) > 0}

    total = sum(positive.values())
    if total <= 0:
        total = 1.0  # avoid divide-by-zero; we'll effectively return zeros

    proportions = {k: v / total for k, v in positive.items()}

    row_vals = []
    matched_hops = []
    for col in hop_features:
        hop_name = col.replace("hop_", "")
        val = proportions.get(hop_name, 0.0)
        if val > 0:
            matched_hops.append(hop_name)
        row_vals.append(val)

    x = np.array(row_vals).reshape(1, -1)
    return x, matched_hops


def predict_hop_profile(hop_bill_dict, hop_model, hop_features, hop_dims):
    """
    Predict aroma intensities from the hop bill.
    Returns:
      hop_profile_dict: {aroma_dim: value}
      matched_hops: which hops from the user's bill the model actually used
    """
    x, matched_hops = build_hop_features(hop_bill_dict, hop_features)
    y_pred = hop_model.predict(x)[0]  # numeric intensities per hop_dims
    hop_profile_dict = dict(zip(hop_dims, y_pred))
    return hop_profile_dict, matched_hops


def hop_profile_all_zero(hop_profile_dict, eps=1e-6):
    """
    Check if all predicted hop aroma values are ~0.
    """
    if not hop_profile_dict:
        return True
    vals = np.array(list(hop_profile_dict.values()), dtype=float)
    return np.all(np.abs(vals) < eps)


# ---- MALTS ----
def get_weighted_malt_vector(malt_selections, malt_df, malt_features):
    """
    malt_selections looks like:
    [
      {"name": "Maris Otter", "pct": 70},
      {"name": "Crystal 60L", "pct": 20},
      {"name": "Flaked Oats", "pct": 10}
    ]
    We'll build a weighted blend of numeric malt features based on pct of grist.
    """
    blend_vec = np.zeros(len(malt_features), dtype=float)

    for item in malt_selections:
        malt_name = item["name"]
        pct       = float(item["pct"])

        row = malt_df[malt_df["PRODUCT NAME"] == malt_name].head(1)
        if row.empty:
            continue

        vec = np.array([row.iloc[0][feat] for feat in malt_features],
                       dtype=float)
        blend_vec += vec * (pct / 100.0)

    return blend_vec.reshape(1, -1)


def predict_malt_profile_from_blend(malt_selections,
                                    malt_model,
                                    malt_df,
                                    malt_features,
                                    malt_dims):
    """
    Predict malt trait flags (often binary-ish like 'bready', 'caramel', etc.)
    """
    x = get_weighted_malt_vector(malt_selections, malt_df, malt_features)
    y_pred = malt_model.predict(x)[0]
    return dict(zip(malt_dims, y_pred))


# ---- YEAST ----
def get_yeast_feature_vector(yeast_name, yeast_df, yeast_features):
    """
    Build one-row model input for chosen yeast strain.
    We assume yeast_df has columns referenced in yeast_features.
    """
    row = yeast_df[yeast_df["Name"] == yeast_name].head(1)
    if row.empty:
        # fallback: all zeros
        return np.zeros(len(yeast_features)).reshape(1, -1)

    vec = [row.iloc[0][feat] for feat in yeast_features]
    return np.array(vec, dtype=float).reshape(1, -1)


def predict_yeast_profile(yeast_name,
                          yeast_model,
                          yeast_df,
                          yeast_features,
                          yeast_dims):
    x = get_yeast_feature_vector(yeast_name, yeast_df, yeast_features)
    y_pred = yeast_model.predict(x)[0]  # often 0/1 trait flags
    return dict(zip(yeast_dims, y_pred))


# ---- COMBINE EVERYTHING / SUMMARY ----
def summarize_beer(hop_bill_dict,
                   malt_selections,
                   yeast_name,
                   hop_model, hop_features, hop_dims,
                   malt_model, malt_df, malt_features, malt_dims,
                   yeast_model, yeast_df, yeast_features, yeast_dims):

    hop_out, matched_hops = predict_hop_profile(
        hop_bill_dict, hop_model, hop_features, hop_dims
    )
    malt_out  = predict_malt_profile_from_blend(
        malt_selections, malt_model, malt_df, malt_features, malt_dims
    )
    yeast_out = predict_yeast_profile(
        yeast_name, yeast_model, yeast_df, yeast_features, yeast_dims
    )

    # rank hop notes
    hop_sorted = sorted(hop_out.items(), key=lambda kv: kv[1], reverse=True)
    top_hops   = [f"{k} ({round(v, 2)})" for k, v in hop_sorted[:2]]

    # Malt traits that fired
    malt_active  = [k for k,v in malt_out.items() if v == 1]

    # Yeast traits that fired
    yeast_active = [k for k,v in yeast_out.items() if v == 1]

    # Heuristic style guess
    style_guess = "Experimental / Hybrid"

    if ("clean_neutral" in yeast_out and yeast_out["clean_neutral"] == 1
        and "dry_finish" in yeast_out and yeast_out["dry_finish"] == 1):
        # very crude "west coast" vibe if top hops are resin/citrus
        if any(("citrus" in n[0] or "resin" in n[0])
               for n in hop_sorted[:2]):
            style_guess = "West Coast IPA / Modern IPA"
        else:
            style_guess = "Clean, dry ale"

    if ("fruity_esters" in yeast_out and yeast_out["fruity_esters"] == 1
        and "tropical" in hop_out and hop_out["tropical"] > 0.6):
        style_guess = "Hazy / NEIPA leaning"

    if ("phenolic_spicy" in yeast_out and yeast_out["phenolic_spicy"] == 1):
        style_guess = "Belgian / Saison leaning"

    if ("caramel" in malt_out and malt_out["caramel"] == 1):
        style_guess = "English / Malt-forward Ale"

    return {
        "hop_out": hop_out,
        "matched_hops": matched_hops,
        "hop_top_notes": top_hops,
        "malt_traits": malt_active,
        "yeast_traits": yeast_active,
        "style_guess": style_guess
    }


# -----------------------------------------------------------------------------
# RADAR PLOT (robust, no tick mismatch)
# -----------------------------------------------------------------------------
def plot_hop_radar(hop_profile, title="Hop Aroma Radar"):
    """
    Radar plot where ticks and labels match exactly.
    We don't reuse the first point for labeling (only for drawing the polygon).
    """

    if not hop_profile:
        hop_profile = {
            "tropical": 0.0,
            "citrus": 0.0,
            "fruity": 0.0,
            "resinous": 0.0,
            "floral": 0.0,
            "herbal": 0.0,
            "spicy": 0.0,
            "earthy": 0.0,
        }

    labels = list(hop_profile.keys())
    values = np.array(list(hop_profile.values()), dtype=float)

    n = len(labels)
    # angles for each unique label
    base_angles = np.linspace(0, 2*np.pi, n, endpoint=False)

    # close polygon
    closed_angles = np.concatenate([base_angles, base_angles[:1]])
    closed_values = np.concatenate([values, values[:1]])

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    ax.plot(closed_angles, closed_values, linewidth=2, color="#1f77b4")
    ax.fill(closed_angles, closed_values, alpha=0.25, color="#1f77b4")

    # annotate numeric values at each vertex
    for ang, val in zip(base_angles, values):
        ax.text(
            ang,
            val,
            f"{val:.4f}",
            color="black",
            ha="center",
            va="center",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.2",
                      fc="white",
                      ec="#1f77b4",
                      lw=1),
        )

    # ticks == unique labels
    ax.set_xticks(base_angles)
    ax.set_xticklabels(labels, fontsize=12)

    # nicer radial grid
    ax.set_rlabel_position(0)
    ax.yaxis.grid(color="gray", linestyle="--", alpha=0.4)
    ax.xaxis.grid(color="gray", linestyle="--", alpha=0.4)

    # y-limit: auto but padded
    vmax = max(1.0, values.max() * 1.2 if values.size else 1.0)
    ax.set_ylim(0, vmax)

    ax.set_title(title, fontsize=32, fontweight="bold", pad=20)

    fig.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# SIDEBAR UI
# -----------------------------------------------------------------------------
st.sidebar.header("Hop Bill")

# Show which hops the model even knows about (collapsed by default in main app
# we don't spam the page; user just sees this section in the sidebar).
with st.sidebar.expander("🧬 Model trained on hops:", expanded=False):
    st.write(", ".join(sorted(get_all_hop_names(hop_features))))

all_hops = sorted(get_all_hop_names(hop_features))

# We'll store hop selections in session_state, so demo button can overwrite
def _init_ss(key, default):
    if key not in st.session_state:
        st.session_state[key] = default

_init_ss("hop1_name", all_hops[0] if len(all_hops) > 0 else "")
_init_ss("hop2_name", all_hops[1] if len(all_hops) > 1 else st.session_state["hop1_name"])
_init_ss("hop3_name", all_hops[2] if len(all_hops) > 2 else st.session_state["hop1_name"])
_init_ss("hop4_name", all_hops[3] if len(all_hops) > 3 else st.session_state["hop2_name"])

_init_ss("hop1_amt", 40)
_init_ss("hop2_amt", 40)
_init_ss("hop3_amt", 0)
_init_ss("hop4_amt", 0)

# Hop selectors / sliders
st.session_state.hop1_name = st.sidebar.selectbox("Hop 1", all_hops, key="hop1_name")
st.session_state.hop1_amt  = st.sidebar.slider(f"{st.session_state.hop1_name} (g)", 0, 120, st.session_state.hop1_amt, 5, key="hop1_amt")

st.session_state.hop2_name = st.sidebar.selectbox("Hop 2", all_hops, key="hop2_name")
st.session_state.hop2_amt  = st.sidebar.slider(f"{st.session_state.hop2_name} (g)", 0, 120, st.session_state.hop2_amt, 5, key="hop2_amt")

st.session_state.hop3_name = st.sidebar.selectbox("Hop 3", all_hops, key="hop3_name")
st.session_state.hop3_amt  = st.sidebar.slider(f"{st.session_state.hop3_name} (g)", 0, 120, st.session_state.hop3_amt, 5, key="hop3_amt")

st.session_state.hop4_name = st.sidebar.selectbox("Hop 4", all_hops, key="hop4_name")
st.session_state.hop4_amt  = st.sidebar.slider(f"{st.session_state.hop4_name} (g)", 0, 120, st.session_state.hop4_amt, 5, key="hop4_amt")

hop_bill = {
    st.session_state.hop1_name: st.session_state.hop1_amt,
    st.session_state.hop2_name: st.session_state.hop2_amt,
    st.session_state.hop3_name: st.session_state.hop3_amt,
    st.session_state.hop4_name: st.session_state.hop4_amt,
}

# Malt section
st.sidebar.header("Malt Bill")

malt_options = sorted(malt_df["PRODUCT NAME"].unique().tolist())

_init_ss("malt1_name", malt_options[0] if malt_options else "")
_init_ss("malt2_name", malt_options[1] if len(malt_options) > 1 else st.session_state["malt1_name"])
_init_ss("malt3_name", malt_options[2] if len(malt_options) > 2 else st.session_state["malt1_name"])

_init_ss("malt1_pct", 70.0)
_init_ss("malt2_pct", 20.0)
_init_ss("malt3_pct", 10.0)

st.session_state.malt1_name = st.sidebar.selectbox("Malt 1", malt_options, key="malt1_name")
st.session_state.malt1_pct  = st.sidebar.number_input("Malt 1 %", min_value=0.0, max_value=100.0,
                                                       value=st.session_state.malt1_pct, step=1.0, key="malt1_pct")

st.session_state.malt2_name = st.sidebar.selectbox("Malt 2", malt_options, key="malt2_name")
st.session_state.malt2_pct  = st.sidebar.number_input("Malt 2 %", min_value=0.0, max_value=100.0,
                                                       value=st.session_state.malt2_pct, step=1.0, key="malt2_pct")

st.session_state.malt3_name = st.sidebar.selectbox("Malt 3", malt_options, key="malt3_name")
st.session_state.malt3_pct  = st.sidebar.number_input("Malt 3 %", min_value=0.0, max_value=100.0,
                                                       value=st.session_state.malt3_pct, step=1.0, key="malt3_pct")

malt_selections = [
    {"name": st.session_state.malt1_name, "pct": st.session_state.malt1_pct},
    {"name": st.session_state.malt2_name, "pct": st.session_state.malt2_pct},
    {"name": st.session_state.malt3_name, "pct": st.session_state.malt3_pct},
]

# Yeast section
st.sidebar.header("Yeast")
yeast_options = sorted(yeast_df["Name"].dropna().unique().tolist())

_init_ss("chosen_yeast", yeast_options[0] if yeast_options else "")
st.session_state.chosen_yeast = st.sidebar.selectbox("Yeast Strain", yeast_options, key="chosen_yeast")

# Action buttons
run_button = st.sidebar.button("Predict Flavor 🧪")

demo = st.sidebar.button("Use demo hop bill (Galaxy + Vic Secret)")
if demo:
    # try to fill with known juicy combo if available
    if "Galaxy" in all_hops:
        st.session_state.hop1_name = "Galaxy"
        st.session_state.hop1_amt = 60
    if "Vic Secret" in all_hops:
        st.session_state.hop2_name = "Vic Secret"
        st.session_state.hop2_amt = 40

    # set rest to 0
    st.session_state.hop3_amt = 0
    st.session_state.hop4_amt = 0
    st.experimental_rerun()

# -----------------------------------------------------------------------------
# MAIN APP LAYOUT
# -----------------------------------------------------------------------------

st.markdown(
    "<h1 style='display:flex;align-items:center;gap:0.5rem;'>"
    "🍺 Beer Recipe Digital Twin"
    "</h1>",
    unsafe_allow_html=True
)

st.caption(
    "Predict hop aroma, malt character, and fermentation profile using trained ML models."
)

if run_button:
    # run inference
    summary = summarize_beer(
        hop_bill,
        malt_selections,
        st.session_state.chosen_yeast,
        hop_model, hop_features, hop_dims,
        malt_model, malt_df, malt_features, malt_dims,
        yeast_model, yeast_df, yeast_features, yeast_dims
    )

    hop_profile   = summary["hop_out"]
    matched_hops  = summary["matched_hops"]
    hop_notes     = summary["hop_top_notes"]
    malt_traits   = summary["malt_traits"]
    yeast_traits  = summary["yeast_traits"]
    style_guess   = summary["style_guess"]

    # top-level layout for results
    col_left, col_right = st.columns([2, 1], vertical_alignment="top")

    with col_left:
        # if model basically didn't recognize any hops, warn instead of plotting empty spider
        if (not matched_hops) or hop_profile_all_zero(hop_profile):
            st.warning(
                "⚠️ No recognizable hops in your bill (or total was 0 once normalized), "
                "so aroma prediction is basically flat.\n\n"
                "Pick hops from the list in the sidebar (open the '🧬 Model trained on hops:' "
                "section), then click **Predict Flavor 🧪** again."
            )
        else:
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
        st.write(", ".join(matched_hops) if matched_hops else "_None_")

    # Debug section (optional): what vector did we feed the hop model?
    with st.expander("🧪 Debug: hop model input (what the model actually sees)"):
        x_for_debug, _matched2 = build_hop_features(hop_bill, hop_features)
        debug_df = pd.DataFrame(x_for_debug, columns=hop_features)
        st.dataframe(debug_df, use_container_width=True)

else:
    st.info(
        "👈 Build your hop bill (up to 4 hops), malt bill (3 malts with %), "
        "choose yeast, then click **Predict Flavor 🧪**.\n\n"
        "Tip: if the radar stays empty, try the **Use demo hop bill** button in the sidebar."
    )

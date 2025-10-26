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
    layout="wide"
)

st.title("🍺 Beer Recipe Digital Twin")
st.caption(
    "Predict hop aroma, malt character, and fermentation profile using trained ML models."
)


# -----------------------------------------------------------------------------
# CACHE: LOAD MODELS AND DATA
# -----------------------------------------------------------------------------
@st.cache_resource
def load_models_and_data():
    """
    Load all trained models + reference data once, cache them for the session.
    Expects the following files in the repo root:
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
    hop_dims       = hop_bundle["aroma_dims"]     # e.g. ["tropical","citrus","fruity",...]

    malt_model     = malt_bundle["model"]
    malt_features  = malt_bundle["feature_cols"]  # e.g. malt chemistry columns
    malt_dims      = malt_bundle["flavor_cols"]   # predicted malt traits (binary flags, etc.)

    yeast_model    = yeast_bundle["model"]
    yeast_features = yeast_bundle["feature_cols"] # e.g. Temp_avg_C, etc.
    yeast_dims     = yeast_bundle["flavor_cols"]  # predicted yeast traits (binary flags, etc.)

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
    hop_features are columns like 'hop_Citra', 'hop_Mosaic', ...
    We convert them to ['Citra','Mosaic',...].
    """
    return [c.replace("hop_", "") for c in hop_features]


def build_hop_features(hop_bill_dict, hop_features):
    """
    hop_bill_dict = {"Citra": 40, "Mosaic": 20, ...}
    Return a single row aligned with hop_features order.
    We'll also normalize by total grams so that the model sees relative
    contributions instead of raw mass, if that's how it was trained.
    (We assume the model was trained on normalized weights; if not, remove
     the normalization part.)
    """
    # vector of raw grams in same order as model expects
    raw_values = []
    for col in hop_features:
        hop_name = col.replace("hop_", "")
        raw_values.append(float(hop_bill_dict.get(hop_name, 0.0)))

    raw_values = np.array(raw_values, dtype=float)
    total = raw_values.sum()

    if total > 0:
        norm_values = raw_values / total
    else:
        # keep zero vector if there's no hops or total is 0
        norm_values = raw_values

    x = norm_values.reshape(1, -1)
    return x, total


def predict_hop_profile(hop_bill_dict, hop_model, hop_features, hop_dims):
    """
    Returns:
        hop_out_dict: { dimension_name -> predicted intensity (float) }
        used_hops:    list of hops actually > 0
        total_g:      total grams (for warnings / messaging)
    """
    x, total_g = build_hop_features(hop_bill_dict, hop_features)
    y_pred = hop_model.predict(x)[0]  # numeric intensities

    hop_out_dict = dict(zip(hop_dims, y_pred))

    # hops used
    used_hops = [h for h, amt in hop_bill_dict.items() if amt and amt > 0]

    return hop_out_dict, used_hops, total_g


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


def predict_malt_profile_from_blend(
    malt_selections, malt_model, malt_df, malt_features, malt_dims
):
    """
    Predict malt-driven character (usually binary flags like 'bready', etc.).
    """
    x = get_weighted_malt_vector(malt_selections, malt_df, malt_features)
    y_pred = malt_model.predict(x)[0]  # often 0/1 flags for traits

    return dict(zip(malt_dims, y_pred))


# ---- YEAST ----
def get_yeast_feature_vector(yeast_name, yeast_df, yeast_features):
    """
    Build single-row model input for chosen yeast strain.
    We'll assume the yeast_df has columns that match yeast_features.
    """
    row = yeast_df[yeast_df["Name"] == yeast_name].head(1)
    if row.empty:
        # fallback zeros if not found
        return np.zeros(len(yeast_features)).reshape(1, -1)

    # We assume the features in yeast_features map to columns in yeast_df,
    # e.g. ["Temp_avg_C","Flocculation_num","Attenuation_pct"]
    vec = [row.iloc[0][feat] for feat in yeast_features]
    return np.array(vec, dtype=float).reshape(1, -1)


def predict_yeast_profile(
    yeast_name, yeast_model, yeast_df, yeast_features, yeast_dims
):
    """
    e.g. returns flags for yeast traits like 'fruity_esters', 'clean_neutral', etc.
    """
    x = get_yeast_feature_vector(yeast_name, yeast_df, yeast_features)
    y_pred = yeast_model.predict(x)[0]  # often 0/1 flags for traits
    return dict(zip(yeast_dims, y_pred))


# ---- COMBINE EVERYTHING / SUMMARY ----
def summarize_beer(
    hop_bill_dict,
    malt_selections,
    yeast_name,
    hop_model, hop_features, hop_dims,
    malt_model, malt_df, malt_features, malt_dims,
    yeast_model, yeast_df, yeast_features, yeast_dims,
):
    hop_out, hops_used, total_g = predict_hop_profile(
        hop_bill_dict, hop_model, hop_features, hop_dims
    )
    malt_out  = predict_malt_profile_from_blend(
        malt_selections, malt_model, malt_df, malt_features, malt_dims
    )
    yeast_out = predict_yeast_profile(
        yeast_name, yeast_model, yeast_df, yeast_features, yeast_dims
    )

    # Top hop notes: sort hop_out by intensity desc
    hop_sorted = sorted(hop_out.items(), key=lambda kv: kv[1], reverse=True)
    top_hops   = [f"{k} ({round(v, 2)})" for k, v in hop_sorted[:2]]

    # Malt traits that fired:
    malt_active  = [k for k, v in malt_out.items() if v == 1]

    # Yeast traits that fired:
    yeast_active = [k for k, v in yeast_out.items() if v == 1]

    # Heuristic style guess:
    style_guess = "Experimental / Hybrid"

    if ("clean_neutral" in yeast_out and yeast_out["clean_neutral"] == 1
        and "dry_finish" in yeast_out and yeast_out["dry_finish"] == 1):
        # pseudo-West Coast
        if any(
            ("citrus" in n.lower() or "resin" in n.lower())
            for n in [k for k,_ in hop_sorted[:2]]
        ):
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
        "hops_used": hops_used,
        "total_g": total_g
    }


# -----------------------------------------------------------------------------
# RADAR / SPIDER PLOT
# -----------------------------------------------------------------------------
def plot_hop_radar(hop_profile, title="Hop Aroma Radar"):
    """
    Make a multi-axis spider/radar chart with dashed grid rings
    and equal angular spacing.

    hop_profile is dict like:
      {"tropical":0.8,"citrus":0.5,"fruity":0.4,"resinous":0.3,...}

    We will:
    - Take keys as labels
    - Take values as the radius at each spoke
    - Close the polygon
    - Show dashed circular gridlines for the "web" look
    """

    # Guard: if empty (shouldn't happen but let's be safe)
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
    values = list(hop_profile.values())

    # close polygon
    values_closed = values + [values[0]]

    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles_closed = angles + [angles[0]]

    fig, ax = plt.subplots(
        figsize=(6, 6),
        subplot_kw=dict(polar=True)
    )

    # Plot polygon + fill
    ax.plot(
        angles_closed,
        values_closed,
        color="#1f77b4",
        linewidth=2
    )
    ax.fill(
        angles_closed,
        values_closed,
        color="#1f77b4",
        alpha=0.15
    )

    # Put numeric label at the center-ish average of the ring near each spoke?
    # Simple version: label each vertex with its numeric value
    for ang, val in zip(angles, values):
        ax.text(
            ang,
            val,
            f"{val:.2f}",
            color="black",
            ha="center",
            va="center",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#1f77b4", lw=1)
        )

    # Set the spoke labels (category names) around the outside
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=12)

    # Make radial grid dashed for spider-web look
    ax.yaxis.grid(True, linestyle="--", color="gray", alpha=0.6)
    ax.xaxis.grid(True, linestyle="--", color="gray", alpha=0.6)

    # If you want to control radial range, you can uncomment:
    # max_val = max(1.0, max(values) * 1.2 if values else 1.0)
    # ax.set_ylim(0, max_val)

    ax.set_title(
        title,
        fontsize=20,
        fontweight="bold",
        pad=20
    )

    fig.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# SIDEBAR UI
# -----------------------------------------------------------------------------

st.sidebar.header("Hop Bill")
all_hops_sorted = sorted(get_all_hop_names(hop_features))

# We'll store UI state in session_state to avoid Streamlit "resets".
# Each hop slot: (name dropdown, grams input)
if "hop1_name" not in st.session_state:
    st.session_state.hop1_name = all_hops_sorted[0] if all_hops_sorted else ""
if "hop2_name" not in st.session_state:
    st.session_state.hop2_name = all_hops_sorted[1] if len(all_hops_sorted) > 1 else st.session_state.hop1_name
if "hop3_name" not in st.session_state:
    st.session_state.hop3_name = all_hops_sorted[2] if len(all_hops_sorted) > 2 else st.session_state.hop1_name
if "hop4_name" not in st.session_state:
    st.session_state.hop4_name = all_hops_sorted[3] if len(all_hops_sorted) > 3 else st.session_state.hop1_name

hop1_name = st.sidebar.selectbox("Hop 1", all_hops_sorted, key="hop1_name")
hop1_amt  = st.sidebar.number_input(f"{hop1_name} (g)", min_value=0.0, value=40.0, step=5.0)

hop2_name = st.sidebar.selectbox("Hop 2", all_hops_sorted, key="hop2_name")
hop2_amt  = st.sidebar.number_input(f"{hop2_name} (g)", min_value=0.0, value=20.0, step=5.0)

hop3_name = st.sidebar.selectbox("Hop 3", all_hops_sorted, key="hop3_name")
hop3_amt  = st.sidebar.number_input(f"{hop3_name} (g)", min_value=0.0, value=0.0, step=5.0)

hop4_name = st.sidebar.selectbox("Hop 4", all_hops_sorted, key="hop4_name")
hop4_amt  = st.sidebar.number_input(f"{hop4_name} (g)", min_value=0.0, value=0.0, step=5.0)

hop_bill = {
    hop1_name: hop1_amt,
    hop2_name: hop2_amt,
    hop3_name: hop3_amt,
    hop4_name: hop4_amt,
}

st.sidebar.header("🌾 Malt Bill")
malt_options = sorted(malt_df["PRODUCT NAME"].dropna().unique().tolist())

if "malt1_name" not in st.session_state:
    st.session_state.malt1_name = malt_options[0] if malt_options else ""
if "malt2_name" not in st.session_state:
    st.session_state.malt2_name = malt_options[1] if len(malt_options) > 1 else st.session_state.malt1_name
if "malt3_name" not in st.session_state:
    st.session_state.malt3_name = malt_options[2] if len(malt_options) > 2 else st.session_state.malt1_name

malt1_name = st.sidebar.selectbox("Malt 1", malt_options, key="malt1_name")
malt1_pct  = st.sidebar.number_input("Malt 1 %", min_value=0.0, max_value=100.0,
                                     value=70.0, step=1.0)
malt2_name = st.sidebar.selectbox("Malt 2", malt_options, key="malt2_name")
malt2_pct  = st.sidebar.number_input("Malt 2 %", min_value=0.0, max_value=100.0,
                                     value=20.0, step=1.0)
malt3_name = st.sidebar.selectbox("Malt 3", malt_options, key="malt3_name")
malt3_pct  = st.sidebar.number_input("Malt 3 %", min_value=0.0, max_value=100.0,
                                     value=10.0, step=1.0)

malt_selections = [
    {"name": malt1_name, "pct": malt1_pct},
    {"name": malt2_name, "pct": malt2_pct},
    {"name": malt3_name, "pct": malt3_pct},
]

st.sidebar.header("🧫 Yeast")
yeast_options = sorted(yeast_df["Name"].dropna().unique().tolist())

if "chosen_yeast" not in st.session_state:
    st.session_state.chosen_yeast = yeast_options[0] if yeast_options else ""

chosen_yeast  = st.sidebar.selectbox("Yeast Strain", yeast_options, key="chosen_yeast")

run_button = st.sidebar.button("Predict Flavor 🧪")


# -----------------------------------------------------------------------------
# MAIN APP LAYOUT
# -----------------------------------------------------------------------------

if run_button:
    summary = summarize_beer(
        hop_bill,
        malt_selections,
        chosen_yeast,
        hop_model, hop_features, hop_dims,
        malt_model, malt_df, malt_features, malt_dims,
        yeast_model, yeast_df, yeast_features, yeast_dims
    )

    hop_profile   = summary["hop_out"]
    hop_notes     = summary["hop_top_notes"]
    malt_traits   = summary["malt_traits"]
    yeast_traits  = summary["yeast_traits"]
    style_guess   = summary["style_guess"]
    hops_used     = summary["hops_used"]
    total_g       = summary["total_g"]

    # layout columns for radar + text summary
    col_left, col_right = st.columns([2, 1], vertical_alignment="top")

    with col_left:
        # Warning banner if you gave 0g or everything normalized to 0
        if total_g == 0:
            st.warning(
                "No recognizable hops in your bill (or total was 0 once normalized), "
                "so aroma prediction is basically flat.\n\n"
                "Pick hops from the list in the sidebar (open the "
                "'🌾 Malt Bill' / 'Hop Bill' section), then click **Predict Flavor 🧪** again."
            )

        st.subheader("Hop Aroma Radar")
        fig = plot_hop_radar(hop_profile, title="Hop Aroma Radar")
        st.pyplot(fig)

    with col_right:
        st.markdown("#### Top hop notes:")
        if hop_notes:
            for n in hop_notes:
                st.write(f"- {n}")
        else:
            st.write("_No dominant hop note_")

        st.markdown("#### Malt character:")
        if malt_traits:
            st.write(", ".join(malt_traits))
        else:
            st.write("None")

        st.markdown("#### Yeast character:")
        if yeast_traits:
            st.write(", ".join(yeast_traits))
        else:
            st.write("None")

        st.markdown("#### Style direction:")
        st.write(f"🧭 {style_guess}")

        st.markdown("#### Hops used by the model:")
        st.write(", ".join(hops_used))

else:
    # Initial instructions (shown before the user clicks Predict)
    st.info(
        "👉 Build your hop bill (up to 4 hops, with nonzero grams), "
        "set malt bill (% grist), choose yeast, then click "
        "**Predict Flavor 🧪** in the sidebar."
    )

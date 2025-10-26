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


# -----------------------------------------------------------------------------
# UTILITIES (SESSION SAFE GET/SET)
# -----------------------------------------------------------------------------
def _safe_get_session(key, default=None):
    """Return st.session_state[key] if it exists, else default."""
    return st.session_state[key] if key in st.session_state else default

def _safe_set_session(key, value):
    """Write a value into session_state (simple helper)."""
    st.session_state[key] = value


# -----------------------------------------------------------------------------
# CACHE: LOAD MODELS + DATA
# -----------------------------------------------------------------------------
@st.cache_resource
def load_models_and_data():
    """
    Load all trained models + reference data once, cache them.
    We expect these files in the same repo directory:
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
    hop_features   = hop_bundle["feature_cols"]    # e.g. ["hop_Citra", "hop_Mosaic", ...]
    hop_dims       = hop_bundle["aroma_dims"]      # e.g. ["tropical","citrus","floral","resinous","fruity","earthy"]

    malt_model     = malt_bundle["model"]
    malt_features  = malt_bundle["feature_cols"]   # numeric malt chemistry columns
    malt_dims      = malt_bundle["flavor_cols"]    # predicted malt traits, e.g. ["bready","caramel","color_intensity",...]

    yeast_model    = yeast_bundle["model"]
    yeast_features = yeast_bundle["feature_cols"]  # e.g. ["Temp_avg_C","Flocculation_num","Attenuation_pct"]
    yeast_dims     = yeast_bundle["flavor_cols"]   # predicted yeast traits, e.g. ["fruity_esters","clean_neutral","phenolic_spicy",..]

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
def get_all_hop_names(hop_features):
    """
    hop_features are like ["hop_Citra", "hop_Mosaic", ...].
    We convert to ["Citra","Mosaic",...].
    """
    clean_names = []
    for col in hop_features:
        if col.startswith("hop_"):
            clean_names.append(col.replace("hop_", "", 1))
        else:
            clean_names.append(col)
    return clean_names


def build_hop_features(hop_bill_dict, hop_features):
    """
    hop_bill_dict might look like:
      {"Citra": 40.0, "Mosaic": 20.0, "Simcoe": 0.0, "Centennial": 0.0}
    We:
      1) build [40,20,0,0,...] aligned to hop_features
      2) normalize so sum = 1 (if total > 0)
    Return shape (1, n_features).
    """
    row = []
    total = sum(hop_bill_dict.values())
    for col in hop_features:
        hop_name = col.replace("hop_", "", 1)
        grams    = hop_bill_dict.get(hop_name, 0.0)
        if total > 0:
            row.append(grams / total)
        else:
            row.append(0.0)
    x = np.array(row).reshape(1, -1)
    return x, total


def predict_hop_profile(hop_bill_dict, hop_model, hop_features, hop_dims):
    """
    Return dict of {aroma_dim -> predicted_value}, e.g.
    {"tropical":0.8,"citrus":0.7,"resinous":0.2,...}
    Also return total_g for warning logic.
    """
    X, total_g = build_hop_features(hop_bill_dict, hop_features)
    y_pred = hop_model.predict(X)[0]  # 1D array
    hop_out = dict(zip(hop_dims, y_pred))
    return hop_out, total_g


def get_weighted_malt_vector(malt_selections, malt_df, malt_features):
    """
    malt_selections is a list of dicts:
      [
        {"name": "Maris Otter", "pct": 70},
        {"name": "Crystal 60L", "pct": 20},
        {"name": "Flaked Oats", "pct": 10}
      ]

    We'll make a weighted average for each numeric malt feature.
    """
    blend_vec = np.zeros(len(malt_features), dtype=float)

    for item in malt_selections:
        malt_name = item["name"]
        pct       = float(item["pct"])  # e.g. 70.0
        row = malt_df[malt_df["PRODUCT NAME"] == malt_name].head(1)
        if row.empty:
            continue
        vec = np.array([row.iloc[0][feat] for feat in malt_features], dtype=float)
        blend_vec += vec * (pct / 100.0)

    return blend_vec.reshape(1, -1)


def predict_malt_profile_from_blend(malt_selections, malt_model, malt_df, malt_features, malt_dims):
    """
    Returns dict {trait -> 0/1 or numeric} from malt model.
    """
    X = get_weighted_malt_vector(malt_selections, malt_df, malt_features)
    y_pred = malt_model.predict(X)[0]
    return dict(zip(malt_dims, y_pred))


def get_yeast_feature_vector(yeast_name, yeast_df, yeast_features):
    """
    Build a single-row feature vector for the chosen yeast.
    We'll specifically pull the columns our yeast model expects.
    """
    row = yeast_df[yeast_df["Name"] == yeast_name].head(1)
    if row.empty:
        # fallback zero vector (same length as yeast_features)
        return np.zeros(len(yeast_features)).reshape(1, -1)

    # We know from training that yeast_features are in a stable order like:
    # ["Temp_avg_C", "Flocculation_num", "Attenuation_pct"]
    vals = []
    for feat in yeast_features:
        vals.append(row.iloc[0].get(feat, 0.0))
    return np.array(vals).reshape(1, -1)


def predict_yeast_profile(yeast_name, yeast_model, yeast_df, yeast_features, yeast_dims):
    """
    Returns dict {trait -> 0/1 or numeric} from yeast model.
    """
    X = get_yeast_feature_vector(yeast_name, yeast_df, yeast_features)
    y_pred = yeast_model.predict(X)[0]
    return dict(zip(yeast_dims, y_pred))


def summarize_beer(
    hop_bill_dict,
    malt_selections,
    yeast_name,
    hop_model, hop_features, hop_dims,
    malt_model, malt_df, malt_features, malt_dims,
    yeast_model, yeast_df, yeast_features, yeast_dims
):
    """
    Runs all 3 models and creates a friendly summary bundle.
    """

    hop_out, hop_total_g = predict_hop_profile(
        hop_bill_dict, hop_model, hop_features, hop_dims
    )
    malt_out = predict_malt_profile_from_blend(
        malt_selections, malt_model, malt_df, malt_features, malt_dims
    )
    yeast_out = predict_yeast_profile(
        yeast_name, yeast_model, yeast_df, yeast_features, yeast_dims
    )

    # --- Build little descriptive pieces ---

    # 1) Top hop notes (sort by predicted intensity)
    hop_sorted = sorted(hop_out.items(), key=lambda kv: kv[1], reverse=True)
    top_hops   = [f"{k} ({round(v, 2)})" for k, v in hop_sorted[:2]]

    # 2) Malt traits that fired
    active_malt_traits = [k for k, v in malt_out.items() if v == 1 or (isinstance(v,(int,float)) and v>0)]
    # If you have a trait like "color_intensity" that's numeric, you might want it even if it's not ==1.
    # We'll keep it simple but let's make sure we don't lose obvious numeric features:
    if "color_intensity" in malt_out:
        if "color_intensity" not in active_malt_traits:
            active_malt_traits.append("color_intensity")

    # 3) Yeast traits that fired
    active_yeast_traits = [k for k, v in yeast_out.items() if v == 1 or (isinstance(v,(int,float)) and v>0)]
    if "clean_neutral" in yeast_out and "clean_neutral" not in active_yeast_traits and yeast_out["clean_neutral"]==1:
        active_yeast_traits.append("clean_neutral")

    # 4) Style guess heuristic
    style_guess = "Experimental / Hybrid"
    if ("clean_neutral" in yeast_out and yeast_out["clean_neutral"] == 1
        and "fruity_esters" in yeast_out and yeast_out["fruity_esters"] == 1):
        # Clean but fruity yeast can suggest modern ale
        style_guess = "Modern Ale / IPA-ish"

    if ("phenolic_spicy" in yeast_out and yeast_out["phenolic_spicy"] == 1):
        style_guess = "Belgian / Saison leaning"

    if ("caramel" in malt_out and malt_out["caramel"] == 1):
        style_guess = "English / Malt-forward Ale"

    # store results
    return {
        "hop_out": hop_out,
        "hop_total_g": hop_total_g,
        "hop_top_notes": top_hops,
        "malt_traits": active_malt_traits,
        "yeast_traits": active_yeast_traits,
        "style_guess": style_guess
    }


# -----------------------------------------------------------------------------
# VIS: RADAR PLOT
# -----------------------------------------------------------------------------
def plot_hop_radar(hop_profile, title="Hop Aroma Radar"):
    """
    Draw radar chart for hop aroma intensities.
    hop_profile is dict: {dimension: value}

    We'll:
      - create angles for each label
      - plot closed polygon
      - show each numeric value in a callout box
    """

    if not hop_profile:
        return None

    labels = list(hop_profile.keys())
    values = np.array(list(hop_profile.values()), dtype=float)

    n = len(labels)
    if n == 0:
        return None

    # angles for each unique label
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    # closed polygon angles/values (repeat first at end)
    angles_closed = np.concatenate([angles, angles[:1]])
    values_closed = np.concatenate([values, values[:1]])

    fig, ax = plt.subplots(
        figsize=(8, 8),
        subplot_kw=dict(polar=True)
    )

    ax.plot(angles_closed, values_closed, color="#1f77b4", linewidth=2)
    ax.fill(angles_closed, values_closed, color="#1f77b4", alpha=0.25)

    # label each vertex with numeric value
    for ang, val, lab in zip(angles, values, labels):
        ax.text(
            ang,
            val,
            f"{val:.4f}",
            ha="center",
            va="center",
            fontsize=10,
            bbox=dict(
                boxstyle="round,pad=0.3",
                fc="white",
                ec="#1f77b4",
                lw=1,
            ),
        )

    # category labels
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=13)

    # radial grid styling
    ax.set_rlabel_position(0)
    ax.yaxis.grid(color="gray", linestyle="--", alpha=0.4)
    ax.xaxis.grid(color="gray", linestyle="--", alpha=0.4)

    ax.set_title(title, fontsize=32, fontweight="bold", pad=20)

    fig.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# SIDEBAR UI
# -----------------------------------------------------------------------------
all_hops_sorted = sorted(get_all_hop_names(hop_features))

st.sidebar.markdown("## 🌿 Hop Bill")

# --- Hop 1 ---
hop1_name = st.sidebar.selectbox(
    "Hop 1",
    all_hops_sorted,
    key="hop1_name",
    index=0 if _safe_get_session("hop1_name") is None else all_hops_sorted.index(_safe_get_session("hop1_name")) if _safe_get_session("hop1_name") in all_hops_sorted else 0
)
hop1_amt = st.sidebar.number_input(
    f"{hop1_name} (g)",
    min_value=0.0,
    step=1.0,
    value=_safe_get_session("hop1_amt", 40.0),
    key="hop1_amt_val"
)

# --- Hop 2 ---
hop2_name = st.sidebar.selectbox(
    "Hop 2",
    all_hops_sorted,
    key="hop2_name",
    index=1 if _safe_get_session("hop2_name") is None else all_hops_sorted.index(_safe_get_session("hop2_name")) if _safe_get_session("hop2_name") in all_hops_sorted else 1
)
hop2_amt = st.sidebar.number_input(
    f"{hop2_name} (g)",
    min_value=0.0,
    step=1.0,
    value=_safe_get_session("hop2_amt", 40.0),
    key="hop2_amt_val"
)

# --- Hop 3 ---
hop3_name = st.sidebar.selectbox(
    "Hop 3",
    all_hops_sorted,
    key="hop3_name",
    index=2 if _safe_get_session("hop3_name") is None else all_hops_sorted.index(_safe_get_session("hop3_name")) if _safe_get_session("hop3_name") in all_hops_sorted else 2
)
hop3_amt = st.sidebar.number_input(
    f"{hop3_name} (g)",
    min_value=0.0,
    step=1.0,
    value=_safe_get_session("hop3_amt", 0.0),
    key="hop3_amt_val"
)

# --- Hop 4 ---
hop4_name = st.sidebar.selectbox(
    "Hop 4",
    all_hops_sorted,
    key="hop4_name",
    index=3 if _safe_get_session("hop4_name") is None else all_hops_sorted.index(_safe_get_session("hop4_name")) if _safe_get_session("hop4_name") in all_hops_sorted else 3
)
hop4_amt = st.sidebar.number_input(
    f"{hop4_name} (g)",
    min_value=0.0,
    step=1.0,
    value=_safe_get_session("hop4_amt", 0.0),
    key="hop4_amt_val"
)

# Immediately sync those updated numeric values back into session_state so they "stick"
_safe_set_session("hop1_amt", hop1_amt)
_safe_set_session("hop2_amt", hop2_amt)
_safe_set_session("hop3_amt", hop3_amt)
_safe_set_session("hop4_amt", hop4_amt)

# (Also keep the chosen hop names, in case Streamlit re-runs)
_safe_set_session("hop1_name", hop1_name)
_safe_set_session("hop2_name", hop2_name)
_safe_set_session("hop3_name", hop3_name)
_safe_set_session("hop4_name", hop4_name)

# Build the hop bill dict for the model:
hop_bill = {
    hop1_name: hop1_amt,
    hop2_name: hop2_amt,
    hop3_name: hop3_amt,
    hop4_name: hop4_amt,
}


# --- Malt Bill ---
st.sidebar.markdown("## 🌽 Malt Bill")

malt_options = sorted(malt_df["PRODUCT NAME"].unique().tolist())

malt1_name = st.sidebar.selectbox(
    "Malt 1",
    malt_options,
    key="malt1_name",
    index=0 if _safe_get_session("malt1_name") is None else malt_options.index(_safe_get_session("malt1_name")) if _safe_get_session("malt1_name") in malt_options else 0
)
malt1_pct = st.sidebar.number_input(
    "Malt 1 %",
    min_value=0.0,
    max_value=100.0,
    step=1.0,
    value=_safe_get_session("malt1_pct", 70.0),
    key="malt1_pct_val"
)

malt2_name = st.sidebar.selectbox(
    "Malt 2",
    malt_options,
    key="malt2_name",
    index=1 if _safe_get_session("malt2_name") is None else malt_options.index(_safe_get_session("malt2_name")) if _safe_get_session("malt2_name") in malt_options else 1
)
malt2_pct = st.sidebar.number_input(
    "Malt 2 %",
    min_value=0.0,
    max_value=100.0,
    step=1.0,
    value=_safe_get_session("malt2_pct", 20.0),
    key="malt2_pct_val"
)

malt3_name = st.sidebar.selectbox(
    "Malt 3",
    malt_options,
    key="malt3_name",
    index=2 if _safe_get_session("malt3_name") is None else malt_options.index(_safe_get_session("malt3_name")) if _safe_get_session("malt3_name") in malt_options else 2
)
malt3_pct = st.sidebar.number_input(
    "Malt 3 %",
    min_value=0.0,
    max_value=100.0,
    step=1.0,
    value=_safe_get_session("malt3_pct", 10.0),
    key="malt3_pct_val"
)

malt_selections = [
    {"name": malt1_name, "pct": malt1_pct},
    {"name": malt2_name, "pct": malt2_pct},
    {"name": malt3_name, "pct": malt3_pct},
]

# Keep them synced in session too
_safe_set_session("malt1_name", malt1_name)
_safe_set_session("malt2_name", malt2_name)
_safe_set_session("malt3_name", malt3_name)
_safe_set_session("malt1_pct", malt1_pct)
_safe_set_session("malt2_pct", malt2_pct)
_safe_set_session("malt3_pct", malt3_pct)


# --- Yeast ---
st.sidebar.markdown("## 🧬 Yeast")
yeast_options = sorted(yeast_df["Name"].dropna().unique().tolist())

chosen_yeast = st.sidebar.selectbox(
    "Yeast Strain",
    yeast_options,
    key="chosen_yeast",
    index=0 if _safe_get_session("chosen_yeast") is None else yeast_options.index(_safe_get_session("chosen_yeast")) if _safe_get_session("chosen_yeast") in yeast_options else 0
)
_safe_set_session("chosen_yeast", chosen_yeast)


# --- Predict button ---
run_button = st.sidebar.button("Predict Flavor 🧪")


# -----------------------------------------------------------------------------
# MAIN PAGE CONTENT
# -----------------------------------------------------------------------------
st.markdown(
    "<h1 style='display:flex;align-items:center;gap:.5rem;'>"
    "🍺 Beer Recipe Digital Twin"
    "</h1>",
    unsafe_allow_html=True
)
st.write(
    "Predict hop aroma, malt character, and fermentation profile using trained ML models."
)


def render_warning_box(msg_lines):
    """Draw a nice yellow info/warning box in main area."""
    st.markdown(
        f"""
        <div style="
            background-color:#fffdd7;
            border:1px solid #f5ec8c;
            border-radius:6px;
            padding:1rem 1.25rem;
            margin-top:1rem;
            color:#6a5500;
            font-size:1.15rem;
            line-height:1.5;
        ">
            {"<br>".join(msg_lines)}
        </div>
        """,
        unsafe_allow_html=True
    )


if run_button:
    # User clicked Predict Flavor 🧪, so actually run all models
    summary = summarize_beer(
        hop_bill,
        malt_selections,
        chosen_yeast,
        hop_model, hop_features, hop_dims,
        malt_model, malt_df, malt_features, malt_dims,
        yeast_model, yeast_df, yeast_features, yeast_dims
    )
    _safe_set_session("last_summary", summary)
else:
    summary = _safe_get_session("last_summary", None)


# If we have a summary *and* some hop weight was non-zero, show results (radar+details).
if summary and summary["hop_total_g"] > 0:
    hop_profile   = summary["hop_out"]
    hop_notes     = summary["hop_top_notes"]
    malt_traits   = summary["malt_traits"]
    yeast_traits  = summary["yeast_traits"]
    style_guess   = summary["style_guess"]

    col_left, col_right = st.columns([2, 1], vertical_alignment="top")

    with col_left:
        st.markdown("### Hop Aroma Radar")
        fig = plot_hop_radar(hop_profile, title="Hop Aroma Radar")
        if fig is not None:
            st.pyplot(fig, use_container_width=True)

        with st.expander("🧪 Debug: hop model input (what the model actually sees)"):
            X_debug, total_g_debug = build_hop_features(hop_bill, hop_features)
            debug_df = pd.DataFrame(X_debug, columns=hop_features)
            st.write(debug_df)

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
        st.write(", ".join([h for h,v in hop_bill.items() if v > 0]))

else:
    # No summary yet OR hop_total_g == 0
    warn_lines = [
        "⚠️ No recognizable hops in your bill (or total was 0 once normalized), "
        "so aroma prediction is basically flat.",
        "Pick hops from the list in the sidebar "
        "(open the '🌿 Model trained on hops:' section), "
        "then click <b>Predict Flavor 🧪</b> again."
    ]
    render_warning_box(warn_lines)

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
# CACHE: LOAD MODELS AND DATA
# -----------------------------------------------------------------------------
@st.cache_resource
def load_models_and_data():
    """
    Load trained models + reference data once.
    """
    hop_bundle   = joblib.load("hop_aroma_model.joblib")
    malt_bundle  = joblib.load("malt_sensory_model.joblib")
    yeast_bundle = joblib.load("yeast_sensory_model.joblib")

    hop_model      = hop_bundle["model"]
    hop_features   = hop_bundle["feature_cols"]   # e.g. ["hop_Citra", ...]
    hop_dims       = hop_bundle["aroma_dims"]     # e.g. ["tropical","citrus",...]

    malt_model     = malt_bundle["model"]
    malt_features  = malt_bundle["feature_cols"]  # numeric malt chemistry cols
    malt_dims      = malt_bundle["flavor_cols"]   # predicted malt traits

    yeast_model    = yeast_bundle["model"]
    yeast_features = yeast_bundle["feature_cols"] # e.g. Temp_avg_C, etc.
    yeast_dims     = yeast_bundle["flavor_cols"]  # predicted yeast traits

    malt_df  = pd.read_pickle("clean_malt_df.pkl")
    yeast_df = pd.read_pickle("clean_yeast_df.pkl")

    # hops list from feature columns (hop_Citra -> Citra)
    all_hops_model_knows = [c.replace("hop_", "") for c in hop_features]

    return (
        hop_model, hop_features, hop_dims,
        malt_model, malt_features, malt_dims,
        yeast_model, yeast_features, yeast_dims,
        malt_df, yeast_df,
        all_hops_model_knows
    )

(
    hop_model, hop_features, hop_dims,
    malt_model, malt_features, malt_dims,
    yeast_model, yeast_features, yeast_dims,
    malt_df, yeast_df,
    all_hops_model_knows
) = load_models_and_data()

# -----------------------------------------------------------------------------
# UTIL
# -----------------------------------------------------------------------------
def _safe_get_session(key, default):
    """Avoid KeyError on first run."""
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]

# ---- HOPS ----
def build_hop_features(hop_bill_dict, hop_features):
    """
    hop_bill_dict = {"Citra": 40, "Mosaic": 20, ...}
    Return a row aligned with hop_features order.
    """
    row = []
    for col in hop_features:
        hop_name = col.replace("hop_", "")
        row.append(hop_bill_dict.get(hop_name, 0))
    x = np.array(row).reshape(1, -1)
    return x

def predict_hop_profile(hop_bill_dict, hop_model, hop_features, hop_dims):
    x = build_hop_features(hop_bill_dict, hop_features)
    y_pred = hop_model.predict(x)[0]  # numeric intensities
    return dict(zip(hop_dims, y_pred))

# ---- MALTS ----
def get_weighted_malt_vector(malt_selections, malt_df, malt_features):
    """
    malt_selections:
    [
      {"name": "Maris Otter", "pct": 70},
      {"name": "Crystal 60L", "pct": 20},
      {"name": "Flaked Oats", "pct": 10}
    ]
    We'll produce a weighted sum of malt_features.
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
    y_pred = malt_model.predict(x)[0]  # often 0/1-ish trait flags
    return dict(zip(malt_dims, y_pred))

# ---- YEAST ----
def get_yeast_feature_vector(yeast_name, yeast_df, yeast_features):
    """
    Build single-row vector for chosen yeast.
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
    y_pred = yeast_model.predict(x)[0]
    return dict(zip(yeast_dims, y_pred))

# ---- STYLE SUMMARY ----
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

    hop_sorted = sorted(hop_out.items(), key=lambda kv: kv[1], reverse=True)
    top_hops   = [f"{k} ({round(v, 2)})" for k, v in hop_sorted[:2]]

    malt_active = [k for k, v in malt_out.items() if v == 1 or v > 0.5]
    yeast_active = [k for k, v in yeast_out.items() if v == 1 or v > 0.5]

    # Heuristic "style guess"
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
        "style_guess": style_guess
    }

# -----------------------------------------------------------------------------
# RADAR PLOT
# -----------------------------------------------------------------------------
def plot_hop_radar(hop_profile, title="Hop Aroma Radar"):
    """
    Radar chart for hop_profile (dict aroma_dim->float).
    Autoscale radius so we don't get a flat donut at 0.00.
    """
    if not hop_profile:
        return None

    labels = list(hop_profile.keys())
    values = np.array(list(hop_profile.values()), dtype=float)

    vmax = values.max() if values.max() > 0 else 1.0
    rmax = vmax * 1.2

    closed_vals = np.concatenate([values, values[:1]])
    n = len(labels)
    base_angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    closed_angles = np.concatenate([base_angles, base_angles[:1]])

    fig, ax = plt.subplots(
        figsize=(7, 7),
        subplot_kw=dict(polar=True)
    )

    ax.plot(closed_angles, closed_vals, linewidth=2, color="#1f77b4")
    ax.fill(closed_angles, closed_vals, alpha=0.25, color="#1f77b4")

    for ang, val in zip(base_angles, values):
        ax.text(
            ang,
            val,
            f"{val:.4f}",
            color="black",
            ha="center",
            va="center",
            fontsize=10,
            bbox=dict(
                boxstyle="round,pad=0.25",
                fc="white",
                ec="#1f77b4",
                lw=1
            )
        )

    ax.set_xticks(base_angles)
    ax.set_xticklabels(labels, fontsize=12)

    ax.set_ylim(0, rmax)
    ax.set_yticks(np.linspace(0, rmax, 5))
    ax.set_yticklabels(
        [f"{t:.2f}" for t in np.linspace(0, rmax, 5)],
        fontsize=10
    )
    ax.yaxis.grid(color="gray", linestyle="--", alpha=0.4)
    ax.xaxis.grid(color="gray", linestyle="--", alpha=0.4)
    ax.set_rlabel_position(0)

    ax.set_title(
        title,
        fontsize=32,
        fontweight="bold",
        pad=20
    )

    fig.tight_layout()
    return fig

# -----------------------------------------------------------------------------
# INIT PERSISTENT RESULT HOLDERS
# -----------------------------------------------------------------------------
_safe_get_session("latest_summary", None)
_safe_get_session("latest_hop_profile", None)

# -----------------------------------------------------------------------------
# SIDEBAR UI
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("🌿 Hop Bill")

    # Prepare defaults (first run)
    if all_hops_model_knows:
        _safe_get_session("hop1_name", all_hops_model_knows[0])
    else:
        _safe_get_session("hop1_name", "")
    if len(all_hops_model_knows) > 1:
        _safe_get_session("hop2_name", all_hops_model_knows[1])
    else:
        _safe_get_session("hop2_name", st.session_state["hop1_name"])
    if len(all_hops_model_knows) > 2:
        _safe_get_session("hop3_name", all_hops_model_knows[2])
    else:
        _safe_get_session("hop3_name", st.session_state["hop1_name"])
    if len(all_hops_model_knows) > 3:
        _safe_get_session("hop4_name", all_hops_model_knows[3])
    else:
        _safe_get_session("hop4_name", st.session_state["hop1_name"])

    _safe_get_session("hop1_amt", 40.0)
    _safe_get_session("hop2_amt", 40.0)
    _safe_get_session("hop3_amt", 0.0)
    _safe_get_session("hop4_amt", 0.0)

    all_hops_sorted = sorted(all_hops_model_knows)

    st.selectbox("Hop 1", all_hops_sorted, key="hop1_name")
    st.number_input(
        f"{st.session_state.hop1_name} (g)",
        min_value=0.0, max_value=200.0, step=1.0,
        key="hop1_amt"
    )

    st.selectbox("Hop 2", all_hops_sorted, key="hop2_name")
    st.number_input(
        f"{st.session_state.hop2_name} (g)",
        min_value=0.0, max_value=200.0, step=1.0,
        key="hop2_amt"
    )

    st.selectbox("Hop 3", all_hops_sorted, key="hop3_name")
    st.number_input(
        f"{st.session_state.hop3_name} (g)",
        min_value=0.0, max_value=200.0, step=1.0,
        key="hop3_amt"
    )

    st.selectbox("Hop 4", all_hops_sorted, key="hop4_name")
    st.number_input(
        f"{st.session_state.hop4_name} (g)",
        min_value=0.0, max_value=200.0, step=1.0,
        key="hop4_amt"
    )

    hop_bill = {
        st.session_state.hop1_name: st.session_state.hop1_amt,
        st.session_state.hop2_name: st.session_state.hop2_amt,
        st.session_state.hop3_name: st.session_state.hop3_amt,
        st.session_state.hop4_name: st.session_state.hop4_amt,
    }

    st.header("🌾 Malt Bill")
    malt_options = sorted(malt_df["PRODUCT NAME"].unique().tolist())

    if malt_options:
        _safe_get_session("malt1_name", malt_options[0])
    else:
        _safe_get_session("malt1_name", "")
    if len(malt_options) > 1:
        _safe_get_session("malt2_name", malt_options[1])
    else:
        _safe_get_session("malt2_name", st.session_state["malt1_name"])
    if len(malt_options) > 2:
        _safe_get_session("malt3_name", malt_options[2])
    else:
        _safe_get_session("malt3_name", st.session_state["malt1_name"])

    _safe_get_session("malt1_pct", 70.0)
    _safe_get_session("malt2_pct", 20.0)
    _safe_get_session("malt3_pct", 10.0)

    st.selectbox("Malt 1", malt_options, key="malt1_name")
    st.number_input("Malt 1 %", min_value=0.0, max_value=100.0, step=1.0, key="malt1_pct")

    st.selectbox("Malt 2", malt_options, key="malt2_name")
    st.number_input("Malt 2 %", min_value=0.0, max_value=100.0, step=1.0, key="malt2_pct")

    st.selectbox("Malt 3", malt_options, key="malt3_name")
    st.number_input("Malt 3 %", min_value=0.0, max_value=100.0, step=1.0, key="malt3_pct")

    malt_selections = [
        {"name": st.session_state.malt1_name, "pct": st.session_state.malt1_pct},
        {"name": st.session_state.malt2_name, "pct": st.session_state.malt2_pct},
        {"name": st.session_state.malt3_name, "pct": st.session_state.malt3_pct},
    ]

    st.header("🧬 Yeast")
    yeast_options = sorted(yeast_df["Name"].dropna().unique().tolist())
    if yeast_options:
        _safe_get_session("chosen_yeast", yeast_options[0])
    else:
        _safe_get_session("chosen_yeast", "")

    st.selectbox("Yeast Strain", yeast_options, key="chosen_yeast")

    run_button = st.button("Predict Flavor 🧪")

    st.divider()
    with st.expander("🍥 Model trained on hops:"):
        st.caption(", ".join(all_hops_model_knows))

# -----------------------------------------------------------------------------
# RUN PREDICTION & STORE RESULT
# -----------------------------------------------------------------------------
if run_button:
    summary = summarize_beer(
        hop_bill,
        malt_selections,
        st.session_state.chosen_yeast,
        hop_model, hop_features, hop_dims,
        malt_model, malt_df, malt_features, malt_dims,
        yeast_model, yeast_df, yeast_features, yeast_dims
    )
    st.session_state.latest_summary = summary
    st.session_state.latest_hop_profile = summary["hop_out"]

# Grab current in-session results (may still be None first load)
summary     = st.session_state.latest_summary
hop_profile = st.session_state.latest_hop_profile

# -----------------------------------------------------------------------------
# MAIN BODY
# -----------------------------------------------------------------------------
st.title("🍺 Beer Recipe Digital Twin")
st.caption("Predict hop aroma, malt character, and fermentation profile using trained ML models.")

col_left, col_right = st.columns([2, 1], vertical_alignment="top")

with col_left:
    if hop_profile and any(v != 0 for v in hop_profile.values()):
        fig = plot_hop_radar(hop_profile, title="Hop Aroma Radar")
        if fig is not None:
            st.pyplot(fig, use_container_width=True)
    else:
        st.warning(
            "⚠️ No recognizable hops in your bill (or total was 0 once normalized), "
            "so aroma prediction is basically flat.\n\n"
            "Pick hops from the list in the sidebar (open the '🍥 Model trained on hops:' section), "
            "then click **Predict Flavor 🧪** again."
        )

with col_right:
    st.markdown("### Top hop notes:")
    if summary and summary["hop_top_notes"]:
        for n in summary["hop_top_notes"]:
            st.write(f"- {n}")
    else:
        st.write("_–_")

    st.markdown("### Malt character:")
    if summary and summary["malt_traits"]:
        st.write(", ".join(summary["malt_traits"]))
    else:
        st.write("_–_")

    st.markdown("### Yeast character:")
    if summary and summary["yeast_traits"]:
        st.write(", ".join(summary["yeast_traits"]))
    else:
        st.write("_–_")

    st.markdown("### Style direction:")
    if summary:
        st.write(f"🍺 {summary['style_guess']}")
    else:
        st.write("_–_")

    st.markdown("### Hops used by the model:")
    st.write(", ".join(list(hop_bill.keys())))

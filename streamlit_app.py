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
# UTILS: SAFE SESSION GET/SET
# -----------------------------------------------------------------------------
def _safe_get_session(key, default=None):
    """Read from st.session_state without throwing KeyErrors."""
    return st.session_state[key] if key in st.session_state else default

def _safe_set_session(key, value):
    """Write to st.session_state and return the value (convenience)."""
    st.session_state[key] = value
    return value

# bootstrap a couple keys we'll reuse
_safe_set_session("latest_summary", _safe_get_session("latest_summary", None))
_safe_set_session("latest_hop_profile", _safe_get_session("latest_hop_profile", {}))

# -----------------------------------------------------------------------------
# LOAD MODELS + DATA (CACHED)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_models_and_data():
    """
    Load all trained models + reference data once, cache them for the session.
    Expecting these files in repo root:
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
    hop_features   = hop_bundle["feature_cols"]   # ["hop_Citra", ...]
    hop_dims       = hop_bundle["aroma_dims"]     # ["tropical","citrus",...]

    malt_model     = malt_bundle["model"]
    malt_features  = malt_bundle["feature_cols"]  # malt chemistry columns
    malt_dims      = malt_bundle["flavor_cols"]   # ["caramel","bready",...]

    yeast_model    = yeast_bundle["model"]
    yeast_features = yeast_bundle["feature_cols"] # e.g. ["Temp_avg_C","Flocculation_num","Attenuation_pct"]
    yeast_dims     = yeast_bundle["flavor_cols"]  # yeast trait labels

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
    # "hop_Citra" → "Citra"
    return [c.replace("hop_", "") for c in hop_features]

def build_hop_features(hop_bill_dict, hop_features):
    """
    hop_bill_dict = {"Citra": 40, "Mosaic": 20, ...}

    We return:
       - x : 1 x N array for model input
       - df_debug : one-row dataframe labeled with hop_*, to show user
       - used_hops_list : list of hops that were nonzero
       - total_g : total grams (for checking zeros)
    """
    rows = []
    used_hops = []
    total_g = 0.0
    debug_dict = {}

    for col in hop_features:
        hop_name = col.replace("hop_", "")
        grams = float(hop_bill_dict.get(hop_name, 0))
        total_g += grams
        debug_dict[col] = grams
        rows.append(grams)
        if grams > 0:
            used_hops.append(hop_name)

    x = np.array(rows, dtype=float).reshape(1, -1)

    df_debug = pd.DataFrame([debug_dict])  # show in UI if we want
    return x, df_debug, used_hops, total_g

def predict_hop_profile(hop_bill_dict, hop_model, hop_features, hop_dims):
    """
    Returns (hop_out_dict, debug_df, used_hops_list, total_g)
    hop_out_dict is aroma_dim -> value
    """
    x, df_debug, used_hops, total_g = build_hop_features(hop_bill_dict, hop_features)

    # If total is 0, prediction is meaningless (or all zeros).
    # We'll just do model.predict anyway, but caller can decide how to display.
    try:
        y_pred = hop_model.predict(x)[0]
    except Exception:
        y_pred = np.zeros(len(hop_dims))

    hop_out = dict(zip(hop_dims, y_pred))
    return hop_out, df_debug, used_hops, total_g


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
    try:
        y_pred = malt_model.predict(x)[0]
    except Exception:
        y_pred = np.zeros(len(malt_dims))
    return dict(zip(malt_dims, y_pred))


# ---- YEAST ----
def get_yeast_feature_vector(yeast_name, yeast_df, yeast_features):
    """
    Build single-row model input for chosen yeast strain.
    """
    row = yeast_df[yeast_df["Name"] == yeast_name].head(1)
    if row.empty:
        # fallback all zeros
        return np.zeros(len(yeast_features)).reshape(1, -1)

    vec = [
        row.iloc[0]["Temp_avg_C"],
        row.iloc[0]["Flocculation_num"],
        row.iloc[0]["Attenuation_pct"]
    ]
    return np.array(vec, dtype=float).reshape(1, -1)

def predict_yeast_profile(yeast_name, yeast_model, yeast_df, yeast_features, yeast_dims):
    x = get_yeast_feature_vector(yeast_name, yeast_df, yeast_features)
    try:
        y_pred = yeast_model.predict(x)[0]
    except Exception:
        y_pred = np.zeros(len(yeast_dims))
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
    hop_out, hop_debug_df, used_hops_list, total_g = predict_hop_profile(
        hop_bill_dict, hop_model, hop_features, hop_dims
    )
    malt_out  = predict_malt_profile_from_blend(
        malt_selections, malt_model, malt_df, malt_features, malt_dims
    )
    yeast_out = predict_yeast_profile(
        yeast_name, yeast_model, yeast_df, yeast_features, yeast_dims
    )

    # build easy-to-read lists
    hop_sorted = sorted(hop_out.items(), key=lambda kv: kv[1], reverse=True)
    top_hops   = [f"{k} ({round(v, 2)})" for k, v in hop_sorted[:2]]

    malt_active  = [k for k,v in malt_out.items() if v == 1]
    yeast_active = [k for k,v in yeast_out.items() if v == 1]

    # style heuristic
    style_guess = "Experimental / Hybrid"
    if ("clean_neutral" in yeast_out and yeast_out["clean_neutral"] == 1
        and "dry_finish" in yeast_out and yeast_out["dry_finish"] == 1):
        if any(("citrus" in n[0] or "resin" in n[0]) for n in hop_sorted[:2]):
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
        "hop_debug_df": hop_debug_df,
        "hop_used": used_hops_list,
        "hop_total_g": total_g,
        "hop_top_notes": top_hops,
        "malt_traits": malt_active,
        "yeast_traits": yeast_active,
        "style_guess": style_guess
    }


# -----------------------------------------------------------------------------
# RADAR PLOT (RESCALED)
# -----------------------------------------------------------------------------
def plot_hop_radar(hop_profile, title="Hop Aroma Radar"):
    """
    Radar plot where:
    - axes are aroma dims (tropical, citrus, fruity, resinous, floral, herbal, spicy, earthy, etc.)
    - values are normalized 0-1 based on that profile's max, so we always see shape
    """

    if not hop_profile:
        # default placeholders
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
    raw_vals = np.array(list(hop_profile.values()), dtype=float)

    max_val = raw_vals.max() if raw_vals.size > 0 else 0.0
    if max_val > 0:
        plot_vals = raw_vals / max_val
    else:
        plot_vals = raw_vals

    closed_vals = np.concatenate([plot_vals, plot_vals[:1]])
    n = len(labels)

    base_angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    closed_angles = np.concatenate([base_angles, base_angles[:1]])

    fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))

    ax.plot(closed_angles, closed_vals, linewidth=2)
    ax.fill(closed_angles, closed_vals, alpha=0.25)

    # numeric value boxes
    for ang, val in zip(base_angles, plot_vals):
        ax.text(
            ang,
            val,
            f"{val:.4f}",
            color="black",
            ha="center",
            va="center",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", lw=1)
        )

    ax.set_xticks(base_angles)
    ax.set_xticklabels(labels, fontsize=12)

    ax.set_rlabel_position(0)
    ax.yaxis.grid(color="gray", linestyle="--", alpha=0.4)
    ax.xaxis.grid(color="gray", linestyle="--", alpha=0.4)

    ax.set_ylim(0, 1.0 if max_val > 0 else 1.0)

    ax.set_title(title, fontsize=28, fontweight="bold", pad=20)

    fig.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# SIDEBAR UI
# -----------------------------------------------------------------------------
st.sidebar.header("🌿 Hop Bill")

all_hops_sorted = sorted(get_all_hop_names(hop_features))

# We'll keep 4 hop slots
hop1_name = st.sidebar.selectbox("Hop 1", all_hops_sorted, key="hop1_name")
hop1_amt  = st.sidebar.number_input(f"{hop1_name} (g)", min_value=0.0, step=1.0, value=_safe_get_session("hop1_amt", 40.0), key="hop1_amt_val")

hop2_name = st.sidebar.selectbox("Hop 2", all_hops_sorted, key="hop2_name")
hop2_amt  = st.sidebar.number_input(f"{hop2_name} (g)", min_value=0.0, step=1.0, value=_safe_get_session("hop2_amt", 40.0), key="hop2_amt_val")

hop3_name = st.sidebar.selectbox("Hop 3", all_hops_sorted, key="hop3_name")
hop3_amt  = st.sidebar.number_input(f"{hop3_name} (g)", min_value=0.0, step=1.0, value=_safe_get_session("hop3_amt", 0.0), key="hop3_amt_val")

hop4_name = st.sidebar.selectbox("Hop 4", all_hops_sorted, key="hop4_name")
hop4_amt  = st.sidebar.number_input(f"{hop4_name} (g)", min_value=0.0, step=1.0, value=_safe_get_session("hop4_amt", 0.0), key="hop4_amt_val")

hop_bill = {
    hop1_name: hop1_amt,
    hop2_name: hop2_amt,
    hop3_name: hop3_amt,
    hop4_name: hop4_amt,
}

st.sidebar.header("🌾 Malt Bill")

malt_options = sorted(malt_df["PRODUCT NAME"].unique().tolist())

malt1_name = st.sidebar.selectbox("Malt 1", malt_options, key="malt1_name")
malt1_pct  = st.sidebar.number_input("Malt 1 %", min_value=0.0, max_value=100.0,
                                     step=1.0, value=_safe_get_session("malt1_pct", 70.0), key="malt1_pct_val")

malt2_name = st.sidebar.selectbox("Malt 2", malt_options, key="malt2_name")
malt2_pct  = st.sidebar.number_input("Malt 2 %", min_value=0.0, max_value=100.0,
                                     step=1.0, value=_safe_get_session("malt2_pct", 20.0), key="malt2_pct_val")

malt3_name = st.sidebar.selectbox("Malt 3", malt_options, key="malt3_name")
malt3_pct  = st.sidebar.number_input("Malt 3 %", min_value=0.0, max_value=100.0,
                                     step=1.0, value=_safe_get_session("malt3_pct", 10.0), key="malt3_pct_val")

malt_selections = [
    {"name": malt1_name, "pct": malt1_pct},
    {"name": malt2_name, "pct": malt2_pct},
    {"name": malt3_name, "pct": malt3_pct},
]

st.sidebar.header("🧬 Yeast")
yeast_options = sorted(yeast_df["Name"].dropna().unique().tolist())
chosen_yeast  = st.sidebar.selectbox("Yeast Strain", yeast_options, key="yeast_name")

run_button = st.sidebar.button("Predict Flavor 🧪")

with st.sidebar.expander("🌀 Model trained on hops:"):
    st.write(", ".join(all_hops_sorted))

# -----------------------------------------------------------------------------
# HEADER
# -----------------------------------------------------------------------------
st.markdown(
    """
    <h1 style='display:flex; align-items:center; gap:0.5rem; font-size:2.5rem;'>
        <span>🍺</span>
        <span>Beer Recipe Digital Twin</span>
    </h1>
    """,
    unsafe_allow_html=True,
)

st.write("Predict hop aroma, malt character, and fermentation profile using trained ML models.")

# -----------------------------------------------------------------------------
# RUN PREDICTION (on click)
# -----------------------------------------------------------------------------
if run_button:
    summary_now = summarize_beer(
        hop_bill,
        malt_selections,
        chosen_yeast,
        hop_model, hop_features, hop_dims,
        malt_model, malt_df, malt_features, malt_dims,
        yeast_model, yeast_df, yeast_features, yeast_dims
    )
    # store so we can render even after rerun
    _safe_set_session("latest_summary", summary_now)
    _safe_set_session("latest_hop_profile", summary_now["hop_out"])

# after possibly updating session_state, read what we have
summary = _safe_get_session("latest_summary", None)
hop_profile = _safe_get_session("latest_hop_profile", {})

# -----------------------------------------------------------------------------
# LAYOUT FOR RESULTS
# -----------------------------------------------------------------------------
col_left, col_right = st.columns([2, 1], vertical_alignment="top")

with col_left:
    # case 1: we have a summary and hops total > 0 → show radar
    if summary and summary["hop_total_g"] > 0:
        fig = plot_hop_radar(summary["hop_out"], title="Hop Aroma Radar")
        st.pyplot(fig, use_container_width=True)

        # debug expand
        with st.expander("🔬 Debug: hop model input (what the model actually sees)"):
            st.dataframe(summary["hop_debug_df"])
    else:
        # warning / instructions box
        st.warning(
            "⚠️ No recognizable hops in your bill (or total was 0 once normalized), "
            "so aroma prediction is basically flat.\n\n"
            "Pick hops from the list in the sidebar (open the '🌀 Model trained on hops:' "
            "section), then click **Predict Flavor 🧪** again."
        )

with col_right:
    st.markdown("### Top hop notes:")
    if summary and summary["hop_top_notes"]:
        for n in summary["hop_top_notes"]:
            st.write(f"- {n}")
    else:
        st.write("-")

    st.markdown("### Malt character:")
    if summary and summary["malt_traits"]:
        st.write(", ".join(summary["malt_traits"]))
    else:
        st.write("-")

    st.markdown("### Yeast character:")
    if summary and summary["yeast_traits"]:
        st.write(", ".join(summary["yeast_traits"]))
    else:
        st.write("-")

    st.markdown("### Style direction:")
    if summary:
        st.write(f"🍻 {summary['style_guess']}")
    else:
        st.write("-")

    st.markdown("### Hops used by the model:")
    if summary and summary["hop_used"]:
        st.write(", ".join(summary["hop_used"]))
    else:
        st.write("-")

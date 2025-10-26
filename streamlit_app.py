import streamlit as st
import numpy as np
import pandas as pd
import joblib
import warnings
import unicodedata
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
# LOAD MODELS & DATA (cached)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_models_and_data():
    """
    Load and cache trained models and reference data.
    We assume these files exist in the repo root:
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
    hop_features   = hop_bundle["feature_cols"]   # e.g. ["hop_Galaxy", ...]
    hop_dims       = hop_bundle["aroma_dims"]     # e.g. ["tropical","citrus",...]

    malt_model     = malt_bundle["model"]
    malt_features  = malt_bundle["feature_cols"]  # columns from malt_df
    malt_dims      = malt_bundle["flavor_cols"]   # predicted malt traits

    yeast_model    = yeast_bundle["model"]
    yeast_features = yeast_bundle["feature_cols"] # ["Temp_avg_C","Flocculation_num","Attenuation_pct", ...]
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
# HELPERS
# -----------------------------------------------------------------------------

# ---- name normalization so "Vic Secret", "vic secret", "VİC SECRET®" match
def normalize_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    # remove accents/diacritics
    name = unicodedata.normalize("NFKD", name)
    name = "".join([c for c in name if not unicodedata.combining(c)])
    # lowercase, strip spaces/punctuation-ish
    name = (
        name.lower()
            .replace("®", "")
            .replace("™", "")
            .replace(" ", "")
            .replace("-", "")
            .replace("_", "")
            .strip()
    )
    return name

# -------------------------
# HOPS
# -------------------------
def get_all_hop_names(hop_features):
    # "hop_Galaxy" -> "Galaxy"
    return [c.replace("hop_", "") for c in hop_features]

def build_hop_features(hop_bill_dict, hop_features):
    """
    hop_bill_dict = {"Citra": 40, "Galaxy": 60, ... grams}
    Return:
      x           -> 1xN numpy array aligned with hop_features
      matched_hops -> list of hops that actually contributed (>0)
    - We normalize names to match model columns.
    - We scale by percentage of total grams (so model sees proportions).
    """
    # map normalized hop name -> column name
    feature_map = {
        normalize_name(f.replace("hop_", "")): f
        for f in hop_features
    }

    # keep only positive amounts
    positive = {
        normalize_name(k): float(v)
        for k, v in hop_bill_dict.items()
        if v is not None and float(v) > 0
    }

    total = sum(positive.values())
    if total <= 0:
        total = 1.0  # avoid div 0; everything will become 0 anyway

    proportions = {k: v / total for k, v in positive.items()}

    row_vals = []
    matched_hops = []
    for feat_norm, full_col in feature_map.items():
        val = proportions.get(feat_norm, 0.0)
        if val > 0:
            matched_hops.append(full_col.replace("hop_", ""))
        row_vals.append(val)

    x = np.array(row_vals).reshape(1, -1)
    return x, matched_hops

def predict_hop_profile(hop_bill_dict, hop_model, hop_features, hop_dims):
    x, matched = build_hop_features(hop_bill_dict, hop_features)
    y_pred = hop_model.predict(x)[0]  # vector of intensities
    profile = dict(zip(hop_dims, y_pred))
    return profile, matched

# -------------------------
# MALTS
# -------------------------
def get_weighted_malt_vector(malt_selections, malt_df, malt_features):
    """
    malt_selections:
    [
      {"name": "Maris Otter", "pct": 70},
      {"name": "Crystal 60L", "pct": 20},
      {"name": "Flaked Oats", "pct": 10}
    ]
    We create a weighted average of the malt chemistry columns.
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
    y_pred = malt_model.predict(x)[0]  # often binary/flag-like outputs
    return dict(zip(malt_dims, y_pred))

# -------------------------
# YEAST
# -------------------------
def get_yeast_feature_vector(yeast_name, yeast_df, yeast_features):
    row = yeast_df[yeast_df["Name"] == yeast_name].head(1)
    if row.empty:
        return np.zeros(len(yeast_features)).reshape(1, -1)

    vec = [
        row.iloc[0]["Temp_avg_C"],
        row.iloc[0]["Flocculation_num"],
        row.iloc[0]["Attenuation_pct"],
    ]
    return np.array(vec).reshape(1, -1)

def predict_yeast_profile(yeast_name, yeast_model, yeast_df, yeast_features, yeast_dims):
    x = get_yeast_feature_vector(yeast_name, yeast_df, yeast_features)
    y_pred = yeast_model.predict(x)[0]  # often binary flags again
    return dict(zip(yeast_dims, y_pred))

# -------------------------
# RADAR PLOT
# -------------------------
def plot_hop_radar(hop_profile, title="Hop Aroma Radar"):
    """
    Make a polar/radar chart of hop aroma intensities.
    FIXED so tick count == label count (no crash).
    """
    if not hop_profile:
        hop_profile = {d: 0.0 for d in [
            "tropical","citrus","fruity","resinous",
            "floral","herbal","spicy","earthy"
        ]}

    labels = list(hop_profile.keys())
    values = np.array(list(hop_profile.values()), dtype=float)

    # angles for unique axes
    n = len(labels)
    base_angles = np.linspace(0, 2*np.pi, n, endpoint=False)

    # closed polygon values/angles (repeat first point)
    closed_values = np.concatenate([values, values[:1]])
    closed_angles = np.concatenate([base_angles, base_angles[:1]])

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    ax.plot(closed_angles, closed_values, color="#1f77b4", linewidth=2)
    ax.fill(closed_angles, closed_values, color="#1f77b4", alpha=0.25)

    # annotate each vertex with numeric value
    for ang, val in zip(base_angles, values):
        ax.text(
            ang,
            val,
            f"{val:.4f}",
            color="black",
            ha="center",
            va="center",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#1f77b4", lw=1),
        )

    # category ticks
    ax.set_xticks(base_angles)
    ax.set_xticklabels(labels, fontsize=12)

    # radial grid styling
    ax.set_rlabel_position(0)
    ax.yaxis.grid(color="gray", linestyle="--", alpha=0.4)
    ax.xaxis.grid(color="gray", linestyle="--", alpha=0.4)

    ax.set_title(title, fontsize=24, fontweight="bold", pad=20)
    fig.tight_layout()
    return fig

# -------------------------
# SUMMARY TEXT (style guess etc.)
# -------------------------
def summarize_beer(
    hop_bill_dict,
    malt_selections,
    yeast_name,
    hop_model, hop_features, hop_dims,
    malt_model, malt_df, malt_features, malt_dims,
    yeast_model, yeast_df, yeast_features, yeast_dims,
):
    hop_out, matched_hops = predict_hop_profile(
        hop_bill_dict, hop_model, hop_features, hop_dims
    )
    malt_out  = predict_malt_profile_from_blend(
        malt_selections, malt_model, malt_df, malt_features, malt_dims
    )
    yeast_out = predict_yeast_profile(
        yeast_name, yeast_model, yeast_df, yeast_features, yeast_dims
    )

    # sort hop aroma descending
    hop_sorted = sorted(hop_out.items(), key=lambda kv: kv[1], reverse=True)
    top_hops   = [f"{k} ({round(v, 2)})" for k, v in hop_sorted[:2]]

    # malt traits flagged
    malt_active  = [k for k,v in malt_out.items() if v == 1 or v is True]

    # yeast traits flagged
    yeast_active = [k for k,v in yeast_out.items() if v == 1 or v is True]

    # style heuristic (super rough demo rules)
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
        "matched_hops": matched_hops
    }

# -----------------------------------------------------------------------------
# SIDEBAR UI CONTROLS
# -----------------------------------------------------------------------------

st.sidebar.markdown("### 🌿 Hop Bill")

all_hops = sorted(get_all_hop_names(hop_features))

# We'll store up to 4 hops
if "hop_rows" not in st.session_state:
    # default 2 rows with 40g each
    st.session_state.hop_rows = [
        {"name": all_hops[0] if all_hops else "", "amt": 40},
        {"name": all_hops[1] if len(all_hops) > 1 else (all_hops[0] if all_hops else ""), "amt": 40},
        {"name": "", "amt": 0},
        {"name": "", "amt": 0},
    ]

# Draw editable hop rows
new_rows = []
for i, row in enumerate(st.session_state.hop_rows):
    col1, col2 = st.sidebar.columns([2,1])
    with col1:
        hop_name = st.selectbox(
            f"Hop {i+1}",
            [""] + all_hops,
            index=([""]+all_hops).index(row["name"]) if row["name"] in ([""]+all_hops) else 0,
            key=f"hop_name_{i}",
        )
    with col2:
        hop_amt = st.number_input(
            f"{row['name'] or 'amount'} (g)",
            min_value=0.0,
            max_value=500.0,
            value=float(row["amt"]),
            step=5.0,
            key=f"hop_amt_{i}",
        )
    new_rows.append({"name": hop_name, "amt": hop_amt})

# persist
st.session_state.hop_rows = new_rows

# Build dict { "Citra": grams, ... }
hop_bill = {
    r["name"]: r["amt"]
    for r in st.session_state.hop_rows
    if r["name"] and r["amt"] > 0
}

st.sidebar.markdown("### 🌾 Malt Bill")

malt_options = sorted(malt_df["PRODUCT NAME"].unique().tolist())

# default malt bill 70/20/10
if "malt_rows" not in st.session_state:
    st.session_state.malt_rows = [
        {"name": malt_options[0] if malt_options else "", "pct": 70.0},
        {"name": malt_options[1] if len(malt_options)>1 else (malt_options[0] if malt_options else ""), "pct": 20.0},
        {"name": malt_options[2] if len(malt_options)>2 else (malt_options[0] if malt_options else ""), "pct": 10.0},
    ]

malt_edits = []
for i, row in enumerate(st.session_state.malt_rows):
    col1, col2 = st.sidebar.columns([2,1])
    with col1:
        malt_name = st.selectbox(
            f"Malt {i+1}",
            malt_options,
            index=malt_options.index(row["name"]) if row["name"] in malt_options else 0,
            key=f"malt_name_{i}",
        )
    with col2:
        malt_pct = st.number_input(
            f"% {i+1}",
            min_value=0.0,
            max_value=100.0,
            value=float(row["pct"]),
            step=1.0,
            key=f"malt_pct_{i}",
        )
    malt_edits.append({"name": malt_name, "pct": malt_pct})

st.session_state.malt_rows = malt_edits
malt_selections = malt_edits

st.sidebar.markdown("### 🧫 Yeast")

yeast_options = sorted(yeast_df["Name"].dropna().unique().tolist())
if "chosen_yeast" not in st.session_state:
    st.session_state.chosen_yeast = yeast_options[0] if yeast_options else ""

chosen_yeast = st.sidebar.selectbox(
    "Yeast Strain",
    yeast_options,
    index=yeast_options.index(st.session_state.chosen_yeast) if st.session_state.chosen_yeast in yeast_options else 0,
    key="chosen_yeast",
)

run_button = st.sidebar.button("Predict Flavor 🧪")

# optional helper section listing trained hops so user knows what works
with st.sidebar.expander("💡 Model trained on hops:"):
    st.write(", ".join(sorted(get_all_hop_names(hop_features))))

# -----------------------------------------------------------------------------
# MAIN LAYOUT
# -----------------------------------------------------------------------------

st.markdown(
    "## 🍺 Beer Recipe Digital Twin\n"
    "Predict hop aroma, malt character, and fermentation profile using trained ML models."
)

if run_button:
    # make predictions / summary
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
    matched_hops  = summary["matched_hops"]

    # warn if no recognizable hops
    if len(matched_hops) == 0:
        st.warning(
            "⚠ No recognizable hops in your bill (or total was 0 once normalized), "
            "so aroma prediction is basically flat.\n\n"
            "Pick hops from the list in the sidebar (open the '💡 Model trained on hops:' "
            "section), then click Predict Flavor 🧪 again."
        )

    # -- two columns: radar on left, text on right
    col_left, col_right = st.columns([2,1], vertical_alignment="top")

    with col_left:
        fig = plot_hop_radar(hop_profile, title="Hop Aroma Radar")
        st.pyplot(fig, use_container_width=True)

    with col_right:
        st.markdown("### Top hop notes:")
        if hop_notes:
            for n in hop_notes:
                st.write(f"- {n}")
        else:
            st.write("_None_")

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
        st.write(f"🍻 {style_guess}")

        st.markdown("### Hops used by the model:")
        st.write(", ".join(matched_hops) if matched_hops else "—")

    # debug accordion
    with st.expander("🧪 Debug: hop model input (what the model actually sees)"):
        x_debug, _ = build_hop_features(hop_bill, hop_features)
        debug_df = pd.DataFrame(x_debug, columns=hop_features)
        st.dataframe(debug_df, use_container_width=True)

else:
    # nothing predicted yet
    st.info(
        "Pick hops, malt %, and yeast in the left sidebar, then click **Predict Flavor 🧪**.\n\n"
        "You'll get aroma radar + style guidance here."
    )

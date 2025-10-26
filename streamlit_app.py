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
# CACHED LOADERS
# -----------------------------------------------------------------------------
@st.cache_resource
def load_models_and_data():
    """
    Load trained models and reference data once per session.
    Assumes these files exist in the repo root:
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
    hop_features   = hop_bundle["feature_cols"]   # e.g. ["hop_Citra", "hop_Mosaic", ...]
    hop_dims       = hop_bundle["aroma_dims"]     # e.g. ["tropical","citrus","resinous",...]

    malt_model     = malt_bundle["model"]
    malt_features  = malt_bundle["feature_cols"]  # numeric malt chemistry cols
    malt_dims      = malt_bundle["flavor_cols"]   # e.g. ["bready","caramel","roast",...]

    yeast_model    = yeast_bundle["model"]
    yeast_features = yeast_bundle["feature_cols"] # ["Temp_avg_C","Flocculation_num","Attenuation_pct"]
    yeast_dims     = yeast_bundle["flavor_cols"]  # e.g. ["fruity_esters","clean_neutral",...]

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

# ---- HOPS --------------------------------------------------------
def get_all_hop_names(hop_features_cols):
    # features look like "hop_Citra", "hop_Mosaic" -> show just "Citra", "Mosaic"
    return [c.replace("hop_", "") for c in hop_features_cols]

def build_hop_features(hop_bill_dict, hop_features_cols):
    """
    hop_bill_dict = {"Citra": 40, "Mosaic": 20, ...}
    Return a single-row numpy array aligned with hop_features_cols.
    We normalize internally (so 40 + 20 + ... won't blow up scale).
    """
    # raw vector in the same order as model expects
    row = []
    for col in hop_features_cols:
        hop_name = col.replace("hop_", "")
        row.append(float(hop_bill_dict.get(hop_name, 0.0)))

    vec = np.array(row, dtype=float)

    total = vec.sum()
    if total > 0:
        vec = vec / total  # normalize so sum = 1
    # shape (1, n_features)
    return vec.reshape(1, -1)

def predict_hop_profile(hop_bill_dict, hop_model, hop_features_cols, hop_dims_out):
    """
    Returns dict: { "tropical": value, "citrus": value, ... }
    or {} if no valid hop data was provided.
    """
    vec = build_hop_features(hop_bill_dict, hop_features_cols)
    if np.allclose(vec.sum(), 0.0):
        # no hops / total = 0 -> can't meaningfully predict
        return {}

    y_pred = hop_model.predict(vec)[0]  # shape (len(hop_dims_out),)
    return dict(zip(hop_dims_out, y_pred))


# ---- MALTS --------------------------------------------------------
def get_weighted_malt_vector(malt_selections, malt_df, malt_features_cols):
    """
    malt_selections = [
        {"name":"Maris Otter","pct":70},
        {"name":"Crystal 60L","pct":20},
        {"name":"Oats","pct":10},
    ]
    We'll build a weighted average of each malt's chemistry features.
    """

    blend_vec = np.zeros(len(malt_features_cols), dtype=float)

    for item in malt_selections:
        malt_name = item["name"]
        pct       = float(item["pct"])

        row = malt_df[malt_df["PRODUCT NAME"] == malt_name].head(1)
        if row.empty:
            continue

        # grab features in correct order
        feat_vals = [row.iloc[0][f] for f in malt_features_cols]
        feat_vals = np.array(feat_vals, dtype=float)

        blend_vec += feat_vals * (pct / 100.0)

    # shape (1, n_features)
    return blend_vec.reshape(1, -1)

def predict_malt_profile(malt_selections, malt_model, malt_df, malt_features_cols, malt_dims_out):
    """
    Returns dict { "bready":1, "caramel":0, ... } etc.
    """
    x = get_weighted_malt_vector(malt_selections, malt_df, malt_features_cols)
    y_pred = malt_model.predict(x)[0]
    # could be continuous or binary 0/1 labels depending on training
    return dict(zip(malt_dims_out, y_pred))


# ---- YEAST --------------------------------------------------------
def build_yeast_feature_vector(yeast_name, yeast_df, yeast_features_cols):
    """
    yeast_df row columns expected:
      "Name", "Temp_avg_C", "Flocculation_num", "Attenuation_pct"
    """
    row = yeast_df[yeast_df["Name"] == yeast_name].head(1)
    if row.empty:
        # fallback zeros
        return np.zeros(len(yeast_features_cols)).reshape(1, -1)

    vals = [
        row.iloc[0]["Temp_avg_C"],
        row.iloc[0]["Flocculation_num"],
        row.iloc[0]["Attenuation_pct"],
    ]
    return np.array(vals, dtype=float).reshape(1, -1)

def predict_yeast_profile(yeast_name, yeast_model, yeast_df, yeast_features_cols, yeast_dims_out):
    """
    Returns dict like { "fruity_esters":1, "clean_neutral":1, ... }
    """
    x = build_yeast_feature_vector(yeast_name, yeast_df, yeast_features_cols)
    y_pred = yeast_model.predict(x)[0]
    return dict(zip(yeast_dims_out, y_pred))


# ---- STYLE SUMMARY ------------------------------------------------
def summarize_beer(
    hop_bill_dict,
    malt_selections,
    yeast_name,
    hop_model, hop_features_cols, hop_dims_out,
    malt_model, malt_df, malt_features_cols, malt_dims_out,
    yeast_model, yeast_df, yeast_features_cols, yeast_dims_out,
):
    hop_out   = predict_hop_profile(hop_bill_dict, hop_model, hop_features_cols, hop_dims_out)
    malt_out  = predict_malt_profile(malt_selections, malt_model, malt_df, malt_features_cols, malt_dims_out)
    yeast_out = predict_yeast_profile(yeast_name, yeast_model, yeast_df, yeast_features_cols, yeast_dims_out)

    # top hop notes
    if hop_out:
        hop_sorted = sorted(hop_out.items(), key=lambda kv: kv[1], reverse=True)
        hop_top    = [f"{k} ({round(v, 2)})" for k,v in hop_sorted[:2]]
    else:
        hop_sorted = []
        hop_top    = []

    # malt traits "on"
    # if they are binary, 1 means active. if continuous, show top few
    malt_traits_active = []
    if len(malt_out) > 0:
        for k,v in malt_out.items():
            # consider >0.5 "active" if numeric
            try:
                if float(v) >= 0.5:
                    malt_traits_active.append(k)
            except:
                # fallback if it's not numeric
                if v == 1:
                    malt_traits_active.append(k)
        # if nothing triggered, show best-scoring anyway:
        if not malt_traits_active:
            malt_sorted = sorted(malt_out.items(), key=lambda kv: kv[1], reverse=True)
            malt_traits_active = [malt_sorted[0][0]] if malt_sorted else []

    # yeast traits "on"
    yeast_traits_active = []
    if len(yeast_out) > 0:
        for k,v in yeast_out.items():
            try:
                if float(v) >= 0.5:
                    yeast_traits_active.append(k)
            except:
                if v == 1:
                    yeast_traits_active.append(k)
        if not yeast_traits_active:
            yeast_sorted = sorted(yeast_out.items(), key=lambda kv: kv[1], reverse=True)
            yeast_traits_active = [yeast_sorted[0][0]] if yeast_sorted else []

    # style guess heuristic
    style_guess = "Experimental / Hybrid"

    # quick / dumb rule-of-thumb classifier:
    if ("clean_neutral" in yeast_out and yeast_out["clean_neutral"] >= 0.5) and \
       ("dry_finish"   in yeast_out and yeast_out["dry_finish"]   >= 0.5):
        style_guess = "West Coast IPA / Modern IPA"

    if ("fruity_esters" in yeast_out and yeast_out["fruity_esters"] >= 0.5) and \
       ("tropical" in hop_out and hop_out["tropical"] >= 0.6):
        style_guess = "Hazy / NEIPA leaning"

    if ("phenolic_spicy" in yeast_out and yeast_out["phenolic_spicy"] >= 0.5):
        style_guess = "Belgian / Saison leaning"

    if ("caramel" in malt_out and malt_out["caramel"] >= 0.5):
        style_guess = "English / Malt-forward Ale"

    return {
        "hop_out": hop_out,
        "hop_top_notes": hop_top,
        "malt_traits": malt_traits_active,
        "yeast_traits": yeast_traits_active,
        "style_guess": style_guess
    }

# ---- RADAR PLOT ---------------------------------------------------
def plot_hop_radar(hop_profile, title="Hop Aroma Radar"):
    """
    Clean polar radar plot. If hop_profile is empty,
    use zeros for standard aroma dims.
    """

    if not hop_profile:
        # fallback default ordering
        hop_profile = {
            "tropical": 0.0,
            "citrus": 0.0,
            "fruity": 0.0,
            "resinous": 0.0,
            "floral": 0.0,
            "earthy": 0.0,
        }

    labels = list(hop_profile.keys())
    values = list(hop_profile.values())
    values_arr = np.array(values, dtype=float)

    # For polygon we repeat first value at end
    closed_vals = np.concatenate([values_arr, values_arr[:1]])
    n = len(labels)

    # Angles for each category (unique)
    base_angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    closed_angles = np.concatenate([base_angles, base_angles[:1]])

    fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))

    ax.plot(closed_angles, closed_vals, linewidth=2, color="#1f77b4")
    ax.fill(closed_angles, closed_vals, alpha=0.25, color="#1f77b4")

    # label each point with numeric
    for ang, val in zip(base_angles, values_arr):
        ax.text(
            ang,
            val,
            f"{val:.4f}",
            color="black",
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

    ax.set_xticks(base_angles)
    ax.set_xticklabels(labels, fontsize=12)

    # style radial grid
    ax.set_rlabel_position(0)
    ax.yaxis.grid(color="gray", linestyle="--", alpha=0.4)
    ax.xaxis.grid(color="gray", linestyle="--", alpha=0.4)

    ax.set_title(title, fontsize=28, fontweight="bold", pad=20)
    fig.tight_layout()
    return fig

# -----------------------------------------------------------------------------
# SIDEBAR INPUTS
# -----------------------------------------------------------------------------
st.sidebar.header("🍃 Hop Bill")

all_hops_sorted = sorted(get_all_hop_names(hop_features))

# pre-populate 4 hop slots
hop1_name = st.sidebar.selectbox("Hop 1", all_hops_sorted, key="hop1_name")
hop1_amt  = st.sidebar.number_input(f"{hop1_name} (g)", min_value=0.0, value=40.0, step=1.0, key="hop1_amt")

hop2_name = st.sidebar.selectbox("Hop 2", all_hops_sorted, key="hop2_name")
hop2_amt  = st.sidebar.number_input(f"{hop2_name} (g)", min_value=0.0, value=40.0, step=1.0, key="hop2_amt")

hop3_name = st.sidebar.selectbox("Hop 3", all_hops_sorted, key="hop3_name")
hop3_amt  = st.sidebar.number_input(f"{hop3_name} (g)", min_value=0.0, value=0.0, step=1.0, key="hop3_amt")

hop4_name = st.sidebar.selectbox("Hop 4", all_hops_sorted, key="hop4_name")
hop4_amt  = st.sidebar.number_input(f"{hop4_name} (g)", min_value=0.0, value=0.0, step=1.0, key="hop4_amt")

hop_bill = {
    hop1_name: hop1_amt,
    hop2_name: hop2_amt,
    hop3_name: hop3_amt,
    hop4_name: hop4_amt,
}

st.sidebar.header("🌾 Malt Bill")
malt_options = sorted(malt_df["PRODUCT NAME"].unique().tolist())

malt1_name = st.sidebar.selectbox("Malt 1", malt_options, key="malt1_name")
malt1_pct  = st.sidebar.number_input("Malt 1 %", min_value=0.0, max_value=100.0, value=70.0, step=1.0, key="malt1_pct")

malt2_name = st.sidebar.selectbox("Malt 2", malt_options, key="malt2_name")
malt2_pct  = st.sidebar.number_input("Malt 2 %", min_value=0.0, max_value=100.0, value=20.0, step=1.0, key="malt2_pct")

malt3_name = st.sidebar.selectbox("Malt 3", malt_options, key="malt3_name")
malt3_pct  = st.sidebar.number_input("Malt 3 %", min_value=0.0, max_value=100.0, value=10.0, step=1.0, key="malt3_pct")

malt_selections = [
    {"name": malt1_name, "pct": malt1_pct},
    {"name": malt2_name, "pct": malt2_pct},
    {"name": malt3_name, "pct": malt3_pct},
]

st.sidebar.header("🧬 Yeast")
yeast_options = sorted(yeast_df["Name"].dropna().unique().tolist())
chosen_yeast  = st.sidebar.selectbox("Yeast Strain", yeast_options, key="yeast_name")

run_button = st.sidebar.button("Predict Flavor 🧪")

# -----------------------------------------------------------------------------
# MAIN PAGE
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div style='display:flex; align-items:center; gap:0.5rem;'>
      <span style='font-size:2.5rem;'>🍺</span>
      <span style='font-size:2.5rem; font-weight:600; line-height:2.5rem;'>Beer Recipe Digital Twin</span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write(
    "Predict hop aroma, malt character, and fermentation profile using trained ML models."
)

# run inference only if button clicked
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

    # warning if no hops / cannot calculate
    no_hops = (not hop_profile) or all(abs(v) < 1e-9 for v in hop_profile.values())

    if no_hops:
        st.warning(
            "No recognizable hops in your bill "
            "(or total was 0 once normalized), so aroma prediction is basically flat.\n\n"
            "Pick hops from the list in the sidebar (open the '🌀 Model trained on hops:' "
            "section), then click **Predict Flavor 🧪** again."
        )

    # layout columns for radar + summary info
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
            st.write("*None / flat*")

        st.markdown("### Malt character:")
        if malt_traits:
            st.write(", ".join(malt_traits))
        else:
            st.write("*None detected*")

        st.markdown("### Yeast character:")
        if yeast_traits:
            st.write(", ".join(yeast_traits))
        else:
            st.write("*None detected*")

        st.markdown("### Style direction:")
        st.write(f"🧭 {style_guess}")

        # also show which hops they actually used (nonzero)
        used_hops = [h for h,a in hop_bill.items() if a and a > 0]
        if used_hops:
            st.markdown("### Hops used by the model:")
            st.write(", ".join(used_hops))

else:
    # initial view: just the safety/warning helper box
    st.warning(
        "No recognizable hops in your bill (or total was 0 once normalized), "
        "so aroma prediction is basically flat.\n\n"
        "Pick hops from the list in the sidebar (open the '🌀 Model trained on hops:' "
        "section), then click **Predict Flavor 🧪** again."
    )

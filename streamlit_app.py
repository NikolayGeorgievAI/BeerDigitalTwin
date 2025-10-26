import streamlit as st
import numpy as np
import pandas as pd
import joblib
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# -----------------------------------------------------------------------------
# STREAMLIT PAGE CONFIG
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Beer Recipe Digital Twin",
    page_icon="🍺",
    layout="wide"
)

# -----------------------------------------------------------------------------
# LOAD MODELS & DATA (CACHED)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_models_and_data():
    """
    Load once per session:
    - hop_aroma_model.joblib
    - malt_sensory_model.joblib
    - yeast_sensory_model.joblib
    - clean_malt_df.pkl
    - clean_yeast_df.pkl
    """

    # --- HOPS MODEL BUNDLE ---
    hop_bundle = joblib.load("hop_aroma_model.joblib")
    hop_model = hop_bundle["model"]
    hop_features = hop_bundle["feature_cols"]      # e.g. ["hop_Citra","hop_Mosaic",...]
    hop_dims = hop_bundle["aroma_dims"]            # e.g. ["tropical","citrus","resinous",...]

    # --- MALT MODEL BUNDLE ---
    malt_bundle = joblib.load("malt_sensory_model.joblib")
    malt_model = malt_bundle["model"]
    malt_features = malt_bundle["feature_cols"]    # chemistry / composition columns
    malt_dims = malt_bundle["flavor_cols"]         # predicted malt traits

    # --- YEAST MODEL BUNDLE ---
    yeast_bundle = joblib.load("yeast_sensory_model.joblib")
    yeast_model = yeast_bundle["model"]
    yeast_features = yeast_bundle["feature_cols"]  # e.g. ["Temp_avg_C","Flocculation_num","Attenuation_pct"]
    yeast_dims = yeast_bundle["flavor_cols"]       # predicted yeast traits

    # --- DATAFRAMES ---
    malt_df = pd.read_pickle("clean_malt_df.pkl")
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

#
# --- HOPS HELPERS ------------------------------------------------------------
#
def get_all_hop_names(hop_features):
    """
    hop_features looks like ["hop_Adeena","hop_Agnum",...].
    We want ["Adeena","Agnum",...]
    """
    return [c.replace("hop_", "") for c in hop_features]


def build_hop_feature_vector(hop_bill_dict, hop_features):
    """
    hop_bill_dict = {"Citra": 40, "Mosaic": 20, "Saaz": 10, ...}

    Output is a single 1 x N array aligned with hop_features order:
    e.g. for ["hop_Citra","hop_Mosaic","hop_Saaz"] → [[40,20,10]]
    Missing hops default to 0.
    """
    row = []
    for col in hop_features:
        hop_name = col.replace("hop_", "")
        row.append(hop_bill_dict.get(hop_name, 0))
    return np.array(row).reshape(1, -1)


def predict_hop_profile(hop_bill_dict, hop_model, hop_features, hop_dims):
    """
    Return dict of predicted aroma intensities:
    {"tropical":0.8,"citrus":0.6,...}
    """
    x = build_hop_feature_vector(hop_bill_dict, hop_features)
    y_pred = hop_model.predict(x)[0]
    return dict(zip(hop_dims, y_pred))


#
# --- MALT HELPERS ------------------------------------------------------------
#
def get_weighted_malt_vector(malt_selections, malt_df, malt_features):
    """
    malt_selections is a list of dicts:
    [
      {"name":"Maris Otter","pct":70},
      {"name":"Crystal 60L","pct":20},
      {"name":"Flaked Oats","pct":10},
    ]

    We build a weighted blend of malt_features, summing pct/100 * each malt's row.
    """
    blend_vec = np.zeros(len(malt_features), dtype=float)

    for item in malt_selections:
        malt_name = item["name"]
        pct = float(item["pct"])

        row = malt_df[malt_df["PRODUCT NAME"] == malt_name].head(1)
        if row.empty:
            continue

        vec = np.array([row.iloc[0][feat] for feat in malt_features], dtype=float)
        blend_vec += vec * (pct / 100.0)

    return blend_vec.reshape(1, -1)


def predict_malt_profile_from_blend(malt_selections, malt_model, malt_df, malt_features, malt_dims):
    """
    Returns dict of predicted malt traits like:
    {"bready":1,"caramel":0,"roasty":1,...}
    """
    x = get_weighted_malt_vector(malt_selections, malt_df, malt_features)
    y_pred = malt_model.predict(x)[0]
    return dict(zip(malt_dims, y_pred))


#
# --- YEAST HELPERS -----------------------------------------------------------
#
def get_yeast_feature_vector(yeast_name, yeast_df, yeast_features):
    """
    Build single-row input for the chosen yeast strain.
    We'll pull columns that the model expects from yeast_df.
    """
    row = yeast_df[yeast_df["Name"] == yeast_name].head(1)
    if row.empty:
        # fallback zeroes if yeast not found (shouldn't really happen)
        return np.zeros(len(yeast_features)).reshape(1, -1)

    # We'll grab them in yeast_features order if possible
    vals = []
    for feat in yeast_features:
        if feat in row.columns:
            vals.append(row.iloc[0][feat])
        else:
            # backward compat / legacy columns
            if feat == "Temp_avg_C" and "Temp_avg_C" in row.columns:
                vals.append(row.iloc[0]["Temp_avg_C"])
            elif feat == "Flocculation_num" and "Flocculation_num" in row.columns:
                vals.append(row.iloc[0]["Flocculation_num"])
            elif feat == "Attenuation_pct" and "Attenuation_pct" in row.columns:
                vals.append(row.iloc[0]["Attenuation_pct"])
            else:
                vals.append(0.0)
    return np.array(vals).reshape(1, -1)


def predict_yeast_profile(yeast_name, yeast_model, yeast_df, yeast_features, yeast_dims):
    """
    Returns dict of predicted yeast-driven traits:
    {"fruity_esters":1,"clean_neutral":0,"phenolic_spicy":1,...}
    """
    x = get_yeast_feature_vector(yeast_name, yeast_df, yeast_features)
    y_pred = yeast_model.predict(x)[0]
    return dict(zip(yeast_dims, y_pred))


#
# --- SUMMARY / STYLE GUESS ---------------------------------------------------
#
def summarize_beer(
    hop_bill_dict,
    malt_selections,
    yeast_name,
    hop_model, hop_features, hop_dims,
    malt_model, malt_df, malt_features, malt_dims,
    yeast_model, yeast_df, yeast_features, yeast_dims
):
    hop_out = predict_hop_profile(hop_bill_dict, hop_model, hop_features, hop_dims)
    malt_out = predict_malt_profile_from_blend(malt_selections, malt_model, malt_df, malt_features, malt_dims)
    yeast_out = predict_yeast_profile(yeast_name, yeast_model, yeast_df, yeast_features, yeast_dims)

    # --- Top hop notes (sort intensities descending, take top 2)
    hop_sorted = sorted(hop_out.items(), key=lambda kv: kv[1], reverse=True)
    top_hops = [f"{k} ({round(v, 2)})" for k, v in hop_sorted[:2]]

    # --- Malt traits that are "on"
    malt_active = [k for k, v in malt_out.items() if v == 1 or v is True]

    # --- Yeast traits that are "on"
    yeast_active = [k for k, v in yeast_out.items() if v == 1 or v is True]

    # --- A light style heuristic
    style_guess = "Experimental / Hybrid"

    # e.g. if yeast is clean & dry and hops are citrusy/resinous -> West Coast IPA lean
    if ("clean_neutral" in yeast_out and yeast_out["clean_neutral"] == 1
        and "dry_finish" in yeast_out and yeast_out["dry_finish"] == 1):
        has_citrus_or_resin = any(
            ("citrus" in name or "resin" in name)
            for name, val in hop_sorted[:2]
        )
        if has_citrus_or_resin:
            style_guess = "West Coast IPA / Modern IPA"
        else:
            style_guess = "Clean, dry ale"

    # If fruity esters + tropical hop
    if (
        ("fruity_esters" in yeast_out and yeast_out["fruity_esters"] == 1)
        and ("tropical" in hop_out and hop_out["tropical"] > 0.6)
    ):
        style_guess = "Hazy / NEIPA leaning"

    # Belgian / Saison lean
    if ("phenolic_spicy" in yeast_out and yeast_out["phenolic_spicy"] == 1):
        style_guess = "Belgian / Saison leaning"

    # Malt caramel flag
    if ("caramel" in malt_out and malt_out["caramel"] == 1):
        style_guess = "English / Malt-forward Ale"

    return {
        "hop_out": hop_out,
        "hop_top_notes": top_hops,
        "malt_traits": malt_active,
        "yeast_traits": yeast_active,
        "style_guess": style_guess,
    }


#
# --- RADAR PLOT --------------------------------------------------------------
#
def plot_hop_radar(hop_profile, title="Hop Aroma Radar"):
    """
    Radar chart of hop_profile dict.
    Keys = aroma dims (tropical, citrus, resinous, ...)
    Values = intensities (float)
    """
    # guard: if empty, fake a zero dict
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
    values_arr = np.array(values, dtype=float)

    # repeat first value to close polygon
    closed_values = np.concatenate([values_arr, values_arr[:1]])

    # set up angles
    n = len(labels)
    base_angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    closed_angles = np.concatenate([base_angles, base_angles[:1]])

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    ax.plot(closed_angles, closed_values, linewidth=2)
    ax.fill(closed_angles, closed_values, alpha=0.25)

    # place numeric labels
    for ang, val in zip(base_angles, values_arr):
        ax.text(
            ang,
            val,
            f"{val:.4f}",
            ha="center",
            va="center",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", lw=1),
        )

    # set category ticks
    ax.set_xticks(base_angles)
    ax.set_xticklabels(labels, fontsize=12)

    # nicer radial ticks
    ax.set_rlabel_position(0)
    ax.yaxis.grid(color="gray", linestyle="--", alpha=0.4)
    ax.xaxis.grid(color="gray", linestyle="--", alpha=0.4)

    # auto radial limit
    vmax = max(1.0, values_arr.max() * 1.2 if values_arr.size else 1.0)
    ax.set_ylim(0, vmax)

    ax.set_title(title, fontsize=24, fontweight="bold", pad=20)
    fig.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# SIDEBAR UI
# -----------------------------------------------------------------------------
st.sidebar.header("🌿 Hop Bill")

all_hops_sorted = sorted(get_all_hop_names(hop_features))

# four hop slots
hop1_name = st.sidebar.selectbox("Hop 1", all_hops_sorted, key="hop1_name")
hop1_amt = st.sidebar.number_input(
    f"{hop1_name} (g)", min_value=0.0, max_value=200.0, value=40.0, step=5.0, key="hop1_amt"
)

hop2_name = st.sidebar.selectbox(
    "Hop 2", all_hops_sorted, index=min(1, len(all_hops_sorted)-1), key="hop2_name"
)
hop2_amt = st.sidebar.number_input(
    f"{hop2_name} (g)", min_value=0.0, max_value=200.0, value=40.0, step=5.0, key="hop2_amt"
)

hop3_name = st.sidebar.selectbox(
    "Hop 3", all_hops_sorted, index=min(2, len(all_hops_sorted)-1), key="hop3_name"
)
hop3_amt = st.sidebar.number_input(
    f"{hop3_name} (g)", min_value=0.0, max_value=200.0, value=0.0, step=5.0, key="hop3_amt"
)

hop4_name = st.sidebar.selectbox(
    "Hop 4", all_hops_sorted, index=min(3, len(all_hops_sorted)-1), key="hop4_name"
)
hop4_amt = st.sidebar.number_input(
    f"{hop4_name} (g)", min_value=0.0, max_value=200.0, value=0.0, step=5.0, key="hop4_amt"
)

hop_bill = {
    hop1_name: hop1_amt,
    hop2_name: hop2_amt,
    hop3_name: hop3_amt,
    hop4_name: hop4_amt,
}

# --- Malt Bill ---
st.sidebar.header("🌾 Malt Bill")

malt_options = sorted(malt_df["PRODUCT NAME"].dropna().unique().tolist())

malt1_name = st.sidebar.selectbox("Malt 1", malt_options, key="malt1_name")
malt1_pct = st.sidebar.number_input("Malt 1 %", 0.0, 100.0, 70.0, 1.0, key="malt1_pct")

malt2_name = st.sidebar.selectbox("Malt 2", malt_options, key="malt2_name")
malt2_pct = st.sidebar.number_input("Malt 2 %", 0.0, 100.0, 20.0, 1.0, key="malt2_pct")

malt3_name = st.sidebar.selectbox("Malt 3", malt_options, key="malt3_name")
malt3_pct = st.sidebar.number_input("Malt 3 %", 0.0, 100.0, 10.0, 1.0, key="malt3_pct")

malt_selections = [
    {"name": malt1_name, "pct": malt1_pct},
    {"name": malt2_name, "pct": malt2_pct},
    {"name": malt3_name, "pct": malt3_pct},
]

# --- Yeast ---
st.sidebar.header("🧫 Yeast")

yeast_options = sorted(yeast_df["Name"].dropna().unique().tolist())
chosen_yeast = st.sidebar.selectbox("Yeast Strain", yeast_options, key="chosen_yeast")

run_button = st.sidebar.button("Predict Flavor 🧪")

# -----------------------------------------------------------------------------
# MAIN PAGE HEADER
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div style='display:flex; align-items:center; gap:0.5rem;'>
      <span style='font-size:2.5rem;'>🍺</span>
      <span style='font-size:2.5rem; font-weight:600; line-height:2.5rem;'>
        Beer Recipe Digital Twin
      </span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write(
    "Predict hop aroma, malt character, and fermentation profile using trained ML models."
)

# -----------------------------------------------------------------------------
# RUN / DISPLAY RESULTS
# -----------------------------------------------------------------------------
if run_button:
    # total hop mass to detect "no hops"
    total_hops = sum(v for v in hop_bill.values() if v > 0)

    if total_hops == 0:
        st.warning(
            "⚠️ No recognizable hops in your bill (or total was 0 once normalized), "
            "so aroma prediction is basically flat.\n\n"
            "Please enter nonzero hop weights, then click **Predict Flavor 🧪** again."
        )
        st.stop()

    summary = summarize_beer(
        hop_bill,
        malt_selections,
        chosen_yeast,
        hop_model, hop_features, hop_dims,
        malt_model, malt_df, malt_features, malt_dims,
        yeast_model, yeast_df, yeast_features, yeast_dims
    )

    hop_profile = summary["hop_out"]
    hop_notes = summary["hop_top_notes"]
    malt_traits = summary["malt_traits"]
    yeast_traits = summary["yeast_traits"]
    style_guess = summary["style_guess"]

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
        st.write(", ".join(malt_traits) if malt_traits else "*None detected*")

        st.markdown("### Yeast character:")
        st.write(", ".join(yeast_traits) if yeast_traits else "*None detected*")

        st.markdown("### Style direction:")
        st.write(f"🧭 {style_guess}")

        # display which hops actually had >0g
        used_hops = [h for h, a in hop_bill.items() if a and a > 0]
        if used_hops:
            st.markdown("### Hops used by the model:")
            st.write(", ".join(used_hops))

else:
    # nice friendly instructions before first run
    st.info(
        "👈 Build your hop bill (up to 4 hops, with nonzero grams), set malt bill (% grist), "
        "choose yeast, then click **Predict Flavor 🧪** in the sidebar."
    )

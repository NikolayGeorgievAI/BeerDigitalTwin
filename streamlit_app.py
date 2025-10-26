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
# CACHED LOADING
# -----------------------------------------------------------------------------
@st.cache_resource
def load_models_and_data():
    """
    Load models and cleaned dataframes once per session.
    We expect these files in the repo root (same folder as this app):
    - hop_aroma_model.joblib
    - malt_sensory_model.joblib
    - yeast_sensory_model.joblib
    - clean_malt_df.pkl
    - clean_yeast_df.pkl
    """

    hop_bundle   = joblib.load("hop_aroma_model.joblib")
    malt_bundle  = joblib.load("malt_sensory_model.joblib")
    yeast_bundle = joblib.load("yeast_sensory_model.joblib")

    # unpack hop model bundle
    hop_model     = hop_bundle["model"]
    hop_features  = hop_bundle["feature_cols"]  # e.g. ["hop_Citra","hop_Mosaic",...]
    hop_dims      = hop_bundle["aroma_dims"]    # e.g. ["tropical","citrus","fruity",...]

    # unpack malt model bundle
    malt_model    = malt_bundle["model"]
    malt_features = malt_bundle["feature_cols"] # numeric chemistry columns
    malt_dims     = malt_bundle["flavor_cols"]  # predicted traits / flags for malt

    # unpack yeast model bundle
    yeast_model     = yeast_bundle["model"]
    yeast_features  = yeast_bundle["feature_cols"] # e.g. Temp_avg_C, etc.
    yeast_dims      = yeast_bundle["flavor_cols"]  # predicted traits / flags for yeast

    # reference dataframes
    malt_df  = pd.read_pickle("clean_malt_df.pkl")
    yeast_df = pd.read_pickle("clean_yeast_df.pkl")

    return (
        hop_model, hop_features, hop_dims,
        malt_model, malt_features, malt_dims,
        yeast_model, yeast_features, yeast_dims,
        malt_df, yeast_df,
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
    # hop_features like ["hop_Citra","hop_Mosaic",...]
    # return ["Citra","Mosaic",...]
    return [c.replace("hop_", "") for c in hop_features]

def build_hop_features(hop_bill_dict, hop_features):
    """
    hop_bill_dict = {"Citra": 40, "Mosaic": 20, ...}
    We'll produce a numeric row of shape (1, len(hop_features)),
    where each column 'hop_Name' = grams (0 if not used).
    """
    row = []
    for col in hop_features:
        hop_name = col.replace("hop_", "")
        row.append(hop_bill_dict.get(hop_name, 0.0))
    x = np.array(row).reshape(1, -1)
    return x

def predict_hop_profile(hop_bill_dict, hop_model, hop_features, hop_dims):
    """
    Return dict aroma_dim -> float for all hop_dims, using the trained hop model.
    """
    x = build_hop_features(hop_bill_dict, hop_features)
    y_pred = hop_model.predict(x)[0]  # 1D array of predicted intensities
    return dict(zip(hop_dims, y_pred))


# ---- MALTS ----
def get_weighted_malt_vector(malt_selections, malt_df, malt_features):
    """
    malt_selections =
    [
      {"name":"Maris Otter","pct":70},
      {"name":"Crystal 60L","pct":20},
      {"name":"Flaked Oats","pct":10}
    ]

    We'll do a weighted average of each numeric malt feature by grist %.
    """
    blend_vec = np.zeros(len(malt_features), dtype=float)

    for item in malt_selections:
        malt_name = item["name"]
        pct       = float(item["pct"])

        # take first row where PRODUCT NAME == malt_name
        row = malt_df[malt_df["PRODUCT NAME"] == malt_name].head(1)
        if row.empty:
            continue

        vec = np.array([row.iloc[0][feat] for feat in malt_features], dtype=float)
        blend_vec += vec * (pct / 100.0)

    return blend_vec.reshape(1, -1)

def predict_malt_profile_from_blend(malt_selections, malt_model, malt_df, malt_features, malt_dims):
    x = get_weighted_malt_vector(malt_selections, malt_df, malt_features)
    y_pred = malt_model.predict(x)[0]  # often binary flags or intensities
    return dict(zip(malt_dims, y_pred))


# ---- YEAST ----
def get_yeast_feature_vector(yeast_name, yeast_df, yeast_features):
    """
    Create the single-row vector for the chosen yeast strain.
    We'll pull e.g. Temp_avg_C, Flocculation_num, Attenuation_pct
    (whatever yeast_features expects).
    """
    row = yeast_df[yeast_df["Name"] == yeast_name].head(1)
    if row.empty:
        return np.zeros(len(yeast_features)).reshape(1, -1)

    vec = []
    for feat in yeast_features:
        vec.append(float(row.iloc[0][feat]))
    return np.array(vec).reshape(1, -1)

def predict_yeast_profile(yeast_name, yeast_model, yeast_df, yeast_features, yeast_dims):
    x = get_yeast_feature_vector(yeast_name, yeast_df, yeast_features)
    y_pred = yeast_model.predict(x)[0]  # typically binary trait flags
    return dict(zip(yeast_dims, y_pred))


# ---- SUMMARY LOGIC ----
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

    # rank hop aroma dims descending:
    hop_sorted = sorted(hop_out.items(), key=lambda kv: kv[1], reverse=True)
    top_hops   = [f"{k} ({round(v, 2)})" for k, v in hop_sorted[:2]]

    # active malt traits (==1 or >0.5 etc.)
    malt_active  = [k for k,v in malt_out.items() if v == 1 or v > 0.5]

    # active yeast traits
    yeast_active = [k for k,v in yeast_out.items() if v == 1 or v > 0.5]

    # simple heuristic style guess
    style_guess = "Experimental / Hybrid"
    # example heuristics:
    if ("clean_neutral" in yeast_out and yeast_out["clean_neutral"] == 1) and \
       ("dry_finish" in yeast_out and yeast_out["dry_finish"] == 1):
        if any("citrus" in dim for dim, val in hop_sorted[:2]):
            style_guess = "West Coast IPA / Modern IPA"
        else:
            style_guess = "Clean, dry ale"

    if ("fruity_esters" in yeast_out and yeast_out["fruity_esters"] == 1) and \
       ("tropical" in hop_out and hop_out["tropical"] > 0.6):
        style_guess = "Hazy / NEIPA leaning"

    if ("phenolic_spicy" in yeast_out and yeast_out["phenolic_spicy"] == 1):
        style_guess = "Belgian / Saison leaning"

    if ("caramel" in malt_out and malt_out["caramel"] == 1) or \
       ("color_intensity" in malt_out and malt_out["color_intensity"] == 1):
        style_guess = "English / Malt-forward Ale"

    return {
        "hop_out": hop_out,
        "hop_top_notes": top_hops,
        "malt_traits": malt_active,
        "yeast_traits": yeast_active,
        "style_guess": style_guess
    }


# -----------------------------------------------------------------------------
# RADAR / SPIDER PLOT
# -----------------------------------------------------------------------------
def plot_hop_radar(hop_profile, title="Hop Aroma Radar"):
    """
    Classic radar/spider chart:
    - one axis per aroma dimension
    - polygon fill
    - radial gridlines (spider web)
    - compact figure so it fits nicely in a column
    """

    # Fallback if for some reason hop_profile is empty
    if not hop_profile:
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

    # Close the polygon
    values_closed = values + [values[0]]

    import numpy as np

    # Angles for each axis, and close the loop
    num_axes = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_axes, endpoint=False)
    angles_closed = list(angles) + [angles[0]]

    # Create polar subplot; smaller so it doesn't dominate
    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))

    # Draw polygon + fill
    ax.plot(angles_closed, values_closed, linewidth=2, color="#1f77b4")
    ax.fill(angles_closed, values_closed, color="#1f77b4", alpha=0.25)

    # Annotate numeric value at each axis
    for ang, val in zip(angles, values):
        ax.text(
            ang,
            val,
            f"{val:.2f}",
            ha="center",
            va="center",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#1f77b4", lw=0.8),
        )

    # Category labels
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=10)

    # Spider web grid
    ax.set_rlabel_position(0)
    ax.yaxis.grid(color="gray", linestyle="--", alpha=0.5)
    ax.xaxis.grid(color="gray", linestyle="--", alpha=0.5)

    # Autoscale radial limit nicely
    vmax = max(1.0, max(values) * 1.2 if values else 1.0)
    ax.set_ylim(0, vmax)

    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

    fig.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# SIDEBAR UI INPUTS
# -----------------------------------------------------------------------------

st.sidebar.title("🍺 Build Your Recipe")

# --- Hop selectors
st.sidebar.header("Hop Bill")
all_hops_sorted = sorted(get_all_hop_names(hop_features))

# We'll give 4 hop slots
hop1_name = st.sidebar.selectbox("Hop 1", all_hops_sorted, index=0 if len(all_hops_sorted) > 0 else None, key="hop1_name")
hop1_amt  = st.sidebar.number_input(f"{hop1_name} (g)", min_value=0.0, value=40.0, step=5.0, key="hop1_amt")

hop2_name = st.sidebar.selectbox("Hop 2", all_hops_sorted, index=1 if len(all_hops_sorted) > 1 else 0, key="hop2_name")
hop2_amt  = st.sidebar.number_input(f"{hop2_name} (g)", min_value=0.0, value=40.0, step=5.0, key="hop2_amt")

hop3_name = st.sidebar.selectbox("Hop 3", all_hops_sorted, index=2 if len(all_hops_sorted) > 2 else 0, key="hop3_name")
hop3_amt  = st.sidebar.number_input(f"{hop3_name} (g)", min_value=0.0, value=0.0, step=5.0, key="hop3_amt")

hop4_name = st.sidebar.selectbox("Hop 4", all_hops_sorted, index=3 if len(all_hops_sorted) > 3 else 0, key="hop4_name")
hop4_amt  = st.sidebar.number_input(f"{hop4_name} (g)", min_value=0.0, value=0.0, step=5.0, key="hop4_amt")

hop_bill = {
    hop1_name: hop1_amt,
    hop2_name: hop2_amt,
    hop3_name: hop3_amt,
    hop4_name: hop4_amt,
}

# --- Malt bill
st.sidebar.header("Malt Bill")

malt_options = sorted(malt_df["PRODUCT NAME"].dropna().unique().tolist())

malt1_name = st.sidebar.selectbox("Malt 1", malt_options, key="malt1_name")
malt1_pct  = st.sidebar.number_input("Malt 1 % of grist", min_value=0.0, max_value=100.0, value=70.0, step=1.0, key="malt1_pct")

malt2_name = st.sidebar.selectbox("Malt 2", malt_options, key="malt2_name")
malt2_pct  = st.sidebar.number_input("Malt 2 % of grist", min_value=0.0, max_value=100.0, value=20.0, step=1.0, key="malt2_pct")

malt3_name = st.sidebar.selectbox("Malt 3", malt_options, key="malt3_name")
malt3_pct  = st.sidebar.number_input("Malt 3 % of grist", min_value=0.0, max_value=100.0, value=10.0, step=1.0, key="malt3_pct")

malt_selections = [
    {"name": malt1_name, "pct": malt1_pct},
    {"name": malt2_name, "pct": malt2_pct},
    {"name": malt3_name, "pct": malt3_pct},
]

# --- Yeast
st.sidebar.header("Yeast")
yeast_options = sorted(yeast_df["Name"].dropna().unique().tolist())
chosen_yeast  = st.sidebar.selectbox("Yeast strain", yeast_options, key="chosen_yeast")

run_button = st.sidebar.button("Predict Flavor 🧪")


# -----------------------------------------------------------------------------
# MAIN PAGE BODY
# -----------------------------------------------------------------------------

st.markdown(
    """
    # 🍺 Beer Recipe Digital Twin
    Predict hop aroma, malt character, and fermentation profile using trained ML models.
    """,
)

if run_button:
    # run inference + summarize
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

    # layout: radar on left, summary on right
    col_plot, col_info = st.columns([1.3, 1])

    with col_plot:
        st.subheader("Hop Aroma Radar")
        fig = plot_hop_radar(hop_profile, title="Hop Aroma Radar")
        st.pyplot(fig, use_container_width=True)

    with col_info:
        st.subheader("Top hop notes:")
        if hop_notes:
            for n in hop_notes:
                st.write(f"- {n}")
        else:
            st.write("_No dominant hop note_")

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

        # which hops we actually used in nonzero amounts
        used_hops = [name for name, grams in hop_bill.items() if grams and grams > 0]
        if used_hops:
            st.subheader("Hops used by the model:")
            st.write(", ".join(used_hops))

else:
    # helpful instructions when we haven't predicted yet
    st.info(
        "👉 Build your hop bill (up to 4 hops, give them nonzero grams), "
        "set malt bill (% grist), choose yeast, then click **Predict Flavor 🧪** in the sidebar."
    )

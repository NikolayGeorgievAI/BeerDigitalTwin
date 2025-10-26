import streamlit as st
import numpy as np
import pandas as pd
import joblib
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


# ---------------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------------
st.set_page_config(
    page_title="Beer Recipe Digital Twin",
    page_icon="🍺",
    layout="wide"
)

# ---------------------------------------------------------------------------------
# CACHED LOADING OF MODELS + DATA
# ---------------------------------------------------------------------------------
@st.cache_resource
def load_models_and_data():
    """
    Load heavy assets once and cache for the Streamlit session.
    """

    # Load bundles
    hop_bundle   = joblib.load("hop_aroma_model.joblib")
    malt_bundle  = joblib.load("malt_sensory_model.joblib")
    yeast_bundle = joblib.load("yeast_sensory_model.joblib")

    # Unpack hop bundle
    hop_model      = hop_bundle["model"]
    hop_features   = hop_bundle["feature_cols"]   # ex: ["hop_Akoya", "hop_African Queen", ...]
    hop_dims       = hop_bundle["aroma_dims"]     # ex: ["tropical","citrus","resinous",...]

    # Unpack malt bundle
    malt_model     = malt_bundle["model"]
    malt_features  = malt_bundle["feature_cols"]
    malt_dims      = malt_bundle["flavor_cols"]

    # Unpack yeast bundle
    yeast_model    = yeast_bundle["model"]
    yeast_features = yeast_bundle["feature_cols"]
    yeast_dims     = yeast_bundle["flavor_cols"]

    # Load reference data
    malt_df  = pd.read_pickle("clean_malt_df.pkl")
    yeast_df = pd.read_pickle("clean_yeast_df.pkl")

    return (
        hop_model, hop_features, hop_dims,
        malt_model, malt_features, malt_dims,
        yeast_model, yeast_features, yeast_dims,
        malt_df, yeast_df
    )


# ---------------------------------------------------------------------------------
# HOP NAME MAPPING
# ---------------------------------------------------------------------------------
def build_hop_display_mapping(hop_features):
    """
    We have model feature columns like "hop_African Queen", "hop_Admiral", ...
    We show the user a clean dropdown name ("African Queen", "Admiral", ...).
    Then we map back from the user's selection to the full column name.

    Returns:
      display_names: sorted list of unique display labels for dropdown
      display_to_feature: dict like {"African Queen": "hop_African Queen", ...}
    """
    display_to_feature = {}
    for feat in hop_features:
        # Strip the "hop_" prefix to get the nice human name
        disp = feat.replace("hop_", "")
        display_to_feature[disp] = feat

    display_names = sorted(display_to_feature.keys())
    return display_names, display_to_feature


def build_hop_feature_vector(hop_bill_user_display, hop_features, display_to_feature):
    """
    hop_bill_user_display looks like:
      {"Akoya": 40, "African Queen": 60, ...}

    We need to output a 1 x N vector aligned with hop_features.
    If hop_features[i] == "hop_African Queen" and hop_bill_user_display["African Queen"] = 60,
    that position gets value 60. Otherwise 0.
    """
    vec = []
    for feat in hop_features:
        nice_name = feat.replace("hop_", "")
        amt = hop_bill_user_display.get(nice_name, 0)
        vec.append(amt)
    return np.array(vec).reshape(1, -1)


def predict_hop_profile(hop_bill_user_display,
                        hop_model, hop_features, hop_dims,
                        display_to_feature):
    """
    Build proper model input, run hop_model.predict,
    return {dimension: value}.
    """
    x = build_hop_feature_vector(hop_bill_user_display,
                                 hop_features,
                                 display_to_feature)

    y_pred = hop_model.predict(x)[0]  # numeric intensities per aroma dim
    return dict(zip(hop_dims, y_pred)), x  # also return x for debug


# ---------------------------------------------------------------------------------
# MALT HELPERS
# ---------------------------------------------------------------------------------
def get_weighted_malt_vector(malt_selections, malt_df, malt_features):
    """
    malt_selections is list of:
    [
      {"name": "...", "pct": 70.0},
      {"name": "...", "pct": 20.0},
      {"name": "...", "pct": 10.0},
    ]

    We'll create a weighted feature vector (1 x len(malt_features)).
    """
    blend_vec = np.zeros(len(malt_features), dtype=float)

    for item in malt_selections:
        malt_name = item["name"]
        pct       = float(item["pct"])

        row = malt_df[malt_df["PRODUCT NAME"] == malt_name].head(1)
        if row.empty:
            continue

        vec = [row.iloc[0][feat] for feat in malt_features]
        vec = np.array(vec, dtype=float)

        blend_vec += vec * (pct / 100.0)

    return blend_vec.reshape(1, -1)


def predict_malt_profile_from_blend(malt_selections,
                                    malt_model, malt_df, malt_features, malt_dims):
    x = get_weighted_malt_vector(malt_selections, malt_df, malt_features)
    y_pred = malt_model.predict(x)[0]  # typically 0/1 per flavor trait
    return dict(zip(malt_dims, y_pred))


# ---------------------------------------------------------------------------------
# YEAST HELPERS
# ---------------------------------------------------------------------------------
def get_yeast_feature_vector(yeast_name, yeast_df, yeast_features):
    """
    Build a row [Temp_avg_C, Flocculation_num, Attenuation_pct] for this strain.
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


def predict_yeast_profile(yeast_name,
                          yeast_model, yeast_df, yeast_features, yeast_dims):
    x = get_yeast_feature_vector(yeast_name, yeast_df, yeast_features)
    y_pred = yeast_model.predict(x)[0]  # typically 0/1
    return dict(zip(yeast_dims, y_pred))


# ---------------------------------------------------------------------------------
# COMBINE EVERYTHING
# ---------------------------------------------------------------------------------
def summarize_beer(
    hop_bill_user_display,
    malt_selections,
    yeast_name,
    hop_model, hop_features, hop_dims,
    malt_model, malt_df, malt_features, malt_dims,
    yeast_model, yeast_df, yeast_features, yeast_dims,
    display_to_feature
):
    hop_out, hop_model_input_vec = predict_hop_profile(
        hop_bill_user_display,
        hop_model, hop_features, hop_dims,
        display_to_feature
    )

    malt_out  = predict_malt_profile_from_blend(
        malt_selections, malt_model, malt_df, malt_features, malt_dims
    )

    yeast_out = predict_yeast_profile(
        yeast_name, yeast_model, yeast_df, yeast_features, yeast_dims
    )

    # Rank hop notes
    hop_sorted = sorted(hop_out.items(), key=lambda kv: kv[1], reverse=True)
    top_hops   = [f"{k} ({round(v, 2)})" for k, v in hop_sorted[:2]]

    # Active flavors
    malt_active  = [k for k,v in malt_out.items() if v == 1]
    yeast_active = [k for k,v in yeast_out.items() if v == 1]

    # Heuristic style
    style_guess = "Experimental / Hybrid"

    if ("clean_neutral" in yeast_out and yeast_out.get("clean_neutral",0) == 1
        and "dry_finish" in yeast_out and yeast_out.get("dry_finish",0) == 1):
        if any("citrus" in n[0] or "resin" in n[0] for n in hop_sorted[:2]):
            style_guess = "West Coast IPA / Modern IPA"
        else:
            style_guess = "Clean, dry ale"

    if (yeast_out.get("fruity_esters",0) == 1) and \
       (hop_out.get("tropical",0) > 0.6):
        style_guess = "Hazy / NEIPA leaning"

    if yeast_out.get("phenolic_spicy",0) == 1:
        style_guess = "Belgian / Saison leaning"

    if malt_out.get("caramel",0) == 1:
        style_guess = "English / Malt-forward Ale"

    return {
        "hop_out": hop_out,
        "hop_model_input_vec": hop_model_input_vec,
        "hop_top_notes": top_hops,
        "malt_traits": malt_active,
        "yeast_traits": yeast_active,
        "style_guess": style_guess
    }


# ---------------------------------------------------------------------------------
# RADAR PLOT
# ---------------------------------------------------------------------------------
def plot_hop_radar(hop_profile, title="Hop Aroma Radar"):
    """
    Create a radar (polar) chart with nice size, filled region, and labels.
    """
    labels = list(hop_profile.keys())
    values = list(hop_profile.values())

    # close loop
    labels_loop = labels + [labels[0]]
    values_loop = values + [values[0]]

    # angles
    ang = np.linspace(0, 2*np.pi, len(labels_loop), endpoint=False)

    fig = plt.figure(figsize=(6,6))
    ax  = fig.add_subplot(111, polar=True)

    # web
    ax.plot(ang, values_loop, color="#1f77b4", linewidth=2)
    ax.fill(ang, values_loop, alpha=0.25, color="#1f77b4")

    # ticks for axes (spokes)
    ax.set_xticks(ang[:-1])
    ax.set_xticklabels(labels, fontsize=12)

    # radial grid aesthetics
    ax.tick_params(axis='y', labelsize=10)
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.6)

    # annotate each point (not the closed loop last point)
    for angle, val in zip(ang[:-1], values):
        ax.text(
            angle,
            val,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.2",
                      fc="white", ec="#1f77b4", alpha=0.7)
        )

    ax.set_title(title, fontsize=22, fontweight="bold", pad=20)
    return fig


# ---------------------------------------------------------------------------------
# LOAD MODELS / DATA
# ---------------------------------------------------------------------------------
(
    hop_model, hop_features, hop_dims,
    malt_model, malt_features, malt_dims,
    yeast_model, yeast_features, yeast_dims,
    malt_df, yeast_df
) = load_models_and_data()

# Build hop name mapping for dropdowns
display_hop_names, display_to_feature = build_hop_display_mapping(hop_features)


# ---------------------------------------------------------------------------------
# SIDEBAR INPUTS
# ---------------------------------------------------------------------------------
st.sidebar.header("Hop Bill")

# hop 1
hop1_name = st.sidebar.selectbox(
    "Hop 1",
    display_hop_names,
    index=0 if len(display_hop_names) > 0 else None,
    key="hop1_name",
)
hop1_amt = st.sidebar.slider(f"{hop1_name} (g)", 0, 120, 40, 5, key="hop1_amt")

# hop 2
hop2_name = st.sidebar.selectbox(
    "Hop 2",
    display_hop_names,
    index=1 if len(display_hop_names) > 1 else 0,
    key="hop2_name",
)
hop2_amt = st.sidebar.slider(f"{hop2_name} (g)", 0, 120, 40, 5, key="hop2_amt")

# hop 3
hop3_name = st.sidebar.selectbox(
    "Hop 3",
    display_hop_names,
    index=2 if len(display_hop_names) > 2 else 0,
    key="hop3_name",
)
hop3_amt = st.sidebar.slider(f"{hop3_name} (g)", 0, 120, 40, 5, key="hop3_amt")

# hop 4
hop4_name = st.sidebar.selectbox(
    "Hop 4",
    display_hop_names,
    index=3 if len(display_hop_names) > 3 else 0,
    key="hop4_name",
)
hop4_amt = st.sidebar.slider(f"{hop4_name} (g)", 0, 120, 40, 5, key="hop4_amt")

# Build hop bill in *display-name keys*
hop_bill_user_display = {
    hop1_name: hop1_amt,
    hop2_name: hop2_amt,
    hop3_name: hop3_amt,
    hop4_name: hop4_amt,
}


# Malt bill
st.sidebar.header("Malt Bill")

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
st.sidebar.header("Yeast")
yeast_options = sorted(yeast_df["Name"].dropna().unique().tolist())
chosen_yeast  = st.sidebar.selectbox("Yeast Strain", yeast_options)

run_button = st.sidebar.button("Predict Flavor 🧪")


# ---------------------------------------------------------------------------------
# MAIN LAYOUT
# ---------------------------------------------------------------------------------
if run_button:
    summary = summarize_beer(
        hop_bill_user_display,
        malt_selections,
        chosen_yeast,
        hop_model, hop_features, hop_dims,
        malt_model, malt_df, malt_features, malt_dims,
        yeast_model, yeast_df, yeast_features, yeast_dims,
        display_to_feature
    )

    hop_profile   = summary["hop_out"]
    hop_notes     = summary["hop_top_notes"]
    malt_traits   = summary["malt_traits"]
    yeast_traits  = summary["yeast_traits"]
    style_guess   = summary["style_guess"]
    hop_vec_debug = summary["hop_model_input_vec"]

    # two-column layout: left = radar, right = summary text
    col1, col2 = st.columns([1.2, 1])

    with col1:
        fig = plot_hop_radar(hop_profile, title="Hop Aroma Radar")
        st.pyplot(fig)

    with col2:
        st.subheader("Top hop notes:")
        if hop_notes:
            for n in hop_notes:
                st.write("•", n)
        else:
            st.write("_no dominant hop note_")

        st.subheader("Malt character:")
        st.write(", ".join(malt_traits) if malt_traits else "_none detected_")

        st.subheader("Yeast character:")
        st.write(", ".join(yeast_traits) if yeast_traits else "_none detected_")

        st.subheader("Style direction:")
        st.write("🧭", style_guess)

    # Debug info (kept for you to sanity-check the vector isn't all zeros)
    with st.expander("Debug / Model I/O"):
        st.markdown("**Hop model input vector shape:**")
        st.write(hop_vec_debug.shape)

        st.markdown("**First 15 hop features (name ➜ value):**")
        debug_table = []
        for feat, val in zip(hop_features[:15], hop_vec_debug[0][:15]):
            debug_table.append({"feature_col": feat, "value": float(val)})
        st.dataframe(pd.DataFrame(debug_table))

        st.markdown("**Raw hop_model output (hop aroma intensities):**")
        st.write(hop_profile)

        st.markdown("**Malt traits (binary flags):**")
        st.write(malt_traits)

        st.markdown("**Yeast traits (binary flags):**")
        st.write(yeast_traits)

else:
    st.info("👈 Build your hop bill (set grams), malt bill (%), choose yeast, then click **Predict Flavor 🧪**.")


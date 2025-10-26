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
# HEADER
# ---------------------------------------------------------------------------------
st.markdown(
    """
    <div style="display:flex; align-items:flex-start; gap:0.5rem; flex-wrap:wrap;">
        <div style="font-size:1.6rem; font-weight:600; line-height:1.2;">
            🍺 Beer Recipe Digital Twin
        </div>
    </div>
    <div style="color:#666; font-size:0.8rem; margin-top:0.25rem;">
        Predict hop aroma, malt character, and fermentation profile using trained ML models.
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------------------------------
# CACHED LOADING OF MODELS + DATA
# ---------------------------------------------------------------------------------
@st.cache_resource
def load_models_and_data():
    """
    Load all heavy assets once, cache them for the session.
    """
    hop_bundle   = joblib.load("hop_aroma_model.joblib")
    malt_bundle  = joblib.load("malt_sensory_model.joblib")
    yeast_bundle = joblib.load("yeast_sensory_model.joblib")

    hop_model      = hop_bundle["model"]
    hop_features   = hop_bundle["feature_cols"]   # e.g. ['hop_Citra', 'hop_Mosaic', ...]
    hop_dims       = hop_bundle["aroma_dims"]     # e.g. ['tropical','citrus','resinous',...]

    malt_model     = malt_bundle["model"]
    malt_features  = malt_bundle["feature_cols"]  # e.g. numeric brewing specs
    malt_dims      = malt_bundle["flavor_cols"]   # e.g. ['bready','caramel','body_full',...]

    yeast_model    = yeast_bundle["model"]
    yeast_features = yeast_bundle["feature_cols"] # e.g. ['Temp_avg_C','Flocculation_num','Attenuation_pct']
    yeast_dims     = yeast_bundle["flavor_cols"]  # e.g. ['fruity_esters','phenolic_spicy','clean_neutral',...]

    malt_df  = pd.read_pickle("clean_malt_df.pkl")
    yeast_df = pd.read_pickle("clean_yeast_df.pkl")

    return (
        hop_model, hop_features, hop_dims,
        malt_model, malt_features, malt_dims,
        yeast_model, yeast_features, yeast_dims,
        malt_df, yeast_df
    )

# load cached objects
(
    hop_model, hop_features, hop_dims,
    malt_model, malt_features, malt_dims,
    yeast_model, yeast_features, yeast_dims,
    malt_df, yeast_df
) = load_models_and_data()

# ---------------------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------------------

# ---- HOPS ----
def get_all_hop_names(hop_features):
    """
    hop_features looks like ["hop_Citra", "hop_Mosaic", ...]
    We recover human-friendly hop names.
    """
    names = []
    for c in hop_features:
        n = c.replace("hop_", "")
        # undo common formatting so dropdowns look nice
        n = n.replace("_", " ").title()
        names.append(n)
    # unique, sorted
    return sorted(list(dict.fromkeys(names)))


def normalize_hop_name_for_feature(hop_name_ui):
    """
    User picks "Citra" or "African Queen" from dropdown.
    Model feature might be "hop_citra" or "hop_African_Queen" etc.

    We'll try to make a normalized key that matches how columns
    were named *after* the 'hop_' prefix.
    """
    return (
        hop_name_ui.strip()
        .lower()
        .replace(" ", "_")
    )


def build_hop_features(hop_bill_dict, hop_features):
    """
    hop_bill_dict like:
    {
        "Citra": 40,
        "Mosaic": 20,
        "African Queen": 10,
        "Admiral": 5
    }

    We construct a 1-row vector aligned with hop_features.
    If feature list has ["hop_citra","hop_mosaic","hop_african_queen",...]
    we map amounts accordingly.
    """
    # Normalize user hop names -> amount
    norm_bill = {
        normalize_hop_name_for_feature(name): amt
        for name, amt in hop_bill_dict.items()
    }

    row_vals = []
    for feat in hop_features:
        # remove "hop_" prefix then normalize to compare
        feat_hop = feat.replace("hop_", "")
        feat_norm = feat_hop.strip().lower().replace(" ", "_")
        val = norm_bill.get(feat_norm, 0)
        row_vals.append(val)

    return np.array(row_vals, dtype=float).reshape(1, -1)


def predict_hop_profile(hop_bill_dict, hop_model, hop_features, hop_dims):
    x = build_hop_features(hop_bill_dict, hop_features)
    y_pred = hop_model.predict(x)[0]  # numeric intensities per aroma dim
    return dict(zip(hop_dims, y_pred))


# ---- MALTS ----
def get_weighted_malt_vector(malt_selections, malt_df, malt_features):
    """
    malt_selections is a list of dicts like:
    [
      {"name": "Maris Otter", "pct": 70},
      {"name": "Crystal 60L", "pct": 20},
      {"name": "Flaked Oats", "pct": 10}
    ]

    We build a weighted blend of the malt_features using percentages.
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
    y_pred = malt_model.predict(x)[0]  # predicted trait flags (0/1 etc.)
    return dict(zip(malt_dims, y_pred))


# ---- YEAST ----
def get_yeast_feature_vector(yeast_name, yeast_df, yeast_features):
    """
    Build [Temp_avg_C, Flocculation_num, Attenuation_pct] row for the chosen strain.
    """
    row = yeast_df[yeast_df["Name"] == yeast_name].head(1)
    if row.empty:
        return np.zeros(len(yeast_features)).reshape(1, -1)

    vec = [
        row.iloc[0]["Temp_avg_C"],
        row.iloc[0]["Flocculation_num"],
        row.iloc[0]["Attenuation_pct"]
    ]
    return np.array(vec, dtype=float).reshape(1, -1)


def predict_yeast_profile(yeast_name, yeast_model, yeast_df, yeast_features, yeast_dims):
    x = get_yeast_feature_vector(yeast_name, yeast_df, yeast_features)
    y_pred = yeast_model.predict(x)[0]  # usually 0/1 flags
    return dict(zip(yeast_dims, y_pred))


# ---- COMBINE EVERYTHING ----
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

    # sort hop notes
    hop_sorted = sorted(hop_out.items(), key=lambda kv: kv[1], reverse=True)
    top_hops   = [f"{k} ({round(v, 2)})" for k, v in hop_sorted[:2]]

    # active malt traits
    malt_active  = [k for k,v in malt_out.items() if v == 1]

    # active yeast traits
    yeast_active = [k for k,v in yeast_out.items() if v == 1]

    # crude style heuristic
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

# ---------------------------------------------------------------------------------
# RADAR PLOT
# ---------------------------------------------------------------------------------
def plot_hop_radar(hop_profile, title="Hop Aroma Radar"):
    """
    Draw a smaller radar chart with readable text.
    """
    # labels and values
    labels = list(hop_profile.keys())
    values = list(hop_profile.values())

    # close the polygon
    labels_cycle = labels + [labels[0]]
    values_cycle = values + [values[0]]

    angles = np.linspace(0, 2*np.pi, len(labels_cycle), endpoint=False)

    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111, polar=True)

    ax.plot(angles, values_cycle, linewidth=2, color="#1f77b4")
    ax.fill(angles, values_cycle, color="#1f77b4", alpha=0.25)

    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=11)

    # nice light radial grid
    ax.set_rlabel_position(0)
    ax.tick_params(axis='y', labelsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)

    ax.set_title(title, fontsize=20, fontweight="bold", pad=20)

    # annotate each vertex (but skip the duplicate last)
    for ang, val in zip(angles[:-1], values):
        ax.annotate(
            f"{val:.4f}",
            xy=(ang, val),
            xytext=(5,5),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#1f77b4", lw=0.8, alpha=0.8),
            color="#1f77b4"
        )

    fig.tight_layout()
    return fig

# ---------------------------------------------------------------------------------
# SIDEBAR CONTROLS
# ---------------------------------------------------------------------------------
st.sidebar.header("Hop Bill")

all_hops = get_all_hop_names(hop_features)

hop1_name = st.sidebar.selectbox("Hop 1", all_hops, index=0 if len(all_hops)>0 else None, key="hop1_name")
hop1_amt  = st.sidebar.slider(f"{hop1_name} (g)", 0, 120, 40, 5, key="hop1_amt")

hop2_name = st.sidebar.selectbox("Hop 2", all_hops, index=1 if len(all_hops)>1 else 0, key="hop2_name")
hop2_amt  = st.sidebar.slider(f"{hop2_name} (g)", 0, 120, 40, 5, key="hop2_amt")

hop3_name = st.sidebar.selectbox("Hop 3", all_hops, index=2 if len(all_hops)>2 else 0, key="hop3_name")
hop3_amt  = st.sidebar.slider(f"{hop3_name} (g)", 0, 120, 40, 5, key="hop3_amt")

hop4_name = st.sidebar.selectbox("Hop 4", all_hops, index=3 if len(all_hops)>3 else 0, key="hop4_name")
hop4_amt  = st.sidebar.slider(f"{hop4_name} (g)", 0, 120, 40, 5, key="hop4_amt")

hop_bill = {
    hop1_name: hop1_amt,
    hop2_name: hop2_amt,
    hop3_name: hop3_amt,
    hop4_name: hop4_amt,
}

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

st.sidebar.header("Yeast")
yeast_options = sorted(yeast_df["Name"].dropna().unique().tolist())
chosen_yeast  = st.sidebar.selectbox("Yeast Strain", yeast_options)

run_button = st.sidebar.button("Predict Flavor 🧪")

# ---------------------------------------------------------------------------------
# MAIN PANEL
# ---------------------------------------------------------------------------------
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

    # LAYOUT: radar on left, text on right
    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        fig = plot_hop_radar(hop_profile, title="Hop Aroma Radar")
        st.pyplot(fig)

    with col_right:
        st.markdown("### Top hop notes:")
        if hop_notes:
            for n in hop_notes:
                st.write("•", n)
        else:
            st.write("_No dominant hop note_")

        st.markdown("### Malt character:")
        st.write(", ".join(malt_traits) if malt_traits else "—")

        st.markdown("### Yeast character:")
        st.write(", ".join(yeast_traits) if yeast_traits else "—")

        st.markdown("### Style direction:")
        st.write(f"🧭 {style_guess}")

    # ---------------------------------
    # DEBUG BLOCKS
    # ---------------------------------
    with st.expander("Debug / Model Outputs"):
        st.write("Hop profile (raw):", hop_profile)
        st.write("Malt traits (flags):", malt_traits)
        st.write("Yeast traits (flags):", yeast_traits)

        st.json({
            "hop_bill (grams)": hop_bill,
            "malt_bill (%)": malt_selections,
            "yeast": chosen_yeast
        })

    # EXTRA DEBUG: show model inputs/outputs so we can diagnose zeros
    st.markdown("### Debug: Hop model input/output check")
    x_debug = build_hop_features(hop_bill, hop_features)
    st.write("Input shape:", x_debug.shape)
    st.write("First 10 hop feature names:", hop_features[:10])
    st.write("First 10 hop feature values:", x_debug[0][:10])
    y_debug = hop_model.predict(x_debug)
    st.write("Raw hop_model output:", y_debug)

else:
    st.info("👈 Build your hop bill (up to 4 hops), malt bill (3 malts with %), choose yeast, then click **Predict Flavor 🧪**.")

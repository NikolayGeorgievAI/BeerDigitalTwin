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
    Load all heavy assets once, cache them for the session.
    """

    # Load the trained bundles
    hop_bundle   = joblib.load("hop_aroma_model.joblib")
    malt_bundle  = joblib.load("malt_sensory_model.joblib")
    yeast_bundle = joblib.load("yeast_sensory_model.joblib")

    # Unpack hop bundle
    hop_model      = hop_bundle["model"]
    hop_features   = hop_bundle["feature_cols"]   # e.g. ['hop_Citra', 'hop_Mosaic', ...]
    hop_dims       = hop_bundle["aroma_dims"]     # e.g. ['tropical','citrus','resinous',...]

    # Unpack malt bundle
    malt_model     = malt_bundle["model"]
    malt_features  = malt_bundle["feature_cols"]  # e.g. ['MOISTURE MAX','EXTRACT TYPICAL',...]
    malt_dims      = malt_bundle["flavor_cols"]   # e.g. ['bready','caramel','body_full',...]

    # Unpack yeast bundle
    yeast_model    = yeast_bundle["model"]
    yeast_features = yeast_bundle["feature_cols"] # e.g. ['Temp_avg_C','Flocculation_num','Attenuation_pct']
    yeast_dims     = yeast_bundle["flavor_cols"]  # e.g. ['fruity_esters','phenolic_spicy','clean_neutral',...]

    # Load cleaned reference tables
    malt_df  = pd.read_pickle("clean_malt_df.pkl")
    yeast_df = pd.read_pickle("clean_yeast_df.pkl")

    return (
        hop_model, hop_features, hop_dims,
        malt_model, malt_features, malt_dims,
        yeast_model, yeast_features, yeast_dims,
        malt_df, yeast_df
    )


# ---------------------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------------------

# ---- HOPS ----
def get_all_hop_names(hop_features):
    # hop_features looks like ["hop_Citra", "hop_Mosaic", ...]
    return [c.replace("hop_", "") for c in hop_features]

def build_hop_features(hop_bill_dict, hop_features):
    """
    hop_bill_dict = { "Citra": 40, "Mosaic": 20, ... }
    We convert to the feature order hop_features and return shape (1, -1)
    """
    row = []
    for col in hop_features:
        hop_name = col.replace("hop_", "")
        row.append(hop_bill_dict.get(hop_name, 0))
    return np.array(row).reshape(1, -1)

def predict_hop_profile(hop_bill_dict, hop_model, hop_features, hop_dims):
    x = build_hop_features(hop_bill_dict, hop_features)
    y_pred = hop_model.predict(x)[0]  # numeric intensities
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

    We build a weighted blend of the malt_features, expressed as % of grist.
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
    y_pred = malt_model.predict(x)[0]  # 0/1 flags for sensory traits
    return dict(zip(malt_dims, y_pred))


# ---- YEAST ----
def get_yeast_feature_vector(yeast_name, yeast_df, yeast_features):
    """
    Build a row [Temp_avg_C, Flocculation_num, Attenuation_pct] for the chosen strain.
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
    y_pred = yeast_model.predict(x)[0]  # 0/1 flags for traits
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
    # Predict each part
    hop_out   = predict_hop_profile(hop_bill_dict, hop_model, hop_features, hop_dims)
    malt_out  = predict_malt_profile_from_blend(malt_selections, malt_model, malt_df, malt_features, malt_dims)
    yeast_out = predict_yeast_profile(yeast_name, yeast_model, yeast_df, yeast_features, yeast_dims)

    # Rank hop notes
    hop_sorted = sorted(hop_out.items(), key=lambda kv: kv[1], reverse=True)
    top_hops   = [f"{k} ({round(v, 2)})" for k, v in hop_sorted[:2]]

    # Malt traits that fired
    malt_active  = [k for k,v in malt_out.items() if v == 1]

    # Yeast traits that fired
    yeast_active = [k for k,v in yeast_out.items() if v == 1]

    # Quick & dirty "style" heuristic
    style_guess = "Experimental / Hybrid"

    # West Coast-ish / dry clean ales
    if ("clean_neutral" in yeast_out and yeast_out["clean_neutral"] == 1
        and "dry_finish" in yeast_out and yeast_out["dry_finish"] == 1):
        if any("citrus" in n[0] or "resin" in n[0] for n in hop_sorted[:2]):
            style_guess = "West Coast IPA / Modern IPA"
        else:
            style_guess = "Clean, dry ale"

    # Hazy-ish
    if ("fruity_esters" in yeast_out and yeast_out["fruity_esters"] == 1) and \
       ("tropical" in hop_out and hop_out["tropical"] > 0.6):
        style_guess = "Hazy / NEIPA leaning"

    # Belgian-ish
    if ("phenolic_spicy" in yeast_out and yeast_out["phenolic_spicy"] == 1):
        style_guess = "Belgian / Saison leaning"

    # Malty English-ish
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
# PLOTTING (RADAR CHART WITH RESCALING)
# ---------------------------------------------------------------------------------
def plot_hop_radar(hop_profile, title="Hop Aroma Radar"):
    """
    Radar (spider) chart of hop aroma dimensions.

    - Scales tiny model outputs up visually so you ALWAYS get a readable shape,
      but still displays the true numeric values as text.
    - Spreads annotation labels out at the polygon tips (so they don't all pile
      up in the center).
    - Keeps a consistent outer radius for compare-ability.
    """

    labels = list(hop_profile.keys())
    vals   = np.array(list(hop_profile.values()), dtype=float)  # true model outputs

    num_axes = len(labels)
    angles = np.linspace(0, 2*np.pi, num_axes, endpoint=False).tolist()

    # --- build "display" values we actually plot (so the shape is visible) ---
    vmin_true = vals.min() if len(vals) else 0.0
    vmax_true = vals.max() if len(vals) else 0.0
    range_true = vmax_true - vmin_true

    low_target  = 0.2   # 20% radius
    high_target = 0.8   # 80% radius

    if range_true < 1e-6:
        # basically flat line: just make a small constant ring
        vals_display = np.full_like(vals, 0.5)  # halfway out for visibility
    else:
        # linear rescale
        vals_norm = (vals - vmin_true) / range_true  # 0..1
        vals_display = low_target + vals_norm * (high_target - low_target)

    # close polygons
    vals_display_closed = np.concatenate([vals_display, [vals_display[0]]])
    angles_closed       = angles + [angles[0]]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    fig.subplots_adjust(left=0.08, right=0.92, top=0.9, bottom=0.1)

    # Draw polygon
    ax.plot(angles_closed, vals_display_closed, linewidth=2, color="tab:blue")
    ax.fill(angles_closed, vals_display_closed, alpha=0.25, color="tab:blue")

    # Category labels around circle
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=11)

    # Fix radial frame 0..1 so charts are visually consistent
    ax.set_ylim(0, 1.0)

    # Nice spider grid
    ax.grid(True, linestyle="--", alpha=0.4)

    # Hide radial tick text (0.2, 0.4...) because it's rescaled
    ax.set_yticklabels([])
    ax.tick_params(axis="y", labelsize=0)

    # Annotate each spoke with the TRUE (unscaled) value
    for angle, disp_val, true_val in zip(angles, vals_display, vals):
        r_text = min(disp_val + 0.05, 0.95)
        ax.text(
            angle,
            r_text,
            f"{true_val:.2f}",
            fontsize=9,
            ha="center",
            va="center",
            bbox=dict(
                boxstyle="round,pad=0.2",
                fc="white",
                ec="gray",
                lw=0.5,
                alpha=0.8
            )
        )

    ax.set_title(title, size=18, weight="bold", pad=20)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------------
# SIDEBAR UI
# ---------------------------------------------------------------------------------

(
    hop_model, hop_features, hop_dims,
    malt_model, malt_features, malt_dims,
    yeast_model, yeast_features, yeast_dims,
    malt_df, yeast_df
) = load_models_and_data()

# HOP BILL INPUTS
st.sidebar.header("Hop Bill")

all_hops = sorted(get_all_hop_names(hop_features))

hop1_name = st.sidebar.selectbox("Hop 1", all_hops, index=0 if len(all_hops) > 0 else None, key="hop1_name")
hop1_amt  = st.sidebar.slider(f"{hop1_name} (g)", 0, 120, 40, 5, key="hop1_amt")

hop2_name = st.sidebar.selectbox("Hop 2", all_hops, index=1 if len(all_hops) > 1 else 0, key="hop2_name")
hop2_amt  = st.sidebar.slider(f"{hop2_name} (g)", 0, 120, 40, 5, key="hop2_amt")

hop3_name = st.sidebar.selectbox("Hop 3", all_hops, index=2 if len(all_hops) > 2 else 0, key="hop3_name")
hop3_amt  = st.sidebar.slider(f"{hop3_name} (g)", 0, 120, 40, 5, key="hop3_amt")

hop4_name = st.sidebar.selectbox("Hop 4", all_hops, index=3 if len(all_hops) > 3 else 0, key="hop4_name")
hop4_amt  = st.sidebar.slider(f"{hop4_name} (g)", 0, 120, 40, 5, key="hop4_amt")

hop_bill = {
    hop1_name: hop1_amt,
    hop2_name: hop2_amt,
    hop3_name: hop3_amt,
    hop4_name: hop4_amt,
}

# MALT BILL INPUTS
st.sidebar.header("Malt Bill")

malt_options = sorted(malt_df["PRODUCT NAME"].unique().tolist())

malt1_name = st.sidebar.selectbox("Malt 1", malt_options, key="malt1_name")
malt1_pct  = st.sidebar.number_input(
    "Malt 1 %",
    min_value=0.0, max_value=100.0,
    value=70.0, step=1.0,
    key="malt1_pct"
)

malt2_name = st.sidebar.selectbox("Malt 2", malt_options, key="malt2_name")
malt2_pct  = st.sidebar.number_input(
    "Malt 2 %",
    min_value=0.0, max_value=100.0,
    value=20.0, step=1.0,
    key="malt2_pct"
)

malt3_name = st.sidebar.selectbox("Malt 3", malt_options, key="malt3_name")
malt3_pct  = st.sidebar.number_input(
    "Malt 3 %",
    min_value=0.0, max_value=100.0,
    value=10.0, step=1.0,
    key="malt3_pct"
)

malt_selections = [
    {"name": malt1_name, "pct": malt1_pct},
    {"name": malt2_name, "pct": malt2_pct},
    {"name": malt3_name, "pct": malt3_pct},
]

# YEAST INPUT
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

    # layout: big chart left, interpretation right
    left_col, right_col = st.columns([3, 1])

    with left_col:
        fig = plot_hop_radar(hop_profile, title="Hop Aroma Radar")
        st.pyplot(fig)

        st.markdown("#### Top hop notes:")
        if hop_notes:
            for n in hop_notes:
                st.write("•", n)
        else:
            st.write("_No dominant hop note_")

    with right_col:
        st.markdown("#### Malt character:")
        st.write(", ".join(malt_traits) if malt_traits else "None detected")

        st.markdown("#### Yeast character:")
        st.write(", ".join(yeast_traits) if yeast_traits else "None detected")

        st.markdown("#### Style direction:")
        st.write(f"🧭 {style_guess}")

    with st.expander("Debug / Model Outputs"):
        st.write("Hop profile (raw model outputs):", hop_profile)
        st.write("Malt traits (flags):", malt_traits)
        st.write("Yeast traits (flags):", yeast_traits)
        st.json({
            "hop_bill (grams)": hop_bill,
            "malt_bill (%)": malt_selections,
            "yeast": chosen_yeast
        })

else:
    st.title("🍺 Beer Recipe Digital Twin")
    st.caption("Predict hop aroma, malt character, and fermentation profile using trained ML models.")
    st.info("👈 Build your hop bill (up to 4 hops), malt bill (3 malts with %), choose yeast, then click **Predict Flavor 🧪**.")

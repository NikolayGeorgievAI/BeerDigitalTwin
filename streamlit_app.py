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

st.title("🍺 Beer Recipe Digital Twin")
st.caption("Predict hop aroma, malt character, and fermentation profile using trained ML models.")


# ---------------------------------------------------------------------------------
# CACHED MODEL / DATA LOAD
# ---------------------------------------------------------------------------------
@st.cache_resource
def load_models_and_data():
    """
    Load and cache all heavy assets one time.
    """

    # Load trained bundles
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

    # Reference tables
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
    """
    hop_features looks like ["hop_Citra", "hop_Mosaic", ...]
    Return ["Citra", "Mosaic", ...]
    """
    return [c.replace("hop_", "") for c in hop_features]

def build_hop_features(hop_bill_dict, hop_features):
    """
    hop_bill_dict = { "Citra": 40, "Mosaic": 20, ... }  # grams
    Convert to a vector matching hop_features order.
    """
    row = []
    for col in hop_features:
        hop_name = col.replace("hop_", "")
        row.append(hop_bill_dict.get(hop_name, 0))
    X = np.array(row).reshape(1, -1)
    return X

def predict_hop_profile(hop_bill_dict, hop_model, hop_features, hop_dims):
    """
    Returns dict like:
    {
      'tropical': 0.12,
      'citrus': 0.05,
      ...
    }
    """
    X = build_hop_features(hop_bill_dict, hop_features)
    y_pred = hop_model.predict(X)[0]
    # Ensure numeric floats / not np types for plotting later
    hop_out = {dim: float(val) for dim, val in zip(hop_dims, y_pred)}
    return hop_out, X


# ---- MALTS ----
def get_weighted_malt_vector(malt_selections, malt_df, malt_features):
    """
    malt_selections = [
      {"name": "Maris Otter", "pct": 70},
      {"name": "Crystal 60L", "pct": 20},
      {"name": "Flaked Oats", "pct": 10},
    ]

    We build a weighted blend of the malt_features columns,
    weighted by pct / 100.
    """
    blend_vec = np.zeros(len(malt_features), dtype=float)

    for item in malt_selections:
        malt_name = item["name"]
        pct       = float(item["pct"])

        row = malt_df[malt_df["PRODUCT NAME"] == malt_name].head(1)
        if row.empty:
            continue

        vals = np.array([row.iloc[0][feat] for feat in malt_features], dtype=float)
        blend_vec += vals * (pct / 100.0)

    return blend_vec.reshape(1, -1)

def predict_malt_profile_from_blend(malt_selections, malt_model, malt_df, malt_features, malt_dims):
    """
    Returns dict like:
    {
      'bready': 1,
      'caramel': 0,
      ...
    }
    where 1 means that malt character is predicted "on".
    """
    X = get_weighted_malt_vector(malt_selections, malt_df, malt_features)
    y_pred = malt_model.predict(X)[0]
    malt_out = {dim: int(val) for dim, val in zip(malt_dims, y_pred)}
    return malt_out


# ---- YEAST ----
def get_yeast_feature_vector(yeast_name, yeast_df, yeast_features):
    """
    Build row [Temp_avg_C, Flocculation_num, Attenuation_pct]
    (or whatever yeast_features are).
    """
    row = yeast_df[yeast_df["Name"] == yeast_name].head(1)
    if row.empty:
        return np.zeros(len(yeast_features)).reshape(1, -1)

    vec = []
    for feat in yeast_features:
        vec.append(row.iloc[0][feat])
    X = np.array(vec, dtype=float).reshape(1, -1)
    return X

def predict_yeast_profile(yeast_name, yeast_model, yeast_df, yeast_features, yeast_dims):
    """
    Returns dict like:
    {
      'fruity_esters': 1,
      'phenolic_spicy': 0,
      'clean_neutral': 1,
      ...
    }
    """
    X = get_yeast_feature_vector(yeast_name, yeast_df, yeast_features)
    y_pred = yeast_model.predict(X)[0]
    yeast_out = {dim: int(val) for dim, val in zip(yeast_dims, y_pred)}
    return yeast_out


# ---- STYLE GUESS ----
def summarize_beer(
    hop_bill_dict,
    malt_selections,
    yeast_name,
    hop_model, hop_features, hop_dims,
    malt_model, malt_df, malt_features, malt_dims,
    yeast_model, yeast_df, yeast_features, yeast_dims,
):
    # Predict each part
    hop_out, hop_input_vec = predict_hop_profile(hop_bill_dict, hop_model, hop_features, hop_dims)
    malt_out               = predict_malt_profile_from_blend(malt_selections, malt_model, malt_df, malt_features, malt_dims)
    yeast_out              = predict_yeast_profile(yeast_name, yeast_model, yeast_df, yeast_features, yeast_dims)

    # Rank hop notes: sort by intensity descending
    hop_sorted = sorted(hop_out.items(), key=lambda kv: kv[1], reverse=True)
    top_hops   = [f"{k} ({round(v, 2)})" for k, v in hop_sorted[:2]]

    # Malt traits that fired
    malt_active  = [k for k,v in malt_out.items() if v == 1]

    # Yeast traits that fired
    yeast_active = [k for k,v in yeast_out.items() if v == 1]

    # Quick "style" heuristic
    style_guess = "Experimental / Hybrid"

    # West Coast-ish / dry clean ales
    if ("clean_neutral" in yeast_out and yeast_out["clean_neutral"] == 1
        and "dry_finish" in yeast_out and yeast_out.get("dry_finish", 0) == 1):
        if any(("citrus" in n[0] or "resin" in n[0]) for n in hop_sorted[:2]):
            style_guess = "West Coast IPA / Modern IPA"
        else:
            style_guess = "Clean, dry ale"

    # Hazy-ish / NEIPA-ish
    if (yeast_out.get("fruity_esters", 0) == 1) and \
       (hop_out.get("tropical", 0) > 0.6):
        style_guess = "Hazy / NEIPA leaning"

    # Belgian-ish
    if yeast_out.get("phenolic_spicy", 0) == 1:
        style_guess = "Belgian / Saison leaning"

    # Malty English-ish
    if malt_out.get("caramel", 0) == 1:
        style_guess = "English / Malt-forward Ale"

    return {
        "hop_out": hop_out,
        "hop_top_notes": top_hops,
        "malt_traits": malt_active,
        "yeast_traits": yeast_active,
        "style_guess": style_guess,
        "hop_input_vec": hop_input_vec,  # for debugging
    }


# ---------------------------------------------------------------------------------
# RADAR PLOT (FIXED VERSION)
# ---------------------------------------------------------------------------------
def plot_hop_radar(hop_profile, title="Hop Aroma Radar"):
    """
    Draw a radar (spider) chart for hop aroma intensities.

    hop_profile example:
        {"tropical":0.12, "citrus":0.05, "resinous":0.03, ...}
    """

    import numpy as np
    import matplotlib.pyplot as plt

    # 1. Stable order
    labels = list(hop_profile.keys())   # e.g. ['tropical','citrus','resinous',...]
    vals   = np.array(list(hop_profile.values()), dtype=float)
    n      = len(labels)

    if n == 0:
        # Avoid crash if somehow empty
        fig = plt.figure(figsize=(4,4), dpi=150)
        ax = fig.add_subplot(111)
        ax.text(0.5,0.5,"No hop data",ha="center",va="center")
        ax.axis("off")
        return fig

    # 2. Base angles for each REAL dimension
    base_angles = np.linspace(0, 2 * np.pi, n, endpoint=False)

    # 3. Closed data for polygon fill (repeat first point)
    closed_vals    = np.concatenate([vals, vals[:1]])
    closed_angles  = np.concatenate([base_angles, base_angles[:1]])

    # 4. Build fig
    fig = plt.figure(figsize=(8,8), dpi=150)
    ax  = fig.add_subplot(111, polar=True)

    # 5. Draw polygon
    ax.plot(closed_angles, closed_vals, color="#1f77b4", linewidth=2)
    ax.fill(closed_angles, closed_vals, color="#1f77b4", alpha=0.25)

    # 6. Set ticks ONLY on the real (non-closed) angles
    ax.set_xticks(base_angles)
    ax.set_xticklabels(labels, fontsize=12)

    # 7. Radial grid styling
    ax.set_rlabel_position(0)
    ax.tick_params(axis="y", labelsize=10)
    ax.grid(True, linestyle="--", color="gray", alpha=0.4)

    # 8. Add numeric labels near each vertex
    for ang, val in zip(base_angles, vals):
        ax.text(
            ang,
            val,
            f"{val:.4f}",
            fontsize=10,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#1f77b4", lw=1, alpha=0.8),
        )

    # 9. Title
    ax.set_title(title, fontsize=28, fontweight="bold", pad=20)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------------
# MAIN APP UI
# ---------------------------------------------------------------------------------

# Load resources
(
    hop_model, hop_features, hop_dims,
    malt_model, malt_features, malt_dims,
    yeast_model, yeast_features, yeast_dims,
    malt_df, yeast_df
) = load_models_and_data()

# --- Sidebar: Hop Bill --------------------------------------------------------
st.sidebar.header("Hop Bill")

all_hops = sorted(get_all_hop_names(hop_features))

# Up to 4 hops with grams each
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


# --- Sidebar: Malt Bill (3-part blend) ----------------------------------------
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


# --- Sidebar: Yeast -----------------------------------------------------------
st.sidebar.header("Yeast")
yeast_options = sorted(yeast_df["Name"].dropna().unique().tolist())
chosen_yeast  = st.sidebar.selectbox("Yeast Strain", yeast_options)

# Predict button
run_button = st.sidebar.button("Predict Flavor 🧪")


# --- Main Panel ---------------------------------------------------------------
if run_button:
    summary = summarize_beer(
        hop_bill,
        malt_selections,
        chosen_yeast,
        hop_model, hop_features, hop_dims,
        malt_model, malt_df, malt_features, malt_dims,
        yeast_model, yeast_df, yeast_features, yeast_dims
    )

    hop_profile   = summary["hop_out"]          # dict aroma_dim -> float
    hop_notes     = summary["hop_top_notes"]
    malt_traits   = summary["malt_traits"]
    yeast_traits  = summary["yeast_traits"]
    style_guess   = summary["style_guess"]

    # Layout: radar left / text right
    col_left, col_right = st.columns([2, 1], vertical_alignment="top")

    with col_left:
        fig = plot_hop_radar(hop_profile, title="Hop Aroma Radar")
        st.pyplot(fig, use_container_width=True)

    with col_right:
        st.subheader("Top hop notes:")
        if hop_notes:
            for n in hop_notes:
                st.write(f"• {n}")
        else:
            st.write("• (none)")

        st.subheader("Malt character:")
        if malt_traits:
            st.write(", ".join(malt_traits))
        else:
            st.write("none")

        st.subheader("Yeast character:")
        if yeast_traits:
            st.write(", ".join(yeast_traits))
        else:
            st.write("none")

        st.subheader("Style direction:")
        st.write(f"🍺 {style_guess}")

    # Debug expander
    with st.expander("Debug / Model Outputs"):
        st.write("Hop profile (raw):", hop_profile)
        st.write("Malt traits (flags):", malt_traits)
        st.write("Yeast traits (flags):", yeast_traits)
        st.json({
            "hop_bill (grams)": hop_bill,
            "malt_bill (%)": malt_selections,
            "yeast": chosen_yeast
        })

else:
    st.info(
        "👈 Build your hop bill (up to 4 hops), malt bill (3 malts with %), "
        "choose yeast, then click **Predict Flavor 🧪**."
    )

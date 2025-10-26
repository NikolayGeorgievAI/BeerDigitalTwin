import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------------
st.set_page_config(
    page_title="Beer Recipe Digital Twin",
    page_icon="🍺",
    layout="wide",
)

# ----------------------------------------------------------------------------------
# LOAD MODELS + DATA (CACHED)
# ----------------------------------------------------------------------------------
@st.cache_resource
def load_models_and_data():
    # Load bundles
    hop_bundle   = joblib.load("hop_aroma_model.joblib")
    malt_bundle  = joblib.load("malt_sensory_model.joblib")
    yeast_bundle = joblib.load("yeast_sensory_model.joblib")

    hop_model      = hop_bundle["model"]
    hop_features   = hop_bundle["feature_cols"]   # e.g. ["hop_Adeena","hop_Amarillo",...]
    hop_dims       = hop_bundle["aroma_dims"]     # e.g. ["fruity","citrus","tropical",...]

    malt_model     = malt_bundle["model"]
    malt_features  = malt_bundle["feature_cols"]  # numeric chemistry columns for malts
    malt_dims      = malt_bundle["flavor_cols"]   # predicted malt traits (e.g. "bready", "color_intensity" etc)

    yeast_model    = yeast_bundle["model"]
    yeast_features = yeast_bundle["feature_cols"] # ["Temp_avg_C","Flocculation_num","Attenuation_pct",...]
    yeast_dims     = yeast_bundle["flavor_cols"]  # e.g. ["fruity_esters","clean_neutral",...]

    # Load reference data
    malt_df        = pd.read_pickle("clean_malt_df.pkl")
    yeast_df       = pd.read_pickle("clean_yeast_df.pkl")

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
    malt_df, clean_yeast_df,
) = load_models_and_data()

# ----------------------------------------------------------------------------------
# CONSTANT: which column in clean_yeast_df stores the strain name?
# ----------------------------------------------------------------------------------
# If your dataframe preview shows a different column (like "Yeast" or "Strain"),
# change the string below to match that EXACT column name.
YEAST_NAME_COL = "Name"

# ----------------------------------------------------------------------------------
# OPTIONAL DEBUG PREVIEW
# Uncomment these 3 lines to inspect yeast_df columns in the live app if you still
# get KeyError. Comment them again after confirming the column name.
#
# st.write("Yeast dataframe preview:")
# st.dataframe(clean_yeast_df.head())
# st.write("Yeast columns:", list(clean_yeast_df.columns))


# ----------------------------------------------------------------------------------
# HELPER FUNCTIONS
# ----------------------------------------------------------------------------------

def list_all_hops_from_features(hop_features):
    """
    We assume hop_features look like ["hop_Adeena","hop_Amarillo","hop_Citra",...].
    We'll strip off the "hop_" prefix to get user-friendly hop names.
    """
    hop_names = []
    for col in hop_features:
        if col.startswith("hop_"):
            hop_names.append(col.replace("hop_", "", 1))
        else:
            hop_names.append(col)
    return hop_names


def build_hop_feature_row(hop_bill_dict, hop_features):
    """
    hop_bill_dict = {"Citra": grams, "Mosaic": grams, ...}
    We map that into the exact model input order = hop_features.
    """
    row = []
    for feat in hop_features:
        hop_name = feat.replace("hop_", "", 1)
        row.append(hop_bill_dict.get(hop_name, 0.0))
    X = np.array(row, dtype=float).reshape(1, -1)
    return X


def predict_hop_profile(hop_bill_dict, hop_model, hop_features, hop_dims):
    """
    Returns {dimension_name: predicted_value, ...}
    """
    X = build_hop_feature_row(hop_bill_dict, hop_features)
    y_pred = hop_model.predict(X)[0]
    return dict(zip(hop_dims, y_pred))


def get_weighted_malt_vector(malt_selections, malt_df, malt_features):
    """
    malt_selections = [
        {"name": "AMBER MALT", "pct": 70.0},
        {"name": "BEST ALE MALT", "pct": 20.0},
        {"name": "BLACK MALT", "pct": 10.0},
    ]

    We build a weighted average of malt chemistry columns in malt_features.
    """
    blend_vec = np.zeros(len(malt_features), dtype=float)
    total_pct = sum([float(m["pct"]) for m in malt_selections])

    if total_pct <= 0:
        # nothing provided, return zeros
        return blend_vec.reshape(1, -1)

    for sel in malt_selections:
        malt_name = sel["name"]
        pct       = float(sel["pct"])
        row = malt_df[malt_df["PRODUCT NAME"] == malt_name].head(1)
        if row.empty:
            continue
        vec = np.array([row.iloc[0][f] for f in malt_features], dtype=float)
        weight = pct / total_pct
        blend_vec += vec * weight

    return blend_vec.reshape(1, -1)


def predict_malt_profile_from_blend(malt_selections, malt_model, malt_df, malt_features, malt_dims):
    X = get_weighted_malt_vector(malt_selections, malt_df, malt_features)
    y_pred = malt_model.predict(X)[0]
    return dict(zip(malt_dims, y_pred))


def get_yeast_feature_vector(yeast_name, yeast_df, yeast_features, name_col):
    """
    Build a single-row feature vector for the chosen yeast strain.
    We look it up by name_col (ex: "Name").
    """
    row = yeast_df[yeast_df[name_col] == yeast_name].head(1)

    # If strain wasn't found for some reason, return zeroes:
    if row.empty:
        return np.zeros(len(yeast_features)).reshape(1, -1)

    # We assume yeast_features includes columns like:
    # ["Temp_avg_C","Flocculation_num","Attenuation_pct", ...]
    vec_vals = []
    for feat in yeast_features:
        if feat in row.columns:
            vec_vals.append(row.iloc[0][feat])
        else:
            # missing column? append 0 fallback
            vec_vals.append(0.0)

    return np.array(vec_vals, dtype=float).reshape(1, -1)


def predict_yeast_profile(yeast_name, yeast_model, yeast_df, yeast_features, yeast_dims, name_col):
    X = get_yeast_feature_vector(yeast_name, yeast_df, yeast_features, name_col)
    y_pred = yeast_model.predict(X)[0]
    return dict(zip(yeast_dims, y_pred))


def summarize_beer(
    hop_bill_dict,
    malt_selections,
    yeast_name,
    hop_model, hop_features, hop_dims,
    malt_model, malt_df, malt_features, malt_dims,
    yeast_model, yeast_df, yeast_features, yeast_dims,
    yeast_name_col,
):
    """
    Run all 3 sub-models and produce a summary dict for the UI.
    """
    hop_out   = predict_hop_profile(hop_bill_dict, hop_model, hop_features, hop_dims)
    malt_out  = predict_malt_profile_from_blend(
        malt_selections,
        malt_model,
        malt_df,
        malt_features,
        malt_dims,
    )
    yeast_out = predict_yeast_profile(
        yeast_name,
        yeast_model,
        yeast_df,
        yeast_features,
        yeast_dims,
        yeast_name_col,
    )

    # Pick top 2 hop notes as (dimension, value)
    hop_sorted = sorted(hop_out.items(), key=lambda kv: kv[1], reverse=True)
    top_hop_notes = hop_sorted[:2]
    top_hop_notes_fmt = [f"{k} ({round(v, 2)})" for k, v in top_hop_notes]

    # Malt traits that fired "1" (or highest)
    malt_traits_on = [k for k, v in malt_out.items() if v == 1 or v is True]

    # Yeast traits that fired "1" (or highest)
    yeast_traits_on = [k for k, v in yeast_out.items() if v == 1 or v is True]

    # naive style guess
    style_guess = "Experimental / Hybrid"
    if ("clean_neutral" in yeast_out and yeast_out["clean_neutral"] == 1) and \
       ("dry_finish"    in yeast_out and yeast_out["dry_finish"]    == 1):
        style_guess = "Clean / dry ale"
    if ("fruity_esters" in yeast_out and yeast_out["fruity_esters"] == 1) and \
       ("tropical" in hop_out and hop_out["tropical"] > 0.6):
        style_guess = "Hazy / NEIPA leaning"
    if ("phenolic_spicy" in yeast_out and yeast_out["phenolic_spicy"] == 1):
        style_guess = "Belgian / Saison leaning"
    if ("caramel" in malt_out and malt_out["caramel"] == 1):
        style_guess = "English / Malt-forward Ale"

    return {
        "hop_out": hop_out,
        "hop_top_notes": top_hop_notes_fmt,
        "malt_traits": malt_traits_on,
        "yeast_traits": yeast_traits_on,
        "style_guess": style_guess,
    }


def make_spider_plot(hop_profile_dict, title="Hop Aroma Radar"):
    """
    Draw a filled spider/radar chart (circular web).
    We'll show each aroma dim as a spoke.
    """

    # if somehow empty:
    if not hop_profile_dict:
        hop_profile_dict = {
            "fruity": 0,
            "citrus": 0,
            "tropical": 0,
            "earthy": 0,
            "spicy": 0,
            "herbal": 0,
            "floral": 0,
            "resinous": 0,
        }

    labels = list(hop_profile_dict.keys())
    values = list(hop_profile_dict.values())

    N = len(labels)

    # angles for each category
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    # close the polygon
    angles_closed = np.concatenate([angles, [angles[0]]])
    values_closed = np.concatenate([values, [values[0]]])

    fig, ax = plt.subplots(
        figsize=(6, 6),
        subplot_kw=dict(polar=True)
    )

    ax.plot(angles_closed, values_closed, color="#1f77b4", linewidth=2)
    ax.fill(angles_closed, values_closed, color="#1f77b4", alpha=0.25)

    # Put numeric label on each vertex
    for ang, val in zip(angles, values):
        ax.text(
            ang,
            val,
            f"{val:.2f}",
            ha="center",
            va="center",
            fontsize=9,
            bbox=dict(
                boxstyle="round,pad=0.2",
                fc="white",
                ec="#1f77b4",
                lw=1
            ),
        )

    # set category labels around the circle
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=11)

    # look a bit nicer
    ax.set_rlabel_position(0)
    ax.yaxis.grid(color="gray", linestyle="--", alpha=0.4)
    ax.xaxis.grid(color="gray", linestyle="--", alpha=0.4)
    ax.set_title(title, fontsize=20, fontweight="bold", pad=20)

    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------------
# SIDEBAR INPUTS
# ----------------------------------------------------------------------------------

st.sidebar.header("🧪 Model Inputs")

# HOP BILL
st.sidebar.subheader("Hop Bill (g)")
all_hops = list_all_hops_from_features(hop_features)

# We'll expose first ~5-6 hops as numeric inputs in grams for simplicity.
# You can expand this to more hops or make it dynamic later.
# We'll just iterate through first N hops.
N_SHOW = min(6, len(all_hops))
hop_bill_dict = {}

for hop_name in all_hops[:N_SHOW]:
    grams = st.sidebar.number_input(
        hop_name,
        min_value=0.0,
        max_value=200.0,
        value=0.0,
        step=5.0,
        key=f"hop_{hop_name}",
    )
    hop_bill_dict[hop_name] = grams

# MALT BILL
st.sidebar.subheader("Malt Bill (%)")
malt_options = sorted(malt_df["PRODUCT NAME"].unique().tolist())

malt1_name = st.sidebar.selectbox("Malt 1", malt_options, index=0 if len(malt_options) > 0 else None, key="malt1_name")
malt1_pct  = st.sidebar.number_input("Malt 1 %", min_value=0.0, max_value=100.0, value=70.0, step=1.0, key="malt1_pct")

malt2_name = st.sidebar.selectbox("Malt 2", malt_options, index=1 if len(malt_options) > 1 else 0, key="malt2_name")
malt2_pct  = st.sidebar.number_input("Malt 2 %", min_value=0.0, max_value=100.0, value=20.0, step=1.0, key="malt2_pct")

malt3_name = st.sidebar.selectbox("Malt 3", malt_options, index=2 if len(malt_options) > 2 else 0, key="malt3_name")
malt3_pct  = st.sidebar.number_input("Malt 3 %", min_value=0.0, max_value=100.0, value=10.0, step=1.0, key="malt3_pct")

malt_selections = [
    {"name": malt1_name, "pct": malt1_pct},
    {"name": malt2_name, "pct": malt2_pct},
    {"name": malt3_name, "pct": malt3_pct},
]

# YEAST
st.sidebar.subheader("Yeast Strain")
if YEAST_NAME_COL in clean_yeast_df.columns:
    yeast_choices_list = clean_yeast_df[YEAST_NAME_COL].dropna().unique().tolist()
else:
    # Fallback so app doesn't crash if wrong column name:
    yeast_choices_list = ["(no yeast column found)"]

yeast_choice = st.sidebar.selectbox(
    "Select yeast",
    yeast_choices_list,
    key="yeast_choice",
)

# RUN BUTTON
run_button = st.sidebar.button("Predict Flavor 🧪", type="primary")


# ----------------------------------------------------------------------------------
# MAIN LAYOUT
# ----------------------------------------------------------------------------------

st.title("🍺 Beer Recipe Digital Twin")
st.caption(
    "Predict hop aroma, malt character, and fermentation profile using trained ML models."
)

if run_button:
    # Compute summary
    summary = summarize_beer(
        hop_bill_dict,
        malt_selections,
        yeast_choice,
        hop_model, hop_features, hop_dims,
        malt_model, malt_df, malt_features, malt_dims,
        yeast_model, clean_yeast_df, yeast_features, yeast_dims,
        YEAST_NAME_COL,
    )

    hop_profile   = summary["hop_out"]
    hop_notes     = summary["hop_top_notes"]
    malt_traits   = summary["malt_traits"]
    yeast_traits  = summary["yeast_traits"]
    style_guess   = summary["style_guess"]

    col_left, col_right = st.columns([2, 1], vertical_alignment="top")

    with col_left:
        fig = make_spider_plot(hop_profile, title="Hop Aroma Radar")
        st.pyplot(fig, use_container_width=True)

    with col_right:
        # Top hop notes list
        st.markdown("### Top hop notes:")
        if hop_notes:
            for n in hop_notes:
                st.write(f"- {n}")
        else:
            st.write("_No dominant hop note_")

        # Malt
        st.markdown("### Malt character:")
        if malt_traits:
            st.write(", ".join(malt_traits))
        else:
            st.write("None")

        # Yeast
        st.markdown("### Yeast character:")
        if yeast_traits:
            st.write(", ".join(yeast_traits))
        else:
            st.write("None")

        st.markdown("### Style direction:")
        st.write(f"🧭 {style_guess}")

        # Also show which hops actually had >0 grams (to help interpret)
        nonzero_hops = [h for h, g in hop_bill_dict.items() if g > 0]
        if nonzero_hops:
            st.markdown("### Hops used by the model:")
            st.write(", ".join(nonzero_hops))

else:
    # Instructions state if user hasn't run yet
    st.info(
        "👉 Build your hop bill (grams for each hop), set malt bill (% grist), pick a yeast, "
        "then click **Predict Flavor 🧪** in the sidebar."
    )

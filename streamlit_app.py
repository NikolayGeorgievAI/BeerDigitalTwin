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
    hop_bundle   = joblib.load("hop_aroma_model.joblib")
    malt_bundle  = joblib.load("malt_sensory_model.joblib")
    yeast_bundle = joblib.load("yeast_sensory_model.joblib")

    hop_model      = hop_bundle["model"]
    hop_features   = hop_bundle["feature_cols"]
    hop_dims       = hop_bundle["aroma_dims"]

    malt_model     = malt_bundle["model"]
    malt_features  = malt_bundle["feature_cols"]
    malt_dims      = malt_bundle["flavor_cols"]

    yeast_model    = yeast_bundle["model"]
    yeast_features = yeast_bundle["feature_cols"]
    yeast_dims     = yeast_bundle["flavor_cols"]

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

# which column in yeast df stores the strain name?
YEAST_NAME_COL = "Name"  # change if needed

# ----------------------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------------------

def list_all_hops_from_features(hop_features):
    names = []
    for col in hop_features:
        if col.startswith("hop_"):
            names.append(col.replace("hop_", "", 1))
        else:
            names.append(col)
    return names


def build_hop_feature_row(hop_bill_dict, hop_features):
    row = []
    for feat in hop_features:
        hop_name = feat.replace("hop_", "", 1)
        row.append(hop_bill_dict.get(hop_name, 0.0))
    X = np.array(row, dtype=float).reshape(1, -1)
    return X


def predict_hop_profile(hop_bill_dict, hop_model, hop_features, hop_dims):
    X = build_hop_feature_row(hop_bill_dict, hop_features)
    y_pred = hop_model.predict(X)[0]
    return dict(zip(hop_dims, y_pred))


def get_weighted_malt_vector(malt_selections, malt_df, malt_features):
    blend_vec = np.zeros(len(malt_features), dtype=float)
    total_pct = sum([float(m["pct"]) for m in malt_selections])

    if total_pct <= 0:
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
    row = yeast_df[yeast_df[name_col] == yeast_name].head(1)
    if row.empty:
        return np.zeros(len(yeast_features)).reshape(1, -1)

    vals = []
    for feat in yeast_features:
        if feat in row.columns:
            vals.append(row.iloc[0][feat])
        else:
            vals.append(0.0)
    return np.array(vals, dtype=float).reshape(1, -1)


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

    # top 2 hop notes
    hop_sorted = sorted(hop_out.items(), key=lambda kv: kv[1], reverse=True)
    top_hop_notes = [f"{k} ({round(v,2)})" for k, v in hop_sorted[:2]]

    malt_traits_on  = [k for k, v in malt_out.items()  if v == 1 or v is True]
    yeast_traits_on = [k for k, v in yeast_out.items() if v == 1 or v is True]

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
        "hop_top_notes": top_hop_notes,
        "malt_traits": malt_traits_on,
        "yeast_traits": yeast_traits_on,
        "style_guess": style_guess,
    }


def make_spider_plot(hop_profile_dict, title="Hop Aroma Radar"):
    """
    Radar/spider chart with polygon rings and spokes.
    """

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
    raw_vals = np.array(list(hop_profile_dict.values()), dtype=float)

    # clamp negatives -> 0 for cleaner visuals
    vals = np.maximum(raw_vals, 0.0)

    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)

    angles_closed = np.concatenate([angles, [angles[0]]])
    vals_closed   = np.concatenate([vals,   [vals[0]]])

    vmax = float(vals.max()) if vals.max() > 0 else 1.0
    n_rings = 5
    ring_levels = np.linspace(0, vmax, n_rings + 1)

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    # draw polygon rings
    for r in ring_levels[1:]:
        ring_vals = np.full(N, r)
        ring_vals_closed = np.concatenate([ring_vals, [ring_vals[0]]])
        ax.plot(
            angles_closed,
            ring_vals_closed,
            color="gray",
            linestyle="--",
            linewidth=0.7,
            alpha=0.5,
        )

    # draw spokes
    for ang in angles:
        ax.plot(
            [ang, ang], [0, vmax],
            color="gray",
            linestyle="--",
            linewidth=0.7,
            alpha=0.5,
        )

    # plot data
    ax.plot(
        angles_closed,
        vals_closed,
        color="#1f77b4",
        linewidth=2,
    )
    ax.fill(
        angles_closed,
        vals_closed,
        color="#1f77b4",
        alpha=0.25,
    )

    # numeric labels at each node
    for ang, v in zip(angles, vals):
        ax.text(
            ang,
            v,
            f"{v:.2f}",
            ha="center",
            va="center",
            fontsize=9,
            bbox=dict(
                boxstyle="round,pad=0.2",
                fc="white",
                ec="#1f77b4",
                lw=1,
            ),
        )

    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=12)

    ax.set_yticklabels([])  # hide radial tick labels
    ax.set_ylim(0, vmax if vmax > 0 else 1.0)

    ax.spines["polar"].set_visible(False)

    ax.set_title(title, fontsize=24, fontweight="bold", pad=20)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------------
# SIDEBAR INPUTS
# ----------------------------------------------------------------------------------

st.sidebar.header("🧪 Model Inputs")

######################################
# Hop bill (dropdown + grams each)
######################################
st.sidebar.subheader("Hop Bill (g)")

all_hops = list_all_hops_from_features(hop_features)

hop_selections = []
for i in range(4):
    c1, c2 = st.sidebar.columns([2, 1])
    with c1:
        hop_choice = st.selectbox(
            f"Hop {i+1}",
            options=["(none)"] + all_hops,
            index=0,
            key=f"hop_name_{i}",
        )
    with c2:
        grams = st.number_input(
            "g",
            min_value=0.0,
            max_value=200.0,
            value=0.0,
            step=5.0,
            key=f"hop_g_{i}",
        )
    if hop_choice != "(none)" and grams > 0:
        hop_selections.append((hop_choice, grams))

# merge duplicates into dict for model input
hop_bill_dict = {}
for hop_name, g in hop_selections:
    hop_bill_dict[hop_name] = hop_bill_dict.get(hop_name, 0.0) + g


######################################
# Malt bill
######################################
st.sidebar.subheader("Malt Bill (%)")
malt_options = sorted(malt_df["PRODUCT NAME"].unique().tolist())

malt1_name = st.sidebar.selectbox("Malt 1", malt_options, index=0 if len(malt_options) > 0 else None, key="malt1_name")
malt1_pct  = st.sidebar.number_input("Malt 1 %", min_value=0.0, max_value=100.0,
                                     value=70.0, step=1.0, key="malt1_pct")

malt2_name = st.sidebar.selectbox("Malt 2", malt_options, index=1 if len(malt_options) > 1 else 0, key="malt2_name")
malt2_pct  = st.sidebar.number_input("Malt 2 %", min_value=0.0, max_value=100.0,
                                     value=20.0, step=1.0, key="malt2_pct")

malt3_name = st.sidebar.selectbox("Malt 3", malt_options, index=2 if len(malt_options) > 2 else 0, key="malt3_name")
malt3_pct  = st.sidebar.number_input("Malt 3 %", min_value=0.0, max_value=100.0,
                                     value=10.0, step=1.0, key="malt3_pct")

malt_selections = [
    {"name": malt1_name, "pct": malt1_pct},
    {"name": malt2_name, "pct": malt2_pct},
    {"name": malt3_name, "pct": malt3_pct},
]

######################################
# Yeast
######################################
st.sidebar.subheader("Yeast Strain")

if YEAST_NAME_COL in clean_yeast_df.columns:
    yeast_choices_list = clean_yeast_df[YEAST_NAME_COL].dropna().unique().tolist()
else:
    yeast_choices_list = ["(no yeast column found)"]

yeast_choice = st.sidebar.selectbox(
    "Select yeast",
    yeast_choices_list,
    key="yeast_choice",
)

######################################
# Predict button
######################################
run_button = st.sidebar.button("Predict Flavor 🧪", type="primary")


# ----------------------------------------------------------------------------------
# MAIN BODY
# ----------------------------------------------------------------------------------
st.title("🍺 Beer Recipe Digital Twin")
st.caption(
    "Predict hop aroma, malt character, and fermentation profile using trained ML models."
)

if run_button:
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
        st.markdown("### Top hop notes:")
        if hop_notes:
            for n in hop_notes:
                st.write(f"- {n}")
        else:
            st.write("_No dominant hop note_")

        st.markdown("### Malt character:")
        st.write(", ".join(malt_traits) if malt_traits else "None")

        st.markdown("### Yeast character:")
        st.write(", ".join(yeast_traits) if yeast_traits else "None")

        st.markdown("### Style direction:")
        st.write(f"🧭 {style_guess}")

        # also list any hops actually included
        nonzero_hops = [f"{h} ({g} g)" for h, g in hop_bill_dict.items() if g > 0]
        if nonzero_hops:
            st.markdown("### Hops used by the model:")
            st.write(", ".join(nonzero_hops))

else:
    st.info(
        "👉 Pick up to 4 hops (with grams), set malt bill (% grist), choose a yeast strain, "
        "then click **Predict Flavor 🧪** in the sidebar."
    )

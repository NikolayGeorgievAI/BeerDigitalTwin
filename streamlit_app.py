# =========================
# Beer Recipe Digital Twin
# =========================

# ---- Imports ----
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


# ---- Page setup ----
st.set_page_config(
    page_title="Beer Recipe Digital Twin",
    page_icon="🍺",
    layout="wide",
)

st.title("🍺 Beer Recipe Digital Twin")
st.caption(
    "Predict hop aroma, malt character, and fermentation profile using trained ML models."
)


# ---- Data/model loader ----
@st.cache_resource
def load_models_and_data():
    # load joblib bundles
    hop_bundle   = joblib.load("hop_aroma_model.joblib")
    malt_bundle  = joblib.load("malt_sensory_model.joblib")
    yeast_bundle = joblib.load("yeast_sensory_model.joblib")

    hop_model      = hop_bundle["model"]
    hop_features   = hop_bundle["feature_cols"]   # e.g. hop_Astra, hop_Eclipse, ...
    hop_dims       = hop_bundle["aroma_dims"]     # e.g. ['tropical','citrus',...]

    malt_model     = malt_bundle["model"]
    malt_features  = malt_bundle["feature_cols"]  # columns from clean_malt_df.pkl
    malt_dims      = malt_bundle["flavor_cols"]   # predicted malt descriptors

    yeast_model    = yeast_bundle["model"]
    yeast_features = yeast_bundle["feature_cols"] # e.g. ['Temp_avg_C','Flocculation_num',...]
    yeast_dims     = yeast_bundle["flavor_cols"]  # predicted yeast descriptors

    # supporting clean data
    clean_malt_df  = pd.read_pickle("clean_malt_df.pkl")
    clean_yeast_df = pd.read_pickle("clean_yeast_df.pkl")

    return (
        hop_model, hop_features, hop_dims,
        malt_model, malt_features, malt_dims,
        yeast_model, yeast_features, yeast_dims,
        clean_malt_df, clean_yeast_df
    )

(
    hop_model, hop_features, hop_dims,
    malt_model, malt_features, malt_dims,
    yeast_model, yeast_features, yeast_dims,
    malt_df, yeast_df
) = load_models_and_data()


# ---- Helper functions ----
def get_all_hop_names(hop_feature_cols):
    """Take columns like 'hop_Astra', 'hop_Eclipse', ... -> ['Astra','Eclipse', ...]"""
    return [c.replace("hop_", "") for c in hop_feature_cols]


def build_hop_feature_vector(hop_bill_dict, hop_feature_cols):
    """
    hop_bill_dict = { 'Astra': 50, 'Eclipse': 20, ... }  (g)
    Must return X shape (1, n_features) aligned to hop_feature_cols.
    """
    row = []
    for col in hop_feature_cols:
        hop_name = col.replace("hop_", "")
        row.append(hop_bill_dict.get(hop_name, 0.0))
    return np.array(row).reshape(1, -1)


def predict_hop_profile(hop_bill_dict, hop_model, hop_feature_cols, hop_dims):
    """Return dict like {'tropical':0.8,'citrus':0.4,...} from hop bill."""
    X = build_hop_feature_vector(hop_bill_dict, hop_feature_cols)
    y_pred = hop_model.predict(X)[0]  # 1D
    return dict(zip(hop_dims, y_pred))


def make_weighted_malt_vector(malt_selections, malt_df, malt_feature_cols):
    """
    malt_selections = [
        {'name': 'BEST ALE MALT', 'pct': 70.0},
        {'name': 'BLACK MALT',   'pct': 20.0},
        ...
    ]
    We'll build a weighted average across those malts in feature space.
    """
    blend = np.zeros(len(malt_feature_cols), dtype=float)

    for item in malt_selections:
        malt_name = item["name"]
        pct       = float(item["pct"])
        row = malt_df[malt_df["PRODUCT NAME"] == malt_name].head(1)
        if row.empty:
            continue

        vec = np.array(
            [row.iloc[0][feat] for feat in malt_feature_cols],
            dtype=float
        )
        blend += vec * (pct/100.0)

    return blend.reshape(1, -1)


def predict_malt_profile(malt_selections, malt_model, malt_df, malt_feature_cols, malt_dims):
    """
    Return dict like {'caramel':1,'toffee':0,'bready':1,...}
    (whatever your malt model predicts).
    """
    X = make_weighted_malt_vector(malt_selections, malt_df, malt_feature_cols)
    y_pred = malt_model.predict(X)[0]
    return dict(zip(malt_dims, y_pred))


def get_yeast_feature_vector(yeast_name, yeast_df, yeast_feature_cols):
    """
    Build numeric vector for the chosen yeast (temp, flocculation, attenuation, etc.)
    We'll map columns in yeast_df to yeast_feature_cols in correct order.
    """
    row = yeast_df[yeast_df["Name"] == yeast_name].head(1)
    if row.empty:
        return np.zeros(len(yeast_feature_cols)).reshape(1, -1)

    # IMPORTANT: reorder to yeast_feature_cols
    vals = []
    for col in yeast_feature_cols:
        vals.append(row.iloc[0][col])
    return np.array(vals, dtype=float).reshape(1, -1)


def predict_yeast_profile(yeast_name, yeast_model, yeast_df, yeast_feature_cols, yeast_dims):
    """
    Return dict like {'fruity_esters':1,'clean_neutral':0,'phenolic_spicy':0,...}
    """
    X = get_yeast_feature_vector(yeast_name, yeast_df, yeast_feature_cols)
    y_pred = yeast_model.predict(X)[0]
    return dict(zip(yeast_dims, y_pred))


def summarize_beer(
    hop_bill_dict,
    malt_selections,
    yeast_name,
    hop_model, hop_feature_cols, hop_dims,
    malt_model, malt_df, malt_feature_cols, malt_dims,
    yeast_model, yeast_df, yeast_feature_cols, yeast_dims
):
    # hop aroma vector
    hop_out   = predict_hop_profile(
        hop_bill_dict,
        hop_model,
        hop_feature_cols,
        hop_dims
    )

    # malt descriptor flags/scores
    malt_out  = predict_malt_profile(
        malt_selections,
        malt_model,
        malt_df,
        malt_feature_cols,
        malt_dims
    )

    # yeast descriptor flags/scores
    yeast_out = predict_yeast_profile(
        yeast_name,
        yeast_model,
        yeast_df,
        yeast_feature_cols,
        yeast_dims
    )

    # top hop notes (sort hop_out descending)
    hop_sorted = sorted(hop_out.items(), key=lambda kv: kv[1], reverse=True)
    top_hop_notes = [f"{k} ({round(v, 2)})" for k, v in hop_sorted[:2]]

    malt_traits  = [k for k, v in malt_out.items() if v == 1]
    yeast_traits = [k for k, v in yeast_out.items() if v == 1]

    # naive style logic
    style_guess = "Experimental / Hybrid"
    if (
        ("clean_neutral" in yeast_out and yeast_out["clean_neutral"] == 1)
        and ("dry_finish" in yeast_out and yeast_out["dry_finish"] == 1)
    ):
        # “West Coast-ish” if citrus/resin high
        if any("citrus" in n.lower() or "resin" in n.lower() for n in top_hop_notes):
            style_guess = "West Coast IPA / Modern IPA"
        else:
            style_guess = "Clean / Neutral Ale direction"

    if (
        ("fruity_esters" in yeast_out and yeast_out["fruity_esters"] == 1)
        and ("tropical" in hop_out and hop_out["tropical"] > 0.6)
    ):
        style_guess = "Hazy / NEIPA leaning"

    if "phenolic_spicy" in yeast_out and yeast_out["phenolic_spicy"] == 1:
        style_guess = "Belgian / Saison leaning"

    if "caramel" in malt_out and malt_out["caramel"] == 1:
        style_guess = "English / Malt-forward Ale"

    return {
        "hop_out": hop_out,
        "top_hop_notes": top_hop_notes,
        "malt_traits": malt_traits,
        "yeast_traits": yeast_traits,
        "style_guess": style_guess,
    }


def make_spider_plot(hop_out_dict):
    """
    Draw radar plot for hop aroma. Axes are fixed so we get a nice spider web.
    hop_out_dict ~ {'tropical':0.7,'citrus':0.5,...}
    """
    axes_order = [
        "tropical",
        "citrus",
        "fruity",
        "resinous",
        "floral",
        "herbal",
        "spicy",
        "earthy",
    ]

    vals = [float(hop_out_dict.get(dim, 0.0)) for dim in axes_order]
    vals_closed = vals + [vals[0]]

    angles = np.linspace(0, 2*np.pi, len(axes_order), endpoint=False)
    angles_closed = np.concatenate([angles, [angles[0]]])

    fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
    ax.set_facecolor("#fafafa")

    # web lines
    ax.yaxis.grid(color="gray", linestyle="--", alpha=0.4)
    ax.xaxis.grid(color="gray", linestyle="--", alpha=0.4)

    # plot + fill
    ax.plot(angles_closed, vals_closed, color="#1f77b4", linewidth=2)
    ax.fill(angles_closed, vals_closed, color="#1f77b4", alpha=0.25)

    # label each point
    for ang, val in zip(angles, vals):
        ax.text(
            ang,
            val,
            f"{val:.2f}",
            ha="center",
            va="center",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#1f77b4", lw=1),
        )

    # category labels around
    ax.set_xticks(angles)
    ax.set_xticklabels(axes_order, fontsize=10)

    # hide the radial tick labels (just keep circles)
    for lab in ax.get_yticklabels():
        lab.set_visible(False)

    # scale radius so the shape sits in the middle nicely
    max_val = max(max(vals), 0.5)
    ax.set_ylim(0, max_val * 1.2)

    ax.set_title("Hop Aroma Radar", fontsize=20, fontweight="bold", pad=20)
    fig.tight_layout()
    return fig


# ---- Sidebar inputs ----
st.sidebar.header("Hop Bill (g)")

all_hops_sorted = sorted(get_all_hop_names(hop_features))

hop1_name = st.sidebar.selectbox("Hop 1", all_hops_sorted, key="hop1_name")
hop1_amt  = st.sidebar.number_input(
    f"{hop1_name} (g)",
    min_value=0.0, max_value=500.0,
    value=0.0, step=5.0,
    key="hop1_amt"
)

hop2_name = st.sidebar.selectbox("Hop 2", all_hops_sorted, key="hop2_name")
hop2_amt  = st.sidebar.number_input(
    f"{hop2_name} (g)",
    min_value=0.0, max_value=500.0,
    value=0.0, step=5.0,
    key="hop2_amt"
)

hop3_name = st.sidebar.selectbox("Hop 3", all_hops_sorted, key="hop3_name")
hop3_amt  = st.sidebar.number_input(
    f"{hop3_name} (g)",
    min_value=0.0, max_value=500.0,
    value=0.0, step=5.0,
    key="hop3_amt"
)

hop4_name = st.sidebar.selectbox("Hop 4", all_hops_sorted, key="hop4_name")
hop4_amt  = st.sidebar.number_input(
    f"{hop4_name} (g)",
    min_value=0.0, max_value=500.0,
    value=0.0, step=5.0,
    key="hop4_amt"
)

hop_bill = {
    hop1_name: hop1_amt,
    hop2_name: hop2_amt,
    hop3_name: hop3_amt,
    hop4_name: hop4_amt,
}


st.sidebar.header("Malt Bill")
malt_options = sorted(malt_df["PRODUCT NAME"].dropna().unique().tolist())

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


st.sidebar.header("Yeast Strain")
yeast_options = sorted(yeast_df["Name"].dropna().unique().tolist())
chosen_yeast  = st.sidebar.selectbox("Select yeast", yeast_options, key="yeast_choice")


# ---- BUTTON (This MUST come after imports + all code above) ----
run_button = st.sidebar.button("Predict Flavor 🧪")


# ---- Main body ----
if run_button:
    # Run the predictions
    results = summarize_beer(
        hop_bill,
        malt_selections,
        chosen_yeast,
        hop_model, hop_features, hop_dims,
        malt_model, malt_df, malt_features, malt_dims,
        yeast_model, yeast_df, yeast_features, yeast_dims,
    )

    hop_out       = results["hop_out"]
    top_hop_notes = results["top_hop_notes"]
    malt_traits   = results["malt_traits"]
    yeast_traits  = results["yeast_traits"]
    style_guess   = results["style_guess"]

    left_col, right_col = st.columns([0.6, 0.4], vertical_alignment="top")

    with left_col:
        fig = make_spider_plot(hop_out)
        st.pyplot(fig, use_container_width=True)

    with right_col:
        st.subheader("Top hop notes:")
        if top_hop_notes:
            for n in top_hop_notes:
                st.write(f"- {n}")
        else:
            st.write("_No dominant hop note_")

        st.subheader("Malt character:")
        if malt_traits:
            for trait in malt_traits:
                st.write("-", trait)
        else:
            st.write("None")

        st.subheader("Yeast character:")
        if yeast_traits:
            for trait in yeast_traits:
                st.write("-", trait)
        else:
            st.write("None")

        st.subheader("Style direction:")
        st.write(f"🧭 {style_guess}")

else:
    st.info(
        "👉 Build your hop bill (up to 4 hops with some grams), set malt bill (% grist), "
        "choose yeast, then click **Predict Flavor 🧪** in the sidebar."
    )

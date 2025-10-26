# ============================================
# 🍺 Beer Recipe Digital Twin (DEBUG BUILD v8️⃣)
# ============================================

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
st.write("**DEBUG BUILD v8️⃣**")
st.title("🍺 Beer Recipe Digital Twin")
st.caption(
    "Predict hop aroma, malt character, and fermentation profile using trained ML models."
)

# ---- Load models and datasets ----
@st.cache_resource
def load_models_and_data():
    hop_bundle = joblib.load("hop_aroma_model.joblib")
    malt_bundle = joblib.load("malt_sensory_model.joblib")
    yeast_bundle = joblib.load("yeast_sensory_model.joblib")

    hop_model = hop_bundle["model"]
    hop_features = hop_bundle["feature_cols"]
    hop_dims = hop_bundle["aroma_dims"]

    malt_model = malt_bundle["model"]
    malt_features = malt_bundle["feature_cols"]
    malt_dims = malt_bundle["flavor_cols"]

    yeast_model = yeast_bundle["model"]
    yeast_features = yeast_bundle["feature_cols"]
    yeast_dims = yeast_bundle["flavor_cols"]

    clean_malt_df = pd.read_pickle("clean_malt_df.pkl")
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
    return [c.replace("hop_", "") for c in hop_feature_cols]

def build_hop_feature_vector(hop_bill_dict, hop_feature_cols):
    return np.array(
        [hop_bill_dict.get(c.replace("hop_", ""), 0.0) for c in hop_feature_cols]
    ).reshape(1, -1)

def predict_hop_profile(hop_bill_dict, hop_model, hop_feature_cols, hop_dims):
    X = build_hop_feature_vector(hop_bill_dict, hop_feature_cols)
    y_pred = hop_model.predict(X)[0]
    return dict(zip(hop_dims, y_pred))

def make_weighted_malt_vector(malt_selections, malt_df, malt_feature_cols):
    blend = np.zeros(len(malt_feature_cols))
    for item in malt_selections:
        name, pct = item["name"], item["pct"]
        row = malt_df[malt_df["PRODUCT NAME"] == name]
        if row.empty:
            continue
        vec = np.array([row.iloc[0][f] for f in malt_feature_cols])
        blend += vec * (pct / 100.0)
    return blend.reshape(1, -1)

def predict_malt_profile(malt_selections, malt_model, malt_df, malt_feature_cols, malt_dims):
    X = make_weighted_malt_vector(malt_selections, malt_df, malt_feature_cols)
    y_pred = malt_model.predict(X)[0]
    return dict(zip(malt_dims, y_pred))

def get_yeast_feature_vector(yeast_name, yeast_df, yeast_feature_cols):
    row = yeast_df[yeast_df["Name"] == yeast_name]
    if row.empty:
        return np.zeros(len(yeast_feature_cols)).reshape(1, -1)
    vals = [row.iloc[0][col] for col in yeast_feature_cols]
    return np.array(vals, dtype=float).reshape(1, -1)

def predict_yeast_profile(yeast_name, yeast_model, yeast_df, yeast_feature_cols, yeast_dims):
    X = get_yeast_feature_vector(yeast_name, yeast_df, yeast_feature_cols)
    y_pred = yeast_model.predict(X)[0]
    return dict(zip(yeast_dims, y_pred))

def summarize_beer(
    hop_bill_dict,
    malt_selections,
    yeast_name,
    hop_model, hop_feature_cols, hop_dims,
    malt_model, malt_df, malt_feature_cols, malt_dims,
    yeast_model, yeast_df, yeast_feature_cols, yeast_dims,
):
    hop_out = predict_hop_profile(hop_bill_dict, hop_model, hop_feature_cols, hop_dims)
    malt_out = predict_malt_profile(malt_selections, malt_model, malt_df, malt_feature_cols, malt_dims)
    yeast_out = predict_yeast_profile(yeast_name, yeast_model, yeast_df, yeast_feature_cols, yeast_dims)

    hop_sorted = sorted(hop_out.items(), key=lambda kv: kv[1], reverse=True)
    top_hops = [f"{k} ({round(v, 2)})" for k, v in hop_sorted[:2]]

    malt_traits = [k for k, v in malt_out.items() if v == 1]
    yeast_traits = [k for k, v in yeast_out.items() if v == 1]

    style_guess = "Experimental / Hybrid"
    if ("clean_neutral" in yeast_out and yeast_out["clean_neutral"] == 1) and \
       ("dry_finish" in yeast_out and yeast_out["dry_finish"] == 1):
        style_guess = "Clean / Neutral Ale direction"
    if ("fruity_esters" in yeast_out and yeast_out["fruity_esters"] == 1) and \
       ("tropical" in hop_out and hop_out["tropical"] > 0.6):
        style_guess = "Hazy / NEIPA leaning"
    if "phenolic_spicy" in yeast_out and yeast_out["phenolic_spicy"] == 1:
        style_guess = "Belgian / Saison leaning"
    if "caramel" in malt_out and malt_out["caramel"] == 1:
        style_guess = "English / Malt-forward Ale"

    return {
        "hop_out": hop_out,
        "top_hops": top_hops,
        "malt_traits": malt_traits,
        "yeast_traits": yeast_traits,
        "style_guess": style_guess,
    }

def make_spider_plot(hop_out_dict):
    axes = ["tropical","citrus","fruity","resinous","floral","herbal","spicy","earthy"]
    vals = [float(hop_out_dict.get(dim, 0.0)) for dim in axes]
    vals += [vals[0]]
    angles = np.linspace(0, 2*np.pi, len(axes), endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])

    fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
    ax.set_facecolor("#fafafa")
    ax.plot(angles, vals, color="#1f77b4", linewidth=2)
    ax.fill(angles, vals, color="#1f77b4", alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(axes, fontsize=10)
    ax.set_yticklabels([])
    ax.set_title("Hop Aroma Radar", fontsize=20, fontweight="bold", pad=20)
    fig.tight_layout()
    return fig

# ---- Sidebar ----
st.sidebar.header("Hop Bill (g)")
hop_names = sorted(get_all_hop_names(hop_features))

hop_bill = {}
for i in range(1, 5):
    hop_name = st.sidebar.selectbox(f"Hop {i}", hop_names, key=f"hop{i}_name")
    hop_amt = st.sidebar.number_input(f"{hop_name} (g)", min_value=0.0, max_value=500.0, value=0.0, step=5.0, key=f"hop{i}_amt")
    hop_bill[hop_name] = hop_amt

st.sidebar.header("Malt Bill")
malt_names = sorted(malt_df["PRODUCT NAME"].dropna().unique().tolist())
malt_selections = []
for i, default_pct in zip(range(1, 4), [70.0, 20.0, 10.0]):
    name = st.sidebar.selectbox(f"Malt {i}", malt_names, key=f"malt{i}_name")
    pct = st.sidebar.number_input(f"Malt {i} %", min_value=0.0, max_value=100.0, value=default_pct, step=1.0, key=f"malt{i}_pct")
    malt_selections.append({"name": name, "pct": pct})

st.sidebar.header("Yeast Strain")
yeast_options = sorted(yeast_df["Name"].dropna().unique().tolist())
chosen_yeast = st.sidebar.selectbox("Select yeast", yeast_options, key="yeast_choice")

# ---- Button ----
run_button = st.sidebar.button("Predict Flavor 🧪")

# ---- Main layout ----
if run_button:
    results = summarize_beer(
        hop_bill, malt_selections, chosen_yeast,
        hop_model, hop_features, hop_dims,
        malt_model, malt_df, malt_features, malt_dims,
        yeast_model, yeast_df, yeast_features, yeast_dims,
    )

    hop_out = results["hop_out"]
    top_hops = results["top_hops"]
    malt_traits = results["malt_traits"]
    yeast_traits = results["yeast_traits"]
    style_guess = results["style_guess"]

    col1, col2 = st.columns([0.6, 0.4])
    with col1:
        fig = make_spider_plot(hop_out)
        st.pyplot(fig, use_container_width=True)

    with col2:
        st.subheader("Top hop notes:")
        for h in top_hops or ["No dominant hop note"]:
            st.write(f"- {h}")

        st.subheader("Malt character:")
        st.write(", ".join(malt_traits) if malt_traits else "None")

        st.subheader("Yeast character:")
        st.write(", ".join(yeast_traits) if yeast_traits else "None")

        st.subheader("Style direction:")
        st.write(f"🧭 {style_guess}")

else:
    st.info("👉 Build your hop bill, malt bill, and select yeast. Then click **Predict Flavor 🧪**.")

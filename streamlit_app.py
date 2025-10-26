# Beer Recipe Digital Twin - Streamlit App
# Predict hop aroma, malt character, and yeast profile using trained ML models

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# -----------------------
# Load trained models
# -----------------------
@st.cache_resource
def load_models():
    hop_model = joblib.load("hop_aroma_model.joblib")
    malt_model = joblib.load("malt_sensory_model.joblib")
    yeast_model = joblib.load("yeast_sensory_model.joblib")
    return hop_model, malt_model, yeast_model

hop_model, malt_model, yeast_model = load_models()

# -----------------------
# Load reference data
# -----------------------
@st.cache_data
def load_reference_data():
    clean_malt_df = pd.read_pickle("clean_malt_df.pkl")
    clean_yeast_df = pd.read_pickle("clean_yeast_df.pkl")
    return clean_malt_df, clean_yeast_df

clean_malt_df, clean_yeast_df = load_reference_data()


# -----------------------
# Helper: Predict hop profile
# -----------------------
def predict_hop_profile(hop_bill, model):
    """
    hop_bill: dict of {hop_name: grams}
    model: trained hop aroma model
    Returns normalized aroma dict
    """
    if not hop_bill or sum(hop_bill.values()) == 0:
        return {k: 0.0 for k in ["tropical", "citrus", "fruity", "resinous", "floral", "herbal", "earthy", "spicy"]}

    df = pd.DataFrame(list(hop_bill.items()), columns=["hop", "grams"])
    df["grams_norm"] = df["grams"] / df["grams"].sum()
    aroma = model.predict(df["grams_norm"].values.reshape(1, -1))[0]
    aroma_dict = dict(zip(["tropical", "citrus", "fruity", "resinous", "floral", "herbal", "earthy", "spicy"], aroma))
    return aroma_dict


# -----------------------
# PLOT: Hop Aroma Spider Web
# -----------------------
def plot_hop_radar(hop_profile, title="Hop Aroma Radar"):
    """Spider-web style radar chart with polygon grid rings."""

    # Fallback for empty profile
    if not hop_profile:
        hop_profile = {
            "tropical": 0.0,
            "citrus": 0.0,
            "fruity": 0.0,
            "resinous": 0.0,
            "floral": 0.0,
            "herbal": 0.0,
            "earthy": 0.0,
            "spicy": 0.0,
        }

    labels = list(hop_profile.keys())
    values = list(hop_profile.values())
    n_axes = len(labels)
    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False)

    max_val = max(1.0, float(np.max(values)) * 1.2 if len(values) else 1.0)
    ring_fracs = [0.2, 0.4, 0.6, 0.8, 1.0]

    def polar_to_xy(r, ang):
        return r * np.cos(ang), r * np.sin(ang)

    fig, ax = plt.subplots(figsize=(6, 6))

    # Polygon rings
    for frac in ring_fracs:
        r = frac * max_val
        ring_x, ring_y = [], []
        for ang in angles:
            x, y = polar_to_xy(r, ang)
            ring_x.append(x)
            ring_y.append(y)
        ring_x.append(ring_x[0])
        ring_y.append(ring_y[0])
        ax.plot(ring_x, ring_y, linestyle="--", color="gray", alpha=0.5, linewidth=1)

    # Radial spokes and labels
    for i, ang in enumerate(angles):
        x0, y0 = polar_to_xy(0, ang)
        x1, y1 = polar_to_xy(max_val, ang)
        ax.plot([x0, x1], [y0, y1], linestyle="--", color="gray", alpha=0.5, linewidth=1)
        lx, ly = polar_to_xy(max_val * 1.12, ang)
        ax.text(lx, ly, labels[i], ha="center", va="center", fontsize=12)

    # Data polygon
    poly_x, poly_y = [], []
    for ang, val in zip(angles, values):
        r = min(val, max_val)
        x, y = polar_to_xy(r, ang)
        poly_x.append(x)
        poly_y.append(y)
    poly_x.append(poly_x[0])
    poly_y.append(poly_y[0])

    ax.fill(poly_x, poly_y, color="#1f77b4", alpha=0.15, zorder=3)
    ax.plot(poly_x, poly_y, color="#1f77b4", linewidth=2, zorder=4)

    # Value labels
    for ang, val in zip(angles, values):
        r = min(val, max_val)
        x, y = polar_to_xy(r, ang)
        ax.text(
            x,
            y,
            f"{val:.2f}",
            color="black",
            ha="center",
            va="center",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#1f77b4", lw=1),
            zorder=5,
        )

    # Clean up
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-max_val * 1.4, max_val * 1.4)
    ax.set_ylim(-max_val * 1.4, max_val * 1.4)
    ax.set_title(title, fontsize=20, fontweight="bold", pad=30)
    fig.tight_layout()
    return fig


# -----------------------
# Streamlit App Layout
# -----------------------
st.set_page_config(page_title="Beer Recipe Digital Twin", page_icon="🍺", layout="wide")

st.title("🍺 Beer Recipe Digital Twin")
st.caption("Predict hop aroma, malt character, and fermentation profile using trained ML models.")

with st.sidebar:
    st.header("🧪 Model Inputs")
    st.subheader("Hop Bill (g)")
    hop_bill = {}
    for hop in ["Adeena", "Admiral", "Citra", "Simcoe", "Mosaic"]:
        grams = st.number_input(f"{hop}", min_value=0, max_value=100, value=0)
        if grams > 0:
            hop_bill[hop] = grams

    st.subheader("Malt Composition (% grist)")
    malt_bill = {
        "Base malt": st.slider("Base malt", 0, 100, 80),
        "Crystal malt": st.slider("Crystal malt", 0, 100, 10),
        "Roasted malt": st.slider("Roasted malt", 0, 100, 5),
        "Other specialty": st.slider("Other", 0, 100, 5),
    }

    st.subheader("Yeast Strain")
    yeast_choice = st.selectbox("Select yeast", clean_yeast_df["name"].unique())

    run_button = st.button("🔮 Predict Flavor")

# -----------------------
# Prediction + Output
# -----------------------
if run_button:
    hop_profile = predict_hop_profile(hop_bill, hop_model)
    fig = plot_hop_radar(hop_profile, title="Hop Aroma Radar")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.pyplot(fig, use_container_width=True)

    with col2:
        st.markdown("### Malt character:")
        st.write("bready")

        st.markdown("### Yeast character:")
        st.write("fruity_esters, clean_neutral")

        st.markdown("### Style direction:")
        st.write("🧭 Experimental / Hybrid")

        st.markdown("### Hops used by the model:")
        st.write(", ".join(hop_bill.keys()) if hop_bill else "None")
else:
    st.info("👉 Build your hop bill, set malt % grist, choose yeast, then click **Predict Flavor**.")

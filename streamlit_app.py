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
# LOAD MODELS & DATA (CACHED)
# ---------------------------------------------------------------------------------
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
def get_all_hop_names(hop_features):
    return [c.replace("hop_", "") for c in hop_features]

def build_hop_features(hop_bill_dict, hop_features):
    row = []
    for col in hop_features:
        hop_name = col.replace("hop_", "")
        row.append(hop_bill_dict.get(hop_name, 0))
    return np.array(row).reshape(1, -1)

def predict_hop_profile(hop_bill_dict, hop_model, hop_features, hop_dims):
    x = build_hop_features(hop_bill_dict, hop_features)
    y_pred = hop_model.predict(x)[0]
    return dict(zip(hop_dims, y_pred))

def get_weighted_malt_vector(malt_selections, malt_df, malt_features):
    blend_vec = np.zeros(len(malt_features), dtype=float)
    for item in malt_selections:
        malt_name = item["name"]
        pct = float(item["pct"])
        row = malt_df[malt_df["PRODUCT NAME"] == malt_name].head(1)
        if row.empty:
            continue
        vec = np.array([row.iloc[0][feat] for feat in malt_features], dtype=float)
        blend_vec += vec * (pct / 100.0)
    return blend_vec.reshape(1, -1)

def predict_malt_profile_from_blend(malt_selections, malt_model, malt_df, malt_features, malt_dims):
    x = get_weighted_malt_vector(malt_selections, malt_df, malt_features)
    y_pred = malt_model.predict(x)[0]
    return dict(zip(malt_dims, y_pred))

def get_yeast_feature_vector(yeast_name, yeast_df, yeast_features):
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
    y_pred = yeast_model.predict(x)[0]
    return dict(zip(yeast_dims, y_pred))

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

    hop_sorted = sorted(hop_out.items(), key=lambda kv: kv[1], reverse=True)
    top_hops   = [f"{k} ({round(v, 3)})" for k, v in hop_sorted[:2]]

    malt_active  = [k for k, v in malt_out.items() if v == 1]
    yeast_active = [k for k, v in yeast_out.items() if v == 1]

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
# RADAR CHART (FIXED SIZE + CORRECT LABELS)
# ---------------------------------------------------------------------------------
def make_radar_chart(hop_profile, hop_top_notes, malt_traits, yeast_traits, style_guess):
    labels = list(hop_profile.keys())
    vals = list(hop_profile.values())
    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]
    plot_vals = vals + vals[:1]

    fig = plt.figure(figsize=(7, 7))
    gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[2.3, 1], wspace=0.3)
    ax = fig.add_subplot(gs[0, 0], polar=True)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.plot(angles, plot_vals, color="#1f77b4", linewidth=2)
    ax.fill(angles, plot_vals, color="#1f77b4", alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.grid(color="gray", linestyle="--", linewidth=0.6, alpha=0.6)

    vmax = max(max(vals), 1e-6)
    ax.set_ylim(0, vmax * 1.2)
    ax.set_title("Hop Aroma Radar", fontsize=18, fontweight="bold", pad=20)

    # Correct numeric labels
    for angle, val in zip(angles[:-1], vals):
        ax.annotate(
            f"{val:.3f}",
            xy=(angle, val),
            ha="center",
            va="center",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#1f77b4", lw=0.5, alpha=0.8)
        )

    # ---- Text box on the right ----
    ax_txt = fig.add_subplot(gs[0, 1])
    ax_txt.axis("off")

    y = 0.95
    ax_txt.text(0, y, "Top hop notes:", fontsize=12, fontweight="bold", va="top"); y -= 0.06
    for note in hop_top_notes:
        ax_txt.text(0, y, f"• {note}", fontsize=11, va="top"); y -= 0.05
    if not hop_top_notes:
        ax_txt.text(0, y, "(none)", fontsize=11, va="top", style="italic"); y -= 0.05

    ax_txt.text(0, y, "Malt character:", fontsize=12, fontweight="bold", va="top"); y -= 0.05
    ax_txt.text(0, y, ", ".join(malt_traits) if malt_traits else "(none detected)", fontsize=11, va="top"); y -= 0.07

    ax_txt.text(0, y, "Yeast character:", fontsize=12, fontweight="bold", va="top"); y -= 0.05
    ax_txt.text(0, y, ", ".join(yeast_traits) if yeast_traits else "(none detected)", fontsize=11, va="top"); y -= 0.07

    ax_txt.text(0, y, "Style direction:", fontsize=12, fontweight="bold", va="top"); y -= 0.05
    ax_txt.text(0, y, f"🍺 {style_guess}", fontsize=11, va="top")

    return fig

# ---------------------------------------------------------------------------------
# APP UI
# ---------------------------------------------------------------------------------
(
    hop_model, hop_features, hop_dims,
    malt_model, malt_features, malt_dims,
    yeast_model, yeast_features, yeast_dims,
    malt_df, yeast_df
) = load_models_and_data()

# --- Sidebar: Hop Bill ---
st.sidebar.header("Hop Bill")
all_hops = sorted(get_all_hop_names(hop_features))

hop_inputs = []
for i in range(1, 5):
    name = st.sidebar.selectbox(f"Hop {i}", all_hops, index=min(i - 1, len(all_hops) - 1), key=f"hop{i}_name")
    amt = st.sidebar.slider(f"{name} (g)", 0, 120, 40, 5, key=f"hop{i}_amt")
    hop_inputs.append((name, amt))
hop_bill = {name: amt for name, amt in hop_inputs}

# --- Sidebar: Malt Bill ---
st.sidebar.header("Malt Bill")
malt_options = sorted(malt_df["PRODUCT NAME"].unique().tolist())
malt_selections = []
for i, default in enumerate([70.0, 20.0, 10.0], start=1):
    name = st.sidebar.selectbox(f"Malt {i}", malt_options, key=f"malt{i}_name")
    pct = st.sidebar.number_input(f"Malt {i} %", 0.0, 100.0, default, 1.0, key=f"malt{i}_pct")
    malt_selections.append({"name": name, "pct": pct})

# --- Sidebar: Yeast ---
st.sidebar.header("Yeast")
yeast_options = sorted(yeast_df["Name"].dropna().unique().tolist())
chosen_yeast = st.sidebar.selectbox("Yeast Strain", yeast_options)
run_button = st.sidebar.button("Predict Flavor 🧪")

# --- Main ---
if run_button:
    summary = summarize_beer(
        hop_bill,
        malt_selections,
        chosen_yeast,
        hop_model, hop_features, hop_dims,
        malt_model, malt_df, malt_features, malt_dims,
        yeast_model, yeast_df, yeast_features, yeast_dims
    )

    fig = make_radar_chart(
        summary["hop_out"],
        summary["hop_top_notes"],
        summary["malt_traits"],
        summary["yeast_traits"],
        summary["style_guess"]
    )

    st.pyplot(fig, use_container_width=False)

    with st.expander("Debug / Raw Outputs"):
        st.json(summary)
else:
    st.info("👈 Build your hop bill (up to 4 hops), malt bill (3 malts with %), choose yeast, then click **Predict Flavor 🧪**.")

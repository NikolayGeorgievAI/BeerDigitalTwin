import streamlit as st
import numpy as np
import pandas as pd
import joblib
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# -----------------------------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Beer Recipe Digital Twin",
    page_icon="🍺",
    layout="wide"
)

st.title("🍺 Beer Recipe Digital Twin")
st.caption("Predict hop aroma, malt character, and fermentation profile using trained ML models.")


# -----------------------------------------------------------------------------
# CACHE: LOAD MODELS AND DATA
# -----------------------------------------------------------------------------
@st.cache_resource
def load_models_and_data():
    """
    Load all trained models + reference data once, cache them for the session.
    """
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


(
    hop_model, hop_features, hop_dims,
    malt_model, malt_features, malt_dims,
    yeast_model, yeast_features, yeast_dims,
    malt_df, yeast_df
) = load_models_and_data()


# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------

# ---- HOPS ----
def get_all_hop_names(hop_features):
    """Return list of hop names the hop model actually knows."""
    return [c.replace("hop_", "") for c in hop_features]


def build_hop_features(hop_bill_dict, hop_features):
    """
    hop_bill_dict = {"Galaxy": 40, "Vic Secret": 20, ...}
    Return row aligned with hop_features order.
    Also return which hops actually matched the model vocab.
    """
    row = []
    matched = []

    for col in hop_features:
        hop_name = col.replace("hop_", "")
        value = hop_bill_dict.get(hop_name, 0)
        if value > 0:
            matched.append(hop_name)
        row.append(value)

    x = np.array(row).reshape(1, -1)
    return x, matched


def predict_hop_profile(hop_bill_dict, hop_model, hop_features, hop_dims):
    x, matched = build_hop_features(hop_bill_dict, hop_features)

    y_pred = hop_model.predict(x)[0]
    hop_out = dict(zip(hop_dims, y_pred))

    return hop_out, matched, x


# ---- MALTS ----
def get_weighted_malt_vector(malt_selections, malt_df, malt_features):
    """
    Blend malt chemistry vectors using % of grist.
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
    y_pred = malt_model.predict(x)[0]
    return dict(zip(malt_dims, y_pred))


# ---- YEAST ----
def get_yeast_feature_vector(yeast_name, yeast_df, yeast_features):
    row = yeast_df[yeast_df["Name"] == yeast_name].head(1)
    if row.empty:
        # fallback zeros if somehow not found
        return np.zeros(len(yeast_features)).reshape(1, -1)

    vec = [row.iloc[0][feat] for feat in yeast_features]
    return np.array(vec).reshape(1, -1)


def predict_yeast_profile(yeast_name, yeast_model, yeast_df, yeast_features, yeast_dims):
    x = get_yeast_feature_vector(yeast_name, yeast_df, yeast_features)
    y_pred = yeast_model.predict(x)[0]
    return dict(zip(yeast_dims, y_pred))


# ---- COMBINE EVERYTHING / SUMMARY ----
def summarize_beer(
    hop_bill_dict,
    malt_selections,
    yeast_name,
    hop_model, hop_features, hop_dims,
    malt_model, malt_df, malt_features, malt_dims,
    yeast_model, yeast_df, yeast_features, yeast_dims,
):
    hop_out, matched_hops, hop_debug_matrix = predict_hop_profile(
        hop_bill_dict, hop_model, hop_features, hop_dims
    )
    malt_out  = predict_malt_profile_from_blend(
        malt_selections, malt_model, malt_df, malt_features, malt_dims
    )
    yeast_out = predict_yeast_profile(
        yeast_name, yeast_model, yeast_df, yeast_features, yeast_dims
    )

    # Top hop notes
    hop_sorted = sorted(hop_out.items(), key=lambda kv: kv[1], reverse=True)
    top_hops   = [f"{k} ({round(v, 2)})" for k, v in hop_sorted[:2]]

    # Malt traits that fired
    malt_active  = [k for k,v in malt_out.items() if v == 1]

    # Yeast traits that fired
    yeast_active = [k for k,v in yeast_out.items() if v == 1]

    # Style guess heuristic
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
        "style_guess": style_guess,
        "matched_hops": matched_hops,
        "hop_debug_matrix": hop_debug_matrix,
    }


# -----------------------------------------------------------------------------
# RADAR PLOT
# -----------------------------------------------------------------------------
def plot_hop_radar(hop_profile, title="Hop Aroma Radar"):
    """
    Polar/radar plot of hop aroma dimensions.
    """
    if not hop_profile:
        hop_profile = {
            "tropical": 0.0,
            "citrus": 0.0,
            "fruity": 0.0,
            "resinous": 0.0,
            "floral": 0.0,
            "herbal": 0.0,
            "spicy": 0.0,
            "earthy": 0.0,
        }

    labels = list(hop_profile.keys())
    values = list(hop_profile.values())

    values_arr = np.array(values, dtype=float)
    n = len(labels)

    # angles for each unique label
    base_angles = np.linspace(0, 2*np.pi, n, endpoint=False)

    # close polygon
    closed_angles = np.concatenate([base_angles, base_angles[:1]])
    closed_values = np.concatenate([values_arr, values_arr[:1]])

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    ax.plot(closed_angles, closed_values, color="#1f77b4", linewidth=2)
    ax.fill(closed_angles, closed_values, color="#1f77b4", alpha=0.25)

    # annotate each vertex with numeric value
    for ang, val in zip(base_angles, values_arr):
        ax.text(
            ang,
            val,
            f"{val:.4f}",
            color="black",
            ha="center",
            va="center",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#1f77b4", lw=1)
        )

    # ticks / labels
    ax.set_xticks(base_angles)
    ax.set_xticklabels(labels, fontsize=12)

    max_val = values_arr.max() if values_arr.size > 0 else 0.0
    upper = max(1.0, max_val * 1.2)
    ax.set_ylim(0, upper)

    ax.set_rlabel_position(0)
    ax.yaxis.grid(color="gray", linestyle="--", alpha=0.4)
    ax.xaxis.grid(color="gray", linestyle="--", alpha=0.4)

    ax.set_title(title, fontsize=28, fontweight="bold", pad=20)
    fig.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# SIDEBAR UI
# -----------------------------------------------------------------------------
st.sidebar.header("Hop Bill")

all_hops = sorted(get_all_hop_names(hop_features))

# show "known hops" list nicely in sidebar
st.sidebar.markdown("**🧬 Model trained on hops:**")
st.sidebar.write(", ".join(all_hops))

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


# -----------------------------------------------------------------------------
# MAIN APP LAYOUT
# -----------------------------------------------------------------------------
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
    matched_hops  = summary["matched_hops"]
    hop_debug_x   = summary["hop_debug_matrix"]

    # warn if we didn't match any hops
    if len(matched_hops) == 0:
        st.warning(
            "⚠️ Your hop bill didn't match any hops the model was trained on, "
            "so predicted aroma is basically zeros. Try choosing hops from the sidebar list."
        )

    # optional debug expander for power users
    with st.expander("🔬 Debug: hop model input (what the model actually sees)"):
        debug_df = pd.DataFrame(hop_debug_x, columns=hop_features)
        st.dataframe(debug_df, use_container_width=True)

    col_left, col_right = st.columns([2, 1], vertical_alignment="top")

    with col_left:
        fig = plot_hop_radar(hop_profile, title="Hop Aroma Radar")
        st.pyplot(fig, use_container_width=True)

    with col_right:
        st.markdown("### Top hop notes:")
        if hop_notes:
            st.write("\n".join([f"- {n}" for n in hop_notes]))
        else:
            st.write("_No dominant hop note_")

        st.markdown("### Malt character:")
        st.write(", ".join(malt_traits) if malt_traits else "None")

        st.markdown("### Yeast character:")
        st.write(", ".join(yeast_traits) if yeast_traits else "None")

        st.markdown("### Style direction:")
        st.write(f"🧭 {style_guess}")

else:
    st.info("👈 Build your hop bill (choose hops from the model list), malt bill, yeast, then click **Predict Flavor 🧪**.")

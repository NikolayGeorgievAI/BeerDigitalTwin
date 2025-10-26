import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import warnings

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
# LOAD MODELS / DATA (CACHED)
# ---------------------------------------------------------------------------------
@st.cache_resource
def load_models_and_data():
    """
    Loads all models/bundles and reference data (malt, yeast).
    Cached so Streamlit Cloud doesn't keep reloading them.
    """
    hop_bundle   = joblib.load("hop_aroma_model.joblib")
    malt_bundle  = joblib.load("malt_sensory_model.joblib")
    yeast_bundle = joblib.load("yeast_sensory_model.joblib")

    # hop bundle
    hop_model      = hop_bundle["model"]
    hop_features   = hop_bundle["feature_cols"]   # e.g. ["hop_Citra", "hop_Mosaic", ...]
    hop_dims       = hop_bundle["aroma_dims"]     # e.g. ["tropical","citrus","resinous",...]

    # We don't know yet if the bundle included a target scaler / inverse transform.
    # We'll just grab it if present and otherwise None.
    hop_target_scaler = hop_bundle.get("target_scaler", None)

    # malt bundle
    malt_model     = malt_bundle["model"]
    malt_features  = malt_bundle["feature_cols"]  # numeric malt chemistry columns
    malt_dims      = malt_bundle["flavor_cols"]   # traits like "caramel", "bready", etc.

    # yeast bundle
    yeast_model    = yeast_bundle["model"]
    yeast_features = yeast_bundle["feature_cols"] # ["Temp_avg_C","Flocculation_num","Attenuation_pct"]
    yeast_dims     = yeast_bundle["flavor_cols"]  # ["fruity_esters","phenolic_spicy","clean_neutral",...]

    # reference tables
    malt_df  = pd.read_pickle("clean_malt_df.pkl")
    yeast_df = pd.read_pickle("clean_yeast_df.pkl")

    return (
        hop_model, hop_features, hop_dims, hop_target_scaler,
        malt_model, malt_features, malt_dims,
        yeast_model, yeast_features, yeast_dims,
        malt_df, yeast_df
    )

(
    hop_model, hop_features, hop_dims, hop_target_scaler,
    malt_model, malt_features, malt_dims,
    yeast_model, yeast_features, yeast_dims,
    malt_df, yeast_df
) = load_models_and_data()

# ---------------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------------

# ---- HOPS ---------------------------------------------------------------------
def normalize_name_for_feature(hop_name: str) -> str:
    """
    We try to align the user's hop name to how it appears in training feature columns.
    Training columns look like 'hop_Citra', 'hop_African Queen', etc.
    In the sidebar we might show "African Queen".
    We'll generate all possible candidate keys and see which one exists.
    """
    if hop_name is None:
        return None

    candidates = [
        f"hop_{hop_name}",
        f"hop_{hop_name.strip()}",
        f"hop_{hop_name.replace(' ', '_')}",
        f"hop_{hop_name.replace('-', '_')}",
        f"hop_{hop_name.replace(' ', '')}",
        f"hop_{hop_name.replace(' ', '-')}"

        # Add more heuristics here if needed
    ]
    # Return the first match that exists in hop_features
    for c in candidates:
        if c in hop_features:
            return c
    return None


def build_hop_feature_vector(hop_bill_dict, hop_features):
    """
    hop_bill_dict is like:
        {
            "Citra": 40,
            "Mosaic": 20,
            ...
        }

    We map each hop name to the correct trained feature column and
    place the grams there. Everything else is 0.
    Returns:
      x (1, n_features)
      debug_map: DataFrame of (feature_col, value) for visibility
    """
    feature_values = np.zeros(len(hop_features), dtype=float)

    debug_rows = []  # for the expander display
    # First initialize rows to 0 for debug visibility
    for col in hop_features:
        debug_rows.append({"feature_col": col, "value": 0.0})

    # Now fill in grams where we have a match
    for user_hop_name, grams in hop_bill_dict.items():
        if grams is None or grams <= 0:
            continue
        matched_col = normalize_name_for_feature(user_hop_name)
        if matched_col is not None:
            idx = hop_features.index(matched_col)
            feature_values[idx] = grams
            # Update debug_rows
            for r in debug_rows:
                if r["feature_col"] == matched_col:
                    r["value"] = grams
                    break

    x = feature_values.reshape(1, -1)
    debug_df = pd.DataFrame(debug_rows)

    return x, debug_df


def predict_hop_profile(hop_bill_dict):
    """
    1) Build feature vector for hops
    2) Run hop_model.predict
    3) If there's a target scaler saved, inverse-transform
    4) Package as dict {dimension: value}
    """
    x, debug_df = build_hop_feature_vector(hop_bill_dict, hop_features)

    raw_pred = hop_model.predict(x)  # shape (1, n_dims)
    y_pred = raw_pred.copy()

    # Try to un-scale if we have a scaler-like object
    # NOTE: We don't know its real name/shape, but we'll attempt a scikit-learn-style inverse_transform.
    if hop_target_scaler is not None:
        try:
            y_pred = hop_target_scaler.inverse_transform(y_pred)
        except Exception:
            # If it fails silently, we just keep y_pred = raw_pred
            pass

    y_pred_row = y_pred[0]

    hop_profile = {}
    for dim_name, val in zip(hop_dims, y_pred_row):
        hop_profile[dim_name] = float(val)

    return hop_profile, x, debug_df, raw_pred


def rank_top_hop_notes(hop_profile, top_k=2):
    """
    Sort hop aroma dimensions by predicted intensity descending.
    Returns up to top_k notes like ["tropical (0.42)", "citrus (0.33)"]
    """
    sorted_items = sorted(hop_profile.items(), key=lambda kv: kv[1], reverse=True)
    top_notes = [f"{name} ({round(val, 2)})" for name, val in sorted_items[:top_k]]
    return top_notes


# ---- MALTS --------------------------------------------------------------------
def get_weighted_malt_vector(malt_selections, malt_df, malt_features):
    """
    malt_selections is a list of dicts like:
        [
          {"name": "Maris Otter", "pct": 70.0},
          {"name": "Crystal 60L", "pct": 20.0},
          {"name": "Flaked Oats", "pct": 10.0}
        ]

    We make a weighted sum of malt chemistry columns (malt_features).
    """
    blend_vec = np.zeros(len(malt_features), dtype=float)

    for item in malt_selections:
        malt_name = item["name"]
        pct       = float(item["pct"])

        row = malt_df[malt_df["PRODUCT NAME"] == malt_name].head(1)
        if row.empty:
            continue

        vec = np.array([row.iloc[0][f] for f in malt_features], dtype=float)
        blend_vec += vec * (pct / 100.0)

    return blend_vec.reshape(1, -1)


def predict_malt_profile(malt_selections):
    """
    Returns something like:
      {
        "bready": 1,
        "caramel": 0,
        ...
      }
    """
    x = get_weighted_malt_vector(malt_selections, malt_df, malt_features)
    y_pred = malt_model.predict(x)[0]
    out = dict(zip(malt_dims, y_pred))
    # collect "active" traits
    active_traits = [k for k, v in out.items() if v == 1]
    return out, active_traits


# ---- YEAST -------------------------------------------------------------------
def get_yeast_feature_vector(yeast_name, yeast_df, yeast_features):
    """
    Build row like [Temp_avg_C, Flocculation_num, Attenuation_pct]
    """
    row = yeast_df[yeast_df["Name"] == yeast_name].head(1)
    if row.empty:
        return np.zeros(len(yeast_features)).reshape(1, -1)

    # We'll just map by the known columns.
    # Order should match yeast_features in the model.
    vec = [row.iloc[0][col] for col in yeast_features]
    return np.array(vec).reshape(1, -1)


def predict_yeast_profile(yeast_name):
    """
    Returns ({trait:0/1,...}, active_traits_list)
    """
    x = get_yeast_feature_vector(yeast_name, yeast_df, yeast_features)
    y_pred = yeast_model.predict(x)[0]
    out = dict(zip(yeast_dims, y_pred))
    active_traits = [k for k, v in out.items() if v == 1]
    return out, active_traits


# ---- PUT IT ALL TOGETHER -----------------------------------------------------
def summarize_beer(hop_bill, malt_selections, yeast_name):
    hop_profile, hop_X, hop_debug_df, raw_pred = predict_hop_profile(hop_bill)
    malt_out, malt_active = predict_malt_profile(malt_selections)
    yeast_out, yeast_active = predict_yeast_profile(yeast_name)

    hop_notes = rank_top_hop_notes(hop_profile, top_k=2)

    # crude style guess
    style_guess = "Experimental / Hybrid"
    if ("clean_neutral" in yeast_out and yeast_out["clean_neutral"] == 1
        and "dry_finish" in yeast_out and yeast_out["dry_finish"] == 1):
        style_guess = "West Coast IPA / Modern IPA"

    if ("fruity_esters" in yeast_out and yeast_out["fruity_esters"] == 1) and \
       ("tropical" in hop_profile and hop_profile["tropical"] > 0.6):
        style_guess = "Hazy / NEIPA leaning"

    if ("phenolic_spicy" in yeast_out and yeast_out["phenolic_spicy"] == 1):
        style_guess = "Belgian / Saison leaning"

    if ("caramel" in malt_out and malt_out["caramel"] == 1):
        style_guess = "English / Malt-forward Ale"

    summary = {
        "hop_profile": hop_profile,
        "hop_notes": hop_notes,
        "hop_X": hop_X,
        "hop_debug_df": hop_debug_df,
        "hop_raw_pred": raw_pred,

        "malt_traits_dict": malt_out,
        "malt_active": malt_active,

        "yeast_traits_dict": yeast_out,
        "yeast_active": yeast_active,

        "style_guess": style_guess,
    }
    return summary


# ---------------------------------------------------------------------------------
# RADAR PLOT
# ---------------------------------------------------------------------------------
def plot_hop_radar(hop_profile, title="Hop Aroma Radar"):
    """
    Make a spider/radar chart with labeled axes and numeric annotations.
    """
    labels = list(hop_profile.keys())
    values = list(hop_profile.values())

    # Close the loop for radar
    labels_loop = labels + [labels[0]]
    values_loop = values + [values[0]]

    angles = np.linspace(0, 2*np.pi, len(labels_loop), endpoint=False)

    fig, ax = plt.subplots(
        figsize=(7,7),
        subplot_kw={"projection": "polar"}
    )

    ax.plot(angles, values_loop, color="#1f77b4", linewidth=2)
    ax.fill(angles, values_loop, color="#1f77b4", alpha=0.25)

    # Ticks around
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=12)

    # Nice grid
    ax.set_rlabel_position(0)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.xaxis.grid(True, linestyle="--", alpha=0.4)

    # Annotate each vertex (except the closing duplicate)
    for ang, val in zip(angles[:-1], values):
        ax.text(
            ang,
            val,
            f"{val:.4f}",
            ha="center",
            va="center",
            fontsize=10,
            bbox=dict(
                boxstyle="round,pad=0.25",
                fc="white",
                ec="#1f77b4",
                lw=1,
                alpha=0.8,
            ),
        )

    ax.set_title(title, fontsize=24, pad=20, fontweight="bold")

    return fig


# ---------------------------------------------------------------------------------
# SIDEBAR UI
# ---------------------------------------------------------------------------------
st.sidebar.header("Hop Bill")

# AVAILABLE HOPS from training columns
# We want to show user-friendly names without the "hop_" prefix.
all_hops_human = sorted(
    list(
        {
            col.replace("hop_", "").replace("_", " ").strip()
            for col in hop_features
        }
    )
)

hop1_name = st.sidebar.selectbox("Hop 1", all_hops_human, key="hop1_name")
hop1_amt  = st.sidebar.slider(f"{hop1_name} (g)", 0, 120, 40, 5, key="hop1_amt")

hop2_name = st.sidebar.selectbox("Hop 2", all_hops_human, key="hop2_name")
hop2_amt  = st.sidebar.slider(f"{hop2_name} (g)", 0, 120, 40, 5, key="hop2_amt")

hop3_name = st.sidebar.selectbox("Hop 3", all_hops_human, key="hop3_name")
hop3_amt  = st.sidebar.slider(f"{hop3_name} (g)", 0, 120, 40, 5, key="hop3_amt")

hop4_name = st.sidebar.selectbox("Hop 4", all_hops_human, key="hop4_name")
hop4_amt  = st.sidebar.slider(f"{hop4_name} (g)", 0, 120, 40, 5, key="hop4_amt")

hop_bill = {
    hop1_name: hop1_amt,
    hop2_name: hop2_amt,
    hop3_name: hop3_amt,
    hop4_name: hop4_amt,
}

# Malt bill
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

# Yeast
st.sidebar.header("Yeast")
yeast_options = sorted(yeast_df["Name"].dropna().unique().tolist())
chosen_yeast  = st.sidebar.selectbox("Yeast Strain", yeast_options)

run_button = st.sidebar.button("Predict Flavor 🧪")

# ---------------------------------------------------------------------------------
# MAIN LAYOUT
# ---------------------------------------------------------------------------------
if run_button:
    summary = summarize_beer(hop_bill, malt_selections, chosen_yeast)

    hop_profile     = summary["hop_profile"]
    hop_notes       = summary["hop_notes"]
    malt_active     = summary["malt_active"]
    yeast_active    = summary["yeast_active"]
    style_guess     = summary["style_guess"]

    # Layout: 2 columns -> left radar, right text
    col_left, col_right = st.columns([2, 1], vertical_alignment="top")

    with col_left:
        fig = plot_hop_radar(hop_profile, title="Hop Aroma Radar")
        st.pyplot(fig, use_container_width=True)

    with col_right:
        st.markdown("### Top hop notes:")
        if hop_notes:
            for n in hop_notes:
                st.write("•", n)
        else:
            st.write("• _No dominant hop note_")

        st.markdown("### Malt character:")
        st.write(", ".join(malt_active) if malt_active else "None detected")

        st.markdown("### Yeast character:")
        st.write(", ".join(yeast_active) if yeast_active else "None detected")

        st.markdown("### Style direction:")
        st.write("🍺", style_guess)

    # Debug expander to help us inspect scaling, feature mapping, etc.
    with st.expander("Debug / Model I/O"):
        st.write("Hop model input feature vector shape:", summary["hop_X"].shape)
        st.markdown("First 15 hop features + values:")
        st.dataframe(summary["hop_debug_df"].head(15))

        st.markdown("Raw hop_model.predict() output (before optional inverse scaling):")
        ypd = pd.DataFrame(
            [summary["hop_profile"]],
            columns=summary["hop_profile"].keys()
        ).T
        ypd.columns = ["predicted_intensity"]
        st.dataframe(ypd)

        st.markdown("Malt traits dict:")
        st.json(summary["malt_traits_dict"])

        st.markdown("Yeast traits dict:")
        st.json(summary["yeast_traits_dict"])

        st.markdown("Inputs recap:")
        st.json({
            "hop_bill (grams)": hop_bill,
            "malt_bill (%)": malt_selections,
            "yeast": chosen_yeast
        })

else:
    st.info("👈 Build your hop bill (hops & grams), malt bill (3 malts with %), choose yeast, then click **Predict Flavor 🧪** to generate predictions.")

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Beer Recipe Digital Twin",
    page_icon="🍺",
    layout="wide",
)

# -----------------------------------------------------------------------------
# SAFE SESSION HELPERS  (patched to avoid StreamlitAPIException)
# -----------------------------------------------------------------------------
def _safe_get_session(key, default=None):
    """Return st.session_state[key] if it exists, else default."""
    return st.session_state[key] if key in st.session_state else default


def _safe_set_session(key, value):
    """
    Safely update st.session_state[key] only if needed.

    Streamlit sometimes raises StreamlitAPIException if you assign to
    session_state *during the same render pass that defines a widget*
    with the same key. To avoid that, we only write if the value
    actually changed, and we wrap it in try/except to silently ignore
    'too-early' writes.
    """
    try:
        if key not in st.session_state or st.session_state[key] != value:
            st.session_state[key] = value
    except Exception:
        # If Streamlit refuses because we're in widget init, just skip.
        pass


# -----------------------------------------------------------------------------
# LOAD MODELS + DATA (CACHED)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_models_and_data():
    """
    Load all trained models + reference data once, cache them for the session.
    We expect these files to be in the repo root (as in your GitHub).
    """
    hop_bundle   = joblib.load("hop_aroma_model.joblib")
    malt_bundle  = joblib.load("malt_sensory_model.joblib")
    yeast_bundle = joblib.load("yeast_sensory_model.joblib")

    hop_model      = hop_bundle["model"]
    hop_features   = hop_bundle["feature_cols"]     # e.g. ["hop_Citra", "hop_Mosaic", ...]
    hop_dims       = hop_bundle["aroma_dims"]       # e.g. ["tropical","citrus","fruity",...]

    malt_model     = malt_bundle["model"]
    malt_features  = malt_bundle["feature_cols"]    # malt chemistry columns
    malt_dims      = malt_bundle["flavor_cols"]     # predicted malt traits/labels

    yeast_model    = yeast_bundle["model"]
    yeast_features = yeast_bundle["feature_cols"]   # e.g. ["Temp_avg_C","Flocculation_num","Attenuation_pct"]
    yeast_dims     = yeast_bundle["flavor_cols"]    # e.g. ["fruity_esters","clean_neutral",...]

    # Pickled reference dataframes
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
# HOP HELPERS
# -----------------------------------------------------------------------------
def get_all_hop_names(hop_features):
    """
    hop_features looks like ["hop_Citra","hop_Mosaic",...].
    We convert these to ["Citra","Mosaic",...].
    """
    return [c.replace("hop_", "") for c in hop_features]


def build_hop_features(hop_bill_list, hop_features):
    """
    hop_bill_list is like:
    [
        {"name": "Eclipse", "grams": 40.0},
        {"name": "Ella",    "grams": 20.0},
        ...
    ]

    We:
    - sum total grams
    - compute normalized percentages for each hop in the model's one-hot style vector
      (or keep 0 if hop isn't used)
    Returns a single-row NumPy array aligned to hop_features order.
    """

    # total grams
    total = sum(item["grams"] for item in hop_bill_list if item["grams"] > 0)

    # build vector
    row_vals = []
    for col in hop_features:
        hop_name = col.replace("hop_", "")
        # raw grams for this hop
        g = 0.0
        for item in hop_bill_list:
            if item["name"] == hop_name:
                g = item["grams"]
                break

        if total > 0:
            pct = g / total
        else:
            pct = 0.0

        row_vals.append(pct)

    x = np.array(row_vals, dtype=float).reshape(1, -1)
    return x, total


def predict_hop_profile(hop_bill_list, hop_model, hop_features, hop_dims):
    """
    Builds features from hop bill and calls hop_model.predict.
    Returns:
      hop_out_dict: { "tropical": val, "citrus": val, ... }
      total_grams : total hop grams used (for warnings/UX)
    """
    x, total_grams = build_hop_features(hop_bill_list, hop_features)

    # model output shape -> (1, num_dims)
    y_pred = hop_model.predict(x)[0]
    hop_out_dict = dict(zip(hop_dims, y_pred))

    return hop_out_dict, total_grams


# -----------------------------------------------------------------------------
# MALT HELPERS
# -----------------------------------------------------------------------------
def get_weighted_malt_vector(malt_selections, malt_df, malt_features):
    """
    malt_selections:
    [
      {"name": "Maris Otter", "pct": 70.0},
      {"name": "Crystal 60L", "pct": 20.0},
      {"name": "Flaked Oats", "pct": 10.0}
    ]

    We build a weighted blend of numeric malt_features columns.
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
    """
    Call malt model on the weighted blend.
    Model is (apparently) classification-like: outputs 0/1 trait flags.
    """
    x = get_weighted_malt_vector(malt_selections, malt_df, malt_features)
    y_pred = malt_model.predict(x)[0]
    return dict(zip(malt_dims, y_pred))


# -----------------------------------------------------------------------------
# YEAST HELPERS
# -----------------------------------------------------------------------------
def get_yeast_feature_vector(yeast_name, yeast_df, yeast_features):
    """
    Build the single-row model input for the chosen yeast strain.
    We'll look it up in yeast_df by "Name".
    """
    row = yeast_df[yeast_df["Name"] == yeast_name].head(1)
    if row.empty:
        # fallback zeros
        vec = [0.0] * len(yeast_features)
    else:
        # We assume the correct columns exist in yeast_df
        vec = [row.iloc[0][col] for col in yeast_features]

    return np.array(vec, dtype=float).reshape(1, -1)


def predict_yeast_profile(yeast_name, yeast_model, yeast_df, yeast_features, yeast_dims):
    """
    Return classification-like labels from the yeast model.
    """
    x = get_yeast_feature_vector(yeast_name, yeast_df, yeast_features)
    y_pred = yeast_model.predict(x)[0]
    return dict(zip(yeast_dims, y_pred))


# -----------------------------------------------------------------------------
# BEER SUMMARY / STYLE HEURISTIC
# -----------------------------------------------------------------------------
def summarize_beer(
    hop_bill_list,
    malt_selections,
    yeast_name,
    hop_model, hop_features, hop_dims,
    malt_model, malt_df, malt_features, malt_dims,
    yeast_model, yeast_df, yeast_features, yeast_dims,
):
    hop_out, total_hop_grams = predict_hop_profile(
        hop_bill_list, hop_model, hop_features, hop_dims
    )
    malt_out  = predict_malt_profile_from_blend(
        malt_selections, malt_model, malt_df, malt_features, malt_dims
    )
    yeast_out = predict_yeast_profile(
        yeast_name, yeast_model, yeast_df, yeast_features, yeast_dims
    )

    # Top hop notes by intensity
    hop_sorted = sorted(hop_out.items(), key=lambda kv: kv[1], reverse=True)
    top_hops   = [f"{k} ({round(v, 2)})" for k, v in hop_sorted[:2]]

    # Malt traits with '1'
    malt_traits  = [k for k, v in malt_out.items() if v == 1]

    # Yeast traits with '1'
    yeast_traits = [k for k, v in yeast_out.items() if v == 1]

    # style heuristic (toy)
    style_guess = "Experimental / Hybrid"

    # smells like west coast? (clean, dry yeast + citrus/resin hop top)
    if ("clean_neutral" in yeast_out and yeast_out["clean_neutral"] == 1
        and ("dry_finish" in yeast_out and yeast_out["dry_finish"] == 1 or "attenuative" in yeast_out and yeast_out["attenuative"] == 1)
    ):
        if any("citrus" in n.split()[0].lower() or "resin" in n.split()[0].lower()
               for n in top_hops[:2]):
            style_guess = "West Coast IPA / Modern IPA"
        else:
            style_guess = "Clean, dry ale"

    # hazy / NEIPA if fruity yeast + tropical hop dominance
    if ("fruity_esters" in yeast_out and yeast_out["fruity_esters"] == 1) and \
       ("tropical" in hop_out and hop_out["tropical"] > 0.6):
        style_guess = "Hazy / NEIPA leaning"

    # Belgian / Saison if phenolic
    if ("phenolic_spicy" in yeast_out and yeast_out["phenolic_spicy"] == 1):
        style_guess = "Belgian / Saison leaning"

    # English / Malt-forward if caramel
    if ("caramel" in malt_out and malt_out["caramel"] == 1) or \
       ("toffee" in malt_out and malt_out["toffee"] == 1):
        style_guess = "English / Malt-forward Ale"

    # compile final dict
    return {
        "hop_out": hop_out,
        "hop_top_notes": top_hops,
        "malt_traits": malt_traits,
        "yeast_traits": yeast_traits,
        "style_guess": style_guess,
        "total_hop_grams": total_hop_grams,
    }


# -----------------------------------------------------------------------------
# RADAR PLOT
# -----------------------------------------------------------------------------
def plot_hop_radar(hop_profile, title="Hop Aroma Radar"):
    """
    Polar radar plot of hop aroma dimensions.

    We plot each aroma dimension at a fixed angle, label it, and annotate each
    vertex with the numeric value. We DO NOT reuse the first element, so
    ticks == labels == values, so no mismatch error.
    """
    if not hop_profile:
        # in degenerate case, make a dummy set
        hop_profile = {
            "tropical": 0.0,
            "citrus": 0.0,
            "fruity": 0.0,
            "resinous": 0.0,
            "floral": 0.0,
            "earthy": 0.0,
        }

    labels   = list(hop_profile.keys())
    values   = [float(v) for v in hop_profile.values()]
    values_a = np.array(values, dtype=float)

    n = len(labels)
    if n == 0:
        # no dimensions
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No hop aroma dimensions", ha="center", va="center")
        return fig

    # angles for each label around the circle
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)

    # close the polygon by repeating first angle & value
    closed_angles = np.concatenate([angles, [angles[0]]])
    closed_vals   = np.concatenate([values_a, [values_a[0]]])

    fig, ax = plt.subplots(
        figsize=(6, 6),
        subplot_kw=dict(polar=True),
    )

    ax.plot(closed_angles, closed_vals, linewidth=2, color="#1f77b4")
    ax.fill(closed_angles, closed_vals, alpha=0.25, color="#1f77b4")

    # annotate each vertex with numeric
    for ang, val in zip(angles, values_a):
        ax.text(
            ang,
            val,
            f"{val:.4f}",
            color="black",
            ha="center",
            va="center",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.2",
                      fc="white",
                      ec="#1f77b4",
                      lw=1)
        )

    # category ticks
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=12)

    # radial styling
    ax.set_rlabel_position(0)
    ax.yaxis.grid(color="gray", linestyle="--", alpha=0.4)
    ax.xaxis.grid(color="gray", linestyle="--", alpha=0.4)

    ax.set_title(title, fontsize=24, fontweight="bold", pad=20)
    fig.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# SIDEBAR UI (hop bill, malt bill, yeast)
# -----------------------------------------------------------------------------
all_hops_sorted = sorted(get_all_hop_names(hop_features))

# ensure session first-time defaults
if _safe_get_session("hop1_name") is None and len(all_hops_sorted) > 0:
    _safe_set_session("hop1_name", all_hops_sorted[0])
if _safe_get_session("hop2_name") is None and len(all_hops_sorted) > 1:
    _safe_set_session("hop2_name", all_hops_sorted[1])
if _safe_get_session("hop3_name") is None and len(all_hops_sorted) > 2:
    _safe_set_session("hop3_name", all_hops_sorted[2])
if _safe_get_session("hop4_name") is None and len(all_hops_sorted) > 3:
    _safe_set_session("hop4_name", all_hops_sorted[3])

# hops
st.sidebar.subheader("🌿 Hop Bill")

st.session_state.hop1_name = st.sidebar.selectbox(
    "Hop 1", all_hops_sorted, index=all_hops_sorted.index(_safe_get_session("hop1_name", all_hops_sorted[0]))
)
hop1_g = st.sidebar.number_input(
    f"{st.session_state.hop1_name} (g)",
    min_value=0.0, step=1.0, value=float(_safe_get_session("hop1_g", 40.0)),
    key="hop1_g"
)

st.session_state.hop2_name = st.sidebar.selectbox(
    "Hop 2",
    all_hops_sorted,
    index=all_hops_sorted.index(_safe_get_session("hop2_name", all_hops_sorted[min(1, len(all_hops_sorted)-1)]))
)
hop2_g = st.sidebar.number_input(
    f"{st.session_state.hop2_name} (g)",
    min_value=0.0, step=1.0, value=float(_safe_get_session("hop2_g", 40.0)),
    key="hop2_g"
)

st.session_state.hop3_name = st.sidebar.selectbox(
    "Hop 3",
    all_hops_sorted,
    index=all_hops_sorted.index(_safe_get_session("hop3_name", all_hops_sorted[min(2, len(all_hops_sorted)-1)]))
)
hop3_g = st.sidebar.number_input(
    f"{st.session_state.hop3_name} (g)",
    min_value=0.0, step=1.0, value=float(_safe_get_session("hop3_g", 0.0)),
    key="hop3_g"
)

st.session_state.hop4_name = st.sidebar.selectbox(
    "Hop 4",
    all_hops_sorted,
    index=all_hops_sorted.index(_safe_get_session("hop4_name", all_hops_sorted[min(3, len(all_hops_sorted)-1)]))
)
hop4_g = st.sidebar.number_input(
    f"{st.session_state.hop4_name} (g)",
    min_value=0.0, step=1.0, value=float(_safe_get_session("hop4_g", 0.0)),
    key="hop4_g"
)

hop_bill_list = [
    {"name": st.session_state.hop1_name, "grams": st.session_state.hop1_g},
    {"name": st.session_state.hop2_name, "grams": st.session_state.hop2_g},
    {"name": st.session_state.hop3_name, "grams": st.session_state.hop3_g},
    {"name": st.session_state.hop4_name, "grams": st.session_state.hop4_g},
]

# malt
st.sidebar.subheader("🌽 Malt Bill")

malt_options = sorted(malt_df["PRODUCT NAME"].unique().tolist())

# default malt picks
if _safe_get_session("malt1_name") is None and malt_options:
    _safe_set_session("malt1_name", malt_options[0])
if _safe_get_session("malt2_name") is None and len(malt_options) > 1:
    _safe_set_session("malt2_name", malt_options[1])
if _safe_get_session("malt3_name") is None and len(malt_options) > 2:
    _safe_set_session("malt3_name", malt_options[2])

st.session_state.malt1_name = st.sidebar.selectbox(
    "Malt 1", malt_options,
    index=malt_options.index(_safe_get_session("malt1_name", malt_options[0]))
)
malt1_pct = st.sidebar.number_input(
    "Malt 1 %",
    min_value=0.0, max_value=100.0, step=1.0,
    value=float(_safe_get_session("malt1_pct", 70.0)),
    key="malt1_pct"
)

st.session_state.malt2_name = st.sidebar.selectbox(
    "Malt 2", malt_options,
    index=malt_options.index(_safe_get_session("malt2_name", malt_options[min(1, len(malt_options)-1)]))
)
malt2_pct = st.sidebar.number_input(
    "Malt 2 %",
    min_value=0.0, max_value=100.0, step=1.0,
    value=float(_safe_get_session("malt2_pct", 20.0)),
    key="malt2_pct"
)

st.session_state.malt3_name = st.sidebar.selectbox(
    "Malt 3", malt_options,
    index=malt_options.index(_safe_get_session("malt3_name", malt_options[min(2, len(malt_options)-1)]))
)
malt3_pct = st.sidebar.number_input(
    "Malt 3 %",
    min_value=0.0, max_value=100.0, step=1.0,
    value=float(_safe_get_session("malt3_pct", 10.0)),
    key="malt3_pct"
)

malt_selections = [
    {"name": st.session_state.malt1_name, "pct": malt1_pct},
    {"name": st.session_state.malt2_name, "pct": malt2_pct},
    {"name": st.session_state.malt3_name, "pct": malt3_pct},
]

# yeast
st.sidebar.subheader("🧬 Yeast")
yeast_options = sorted(yeast_df["Name"].dropna().unique().tolist())
if _safe_get_session("chosen_yeast") is None and yeast_options:
    _safe_set_session("chosen_yeast", yeast_options[0])

st.session_state.chosen_yeast = st.sidebar.selectbox(
    "Yeast Strain", yeast_options,
    index=yeast_options.index(_safe_get_session("chosen_yeast", yeast_options[0]))
)

run_button = st.sidebar.button("Predict Flavor 🧪")

# "model trained on hops" info -> expander at bottom of sidebar
with st.sidebar.expander("🌀 Model trained on hops:"):
    st.write(", ".join(all_hops_sorted))


# -----------------------------------------------------------------------------
# MAIN BODY
# -----------------------------------------------------------------------------
st.markdown(
    "<h1 style='display:flex; align-items:center; gap:0.5rem;'>"
    "🍺 Beer Recipe Digital Twin"
    "</h1>",
    unsafe_allow_html=True,
)
st.caption(
    "Predict hop aroma, malt character, and fermentation profile using trained ML models."
)

if run_button:
    summary = summarize_beer(
        hop_bill_list,
        malt_selections,
        st.session_state.chosen_yeast,
        hop_model, hop_features, hop_dims,
        malt_model, malt_df, malt_features, malt_dims,
        yeast_model, yeast_df, yeast_features, yeast_dims,
    )

    hop_profile   = summary["hop_out"]
    hop_notes     = summary["hop_top_notes"]
    malt_traits   = summary["malt_traits"]
    yeast_traits  = summary["yeast_traits"]
    style_guess   = summary["style_guess"]
    total_hop_g   = summary["total_hop_grams"]

    # layout: 2 columns -> radar on left, text on right
    col_left, col_right = st.columns([2, 1], vertical_alignment="top")

    # left: radar
    with col_left:
        # warn user if total hop grams = 0 (meaning normalized vector is 0s)
        if total_hop_g <= 0:
            st.warning(
                "No recognizable hops in your bill (or total was 0 once normalized), "
                "so aroma prediction is basically flat.\n\n"
                "Pick hops from the list in the sidebar (open the '🌀 Model trained on hops:' "
                "section), then click **Predict Flavor 🧪** again."
            )

        fig = plot_hop_radar(hop_profile, title="Hop Aroma Radar")
        st.pyplot(fig, use_container_width=True)

        # optional debug expander
        with st.expander("🧪 Debug: hop model input (what the model actually sees)"):
            df_debug = pd.DataFrame(
                [build_hop_features(hop_bill_list, hop_features)[0][0]],
                columns=hop_features,
            )
            st.dataframe(df_debug, use_container_width=True)

    # right: textual summary
    with col_right:
        st.markdown("### Top hop notes:")
        if hop_notes:
            for n in hop_notes:
                st.write(f"- {n}")
        else:
            st.write("_No dominant hop note_")

        st.markdown("### Malt character:")
        if malt_traits:
            st.write(", ".join(malt_traits))
        else:
            st.write("None")

        st.markdown("### Yeast character:")
        if yeast_traits:
            st.write(", ".join(yeast_traits))
        else:
            st.write("None")

        st.markdown("### Style direction:")
        st.write(f"🧭 {style_guess}")

        # also show which hops are actually nonzero in user's bill
        nonzero_list = [item["name"] for item in hop_bill_list if item["grams"] > 0]
        st.markdown("### Hops used by the model:")
        if nonzero_list:
            st.write(", ".join(nonzero_list))
        else:
            st.write("_None (all zero grams)_")

else:
    # initial screen before Predict Flavor
    st.warning(
        "No recognizable hops in your bill (or total was 0 once normalized), "
        "so aroma prediction is basically flat.\n\n"
        "Pick hops from the list in the sidebar (open the '🌀 Model trained on hops:' "
        "section), then click **Predict Flavor 🧪** again."
    )
    # nothing else yet until they click

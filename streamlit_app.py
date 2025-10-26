import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# Streamlit page config
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Beer Recipe Digital Twin",
    page_icon="🍺",
    layout="wide"
)

# Small tweak so matplotlib fonts don't look gigantic in Streamlit
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 26,
    "axes.labelsize": 14
})

# ------------------------------------------------------------------
# Cached loaders
# ------------------------------------------------------------------
@st.cache_resource
def load_models_and_data():
    """
    Load + cache models and reference data.
    Assumptions:
      hop_aroma_model.joblib -> {
          "model": fitted_regressor,
          "feature_cols": ["hop_Citra", "hop_Simcoe", ...],
          "aroma_dims": ["fruity","citrus","tropical","resinous",
                         "floral","herbal","spicy","earthy"]
      }

      malt_sensory_model.joblib -> {
          "model": classifier OR multilabel,
          "feature_cols": [... malt chemistry columns ...],
          "flavor_cols": ["bready","sweet_malt","caramel",...]
      }

      yeast_sensory_model.joblib -> {
          "model": classifier,
          "feature_cols": ["Temp_avg_C","Flocculation_num","Attenuation_pct"],
          "flavor_cols": ["clean_neutral","fruity_esters","phenolic_spicy",...]
      }

      clean_malt_df.pkl -> rows of malts w/ columns including feature_cols
      clean_yeast_df.pkl -> rows of yeast strains w/ columns:
          Name, Temp_avg_C, Flocculation_num, Attenuation_pct, ...
    """
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

    malt_df = pd.read_pickle("clean_malt_df.pkl")
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


# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------

def get_all_hops_from_features(hop_features):
    """
    hop_features looks like ["hop_Citra","hop_Simcoe",...]
    we return ["Citra","Simcoe",...]
    """
    names = []
    for col in hop_features:
        # only strip the first "hop_" prefix
        if col.lower().startswith("hop_"):
            names.append(col[4:])
        else:
            names.append(col)
    return sorted(names)


def build_hop_feature_row(hop_bill_dict, hop_features):
    """
    hop_bill_dict = {"Citra": 50, "Simcoe": 40, ...} grams
    We create a 1-row numpy array in the SAME ORDER as hop_features.
    We'll also normalize hop names to lowercase for matching.
    """
    hop_bill_lower = {k.lower(): float(v) for k, v in hop_bill_dict.items() if float(v) > 0}

    row_vals = []
    for feat in hop_features:
        feat_name = feat[4:] if feat.lower().startswith("hop_") else feat
        val = hop_bill_lower.get(feat_name.lower(), 0.0)
        row_vals.append(val)

    # normalize so sum of grams doesn't dominate
    total = sum(row_vals)
    if total > 0:
        row_vals = [v / total for v in row_vals]

    return np.array(row_vals, dtype=float).reshape(1, -1)


def predict_hop_profile(hop_bill_dict):
    """
    Returns dict of aroma dimension intensities.
    If hop_bill is empty/zero, returns zeros safely.
    """
    X = build_hop_feature_row(hop_bill_dict, hop_features)
    y_pred = hop_model.predict(X)[0]
    return {dim: float(val) for dim, val in zip(hop_dims, y_pred)}


def blend_malt_vector(malt_selections):
    """
    malt_selections = [
      {"name":"BEST ALE MALT","pct": 70},
      {"name":"AMBER MALT","pct":20},
      {"name":"BLACK MALT","pct":10},
    ]

    We build a weighted average of malt_features.
    """
    vec = np.zeros(len(malt_features), dtype=float)

    total_pct = sum([float(m["pct"]) for m in malt_selections if float(m["pct"]) > 0])
    if total_pct == 0:
        # no malt? just zeros
        return vec.reshape(1, -1)

    for item in malt_selections:
        name = item["name"]
        pct = float(item["pct"])
        if pct <= 0:
            continue

        row = malt_df[malt_df["PRODUCT NAME"] == name]
        if row.empty:
            continue

        contrib = np.array([row.iloc[0][col] for col in malt_features], dtype=float)
        weight = pct / total_pct
        vec += contrib * weight

    return vec.reshape(1, -1)


def predict_malt_traits(malt_selections):
    X = blend_malt_vector(malt_selections)
    raw = malt_model.predict(X)

    # raw might be:
    # - binary vector for multi-label
    # - single class label
    # We'll try smart interpretation:
    if hasattr(raw, "toarray"):
        raw_arr = raw.toarray()[0]
    else:
        raw_arr = np.array(raw[0]).ravel()

    out = {}
    for dim, val in zip(malt_dims, raw_arr):
        # if it's clearly binary-ish, cast to int
        out[dim] = int(val) if val in [0, 1] or str(val) in ["0", "1"] else float(val)
    return out


def build_yeast_feature_row(yeast_name):
    """
    Use core numeric features from yeast_df.
    If yeast_name not found, return zeros.
    """
    row = yeast_df[yeast_df["Name"] == yeast_name]
    if row.empty:
        return np.zeros(len(yeast_features)).reshape(1, -1)

    return np.array([row.iloc[0][feat] for feat in yeast_features], dtype=float).reshape(1, -1)


def predict_yeast_traits(yeast_name):
    X = build_yeast_feature_row(yeast_name)
    raw = yeast_model.predict(X)

    if hasattr(raw, "toarray"):
        raw_arr = raw.toarray()[0]
    else:
        raw_arr = np.array(raw[0]).ravel()

    out = {}
    for dim, val in zip(yeast_dims, raw_arr):
        out[dim] = int(val) if val in [0, 1] or str(val) in ["0", "1"] else float(val)
    return out


def summarize_style(hop_out, malt_out, yeast_out):
    """
    Very lightweight heuristic for "style direction".
    You can tune this later.
    """
    style_direction = "Experimental / Hybrid"

    # hazy-ish if fruity esters + tropical
    if ("fruity_esters" in yeast_out and yeast_out["fruity_esters"] == 1) and \
       ("tropical" in hop_out and hop_out["tropical"] > 0.5):
        style_direction = "Juicy / Hazy IPA direction"

    # clean + neutral + not much tropical => clean ale / lager-ish
    if yeast_out.get("clean_neutral", 0) == 1 and hop_out.get("tropical", 0) < 0.4:
        style_direction = "Clean / Neutral Ale direction"

    # caramel malt note => malt-forward English-ish
    if malt_out.get("caramel", 0) == 1 or malt_out.get("sweet_malt", 0) == 1:
        style_direction = "Malt-forward / English-inspired"

    return style_direction


def top_two_hop_notes(hop_out):
    """
    Return [(note, value), (note,value)] sorted desc
    """
    if not hop_out:
        return []
    pairs = sorted(hop_out.items(), key=lambda kv: kv[1], reverse=True)
    return pairs[:2]


# ------------------------------------------------------------------
# Plot: spider / radar
# ------------------------------------------------------------------
def make_spider_plot(hop_out_dict):
    """
    Builds a classic 360° radar plot:
    1) order labels -> values
    2) add closing point
    3) polar axes with fill
    """

    # default order for display
    default_order = [
        "fruity", "citrus", "tropical", "earthy",
        "spicy", "herbal", "floral", "resinous"
    ]

    # fill any missing dims with 0.0
    values_core = [float(hop_out_dict.get(dim, 0.0)) for dim in default_order]

    # close polygon
    values_closed = values_core + [values_core[0]]

    # angles for each label
    angles = np.linspace(0, 2 * np.pi, len(default_order), endpoint=False)
    angles_closed = np.concatenate([angles, [angles[0]]])

    fig, ax = plt.subplots(
        figsize=(8, 8),
        subplot_kw=dict(polar=True)
    )
    # light bg so polygon is visible
    ax.set_facecolor("#fafafa")

    # draw polygon fill + outline
    ax.plot(angles_closed, values_closed, color="#1f77b4", linewidth=2)
    ax.fill(angles_closed, values_closed, color="#1f77b4", alpha=0.25)

    # label each vertex with its numeric intensity
    for ang, val in zip(angles, values_core):
        ax.text(
            ang,
            val,
            f"{val:.2f}",
            ha="center",
            va="center",
            fontsize=10,
            bbox=dict(
                boxstyle="round,pad=0.2",
                fc="white",
                ec="#1f77b4",
                lw=1,
            ),
        )

    # category tick labels go on the outer ring
    ax.set_xticks(angles)
    ax.set_xticklabels(default_order, fontsize=12)

    # radial grid styling
    ax.set_rlabel_position(0)
    ax.yaxis.grid(color="gray", linestyle="--", alpha=0.4)
    ax.xaxis.grid(color="gray", linestyle="--", alpha=0.4)

    # dynamic radial limit:
    vmax = max(max(values_core), 0.5)  # avoid super tiny ring when nearly zero
    ax.set_ylim(0, vmax * 1.2)

    ax.set_title("Hop Aroma Radar", fontsize=24, fontweight="bold", pad=20)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------
# Sidebar UI
# ------------------------------------------------------------------
st.sidebar.markdown("## 🧪 Model Inputs")

# --- Hop bill section
st.sidebar.markdown("### Hop Bill (g)")
all_hops = get_all_hops_from_features(hop_features)

# We'll offer up to 4 hop slots
def hop_input(label, default_index=0):
    # protect index range
    idx = default_index if default_index < len(all_hops) else 0
    hop_name = st.sidebar.selectbox(label, all_hops, index=idx)
    amt = st.sidebar.number_input(
        f"{hop_name} (g)", min_value=0.0, max_value=500.0, value=0.0, step=5.0
    )
    return hop_name, amt

hop1_name, hop1_amt = hop_input("Hop 1", default_index=0)
hop2_name, hop2_amt = hop_input("Hop 2", default_index=1)
hop3_name, hop3_amt = hop_input("Hop 3", default_index=2)
hop4_name, hop4_amt = hop_input("Hop 4", default_index=3)

hop_bill = {
    hop1_name: hop1_amt,
    hop2_name: hop2_amt,
    hop3_name: hop3_amt,
    hop4_name: hop4_amt,
}

# --- Malt bill section
st.sidebar.markdown("### Malt Bill")
malt_options = sorted(malt_df["PRODUCT NAME"].unique().tolist())

malt1_name = st.sidebar.selectbox("Malt 1", malt_options, index=0 if len(malt_options) > 0 else 0)
malt1_pct = st.sidebar.number_input("Malt 1 %", min_value=0.0, max_value=100.0,
                                    value=70.0, step=1.0)

malt2_name = st.sidebar.selectbox("Malt 2", malt_options, index=1 if len(malt_options) > 1 else 0)
malt2_pct = st.sidebar.number_input("Malt 2 %", min_value=0.0, max_value=100.0,
                                    value=20.0, step=1.0)

malt3_name = st.sidebar.selectbox("Malt 3", malt_options, index=2 if len(malt_options) > 2 else 0)
malt3_pct = st.sidebar.number_input("Malt 3 %", min_value=0.0, max_value=100.0,
                                    value=10.0, step=1.0)

malt_selections = [
    {"name": malt1_name, "pct": malt1_pct},
    {"name": malt2_name, "pct": malt2_pct},
    {"name": malt3_name, "pct": malt3_pct},
]

# --- Yeast strain section
st.sidebar.markdown("### Yeast Strain")
yeast_options = sorted(yeast_df["Name"].dropna().unique().tolist())
yeast_choice = st.sidebar.selectbox("Select yeast", yeast_options, index=0 if len(yeast_options) else None)

run_button = st.sidebar.button("Predict Flavor 🧪")

# ------------------------------------------------------------------
# Main layout
# ------------------------------------------------------------------
st.title("Beer Recipe Digital Twin")
st.caption("Predict hop aroma, malt character, and fermentation profile using trained ML models.")

# We'll lay out results in two columns:
col_plot, col_info = st.columns([2, 1], vertical_alignment="top")

if run_button:
    # 1. run predictions
    hop_pred = predict_hop_profile(hop_bill)
    malt_pred = predict_malt_traits(malt_selections)
    yeast_pred = predict_yeast_traits(yeast_choice)

    style_guess = summarize_style(hop_pred, malt_pred, yeast_pred)
    top_hops_list = top_two_hop_notes(hop_pred)

    # 2. spider plot
    with col_plot:
        fig = make_spider_plot(hop_pred)
        st.pyplot(fig, clear_figure=True, use_container_width=True)

    # 3. summary panel
    with col_info:
        st.markdown("### Top hop notes:")
        if top_hops_list:
            for (note, val) in top_hops_list:
                st.write(f"- **{note}** ({round(val, 2)})")
        else:
            st.write("_No dominant hop note_")

        st.markdown("### Malt character:")
        # show only active traits == 1 or high
        malt_hits = [k for k, v in malt_pred.items() if v == 1 or (isinstance(v, float) and v > 0.5)]
        if malt_hits:
            for trait in malt_hits:
                st.write(f"- {trait}")
        else:
            st.write("_none detected_")

        st.markdown("### Yeast character:")
        yeast_hits = [k for k, v in yeast_pred.items() if v == 1 or (isinstance(v, float) and v > 0.5)]
        if yeast_hits:
            for trait in yeast_hits:
                st.write(f"- {trait}")
        else:
            st.write("_none detected_")

        st.markdown("### Style direction:")
        st.write(f"🧭 {style_guess}")

        st.markdown("### Hops used by the model:")
        used_hops = [name for name, grams in hop_bill.items() if grams and grams > 0]
        if used_hops:
            st.write(", ".join(used_hops))
        else:
            st.write("_No hops provided_")

else:
    # Initial instructions
    st.info(
        "👉 Build your hop bill (up to 4 hops with >0 g each), set malt bill (% grist), "
        "choose yeast, then click **Predict Flavor 🧪** in the sidebar."
    )

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import math
import matplotlib.pyplot as plt

# -----------------------------
# ---- CONFIG / PAGE SETUP ----
# -----------------------------
st.set_page_config(
    page_title="Beer Recipe Digital Twin",
    page_icon="🍺",
    layout="wide"
)

st.markdown("""
<style>
    /* Make sidebar scroll independently */
    [data-testid="stSidebar"] > div:first-child {
        height: 100vh;
        overflow-y: auto;
    }
    /* Nicer font scale */
    .big-header {
        font-size: 2.2rem;
        font-weight: 600;
        line-height: 2.6rem;
        margin-bottom: 0.5rem;
    }
    .subtle {
        font-size: 0.9rem;
        color: #636363;
        margin-bottom: 2rem;
    }
    .section-header {
        font-weight: 600;
        font-size: 1.05rem;
        margin-top: 1.6rem;
        margin-bottom: 0.4rem;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# ------- DATA LOADING --------
# -----------------------------
@st.cache_resource
def load_reference_data():
    """
    Load reference pickles for malts, yeast, etc.
    """
    try:
        clean_malt_df = pd.read_pickle("clean_malt_df.pkl")
    except Exception:
        clean_malt_df = pd.DataFrame(columns=["malt_name","malt_character"])

    try:
        clean_yeast_df = pd.read_pickle("clean_yeast_df.pkl")
    except Exception:
        clean_yeast_df = pd.DataFrame(columns=["name","profile"])

    return clean_malt_df, clean_yeast_df

@st.cache_resource
def load_models():
    """
    Load trained models. We assume:
      - hop_aroma_model.joblib is a regression model
        that outputs aroma intensities for certain descriptors
      - You may later add malt_sensory_model.joblib, yeast_sensory_model.joblib
        if you want fully ML-driven predictions for those.
    """
    hop_aroma_model = None
    try:
        hop_aroma_model = joblib.load("hop_aroma_model.joblib")
    except Exception:
        hop_aroma_model = None

    # placeholders for future
    malt_sensory_model = None
    yeast_sensory_model = None
    return hop_aroma_model, malt_sensory_model, yeast_sensory_model


clean_malt_df, clean_yeast_df = load_reference_data()
hop_aroma_model, malt_sensory_model, yeast_sensory_model = load_models()


# -----------------------------
# ---- HELPER: RADAR PLOT  ----
# -----------------------------
def plot_hop_radar(aroma_scores):
    """
    Draw a spider / radar chart from aroma_scores, which is expected to be
    dict like:
      {
        "tropical": float,
        "citrus": float,
        "fruity": float,
        "resinous": float,
        "floral": float,
        "herbal": float,
        "spicy": float,
        "earthy": float
      }

    We'll arrange them clockwise with tropical at top for aesthetics.
    """
    # Labels in display order (clockwise)
    labels = [
        "tropical",
        "citrus",
        "fruity",
        "resinous",
        "floral",
        "herbal",
        "spicy",
        "earthy",
    ]

    # Make sure we have them all, default 0 if not
    values = [float(aroma_scores.get(lbl, 0.0)) for lbl in labels]

    # Radar plot trick: close the polygon by repeating first value/label
    values += values[:1]
    angles = np.linspace(0, 2*math.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(
        figsize=(6,6),
        subplot_kw=dict(polar=True)
    )

    ax.plot(angles, values, color="#1f77b4", linewidth=2)
    ax.fill(angles, values, color="#1f77b4", alpha=0.25)

    # Put category labels at correct angular locations
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)

    # radial ticks
    ax.set_yticklabels([])
    ax.set_ylim(0, max(1.0, max(values)))  # dynamic radius

    # center annotation showing ~mean intensity
    center_val = np.mean(values[:-1]) if len(values) > 1 else 0.0
    ax.text(
        0.5, 0.5,
        f"{center_val:.2f}",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=14,
        bbox=dict(facecolor="white", edgecolor="#1f77b4", boxstyle="round,pad=0.3")
    )

    # Title is handled externally
    st.pyplot(fig)


# -----------------------------
# --- HELPER: HOP FEATURES ----
# -----------------------------
def build_hop_feature_row(hop_bill):
    """
    Build a 1-row DataFrame of model features from hop_bill input.

    hop_bill is a list of dicts like:
      [
        {"name": "Simcoe", "grams": 100.0},
        {"name": "Amarillo", "grams": 50.0},
        ...
      ]

    We'll create columns like hop_<variety> with grams. Any unmentioned
    varieties get 0. We'll also keep a debug df to show the user.
    """

    # If model didn't load we can't know its feature space, so fallback
    if hop_aroma_model is None:
        # We'll at least build a small "debug" df with what user entered
        user_cols = {}
        for item in hop_bill:
            nm = item["name"]
            amt = item["grams"]
            if nm not in user_cols:
                user_cols[nm] = 0.0
            user_cols[nm] += amt
        df = pd.DataFrame([user_cols])
        return df, df

    # If the model HAS .feature_names_in_ (sklearn style),
    # we will align to them. Otherwise we'll guess from hop_bill only.
    if hasattr(hop_aroma_model, "feature_names_in_"):
        model_cols = list(hop_aroma_model.feature_names_in_)
    else:
        # fallback guess: columns named "hop_<hopname>"
        # using only what user gave
        model_cols = []
        for item in hop_bill:
            hop_col = f"hop_{item['name']}"
            if hop_col not in model_cols:
                model_cols.append(hop_col)

    # Build a dict of col->0
    row_dict = {col: 0.0 for col in model_cols}
    # Fill in from user data
    for item in hop_bill:
        nm = item["name"]
        grams = float(item["grams"])
        # we try "hop_<name>" first
        guess_col = f"hop_{nm}"
        if guess_col in row_dict:
            row_dict[guess_col] += grams
        else:
            # if not found, maybe the model column literally equals the hop name:
            if nm in row_dict:
                row_dict[nm] += grams
            # If still not found, then that hop is invisible to the model.
            # We'll just ignore it.

    feature_df = pd.DataFrame([row_dict])

    # Build a debug df with just user data for easier reading
    debug_user_cols = {}
    for item in hop_bill:
        nm = item["name"]
        grams = float(item["grams"])
        debug_user_cols[nm] = debug_user_cols.get(nm, 0.0) + grams
    debug_df = pd.DataFrame([debug_user_cols])

    return feature_df, debug_df


# -----------------------------
# -- HELPER: PREDICT AROMA ----
# -----------------------------
def predict_hop_aroma(hop_bill):
    """
    Use hop_aroma_model to predict hop aroma intensities from the hop bill.
    Return:
      aroma_scores (dict),
      model_input_debug (DataFrame),
      model_is_none (bool)
    """
    feature_row, debug_df = build_hop_feature_row(hop_bill)

    if hop_aroma_model is None:
        # fallback: all 0
        aroma_scores = {
            "tropical": 0.0,
            "citrus": 0.0,
            "fruity": 0.0,
            "resinous": 0.0,
            "floral": 0.0,
            "herbal": 0.0,
            "spicy": 0.0,
            "earthy": 0.0,
        }
        return aroma_scores, debug_df, True

    try:
        # we assume model outputs array-like shape (1, 8) for these 8 descriptors
        y_pred = hop_aroma_model.predict(feature_row)
        # If y_pred is shape (1, N), flatten:
        if isinstance(y_pred, (list, np.ndarray)) and len(np.array(y_pred).shape) > 1:
            arr = np.array(y_pred)[0]
        else:
            arr = np.array(y_pred)

        # Map them onto fixed label order we use in the radar
        # If model's output ordering differs, update this mapping.
        labels = [
            "tropical",
            "citrus",
            "fruity",
            "resinous",
            "floral",
            "herbal",
            "spicy",
            "earthy",
        ]
        aroma_scores = {}
        for i, lab in enumerate(labels):
            if i < len(arr):
                aroma_scores[lab] = float(arr[i])
            else:
                aroma_scores[lab] = 0.0

    except Exception:
        # if model predict fails, just zero
        aroma_scores = {
            "tropical": 0.0,
            "citrus": 0.0,
            "fruity": 0.0,
            "resinous": 0.0,
            "floral": 0.0,
            "herbal": 0.0,
            "spicy": 0.0,
            "earthy": 0.0,
        }

    return aroma_scores, debug_df, False


# -----------------------------
# -------- SIDEBAR UI ---------
# -----------------------------
def sidebar_inputs():
    """
    Build the entire sidebar:
      - up to 4 hops (name + grams)
      - 3 malts (type + % grist)
      - yeast dropdown
      - predict button

    Returns all the structured user inputs
    """

    st.sidebar.markdown("### Hop Bill (g)")
    st.sidebar.caption("Select up to 4 hops and assign grams. Non-zero gets used.")

    # We'll keep a list of hop names we think the model can handle.
    # For now we pull from hop names we see in the model's feature set if present,
    # else we just allow a short example list.
    if hop_aroma_model is not None and hasattr(hop_aroma_model, "feature_names_in_"):
        # guess hop names from feature columns like hop_Simcoe
        hop_candidates = []
        for c in hop_aroma_model.feature_names_in_:
            # look for "hop_<name>"
            if c.startswith("hop_"):
                hop_nm = c.split("hop_",1)[1]
                hop_candidates.append(hop_nm)
            else:
                hop_candidates.append(c)
        hop_candidates = sorted(list(set(hop_candidates)))
    else:
        # fallback static list
        hop_candidates = [
            "Simcoe","Amarillo","Citra","Mosaic","Galaxy","Nelson Sauvin",
            "Cascade","Centennial","Astra","Eclipse","Ella","Enigma"
        ]
        hop_candidates = sorted(list(set(hop_candidates)))

    # We'll define a helper to render one hop block
    def hop_block(label_prefix, key_prefix):
        hop_name = st.sidebar.selectbox(
            f"{label_prefix} name",
            ["-"] + hop_candidates,
            index=0,
            key=f"{key_prefix}_name"
        )
        hop_amt = st.sidebar.number_input(
            f"{hop_name} (g)",
            min_value=0.0,
            max_value=500.0,
            value=0.0,
            step=5.0,
            key=f"{key_prefix}_amt"
        )
        return hop_name, hop_amt

    hop1_name, hop1_amt = hop_block("Hop 1", "hop1")
    hop2_name, hop2_amt = hop_block("Hop 2", "hop2")
    hop3_name, hop3_amt = hop_block("Hop 3", "hop3")
    hop4_name, hop4_amt = hop_block("Hop 4", "hop4")

    # MALT BILL
    st.sidebar.markdown("### Malt Bill")
    st.sidebar.caption("Pick 3 malts and % grist each.")

    # We'll allow free text / fallback if we can't parse clean_malt_df
    if "malt_name" in clean_malt_df.columns:
        malt_options = sorted(clean_malt_df["malt_name"].dropna().unique().tolist())
    else:
        # fallback generic list
        malt_options = [
            "PILSNER MALT","PALE ALE MALT","AMBER MALT","CARAMEL MALT",
            "BLACK MALT","WHEAT MALT","BEST ALE MALT"
        ]
        malt_options = sorted(list(set(malt_options)))

    def malt_block(block_label, default_name, default_pct, key_prefix):
        malt_choice = st.sidebar.selectbox(
            block_label,
            malt_options,
            index=(malt_options.index(default_name) 
                   if default_name in malt_options else 0),
            key=f"{key_prefix}_name"
        )
        col_dec, col_inc = st.sidebar.columns([1,1], vertical_alignment="center")
        with col_dec:
            dummy_dec = st.button("-", key=f"{key_prefix}_dec", help="Nudge down", use_container_width=True)
        with col_inc:
            dummy_inc = st.button("+", key=f"{key_prefix}_inc", help="Nudge up", use_container_width=True)

        pct_val = st.sidebar.number_input(
            f"{block_label} %",
            min_value=0.0,
            max_value=100.0,
            value=float(default_pct),
            step=1.0,
            key=f"{key_prefix}_pct"
        )
        # Increment / decrement logic:
        if dummy_dec and pct_val > 0:
            pct_val -= 1
            st.session_state[f"{key_prefix}_pct"] = pct_val
        if dummy_inc and pct_val < 100:
            pct_val += 1
            st.session_state[f"{key_prefix}_pct"] = pct_val

        return malt_choice, pct_val

    malt1_name, malt1_pct = malt_block("Malt 1", "AMBER MALT", 70.0, "malt1")
    malt2_name, malt2_pct = malt_block("Malt 2", "BEST ALE MALT", 20.0, "malt2")
    malt3_name, malt3_pct = malt_block("Malt 3", "BLACK MALT", 10.0, "malt3")

    # YEAST STRAIN
    st.sidebar.markdown("### Yeast Strain")

    # figure out which column in clean_yeast_df likely holds the strain text
    yeast_col_candidates = ["name","yeast_name","strain","Yeast","yeast"]
    yeast_col = None
    for c in yeast_col_candidates:
        if c in clean_yeast_df.columns:
            yeast_col = c
            break

    if yeast_col is not None and len(clean_yeast_df) > 0:
        yeast_options = ["-"] + sorted(clean_yeast_df[yeast_col].dropna().unique().tolist())
    else:
        yeast_options = ["-"]

    yeast_choice = st.sidebar.selectbox(
        "Select yeast",
        yeast_options,
        index=0,
        key="yeast_choice"
    )

    run_button = st.sidebar.button("Predict Flavor 🧪", key="run_button")

    # Bundle up hops in a structured list for downstream
    hop_bill = []
    if hop1_name != "-" and hop1_amt > 0:
        hop_bill.append({"name": hop1_name, "grams": hop1_amt})
    if hop2_name != "-" and hop2_amt > 0:
        hop_bill.append({"name": hop2_name, "grams": hop2_amt})
    if hop3_name != "-" and hop3_amt > 0:
        hop_bill.append({"name": hop3_name, "grams": hop3_amt})
    if hop4_name != "-" and hop4_amt > 0:
        hop_bill.append({"name": hop4_name, "grams": hop4_amt})

    malt_bill = [
        {"malt": malt1_name, "pct": malt1_pct},
        {"malt": malt2_name, "pct": malt2_pct},
        {"malt": malt3_name, "pct": malt3_pct},
    ]

    return hop_bill, malt_bill, yeast_choice, run_button


# -----------------------------
# ------- MAIN CONTENT --------
# -----------------------------
def main():
    # LAYOUT: 3 columns
    # left col is sidebar already, we place main content in body
    st.markdown(
        "<div class='big-header'>🍺 Beer Recipe Digital Twin</div>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<div class='subtle'>Predict hop aroma, malt character, and fermentation profile using trained ML models (in progress).</div>",
        unsafe_allow_html=True
    )

    # gather user inputs
    hop_bill, malt_bill, yeast_choice, run_button = sidebar_inputs()

    # 3 main columns across the body
    col_radar, col_desc = st.columns([2,1], gap="large")

    # If user clicked the button, generate predictions
    if run_button:
        # get hop aroma from the hop_aroma_model
        aroma_scores, debug_hop_input_df, model_is_none = predict_hop_aroma(hop_bill)

        # MALTS (placeholder)
        #  - If you later add malt_sensory_model, compute something real
        if len(malt_bill) > 0:
            # naive "malt character": choose first malt's name or fallback
            malt_character = "bready"
            # (you can enhance this with real model or heuristics)
        else:
            malt_character = "bready"

        # YEAST (placeholder)
        #  - If you later add yeast_sensory_model, compute something real
        if yeast_choice and yeast_choice != "-":
            yeast_character = "fruity_esters, clean_neutral"
        else:
            yeast_character = "fruity_esters, clean_neutral"

        style_direction = "🍻 Experimental / Hybrid"

        # ---------- Radar plot ----------
        with col_radar:
            st.markdown("## Hop Aroma Radar")
            plot_hop_radar(aroma_scores)

        # ---------- Textual descriptors ----------
        with col_desc:
            st.markdown("### Top hop notes:")
            # naive top2 from aroma_scores
            sorted_notes = sorted(
                aroma_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
            top_notes = sorted_notes[:2] if sorted_notes else []
            for note, val in top_notes:
                st.markdown(f"- **{note} ({val:.2f})**")

            st.markdown("### Malt character:")
            st.markdown(f"{malt_character}")

            st.markdown("### Yeast character:")
            st.markdown(f"{yeast_character}")

            st.markdown("### Style direction:")
            st.markdown(style_direction)

            # show hops used
            st.markdown("### Hops used by the model:")
            if hop_bill:
                hop_lines = []
                for h in hop_bill:
                    hop_lines.append(f"{h['name']} ({h['grams']} g)")
                st.markdown(", ".join(hop_lines))
            else:
                st.markdown("_No hops in bill_")

        # ---------- Debug block for hop model input ----------
        with st.expander("🔬 Debug: hop model input (what the model actually sees)"):
            st.write("hop_aroma_model is None?", model_is_none)
            if hop_aroma_model is not None and hasattr(hop_aroma_model, "feature_names_in_"):
                st.write("Model feature_names_in_:", list(hop_aroma_model.feature_names_in_))
            else:
                st.write("Model has no .feature_names_in_ attr.")
            st.write("DataFrame passed to model for prediction (aligned features):")
            # build the aligned features again to show user:
            aligned_df, raw_debug_df = build_hop_feature_row(hop_bill)
            st.dataframe(aligned_df)

            st.write("User aggregate hop grams by hop name:")
            st.dataframe(raw_debug_df)

    else:
        # BEFORE the click, just instruct user
        st.info(
            "🧪 Build your hop bill (up to 4 hops, each >0 g), set malt bill (% grist), "
            "choose yeast, then click **Predict Flavor 🧪** in the sidebar."
        )

    # ---------- Debug block for yeast dataset ----------
    with st.expander("🧫 Debug: yeast dataset (clean_yeast_df)"):
        st.write("Columns:", list(clean_yeast_df.columns))
        st.dataframe(clean_yeast_df.head())


# -----------------------------
# ---------- RUN APP ----------
# -----------------------------
if __name__ == "__main__":
    main()

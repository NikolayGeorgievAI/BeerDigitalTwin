import streamlit as st
import pandas as pd
import numpy as np
import joblib
import math
import matplotlib.pyplot as plt
import inspect

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
        clean_yeast_df = pd.DataFrame(columns=["Name","Lab","Type","Form","Temp"])

    return clean_malt_df, clean_yeast_df

@st.cache_resource
def load_models():
    """
    Load trained models.
    hop_aroma_model.joblib is assumed to be some ML model that predicts
    hop aroma intensities.
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
def plot_hop_radar(aroma_scores, title="Hop Aroma Radar"):
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

    We'll arrange them clockwise with tropical at top.
    """
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

    values = [float(aroma_scores.get(lbl, 0.0)) for lbl in labels]
    # close polygon
    values += values[:1]

    angles = np.linspace(0, 2*math.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(
        figsize=(6,6),
        subplot_kw=dict(polar=True)
    )

    ax.plot(angles, values, color="#1f77b4", linewidth=2)
    ax.fill(angles, values, color="#1f77b4", alpha=0.25)

    # category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)

    ax.set_yticklabels([])
    ax.set_ylim(0, max(1.0, max(values)))  # dynamic radius

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

    ax.set_title(title, pad=20, fontsize=16, fontweight="bold")
    st.pyplot(fig)


# -----------------------------
# --- HELPER: FEATURE ROW  ----
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

    Return:
        feature_df (DataFrame)  -> what we'll send to model.predict()
        debug_df   (DataFrame)  -> nice summary of hop grams by name
    """

    # If model didn't load, create debug only
    if hop_aroma_model is None:
        user_cols = {}
        for item in hop_bill:
            nm = item["name"]
            amt = item["grams"]
            user_cols[nm] = user_cols.get(nm, 0.0) + amt
        df = pd.DataFrame([user_cols]) if user_cols else pd.DataFrame([{}])
        return df, df

    # If model HAS .feature_names_in_, we align to that
    if hasattr(hop_aroma_model, "feature_names_in_"):
        model_cols = list(hop_aroma_model.feature_names_in_)
    else:
        # We don't know the feature names. We'll GUESS:
        # For each hop in hop_bill, build "hop_<Hopname>"
        model_cols = []
        for item in hop_bill:
            hop_col = f"hop_{item['name']}"
            if hop_col not in model_cols:
                model_cols.append(hop_col)

    # Build row dict with zeros
    row_dict = {col: 0.0 for col in model_cols}

    # Fill from user data
    for item in hop_bill:
        nm = item["name"]
        grams = float(item["grams"])
        guess_cols = [
            f"hop_{nm}",
            nm
        ]
        for gc in guess_cols:
            if gc in row_dict:
                row_dict[gc] += grams
                break
        # If we still didn't find a match, that hop won't be represented.

    feature_df = pd.DataFrame([row_dict])

    # Build debug of user totals
    user_dict = {}
    for item in hop_bill:
        nm = item["name"]
        grams = float(item["grams"])
        user_dict[nm] = user_dict.get(nm, 0.0) + grams
    debug_df = pd.DataFrame([user_dict]) if user_dict else pd.DataFrame([{}])

    return feature_df, debug_df


# -----------------------------
# -- HELPER: PREDICT AROMA ----
# -----------------------------
def predict_hop_aroma(hop_bill):
    """
    Use hop_aroma_model to predict hop aroma intensities.
    Return:
      aroma_scores (dict),
      aligned_df (DataFrame),
      raw_debug_df (DataFrame),
      model_is_none (bool),
      model_exc (str|None)
    """
    aligned_df, raw_debug_df = build_hop_feature_row(hop_bill)

    if hop_aroma_model is None:
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
        return aroma_scores, aligned_df, raw_debug_df, True, None

    # Try to predict
    try:
        y_pred = hop_aroma_model.predict(aligned_df)
        arr = np.array(y_pred)
        # If shape is (1, N), flatten that row
        if len(arr.shape) > 1:
            arr = arr[0]

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

        return aroma_scores, aligned_df, raw_debug_df, False, None

    except Exception as e:
        # If prediction fails, fallback zeros
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
        return aroma_scores, aligned_df, raw_debug_df, False, str(e)


# -----------------------------
# HELPER: YEAST CHARACTER TEXT
# -----------------------------
def describe_yeast(clean_yeast_df, yeast_choice):
    """
    Build a textual yeast character using the chosen strain row.
    We'll look up columns like fruity_esters, clean_neutral, etc.
    """
    if not yeast_choice or yeast_choice == "-":
        return "fruity_esters, clean_neutral"

    # Find row in df where 'Name' == that yeast_choice
    if "Name" in clean_yeast_df.columns:
        row = clean_yeast_df.loc[clean_yeast_df["Name"] == yeast_choice]
    else:
        row = pd.DataFrame()

    if row.empty:
        return "fruity_esters, clean_neutral"

    row = row.iloc[0]

    sensory_flags = []
    for col in [
        "fruity_esters",
        "phenolic_spicy",
        "clean_neutral",
        "dry_finish",
        "malty",
        "sulfur_note",
    ]:
        if col in row.index:
            val = row[col]
            # Any truthy / nonzero -> include
            if isinstance(val, (int, float)) and val > 0:
                sensory_flags.append(col)
            elif isinstance(val, str) and val.strip() and val.strip().lower() not in ["0","false","none","no"]:
                sensory_flags.append(col)

    if not sensory_flags:
        sensory_flags = ["clean_neutral"]

    # nice formatting
    pretty = ", ".join(sensory_flags).replace("_"," ")
    return pretty


# -----------------------------
# -------- SIDEBAR UI ---------
# -----------------------------
def sidebar_inputs():
    """
    Build sidebar:
      - up to 4 hops (name + grams)
      - 3 malts
      - yeast
      - Predict button
    """
    st.sidebar.markdown("### Hop Bill (g)")
    st.sidebar.caption("Select up to 4 hops and assign grams (non-zero).")

    # Build hop list either from model or fallback
    if hop_aroma_model is not None and hasattr(hop_aroma_model, "feature_names_in_"):
        hop_candidates = []
        for c in hop_aroma_model.feature_names_in_:
            if c.startswith("hop_"):
                hop_nm = c.split("hop_",1)[1]
                hop_candidates.append(hop_nm)
            else:
                hop_candidates.append(c)
        hop_candidates = sorted(list(set(hop_candidates)))
    else:
        hop_candidates = sorted(list(set([
            "Simcoe","Amarillo","Citra","Mosaic","Galaxy","Nelson Sauvin",
            "Cascade","Centennial","Astra","Eclipse","Ella","Enigma","Black Pearl"
        ])))

    def hop_block(idx):
        hop_name = st.sidebar.selectbox(
            f"Hop {idx} name",
            ["-"] + hop_candidates,
            index=0,
            key=f"hop{idx}_name"
        )
        hop_amt = st.sidebar.number_input(
            f"{hop_name} (g)",
            min_value=0.0,
            max_value=500.0,
            value=0.0,
            step=5.0,
            key=f"hop{idx}_amt"
        )
        return hop_name, hop_amt

    hop1_name, hop1_amt = hop_block(1)
    hop2_name, hop2_amt = hop_block(2)
    hop3_name, hop3_amt = hop_block(3)
    hop4_name, hop4_amt = hop_block(4)

    # Malt inputs
    st.sidebar.markdown("### Malt Bill")
    st.sidebar.caption("Pick up to 3 malts and % grist.")

    if "malt_name" in clean_malt_df.columns:
        malt_options = sorted(clean_malt_df["malt_name"].dropna().unique().tolist())
    else:
        malt_options = sorted(list(set([
            "PILSNER MALT","PALE ALE MALT","AMBER MALT","CARAMEL MALT",
            "BLACK MALT","WHEAT MALT","BEST ALE MALT"
        ])))

    def malt_block(idx, default_name, default_pct):
        malt_choice = st.sidebar.selectbox(
            f"Malt {idx}",
            malt_options,
            index=(malt_options.index(default_name)
                   if default_name in malt_options else 0),
            key=f"malt{idx}_name"
        )
        # little +/- row
        col_dec, col_inc = st.sidebar.columns([1,1])
        with col_dec:
            dec_btn = st.button("-", key=f"malt{idx}_dec", use_container_width=True)
        with col_inc:
            inc_btn = st.button("+", key=f"malt{idx}_inc", use_container_width=True)

        pct_val = st.sidebar.number_input(
            f"Malt {idx} %",
            min_value=0.0,
            max_value=100.0,
            value=float(default_pct),
            step=1.0,
            key=f"malt{idx}_pct"
        )
        if dec_btn and pct_val > 0:
            pct_val -= 1
            st.session_state[f"malt{idx}_pct"] = pct_val
        if inc_btn and pct_val < 100:
            pct_val += 1
            st.session_state[f"malt{idx}_pct"] = pct_val

        return malt_choice, pct_val

    malt1_name, malt1_pct = malt_block(1, "AMBER MALT", 70.0)
    malt2_name, malt2_pct = malt_block(2, "BEST ALE MALT", 20.0)
    malt3_name, malt3_pct = malt_block(3, "BLACK MALT", 10.0)

    # Yeast
    st.sidebar.markdown("### Yeast Strain")

    # figure out which column in clean_yeast_df to use
    yeast_col_candidates = [
        "Name","name","yeast_name","strain","Yeast","yeast"
    ]
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

    # Build structured bills
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
    st.markdown(
        "<div class='big-header'>🍺 Beer Recipe Digital Twin</div>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<div class='subtle'>Predict hop aroma, malt character, and fermentation profile using trained ML models (work in progress).</div>",
        unsafe_allow_html=True
    )

    hop_bill, malt_bill, yeast_choice, run_button = sidebar_inputs()

    col_radar, col_desc = st.columns([2,1], gap="large")

    if run_button:
        (
            aroma_scores,
            aligned_df,
            raw_debug_df,
            model_is_none,
            model_exc
        ) = predict_hop_aroma(hop_bill)

        # Malt "character" placeholder
        if len(malt_bill) > 0:
            malt_character = "bready"
        else:
            malt_character = "bready"

        # Yeast character using real row info if available
        yeast_character = describe_yeast(clean_yeast_df, yeast_choice)

        # Style direction: just a placeholder vibe
        style_direction = "🍻 Experimental / Hybrid"

        # --- Radar ---
        with col_radar:
            plot_hop_radar(aroma_scores, title="Hop Aroma Radar")

        # --- Descriptors ---
        with col_desc:
            st.markdown("### Top hop notes:")
            sorted_notes = sorted(
                aroma_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
            top_notes = sorted_notes[:2] if sorted_notes else []
            for note, val in top_notes:
                st.markdown(f"- **{note} ({val:.2f})**")

            st.markdown("### Malt character:")
            st.markdown(malt_character)

            st.markdown("### Yeast character:")
            st.markdown(yeast_character)

            st.markdown("### Style direction:")
            st.markdown(style_direction)

            st.markdown("### Hops used by the model:")
            if hop_bill:
                hop_lines = [f"{h['name']} ({h['grams']} g)" for h in hop_bill]
                st.markdown(", ".join(hop_lines))
            else:
                st.markdown("_No hops in bill_")

        # --- Debug: hop model input ---
        with st.expander("🔬 Debug: hop model input (what the model actually sees)"):
            st.write("hop_aroma_model is None?", model_is_none)
            if model_exc:
                st.error(f"Model predict() exception: {model_exc}")

            # Show final DF that went to predict():
            st.write("DataFrame passed to model for prediction (aligned_df):")
            st.dataframe(aligned_df)

            st.write("User aggregate hop grams by hop name (raw_debug_df):")
            st.dataframe(raw_debug_df)

        # --- Debug: hop model internals ---
        with st.expander("🧠 Debug: hop model internals"):
            st.write("Type(hop_aroma_model):", type(hop_aroma_model))
            st.write("dir(hop_aroma_model)[:50]:", dir(hop_aroma_model)[:50])

            # Try to guess attributes that might hold feature names
            guess_attrs = [
                "feature_names_in_",
                "feature_names_",
                "columns_",
                "input_features_",
                "coef_",
                "n_features_in_"
            ]
            for ga in guess_attrs:
                if hasattr(hop_aroma_model, ga):
                    st.write(f"{ga}:", getattr(hop_aroma_model, ga))

            # If it's something like a sklearn Pipeline, show steps
            if hasattr(hop_aroma_model, "steps"):
                st.write("Pipeline steps:")
                for step_name, submodel in hop_aroma_model.steps:
                    st.write(" -", step_name, ":", type(submodel))
                    st.write("   dir() subset:", dir(submodel)[:30])
                    # also inspect common attrs
                    for ga in guess_attrs:
                        if hasattr(submodel, ga):
                            st.write(f"   {ga}:", getattr(submodel, ga))

            # show model source if possible (just best effort)
            try:
                src = inspect.getsource(hop_aroma_model.__class__)
                st.code(src[:2000], language="python")
            except Exception:
                pass

    else:
        st.info(
            "🧪 Build your hop bill, set malt bill, choose yeast, "
            "then click **Predict Flavor 🧪** in the sidebar."
        )

    # --- Debug: yeast dataset ---
    with st.expander("🧫 Debug: yeast dataset (clean_yeast_df)"):
        st.write("Columns:", list(clean_yeast_df.columns))
        st.dataframe(clean_yeast_df.head())


# -----------------------------
# ---------- RUN APP ----------
# -----------------------------
if __name__ == "__main__":
    main()

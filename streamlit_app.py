import os
import re
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from openai import OpenAI

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# -----------------------------------------------------------------------------
# ENV + CONSTANTS
# -----------------------------------------------------------------------------

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "").strip()

ROOT_DIR = os.path.dirname(__file__)

HOP_MODEL_PATH = os.path.join(ROOT_DIR, "hop_aroma_model.joblib")
MALT_MODEL_PATH = os.path.join(ROOT_DIR, "malt_sensory_model.joblib")
YEAST_MODEL_PATH = os.path.join(ROOT_DIR, "yeast_sensory_model.joblib")

# -----------------------------------------------------------------------------
# LOADING MODELS
# -----------------------------------------------------------------------------

hop_bundle = joblib.load(HOP_MODEL_PATH)
hop_model = hop_bundle["model"]
hop_feature_cols = hop_bundle["feature_cols"]
hop_dims = [a for a in hop_bundle["aroma_dims"] if str(a).lower() not in ("nan", "", "none")]

malt_bundle = joblib.load(MALT_MODEL_PATH)
malt_model = malt_bundle["model"]
malt_feature_cols = malt_bundle["feature_cols"]
malt_dims = malt_bundle["flavor_cols"]

yeast_bundle = joblib.load(YEAST_MODEL_PATH)
yeast_model = yeast_bundle["model"]
yeast_feature_cols = yeast_bundle["feature_cols"]
yeast_dims = yeast_bundle["flavor_cols"]

# -----------------------------------------------------------------------------
# SMALL TEXT HELPERS
# -----------------------------------------------------------------------------

def _clean_name(name: str) -> str:
    """Lowercase, strip symbols, keep alphanum only."""
    if not name:
        return ""
    s = str(name).lower()
    s = s.replace("®", "").replace("™", "")
    s = re.sub(r"[^a-z0-9]", "", s)
    return s

def _best_feature_match(user_name: str, feature_cols: list, prefix: str):
    """
    Fuzzy-ish match: find the column with given prefix whose label
    overlaps the cleaned user_name.
    """
    cleaned_user = _clean_name(user_name)
    best_col = None
    best_score = -1

    for col in feature_cols:
        if not col.startswith(prefix):
            continue
        label_raw = col[len(prefix):]
        cleaned_label = _clean_name(label_raw)

        # rough char-overlap heuristic
        common = set(cleaned_user) & set(cleaned_label)
        score = len(common)

        # mild check so we don't match random if there's no overlap
        if len(cleaned_user) >= 3 and cleaned_user[:3] not in cleaned_label:
            continue

        if score > best_score:
            best_score = score
            best_col = col

    return best_col

def _choices_from_features(feature_cols, preferred_prefix=None):
    """
    Build nice human-readable dropdowns from model feature cols.
    We prefer items with a certain prefix (e.g. 'malt_','yeast_','hop_'),
    fallback to everything if no matches.
    """
    def prettify(label: str) -> str:
        label = label.replace("®", "").replace("™", "")
        label = label.replace("_", " ").strip()
        return label

    subset = []
    if preferred_prefix:
        for c in feature_cols:
            if c.startswith(preferred_prefix):
                raw_label = c[len(preferred_prefix):]
                subset.append(prettify(raw_label))

    if not subset:
        for c in feature_cols:
            cand = c
            for p in ["hop_", "malt_", "grain_", "base_", "yeast_", "strain_", "y_", "m_"]:
                if cand.startswith(p):
                    cand = cand[len(p):]
            subset.append(prettify(cand))

    cleaned = []
    for nm in subset:
        if nm and nm not in cleaned:
            cleaned.append(nm)

    cleaned = sorted(cleaned, key=lambda s: s.lower())
    return cleaned

HOP_CHOICES = _choices_from_features(hop_feature_cols, preferred_prefix="hop_")
MALT_CHOICES = _choices_from_features(malt_feature_cols, preferred_prefix="malt_")
YEAST_CHOICES = _choices_from_features(yeast_feature_cols, preferred_prefix="yeast_")

# -----------------------------------------------------------------------------
# FEATURE BUILDERS + PREDICTORS
# -----------------------------------------------------------------------------

def build_hop_features(user_hops):
    """
    user_hops: list of {name: str, amt: float} in grams
    returns 1-row DF with columns=hop_feature_cols
    """
    totals = {c: 0.0 for c in hop_feature_cols}
    for entry in user_hops:
        nm = entry.get("name", "")
        amt = float(entry.get("amt", 0.0))
        if amt <= 0 or not nm.strip():
            continue
        match = _best_feature_match(nm, hop_feature_cols, prefix="hop_")
        if match:
            totals[match] += amt
    return pd.DataFrame([totals], columns=hop_feature_cols)

def predict_hop_profile(user_hops):
    X = build_hop_features(user_hops)
    y_pred = hop_model.predict(X)[0]
    return {dim: float(val) for dim, val in zip(hop_dims, y_pred)}

def build_malt_features(user_malts):
    """
    user_malts: list of {name: str, pct: float} as % of grist
    returns 1-row DF with columns=malt_feature_cols
    """
    totals = {c: 0.0 for c in malt_feature_cols}

    for entry in user_malts:
        nm = entry.get("name", "")
        pct = float(entry.get("pct", 0.0))
        if pct <= 0 or not nm.strip():
            continue

        chosen_col = _best_feature_match(nm, malt_feature_cols, prefix="malt_")
        if chosen_col is None:
            # fallback across known alt prefixes
            for pfx in ["grain_", "base_", "malt_", "m_"]:
                chosen_col = _best_feature_match(nm, malt_feature_cols, prefix=pfx)
                if chosen_col:
                    break
        if chosen_col:
            totals[chosen_col] += pct

    return pd.DataFrame([totals], columns=malt_feature_cols)

def predict_malt_profile(user_malts):
    X = build_malt_features(user_malts)
    y_pred = malt_model.predict(X)[0]
    return {dim: float(val) for dim, val in zip(malt_dims, y_pred)}

def build_yeast_features(user_yeast):
    """
    user_yeast:
      { "strain": str, "ferm_temp_c": float }
    We'll encode strain as 1-hot
    NOTE: the regression we trained uses only strain 1-hot, not temp yet
    """
    totals = {c: 0.0 for c in yeast_feature_cols}
    strain = user_yeast.get("strain", "")
    match = _best_feature_match(strain, yeast_feature_cols, prefix="yeast_")
    if match is None:
        # fallback across alt prefixes
        for pfx in ["strain_", "y_", "yeast_"]:
            m2 = _best_feature_match(strain, yeast_feature_cols, prefix=pfx)
            if m2:
                match = m2
                break
    if match:
        totals[match] = 1.0

    return pd.DataFrame([totals], columns=yeast_feature_cols)

def predict_yeast_profile(user_yeast):
    """
    predicted fermentation-ish traits from yeast model
    """
    X = build_yeast_features(user_yeast)
    y_pred = yeast_model.predict(X)[0]
    return {dim: float(val) for dim, val in zip(yeast_dims, y_pred)}

# -----------------------------------------------------------------------------
# RADAR PLOT HELPER
# -----------------------------------------------------------------------------

def make_clean_radar(labels, values, title=None, max_val=1.0):
    """
    Create a compact radar chart with:
    - category labels around edge
    - no numeric tick labels
    - values scaled to max_val -> range 0..1
    - shared radius=1 for consistent shape comparison
    """
    # safety
    if not labels or not values or len(labels) != len(values):
        # basic empty fig
        fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(3.5,3.5))
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_ylim(0,1)
        ax.set_title(title or "", fontsize=11, pad=14, fontweight="bold")
        return fig

    max_val = max(max_val, 1e-6)

    ang = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
    ang = np.concatenate([ang, ang[:1]])

    scaled = [v / max_val for v in values]
    scaled.append(scaled[0])

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(3.5,3.5))

    ax.plot(ang, scaled, color="#1f77b4", linewidth=2)
    ax.fill(ang, scaled, color="#1f77b4", alpha=0.25)

    # category labels
    ax.set_xticks(ang[:-1])
    ax.set_xticklabels(labels, fontsize=9)

    # kill radial tick labels
    ax.set_yticks([])
    ax.set_yticklabels([])
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_visible(False)

    # subtle grid
    ax.grid(True, color="#cccccc", alpha=0.4)

    # fix radius range
    ax.set_ylim(0,1)

    if title:
        ax.set_title(title, fontsize=11, pad=14, fontweight="bold")

    return fig

# -----------------------------------------------------------------------------
# AZURE OPENAI BREWMASTER NOTES
# -----------------------------------------------------------------------------

def call_azure_brewmaster_notes(
    brewer_goal_text: str,
    hop_vec: dict,
    malt_vec: dict,
    yeast_vec: dict,
):
    """
    Ask Azure OpenAI (gpt-4.1-mini deployment) for guidance,
    or fallback to a small local template if call fails.
    """
    # craft some structured summary from predicted vectors:
    def short_block(vec: dict, topn=5):
        # pick up to 5 largest dims
        if not vec:
            return "N/A"
        items = sorted(vec.items(), key=lambda kv: kv[1], reverse=True)
        out_lines = []
        for k,v in items[:topn]:
            out_lines.append(f"{k}: {v:.2f}")
        return "; ".join(out_lines)

    hop_desc = short_block(hop_vec, topn=5)
    malt_desc = short_block(malt_vec, topn=5)
    yeast_desc = short_block(yeast_vec, topn=5)

    system_msg = (
        "You are a professional brewmaster. "
        "Given hop aroma, malt/body, and fermentation character predictions, "
        "you will suggest concrete recipe/process adjustments to move the beer "
        "toward the brewer's style goal. Keep it practical and specific. "
        "Keep bitterness, late hops, grist tweaks, and fermentation notes clear."
    )

    user_msg = (
        f"Brewer's style/goal:\n{brewer_goal_text}\n\n"
        f"Hop aroma profile (top notes, scale ~0-1+): {hop_desc}\n"
        f"Malt/body profile (sweetness, fullness, color): {malt_desc}\n"
        f"Yeast/fermentation traits: {yeast_desc}\n\n"
        "Please respond in 3-4 short numbered sections:\n"
        "1. Hop adjustments (varieties, timing, bitterness mgmt)\n"
        "2. Malt/grist tweaks (base/specialty malts for body, sweetness, color)\n"
        "3. Fermentation guidance (yeast choice, temp, attenuation)\n"
        "4. One-sentence summary of what to change to hit the goal.\n"
        "Be concise, bullet-point friendly."
    )

    if (AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY and AZURE_OPENAI_DEPLOYMENT):
        try:
            client = OpenAI(
                base_url=f"{AZURE_OPENAI_ENDPOINT}openai/deployments/{AZURE_OPENAI_DEPLOYMENT}/",
                api_key=AZURE_OPENAI_API_KEY,
                default_headers={
                    "api-key": AZURE_OPENAI_API_KEY,
                    "Content-Type": "application/json",
                },
            )

            completion = client.chat.completions.create(
                model=AZURE_OPENAI_DEPLOYMENT,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.4,
                max_tokens=400,
            )

            if completion and completion.choices:
                ai_text = completion.choices[0].message.get("content", "").strip()
            else:
                ai_text = ""
        except Exception as e:
            ai_text = ""

    else:
        ai_text = ""

    if not ai_text:
        # fallback note if azure not working / 404, etc.
        ai_text = (
            "Brewmaster Notes (prototype)\n"
            "• Hop toward tropical/stone-fruit character using high-oil late/dry hop "
            "additions; keep bitterness low to maintain softness.\n"
            "• Add oats or wheat for pillowy mouthfeel and stable haze.\n"
            "• Avoid overly toasty/crystal malts if you want a pale, juicy profile.\n"
            "• Use a moderately ester-friendly yeast and keep fermentation temps "
            "in a softer range (around 18–20°C) to avoid harsh fusels.\n"
            "• Goal: juicy, saturated fruit aroma and plush mouthfeel without "
            "sharp bitterness."
        )

    return ai_text

# -----------------------------------------------------------------------------
# STREAMLIT LAYOUT
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="Beer Recipe Digital Twin",
    page_icon="🍺",
    layout="centered"
)

st.title("🍺 Beer Recipe Digital Twin")
st.markdown(
    """
Your AI brew assistant:

1. Build a hop bill, grain bill, and fermentation plan.  
2. Predict aroma, body, color, esters, mouthfeel — together.  
3. Get brewmaster-style guidance based on your style goal.  
"""
)
st.markdown("---")

# ---------------------------
# INPUTS: HOPS
# ---------------------------

st.header("🌿 Hops (late/aroma additions)")

c1, c2 = st.columns([1,1])
with c1:
    hop1 = st.selectbox(
        "Main Hop Variety",
        HOP_CHOICES,
        index=HOP_CHOICES.index("Citra") if "Citra" in HOP_CHOICES else 0,
        key="hop1_select",
    )
    hop2 = st.selectbox(
        "Secondary Hop Variety",
        HOP_CHOICES,
        index=HOP_CHOICES.index("Mosaic") if "Mosaic" in HOP_CHOICES else 0,
        key="hop2_select",
    )

with c2:
    hop1_amt = st.number_input(
        "Hop 1 amount (g)",
        min_value=0.0, max_value=500.0, step=5.0, value=30.0,
    )
    hop2_amt = st.number_input(
        "Hop 2 amount (g)",
        min_value=0.0, max_value=500.0, step=5.0, value=20.0,
    )

# ---------------------------
# INPUTS: MALT
# ---------------------------

st.header("🌾 Malt / Grain Bill")

m1, m2 = st.columns([1,1])
with m1:
    malt1 = st.selectbox(
        "Base / primary malt",
        MALT_CHOICES,
        index=MALT_CHOICES.index("EXTRA PALE MALT") if "EXTRA PALE MALT" in MALT_CHOICES else 0,
        key="malt1_select",
    )
    malt2 = st.selectbox(
        "Specialty / character malt",
        MALT_CHOICES,
        index=MALT_CHOICES.index("HANÁ MALT") if "HANÁ MALT" in MALT_CHOICES else 0,
        key="malt2_select",
    )
with m2:
    malt1_pct = st.number_input(
        "Malt 1 (% grist)",
        min_value=0.0, max_value=100.0, step=1.0, value=70.0,
    )
    malt2_pct = st.number_input(
        "Malt 2 (% grist)",
        min_value=0.0, max_value=100.0, step=1.0, value=8.0,
    )

# ---------------------------
# INPUTS: YEAST
# ---------------------------

st.header("🧫 Yeast & Fermentation")

y1, y2 = st.columns([1,1])
with y1:
    yeast_strain = st.selectbox(
        "Yeast strain",
        YEAST_CHOICES,
        index=YEAST_CHOICES.index("Nottingham Ale Yeast") if "Nottingham Ale Yeast" in YEAST_CHOICES else 0,
        key="yeast_select",
    )

with y2:
    ferm_temp_c = st.number_input(
        "Fermentation temp (°C)",
        min_value=10.0, max_value=30.0, step=0.5, value=20.0,
    )

st.markdown("---")

# -----------------------------------------------------------------------------
# PREDICTION BUTTON
# -----------------------------------------------------------------------------

st.subheader("🍻 Predict Beer Flavor & Balance")

st.caption(
    "Fill hops, malt, and yeast above — then click **'Predict Beer Flavor & Balance'** "
    "to simulate aroma, body, esters, color, etc."
)

run_sim = st.button("🍻 Predict Beer Flavor & Balance")

hop_profile = {}
malt_profile = {}
yeast_profile = {}

if run_sim:
    # 1. build user inputs
    user_hops = []
    if hop1 and hop1_amt > 0:
        user_hops.append({"name": hop1, "amt": hop1_amt})
    if hop2 and hop2_amt > 0:
        user_hops.append({"name": hop2, "amt": hop2_amt})

    user_malts = []
    if malt1 and malt1_pct > 0:
        user_malts.append({"name": malt1, "pct": malt1_pct})
    if malt2 and malt2_pct > 0:
        user_malts.append({"name": malt2, "pct": malt2_pct})

    user_yeast = {
        "strain": yeast_strain,
        "ferm_temp_c": ferm_temp_c,
    }

    # 2. predictions
    hop_profile = predict_hop_profile(user_hops)        # dict dim->val
    malt_profile = predict_malt_profile(user_malts)     # dict dim->val
    yeast_profile = predict_yeast_profile(user_yeast)   # dict dim->val

    st.markdown("### 📊 Predicted Flavor Snapshot")

    # ------------------ HOP SNAPSHOT ------------------
    st.markdown("**Hop aroma / character:**")
    if hop_profile:
        for k, v in hop_profile.items():
            st.write(f"- {k}: {v:.2f}")
    else:
        st.write("- (no hop data)")

    # ------------------ MALT SNAPSHOT ------------------
    st.markdown("**Malt body / sweetness / color:**")
    if malt_profile:
        for k, v in malt_profile.items():
            st.write(f"- {k}: {v:.2f}")
    else:
        st.write("- (no malt data)")

    # ------------------ YEAST SNAPSHOT ------------------
    st.markdown("**Yeast / fermentation profile:**")
    if yeast_profile:
        # We'll also display actual fermentation temp in °F, plus predicted dims
        # We do some guess fields: let's call them 'flocculation_num', 'attenuation_num' if present
        temp_f = ferm_temp_c * 9.0/5.0 + 32.0
        st.write(f"- Temp_avg_F: {temp_f:.1f}")
        for k, v in yeast_profile.items():
            st.write(f"- {k}: {v:.2f}")
    else:
        st.write("- (no yeast data)")

    # ---------------- RADAR OVERVIEW -------------------
    st.markdown("### 📈 Radar Overview")
    st.caption("Relative shape only. Axes are labeled, numeric tick rings are hidden.")

    # Prepare data for 3 radars.
    # Hops radar -> from hop_profile keys
    hop_labels = list(hop_profile.keys())
    hop_values = [hop_profile[k] for k in hop_labels]

    # Malt radar -> from malt_profile keys
    malt_labels = list(malt_profile.keys())
    malt_values = [malt_profile[k] for k in malt_labels]

    # Yeast radar -> we make a small set of dims from yeast_profile + temp
    # We'll define final yeast dims for shape:
    #   - temp_avg_F
    #   - flocculation_num
    #   - attenuation_num
    # We must only include dims that exist or we fill 0.
    yeast_labels = []
    yeast_values = []

    # collect from yeast_profile
    # figure out typical dimension names
    flocc_keys = [k for k in yeast_profile.keys() if "flocc" in k.lower()]
    atten_keys = [k for k in yeast_profile.keys() if "atten" in k.lower()]

    temp_f = ferm_temp_c * 9.0/5.0 + 32.0

    # We'll scale them relatively; store raw, and we'll compute max range below
    y_dims_local = []
    y_vals_local = []

    # Temp
    y_dims_local.append("temp_avg_F")
    y_vals_local.append(temp_f)

    # Flocculation
    if flocc_keys:
        y_dims_local.append("flocculation")
        y_vals_local.append(yeast_profile[flocc_keys[0]])
    # Attenuation
    if atten_keys:
        y_dims_local.append("attenuation")
        y_vals_local.append(yeast_profile[atten_keys[0]])

    yeast_labels = y_dims_local[:]
    yeast_values = y_vals_local[:]

    # Build columns for 3 radars
    colA, colB, colC = st.columns(3)

    with colA:
        if hop_labels and hop_values:
            fig_hops = make_clean_radar(
                labels=hop_labels,
                values=hop_values,
                title="Hops / Aroma",
                max_val=max(hop_values+[1.0]),
            )
            st.pyplot(fig_hops, clear_figure=True)
        else:
            st.write("(no hop radar)")

    with colB:
        if malt_labels and malt_values:
            fig_malt = make_clean_radar(
                labels=malt_labels,
                values=malt_values,
                title="Malt / Body-Sweetness",
                max_val=max(malt_values+[1.0]),
            )
            st.pyplot(fig_malt, clear_figure=True)
        else:
            st.write("(no malt radar)")

    with colC:
        if yeast_labels and yeast_values:
            fig_yeast = make_clean_radar(
                labels=yeast_labels,
                values=yeast_values,
                title="Yeast / Fermentation",
                max_val=max(yeast_values+[1.0]),
            )
            st.pyplot(fig_yeast, clear_figure=True)
        else:
            st.write("(no yeast radar)")

    st.markdown("---")

# -----------------------------------------------------------------------------
# AI BREWMASTER GUIDANCE (AZURE)
# -----------------------------------------------------------------------------

st.header("🧪 AI Brewmaster Guidance")

goal_text = st.text_area(
    "What's your intent for this beer? (e.g. 'Soft hazy IPA with saturated stone fruit and pineapple, low bitterness, pillowy mouthfeel')",
    "i want to increase mango aroma without increasing bitterness."
)

gen_notes = st.button("🧪 Generate Brewmaster Notes")

if gen_notes:
    # We want to pass the *latest* predicted profiles if available
    # If user didn't click Predict Flavor yet this run, these dicts might be empty
    ai_md = call_azure_brewmaster_notes(
        brewer_goal_text=goal_text,
        hop_vec=hop_profile,
        malt_vec=malt_profile,
        yeast_vec=yeast_profile,
    )

    st.subheader("Brewmaster Notes")
    st.write(ai_md)
    st.caption(
        "Prototype — not production brewing advice. "
        "Always match your yeast strain's process window."
    )

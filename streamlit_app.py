import os
import re
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from openai import AzureOpenAI

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# =========================================================
# Utility: text cleaning + fuzzy feature matching
# =========================================================

def _clean_name(name: str) -> str:
    """Lowercase, strip unicode marks, keep only a–z0–9."""
    if name is None:
        return ""
    s = str(name).lower()
    s = s.replace("®", "").replace("™", "")
    s = re.sub(r"[^a-z0-9]", "", s)
    return s

def _best_feature_match(user_name: str, feature_cols: list, prefixes: list):
    """
    Fuzzy match a user-provided label (e.g. 'Maris Otter')
    to the closest model column, trying each prefix (e.g. ['malt_', 'grain_']).
    We score by simple character overlap after cleaning.

    Returns best matching column or None.
    """
    cleaned_user = _clean_name(user_name)
    if not cleaned_user:
        return None

    best_col = None
    best_score = -1

    for pfx in prefixes:
        for col in feature_cols:
            if pfx and not col.startswith(pfx):
                continue

            raw_label = col[len(pfx):] if pfx else col
            cleaned_label = _clean_name(raw_label)

            # quick gate
            if len(cleaned_user) >= 3 and cleaned_user[:3] not in cleaned_label:
                continue

            # naive overlap score
            common = set(cleaned_user) & set(cleaned_label)
            score = len(common)
            if score > best_score:
                best_score = score
                best_col = col

    # fallback across entire feature_cols if nothing matched prefixes
    if best_col is None:
        for col in feature_cols:
            cleaned_label = _clean_name(col)
            common = set(cleaned_user) & set(cleaned_label)
            score = len(common)
            if score > best_score:
                best_score = score
                best_col = col

    return best_col


def _choices_from_features(feature_cols, preferred_prefixes=None):
    """
    Turn model feature columns into human-friendly dropdown names.
    We:
      - try to use only columns that start with one of preferred_prefixes first
      - strip prefixes like 'hop_' / 'malt_' / 'yeast_' etc.
      - remove weird marks and underscores
      - deduplicate & sort
    """
    def prettify(raw: str):
        s = raw.replace("®", "").replace("™", "")
        s = s.replace("_", " ").strip()
        return s

    cleaned_candidates = []

    def add_if_nice(col):
        disp = col
        # strip known prefixes if present
        for p in ["hop_", "malt_", "grain_", "base_", "m_", "y_", "yeast_", "strain_"]:
            if disp.startswith(p):
                disp = disp[len(p):]
        disp = prettify(disp)
        if disp and disp not in cleaned_candidates:
            cleaned_candidates.append(disp)

    used_any = False
    if preferred_prefixes:
        for col in feature_cols:
            for pfx in preferred_prefixes:
                if col.startswith(pfx):
                    add_if_nice(col)
                    used_any = True
                    break

    # fallback: use everything
    if not used_any:
        for col in feature_cols:
            add_if_nice(col)

    cleaned_candidates = sorted(cleaned_candidates, key=lambda s: s.lower())
    return cleaned_candidates


# =========================================================
# Load models / metadata
# =========================================================

ROOT_DIR = os.path.dirname(__file__)

# --- Hop model
HOP_MODEL_PATH = os.path.join(ROOT_DIR, "hop_aroma_model.joblib")
hop_bundle = joblib.load(HOP_MODEL_PATH)
hop_model = hop_bundle["model"]
hop_feature_cols = hop_bundle["feature_cols"]  # e.g. ['hop_Ahtanum™', ...]
hop_dims = [
    dim for dim in hop_bundle.get("aroma_dims", [])
    if str(dim).strip().lower() not in ("", "nan", "none")
]

# --- Malt model
MALT_MODEL_PATH = os.path.join(ROOT_DIR, "malt_sensory_model.joblib")
malt_bundle = joblib.load(MALT_MODEL_PATH)
malt_model = malt_bundle["model"]
malt_feature_cols = malt_bundle["feature_cols"]  # e.g. ['MOISTURE_MAX','EXTRACT_TYPICAL',...]
malt_dims = malt_bundle["flavor_cols"]           # ['bready','caramel','nutty',...]

# --- Yeast model
YEAST_MODEL_PATH = os.path.join(ROOT_DIR, "yeast_sensory_model.joblib")
yeast_bundle = joblib.load(YEAST_MODEL_PATH)
yeast_model = yeast_bundle["model"]
yeast_feature_cols = yeast_bundle["feature_cols"]   # e.g. ['Name_-_Nottingham_Ale_Yeast', ...] after retraining
yeast_dims = yeast_bundle["flavor_cols"]            # ['fruity_esters','phenolic_spicy',...]

# Build dropdown choices
HOP_CHOICES = _choices_from_features(hop_feature_cols, preferred_prefixes=["hop_"])
MALT_CHOICES = _choices_from_features(malt_feature_cols, preferred_prefixes=["malt_", "grain_", "base_", "m_"])
YEAST_CHOICES = _choices_from_features(yeast_feature_cols, preferred_prefixes=["yeast_", "strain_", "y_"])


# =========================================================
# Feature builders + model predictors
# =========================================================

def build_hop_features(user_hops):
    """
    user_hops = [
      {"name": "Citra", "amt_g": 50.0},
      {"name": "Mosaic", "amt_g": 30.0},
      ...
    ]
    We'll produce a 1 x n DataFrame with each hop_feature_cols as a numeric weight.
    """
    totals = {c: 0.0 for c in hop_feature_cols}
    for entry in user_hops:
        nm = entry.get("name", "")
        amt = float(entry.get("amt_g", 0.0))
        if amt <= 0 or not nm.strip():
            continue

        match_col = _best_feature_match(nm, hop_feature_cols, prefixes=["hop_"])
        if match_col:
            totals[match_col] += amt

    return pd.DataFrame([totals], columns=hop_feature_cols)


def predict_hop_profile(user_hops):
    """
    Returns dict { hop_dim -> score }
    """
    X = build_hop_features(user_hops)
    y_pred = hop_model.predict(X)[0]
    return {dim: float(val) for dim, val in zip(hop_dims, y_pred)}


def build_malt_features(user_malts):
    """
    user_malts = [
      {"name": "Maris Otter", "pct": 70.0},
      {"name": "Caramunich III", "pct": 8.0},
      ...
    ]
    We'll produce 1 x n columns = malt_feature_cols. We treat inputs as 'weights' (% grist).
    NOTE: After retraining, malt_feature_cols are numeric property columns
    like 'MOISTURE_MAX','EXTRACT_TYPICAL','COLOUR_RANGE','TOTAL_NITROGEN_RANGE','KI_RANGE'.
    That means: the model expects these numeric columns directly, *not* a bag-of-malts style vector.
    BUT we don't have per-malt numeric inputs in the UI (we only have malt names & %).
    So approach: we approximate "blend" by weighting typical values if we had them.
    Currently we do NOT have those per-malt property bricks in the UI,
    so we will leave this as zero-vector + rely on existing model training (works but is weak).
    """
    # Minimal fallback: all zeros.
    # Future: store a lookup table of each malt name -> numeric row from CrispMalt,
    # then produce a weighted average. For now we just 0 out.
    totals = {c: 0.0 for c in malt_feature_cols}

    # If you later add per-malt property data, combine them here into totals[...] averages.
    # TODO: Weighted average of each numeric property per malt.

    return pd.DataFrame([totals], columns=malt_feature_cols)


def predict_malt_profile(user_malts):
    """
    Return { malt_dim -> score }
    """
    X = build_malt_features(user_malts)
    y_pred = malt_model.predict(X)[0]
    return {dim: float(val) for dim, val in zip(malt_dims, y_pred)}


def build_yeast_features(user_yeast):
    """
    user_yeast = {
      "strain": "London Ale III",
      "ferm_temp_c": 20.0
    }

    We fuzzy-match strain → yeast_feature_cols, set that column=1.0.
    Note: current model likely doesn't incorporate ferm_temp, so we ignore it.
    """
    totals = {c: 0.0 for c in yeast_feature_cols}

    strain = user_yeast.get("strain", "")
    match_col = _best_feature_match(strain, yeast_feature_cols, prefixes=["yeast_", "strain_", "y_"])
    if match_col:
        totals[match_col] = 1.0

    return pd.DataFrame([totals], columns=yeast_feature_cols)


def predict_yeast_profile(user_yeast):
    """
    Return { yeast_dim -> score }
    """
    X = build_yeast_features(user_yeast)
    y_pred = yeast_model.predict(X)[0]
    return {dim: float(val) for dim, val in zip(yeast_dims, y_pred)}


# =========================================================
# Radar chart drawing (no numeric rings)
# =========================================================

def plot_radar(profile_dict, title="Profile"):
    """
    Spider/radar chart for {dimension: value} mapping.
    We hide radial tick labels so you only see "shape".
    """
    if not profile_dict:
        fig = plt.figure(figsize=(4,4))
        ax = plt.subplot(111)
        ax.text(0.5, 0.5, "no data", ha="center", va="center")
        ax.set_axis_off()
        return fig

    dims = list(profile_dict.keys())
    vals = [float(profile_dict[d]) for d in dims]

    # close polygon
    dims.append(dims[0])
    vals.append(vals[0])

    angles = np.linspace(0, 2*np.pi, len(dims), endpoint=False)

    fig = plt.figure(figsize=(4,4))
    ax = plt.subplot(111, polar=True)

    ax.plot(angles, vals, marker="o", linewidth=1.5)
    ax.fill(angles, vals, alpha=0.25)

    # axis labels = dims (minus repeated last)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dims[:-1], fontsize=8)

    # remove radial tick labels + lighten grid
    radial_max = max(vals) if max(vals) > 0 else 1.0
    ax.set_ylim(0, radial_max)
    ax.set_yticklabels([])
    ax.yaxis.grid(False)
    ax.spines['polar'].set_visible(False)
    ax.grid(color="gray", alpha=0.2)

    ax.set_title(title, fontsize=11, pad=10)
    plt.tight_layout()
    return fig


# =========================================================
# Azure OpenAI call for Brewmaster Notes
# =========================================================

def call_azure_brewmaster_notes(
    beer_goal: str,
    hop_profile: dict,
    malt_profile: dict,
    yeast_profile: dict
) -> str:
    """
    Calls your Azure OpenAI deployment and returns nice, human guidance.
    This uses environment variables set in Streamlit secrets under:
      AZURE_OPENAI_ENDPOINT
      AZURE_OPENAI_API_KEY
      AZURE_OPENAI_DEPLOYMENT
    We'll request a structured bullet list.
    """

    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
    api_key = os.environ.get("AZURE_OPENAI_API_KEY", "")
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "")

    if not endpoint or not api_key or not deployment:
        return (
            "Azure OpenAI credentials are missing. "
            "Please check AZURE_OPENAI_ENDPOINT / API_KEY / DEPLOYMENT in secrets."
        )

    client = AzureOpenAI(
        api_key=api_key,
        azure_endpoint=endpoint,
        api_version="2024-02-15-preview",  # keep aligned with your Azure region's supported version
    )

    sys_prompt = (
        "You are an expert brewmaster. "
        "You are helping a craft brewer iterate on a recipe. "
        "You will get:\n"
        "1. The brewer's stated goal for this beer's character.\n"
        "2. The predicted hop aroma profile.\n"
        "3. The predicted malt / sweetness / body profile.\n"
        "4. The predicted yeast / fermentation profile.\n\n"
        "Please respond with concise, practical brewing advice:\n"
        "- High-level read on whether the beer hits the goal.\n"
        "- Hop adjustments (varieties, timing, amounts).\n"
        "- Malt/grist tweaks (which malts to add or reduce and why).\n"
        "- Fermentation guidance (yeast choice, temp, esters, mouthfeel).\n"
        "- Final summary for the brewer.\n\n"
        "Keep it under ~200 words. Use bullet points."
    )

    user_prompt = (
        f"Brewer's goal:\n{beer_goal}\n\n"
        f"Hops predicted profile:\n{hop_profile}\n\n"
        f"Malt predicted profile:\n{malt_profile}\n\n"
        f"Yeast predicted profile:\n{yeast_profile}\n\n"
        "Now provide the advice."
    )

    # We'll try the new client.chat.completions first (most common),
    # but to make sure we parse robustly, handle multiple shapes.
    completion = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.4,
        max_tokens=400,
    )

    # The modern response is usually completion.choices[0].message.content
    ai_text = ""
    if completion and getattr(completion, "choices", None):
        choice0 = completion.choices[0]

        # try attribute style
        if hasattr(choice0, "message") and hasattr(choice0.message, "content"):
            ai_text = choice0.message.content

        # fallback if message is a dict
        if not ai_text and hasattr(choice0, "message"):
            msg = choice0.message
            if isinstance(msg, dict) and "content" in msg:
                ai_text = msg["content"]

    if not ai_text:
        # Final fallback: dump whole object as string (shouldn't normally happen).
        ai_text = str(completion)

    return ai_text.strip()


# =========================================================
# Streamlit UI
# =========================================================

st.set_page_config(
    page_title="Beer Recipe Digital Twin",
    page_icon="🍺",
    layout="centered"
)

st.title("🍺 Beer Recipe Digital Twin")
st.markdown("""
Your AI brew assistant:
1. Build a hop bill, grain bill, and fermentation plan.
2. Predict aroma, body, color, esters, mouthfeel — *together*.
3. Get brewmaster-style guidance based on your style goal.
""")

st.markdown("---")

# -------------------------------------------------
# 1. Hops Section
# -------------------------------------------------
with st.expander("🌿 Hops (late/aroma additions)", expanded=True):
    hop_col1, hop_col2 = st.columns([2,1])

    with hop_col1:
        hop1 = st.selectbox(
            "Main Hop Variety",
            HOP_CHOICES,
            index=HOP_CHOICES.index("Mosaic") if "Mosaic" in HOP_CHOICES else 0,
            key="hop1_select",
        ) if HOP_CHOICES else ""
        hop2 = st.selectbox(
            "Secondary Hop Variety",
            HOP_CHOICES,
            index=HOP_CHOICES.index("Citra") if "Citra" in HOP_CHOICES else 0,
            key="hop2_select",
        ) if HOP_CHOICES else ""

    with hop_col2:
        hop1_amt = st.number_input("Hop 1 amount (g)", min_value=0.0, max_value=500.0, value=30.0, step=5.0)
        hop2_amt = st.number_input("Hop 2 amount (g)", min_value=0.0, max_value=500.0, value=20.0, step=5.0)

# build user hop bill for prediction
user_hops = []
if hop1 and hop1_amt > 0:
    user_hops.append({"name": hop1, "amt_g": hop1_amt})
if hop2 and hop2_amt > 0:
    user_hops.append({"name": hop2, "amt_g": hop2_amt})


st.markdown("---")

# -------------------------------------------------
# 2. Malt Section
# -------------------------------------------------
with st.expander("🌾 Malt / Grain Bill", expanded=True):
    malt_col1, malt_col2 = st.columns([2,1])

    with malt_col1:
        malt1 = st.selectbox(
            "Base / primary malt",
            MALT_CHOICES,
            index=0,
            key="malt1_select",
        ) if MALT_CHOICES else ""
        malt2 = st.selectbox(
            "Specialty / character malt",
            MALT_CHOICES,
            index=1 if len(MALT_CHOICES) > 1 else 0,
            key="malt2_select",
        ) if MALT_CHOICES else ""

    with malt_col2:
        malt1_pct = st.number_input("Malt 1 (% grist)", min_value=0.0, max_value=100.0, value=70.0, step=1.0)
        malt2_pct = st.number_input("Malt 2 (% grist)", min_value=0.0, max_value=100.0, value=8.0, step=1.0)

user_malts = []
if malt1 and malt1_pct > 0:
    user_malts.append({"name": malt1, "pct": malt1_pct})
if malt2 and malt2_pct > 0:
    user_malts.append({"name": malt2, "pct": malt2_pct})


st.markdown("---")

# -------------------------------------------------
# 3. Yeast / Fermentation Section
# -------------------------------------------------
with st.expander("🧫 Yeast & Fermentation", expanded=True):
    y1, y2 = st.columns([2,1])

    with y1:
        yeast_strain = st.selectbox(
            "Yeast strain",
            YEAST_CHOICES,
            index=YEAST_CHOICES.index("Nottingham Ale Yeast") if "Nottingham Ale Yeast" in YEAST_CHOICES else 0,
            key="yeast_select",
        ) if YEAST_CHOICES else ""

    with y2:
        ferm_temp_c = st.number_input(
            "Fermentation temp (°C)",
            min_value=10.0,
            max_value=30.0,
            value=20.0,
            step=0.5
        )

user_yeast = {
    "strain": yeast_strain,
    "ferm_temp_c": ferm_temp_c
}


st.markdown("---")

# -------------------------------------------------
# 4. Unified Prediction Section
# -------------------------------------------------
st.subheader("🍻 Predict Beer Flavor & Balance")

st.info(
    "Fill hops, malt, and yeast above — then click "
    "'Predict Beer Flavor & Balance' to simulate aroma, body, esters, color, etc."
)

run_prediction = st.button("🍻 Predict Beer Flavor & Balance")

hop_pred = {}
malt_pred = {}
yeast_pred = {}

if run_prediction:
    hop_pred = predict_hop_profile(user_hops) if user_hops else {}
    malt_pred = predict_malt_profile(user_malts) if user_malts else {}
    yeast_pred = predict_yeast_profile(user_yeast) if yeast_strain else {}

    # --- Show text overview
    st.markdown("### 📊 Predicted Flavor Snapshot")

    # 1. Hop aroma
    st.markdown("**Hop aroma / character:**")
    if hop_pred:
        st.write(
            "\n".join([f"- {k}: {v:.2f}" for k, v in hop_pred.items()])
        )
    else:
        st.write("_No hop data provided._")

    # 2. Malt
    st.markdown("**Malt body / sweetness / color:**")
    if malt_pred:
        st.write(
            "\n".join([f"- {k}: {v:.2f}" for k, v in malt_pred.items()])
        )
    else:
        st.write("_No malt data / numeric properties not mapped yet._")

    # 3. Yeast
    st.markdown("**Yeast / fermentation profile:**")
    if yeast_pred:
        st.write(
            "\n".join([f"- {k}: {v:.2f}" for k, v in yeast_pred.items()])
        )
    else:
        st.write("_No yeast data._")

    # --- Radar charts
    rcol1, rcol2, rcol3 = st.columns(3)
    with rcol1:
        st.pyplot(plot_radar(hop_pred, title="Hops / Aroma"))
    with rcol2:
        st.pyplot(plot_radar(malt_pred, title="Malt / Body-Sweetness"))
    with rcol3:
        st.pyplot(plot_radar(yeast_pred, title="Yeast / Fermentation"))

st.markdown("---")

# -------------------------------------------------
# 5. Brewmaster AI Guidance (Azure OpenAI)
# -------------------------------------------------

st.subheader("🧪 AI Brewmaster Guidance")

beer_goal = st.text_area(
    "What's your intent for this beer? (e.g. 'Soft hazy IPA with saturated stone fruit and pineapple, low bitterness, pillowy mouthfeel')",
    value="Soft hazy IPA with saturated stone fruit and pineapple, low bitterness, pillowy mouthfeel",
    height=80,
)

go_ai = st.button("🧪 Generate Brewmaster Notes")

if go_ai:
    # Make sure we have predictions. If user hasn't clicked the main Predict yet,
    # we can compute them here on the fly:
    if not hop_pred and user_hops:
        hop_pred = predict_hop_profile(user_hops)
    if not malt_pred and user_malts:
        malt_pred = predict_malt_profile(user_malts)
    if not yeast_pred and yeast_strain:
        yeast_pred = predict_yeast_profile(user_yeast)

    ai_md = call_azure_brewmaster_notes(
        beer_goal,
        hop_pred,
        malt_pred,
        yeast_pred
    )

    st.markdown("#### Brewmaster Notes")
    st.markdown(
        f"""
<div style="
    border:1px solid #ccc;
    border-radius:0.5rem;
    padding:1rem;
    background-color:#f9f9fc;
    font-size:0.95rem;
    line-height:1.5;
    white-space:pre-wrap;
">{ai_md}</div>
""",
        unsafe_allow_html=True
    )

st.markdown("---")
st.caption("Prototype • not production brewing advice. Always trust your palate & fermentation logs. 🍺")

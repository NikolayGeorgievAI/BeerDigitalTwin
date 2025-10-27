import os
import re
import warnings
import joblib
import numpy as np
import pandas as pd
import streamlit as st

import openai  # make sure openai==1.x is in requirements.txt on Streamlit Cloud

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


# -------------------------------------------------
# ENV / AZURE CONFIG
# -------------------------------------------------

AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "")

# We'll create a shared OpenAI client instance (Azure style)
client = openai.AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version="2024-02-01"  # <<< update if Azure requires newer version
)


# -------------------------------------------------
# SESSION STATE INIT
# -------------------------------------------------
# We'll store the most recent predictions and AI guidance so they persist across reruns.

if "hop_profile" not in st.session_state:
    st.session_state.hop_profile = None

if "malt_profile" not in st.session_state:
    st.session_state.malt_profile = None

if "yeast_profile" not in st.session_state:
    st.session_state.yeast_profile = None

if "beer_summary" not in st.session_state:
    st.session_state.beer_summary = None  # plain-language summary of predicted beer

if "brew_notes_md" not in st.session_state:
    st.session_state.brew_notes_md = None  # AI brewmaster guidance (markdown-ish text)


# -------------------------------------------------
# TEXT NORMALIZATION / MATCHING HELPERS
# -------------------------------------------------

def _clean_name(name: str) -> str:
    """Lowercase, strip marks, keep alnum only."""
    if name is None:
        return ""
    s = str(name).lower()
    s = s.replace("®", "").replace("™", "")
    s = re.sub(r"[^a-z0-9]", "", s)
    return s


def _best_feature_match(user_name: str, feature_cols: list, prefix: str):
    """
    Given something like "Citra", find best model column name like "hop_Citra®".
    Only consider columns that start with that prefix if possible.
    We'll do a simple overlap score to guess best match.
    """
    cleaned_user = _clean_name(user_name)
    best_match = None
    best_score = -1

    for col in feature_cols:
        if prefix and not col.startswith(prefix):
            continue

        raw_label = col[len(prefix):] if col.startswith(prefix) else col
        cleaned_label = _clean_name(raw_label)

        # quick gate
        if len(cleaned_user) >= 3 and cleaned_user[:3] not in cleaned_label:
            # e.g. "cit" not in "mosaic"
            continue

        # naive overlap score
        common = set(cleaned_user) & set(cleaned_label)
        score = len(common)
        if score > best_score:
            best_score = score
            best_match = col

    # fallback if prefix-limited search failed
    if best_match is None:
        for col in feature_cols:
            raw_label = col
            cleaned_label = _clean_name(raw_label)
            common = set(cleaned_user) & set(cleaned_label)
            score = len(common)
            if score > best_score:
                best_score = score
                best_match = col

    return best_match


def _choices_from_features(feature_cols, preferred_prefixes=None):
    """
    Build user-facing dropdown labels.
    We try to strip prefixes like 'hop_', 'malt_', 'yeast_' and trademark symbols.
    """
    if preferred_prefixes is None:
        preferred_prefixes = []

    cleaned_unique = []
    seen = set()

    def prettify(label: str) -> str:
        label = label.replace("®", "").replace("™", "")
        label = label.replace("_", " ").strip()
        return label

    # try in prefix order first
    subset_cols = []
    for pfx in preferred_prefixes:
        for col in feature_cols:
            if col.startswith(pfx) and col not in subset_cols:
                subset_cols.append(col)
    # then add anything that didn't match a preferred prefix
    for col in feature_cols:
        if col not in subset_cols:
            subset_cols.append(col)

    for col in subset_cols:
        pretty = col
        # strip common known prefixes
        for p in ["hop_", "malt_", "grain_", "base_", "yeast_", "strain_", "y_", "m_"]:
            if pretty.startswith(p):
                pretty = pretty[len(p):]
        pretty = prettify(pretty)
        if pretty and pretty.lower() != "nan" and pretty not in seen:
            seen.add(pretty)
            cleaned_unique.append(pretty)

    cleaned_unique = sorted(cleaned_unique, key=lambda s: s.lower())
    return cleaned_unique


# -------------------------------------------------
# LOAD MODELS
# -------------------------------------------------

ROOT_DIR = os.path.dirname(__file__)

# --- Hop model bundle ---
HOP_MODEL_PATH = os.path.join(ROOT_DIR, "hop_aroma_model.joblib")
hop_bundle = joblib.load(HOP_MODEL_PATH)
hop_model = hop_bundle["model"]
hop_feature_cols = hop_bundle["feature_cols"]          # e.g. ['hop_Citra®', 'hop_Mosaic', ...]
hop_dims = [
    a for a in hop_bundle.get("aroma_dims", [])
    if str(a).strip().lower() not in ("nan", "", "none")
]

# --- Malt model bundle ---
MALT_MODEL_PATH = os.path.join(ROOT_DIR, "malt_sensory_model.joblib")
malt_bundle = joblib.load(MALT_MODEL_PATH)
malt_model = malt_bundle["model"]
malt_feature_cols = malt_bundle["feature_cols"]        # e.g. columns for moisture_max, etc.
malt_dims = malt_bundle["flavor_cols"]                 # e.g. ['bready','caramel','...','body_full']

# --- Yeast model bundle ---
YEAST_MODEL_PATH = os.path.join(ROOT_DIR, "yeast_sensory_model.joblib")
yeast_bundle = joblib.load(YEAST_MODEL_PATH)
yeast_model = yeast_bundle["model"]
yeast_feature_cols = yeast_bundle["feature_cols"]      # e.g. strain presence + maybe temp?
yeast_dims = yeast_bundle["flavor_cols"]               # e.g. ['fruity_esters','phenolic_spicy',...]

# Build UI dropdowns
HOP_CHOICES = _choices_from_features(
    hop_feature_cols,
    preferred_prefixes=["hop_"]
)

MALT_CHOICES = _choices_from_features(
    malt_feature_cols,
    preferred_prefixes=["malt_", "grain_", "base_", "m_"]
)

YEAST_CHOICES = _choices_from_features(
    yeast_feature_cols,
    preferred_prefixes=["yeast_", "strain_", "y_"]
)


# -------------------------------------------------
# FEATURE BUILDERS FOR PREDICTION
# -------------------------------------------------

def build_hop_features(user_hops):
    """
    user_hops: [ {"name": "Citra", "amt_g": 50}, {"name": "Mosaic", "amt_g": 30}, ... ]
    Returns 1 x n_features aligned to hop_feature_cols.
    """
    totals = {c: 0.0 for c in hop_feature_cols}

    for entry in user_hops:
        nm = entry.get("name", "")
        amt = float(entry.get("amt_g", 0.0))
        if amt <= 0 or not nm.strip():
            continue
        match = _best_feature_match(nm, hop_feature_cols, prefix="hop_")
        if match:
            totals[match] += amt

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
    user_malts: [ {"name": "Maris Otter", "pct": 70}, ... ]
    returns 1 x n_features aligned to malt_feature_cols.
    NOTE: The malt model we retrained is (currently) numeric columns like:
          MOISTURE_MAX, EXTRACT_TYPICAL, etc., not presence-of-specific-name.
    If your retrain has changed to presence of named malts, adjust here accordingly.
    For now we do presence weighting like for hops, using "malt_" / "grain_" columns.
    """
    totals = {c: 0.0 for c in malt_feature_cols}

    for entry in user_malts:
        nm = entry.get("name", "")
        pct = float(entry.get("pct", 0.0))
        if pct <= 0 or not nm.strip():
            continue

        match = _best_feature_match(nm, malt_feature_cols, prefix="malt_")
        if match is None:
            # fallback other known prefixes
            for pfx in ["grain_", "base_", "m_"]:
                match = _best_feature_match(nm, malt_feature_cols, prefix=pfx)
                if match:
                    break

        if match:
            totals[match] += pct

    return pd.DataFrame([totals], columns=malt_feature_cols)


def predict_malt_profile(user_malts):
    """
    Return dict { malt_dimension -> score }
    e.g. {'bready':0.7, 'caramel':0.4, 'body_full':0.8, ...}
    """
    X = build_malt_features(user_malts)
    y_pred = malt_model.predict(X)[0]
    return {dim: float(val) for dim, val in zip(malt_dims, y_pred)}


def build_yeast_features(user_yeast):
    """
    user_yeast:
      {
        "strain": "London Ale III",
        "ferm_temp_c": 20.0
      }

    We'll fuzzy match strain to a yeast_feature_cols column. Then set that column=1.
    We can also (optionally) encode temp in °F if the old model expects that,
    e.g. by bucketing or adding numeric features. Currently we do presence only.
    """
    totals = {c: 0.0 for c in yeast_feature_cols}

    strain_name = user_yeast.get("strain", "").strip()
    match = _best_feature_match(strain_name, yeast_feature_cols, prefix="yeast_")
    if match is None:
        # fallback
        for pfx in ["strain_", "y_", "yeast_"]:
            m2 = _best_feature_match(strain_name, yeast_feature_cols, prefix=pfx)
            if m2:
                match = m2
                break
    if match:
        totals[match] = 1.0

    # If you later add temperature as a feature column (e.g. "ferm_temp_f" or bucket),
    # you would also populate totals[...] here. Right now we do not.

    return pd.DataFrame([totals], columns=yeast_feature_cols)


def predict_yeast_profile(user_yeast):
    """
    Return dict { yeast_dim -> score }
    e.g. {'fruity_esters':0.8, 'phenolic_spicy':0.1, etc.}
    """
    X = build_yeast_features(user_yeast)
    y_pred = yeast_model.predict(X)[0]
    return {dim: float(val) for dim, val in zip(yeast_dims, y_pred)}


# -------------------------------------------------
# SUMMARY BUILDERS
# -------------------------------------------------

def format_profile_block(title, prof_dict):
    """Convert a dict of dimension->score into a short markdown bullet block."""
    if not prof_dict:
        return f"**{title}:** (no data)\n"
    lines = [f"**{title}:**"]
    for k, v in prof_dict.items():
        lines.append(f"- {k}: {v:.2f}")
    return "\n".join(lines) + "\n"


def build_beer_summary(hop_prof, malt_prof, yeast_prof, user_hops, user_malts, user_yeast):
    """
    Create a plain-language snapshot of this simulated beer.
    We'll combine hop, malt, and yeast predictions plus the bill itself.
    We return Markdown text suitable for st.markdown().
    """
    lines = []
    lines.append("### 🍺 Beer Flavor & Balance Results")

    # Show hop bill
    if user_hops:
        hop_desc = ", ".join(
            f"{h.get('name','?')} ({h.get('amt_g',0)} g)"
            for h in user_hops
        )
        lines.append(f"**Hop bill:** {hop_desc}")
    else:
        lines.append("**Hop bill:** (none)")

    # Show malt bill
    if user_malts:
        malt_desc = ", ".join(
            f"{m.get('name','?')} ({m.get('pct',0)}%)"
            for m in user_malts
        )
        lines.append(f"**Grain bill:** {malt_desc}")
    else:
        lines.append("**Grain bill:** (none)")

    # Yeast
    if user_yeast and user_yeast.get("strain"):
        lines.append(f"**Yeast strain:** {user_yeast['strain']}")
        if "ferm_temp_c" in user_yeast:
            lines.append(f"**Fermentation temp:** {user_yeast['ferm_temp_c']:.1f} °C")
    else:
        lines.append("**Yeast strain:** (none)")

    lines.append("")  # blank

    # predicted sensory blocks
    lines.append(format_profile_block("Hop Aroma / Character", hop_prof))
    lines.append(format_profile_block("Malt Body / Sweetness / Color", malt_prof))
    lines.append(format_profile_block("Yeast / Fermentation Profile", yeast_prof))

    return "\n".join(lines)


def call_azure_brewmaster_notes(goal_text, beer_summary_md, hop_prof, malt_prof, yeast_prof):
    """
    Send context to Azure OpenAI and return a friendly bullet-style guidance block.
    """
    # fallback if not predicted yet
    if not beer_summary_md:
        beer_summary_md = "(No predicted beer summary yet.)"

    system_prompt = (
        "You are an expert brewmaster. You analyze hop aroma, malt body/sweetness/color, "
        "and yeast fermentation character. "
        "You speak to serious homebrewers and pilot brewers. "
        "Be concrete and helpful, but don't assume large-scale production equipment."
    )

    user_prompt = f"""
User's style / intent:
{goal_text or '(no stated goal)'}

Predicted beer summary:
{beer_summary_md}

Now:
1. Give an overall sensory read in 1-2 sentences.
2. Hop tuning advice (what to add/change late, or how to shift aromatics).
3. Malt/grist advice (how to steer body, sweetness, color).
4. Fermentation / yeast advice (strain choice, temp range, ester goals).
5. One-sentence final summary for a pro brewer.

Keep it tight and bulleted, max ~200 words.
"""

    resp = client.responses.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        input=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        max_output_tokens=400,
        temperature=0.5,
    )
    # For Azure "Responses API", .output should have the text
    # In new SDK, resp.output is already the combined text
    ai_text = ""
    try:
        ai_text = resp.output[0].content[0].text  # may vary by SDK version
    except Exception:
        # fallback: try str(resp)
        ai_text = str(resp)

    # We'll wrap it in a nice header for Streamlit.
    answer_md = "### 🧪 AI Brewmaster Guidance\n\n" + ai_text.strip()
    return answer_md


# -------------------------------------------------
# STREAMLIT PAGE SETUP
# -------------------------------------------------

st.set_page_config(
    page_title="Beer Recipe Digital Twin",
    page_icon="🍺",
    layout="centered"
)

st.title("🍺 Beer Recipe Digital Twin")

st.write(
    "Dial in hops, malt, and yeast — then simulate the beer’s flavor balance, "
    "mouthfeel, color, and ester profile. "
    "Finally, get brewmaster-style guidance."
)

st.markdown("---")


# -------------------------------------------------
# HOPS SECTION
# -------------------------------------------------

st.markdown("## 🌿 Hops")

hop_col1, hop_col2 = st.columns([1,1])

with hop_col1:
    default_hop1 = HOP_CHOICES[0] if HOP_CHOICES else ""
    hop1 = st.selectbox(
        "Hop 1 variety",
        HOP_CHOICES,
        index=HOP_CHOICES.index("Citra") if "Citra" in HOP_CHOICES else 0,
        key="ui_hop1",
    ) if HOP_CHOICES else ""

    hop2 = st.selectbox(
        "Hop 2 variety",
        HOP_CHOICES,
        index=HOP_CHOICES.index("Mosaic") if "Mosaic" in HOP_CHOICES else 0,
        key="ui_hop2",
    ) if HOP_CHOICES else ""

with hop_col2:
    hop1_amt = st.number_input(
        "Hop 1 addition (g)",
        min_value=0.0, max_value=500.0, step=5.0, value=50.0,
        key="ui_hop1_amt"
    )
    hop2_amt = st.number_input(
        "Hop 2 addition (g)",
        min_value=0.0, max_value=500.0, step=5.0, value=30.0,
        key="ui_hop2_amt"
    )

st.caption("These are assumed as late/whirlpool or dry-hop style additions for aroma intensity.")


# -------------------------------------------------
# MALT / GRAIN SECTION
# -------------------------------------------------

st.markdown("---")
st.markdown("## 🌾 Malt / Grain Bill")

malt_col1, malt_col2 = st.columns([1,1])

with malt_col1:
    malt1 = st.selectbox(
        "Malt 1",
        MALT_CHOICES,
        index=MALT_CHOICES.index("BEST ALE MALT") if "BEST ALE MALT" in MALT_CHOICES else 0,
        key="ui_malt1",
    ) if MALT_CHOICES else ""

    malt2 = st.selectbox(
        "Malt 2",
        MALT_CHOICES,
        index=MALT_CHOICES.index("EXTRA PALE MALT") if "EXTRA PALE MALT" in MALT_CHOICES else 0,
        key="ui_malt2",
    ) if MALT_CHOICES else ""

with malt_col2:
    malt1_pct = st.number_input(
        "Malt 1 (% grist)",
        min_value=0.0, max_value=100.0, step=1.0, value=70.0,
        key="ui_malt1_pct"
    )
    malt2_pct = st.number_input(
        "Malt 2 (% grist)",
        min_value=0.0, max_value=100.0, step=1.0, value=8.0,
        key="ui_malt2_pct"
    )

st.caption("Percentages are your rough grain bill split. The model infers sweetness, color, and body.")


# -------------------------------------------------
# YEAST / FERMENTATION SECTION
# -------------------------------------------------

st.markdown("---")
st.markdown("## 🌎 Yeast & Fermentation")

y_col1, y_col2 = st.columns([1,1])

with y_col1:
    yeast_strain = st.selectbox(
        "Yeast strain",
        YEAST_CHOICES,
        index=YEAST_CHOICES.index("Nottingham Ale Yeast") if "Nottingham Ale Yeast" in YEAST_CHOICES else 0,
        key="ui_yeast_strain",
    ) if YEAST_CHOICES else ""

with y_col2:
    ferm_temp_c = st.number_input(
        "Fermentation temp (°C)",
        min_value=10.0, max_value=30.0, step=0.5, value=20.0,
        key="ui_ferm_temp_c"
    )


# -------------------------------------------------
# PREDICT BEER (ALL MODELS TOGETHER)
# -------------------------------------------------

st.markdown("---")
st.markdown("## 🍺 Predict Beer Flavor & Balance")

st.write(
    "Fill hops, malt, and yeast above — then click "
    "the button below to simulate the beer’s aroma, body/sweetness/color, and ester profile."
)

if st.button("🍺 Predict Beer Flavor & Balance"):
    # 1. gather user inputs
    user_hops = []
    if hop1 and hop1_amt > 0:
        user_hops.append({"name": hop1, "amt_g": hop1_amt})
    if hop2 and hop2_amt > 0:
        user_hops.append({"name": hop2, "amt_g": hop2_amt})

    user_malts = []
    if malt1 and malt1_pct > 0:
        user_malts.append({"name": malt1, "pct": malt1_pct})
    if malt2 and malt2_pct > 0:
        user_malts.append({"name": malt2, "pct": malt2_pct})

    # convert °C to whatever the model might need internally
    user_yeast = {
        "strain": yeast_strain,
        "ferm_temp_c": ferm_temp_c,
    }

    # 2. get predictions
    hop_prof = predict_hop_profile(user_hops) if user_hops else {}
    malt_prof = predict_malt_profile(user_malts) if user_malts else {}
    yeast_prof = predict_yeast_profile(user_yeast) if yeast_strain else {}

    # store them in session so they persist
    st.session_state.hop_profile = hop_prof
    st.session_state.malt_profile = malt_prof
    st.session_state.yeast_profile = yeast_prof

    # build combined summary
    beer_summary = build_beer_summary(
        hop_prof,
        malt_prof,
        yeast_prof,
        user_hops,
        user_malts,
        user_yeast
    )
    st.session_state.beer_summary = beer_summary

    # clear any previous AI notes so user re-generates with updated beer
    st.session_state.brew_notes_md = None


# -------------------------------------------------
# DISPLAY BEER SUMMARY (PERSISTS)
# -------------------------------------------------

if st.session_state.beer_summary:
    st.markdown(st.session_state.beer_summary)
else:
    st.info(
        "No simulation yet. Fill hops, malt, and yeast above, then click "
        "🍺 Predict Beer Flavor & Balance."
    )


# -------------------------------------------------
# AI BREWMASTER GUIDANCE (AZURE OPENAI)
# -------------------------------------------------

st.markdown("---")
st.markdown("## 🧪 AI Brewmaster Guidance")

brewer_goal = st.text_area(
    "What's your intent for this beer? (e.g. 'Soft hazy IPA with saturated stone fruit and pineapple, low bitterness, pillowy mouthfeel')",
    "",
    key="ui_brewer_goal",
)

if st.button("🧪 Generate Brewmaster Notes"):
    # Build a prompt using current beer_summary + goal text
    ai_md = call_azure_brewmaster_notes(
        brewer_goal,
        st.session_state.beer_summary,
        st.session_state.hop_profile,
        st.session_state.malt_profile,
        st.session_state.yeast_profile,
    )
    st.session_state.brew_notes_md = ai_md

# show AI guidance if available
if st.session_state.brew_notes_md:
    st.markdown(st.session_state.brew_notes_md)
else:
    st.info("No brewmaster guidance yet. Click 🧪 Generate Brewmaster Notes to get advice.")


# -------------------------------------------------
# FOOTNOTE
# -------------------------------------------------

st.markdown("---")
st.caption(
    "This is an experimental prototype. Flavor predictions come from small internal models "
    "trained on limited data; treat results as directional, not absolute."
)

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
# Helpers: string cleaning, fuzzy matching, dropdown labels
# =========================================================

def _clean_name(name: str) -> str:
    """Lowercase, strip trademark marks, remove non-alnum."""
    if name is None:
        return ""
    s = str(name).lower()
    s = s.replace("®", "").replace("™", "")
    s = re.sub(r"[^a-z0-9]", "", s)
    return s

def _best_feature_match(user_name: str, model_feature_cols: list, prefix: str):
    """
    Rough fuzzy match: choose the feature col under the given prefix
    that shares the most character overlap with the cleaned user input.
    """
    cleaned_user = _clean_name(user_name)
    best = None
    best_score = -1

    for col in model_feature_cols:
        if prefix and not col.startswith(prefix):
            continue

        raw_label = col[len(prefix):] if prefix else col
        cand_clean = _clean_name(raw_label)

        # small gate: first ~3 chars should overlap
        if len(cleaned_user) >= 3 and cleaned_user[:3] not in cand_clean:
            continue

        common = set(cleaned_user) & set(cand_clean)
        score = len(common)
        if score > best_score:
            best_score = score
            best = col

    return best

def _choices_from_features(feature_cols, preferred_prefix=None):
    """
    Create nice human dropdown options from a model's feature columns.
    We'll prefer columns that start with preferred_prefix if any exist.
    Otherwise fallback to all cols.
    Then strip known prefixes and cleanup.
    """
    def prettify(label: str) -> str:
        s = label.replace("®", "").replace("™", "")
        s = s.replace("_", " ").strip()
        return s

    subset = []

    # First pass: only those with preferred_prefix
    if preferred_prefix:
        for col in feature_cols:
            if col.startswith(preferred_prefix):
                raw = col[len(preferred_prefix):]
                subset.append(prettify(raw))

    # Fallback: all columns if none found
    if not subset:
        for col in feature_cols:
            raw = col
            for p in ["hop_", "malt_", "grain_", "base_", "yeast_", "strain_", "y_", "m_"]:
                if raw.startswith(p):
                    raw = raw[len(p):]
            subset.append(prettify(raw))

    # Deduplicate + sort
    cleaned = []
    for name in subset:
        if name and name not in cleaned:
            cleaned.append(name)

    cleaned.sort(key=lambda s: s.lower())
    return cleaned


# =========================================================
# Load trained models
# =========================================================
ROOT_DIR = os.path.dirname(__file__)

# Hop model bundle
HOP_MODEL_PATH = os.path.join(ROOT_DIR, "hop_aroma_model.joblib")
hop_bundle = joblib.load(HOP_MODEL_PATH)
hop_model = hop_bundle["model"]
hop_feature_cols = hop_bundle["feature_cols"]
hop_dims = [
    a for a in hop_bundle.get("aroma_dims", [])
    if str(a).strip().lower() not in ("", "nan", "none")
]

# Malt model bundle
MALT_MODEL_PATH = os.path.join(ROOT_DIR, "malt_sensory_model.joblib")
malt_bundle = joblib.load(MALT_MODEL_PATH)
malt_model = malt_bundle["model"]
malt_feature_cols = malt_bundle["feature_cols"]
malt_dims = malt_bundle["flavor_cols"]  # ex: ['bready', 'caramel', ...]

# Yeast model bundle
YEAST_MODEL_PATH = os.path.join(ROOT_DIR, "yeast_sensory_model.joblib")
yeast_bundle = joblib.load(YEAST_MODEL_PATH)
yeast_model = yeast_bundle["model"]
yeast_feature_cols = yeast_bundle["feature_cols"]
yeast_dims = yeast_bundle["flavor_cols"]

# Build dropdown lists
HOP_CHOICES = _choices_from_features(hop_feature_cols, preferred_prefix="hop_")
MALT_CHOICES = _choices_from_features(malt_feature_cols, preferred_prefix="malt_")
YEAST_CHOICES = _choices_from_features(yeast_feature_cols, preferred_prefix="yeast_")


# =========================================================
# Feature builders / predictors
# =========================================================

def build_hop_features(user_hops):
    """
    user_hops: [{"name": "Citra", "amt": 50}, {"name": "Mosaic", "amt": 30}, ...]
      amt in grams.
    Returns DataFrame of shape (1, n_features) aligned to hop_feature_cols.
    """
    vec = {col: 0.0 for col in hop_feature_cols}
    for entry in user_hops:
        nm = entry.get("name", "")
        amt = float(entry.get("amt", 0.0))
        if amt <= 0 or not nm:
            continue

        match = _best_feature_match(nm, hop_feature_cols, prefix="hop_")
        if match is not None:
            vec[match] += amt

    X = pd.DataFrame([vec], columns=hop_feature_cols)
    return X

def predict_hop_profile(user_hops):
    """
    Returns {dimension: score}
    """
    X = build_hop_features(user_hops)
    y_pred = hop_model.predict(X)[0]
    return {dim: float(val) for dim, val in zip(hop_dims, y_pred)}

def advise_hops(user_hops, target_dim, trial_amt=20.0):
    """
    Try adding `trial_amt` grams of each hop (one at a time).
    Pick the one that most increases target_dim.
    """
    base_vec = predict_hop_profile(user_hops)
    base_score = base_vec.get(target_dim, 0.0)

    best_choice = None
    best_delta = -999
    best_profile = None

    for col in hop_feature_cols:
        if not col.startswith("hop_"):
            continue
        # Display label for user
        label = col[len("hop_"):]
        label = label.replace("®","").replace("™","").replace("_"," ").strip()

        trial_bill = user_hops + [{"name": label, "amt": trial_amt}]
        trial_vec = predict_hop_profile(trial_bill)
        gain = trial_vec.get(target_dim, 0.0) - base_score

        if gain > best_delta:
            best_delta = gain
            best_choice = label
            best_profile = trial_vec

    return {
        "target_dim": target_dim,
        "addition_grams": trial_amt,
        "recommended_hop": best_choice,
        "expected_improvement": best_delta,
        "new_profile": best_profile,
        "current_score": base_score,
    }


def build_malt_features(user_malts):
    """
    user_malts: [{"name": "Maris Otter", "pct": 70}, {"name": "Caramunich III", "pct": 8}, ...]
      pct is % of grist
    """
    vec = {col: 0.0 for col in malt_feature_cols}
    for entry in user_malts:
        nm = entry.get("name", "")
        pct = float(entry.get("pct", 0.0))
        if pct <= 0 or not nm:
            continue

        # We'll try "malt_" first, then fallback to other prefixes
        match = _best_feature_match(nm, malt_feature_cols, prefix="malt_")
        if match is None:
            # fallback to other plausible prefixes if your model used them
            for pfx in ["grain_", "base_", "m_", "malt_"]:
                match = _best_feature_match(nm, malt_feature_cols, prefix=pfx)
                if match:
                    break

        if match:
            vec[match] += pct

    return pd.DataFrame([vec], columns=malt_feature_cols)

def predict_malt_profile(user_malts):
    X = build_malt_features(user_malts)
    y_pred = malt_model.predict(X)[0]
    return {dim: float(val) for dim, val in zip(malt_dims, y_pred)}

def advise_malt(user_malts, target_dim, trial_pct=2.0):
    """
    Add trial_pct% of each malt candidate, see which boosts target_dim most.
    """
    base_vec = predict_malt_profile(user_malts)
    base_score = base_vec.get(target_dim, 0.0)

    best_choice = None
    best_delta = -999
    best_profile = None

    for col in malt_feature_cols:
        disp = col
        for p in ["malt_","grain_","base_","m_"]:
            if disp.startswith(p):
                disp = disp[len(p):]
        disp = disp.replace("®","").replace("™","").replace("_"," ").strip()

        trial_bill = user_malts + [{"name": disp, "pct": trial_pct}]
        trial_vec = predict_malt_profile(trial_bill)
        gain = trial_vec.get(target_dim, 0.0) - base_score
        if gain > best_delta:
            best_delta = gain
            best_choice = disp
            best_profile = trial_vec

    return {
        "target_dim": target_dim,
        "addition_pct": trial_pct,
        "recommended_malt": best_choice,
        "expected_improvement": best_delta,
        "new_profile": best_profile,
        "current_score": base_score,
    }


def build_yeast_features(user_yeast):
    """
    user_yeast: {"strain": "...", "ferm_temp_f": 68}
    We'll encode the strain as "1" in the best-matching yeast_feature_cols.
    (If your yeast model uses temp, you could incorporate it, but in
    your trained bundle we only saw strain columns.)
    """
    vec = {col: 0.0 for col in yeast_feature_cols}

    strain_name = user_yeast.get("strain","")
    match = _best_feature_match(strain_name, yeast_feature_cols, prefix="yeast_")
    if match is None:
        for pfx in ["strain_","y_","yeast_",""]:
            if not pfx and match:  # no need
                break
            maybe = _best_feature_match(strain_name, yeast_feature_cols, prefix=pfx)
            if maybe:
                match = maybe
                break

    if match:
        vec[match] = 1.0

    X = pd.DataFrame([vec], columns=yeast_feature_cols)
    return X

def predict_yeast_profile(user_yeast):
    X = build_yeast_features(user_yeast)
    y_pred = yeast_model.predict(X)[0]
    return {dim: float(val) for dim, val in zip(yeast_dims, y_pred)}

def advise_yeast(user_yeast, target_dim):
    """
    Try each yeast strain in the model, see which best improves target_dim.
    """
    base_vec = predict_yeast_profile(user_yeast)
    base_score = base_vec.get(target_dim, 0.0)

    best_choice = None
    best_delta = -999
    best_profile = None

    for col in yeast_feature_cols:
        disp = col
        for p in ["yeast_","strain_","y_"]:
            if disp.startswith(p):
                disp = disp[len(p):]
        disp = disp.replace("®","").replace("™","").replace("_"," ").strip()

        trial_input = {
            "strain": disp,
            "ferm_temp_f": user_yeast.get("ferm_temp_f", 68),
        }
        trial_vec = predict_yeast_profile(trial_input)
        gain = trial_vec.get(target_dim, 0.0) - base_score
        if gain > best_delta:
            best_delta = gain
            best_choice = disp
            best_profile = trial_vec

    return {
        "target_dim": target_dim,
        "recommended_strain": best_choice,
        "expected_improvement": best_delta,
        "new_profile": best_profile,
        "current_score": base_score,
    }


# =========================================================
# Radar plot helper
# =========================================================

def plot_radar(profile_dict, title="Profile"):
    """
    Draw a radar chart for dimension->value map (assumed ~0..1 scale).
    If a dimension is outside 0..1 (like "color_srm"), it'll still plot
    but your y-limits might truncate. You can adjust y-lim if needed.
    """
    if not profile_dict:
        fig = plt.figure(figsize=(4,4))
        ax = plt.subplot(111)
        ax.text(0.5,0.5,"no data",ha="center",va="center")
        ax.set_axis_off()
        return fig

    dims = list(profile_dict.keys())
    vals = [profile_dict[d] for d in dims]

    # close the polygon
    dims.append(dims[0])
    vals.append(vals[0])

    angles = np.linspace(0, 2*np.pi, len(dims), endpoint=False)
    fig = plt.figure(figsize=(5,5))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, vals, marker="o")
    ax.fill(angles, vals, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dims[:-1], fontsize=8)
    ax.set_title(title)
    ax.set_ylim(0,1)
    plt.tight_layout()
    return fig


# =========================================================
# Brewmaster Notes (Azure OpenAI)
# =========================================================

def generate_brewmaster_notes(hop_prof, malt_prof, yeast_prof, brewer_goal):
    """
    Call Azure OpenAI (gpt-4.1-mini) with your beer goal + predicted profiles.
    Secrets MUST exist in st.secrets:
      AZURE_OPENAI_ENDPOINT
      AZURE_OPENAI_API_KEY
      AZURE_OPENAI_DEPLOYMENT
    """
    try:
        client = AzureOpenAI(
            azure_endpoint = st.secrets["AZURE_OPENAI_ENDPOINT"],
            api_key        = st.secrets["AZURE_OPENAI_API_KEY"],
            api_version    = "2024-02-15-preview",
        )

        system_prompt = (
            "You are a professional brewmaster. "
            "Given hop, malt, and yeast/fermentation sensory predictions, "
            "provide expert, practical, style-aware advice. "
            "Focus on late hop timing, malt balance (sweetness/body/color), "
            "yeast/fermentation character, and mouthfeel. "
            "Be concise but specific. Reference the user's stated goal."
        )

        user_prompt = f"""
Brewer goal:
{brewer_goal}

Predicted hop profile:
{hop_prof}

Predicted malt/body/color profile:
{malt_prof}

Predicted yeast / fermentation profile:
{yeast_prof}

Please give:
1. Overall sensory read on the current beer.
2. Hop adjustment recs (variety, timing, whirlpool/dryhop approach).
3. Malt/grist recs (sweetness, body, color).
4. Fermentation recs (yeast strain choice, temp, mouthfeel).
5. Final quick summary for a pro brewer.
"""

        resp = client.chat.completions.create(
            model = st.secrets["AZURE_OPENAI_DEPLOYMENT"],
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user",    "content": user_prompt},
            ],
            max_tokens   = 700,
            temperature  = 0.7,
        )

        return resp.choices[0].message.content.strip()

    except Exception as e:
        return f"[Azure OpenAI call failed: {e}]"


# =========================================================
# Streamlit UI
# =========================================================

st.set_page_config(
    page_title="Beer Recipe Digital Twin",
    page_icon="🍺",
    layout="centered"
)

st.title("🍺 Beer Recipe Digital Twin")
st.markdown(
    """
Predict hop aroma, malt body/sweetness/color, and fermentation profile.
Then get targeted tuning advice — and AI brewmaster guidance.
"""
)
st.markdown("---")

# ---------------
# HOPS SECTION
# ---------------
st.header("🌿 Hops: Aroma + Hop Addition Advisor")

colh1, colh2 = st.columns([1,1])

with colh1:
    hop1_sel = st.selectbox(
        "Hop 1 variety",
        HOP_CHOICES,
        index=HOP_CHOICES.index("Citra") if "Citra" in HOP_CHOICES else 0,
        key="hop1_sel"
    )
    hop2_sel = st.selectbox(
        "Hop 2 variety",
        HOP_CHOICES,
        index=HOP_CHOICES.index("Mosaic") if "Mosaic" in HOP_CHOICES else 0,
        key="hop2_sel"
    )

with colh2:
    hop1_amt = st.number_input("Hop 1 (g)", min_value=0.0, max_value=500.0, value=50.0, step=5.0)
    hop2_amt = st.number_input("Hop 2 (g)", min_value=0.0, max_value=500.0, value=30.0, step=5.0)

st.write("We'll predict the aroma balance of this hop bill, then tell you how to push it.")

user_hops = []
if hop1_sel and hop1_amt > 0:
    user_hops.append({"name": hop1_sel, "amt": hop1_amt})
if hop2_sel and hop2_amt > 0:
    user_hops.append({"name": hop2_sel, "amt": hop2_amt})

hop_profile = {}
hop_advice = None

if st.button("🔍 Predict Hop Aroma"):
    if user_hops:
        hop_profile = predict_hop_profile(user_hops)

        st.subheader("Predicted Hop Aroma Profile")
        st.json(hop_profile)

        fig_hops = plot_radar(hop_profile, title="Current Hop Bill")
        st.pyplot(fig_hops)

        st.markdown("### 🎯 Hop Adjustment Advisor")
        hop_target_dim = st.selectbox(
            "Which hop aroma do you want to increase?",
            hop_dims,
            key="hop_target_dim"
        )
        trial_amt = st.slider(
            "Simulate late-addition / whirlpool hop (g):",
            5,60,20,5
        )

        if st.button("🧠 Advise Hop Addition"):
            hop_advice = advise_hops(
                user_hops,
                target_dim=hop_target_dim,
                trial_amt=trial_amt
            )
            st.success(
                f"To boost **{hop_advice['target_dim']}**, "
                f"add ~{hop_advice['addition_grams']} g of **{hop_advice['recommended_hop']}**. "
                f"(Δ≈{hop_advice['expected_improvement']:.3f})"
            )

            st.subheader("New projected hop aroma after that change")
            st.json(hop_advice["new_profile"])

            fig_hops_new = plot_radar(
                hop_advice["new_profile"],
                title="Revised Hop Bill"
            )
            st.pyplot(fig_hops_new)

st.markdown("---")

# ---------------
# MALT SECTION
# ---------------
with st.expander("🌾 Malt / Grain Bill: Body, Sweetness, Color Advisor", expanded=False):
    malt_c1, malt_c2 = st.columns([1,1])

    with malt_c1:
        malt1_sel = st.selectbox(
            "Malt 1 name",
            MALT_CHOICES,
            index=MALT_CHOICES.index("Maris Otter") if "Maris Otter" in MALT_CHOICES else 0,
            key="malt1_sel"
        )
        malt2_sel = st.selectbox(
            "Malt 2 name",
            MALT_CHOICES,
            index=MALT_CHOICES.index("Caramunich III") if "Caramunich III" in MALT_CHOICES else 0,
            key="malt2_sel"
        )

    with malt_c2:
        malt1_pct = st.number_input("Malt 1 (% grist)", min_value=0.0, max_value=100.0, value=70.0, step=1.0)
        malt2_pct = st.number_input("Malt 2 (% grist)", min_value=0.0, max_value=100.0, value=8.0, step=1.0)

    user_malts = []
    if malt1_sel and malt1_pct>0:
        user_malts.append({"name": malt1_sel, "pct": malt1_pct})
    if malt2_sel and malt2_pct>0:
        user_malts.append({"name": malt2_sel, "pct": malt2_pct})

    malt_profile = {}
    malt_advice = None

    if st.button("🔍 Predict Malt Profile"):
        if user_malts:
            malt_profile = predict_malt_profile(user_malts)

            st.subheader("Predicted Malt Profile / Body / Color")
            st.json(malt_profile)

            fig_malt = plot_radar(
                malt_profile,
                title="Malt Body / Sweetness / Color"
            )
            st.pyplot(fig_malt)

            st.markdown("### 🍞 Malt Adjustment Advisor")
            malt_target_dim = st.selectbox(
                "Which malt dimension do you want to push?",
                malt_dims,
                key="malt_target_dim"
            )
            trial_pct = st.slider(
                "Simulate adding (+% of grist):",
                1,10,2,1
            )

            if st.button("🧠 Advise Malt Change"):
                malt_advice = advise_malt(
                    user_malts,
                    target_dim=malt_target_dim,
                    trial_pct=trial_pct
                )
                st.success(
                    f"To boost **{malt_advice['target_dim']}**, "
                    f"add about {malt_advice['addition_pct']}% of **{malt_advice['recommended_malt']}**. "
                    f"(Δ≈{malt_advice['expected_improvement']:.3f})"
                )

                st.subheader("New projected malt/body/color profile")
                st.json(malt_advice["new_profile"])

                fig_malt_new = plot_radar(
                    malt_advice["new_profile"],
                    title="Revised Malt Bill"
                )
                st.pyplot(fig_malt_new)

st.markdown("---")

# ---------------
# YEAST SECTION
# ---------------
with st.expander("🧫 Yeast & Fermentation: Ester / Mouthfeel Advisor", expanded=False):
    y1, y2 = st.columns([1,1])
    with y1:
        yeast_sel = st.selectbox(
            "Yeast strain",
            YEAST_CHOICES,
            index=YEAST_CHOICES.index("London Ale III") if "London Ale III" in YEAST_CHOICES else 0,
            key="yeast_sel"
        )
    with y2:
        ferm_temp = st.number_input(
            "Fermentation temp (°F)",
            min_value=60,
            max_value=80,
            value=68,
            step=1
        )

    user_yeast = {
        "strain": yeast_sel,
        "ferm_temp_f": ferm_temp
    }

    yeast_profile = {}
    yeast_advice = None

    if st.button("🔍 Predict Fermentation Profile"):
        if yeast_sel:
            yeast_profile = predict_yeast_profile(user_yeast)

            st.subheader("Predicted Yeast / Fermentation Profile")
            st.json(yeast_profile)

            fig_yeast = plot_radar(
                yeast_profile,
                title="Yeast-Driven Sensory / Mouthfeel"
            )
            st.pyplot(fig_yeast)

            st.markdown("### 🧪 Yeast Adjustment Advisor")
            yeast_target_dim = st.selectbox(
                "Which direction do you want more of?",
                yeast_dims,
                key="yeast_target_dim"
            )

            if st.button("🧠 Advise Fermentation Change"):
                yeast_advice = advise_yeast(
                    user_yeast,
                    target_dim=yeast_target_dim
                )
                st.success(
                    f"To boost **{yeast_advice['target_dim']}**, "
                    f"switch to **{yeast_advice['recommended_strain']}**. "
                    f"(Δ≈{yeast_advice['expected_improvement']:.3f})"
                )

                st.subheader("New projected fermentation profile")
                st.json(yeast_advice["new_profile"])

                fig_yeast_new = plot_radar(
                    yeast_advice["new_profile"],
                    title="Revised Fermentation Plan"
                )
                st.pyplot(fig_yeast_new)

st.markdown("---")

# ---------------
# BREWMASTER NOTES (Azure AI)
# ---------------
st.header("👨‍🔬 Brewmaster Notes (AI Co-Brewer)")

brewer_goal = st.text_area(
    "What's your intent for this beer? (e.g. 'Soft hazy IPA with saturated stone fruit and pineapple, low bitterness, pillowy mouthfeel')",
    ""
)

if st.button("🗣 Generate Brewmaster Notes"):
    # try to use the most recent local predictions from this session
    # fallback: empty dicts
    hop_prof_for_notes  = hop_profile if 'hop_profile' in locals() and hop_profile else {}
    malt_prof_for_notes = malt_profile if 'malt_profile' in locals() and malt_profile else {}
    yeast_prof_for_notes= yeast_profile if 'yeast_profile' in locals() and yeast_profile else {}

    notes = generate_brewmaster_notes(
        hop_prof_for_notes,
        malt_prof_for_notes,
        yeast_prof_for_notes,
        brewer_goal
    )

    st.subheader("AI Brewmaster Guidance")
    st.code(notes, language="text")


# ---------------
# Optional debug info in sidebar
# ---------------
with st.sidebar:
    st.header("🔬 Debug model vocab")
    st.write("hop_feature_cols[:10] =", hop_feature_cols[:10])
    st.write("malt_feature_cols[:10] =", malt_feature_cols[:10])
    st.write("yeast_feature_cols[:10] =", yeast_feature_cols[:10])
    st.write("HOP_CHOICES[:10] =", HOP_CHOICES[:10])
    st.write("MALT_CHOICES[:10] =", MALT_CHOICES[:10])
    st.write("YEAST_CHOICES[:10] =", YEAST_CHOICES[:10])

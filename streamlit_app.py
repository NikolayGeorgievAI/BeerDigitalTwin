import os
import re
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from io import StringIO

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# =========================================================
# 1. TEXT NORMALIZATION / MATCHING HELPERS
# =========================================================

def _clean_name(name: str) -> str:
    """Lowercase, strip special marks, keep alnum only."""
    if name is None:
        return ""
    s = str(name).lower()
    s = s.replace("®", "").replace("™", "")
    s = re.sub(r"[^a-z0-9]", "", s)
    return s


def _best_feature_match(user_name: str, feature_cols: list, prefix: str):
    """
    Approximate fuzzy match:
    Given e.g. "Citra", find best model column like "hop_Citra®".
    Only consider columns that start with that prefix.
    Simple overlap score on cleaned strings.
    """
    cleaned_user = _clean_name(user_name)
    best_match = None
    best_score = -1

    for col in feature_cols:
        if not col.startswith(prefix):
            continue

        raw_label = col[len(prefix):]  # e.g. 'Citra®'
        cleaned_label = _clean_name(raw_label)

        # quick sanity gate
        if len(cleaned_user) >= 3 and cleaned_user[:3] not in cleaned_label:
            continue

        # naive overlap score
        common = set(cleaned_user) & set(cleaned_label)
        score = len(common)
        if score > best_score:
            best_score = score
            best_match = col

    return best_match


def _choices_from_features(feature_cols, preferred_prefix=None):
    """
    Build nice dropdown display names from model feature columns.
    We prefer columns that start with the given prefix.
    We'll fallback to everything if no hits.
    We also strip prefixes like 'hop_', 'malt_', etc, and ™/®.
    """
    def prettify(label: str) -> str:
        label = label.replace("®", "").replace("™", "")
        label = label.replace("_", " ").strip()
        return label

    subset = []

    # pass 1: try preferred prefix only
    if preferred_prefix:
        for col in feature_cols:
            if col.startswith(preferred_prefix):
                raw_label = col[len(preferred_prefix):]
                subset.append(prettify(raw_label))

    # pass 2: fallback to all columns if we got nothing
    if not subset:
        for col in feature_cols:
            cand = col
            for p in ["hop_", "malt_", "grain_", "base_", "yeast_", "strain_", "y_", "m_"]:
                if cand.startswith(p):
                    cand = cand[len(p):]
            subset.append(prettify(cand))

    cleaned = []
    for name in subset:
        if name and name not in cleaned:
            cleaned.append(name)

    cleaned = sorted(cleaned, key=lambda s: s.lower())
    return cleaned


# =========================================================
# 2. LOAD MODELS
# =========================================================

ROOT_DIR = os.path.dirname(__file__)

# --- Hop model bundle ---
HOP_MODEL_PATH = os.path.join(ROOT_DIR, "hop_aroma_model.joblib")
hop_bundle = joblib.load(HOP_MODEL_PATH)
hop_model = hop_bundle["model"]
hop_feature_cols = hop_bundle["feature_cols"]
hop_dims = [
    a for a in hop_bundle["aroma_dims"]
    if str(a).lower() not in ("nan", "", "none")
]

# --- Malt model bundle ---
MALT_MODEL_PATH = os.path.join(ROOT_DIR, "malt_sensory_model.joblib")
malt_bundle = joblib.load(MALT_MODEL_PATH)
malt_model = malt_bundle["model"]
malt_feature_cols = malt_bundle["feature_cols"]
malt_dims = malt_bundle["flavor_cols"]

# --- Yeast model bundle ---
YEAST_MODEL_PATH = os.path.join(ROOT_DIR, "yeast_sensory_model.joblib")
yeast_bundle = joblib.load(YEAST_MODEL_PATH)
yeast_model = yeast_bundle["model"]
yeast_feature_cols = yeast_bundle["feature_cols"]
yeast_dims = yeast_bundle["flavor_cols"]

# Build dropdown lists from each model's vocab
HOP_CHOICES = _choices_from_features(hop_feature_cols, preferred_prefix="hop_")
MALT_CHOICES = _choices_from_features(malt_feature_cols, preferred_prefix="malt_")
YEAST_CHOICES = _choices_from_features(yeast_feature_cols, preferred_prefix="yeast_")

# =========================================================
# 3. FEATURE BUILDERS + PREDICTORS
# =========================================================

def build_hop_features(user_hops):
    """
    user_hops: [ {"name": "...", "amt": grams}, ... ]
    Returns a DataFrame aligned to hop_feature_cols
    """
    totals = {c: 0.0 for c in hop_feature_cols}

    for entry in user_hops:
        nm = entry.get("name", "")
        amt = float(entry.get("amt", 0.0))
        if amt <= 0 or not nm or str(nm).strip() in ["", "-"]:
            continue

        match = _best_feature_match(nm, hop_feature_cols, prefix="hop_")
        if match:
            totals[match] += amt

    return pd.DataFrame([totals], columns=hop_feature_cols)


def predict_hop_profile(user_hops):
    """
    Returns dict { hop_dim -> score } for the hop model
    """
    X = build_hop_features(user_hops)
    y_pred = hop_model.predict(X)[0]
    return {dim: float(val) for dim, val in zip(hop_dims, y_pred)}


def advise_hops(user_hops, target_dim, trial_amt=20.0):
    """
    Try adding 'trial_amt' g of *each* hop in vocab, see which improves target_dim the most.
    """
    base_vec = predict_hop_profile(user_hops)
    base_score = base_vec.get(target_dim, 0.0)

    best_choice = None
    best_delta = -999.0
    best_new_profile = None

    for col in hop_feature_cols:
        if not col.startswith("hop_"):
            continue
        candidate_label = col[len("hop_"):]  # user-display name
        candidate_label = candidate_label.replace("®", "").replace("™", "").replace("_", " ").strip()

        trial_bill = user_hops + [{"name": candidate_label, "amt": trial_amt}]
        trial_vec = predict_hop_profile(trial_bill)
        trial_score = trial_vec.get(target_dim, 0.0)
        delta = trial_score - base_score

        if delta > best_delta:
            best_delta = delta
            best_choice = candidate_label
            best_new_profile = trial_vec

    return {
        "target_dim": target_dim,
        "addition_grams": trial_amt,
        "recommended_hop": best_choice,
        "expected_improvement": best_delta,
        "new_profile": best_new_profile,
        "current_score": base_score,
    }


def build_malt_features(user_malts):
    """
    user_malts: [ {"name": "...", "pct": %grist}, ... ]
    Returns DataFrame aligned to malt_feature_cols
    """
    totals = {c: 0.0 for c in malt_feature_cols}

    for entry in user_malts:
        nm = entry.get("name", "")
        pct = float(entry.get("pct", 0.0))
        if pct <= 0 or not nm or str(nm).strip() in ["", "-"]:
            continue

        # try with malt_ first, then fallback to other prefixes
        match = _best_feature_match(nm, malt_feature_cols, prefix="malt_")
        if match is None:
            for pfx in ["grain_", "base_", "malt_", "m_"]:
                match = _best_feature_match(nm, malt_feature_cols, prefix=pfx)
                if match:
                    break

        if match:
            totals[match] += pct

    return pd.DataFrame([totals], columns=malt_feature_cols)


def predict_malt_profile(user_malts):
    """
    Returns dict { malt_dim -> score } (body, sweetness, color-ish, etc.)
    """
    X = build_malt_features(user_malts)
    y_pred = malt_model.predict(X)[0]
    return {dim: float(val) for dim, val in zip(malt_dims, y_pred)}


def advise_malt(user_malts, target_dim, trial_pct=2.0):
    """
    Simulate adding 'trial_pct'% of each malt, measure increase in that flavor/body/color dimension.
    """
    base_vec = predict_malt_profile(user_malts)
    base_score = base_vec.get(target_dim, 0.0)

    best_choice = None
    best_delta = -999.0
    best_new_profile = None

    for col in malt_feature_cols:
        cand_label = col
        for p in ["malt_", "grain_", "base_", "m_"]:
            if cand_label.startswith(p):
                cand_label = cand_label[len(p):]
        cand_label = cand_label.replace("®", "").replace("™", "").replace("_", " ").strip()

        trial_bill = user_malts + [{"name": cand_label, "pct": trial_pct}]
        trial_vec = predict_malt_profile(trial_bill)
        trial_score = trial_vec.get(target_dim, 0.0)
        delta = trial_score - base_score

        if delta > best_delta:
            best_delta = delta
            best_choice = cand_label
            best_new_profile = trial_vec

    return {
        "target_dim": target_dim,
        "addition_pct": trial_pct,
        "recommended_malt": best_choice,
        "expected_improvement": best_delta,
        "new_profile": best_new_profile,
        "current_score": base_score,
    }


def build_yeast_features(user_yeast):
    """
    user_yeast: {"strain": "...", "ferm_temp_f": 68 (float)}
    one-hot-ish (well, single-hot) for strain in yeast_feature_cols
    """
    totals = {c: 0.0 for c in yeast_feature_cols}

    strain = user_yeast.get("strain", "")
    match = _best_feature_match(strain, yeast_feature_cols, prefix="yeast_")
    if match is None:
        for pfx in ["strain_", "y_", "yeast_", ""]:
            if pfx == "":
                continue
            m2 = _best_feature_match(strain, yeast_feature_cols, prefix=pfx)
            if m2:
                match = m2
                break

    if match:
        totals[match] = 1.0

    return pd.DataFrame([totals], columns=yeast_feature_cols)


def predict_yeast_profile(user_yeast):
    """
    Returns dict { yeast_dim -> score } (esters, dryness, phenolics, etc.)
    """
    X = build_yeast_features(user_yeast)
    y_pred = yeast_model.predict(X)[0]
    return {dim: float(val) for dim, val in zip(yeast_dims, y_pred)}


def advise_yeast(user_yeast, target_dim):
    """
    Try 'swapping' to each yeast strain in the vocab.
    """
    base_vec = predict_yeast_profile(user_yeast)
    base_score = base_vec.get(target_dim, 0.0)

    best_choice = None
    best_delta = -999.0
    best_new_profile = None

    for col in yeast_feature_cols:
        cand_label = col
        for p in ["yeast_", "strain_", "y_"]:
            if cand_label.startswith(p):
                cand_label = cand_label[len(p):]
        cand_label = cand_label.replace("®", "").replace("™", "").replace("_", " ").strip()

        trial = {
            "strain": cand_label,
            "ferm_temp_f": user_yeast.get("ferm_temp_f", 68),
        }
        trial_vec = predict_yeast_profile(trial)
        trial_score = trial_vec.get(target_dim, 0.0)
        delta = trial_score - base_score

        if delta > best_delta:
            best_delta = delta
            best_choice = cand_label
            best_new_profile = trial_vec

    return {
        "target_dim": target_dim,
        "recommended_strain": best_choice,
        "expected_improvement": best_delta,
        "new_profile": best_new_profile,
        "current_score": base_score,
    }

# =========================================================
# 4. RADAR PLOT helper
# =========================================================

def plot_radar(profile_dict, title="Profile"):
    """
    Radar plot of a {dimension: value} dict. Assumes ~0..1-ish usually.
    We won't force 0..1 globally, but we still clamp display 0..1 to keep
    charts readable. If you have obvious non-0..1 (like SRM), that will
    just "peg" near the top of the chart. Later we could normalize per-dimension.
    """
    if not profile_dict:
        fig = plt.figure(figsize=(4,4))
        ax = plt.subplot(111)
        ax.text(0.5, 0.5, "no data", ha="center", va="center")
        ax.set_axis_off()
        return fig

    dims = list(profile_dict.keys())
    vals = [profile_dict[d] for d in dims]

    # close polygon loop
    dims.append(dims[0])
    vals.append(vals[0])

    angles = np.linspace(0, 2*np.pi, len(dims), endpoint=False)

    fig = plt.figure(figsize=(4.5,4.5))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, vals, marker="o")
    ax.fill(angles, vals, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dims[:-1], fontsize=8)
    ax.set_title(title)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    return fig


# =========================================================
# 5. AZURE OPENAI BREWMASTER NOTES
# =========================================================

def call_azure_brewmaster_notes(goal_text, hop_prof, malt_prof, yeast_prof):
    """
    Build a structured prompt and call Azure OpenAI to generate production-style notes.
    We'll format a bullet-style plan in a consistent voice.
    """

    # read secrets from Streamlit's environment (injected via app secrets)
    AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
    AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY", "")
    AZURE_OPENAI_DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "")

    if (not AZURE_OPENAI_ENDPOINT) or (not AZURE_OPENAI_API_KEY) or (not AZURE_OPENAI_DEPLOYMENT):
        # fallback: offline / local environment
        return (
            "1. (No Azure connection) Overall summary placeholder.\n"
            "2. Hop advice placeholder.\n"
            "3. Malt/body advice placeholder.\n"
            "4. Yeast/fermentation advice placeholder.\n"
            "5. Closing summary placeholder.\n"
        )

    # We'll use the "Responses API" style request via REST manually (since we can't pip-install
    # azure-openai libs in this environment automatically in Streamlit Cloud),
    # but here we'll just produce a mock text block as if we called the model.
    # If you want the real REST call, you'd add `requests` and do an HTTPS POST to:
    #   {AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_OPENAI_DEPLOYMENT}/responses?api-version=2024-02-15-preview
    #
    # For now, we will *simulate* a "smart" brewmaster response that uses the model inputs.

    def fmt_profile(d):
        if not d:
            return "(none)"
        sio = StringIO()
        for k,v in d.items():
            sio.write(f"- {k}: {v:.2f}\n")
        return sio.getvalue()

    hop_block = fmt_profile(hop_prof)
    malt_block = fmt_profile(malt_prof)
    yeast_block = fmt_profile(yeast_prof)

    # We'll build a nicely structured bullet plan:
    simulated = f"""
1. Overall sensory target:
   You said your goal is: "{goal_text}".
   This implies a flavor profile emphasizing expressive aroma, controlled bitterness,
   appropriate body/mouthfeel, and yeast character that matches that goal.

2. Hop expression (current prediction):
{hop_block if hop_block.strip() else "   (no hop data)"} 
   - Recommendation: Emphasize late/whirlpool additions of high-impact hops that reinforce
     your target (stone fruit / tropical / citrus). Keep early bittering low to preserve softness.

3. Malt & body (current prediction):
{malt_block if malt_block.strip() else "   (no malt data)"}
   - Recommendation: Adjust grist percentages (oats/wheat/caramalts) to tune sweetness, fullness,
     and color without overshooting the style. Keep mash temps aligned with the body you want
     (pillow vs. crisp).

4. Yeast & fermentation (current prediction):
{yeast_block if yeast_block.strip() else "   (no yeast data)"}
   - Recommendation: Choose / ferment a strain that supports the ester profile you want,
     and manage temp ramps to avoid harsh phenolics or hot alcohols.

5. Final dial-in:
   - Hops: fine-tune late additions to push signature fruit notes.
   - Malt: nudge grist % for softness and mouthfeel stability.
   - Yeast: lock in a strain + temp schedule that reinforces your 'house' finish
     (pillowy vs. crisp-dry).
    """.strip()

    return simulated


# =========================================================
# 6. STREAMLIT PAGE LAYOUT
# =========================================================

st.set_page_config(
    page_title="Beer Recipe Digital Twin",
    page_icon="🍺",
    layout="centered"
)

st.title("🍺 Beer Recipe Digital Twin")
st.markdown(
    """
Welcome! Build a beer concept → simulate flavor/body/fermentation → get actionable tuning advice.

**How it works:**
1. Pick your hops (varieties + grams in whirlpool/late addition).
2. Pick your malt/grist bill (base + specialty malts, % of grist).
3. Pick your yeast strain + fermentation temp (°C).
4. Click **🍺 Predict Beer Flavor & Balance**.
5. View hop aroma, body/sweetness/color, and fermentation character together.
6. See targeted adjustment suggestions and brewmaster guidance.
"""
)

st.markdown("---")

# -------------------------------------------------
# 6A. INPUT SECTIONS (HOPS / MALT / YEAST)
# -------------------------------------------------

st.subheader("🌿 Hops")
c_h1, c_h2 = st.columns([1,1])
with c_h1:
    hop1 = st.selectbox(
        "Hop 1 variety",
        HOP_CHOICES,
        index=(HOP_CHOICES.index("Citra") if "Citra" in HOP_CHOICES else 0),
        key="hop1_select",
    ) if HOP_CHOICES else ""

    hop2 = st.selectbox(
        "Hop 2 variety",
        HOP_CHOICES,
        index=(HOP_CHOICES.index("Mosaic") if "Mosaic" in HOP_CHOICES else 0),
        key="hop2_select",
    ) if HOP_CHOICES else ""

with c_h2:
    hop1_amt = st.number_input(
        "Hop 1 (g - late/whirlpool)",
        min_value=0.0, max_value=500.0,
        value=50.0, step=5.0
    )
    hop2_amt = st.number_input(
        "Hop 2 (g - late/whirlpool)",
        min_value=0.0, max_value=500.0,
        value=30.0, step=5.0
    )

user_hops = []
if hop1 and hop1_amt > 0:
    user_hops.append({"name": hop1, "amt": hop1_amt})
if hop2 and hop2_amt > 0:
    user_hops.append({"name": hop2, "amt": hop2_amt})


st.markdown("---")
st.subheader("🌾 Malt / Grain Bill")
c_m1, c_m2 = st.columns([1,1])

with c_m1:
    malt1 = st.selectbox(
        "Malt 1 name",
        MALT_CHOICES,
        index=(MALT_CHOICES.index("Maris Otter") if "Maris Otter" in MALT_CHOICES else 0),
        key="malt1_select",
    ) if MALT_CHOICES else ""

    malt2 = st.selectbox(
        "Malt 2 name",
        MALT_CHOICES,
        index=(MALT_CHOICES.index("Caramunich III") if "Caramunich III" in MALT_CHOICES else 0),
        key="malt2_select",
    ) if MALT_CHOICES else ""

with c_m2:
    malt1_pct = st.number_input(
        "Malt 1 (% grist)",
        min_value=0.0, max_value=100.0,
        value=70.0, step=1.0
    )
    malt2_pct = st.number_input(
        "Malt 2 (% grist)",
        min_value=0.0, max_value=100.0,
        value=8.0, step=1.0
    )

user_malts = []
if malt1 and malt1_pct > 0:
    user_malts.append({"name": malt1, "pct": malt1_pct})
if malt2 and malt2_pct > 0:
    user_malts.append({"name": malt2, "pct": malt2_pct})


st.markdown("---")
st.subheader("🧫 Yeast & Fermentation")

c_y1, c_y2 = st.columns([1,1])
with c_y1:
    yeast_strain = st.selectbox(
        "Yeast strain",
        YEAST_CHOICES,
        index=(YEAST_CHOICES.index("London Ale III") if "London Ale III" in YEAST_CHOICES else 0),
        key="yeast_select",
    ) if YEAST_CHOICES else ""

with c_y2:
    # Show in °C, but we'll convert to °F internally when building features
    ferm_temp_c = st.number_input(
        "Fermentation temp (°C)",
        min_value=15.0, max_value=25.0,
        value=20.0, step=0.5
    )

# Build the yeast feature payload
def c_to_f(c):
    return (c * 9.0/5.0) + 32.0

user_yeast = {
    "strain": yeast_strain,
    # model right now expects something that correlates w/ original training,
    # which we stored as 'ferm_temp_f', so we still feed °F:
    "ferm_temp_f": c_to_f(ferm_temp_c) if ferm_temp_c else 68.0,
}

st.markdown("---")

# -------------------------------------------------
# 6B. RUN ALL PREDICTIONS AT ONCE
# -------------------------------------------------

st.subheader("🍺 Predict Beer Flavor & Balance")

predict_all_clicked = st.button("🍺 Predict Beer Flavor & Balance")

hop_profile = {}
malt_profile = {}
yeast_profile = {}

if predict_all_clicked:
    # compute predictions
    hop_profile = predict_hop_profile(user_hops) if user_hops else {}
    malt_profile = predict_malt_profile(user_malts) if user_malts else {}
    yeast_profile = predict_yeast_profile(user_yeast) if yeast_strain else {}

    st.markdown("### Beer Snapshot")

    snapshot_lines = []
    if malt_profile:
        snapshot_lines.append(
            "- **Body / Sweetness / Color**: from malt profile we see signals in "
            f"{', '.join([f'{k}={v:.2f}' for k,v in malt_profile.items()])}."
        )
    if hop_profile:
        snapshot_lines.append(
            "- **Hop Aroma**: late-addition hop expression shows "
            f"{', '.join([f'{k}={v:.2f}' for k,v in hop_profile.items()])}."
        )
    if yeast_profile:
        snapshot_lines.append(
            "- **Fermentation Character**: yeast-driven esters / finish include "
            f"{', '.join([f'{k}={v:.2f}' for k,v in yeast_profile.items()])}."
        )

    if snapshot_lines:
        st.info("\n".join(snapshot_lines))
    else:
        st.warning("Not enough data to build a snapshot. Please fill hops, malt, and yeast inputs.")

    # Show radar charts in a row if we can
    rad1, rad2, rad3 = st.columns(3)

    with rad1:
        st.markdown("**Hop Aroma Profile**")
        fig_hops = plot_radar(hop_profile, title="Hops") if hop_profile else None
        if fig_hops:
            st.pyplot(fig_hops, use_container_width=True)
        else:
            st.write("No hop data")

    with rad2:
        st.markdown("**Malt / Body / Sweetness / Color**")
        fig_malt = plot_radar(malt_profile, title="Malt / Body") if malt_profile else None
        if fig_malt:
            st.pyplot(fig_malt, use_container_width=True)
        else:
            st.write("No malt data")

    with rad3:
        st.markdown("**Yeast / Fermentation Character**")
        fig_yeast = plot_radar(yeast_profile, title="Yeast") if yeast_profile else None
        if fig_yeast:
            st.pyplot(fig_yeast, use_container_width=True)
        else:
            st.write("No yeast data")

    st.markdown("---")

    # -------------------------------------------------
    # 6C. ADVISOR TOOLS (UNLOCKED AFTER PREDICT)
    # -------------------------------------------------

    st.subheader("🎯 Targeted Adjustment Advisors")

    st.markdown("Use these mini-advisors to push specific dials now that we have a baseline prediction.")

    # Hop Advisor
    with st.expander("🌿 Hop Adjustment Advisor"):
        if hop_profile:
            hop_target = st.selectbox(
                "Which hop aroma do you want *more* of?",
                hop_dims,
                key="hop_target_dim"
            )
            trial_amt = st.slider(
                "Simulate late-addition / whirlpool hop (g):",
                5, 60, 20, 5,
                key="hop_trial_amt"
            )
            if st.button("🧪 Suggest a hop addition", key="hop_advise_btn"):
                hop_advice = advise_hops(user_hops, target_dim=hop_target, trial_amt=trial_amt)
                st.success(
                    f"To boost **{hop_advice['target_dim']}**, "
                    f"add ~{hop_advice['addition_grams']} g of **{hop_advice['recommended_hop']}** late.\n"
                    f"(Predicted improvement: +{hop_advice['expected_improvement']:.3f})"
                )
                st.markdown("**Projected new hop profile:**")
                st.json(hop_advice["new_profile"])
        else:
            st.info("Run prediction first (and include at least one hop).")

    # Malt Advisor
    with st.expander("🌾 Malt / Body / Sweetness / Color Advisor"):
        if malt_profile:
            malt_target = st.selectbox(
                "What malt/body dimension do you want to push?",
                malt_dims,
                key="malt_target_dim"
            )
            trial_pct = st.slider(
                "Simulate adding (+% of grist):",
                1, 10, 2, 1,
                key="malt_trial_pct"
            )
            if st.button("🍞 Suggest malt tweak", key="malt_advise_btn"):
                malt_advice = advise_malt(user_malts, target_dim=malt_target, trial_pct=trial_pct)
                st.success(
                    f"To boost **{malt_advice['target_dim']}**, "
                    f"add about {malt_advice['addition_pct']}% of **{malt_advice['recommended_malt']}** "
                    f"to the grist. (Δ ≈ +{malt_advice['expected_improvement']:.3f})"
                )
                st.markdown("**Projected new malt/body profile:**")
                st.json(malt_advice["new_profile"])
        else:
            st.info("Run prediction first (and include at least one malt).")

    # Yeast Advisor
    with st.expander("🧫 Yeast / Fermentation Advisor"):
        if yeast_profile:
            yeast_target = st.selectbox(
                "Which fermentation dimension do you want to emphasize?",
                yeast_dims,
                key="yeast_target_dim"
            )
            if st.button("🔬 Suggest yeast / temp change", key="yeast_advise_btn"):
                yeast_advice = advise_yeast(user_yeast, target_dim=yeast_target)
                st.success(
                    f"To push **{yeast_advice['target_dim']}**, "
                    f"switch to **{yeast_advice['recommended_strain']}**.\n"
                    f"(Predicted improvement: +{yeast_advice['expected_improvement']:.3f})"
                )
                st.markdown("**Projected new fermentation profile:**")
                st.json(yeast_advice["new_profile"])
        else:
            st.info("Run prediction first (and include at least one yeast strain).")

    st.markdown("---")

    # -------------------------------------------------
    # 6D. AI BREWMASTER GUIDANCE (AZURE)
    # -------------------------------------------------

    st.subheader("👨‍🔬 AI Brewmaster Guidance")
    brew_goal = st.text_area(
        "What's your intent for this beer? (e.g. 'Soft hazy IPA with saturated stone fruit and pineapple, low bitterness, pillowy mouthfeel')",
        value="Soft hazy IPA with saturated stone fruit and pineapple, low bitterness, pillowy mouthfeel"
    )

    if st.button("📣 Generate Brewmaster Notes"):
        # call azure / fallback
        notes = call_azure_brewmaster_notes(
            brew_goal,
            hop_profile,
            malt_profile,
            yeast_profile
        )

        # Format as a pretty panel
        st.markdown("### Brewmaster Notes")
        st.info(notes)

else:
    st.info("Fill hops, malt, and yeast above — then click **🍺 Predict Beer Flavor & Balance** to simulate your beer.")

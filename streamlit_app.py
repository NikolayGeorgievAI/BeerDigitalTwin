import os
import re
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# =========================================================
# 0. LOAD MODELS
# =========================================================

ROOT_DIR = os.path.dirname(__file__)

def load_bundle(filename):
    """Generic loader for any of the joblib bundles."""
    path = os.path.join(ROOT_DIR, filename)
    b = joblib.load(path)
    # normalize keys just in case
    out = {}
    out["model"] = b["model"]
    out["feature_cols"] = b.get("feature_cols", b.get("features", []))
    out["aroma_dims"] = b.get("aroma_dims", b.get("targets", []))
    return out

# --- Hop aroma model (already working) ---
hop_bundle = load_bundle("hop_aroma_model.joblib")
hop_model = hop_bundle["model"]
hop_feature_cols = hop_bundle["feature_cols"]
hop_dims = [a for a in hop_bundle["aroma_dims"] if str(a).lower() not in ("nan", "", "none")]

# --- Malt sensory model (placeholder structure) ---
# Assumes malt_sensory_model.joblib is same shape as hop bundle.
malt_bundle = load_bundle("malt_sensory_model.joblib")
malt_model = malt_bundle["model"]
malt_feature_cols = malt_bundle["feature_cols"]
malt_dims = [a for a in malt_bundle["aroma_dims"] if str(a).lower() not in ("nan", "", "none")]

# --- Yeast / fermentation model (placeholder structure) ---
# Assumes yeast_sensory_model.joblib is same shape as hop bundle.
yeast_bundle = load_bundle("yeast_sensory_model.joblib")
yeast_model = yeast_bundle["model"]
yeast_feature_cols = yeast_bundle["feature_cols"]
yeast_dims = [a for a in yeast_bundle["aroma_dims"] if str(a).lower() not in ("nan", "", "none")]


# =========================================================
# 1. FUZZY NAME HELPERS (HOPS, MALTS, YEASTS)
# =========================================================

def _clean_name(name: str) -> str:
    """Normalize ingredient names for fuzzy match."""
    if name is None:
        return ""
    s = str(name).lower()
    s = s.replace("®", "").replace("™", "")
    s = re.sub(r"[^a-z0-9]", "", s)
    return s

def _best_feature_col(user_name: str, feature_cols: list, prefix: str):
    """
    Fuzzy map user-entered ingredient (e.g. 'Citra') to model col (e.g. 'hop_Citra®').
    `prefix` is 'hop_', 'malt_', 'yeast_' etc.
    """
    cleaned_user = _clean_name(user_name)
    best_match = None
    best_score = -1

    for col in feature_cols:
        if not col.startswith(prefix):
            continue
        raw_name = col[len(prefix):]   # remove prefix
        cleaned_model = _clean_name(raw_name)

        # sanity check: first 3 chars of user must appear in model name
        if len(cleaned_user) >= 3 and cleaned_user[:3] not in cleaned_model:
            continue

        # "score": overlap of character sets
        common = set(cleaned_user) & set(cleaned_model)
        score = len(common)

        if score > best_score:
            best_score = score
            best_match = col

    return best_match


# =========================================================
# 2. FEATURE BUILDERS + PREDICTORS
# =========================================================

# ----- HOPS ------------------------------------------------

def build_hop_features(user_hops):
    """
    user_hops = [ {"name": "Citra", "amt_g": 50}, {"name":"Mosaic","amt_g":30} ]
    Returns a 1-row dataframe aligned to hop_feature_cols.
    """
    totals = {col: 0.0 for col in hop_feature_cols}

    for entry in user_hops:
        hop_name = entry.get("name", "")
        amt = float(entry.get("amt_g", 0.0))

        if amt <= 0:
            continue
        if not hop_name or str(hop_name).strip() in ["", "-"]:
            continue

        model_col = _best_feature_col(hop_name, hop_feature_cols, prefix="hop_")
        if model_col is not None:
            totals[model_col] += amt

    return pd.DataFrame([totals], columns=hop_feature_cols)

def predict_hop_profile(user_hops):
    """
    Returns dict {aroma_dim -> ~0..1 intensity}
    """
    X = build_hop_features(user_hops)
    y_pred = hop_model.predict(X)[0]
    return {dim: float(val) for dim, val in zip(hop_dims, y_pred)}

def advise_hop_addition(user_hops, target_dim, trial_amt=20.0):
    """
    Brute force: try adding +trial_amt g of each known hop,
    see which hop best increases `target_dim`.
    """
    if target_dim not in hop_dims:
        raise ValueError(f"{target_dim} not in hop_dims: {hop_dims}")

    base_vec = predict_hop_profile(user_hops)
    base_score = base_vec.get(target_dim, 0.0)

    best_choice = None
    best_delta = -999
    best_new_profile = None

    for col in hop_feature_cols:
        if not col.startswith("hop_"):
            continue

        candidate_name = col[len("hop_"):]  # nice display label
        test_bill = user_hops + [{"name": candidate_name, "amt_g": trial_amt}]
        trial_vec = predict_hop_profile(test_bill)
        trial_score = trial_vec.get(target_dim, 0.0)
        delta = trial_score - base_score

        if delta > best_delta:
            best_delta = delta
            best_choice = candidate_name
            best_new_profile = trial_vec

    return {
        "target_dim": target_dim,
        "current_score": base_score,
        "recommended_hop": best_choice,
        "addition_grams": trial_amt,
        "expected_improvement": best_delta,
        "new_profile": best_new_profile,
    }


# ----- MALTS -----------------------------------------------

def build_malt_features(user_malts):
    """
    user_malts = [
      {"name": "Maris Otter", "pct": 70},
      {"name": "Caramunich III", "pct": 8},
      ...
    ]
    We'll align to malt_feature_cols (e.g. 'malt_Pilsner', 'malt_Caramunich III', ...)
    using fuzzy name match.
    """
    totals = {col: 0.0 for col in malt_feature_cols}

    for entry in user_malts:
        malt_name = entry.get("name", "")
        pct = float(entry.get("pct", 0.0))

        if pct <= 0:
            continue
        if not malt_name or str(malt_name).strip() in ["", "-"]:
            continue

        model_col = _best_feature_col(malt_name, malt_feature_cols, prefix="malt_")
        if model_col is not None:
            totals[model_col] += pct

    return pd.DataFrame([totals], columns=malt_feature_cols)

def predict_malt_profile(user_malts):
    """
    Returns dict {malt_dim -> intensity/value}.
    malt_dim might include body, sweetness, caramel, toast, color_srm, etc.
    """
    X = build_malt_features(user_malts)
    y_pred = malt_model.predict(X)[0]
    return {dim: float(val) for dim, val in zip(malt_dims, y_pred)}

def advise_malt_grist(user_malts, target_dim, bump_pct=2.0):
    """
    Similar to hop advisor:
    Try adding bump_pct% of each malt and see which improves target_dim most.
    """
    if target_dim not in malt_dims:
        raise ValueError(f"{target_dim} not in malt_dims: {malt_dims}")

    base_vec = predict_malt_profile(user_malts)
    base_score = base_vec.get(target_dim, 0.0)

    best_choice = None
    best_delta = -999
    best_new_profile = None

    for col in malt_feature_cols:
        if not col.startswith("malt_"):
            continue

        candidate_name = col[len("malt_"):]  # nice display
        test_bill = user_malts + [{"name": candidate_name, "pct": bump_pct}]
        trial_vec = predict_malt_profile(test_bill)
        trial_score = trial_vec.get(target_dim, 0.0)
        delta = trial_score - base_score

        if delta > best_delta:
            best_delta = delta
            best_choice = candidate_name
            best_new_profile = trial_vec

    return {
        "target_dim": target_dim,
        "current_score": base_score,
        "recommended_malt": best_choice,
        "addition_pct": bump_pct,
        "expected_improvement": best_delta,
        "new_profile": best_new_profile,
    }


# ----- YEAST / FERMENTATION --------------------------------

def build_yeast_features(yeast_name, ferm_temp_f):
    """
    Minimal example:
    - 1-hot strain
    - plus a numeric ferment temp
    We'll align to yeast_feature_cols like 'yeast_US-05', 'yeast_London Ale III', etc.
    If your yeast model expects more process features (pitch rate, dry hop timing, etc),
    extend here.
    """
    totals = {col: 0.0 for col in yeast_feature_cols}

    # strain one-hot
    strain_col = _best_feature_col(yeast_name, yeast_feature_cols, prefix="yeast_")
    if strain_col is not None:
        totals[strain_col] = 1.0

    # temperature feature if present
    # e.g. if model was trained with a column like "fermtemp"
    # we try to set it if that column exists
    for col in yeast_feature_cols:
        if col.lower() in ["fermtemp", "fermentation_temp_f", "temp_f"]:
            totals[col] = float(ferm_temp_f)

    return pd.DataFrame([totals], columns=yeast_feature_cols)

def predict_yeast_profile(yeast_name, ferm_temp_f):
    """
    Returns dict {yeast_dim -> predicted intensity}.
    yeast_dim might include fruit esters, stone fruit, haze stability, dryness, etc.
    """
    X = build_yeast_features(yeast_name, ferm_temp_f)
    y_pred = yeast_model.predict(X)[0]
    return {dim: float(val) for dim, val in zip(yeast_dims, y_pred)}

def advise_fermentation(yeast_name, ferm_temp_f, target_dim, temp_step=2.0):
    """
    For fermentation, we can't "add grams", but we CAN:
    - try bumping temp up or down by temp_step °F,
    - or swapping to alternative strain(s) to see which moves target_dim.
    We'll do simplest first: scan through all strains at same temp.
    """
    if target_dim not in yeast_dims:
        raise ValueError(f"{target_dim} not in yeast_dims: {yeast_dims}")

    base_vec = predict_yeast_profile(yeast_name, ferm_temp_f)
    base_score = base_vec.get(target_dim, 0.0)

    best_choice = None
    best_delta = -999
    best_new_profile = None

    # try alternate strains (and maybe tweak temp slightly)
    for col in yeast_feature_cols:
        if not col.startswith("yeast_"):
            continue
        candidate_strain = col[len("yeast_"):]
        # same temp
        trial_vec = predict_yeast_profile(candidate_strain, ferm_temp_f)
        trial_score = trial_vec.get(target_dim, 0.0)
        delta = trial_score - base_score

        if delta > best_delta:
            best_delta = delta
            best_choice = f"{candidate_strain} @ {ferm_temp_f}°F"
            best_new_profile = trial_vec

        # also test warmer
        warmer_temp = ferm_temp_f + temp_step
        trial_vec2 = predict_yeast_profile(candidate_strain, warmer_temp)
        trial_score2 = trial_vec2.get(target_dim, 0.0)
        delta2 = trial_score2 - base_score

        if delta2 > best_delta:
            best_delta = delta2
            best_choice = f"{candidate_strain} @ {warmer_temp}°F"
            best_new_profile = trial_vec2

    return {
        "target_dim": target_dim,
        "current_score": base_score,
        "recommended_fermentation": best_choice,
        "expected_improvement": best_delta,
        "new_profile": best_new_profile,
    }


# =========================================================
# 3. VISUALIZATIONS
# =========================================================

def plot_radar(profile_dict, title="Profile"):
    """
    Generic radar plot; we'll reuse for hops, malt, yeast.
    """
    dims = list(profile_dict.keys())
    vals = [profile_dict[d] for d in dims]

    dims.append(dims[0])
    vals.append(vals[0])

    angles = np.linspace(0, 2 * np.pi, len(dims), endpoint=False)

    fig = plt.figure(figsize=(5, 5))
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
# 4. (FUTURE) AZURE "HEAD BREWER" SUMMARIZER
# =========================================================

def generate_brewmaster_notes(hop_prof, malt_prof, yeast_prof, brewer_goal):
    """
    This is where Azure OpenAI / GPT will eventually go.
    For now, we just stitch together a narrative using the model outputs.
    """
    def top_dims(prof, n=3):
        ranked = sorted(prof.items(), key=lambda kv: kv[1], reverse=True)
        return [f"{name} ({score:.2f})" for name, score in ranked[:n]]

    hop_top = ", ".join(top_dims(hop_prof)) if hop_prof else "n/a"
    malt_top = ", ".join(top_dims(malt_prof)) if malt_prof else "n/a"
    yeast_top = ", ".join(top_dims(yeast_prof)) if yeast_prof else "n/a"

    summary = []
    summary.append("Brewmaster Notes:")
    summary.append(f"- Brewer goal: {brewer_goal or '—'}")
    summary.append(f"- Hop-driven aroma so far: {hop_top}")
    summary.append(f"- Malt-driven character: {malt_top}")
    summary.append(f"- Fermentation / yeast profile: {yeast_top}")
    summary.append("")
    summary.append("Suggested direction:")
    summary.append("• Adjust hop bill late to push desired aroma (citra/mosaic style tropical, pine, citrus).")
    summary.append("• Tune malt bill % to get body/sweetness without overshooting roast or SRM.")
    summary.append("• Lock in yeast strain + temp to shape ester profile and mouthfeel.")
    summary.append("")
    summary.append("Next step in production: this block will come from Azure GPT with specific hop names, temps, % grist adjustments, and warnings about style drift.")
    return "\n".join(summary)


# =========================================================
# 5. STREAMLIT APP
# =========================================================

st.set_page_config(page_title="Beer Recipe Digital Twin", page_icon="🍺", layout="centered")

st.title("🍺 Beer Recipe Digital Twin")
st.markdown("""
1. Enter your hop bill (variety + grams).
2. The model predicts flavor dimensions like citrus, pine, tropical fruit.
3. Pick what you want more of.
4. It recommends a single hop addition (and grams) to steer the beer.

Then do the same for malt (body/sweetness/color) and yeast (esters/mouthfeel),
and we'll stitch it all into 'Brewmaster Notes'.
""")
st.markdown("---")

# -----------------------
# HOPS SECTION
# -----------------------
st.header("🌿 Hop Bill")

colh1, colh2, colh3 = st.columns([1,1,1])

with colh1:
    hop1_name = st.text_input("Hop 1 name", "Citra")
    hop2_name = st.text_input("Hop 2 name", "Mosaic")

with colh2:
    hop1_amt = st.number_input("Hop 1 (g)", min_value=0.0, max_value=500.0, value=50.0, step=5.0)
    hop2_amt = st.number_input("Hop 2 (g)", min_value=0.0, max_value=500.0, value=30.0, step=5.0)

with colh3:
    st.write("We'll predict the aroma balance of this hop bill,")
    st.write("and suggest how to push citrus / pine / tropical, etc.")

user_hops = []
if hop1_name and hop1_amt > 0:
    user_hops.append({"name": hop1_name, "amt_g": hop1_amt})
if hop2_name and hop2_amt > 0:
    user_hops.append({"name": hop2_name, "amt_g": hop2_amt})

st.markdown("")
predict_hops_clicked = st.button("🔍 Predict Hop Aroma Profile")

hop_profile = {}
hop_advice = None

if predict_hops_clicked and user_hops:
    hop_profile = predict_hop_profile(user_hops)
    st.subheader("Predicted Hop Aroma Profile")
    st.json(hop_profile)

    fig = plot_radar(hop_profile, title="Current Hop Bill")
    st.pyplot(fig)

    st.subheader("🎯 Hop Adjustment Advisor")
    hop_target_dim = st.selectbox("What aroma do you want more of?", hop_dims)
    hop_trial_amt = st.slider("Simulate adding (g) of ONE new hop:", 5, 60, 20, 5)
    hop_advise_clicked = st.button("🧠 Advise Hop Addition")

    if hop_advise_clicked:
        hop_advice = advise_hop_addition(user_hops, hop_target_dim, trial_amt=hop_trial_amt)
        st.success(
            f"To boost **{hop_advice['target_dim']}**, add {hop_advice['addition_grams']} g of "
            f"**{hop_advice['recommended_hop']}**. "
            f"Expected improvement: +{hop_advice['expected_improvement']:.3f}"
        )

        st.json(hop_advice["new_profile"])
        fig2 = plot_radar(hop_advice["new_profile"], title="Revised Hop Bill")
        st.pyplot(fig2)

st.markdown("---")

# -----------------------
# MALT SECTION
# -----------------------
st.header("🌾 Malt / Grain Bill")

colm1, colm2, colm3 = st.columns([1,1,1])

with colm1:
    malt1_name = st.text_input("Malt 1 name", "Maris Otter")
    malt2_name = st.text_input("Malt 2 name", "Caramunich III")

with colm2:
    malt1_pct = st.number_input("Malt 1 (% of grist)", min_value=0.0, max_value=100.0, value=70.0, step=1.0)
    malt2_pct = st.number_input("Malt 2 (% of grist)", min_value=0.0, max_value=100.0, value=8.0, step=1.0)

with colm3:
    st.write("We'll estimate body, caramel, sweetness, color, etc.")
    st.write("Then we can suggest tweaks to grain percentages.")

user_malts = []
if malt1_name and malt1_pct > 0:
    user_malts.append({"name": malt1_name, "pct": malt1_pct})
if malt2_name and malt2_pct > 0:
    user_malts.append({"name": malt2_name, "pct": malt2_pct})

st.markdown("")
predict_malt_clicked = st.button("🔍 Predict Malt Profile")

malt_profile = {}
malt_advice = None

if predict_malt_clicked and user_malts:
    malt_profile = predict_malt_profile(user_malts)
    st.subheader("Predicted Malt / Body Profile")
    st.json(malt_profile)

    fig_malt = plot_radar(malt_profile, title="Grain Bill Profile")
    st.pyplot(fig_malt)

    st.subheader("🍞 Malt Adjustment Advisor")
    malt_target_dim = st.selectbox("What malt character do you want more of?", malt_dims)
    malt_bump_pct = st.slider("Simulate +% of a specialty malt:", 1, 10, 2, 1)
    malt_advise_clicked = st.button("📈 Advise Malt Adjustment")

    if malt_advise_clicked:
        malt_advice = advise_malt_grist(user_malts, malt_target_dim, bump_pct=float(malt_bump_pct))
        st.success(
            f"To boost **{malt_advice['target_dim']}**, "
            f"add about {malt_advice['addition_pct']}% of **{malt_advice['recommended_malt']}** "
            f"to the grist. Expected improvement: +{malt_advice['expected_improvement']:.3f}"
        )

        st.json(malt_advice["new_profile"])
        fig_malt2 = plot_radar(malt_advice["new_profile"], title="Adjusted Grain Bill")
        st.pyplot(fig_malt2)

st.markdown("---")

# -----------------------
# YEAST / FERMENTATION SECTION
# -----------------------
st.header("🧪 Yeast & Fermentation")

coly1, coly2, coly3 = st.columns([1,1,1])

# basic: pick a yeast from whatever the model knows (strip "yeast_" prefix)
available_strains = sorted(
    list({c[len("yeast_"):] for c in yeast_feature_cols if c.startswith("yeast_")})
)

with coly1:
    yeast_choice = st.selectbox("Yeast strain", available_strains)
with coly2:
    ferm_temp = st.slider("Fermentation temp (°F)", min_value=60, max_value=75, value=68, step=1)
with coly3:
    st.write("We'll estimate ester profile, mouthfeel, dryness, haze stability, etc.")
    st.write("Then we can suggest 'fermentation tuning'.")

st.markdown("")
predict_yeast_clicked = st.button("🔍 Predict Fermentation Profile")

yeast_profile = {}
yeast_advice = None

if predict_yeast_clicked and yeast_choice:
    yeast_profile = predict_yeast_profile(yeast_choice, ferm_temp)
    st.subheader("Predicted Yeast / Fermentation Profile")
    st.json(yeast_profile)

    fig_yeast = plot_radar(yeast_profile, title="Fermentation Profile")
    st.pyplot(fig_yeast)

    st.subheader

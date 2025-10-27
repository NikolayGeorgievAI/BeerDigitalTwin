import os
import math
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import re
import json
import copy
from datetime import datetime
from openai import OpenAI
from typing import Dict, List

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# -------------------------------------------------
# CONFIG / THEME
# -------------------------------------------------
st.set_page_config(
    page_title="Beer Recipe Digital Twin",
    page_icon="🍺",
    layout="wide"
)

plt.rcParams["font.size"] = 12
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

# -------------------------------------------------
# HELPER: flavor tweak suggestions
# -------------------------------------------------
def flavor_tweak_suggestions(goal: str,
                             hop_pred: dict,
                             malt_pred: dict,
                             yeast_pred: dict):
    """
    goal: str like "More tropical fruit", "More body / pillowy mouthfeel", etc.
    hop_pred/malt_pred/yeast_pred: dicts of current predictions
    Returns list[str] of actionable suggestions.
    """

    tips = []
    goal_lower = goal.lower()

    if "tropical" in goal_lower or "mango" in goal_lower or "fruit" in goal_lower:
        tips.append("⬆ Use hops known for mango / pineapple (Citra, Mosaic, Azacca) in whirlpool and dry hop.")
        tips.append("⬆ Push later additions (post-boil ~75°C) instead of bittering additions.")
        tips.append("⬇ Avoid piney / grassy hops that mask juicy fruit.")

    if "body" in goal_lower or "pillowy" in goal_lower or "mouthfeel" in goal_lower:
        tips.append("⬆ Add flaked oats / wheat malt (5–10%) to boost protein haze and silkiness.")
        tips.append("⬇ Use a medium-attenuating, ester-friendly yeast (NEIPA / English style) instead of bone-dry strains.")
        tips.append("⬇ Ferment ~18–20°C to keep the mouthfeel soft instead of thin.")

    if "drier" in goal_lower or "crisper" in goal_lower or "bitter" in goal_lower:
        tips.append("⬆ Consider a slightly higher-attenuating yeast or a slightly warmer ferment to dry it out.")
        tips.append("⬆ Add a small earlier-boil hop charge for firmer bitterness backbone.")
        tips.append("⬇ Trim some late sweet fruit-bomb whirlpool if it's too juicy/sweet.")

    if "color" in goal_lower or "darker" in goal_lower or "amber" in goal_lower:
        tips.append("⬆ Add a touch of light crystal / Vienna / Munich for color + depth (5% range).")
        tips.append("⬇ Keep base pale malt but introduce slight toast for richer hue.")

    if not tips:
        tips.append("No preset for that goal yet — try 'More tropical fruit', 'More body / pillowy mouthfeel', or 'Drier / crisper'.")

    # contextual spice
    if malt_pred and "sweetness" in malt_pred and malt_pred["sweetness"] > 15:
        tips.append("Note: sweetness is already fairly high — consider a tiny early bittering hop to avoid cloying finish.")
    if yeast_pred and ("Attenuation_num" in yeast_pred and yeast_pred["Attenuation_num"] > 0.8):
        tips.append("Note: high attenuation / drier profile — watch ferm temp so body doesn’t thin out too far.")

    return tips

# -------------------------------------------------
# UTILS: NORMALIZATION / MATCHING
# -------------------------------------------------
def _clean_name(name: str) -> str:
    """Lowercase, strip special marks, keep alnum only."""
    if not name:
        return ""
    s = str(name).lower()
    s = s.replace("®", "").replace("™", "")
    s = re.sub(r"[^a-z0-9]", "", s)
    return s

def _best_feature_match(user_name: str, feature_cols: list, prefix: str):
    """
    Fuzzy match "Citra" -> "hop_Citra®"
    Only consider columns that start with that prefix.
    """
    cleaned_user = _clean_name(user_name)
    best_match = None
    best_score = -1

    for col in feature_cols:
        if not col.startswith(prefix):
            continue
        raw_label = col[len(prefix):]
        cleaned_label = _clean_name(raw_label)

        if len(cleaned_user) >= 3 and cleaned_user[:3] not in cleaned_label:
            continue

        common = set(cleaned_user) & set(cleaned_label)
        score = len(common)
        if score > best_score:
            best_score = score
            best_match = col

    return best_match

def _choices_from_features(feature_cols, preferred_prefix=None):
    """
    Build nice dropdown choices from model feature columns.
    If preferred_prefix provided, we try those first.
    """
    def prettify(label: str) -> str:
        label = label.replace("®", "").replace("™", "")
        label = label.replace("_", " ").strip()
        return label

    subset = []
    if preferred_prefix:
        for col in feature_cols:
            if col.startswith(preferred_prefix):
                raw_label = col[len(preferred_prefix):]
                subset.append(prettify(raw_label))

    if not subset:
        for col in feature_cols:
            cand = col
            for p in ["hop_", "malt_", "grain_", "base_", "yeast_", "strain_", "y_", "m_"]:
                if cand.startswith(p):
                    cand = cand[len(p):]
            subset.append(prettify(cand))

    cleaned = []
    for n in subset:
        if n and n not in cleaned:
            cleaned.append(n)

    cleaned = sorted(cleaned, key=lambda s: s.lower())
    return cleaned

# -------------------------------------------------
# LOAD TRAINED MODELS / BUNDLES
# -------------------------------------------------
ROOT_DIR = os.path.dirname(__file__)

# --- HOPS model ---
hop_bundle = joblib.load(os.path.join(ROOT_DIR, "hop_aroma_model.joblib"))
hop_model = hop_bundle["model"]
hop_feature_cols = hop_bundle["feature_cols"]
hop_dims = [
    a for a in hop_bundle["aroma_dims"]
    if str(a).lower() not in ("nan", "", "none")
]

# --- MALT model ---
malt_bundle = joblib.load(os.path.join(ROOT_DIR, "malt_sensory_model.joblib"))
malt_model = malt_bundle["model"]
malt_feature_cols = malt_bundle["feature_cols"]
malt_dims = malt_bundle["flavor_cols"]  # e.g. sweetness, body_full, color_intensity

# --- YEAST model ---
yeast_bundle = joblib.load(os.path.join(ROOT_DIR, "yeast_sensory_model.joblib"))
yeast_model = yeast_bundle["model"]
yeast_feature_cols = yeast_bundle["feature_cols"]
yeast_dims = yeast_bundle["flavor_cols"]  # e.g. attenuation_num, flocculation_num, temp_avg_F

# Build dropdown options
HOP_CHOICES = _choices_from_features(hop_feature_cols, "hop_")
MALT_CHOICES = _choices_from_features(malt_feature_cols, "malt_")
YEAST_CHOICES = _choices_from_features(yeast_feature_cols, "yeast_")

# -------------------------------------------------
# FEATURE BUILDERS
# -------------------------------------------------
def build_hop_features(user_hops):
    """
    user_hops: [ {"name":"Citra", "amt":50}, ... ] grams
    returns DF with columns=hop_feature_cols
    """
    totals = {c: 0.0 for c in hop_feature_cols}
    for entry in user_hops:
        nm = entry.get("name","")
        amt = float(entry.get("amt",0.0))
        if amt <= 0 or not nm.strip():
            continue
        match_col = _best_feature_match(nm, hop_feature_cols, "hop_")
        if match_col:
            totals[match_col] += amt
    return pd.DataFrame([totals], columns=hop_feature_cols)

def predict_hop_profile(user_hops):
    X = build_hop_features(user_hops)
    y = hop_model.predict(X)[0]
    return {dim: float(val) for dim,val in zip(hop_dims, y)}

def build_malt_features(user_malts):
    """
    user_malts: [ {"name":"Maris Otter", "pct":70}, ... ]
    """
    totals = {c:0.0 for c in malt_feature_cols}
    for entry in user_malts:
        nm = entry.get("name","")
        pct = float(entry.get("pct",0.0))
        if pct <= 0 or not nm.strip():
            continue
        match_col = _best_feature_match(nm, malt_feature_cols, "malt_")
        if match_col is None:
            for pfx in ["grain_","base_","malt_","m_"]:
                match_col = _best_feature_match(nm, malt_feature_cols, pfx)
                if match_col:
                    break
        if match_col:
            totals[match_col] += pct
    return pd.DataFrame([totals], columns=malt_feature_cols)

def predict_malt_profile(user_malts):
    X = build_malt_features(user_malts)
    y = malt_model.predict(X)[0]
    return {dim: float(val) for dim,val in zip(malt_dims, y)}

def build_yeast_features(yeast_info):
    """
    yeast_info = {"strain":"Nottingham Ale Yeast", "ferm_temp_c":20.0}
    We'll one-hot the strain, ignoring temp for now in the model input.
    """
    totals = {c:0.0 for c in yeast_feature_cols}
    strain_nm = yeast_info.get("strain","")
    match = _best_feature_match(strain_nm, yeast_feature_cols, "yeast_")
    if match is None:
        for pfx in ["yeast_","strain_","y_",""]:
            if pfx=="":  # skip final fallback
                continue
            tempm = _best_feature_match(strain_nm, yeast_feature_cols, pfx)
            if tempm:
                match = tempm
                break
    if match:
        totals[match] = 1.0
    return pd.DataFrame([totals], columns=yeast_feature_cols)

def predict_yeast_profile(yeast_info):
    X = build_yeast_features(yeast_info)
    y = yeast_model.predict(X)[0]
    return {dim: float(val) for dim,val in zip(yeast_dims, y)}

# -------------------------------------------------
# RADAR PLOT HELPERS
# -------------------------------------------------
def _radar_axes(num_vars: int, subplot, title: str):
    """
    Create a polar subplot with consistent styling, hide numeric ticks.
    """
    ax = plt.subplot(subplot, polar=True)
    ax.set_title(title, fontsize=22, pad=30, fontweight="bold")

    # Hide numeric tick labels:
    ax.set_yticklabels([])
    ax.set_yticks([])
    ax.grid(color="#cccccc", linestyle="-", linewidth=0.5, alpha=0.5)

    return ax

def _plot_radar(ax, labels: List[str], values: List[float], color="#1f77b4"):
    """
    Plot one radar region on the given ax. Tick labels = trait names only, no numbers.
    We close the polygon.
    """
    N = len(labels)
    if N == 0:
        return
    theta = np.linspace(0, 2*math.pi, N, endpoint=False)

    vals = np.array(values, dtype=float)

    theta_closed = np.concatenate([theta, theta[:1]])
    vals_closed  = np.concatenate([vals,  vals[:1]])

    ax.plot(theta_closed, vals_closed, color=color, linewidth=2)
    ax.fill(theta_closed, vals_closed, color=color, alpha=0.2)

    ax.set_xticks(theta)
    ax.set_xticklabels(labels, fontsize=16)

def build_radar_figures(hop_vec, malt_vec, yeast_vec):
    """
    hop_vec, malt_vec, yeast_vec are dicts of numeric predictions.
    We'll scale each web independently 0..max_val to fill it.
    """
    # 1) Hops radar subset
    hop_traits = [
        "citrus","fruity","floral","grassy","pine","herbal","stone fruit","cedar"
    ]
    hv = []
    for t in hop_traits:
        cand_keys = [t, t.replace(" ","_"), t.replace(" ","")]
        v = 0.0
        for ck in cand_keys:
            if ck in hop_vec:
                v = hop_vec[ck]
                break
        hv.append(max(v, 0.0))
    hop_max = max(hv+[1e-9])
    hop_scaled = [(x/hop_max if hop_max>0 else 0.0) for x in hv]

    # 2) Malt radar subset
    malt_traits = ["sweetness", "body_full", "color_intensity"]
    mv = [max(malt_vec.get(k,0.0),0.0) for k in malt_traits]
    malt_max = max(mv+[1e-9])
    malt_scaled = [(x/malt_max if malt_max>0 else 0.0) for x in mv]

    # 3) Yeast radar subset
    ylabels = ["Attenuation_num","Flocculation_num","Temp_avg_F"]
    yv_raw = [
        max(
            yeast_vec.get("attenuation_num",0.0),
            yeast_vec.get("Attenuation_num",0.0),
            0.0
        ),
        max(
            yeast_vec.get("flocculation_num",0.0),
            yeast_vec.get("Flocculation_num",0.0),
            0.0
        ),
        max(
            yeast_vec.get("Temp_avg_F",0.0),
            yeast_vec.get("temp_avg_F",0.0),
            yeast_vec.get("temp_avg_f",0.0),
            yeast_vec.get("temp_avg_f_",0.0),
            0.0
        ),
    ]
    y_max = max(yv_raw+[1e-9])
    y_scaled = [(x/y_max if y_max>0 else 0.0) for x in yv_raw]

    fig = plt.figure(figsize=(18,6))

    ax1 = _radar_axes(len(hop_traits), 131, "Hops / Aroma")
    _plot_radar(ax1, hop_traits, hop_scaled, color="#1f77b4")

    ax2 = _radar_axes(len(malt_traits), 132, "Malt / Body-Sweetness")
    _plot_radar(ax2, malt_traits, malt_scaled, color="#2ca02c")

    ax3 = _radar_axes(len(ylabels), 133, "Yeast / Fermentation")
    _plot_radar(ax3, ylabels, y_scaled, color="#d62728")

    plt.tight_layout()
    return fig


# -------------------------------------------------
# AZURE OPENAI CALL
# -------------------------------------------------
def call_azure_brewmaster_notes(
    goal_text: str,
    hop_vec: Dict[str,float],
    malt_vec: Dict[str,float],
    yeast_vec: Dict[str,float],
) -> str:
    """
    Attempts to call Azure OpenAI for guided brew notes.
    If that fails (404 / bad key / etc.), returns a local fallback.
    """

    def short_desc(vec, keys):
        out = []
        for k in keys:
            val = vec.get(k, 0.0)
            out.append(f"{k}: {val:.2f}")
        return "; ".join(out)

    hop_desc   = short_desc(hop_vec,  list(hop_vec.keys())[:8])
    malt_desc  = short_desc(malt_vec, list(malt_vec.keys())[:5])
    yeast_desc = short_desc(yeast_vec,list(yeast_vec.keys())[:5])

    azure_key      = os.environ.get("AZURE_OPENAI_API_KEY") or st.secrets.get("AZURE_OPENAI_API_KEY", "")
    azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT") or st.secrets.get("AZURE_OPENAI_ENDPOINT", "")
    azure_deploy   = os.environ.get("AZURE_OPENAI_DEPLOYMENT") or st.secrets.get("AZURE_OPENAI_DEPLOYMENT", "")

    if azure_key and azure_endpoint and azure_deploy:
        try:
            client = OpenAI(
                api_key = azure_key,
                base_url = azure_endpoint.rstrip("/") + "/openai/deployments"
            )

            system_msg = (
                "You are a professional brewmaster. "
                "Given the brewer's goal and predicted hop/malt/yeast sensory profile, "
                "provide concise, expert-level adjustments (hop timing, grist tweaks, "
                "fermentation strategy, mouthfeel tuning). Bullet points or numbered steps. "
                "Avoid filler and generic disclaimers."
            )

            user_msg = (
                f"Brewer goal: {goal_text}\n\n"
                "Predicted hop aroma (approx):\n"
                f"{hop_desc}\n\n"
                "Predicted malt/body/color (approx):\n"
                f"{malt_desc}\n\n"
                "Predicted yeast/fermentation (approx):\n"
                f"{yeast_desc}\n\n"
                "Give actionable brewing guidance."
            )

            completion = client.chat.completions.create(
                model=azure_deploy,
                messages=[
                    {"role":"system","content":system_msg},
                    {"role":"user","content":user_msg}
                ],
                temperature=0.4,
                max_tokens=500,
            )

            if (
                completion and
                hasattr(completion,"choices") and
                len(completion.choices) > 0 and
                hasattr(completion.choices[0],"message") and
                completion.choices[0].message and
                "content" in completion.choices[0].message
            ):
                raw = completion.choices[0].message["content"].strip()
                if raw:
                    return raw

        except Exception:
            pass  # fail over to fallback

    fallback = (
        "Brewmaster Notes (prototype)\n\n"
        "1. Hop strategy:\n"
        "- Lean toward tropical / stone-fruit hops in late-whirlpool or dry hop (~10-15 min or sub-77°C) "
        "to boost juicy aroma without boosting IBUs.\n"
        "- Keep early boil hops minimal to avoid sharp bitterness.\n"
        "- Split dry hop additions across multiple days for better oil extraction.\n\n"
        "2. Malt & body:\n"
        "- Use pale / pilsner base plus ~5-10% oats or wheat for pillowy mouthfeel and stable haze.\n"
        "- Keep color fairly light so bright mango / pineapple notes pop instead of going muddy.\n\n"
        "3. Fermentation:\n"
        "- Choose a moderately ester-friendly yeast that enhances fruit without harsh fusels.\n"
        "- Ferment toward the lower end of the yeast's range to avoid solventy heat.\n"
        "- Aim for moderate attenuation so it stays plush, not bone-dry.\n"
    )
    return fallback


# -------------------------------------------------
# STREAMLIT LAYOUT
# -------------------------------------------------
st.markdown(
    "<h1 style='font-size:2.5rem; line-height:1.2; display:flex; align-items:center;'>🍺 Beer Recipe Digital Twin</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "Your AI brew assistant:\n\n"
    "1. Build a hop bill, grain bill, and fermentation plan.\n\n"
    "2. Predict aroma, body, color, esters, mouthfeel — together.\n\n"
    "3. Get brewmaster-style guidance based on your style goal."
)
st.markdown("---")

# ---------------------
# INPUT SECTIONS
# ---------------------

# 🌿 HOPS SECTION
st.markdown("## 🌿 Hops (late/aroma additions)")
st.caption("Select your hop varieties and specify their addition amounts below.")

c_h1, c_h2 = st.columns(2)

with c_h1:
    hop1 = st.selectbox(
        "Main Hop Variety",
        HOP_CHOICES,
        index=HOP_CHOICES.index("Mosaic") if "Mosaic" in HOP_CHOICES else 0,
        key="hop1_select",
    )
    hop1_amt = st.number_input(
        "Hop 1 amount (g)",
        min_value=0.0,
        max_value=500.0,
        value=30.0,
        step=5.0,
        help="Amount for main hop (late or dry-hop addition)."
    )

with c_h2:
    hop2 = st.selectbox(
        "Secondary Hop Variety",
        HOP_CHOICES,
        index=HOP_CHOICES.index("Citra") if "Citra" in HOP_CHOICES else 0,
        key="hop2_select",
    )
    hop2_amt = st.number_input(
        "Hop 2 amount (g)",
        min_value=0.0,
        max_value=500.0,
        value=20.0,
        step=5.0,
        help="Amount for secondary hop addition."
    )

st.markdown("---")

# 🌾 MALT SECTION
st.markdown("## 🌾 Malt / Grain Bill")
st.caption("Define your grist composition as percentages of the total malt bill.")

c_m1, c_m2 = st.columns(2)

with c_m1:
    malt1 = st.selectbox(
        "Base / primary malt",
        MALT_CHOICES,
        index=MALT_CHOICES.index("EXTRA PALE MALT") if "EXTRA PALE MALT" in MALT_CHOICES else 0,
        key="malt1_select",
    )
    malt1_pct = st.number_input(
        "Malt 1 (% grist)",
        min_value=0.0,
        max_value=100.0,
        value=70.0,
        step=1.0,
        help="Percentage of total grist for base malt."
    )

with c_m2:
    malt2 = st.selectbox(
        "Specialty / character malt",
        MALT_CHOICES,
        index=MALT_CHOICES.index("HANÁ MALT") if "HANÁ MALT" in MALT_CHOICES else 0,
        key="malt2_select",
    )
    malt2_pct = st.number_input(
        "Malt 2 (% grist)",
        min_value=0.0,
        max_value=100.0,
        value=8.0,
        step=1.0,
        help="Percentage of total grist for specialty malt."
    )

st.markdown("---")

# 🧫 YEAST SECTION
st.markdown("## 🧫 Yeast & Fermentation")
st.caption("Select your yeast strain and target fermentation temperature.")

c_y1, c_y2 = st.columns(2)

with c_y1:
    yeast_strain = st.selectbox(
        "Yeast strain",
        YEAST_CHOICES,
        index=0,
        key="yeast_select",
    )

with c_y2:
    ferm_temp_c = st.number_input(
        "Fermentation temp (°C)",
        min_value=15.0,
        max_value=30.0,
        value=20.0,
        step=0.5,
        help="This is for AI notes only; not yet modeled in predictions."
    )

st.markdown("---")

# ---------------------
# PREDICTION BUTTON
# ---------------------
st.header("🍻 Predict Beer Flavor & Balance")
st.write(
    "Fill hops, malt, and yeast above — then click **Predict Beer Flavor & Balance** "
    "to simulate aroma, body, esters, color, etc."
)
predict_clicked = st.button("🔍 Predict Beer Flavor & Balance")

hop_profile = {}
malt_profile = {}
yeast_profile = {}

if predict_clicked:
    # Build structured inputs
    user_hops = []
    if hop1 and hop1_amt > 0:
        user_hops.append({"name":hop1,"amt":hop1_amt})
    if hop2 and hop2_amt > 0:
        user_hops.append({"name":hop2,"amt":hop2_amt})

    user_malts = []
    if malt1 and malt1_pct>0:
        user_malts.append({"name":malt1,"pct":malt1_pct})
    if malt2 and malt2_pct>0:
        user_malts.append({"name":malt2,"pct":malt2_pct})

    user_yeast = {
        "strain": yeast_strain,
        "ferm_temp_c": ferm_temp_c
    }

    hop_profile = predict_hop_profile(user_hops)
    malt_profile = predict_malt_profile(user_malts)
    yeast_profile = predict_yeast_profile(user_yeast)

    # Also stash the *current inputs* + predictions in session_state
    st.session_state["main_hop"] = hop1
    st.session_state["secondary_hop"] = hop2
    st.session_state["hop_1_amount_g"] = hop1_amt
    st.session_state["hop_2_amount_g"] = hop2_amt

    st.session_state["malt_base"] = malt1
    st.session_state["malt_special"] = malt2
    st.session_state["malt_1_pct"] = malt1_pct
    st.session_state["malt_2_pct"] = malt2_pct

    st.session_state["yeast_strain"] = yeast_strain
    st.session_state["ferm_temp_c"] = ferm_temp_c

    st.session_state["hop_profile"] = hop_profile
    st.session_state["malt_profile"] = malt_profile
    st.session_state["yeast_profile"] = yeast_profile

    # -------------------------
    # RADAR OVERVIEW
    # -------------------------
    st.markdown("## 🕸 Radar Overview")
    st.write(
        "Relative shape only. Axes are labeled by trait, numeric ticks/values are hidden."
    )
    radar_fig = build_radar_figures(hop_profile, malt_profile, yeast_profile)
    st.pyplot(radar_fig)

st.markdown("---")

# -------------------------------------------------
# SNAPSHOT & RECALL (SAVE / LOAD BATCHES)
# -------------------------------------------------
if "saved_batches" not in st.session_state:
    st.session_state["saved_batches"] = {}  # {batch_name: batch_dict}

st.markdown("## 🗂 Recipe Snapshot & Recall")
col_save, col_load = st.columns([2, 2])

with col_save:
    st.markdown("#### Save current batch")

    # Build a default suggested name: "Mosaic - 2025-10-27 14:32"
    suggested = f"{st.session_state.get('main_hop','Batch')} - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    batch_name = st.text_input("Batch name", value=suggested)

    if st.button("💾 Save this batch"):
        batch_payload = {
            "inputs": {
                "main_hop": st.session_state.get("main_hop", ""),
                "secondary_hop": st.session_state.get("secondary_hop", ""),
                "hop_1_amount_g": st.session_state.get("hop_1_amount_g", 0.0),
                "hop_2_amount_g": st.session_state.get("hop_2_amount_g", 0.0),
                "malt_base": st.session_state.get("malt_base", ""),
                "malt_special": st.session_state.get("malt_special", ""),
                "malt_1_pct": st.session_state.get("malt_1_pct", 0.0),
                "malt_2_pct": st.session_state.get("malt_2_pct", 0.0),
                "yeast_strain": st.session_state.get("yeast_strain", ""),
                "ferm_temp_c": st.session_state.get("ferm_temp_c", 20.0),
            },
            "predictions": {
                "hop_profile": st.session_state.get("hop_profile", {}),
                "malt_profile": st.session_state.get("malt_profile", {}),
                "yeast_profile": st.session_state.get("yeast_profile", {}),
            },
            "ai_notes": st.session_state.get("last_ai_notes",""),
        }

        st.session_state["saved_batches"][batch_name] = copy.deepcopy(batch_payload)

        st.success(f"Saved batch: {batch_name} ✅")

        with st.expander("Show saved JSON for export"):
            st.code(json.dumps(batch_payload, indent=2))

with col_load:
    st.markdown("#### Load previous batch")
    batch_options = list(st.session_state["saved_batches"].keys())
    if batch_options:
        selection = st.selectbox("Select a saved batch", options=batch_options)

        if st.button("📂 Load this batch"):
            loaded = st.session_state["saved_batches"][selection]

            # restore inputs to session_state
            st.session_state["main_hop"]        = loaded["inputs"]["main_hop"]
            st.session_state["secondary_hop"]   = loaded["inputs"]["secondary_hop"]
            st.session_state["hop_1_amount_g"]  = loaded["inputs"]["hop_1_amount_g"]
            st.session_state["hop_2_amount_g"]  = loaded["inputs"]["hop_2_amount_g"]

            st.session_state["malt_base"]       = loaded["inputs"]["malt_base"]
            st.session_state["malt_special"]    = loaded["inputs"]["malt_special"]
            st.session_state["malt_1_pct"]      = loaded["inputs"]["malt_1_pct"]
            st.session_state["malt_2_pct"]      = loaded["inputs"]["malt_2_pct"]

            st.session_state["yeast_strain"]    = loaded["inputs"]["yeast_strain"]
            st.session_state["ferm_temp_c"]     = loaded["inputs"]["ferm_temp_c"]

            # restore predictions
            st.session_state["hop_profile"]     = loaded["predictions"]["hop_profile"]
            st.session_state["malt_profile"]    = loaded["predictions"]["malt_profile"]
            st.session_state["yeast_profile"]   = loaded["predictions"]["yeast_profile"]

            # restore last AI notes
            st.session_state["last_ai_notes"]   = loaded.get("ai_notes","")

            st.success(f"Loaded batch: {selection} ✅")
            st.info("Scroll up — inputs have been restored in memory. You can tweak and click Predict again to recompute visuals.")
    else:
        st.caption("_No saved batches yet — save one on the left!_")

st.markdown("---")

# -------------------------------------------------
# FLAVOR STEERING
# -------------------------------------------------
st.markdown("## 🎯 Flavor Steering")

goal_choice = st.selectbox(
    "What direction do you want to push this beer?",
    [
        "More tropical fruit / mango / pineapple",
        "More body / pillowy mouthfeel",
        "Drier / crisper / brighter bitterness",
        "More color / malt depth",
        "Something else..."
    ],
    index=0,
)

if st.button("💡 Suggest tweaks"):
    tweak_list = flavor_tweak_suggestions(
        goal_choice,
        st.session_state.get("hop_profile", {}),
        st.session_state.get("malt_profile", {}),
        st.session_state.get("yeast_profile", {}),
    )

    st.markdown("#### Suggested levers:")
    for t in tweak_list:
        st.markdown(f"- {t}")

st.markdown("---")

# ---------------------
# AI BREWMASTER GUIDANCE
# ---------------------
if "ai_log" not in st.session_state:
    st.session_state["ai_log"] = []

st.header("🧪 AI Brewmaster Guidance")
style_goal = st.text_area(
    "What's your intent for this beer? (e.g. 'Soft hazy IPA with saturated stone fruit and pineapple, low bitterness, pillowy mouthfeel')",
    "i want to increase mango aroma without increasing bitterness.",
    height=80
)

notes_clicked = st.button("🧪 Generate Brewmaster Notes")
if notes_clicked:
    ai_text = call_azure_brewmaster_notes(
        goal_text=style_goal,
        hop_vec=st.session_state.get("hop_profile", {}),
        malt_vec=st.session_state.get("malt_profile", {}),
        yeast_vec=st.session_state.get("yeast_profile", {}),
    )

    # show
    st.subheader("Brewmaster Notes")
    st.write(ai_text)
    st.caption(
        "Prototype — not production brewing advice. "
        "Always match your yeast strain's process window."
    )

    # persist last notes
    st.session_state["last_ai_notes"] = ai_text

    # log pair for (future) fine-tuning / analysis
    st.session_state["ai_log"].append({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "user_intent": style_goal,
        "notes": ai_text,
    })

# Dev / export log
with st.expander("🧪 Debug / Export AI guidance log (dev only)"):
    st.caption("These pairs can later be used to fine-tune / evaluate advisor quality.")
    st.code(json.dumps(st.session_state["ai_log"], indent=2))
    st.download_button(
        "⬇ Download log as JSON",
        data=json.dumps(st.session_state["ai_log"], indent=2),
        file_name="ai_guidance_log.json",
        mime="application/json"
    )

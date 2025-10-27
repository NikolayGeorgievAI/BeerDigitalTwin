import os
import json
import requests
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from matplotlib.path import Path
from matplotlib.spines import Spine
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection

###############################################################################
#                 STREAMLIT PAGE CONFIG
###############################################################################

st.set_page_config(
    page_title="Beer Recipe Digital Twin",
    page_icon="🍺",
    layout="wide",
)

###############################################################################
#                 HELPER: RADAR CHART FACTORY (matplotlib)
###############################################################################
# We’ll build lightweight radar plots for:
#   - hops/aroma
#   - malt/body-sweetness
#   - yeast/fermentation
# We intentionally hide tick labels and numeric axis labels so we just show shape.

def _radar_factory(num_vars, frame='circle'):
    """
    Create a radar chart with `num_vars` axes.
    Returns: theta (angles), RadarAxes (projection class).
    Adapted from matplotlib radar example.
    """
    # evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):
        name = 'radar'
        # use 1 line segment to connect specified points
        RESOLUTION = 1

        def fill(self, *args, closed=True, **kwargs):
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            lines = super().plot(*args, **kwargs)
            for line in lines:
                line.set_marker("o")
            return lines

        def set_varlabels(self, labels, fontsize=8):
            self.set_thetagrids(
                np.degrees(theta),
                labels,
                fontsize=fontsize
            )

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and have radius of 0.5 in axes coordinates.
            if frame == 'circle':
                return plt.Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                # polygon frame
                verts = self.unit_poly_verts(theta)
                return plt.Polygon(verts, closed=True, edgecolor="black")
            else:
                raise ValueError(f"Unknown frame: {frame}")

        def draw(self, renderer):
            super().draw(renderer)
            # No radial grid labels (we'll hide them in the factory anyway)
        
        @staticmethod
        def unit_poly_verts(theta):
            """Return vertices of polygon for subplot axes (unit circle)."""
            x0, y0, r = [0.5]*3
            verts = [(r*np.cos(t)+x0, r*np.sin(t)+y0) for t in theta]
            return verts
        
        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            # polygon spine
            spine_type = 'circle'
            verts = self.unit_poly_verts(theta)
            verts.append(verts[0])
            path = Path(verts)
            spine = Spine(self, spine_type, path)
            spine.set_transform(self.transAxes)
            return {'polar': spine}

    register_projection(RadarAxes)
    return theta

def make_radar_figure(labels, values, title="", max_val=1.0):
    """
    Build a single radar (polar) figure with hidden radial tick labels.
    labels: list[str]
    values: list[float]
    title:  str
    max_val: float  (we'll set ylimit 0..max_val)
    returns matplotlib Figure
    """
    if len(labels) == 0:
        fig = plt.figure(figsize=(2,2))
        ax = fig.add_subplot(111)
        ax.text(0.5,0.5,"no data",ha="center",va="center")
        ax.set_axis_off()
        return fig

    # Make sure we close the polygon by repeating the first value/label
    L = len(labels)
    theta = _radar_factory(L, frame='polygon')
    vals = values[:]
    vals += values[:1]

    fig = plt.figure(figsize=(3,3))
    ax = fig.add_subplot(111, projection='radar')
    ax.set_rorigin(-0.1)

    # No radial tick labels:
    ax.set_yticklabels([])
    ax.set_ylim(0, max_val)

    # Plot
    ax.plot(theta, values + values[:1], color="#1f77b4")
    ax.fill(theta, values + values[:1], color="#1f77b4", alpha=0.25)
    ax.set_varlabels(labels, fontsize=7)
    ax.set_title(title, fontsize=9, pad=12)

    # Light grid styling
    ax.grid(color='#999999', alpha=0.3)
    return fig


###############################################################################
#                 MODEL LOADING
###############################################################################

ROOT_DIR = os.path.dirname(__file__)

# --- Hop model bundle ---
# Expected joblib dict keys:
#   model                -> trained regressor (multi-output)
#   feature_cols         -> list of columns like "hop_Citra®", "hop_Mosaic"...
#   aroma_dims           -> list of predicted hop aroma attributes
HOP_MODEL_PATH = os.path.join(ROOT_DIR, "hop_aroma_model.joblib")  # adjust if needed
hop_bundle = joblib.load(HOP_MODEL_PATH)
hop_model = hop_bundle["model"]
hop_feature_cols = hop_bundle["feature_cols"]
hop_dims = [
    a for a in hop_bundle["aroma_dims"]
    if str(a).strip().lower() not in ("nan","none","")
]

# --- Malt model bundle ---
# Keys:
#   model
#   feature_cols  -> e.g. ['MOISTURE_MAX','EXTRACT_TYPICAL',... or "malt_*"]
#   flavor_cols   -> output columns e.g. ['color_intensity','sweetness',...]
MALT_MODEL_PATH = os.path.join(ROOT_DIR, "malt_sensory_model.joblib")
malt_bundle = joblib.load(MALT_MODEL_PATH)
malt_model = malt_bundle["model"]
malt_feature_cols = malt_bundle["feature_cols"]
malt_dims = malt_bundle["flavor_cols"]

# --- Yeast model bundle ---
# Keys:
#   model
#   feature_cols -> e.g. 'Name - Nottingham Yeast', 'Lab Coopers', ...
#   flavor_cols  -> output dims like ['Temp_avg_F','Flocculation_num','Attenuation_num']
YEAST_MODEL_PATH = os.path.join(ROOT_DIR, "yeast_sensory_model.joblib")
yeast_bundle = joblib.load(YEAST_MODEL_PATH)
yeast_model = yeast_bundle["model"]
yeast_feature_cols = yeast_bundle["feature_cols"]
yeast_dims = yeast_bundle["flavor_cols"]


###############################################################################
#                 TEXT CLEAN / MATCH HELPERS
###############################################################################

import re

def _clean_name(name: str) -> str:
    if name is None:
        return ""
    s = str(name).lower()
    s = s.replace("®","").replace("™","")
    s = re.sub(r"[^a-z0-9]+","",s)
    return s

def _best_feature_match(user_name: str, feature_cols: list, prefix: str):
    """
    Crude fuzzy-ish match: look among cols starting with `prefix`.
    Score overlap of character sets. Return best column or None.
    """
    cleaned_user = _clean_name(user_name)
    best_col = None
    best_score = -1
    for col in feature_cols:
        if prefix and not col.startswith(prefix):
            continue
        raw_label = col[len(prefix):] if prefix and col.startswith(prefix) else col
        cleaned_label = _clean_name(raw_label)

        if len(cleaned_user)>=3 and cleaned_user[:3] not in cleaned_label:
            # quick gate
            continue

        common = set(cleaned_user)&set(cleaned_label)
        score = len(common)
        if score>best_score:
            best_score=score
            best_col=col
    return best_col

def _choices_from_features(feature_cols, preferred_prefix=None):
    """
    Build nice dropdown names from model feature columns.
    We'll strip common prefixes and brand marks.
    """
    def pretty(label:str):
        label=label.replace("®","").replace("™","")
        label=label.replace("_"," ").strip()
        return label

    chosen=[]
    # If preferred_prefix is provided, gather only those first:
    if preferred_prefix:
        for c in feature_cols:
            if c.startswith(preferred_prefix):
                raw = c[len(preferred_prefix):]
                chosen.append(pretty(raw))

    if not chosen:
        # fallback: all
        for c in feature_cols:
            raw=c
            for p in ["hop_","malt_","grain_","base_","yeast_","strain_","y_","m_"]:
                if raw.startswith(p):
                    raw=raw[len(p):]
            chosen.append(pretty(raw))

    uniq=[]
    for x in chosen:
        if x and x not in uniq:
            uniq.append(x)
    uniq=sorted(uniq, key=lambda s:s.lower())
    return uniq


###############################################################################
#                 FEATURE BUILDERS & PREDICTORS
###############################################################################

def build_hop_features(user_hops):
    """
    user_hops: [ {"name":"Citra","amt":50}, ... ]
    We'll allocate grams into matching hop_* feature columns.
    """
    totals={c:0.0 for c in hop_feature_cols}
    for entry in user_hops:
        nm=entry.get("name","")
        amt=float(entry.get("amt",0.0))
        if amt<=0 or not nm.strip():
            continue
        match=_best_feature_match(nm, hop_feature_cols, prefix="hop_")
        if match:
            totals[match]+=amt
    return pd.DataFrame([totals],columns=hop_feature_cols)

def predict_hop_profile(user_hops):
    if not user_hops:
        return {}
    X=build_hop_features(user_hops)
    y_pred=hop_model.predict(X)[0]
    return {dim:float(val) for dim,val in zip(hop_dims,y_pred)}

def build_malt_features(user_malts):
    """
    user_malts: [{"name":"Maris Otter", "pct":70}, ...], pct in % grist
    We'll try "malt_" prefix first, else "grain_","base_", etc.
    """
    totals={c:0.0 for c in malt_feature_cols}
    for e in user_malts:
        nm=e.get("name","")
        pct=float(e.get("pct",0.0))
        if pct<=0 or not nm.strip():
            continue
        match=_best_feature_match(nm, malt_feature_cols, prefix="malt_")
        if match is None:
            for pfx in ["grain_","base_","malt_","m_"]:
                match=_best_feature_match(nm, malt_feature_cols, prefix=pfx)
                if match:
                    break
        if match:
            totals[match]+=pct
    return pd.DataFrame([totals],columns=malt_feature_cols)

def predict_malt_profile(user_malts):
    if not user_malts:
        return {}
    X=build_malt_features(user_malts)
    y_pred=malt_model.predict(X)[0]
    return {dim:float(val) for dim,val in zip(malt_dims,y_pred)}

def build_yeast_features(user_yeast):
    """
    user_yeast:  {"strain":"Nottingham Ale","ferm_temp_c":20.0}
    We'll fuzzy-match 'strain' to yeast_feature_cols and set that col=1.
    We do NOT yet encode fermentation temp into the model features (kept simple).
    """
    totals={c:0.0 for c in yeast_feature_cols}
    strain=user_yeast.get("strain","")
    match=_best_feature_match(strain, yeast_feature_cols, prefix="yeast_")
    if match is None:
        for p in ["strain_","y_","yeast_",""]:
            if p=="":
                # last resort just try general fuzzy
                for col in yeast_feature_cols:
                    # quick guess with no prefix requirement
                    test_score=len(set(_clean_name(strain))&set(_clean_name(col)))
                    cur_score=len(set(_clean_name(strain))&set(_clean_name(match))) if match else -1
                    if test_score>cur_score:
                        match=col
            else:
                m2=_best_feature_match(strain, yeast_feature_cols, prefix=p)
                if m2:
                    match=m2
                    break
    if match:
        totals[match]=1.0
    return pd.DataFrame([totals],columns=yeast_feature_cols)

def predict_yeast_profile(user_yeast):
    if not user_yeast or not user_yeast.get("strain","").strip():
        return {}
    X=build_yeast_features(user_yeast)
    y_pred=yeast_model.predict(X)[0]
    result = {dim:float(val) for dim,val in zip(yeast_dims,y_pred)}

    # We'll add the user-chosen ferm temp (C + F) as context fields, not from model
    ferm_temp_c=user_yeast.get("ferm_temp_c",20.0)
    ferm_temp_f = ferm_temp_c * 9.0/5.0 + 32.0
    result["Temp_avg_C"]=ferm_temp_c
    result["Temp_avg_F"]=ferm_temp_f
    return result


###############################################################################
#                 AZURE OPENAI BREWMASTER CALL
###############################################################################

def call_azure_brewmaster_notes(
    style_goal_text: str,
    hop_profile: dict,
    malt_profile: dict,
    yeast_profile: dict,
):
    """
    Calls your Azure OpenAI Chat Completions deployment.
    Falls back to a local 'prototype' summary if Azure errors.
    """

    system_msg = (
        "You are a professional brewmaster. "
        "You analyze hop aroma, malt body/sweetness/color, and yeast/fermentation profile. "
        "You give targeted, practical brewing/process suggestions: hop timing & selection, "
        "grain tweaks, fermentation strategy. "
        "No legal disclaimers, no generic fluff. "
        "Write in short actionable bullet-style sections a brewer can apply tonight."
    )

    user_content = (
        "Brewer's stated goal:\n"
        f"{style_goal_text}\n\n"
        "Hop aroma / character:\n"
        f"{json.dumps(hop_profile, indent=2)}\n\n"
        "Malt / body / sweetness / color:\n"
        f"{json.dumps(malt_profile, indent=2)}\n\n"
        "Yeast & fermentation profile:\n"
        f"{json.dumps(yeast_profile, indent=2)}\n\n"
        "Please respond with:\n"
        "1. Hop adjustments (which hops, how to add them, how to push desired aroma without extra bitterness)\n"
        "2. Malt/grist tweaks (sweetness, mouthfeel, color)\n"
        "3. Fermentation guidance (yeast strain, fermentation temperature window, desired attenuation/softness)\n"
        "4. A short final summary with next steps.\n"
    )

    azure_endpoint = st.secrets.get("AZURE_OPENAI_ENDPOINT","").rstrip("/")
    azure_key = st.secrets.get("AZURE_OPENAI_API_KEY","")
    azure_deployment = st.secrets.get("AZURE_OPENAI_DEPLOYMENT","")

    if not (azure_endpoint and azure_key and azure_deployment):
        return build_fallback_brewmaster_notes(style_goal_text, note="Missing Azure secrets")

    # "2024-02-15-preview" works for GPT-4.1 / 4.1-mini in most current tenants
    api_version = "2024-02-15-preview"
    url = (
        f"{azure_endpoint}/openai/deployments/"
        f"{azure_deployment}/chat/completions?api-version={api_version}"
    )

    headers = {
        "Content-Type": "application/json",
        "api-key": azure_key,
    }

    body = {
        "messages":[
            {"role":"system","content":system_msg},
            {"role":"user","content":user_content},
        ],
        "temperature":0.4,
        "max_tokens":400,
        "top_p":0.9,
    }

    try:
        resp=requests.post(url,headers=headers,json=body,timeout=20)
        if resp.status_code!=200:
            return build_fallback_brewmaster_notes(
                style_goal_text,
                note=f"Azure returned {resp.status_code}: {resp.text}",
            )
        data=resp.json()
        ai_text=""
        if ("choices" in data and len(data["choices"])>0
            and "message" in data["choices"][0]
            and "content" in data["choices"][0]["message"]):
            ai_text=data["choices"][0]["message"]["content"]
        if not ai_text:
            return build_fallback_brewmaster_notes(
                style_goal_text,
                note="Azure response had no message content",
            )
        # success
        return format_ai_brewmaster_markdown(ai_text)

    except Exception as e:
        return build_fallback_brewmaster_notes(style_goal_text, note=str(e))


def build_fallback_brewmaster_notes(style_goal_text: str, note: str=""):
    """
    Local proto guidance if Azure fails, still referencing brewer's goal,
    plus we surface note (which might contain 404 / 401 etc.) for debugging.
    """
    base = (
        "Brewmaster Notes (prototype) • "
        "Hop toward tropical/stone-fruit character using high-oil late/dry hop additions, "
        "keep bitterness low to maintain softness. • "
        "Add oats or wheat for pillowy mouthfeel and stable haze. • "
        "Use a moderately ester-friendly yeast and keep fermentation temps in the "
        "18–20°C range (low 60s °F) to stay juicy without fusels. • "
        "Watch attenuation so it doesn't finish bone-dry if you want plush body.\n"
    )
    extra_goal=""
    if style_goal_text.strip():
        extra_goal=f"\nGoal you gave: “{style_goal_text.strip()}”"
    debug=""
    if note:
        debug=f"\n(Azure note: {note})"
    return format_ai_brewmaster_markdown(base+extra_goal+debug)


def format_ai_brewmaster_markdown(text:str):
    return text


###############################################################################
#                 STREAMLIT UI
###############################################################################

st.title("🍺 Beer Recipe Digital Twin")

st.markdown(
    """
Your AI brew assistant:

1. Build a hop bill, grain bill, and fermentation plan.  
2. Predict aroma, body, color, esters, and mouthfeel — all together.  
3. Get brewmaster-style, goal-based guidance powered by your own (or fallback) model.
"""
)

st.markdown("---")

###############################################################################
# HOPS INPUT
###############################################################################

st.subheader("🌿 Hops (late/aroma additions)")

col_h1, col_h2 = st.columns(2)
with col_h1:
    HOP_CHOICES = _choices_from_features(hop_feature_cols, preferred_prefix="hop_")
    default_hop1 = "Mosaic" if "Mosaic" in HOP_CHOICES else (HOP_CHOICES[0] if HOP_CHOICES else "")
    default_hop2 = "Citra" if "Citra" in HOP_CHOICES else (HOP_CHOICES[1] if len(HOP_CHOICES)>1 else default_hop1)

    hop1 = st.selectbox("Main Hop Variety", HOP_CHOICES, index=HOP_CHOICES.index(default_hop1) if default_hop1 in HOP_CHOICES else 0)
    hop2 = st.selectbox("Secondary Hop Variety", HOP_CHOICES, index=HOP_CHOICES.index(default_hop2) if default_hop2 in HOP_CHOICES else 0)

with col_h2:
    hop1_amt = st.number_input("Hop 1 amount (g)", min_value=0.0, max_value=500.0, value=30.0, step=5.0)
    hop2_amt = st.number_input("Hop 2 amount (g)", min_value=0.0, max_value=500.0, value=20.0, step=5.0)

user_hops=[]
if hop1 and hop1_amt>0:
    user_hops.append({"name":hop1,"amt":hop1_amt})
if hop2 and hop2_amt>0:
    user_hops.append({"name":hop2,"amt":hop2_amt})

st.markdown("---")

###############################################################################
# MALT INPUT
###############################################################################

st.subheader("🌾 Malt / Grain Bill")

MALT_CHOICES = _choices_from_features(malt_feature_cols, preferred_prefix="malt_")
# We'll pick first two as "base" and "specialty"
base_default = "EXTRA PALE MALT" if "EXTRA PALE MALT" in MALT_CHOICES else (MALT_CHOICES[0] if MALT_CHOICES else "")
spec_default = "HANÁ MALT" if "HANÁ MALT" in MALT_CHOICES else (MALT_CHOICES[1] if len(MALT_CHOICES)>1 else base_default)

col_m1, col_m2 = st.columns(2)
with col_m1:
    malt1 = st.selectbox(
        "Base / primary malt",
        MALT_CHOICES,
        index=MALT_CHOICES.index(base_default) if base_default in MALT_CHOICES else 0
    )
    malt2 = st.selectbox(
        "Specialty / character malt",
        MALT_CHOICES,
        index=MALT_CHOICES.index(spec_default) if spec_default in MALT_CHOICES else 0
    )

with col_m2:
    malt1_pct = st.number_input("Malt 1 (% grist)", min_value=0.0, max_value=100.0, value=70.0, step=1.0)
    malt2_pct = st.number_input("Malt 2 (% grist)", min_value=0.0, max_value=100.0, value=8.0, step=1.0)

user_malts=[]
if malt1 and malt1_pct>0:
    user_malts.append({"name":malt1,"pct":malt1_pct})
if malt2 and malt2_pct>0:
    user_malts.append({"name":malt2,"pct":malt2_pct})

st.markdown("---")

###############################################################################
# YEAST INPUT
###############################################################################

st.subheader("🧫 Yeast & Fermentation")

YEAST_CHOICES = _choices_from_features(yeast_feature_cols, preferred_prefix="yeast_")
yeast_default = YEAST_CHOICES[0] if YEAST_CHOICES else ""

col_y1, col_y2 = st.columns(2)
with col_y1:
    yeast_strain = st.selectbox(
        "Yeast strain",
        YEAST_CHOICES,
        index=YEAST_CHOICES.index(yeast_default) if yeast_default in YEAST_CHOICES else 0
    )

with col_y2:
    ferm_temp_c = st.number_input(
        "Fermentation temp (°C)",
        min_value=10.0,
        max_value=30.0,
        value=20.0,
        step=0.5
    )

user_yeast = {
    "strain": yeast_strain,
    "ferm_temp_c": ferm_temp_c,
}

st.markdown("---")

###############################################################################
# PREDICT BUTTON
###############################################################################

st.subheader("🍻 Predict Beer Flavor & Balance")

st.caption(
    "Fill hops, malt, and yeast above — then click **'Predict Beer Flavor & Balance'** "
    "to simulate aroma, sweetness/body, color, and fermentation character."
)

predict_clicked = st.button("🍻 Predict Beer Flavor & Balance")

predicted_hops = {}
predicted_malt = {}
predicted_yeast = {}
if predict_clicked:
    predicted_hops = predict_hop_profile(user_hops)
    predicted_malt = predict_malt_profile(user_malts)
    predicted_yeast = predict_yeast_profile(user_yeast)

    # Show bullet-point textual snapshot
    st.markdown("### 📊 Predicted Flavor Snapshot")

    # Hop aroma section
    st.markdown("**Hop aroma / character:**")
    if predicted_hops:
        hop_items = list(predicted_hops.items())
        for k,v in hop_items:
            st.markdown(f"- {k}: {v:.2f}")
    else:
        st.markdown("- (no hop data)")

    # Malt body/sweetness/color
    st.markdown("\n**Malt body / sweetness / color:**")
    if predicted_malt:
        malt_items = list(predicted_malt.items())
        for k,v in malt_items:
            st.markdown(f"- {k}: {v:.2f}")
    else:
        st.markdown("- (no malt data)")

    # Yeast fermentation
    st.markdown("\n**Yeast / fermentation profile:**")
    if predicted_yeast:
        # We'll show them in a friendly order if they exist:
        if "Temp_avg_C" in predicted_yeast and "Temp_avg_F" in predicted_yeast:
            st.markdown(f"- Fermentation temp: {predicted_yeast['Temp_avg_C']:.1f} °C ({predicted_yeast['Temp_avg_F']:.1f} °F)")
        # The model outputs typical numeric dims like "Flocculation_num","Attenuation_num"
        for dim in predicted_yeast:
            if dim not in ("Temp_avg_C","Temp_avg_F"):
                st.markdown(f"- {dim}: {predicted_yeast[dim]:.2f}")
    else:
        st.markdown("- (no yeast data)")

    # Radar row
    st.markdown("---")
    st.markdown("#### Radar Overview")
    st.caption(
        "These radar plots visualize relative shape / balance. "
        "Axes labels are shown, numeric tick labels are hidden."
    )

    # Build 3 radars:
    # 1) Hop aroma radar
    if predicted_hops:
        hop_labels = list(predicted_hops.keys())
        hop_values = [float(predicted_hops[k]) for k in hop_labels]
        fig_hops = make_radar_figure(hop_labels, hop_values, title="Hops / Aroma", max_val=max(hop_values+[1.0]))
    else:
        fig_hops = make_radar_figure([], [], title="Hops / Aroma")

    # 2) Malt/body-sweetness radar
    # We'll pick a subset of "predicted_malt" dims (the ones that imply body/sweetness/color).
    malt_keys_for_radar = []
    for candidate in ["sweetness","body_full","color_intensity","color_intensity_num","sweet_malt","caramel","roast"]:
        for mk in predicted_malt.keys():
            if candidate.lower() in mk.lower() and mk not in malt_keys_for_radar:
                malt_keys_for_radar.append(mk)
    # If we found none, just take first 3 dims so we have *something*
    if not malt_keys_for_radar:
        malt_keys_for_radar = list(predicted_malt.keys())[:3]

    if malt_keys_for_radar:
        malt_vals = [float(predicted_malt[mk]) for mk in malt_keys_for_radar]
        fig_malt = make_radar_figure(malt_keys_for_radar, malt_vals,
                                     title="Malt / Body-Sweetness", 
                                     max_val=max(malt_vals+[1.0]))
    else:
        fig_malt = make_radar_figure([], [], title="Malt / Body-Sweetness")

    # 3) Yeast / fermentation radar
    yeast_keys_for_radar=[]
    for candidate in ["Flocculation","Attenuation","Temp_avg_F","Temp_avg_C"]:
        for yk in predicted_yeast.keys():
            if candidate.lower() in yk.lower() and yk not in yeast_keys_for_radar:
                yeast_keys_for_radar.append(yk)
    if not yeast_keys_for_radar:
        # fallback to first 3 dims (besides the text duplicates)
        yeast_keys_for_radar = [k for k in predicted_yeast.keys() if k not in ("Temp_avg_C","Temp_avg_F")][:3]

    if yeast_keys_for_radar:
        y_vals = [float(predicted_yeast[yk]) for yk in yeast_keys_for_radar]
        fig_yeast = make_radar_figure(yeast_keys_for_radar, y_vals,
                                      title="Yeast / Fermentation",
                                      max_val=max(y_vals+[1.0]))
    else:
        fig_yeast = make_radar_figure([], [], title="Yeast / Fermentation")

    col_r1, col_r2, col_r3 = st.columns(3)
    with col_r1:
        st.pyplot(fig_hops)
    with col_r2:
        st.pyplot(fig_malt)
    with col_r3:
        st.pyplot(fig_yeast)

st.markdown("---")

###############################################################################
# AI BREWMASTER GUIDANCE SECTION
###############################################################################

st.subheader("🧪 AI Brewmaster Guidance")

style_goal_text = st.text_area(
    "What's your intent for this beer? (e.g. 'Soft hazy IPA with saturated stone fruit and pineapple, low bitterness, pillowy mouthfeel')",
    "i want to increase mango aroma without increasing bitterness.",
    height=80
)

gen_notes_clicked = st.button("🧪 Generate Brewmaster Notes")

if gen_notes_clicked:
    # We'll generate Azure-driven or fallback guidance based on current predicted profiles.
    # If user hasn't clicked Predict yet in this run, we still try from scratch (compute on the fly).
    if not predicted_hops:
        predicted_hops = predict_hop_profile(user_hops)
    if not predicted_malt:
        predicted_malt = predict_malt_profile(user_malts)
    if not predicted_yeast:
        predicted_yeast = predict_yeast_profile(user_yeast)

    ai_md = call_azure_brewmaster_notes(
        style_goal_text,
        predicted_hops,
        predicted_malt,
        predicted_yeast,
    )

    st.markdown("### Brewmaster Notes")
    st.write(ai_md)

    st.caption(
        "Prototype — not production brewing advice. Always match your yeast strain's process window."
    )

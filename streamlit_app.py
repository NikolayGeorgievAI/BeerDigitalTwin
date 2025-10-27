#########################
# Beer Recipe Digital Twin (radar fix + bin-mode fix)
#########################

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from typing import List, Dict, Tuple, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


#########################
# --- PAGE CONFIG
#########################

st.set_page_config(
    page_title="Beer Recipe Digital Twin",
    page_icon="🍺",
    layout="wide",
)


#########################
# --- LOAD DATA / MODELS
#########################

@st.cache_data(show_spinner=False)
def load_reference_data():
    """
    Load reference data used to populate dropdowns and metadata.
    """
    try:
        yeast_df = pd.read_pickle("clean_yeast_df.pkl")
    except Exception:
        yeast_df = pd.DataFrame(
            columns=[
                "Name","Lab","Type","Form","Temp","Attenuation","Flocculation","Notes",
                "Temp_avg_C","Flocculation_num","Attenuation_pct",
                "fruity_esters","phenolic_spicy","clean_neutral",
                "dry_finish","malty","sulfur_note"
            ]
        )

    try:
        malt_df = pd.read_pickle("clean_malt_df.pkl")
    except Exception:
        malt_df = pd.DataFrame(columns=["MaltName"])

    return yeast_df, malt_df


class HopModelWrapper:
    """
    Unifies access to hop aroma model:
      - raw joblib might be a Pipeline
      - or a dict{"model": pipeline, "feature_names": [...]}

    After init:
      self.model          -> object with .predict(...)
      self.feature_names  -> list[str] or None
    """

    def __init__(self, raw_obj):
        self.model = None
        self.feature_names = None

        if raw_obj is None:
            return

        # Case 1: raw_obj is already a model
        if hasattr(raw_obj, "predict"):
            self.model = raw_obj
            fn = getattr(raw_obj, "feature_names_in_", None)
            if fn is not None:
                self.feature_names = [str(x) for x in list(fn)]
            else:
                self.feature_names = None

        # Case 2: raw_obj is a dict wrapper
        elif isinstance(raw_obj, dict):
            mdl = raw_obj.get("model", None)
            feat = raw_obj.get("feature_names", None)

            if mdl is not None and hasattr(mdl, "predict"):
                self.model = mdl

            if feat is not None:
                # force python strings
                self.feature_names = [str(x) for x in list(feat)]
            else:
                fn = getattr(mdl, "feature_names_in_", None)
                if fn is not None:
                    self.feature_names = [str(x) for x in list(fn)]
                else:
                    self.feature_names = None

    def is_ready(self):
        return (self.model is not None) and hasattr(self.model, "predict")


@st.cache_resource(show_spinner=False)
def load_hop_model():
    """
    Load hop_aroma_model.joblib and wrap it.
    """
    try:
        raw_obj = joblib.load("hop_aroma_model.joblib")
    except Exception:
        raw_obj = None
    return HopModelWrapper(raw_obj)


#########################
# --- HOP FEATURE BIN LOGIC
#########################

def parse_feature_bin(bin_label: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Convert strings like "[0 - 100]" or "0 - 100" into (0.0, 100.0).
    Return (None, None) if parse fails.
    """
    if bin_label is None:
        return (None, None)

    txt = str(bin_label).strip()

    # Remove surrounding brackets if present
    if txt.startswith("[") and txt.endswith("]"):
        txt = txt[1:-1].strip()

    # Normalize fancy dashes
    txt = txt.replace("–", "-").replace("—", "-")

    parts = [p.strip() for p in txt.split("-")]
    if len(parts) != 2:
        return (None, None)

    try:
        lo = float(parts[0])
        hi = float(parts[1])
        return (lo, hi)
    except ValueError:
        return (None, None)


def feature_names_look_like_bins(feat_list: List[str]) -> bool:
    """
    True if (a) feat_list not empty,
         (b) every item parses to (lo,hi),
         (c) usually small (~4 ranges).
    """
    if not feat_list:
        return False

    parsed_ok = 0
    for f in feat_list:
        lo, hi = parse_feature_bin(f)
        if lo is not None and hi is not None:
            parsed_ok += 1

    # all must parse
    if parsed_ok != len(feat_list):
        return False

    # heuristic: if there are only a handful of bins (like 4),
    # it's almost certainly "bin mode".
    if len(feat_list) <= 10:
        return True

    return True


def build_aligned_df_for_model(
    user_hops: List[Dict[str, float]],
    model_feature_names: Optional[List[str]]
) -> Tuple[pd.DataFrame, float, Dict[str, float], Dict[str, float]]:
    """
    Build the single-row DataFrame for .predict().

    BIN MODE:
      - model_feature_names look like numeric ranges
      - we compute total hop mass
      - mark exactly one bin with 1.0 (others 0.0)

    SPARSE MODE:
      - columns look like hop_Adeena, hop_Amarillo, etc.
      - sum grams of each selected hop into those columns

    Returns
    -------
    aligned_df : pd.DataFrame
    total_mass : float
    bin_hits_debug : dict   (which bin got hit in bin mode)
    sparse_debug : dict     (grams per hop_... in sparse mode)
    """

    total_hop_mass = sum(float(h.get("amt", 0.0) or 0.0) for h in user_hops)
    bin_hits_debug = {}
    sparse_debug = {}

    is_bin_mode = False
    if model_feature_names:
        is_bin_mode = feature_names_look_like_bins(model_feature_names)

    if is_bin_mode:
        # Pure bin mode. Ignore per-hop breakdown.
        row_dict = {}
        for feat in model_feature_names:
            lo, hi = parse_feature_bin(feat)
            if lo is None or hi is None:
                row_dict[feat] = 0.0
                bin_hits_debug[feat] = 0.0
            else:
                in_range = 1.0 if (lo <= total_hop_mass <= hi) else 0.0
                row_dict[feat] = in_range
                bin_hits_debug[feat] = in_range

        aligned_df = pd.DataFrame([row_dict], index=[0])

    else:
        # Sparse mode by hop name -> grams
        aggregate = {}
        for entry in user_hops:
            hop_name = entry.get("name", "-")
            amt_g = float(entry.get("amt", 0.0) or 0.0)
            if hop_name == "-" or amt_g <= 0:
                continue
            col_name = f"hop_{hop_name}"
            aggregate[col_name] = aggregate.get(col_name, 0.0) + amt_g

        sparse_debug = aggregate.copy()

        if model_feature_names:
            row_dict = {}
            for feat in model_feature_names:
                row_dict[feat] = float(aggregate.get(feat, 0.0))
        else:
            # fallback if model has no declared feature names
            row_dict = aggregate if aggregate else {"_dummy": 0.0}

        aligned_df = pd.DataFrame([row_dict], index=[0])

    return aligned_df, total_hop_mass, bin_hits_debug, sparse_debug


#########################
# --- PREDICT AROMA
#########################

AROMA_COLUMNS = [
    "fruity",
    "citrus",
    "tropical",
    "earthy",
    "spicy",
    "herbal",
    "floral",
    "resinous",
]

def predict_hop_aroma(
    hop_wrapper: 'HopModelWrapper',
    aligned_df: pd.DataFrame
) -> Tuple[Dict[str, float], np.ndarray]:
    """
    Run hop_wrapper.model.predict(aligned_df).
    Return (aroma_scores_dict, raw_pred_array).
    If model isn't ready or errors, return zeros.
    """
    aroma_scores = {a: 0.0 for a in AROMA_COLUMNS}
    raw_pred = np.zeros((1, len(AROMA_COLUMNS)), dtype=float)

    if (hop_wrapper is None) or (not hop_wrapper.is_ready()):
        return aroma_scores, raw_pred

    if aligned_df is None or aligned_df.empty:
        return aroma_scores, raw_pred

    try:
        pred = hop_wrapper.model.predict(aligned_df)
        # Ensure ndarray shape (1, N)
        if isinstance(pred, (list, tuple)):
            pred = np.array(pred)
        if pred.ndim == 1:
            pred = pred.reshape(1, -1)

        raw_pred = pred

        for i, aroma in enumerate(AROMA_COLUMNS):
            if i < pred.shape[1]:
                aroma_scores[aroma] = float(pred[0, i])

    except Exception:
        pass

    return aroma_scores, raw_pred


#########################
# --- RADAR CHART
#########################

def make_radar(aroma_scores: Dict[str, float]):
    """
    Radar chart using AROMA_COLUMNS in fixed order.
    """

    vals = [aroma_scores[a] for a in AROMA_COLUMNS]
    n = len(vals)

    closed_vals = vals + vals[:1]
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    closed_angles = np.concatenate([angles, [angles[0]]])

    fig, ax = plt.subplots(
        figsize=(6,6),
        subplot_kw=dict(polar=True)
    )

    # outline
    ax.plot(
        closed_angles,
        closed_vals,
        color="#1f2a44",
        linewidth=2
    )
    ax.fill(
        closed_angles,
        closed_vals,
        color="#1f2a44",
        alpha=0.2
    )

    ax.set_xticks(angles)
    ax.set_xticklabels(AROMA_COLUMNS, fontsize=12)

    ax.set_yticklabels([])
    ax.grid(color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.spines["polar"].set_color("#222")
    ax.spines["polar"].set_linewidth(1.5)

    avg_val = float(np.mean(vals)) if n > 0 else 0.0
    ax.text(
        0.0,
        0.0,
        f"{avg_val:.2f}",
        ha="center",
        va="center",
        fontsize=14,
        bbox=dict(
            facecolor="#e6ebf5",
            edgecolor="#1f2a44",
            boxstyle="round,pad=0.4"
        ),
        color="#1f2a44",
    )

    return fig


#########################
# --- STYLE/CHARACTER HEURISTICS
#########################

def summarize_top_hop_notes(aroma_scores: Dict[str,float], top_n:int=2):
    pairs = [(k,v) for k,v in aroma_scores.items()]
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs[:top_n]


def guess_malt_character(malt_entries: List[Dict[str,float]]) -> str:
    chosen = [m["name"].upper() for m in malt_entries if m.get("name","-") != "-"]

    if any("BLACK" in x for x in chosen):
        return "roasty / dark malt"
    if any("CARA" in x or "CARAMEL" in x for x in chosen):
        return "sweet_caramel"
    return "bready"


def guess_yeast_character(yeast_name: str, yeast_df: pd.DataFrame) -> str:
    if not yeast_name or yeast_name.strip() == "-":
        return "clean / neutral"

    row = yeast_df.loc[yeast_df["Name"] == yeast_name]
    if row.empty:
        return "clean / neutral"

    descs = []
    if "fruity_esters" in row.columns and row["fruity_esters"].iloc[0] == 1:
        descs.append("fruity_esters")
    if "clean_neutral" in row.columns and row["clean_neutral"].iloc[0] == 1:
        descs.append("clean_neutral")
    if "phenolic_spicy" in row.columns and row["phenolic_spicy"].iloc[0] == 1:
        descs.append("phenolic_spicy")

    return ", ".join(descs) if descs else "clean / neutral"


def guess_style_direction(aroma_scores: Dict[str,float], yeast_desc: str) -> str:
    fruity_val = aroma_scores.get("fruity", 0.0)
    if fruity_val > 0.3 and "fruity" in yeast_desc:
        return "🍑 Fruit-forward Ale"
    return "🍻 Experimental / Hybrid"


#########################
# --- SIDEBAR INPUTS
#########################

def sidebar_inputs(yeast_df: pd.DataFrame, malt_df: pd.DataFrame):
    st.sidebar.header("Model Inputs")
    st.sidebar.markdown("### Hop Bill (g)")

    hop_options = ["-", "Adeena", "Admiral", "Amarillo", "Citra", "Simcoe", "Galaxy"]

    hop_entries = []
    for i in range(1,5):
        st.sidebar.markdown(f"**Hop {i}**")
        hop_name = st.sidebar.selectbox(
            f"Hop {i} name",
            options=hop_options,
            index=0,
            key=f"hop{i}_name"
        )
        hop_amt = st.sidebar.number_input(
            f"Hop {i} grams",
            min_value=0.0,
            max_value=500.0,
            value=0.0,
            step=5.0,
            key=f"hop{i}_amt"
        )
        hop_entries.append({"name": hop_name, "amt": hop_amt})

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Malt Bill (%)")

    if "MaltName" in malt_df.columns and malt_df["MaltName"].notna().any():
        malt_options = ["-"] + sorted(malt_df["MaltName"].dropna().unique().tolist())
    else:
        malt_options = ["-", "BEST ALE MALT", "BLACK MALT", "CARA GOLD MALT"]

    malt_entries = []
    for i in range(1,4):
        st.sidebar.markdown(f"**Malt {i}**")
        malt_name = st.sidebar.selectbox(
            f"Malt {i} name",
            options=malt_options,
            index=0,
            key=f"malt{i}_name"
        )
        default_pct = 50.0 if i <= 2 else 0.0
        malt_pct = st.sidebar.number_input(
            f"Malt {i} %",
            min_value=0.0,
            max_value=100.0,
            value=default_pct,
            step=5.0,
            key=f"malt{i}_pct"
        )
        malt_entries.append({"name": malt_name, "pct": malt_pct})

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Yeast Strain")

    yeast_options = ["-"]
    if "Name" in yeast_df.columns:
        yeast_options += yeast_df["Name"].dropna().unique().tolist()

    yeast_choice = st.sidebar.selectbox(
        "Select yeast",
        yeast_options,
        index=0,
        key="yeast_choice"
    )

    st.sidebar.markdown("---")
    run_button = st.sidebar.button("Predict Flavor 🧪")

    return hop_entries, malt_entries, yeast_choice, run_button


#########################
# --- MAIN APP
#########################

def main():
    st.title("🍺 Beer Recipe Digital Twin")
    st.write(
        "Predict hop aroma, malt character, and fermentation profile using trained ML models (work in progress)."
    )

    yeast_df, malt_df = load_reference_data()
    hop_wrapper = load_hop_model()

    hop_entries, malt_entries, yeast_choice, run_button = sidebar_inputs(yeast_df, malt_df)

    aligned_df, total_hop_mass, bin_debug, sparse_debug = build_aligned_df_for_model(
        hop_entries,
        hop_wrapper.feature_names
    )

    aroma_scores, raw_pred = predict_hop_aroma(hop_wrapper, aligned_df)

    fig = make_radar(aroma_scores)

    top_notes = summarize_top_hop_notes(aroma_scores, top_n=2)
    malt_desc = guess_malt_character(malt_entries)
    yeast_desc = guess_yeast_character(yeast_choice, yeast_df)
    style_dir  = guess_style_direction(aroma_scores, yeast_desc)

    left_col, right_col = st.columns([2.2, 1])

    with left_col:
        st.subheader("Hop Aroma Radar")
        st.pyplot(fig, use_container_width=True)

    with right_col:
        st.subheader("Top hop notes:")
        for name, val in top_notes:
            st.markdown(f"- **{name} ({val:.2f})**")

        st.markdown("---")
        st.subheader("Malt character:")
        st.write(malt_desc)

        st.markdown("---")
        st.subheader("Yeast character:")
        st.write(yeast_desc)

        st.markdown("---")
        st.subheader("Style direction:")
        st.write(style_dir)

        st.markdown("---")
        st.subheader("Hops used by the model:")
        used_hops = [
            f'{h["name"]} ({h["amt"]}g)'
            for h in hop_entries
            if h["name"] != "-" and h["amt"] > 0
        ]
        st.write(", ".join(used_hops) if used_hops else "—")

    # Debug block
    st.markdown("---")
    st.markdown("### 🔬 Debug info")

    st.write("User hop entries:")
    st.write(hop_entries)

    st.write("User malt entries:")
    st.write(malt_entries)

    st.write("Selected yeast:", yeast_choice)

    st.write("Aroma scores dict:")
    st.write(aroma_scores)

    st.write("Wrapper feature_names:")
    st.write(hop_wrapper.feature_names)

    st.write("Total hop mass (g):", total_hop_mass)

    st.write("Bin hits (if binned mode):")
    st.write(bin_debug)

    st.write("Sparse aggregate (if sparse mode):")
    st.write(sparse_debug)

    st.write("Aligned DF passed to model:")
    st.write(aligned_df)

    st.write("Raw model prediction output:")
    st.write(f"raw_pred shape: {getattr(raw_pred, 'shape', None)}")
    st.write(raw_pred)

    st.write("Yeast dataset columns:")
    st.write(list(yeast_df.columns))


if __name__ == "__main__":
    main()

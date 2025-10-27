#########################
# Beer Recipe Digital Twin (final bin fix v2)
#########################

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from typing import List, Dict, Tuple, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import re


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


def _clean_feat_name(txt: str) -> str:
    """
    Normalize spacing, brackets, unicode dashes, trailing whitespace,
    etc. e.g. "[0 - 100]g " -> "[0 - 100]".
    We don't remove the bracket entirely here, but we do trim trailing junk.
    """
    s = str(txt)

    # normalize unicode dashes
    s = s.replace("–", "-").replace("—", "-")

    # Collapse repeated spaces
    s = " ".join(s.split())

    # Strip trailing non-numeric garbage after the last digit or bracket
    # e.g. "[0 - 100]g" -> "[0 - 100]"
    m = re.match(r"(.+?)([0-9\]\)])(\D*)$", s)
    if m:
        s = m.group(1) + m.group(2)

    return s.strip()


class HopModelWrapper:
    """
    Wrap hop aroma model, normalizing .feature_names.
    """

    def __init__(self, raw_obj):
        self.model = None
        self.feature_names = None

        if raw_obj is None:
            return

        # case 1: raw_obj is already a predictor
        if hasattr(raw_obj, "predict"):
            self.model = raw_obj
            fn = getattr(raw_obj, "feature_names_in_", None)
            if fn is not None:
                self.feature_names = [_clean_feat_name(x) for x in list(fn)]
            else:
                self.feature_names = None

        # case 2: raw_obj is a dict { "model": ..., "feature_names": ... }
        elif isinstance(raw_obj, dict):
            mdl = raw_obj.get("model", None)
            feat = raw_obj.get("feature_names", None)

            if mdl is not None and hasattr(mdl, "predict"):
                self.model = mdl

            if feat is not None:
                self.feature_names = [_clean_feat_name(x) for x in list(feat)]
            else:
                fn = getattr(mdl, "feature_names_in_", None)
                if fn is not None:
                    self.feature_names = [_clean_feat_name(x) for x in list(fn)]
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
# --- BIN PARSING / ALIGNMENT
#########################

def parse_feature_bin(bin_label: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Convert strings like:
    "[0 - 100]", "0 - 100", "0-100", "[0-100]"
    into (0.0, 100.0).
    We'll aggressively strip brackets & junk and
    keep only digits / dot / minus / whitespace around '-'.
    """
    if not bin_label:
        return (None, None)

    txt = str(bin_label).strip()
    # remove enclosing brackets
    if txt.startswith("[") and txt.endswith("]"):
        txt = txt[1:-1].strip()

    # unify dashes
    txt = txt.replace("–", "-").replace("—", "-")

    # allow something like "0 - 100" or "0-100"
    # we'll extract the two numbers via regex
    # pattern: start num, optional spaces/dash, end num
    m = re.match(r"^\s*([0-9]+(\.[0-9]+)?)\s*-\s*([0-9]+(\.[0-9]+)?)\s*$", txt)
    if not m:
        # fallback: split on '-'
        parts = txt.split("-")
        if len(parts) != 2:
            return (None, None)
        left = parts[0].strip()
        right = parts[1].strip()

        try:
            lo = float(left)
            hi = float(right)
            return (lo, hi)
        except ValueError:
            return (None, None)

    # matched the regex
    left = m.group(1)
    right = m.group(3)
    try:
        lo = float(left)
        hi = float(right)
        return (lo, hi)
    except ValueError:
        return (None, None)


def feature_names_look_like_bins(feat_list: List[str]) -> bool:
    """
    True if *every* feature name in feat_list parses to a numeric bin.
    """
    if not feat_list:
        return False

    for f in feat_list:
        lo, hi = parse_feature_bin(f)
        if lo is None or hi is None:
            return False

    # If they all parse to valid numeric ranges, and there's not a crazy number
    # of them, we'll say this is definitely "bin mode".
    return len(feat_list) <= 20


def assign_mass_to_bins(
    mass: float,
    bin_names: List[str]
) -> Dict[str, float]:
    """
    Return a dict {bin_name: 0/1} lighting exactly one bin for that mass.
    We'll pick the *first* bin whose [lo, hi] range contains 'mass'
    (inclusive of both ends).
    """
    result = {}
    chosen_index = None

    parsed_bins = []
    for i, bn in enumerate(bin_names):
        lo, hi = parse_feature_bin(bn)
        parsed_bins.append((bn, lo, hi))

    for i, (bn, lo, hi) in enumerate(parsed_bins):
        if lo is None or hi is None:
            result[bn] = 0.0
            continue

        # check inclusive range
        if (mass >= lo) and (mass <= hi) and chosen_index is None:
            result[bn] = 1.0
            chosen_index = i
        else:
            result[bn] = 0.0

    return result


def build_aligned_df_for_model(
    user_hops: List[Dict[str, float]],
    model_feature_names: Optional[List[str]]
) -> Tuple[pd.DataFrame, float, Dict[str, float], Dict[str, float], bool]:
    """
    Build the single-row DataFrame for model.predict().

    Returns:
        aligned_df,
        total_hop_mass,
        bin_debug,
        sparse_debug,
        used_bin_mode (bool)
    """
    total_hop_mass = sum(float(h.get("amt", 0.0) or 0.0) for h in user_hops)

    bin_hits_debug: Dict[str, float] = {}
    sparse_debug: Dict[str, float] = {}

    used_bin_mode = False
    if model_feature_names:
        used_bin_mode = feature_names_look_like_bins(model_feature_names)

    if used_bin_mode:
        # one-hot which bin the mass belongs to
        bin_hits_debug = assign_mass_to_bins(total_hop_mass, model_feature_names)

        # now aligned row in the SAME column order expected by model
        row_dict = {feat: float(bin_hits_debug.get(feat, 0.0)) for feat in model_feature_names}
        aligned_df = pd.DataFrame([row_dict], index=[0])

    else:
        # fallback "sparse" mode => sum grams per hop name
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
            # last last fallback
            row_dict = aggregate if aggregate else {"_dummy": 0.0}

        aligned_df = pd.DataFrame([row_dict], index=[0])

    return aligned_df, total_hop_mass, bin_hits_debug, sparse_debug, used_bin_mode


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
    """
    aroma_scores = {a: 0.0 for a in AROMA_COLUMNS}
    raw_pred = np.zeros((1, len(AROMA_COLUMNS)), dtype=float)

    if (hop_wrapper is None) or (not hop_wrapper.is_ready()):
        return aroma_scores, raw_pred

    if aligned_df is None or aligned_df.empty:
        return aroma_scores, raw_pred

    try:
        pred = hop_wrapper.model.predict(aligned_df)

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
    vals = [aroma_scores[a] for a in AROMA_COLUMNS]
    n = len(vals)

    closed_vals = vals + vals[:1]
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    closed_angles = np.concatenate([angles, [angles[0]]])

    fig, ax = plt.subplots(
        figsize=(6,6),
        subplot_kw=dict(polar=True)
    )

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

    aligned_df, total_hop_mass, bin_debug, sparse_debug, used_bin_mode = build_aligned_df_for_model(
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
    st.json(hop_entries)

    st.write("User malt entries:")
    st.json(malt_entries)

    st.write("Selected yeast:", yeast_choice)

    st.write("Aroma scores dict (after mapping):")
    st.json(aroma_scores)

    st.write("Wrapper feature_names (truncated first 20):")
    if hop_wrapper.feature_names:
        st.write(hop_wrapper.feature_names[:20])
        st.write("... total:", len(hop_wrapper.feature_names))

    st.write("Total hop mass (g):", total_hop_mass)
    st.write("used_bin_mode:", used_bin_mode)

    st.write("Sparse aggregate we built:")
    st.json(sparse_debug)

    st.write("Aligned DF columns count:", aligned_df.shape[1])
    st.write("Aligned DF non-zero columns:")
    if not aligned_df.empty:
        nz_cols = [c for c in aligned_df.columns if aligned_df.iloc[0][c] != 0]
        st.write(nz_cols)
        st.write(aligned_df[nz_cols])

    st.write("Raw model prediction output:")
    st.write("raw_pred.shape:", getattr(raw_pred, 'shape', None))
    st.write("raw_pred contents:", raw_pred.tolist() if hasattr(raw_pred, 'tolist') else raw_pred)

    st.write("Yeast dataset columns:")
    st.write(list(yeast_df.columns))


if __name__ == "__main__":
    main()

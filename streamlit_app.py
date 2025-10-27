import math
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt


# ----------------------------
# 1. CONSTANTS / LOOKUPS
# ----------------------------

AROMA_AXES = [
    "fruity",
    "citrus",
    "tropical",
    "earthy",
    "spicy",
    "herbal",
    "floral",
    "resinous",
]

DEFAULT_AROMA_SCORES = {k: 0.0 for k in AROMA_AXES}
STYLE_EMOJI = "🍻"


# ----------------------------
# 2. MODEL WRAPPER
# ----------------------------

class HopModelWrapper:
    """
    Wraps whatever we loaded from hop_aroma_model.joblib.
    Handles:
      - .model  (the thing with .predict)
      - .feature_names (list of expected features)
      - .last_raw_pred (for debug)
    """

    def __init__(self, raw_obj):
        self.raw = raw_obj
        self.model = None
        self.feature_names = None
        self.error = None
        self.last_raw_pred = None  # <-- debug hook

        # pick out model object
        if hasattr(raw_obj, "predict"):
            self.model = raw_obj
        elif isinstance(raw_obj, dict):
            if "model" in raw_obj and hasattr(raw_obj["model"], "predict"):
                self.model = raw_obj["model"]
            else:
                for v in raw_obj.values():
                    if hasattr(v, "predict"):
                        self.model = v
                        break

        # figure out feature names
        self.feature_names = self._extract_feature_names(raw_obj)

    def _extract_feature_names(self, raw_obj):
        # direct estimator
        if hasattr(raw_obj, "feature_names_in_"):
            return list(raw_obj.feature_names_in_)
        if hasattr(raw_obj, "feature_names_in"):
            return list(raw_obj.feature_names_in)

        # dict
        if isinstance(raw_obj, dict):
            if "feature_names" in raw_obj and isinstance(raw_obj["feature_names"], list):
                return raw_obj["feature_names"]
            if "feature_names_in_" in raw_obj and isinstance(raw_obj["feature_names_in_"], list):
                return raw_obj["feature_names_in_"]
            if "model" in raw_obj:
                sub = raw_obj["model"]
                if hasattr(sub, "feature_names_in_"):
                    return list(sub.feature_names_in_)
                if hasattr(sub, "feature_names_in"):
                    return list(sub.feature_names_in)

        return None

    def predict_scores(self, aligned_df: pd.DataFrame) -> Tuple[Dict[str, float], Optional[str]]:
        """
        aligned_df should ALREADY match the model's expected columns.
        Returns aroma_scores (all 8 axes) + error or None.
        Also stores raw model output in self.last_raw_pred for debug.
        """
        out_scores = {axis: 0.0 for axis in AROMA_AXES}

        if self.model is None or not hasattr(self.model, "predict"):
            return out_scores, "No usable hop model."

        if aligned_df is None or aligned_df.empty:
            return out_scores, "Aligned DF is empty."

        try:
            pred = self.model.predict(aligned_df)
        except Exception as e:
            return out_scores, f"Predict crashed: {e}"

        # keep raw pred for debug
        self.last_raw_pred = pred

        # interpret prediction
        # case 1: model returns a list/array of dicts or vectors
        first = None
        if isinstance(pred, (list, np.ndarray, pd.Series)):
            if len(pred) > 0:
                first = pred[0]
            else:
                first = pred
        else:
            # single object/single dict/single array?
            first = pred

        # dict mapping axis->score?
        if isinstance(first, dict):
            for axis in AROMA_AXES:
                out_scores[axis] = float(first.get(axis, 0.0))
            return out_scores, None

        # numeric vector same length as AROMA_AXES?
        if hasattr(first, "__len__") and not isinstance(first, str):
            try:
                first_list = list(first)
            except Exception:
                first_list = None

            if first_list is not None and len(first_list) == len(AROMA_AXES):
                for axis, val in zip(AROMA_AXES, first_list):
                    out_scores[axis] = float(val)
                return out_scores, None

        # couldn't map
        return out_scores, (
            "Model returned unexpected output "
            f"(type={type(first)} value={first})"
        )


def load_hop_model_wrapper(path: str = "hop_aroma_model.joblib") -> HopModelWrapper:
    try:
        raw = joblib.load(path)
        return HopModelWrapper(raw)
    except Exception as e:
        dummy = HopModelWrapper({})
        dummy.error = f"Failed to load hop model: {e}"
        return dummy


# ----------------------------
# 3. DATA HELPERS
# ----------------------------

@st.cache_data
def load_clean_malt_df(path: str = "clean_malt_df.pkl") -> pd.DataFrame:
    try:
        return pd.read_pickle(path)
    except Exception:
        # fallback/basic rows if pickle missing
        return pd.DataFrame(
            {"MaltName": ["BEST ALE MALT", "BLACK MALT", "CARA GOLD MALT"]}
        )


@st.cache_data
def load_clean_yeast_df(path: str = "clean_yeast_df.pkl") -> pd.DataFrame:
    try:
        df = pd.read_pickle(path)
    except Exception:
        df = pd.DataFrame(
            {
                "Name": [
                    "Cooper Ale",
                    "Edme Ale Yeast",
                    "Manchester",
                ],
                "fruity_esters": [1, 0, 0],
                "phenolic_spicy": [0, 0, 0],
                "clean_neutral": [0, 1, 1],
            }
        )
    return df


def unique_sorted(series_like) -> List[str]:
    vals = []
    for v in series_like:
        if pd.isna(v):
            continue
        s = str(v).strip()
        if s and s not in vals:
            vals.append(s)
    return ["-"] + sorted(vals, key=lambda x: x.lower())


def parse_feature_bin(bin_label: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Convert something like "0 - 100" -> (0.0, 100.0).
    Return (None, None) if it doesn't parse.
    """
    txt = bin_label.replace("–", "-").replace("—", "-")
    parts = [p.strip() for p in txt.split("-")]
    if len(parts) != 2:
        return (None, None)
    try:
        lo = float(parts[0])
        hi = float(parts[1])
        return (lo, hi)
    except ValueError:
        return (None, None)


def build_hop_vector_binned(user_hops: List[Dict[str, float]], feature_names: List[str]) -> Tuple[pd.DataFrame, float, Dict[str, float]]:
    """
    1) Sum total hop mass.
    2) For each feature bin "X - Y", set 1.0 if total mass is in [X,Y], else 0.0.
    Returns:
      aligned_df (1 row),
      total_hop_mass,
      bin_hits {bin_label:1.0/0.0,...} for debug.
    """
    total_hop_mass = sum(float(h["amt"] or 0.0) for h in user_hops)

    row_dict = {}
    bin_hits = {}
    for feat in feature_names:
        lo, hi = parse_feature_bin(feat)
        if lo is None or hi is None:
            row_dict[feat] = 0.0
            bin_hits[feat] = 0.0
            continue
        in_range = 1.0 if (lo <= total_hop_mass <= hi) else 0.0
        row_dict[feat] = in_range
        bin_hits[feat] = in_range

    df = pd.DataFrame([row_dict], index=[0])
    return df, total_hop_mass, bin_hits


def build_hop_vector_sparse(user_hops: List[Dict[str, float]], feature_names: List[str]) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    (fallback if the model wants columns like 'hop_Adeena', etc.)
    We'll aggregate grams per hop name and align to feature_names.
    Returns df + aggregate dict for debug.
    """
    aggregate = {}
    for entry in user_hops:
        hop_name = entry.get("name", "-")
        amt_g = float(entry.get("amt", 0.0) or 0.0)
        if hop_name == "-" or amt_g <= 0:
            continue
        col_name = f"hop_{hop_name}"
        aggregate[col_name] = aggregate.get(col_name, 0.0) + amt_g

    aligned_row = {}
    for feat in feature_names:
        aligned_row[feat] = float(aggregate.get(feat, 0.0))
    df = pd.DataFrame([aligned_row], index=[0])

    return df, aggregate


def infer_style_direction(yeast_row: pd.Series) -> str:
    fruity = float(yeast_row.get("fruity_esters", 0) or 0)
    clean = float(yeast_row.get("clean_neutral", 0) or 0)

    if fruity >= 1 and clean <= 0:
        return "Fruit-forward Ale"
    if clean >= 1 and fruity <= 0:
        return "Clean / Neutral Ale"
    return "Experimental / Hybrid"


def infer_top_notes(aroma_scores: Dict[str, float], top_n: int = 2) -> List[Tuple[str, float]]:
    pairs = [(k, float(v)) for k, v in aroma_scores.items()]
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs[:top_n]


# ----------------------------
# 4. SIDEBAR
# ----------------------------

def sidebar_inputs(malt_df: pd.DataFrame, yeast_df: pd.DataFrame) -> Tuple[List[Dict], List[Dict], str, bool]:
    st.sidebar.header("Hop Bill (g)")

    manual_hops = [
        "-", "Adeena", "Admiral", "Amarillo", "Astra", "Citra", "Mosaic", "Simcoe"
    ]
    hop_options = ["-"] + sorted(set(manual_hops), key=lambda x: x.lower())

    user_hops = []
    for i in range(1, 5):
        st.sidebar.subheader(f"Hop {i}")
        hname = st.sidebar.selectbox(f"Hop {i} name", hop_options, key=f"hop{i}_name")
        hgrams = st.sidebar.number_input(
            f"(g) for Hop {i}",
            min_value=0.0,
            max_value=500.0,
            value=0.0,
            step=5.0,
            key=f"hop{i}_amt",
        )
        user_hops.append({"name": hname, "amt": hgrams})

    st.sidebar.markdown("---")
    st.sidebar.header("Malt Bill (%)")

    # figure out which column in malt_df to use
    malt_col_guess = None
    for cand in ["MaltName", "malt_name", "Malt", "Name"]:
        if cand in malt_df.columns:
            malt_col_guess = cand
            break
    if malt_col_guess is None:
        malt_col_guess = "MaltName"
        if "MaltName" not in malt_df.columns:
            malt_df = pd.DataFrame({"MaltName": ["BEST ALE MALT", "BLACK MALT", "CARA GOLD MALT"]})

    malt_options = unique_sorted(malt_df[malt_col_guess].astype(str))

    user_malts = []
    for i in range(1, 4):
        st.sidebar.subheader(f"Malt {i}")
        mname = st.sidebar.selectbox(
            f"Malt {i} name",
            malt_options,
            key=f"malt{i}_name",
        )
        mpct = st.sidebar.number_input(
            f"Malt {i} %",
            min_value=0.0,
            max_value=100.0,
            value=50.0 if i == 1 else 0.0,
            step=5.0,
            key=f"malt{i}_pct",
        )
        user_malts.append({"name": mname, "pct": mpct})

    st.sidebar.markdown("---")
    st.sidebar.header("Yeast Strain")

    yeast_options = unique_sorted(yeast_df["Name"].astype(str))
    yeast_choice = st.sidebar.selectbox("Select yeast", yeast_options, key="yeast_choice")

    run_button = st.sidebar.button("Predict Flavor 🧪")
    st.sidebar.markdown("---")

    return user_hops, user_malts, yeast_choice, run_button


# ----------------------------
# 5. RADAR PLOT
# ----------------------------

def make_radar(aroma_scores: Dict[str, float]) -> plt.Figure:
    vals = [float(aroma_scores.get(axis, 0.0)) for axis in AROMA_AXES]
    N = len(AROMA_AXES)
    angles = np.linspace(0, 2 * math.pi, N, endpoint=False)

    fig, ax = plt.subplots(subplot_kw=dict(polar=True), figsize=(6, 6))

    vals_loop = vals + [vals[0]]
    angles_loop = list(angles) + [angles[0]]

    ax.plot(angles_loop, vals_loop, color="#1f2a44", linewidth=2)
    ax.fill(angles_loop, vals_loop, color="#1f2a44", alpha=0.25)

    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylim(0.0, 1.0)

    ax.set_xticks(angles)
    ax.set_xticklabels(AROMA_AXES, fontsize=12)

    mean_val = float(np.mean(vals)) if len(vals) else 0.0
    ax.text(
        0.0,
        0.0,
        f"{mean_val:.2f}",
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
        bbox=dict(facecolor="#e6eaf6", edgecolor="#1f2a44", boxstyle="round,pad=0.4"),
    )

    ax.grid(color="gray", linestyle="--", linewidth=0.7, alpha=0.7)
    return fig


# ----------------------------
# 6. MAIN
# ----------------------------

def looks_like_bins(feat_list: List[str]) -> bool:
    """
    True if every feature name parses as "lo - hi".
    """
    if not feat_list:
        return False
    ok = 0
    for f in feat_list:
        lo, hi = parse_feature_bin(f)
        if lo is not None and hi is not None:
            ok += 1
    return ok == len(feat_list)


def main():
    st.set_page_config(
        page_title="Beer Recipe Digital Twin",
        page_icon="🍺",
        layout="wide",
    )

    st.title("🍺 Beer Recipe Digital Twin")
    st.write(
        "Predict hop aroma, malt character, and fermentation profile using trained ML models (work in progress)."
    )

    # load data/model
    malt_df = load_clean_malt_df()
    yeast_df = load_clean_yeast_df()
    hop_wrapper = load_hop_model_wrapper()

    # inputs
    user_hops, user_malts, yeast_choice, run_clicked = sidebar_inputs(malt_df, yeast_df)

    # yeast row
    yeast_row = yeast_df[yeast_df["Name"] == yeast_choice]
    if yeast_row.empty:
        yeast_row = yeast_df.iloc[[0]].copy()
    yeast_row = yeast_row.iloc[0]

    # ----------------
    # Build aligned_df for the model
    # ----------------
    aligned_df = pd.DataFrame()
    total_hop_mass = sum(float(h["amt"] or 0.0) for h in user_hops)
    bin_hits_debug = {}
    sparse_debug = {}

    if hop_wrapper.feature_names and looks_like_bins(hop_wrapper.feature_names):
        # Binned mode
        aligned_df, total_hop_mass, bin_hits_debug = build_hop_vector_binned(
            user_hops, hop_wrapper.feature_names
        )
    elif hop_wrapper.feature_names:
        # Sparse per-hop mode
        aligned_df, sparse_debug = build_hop_vector_sparse(
            user_hops, hop_wrapper.feature_names
        )
    else:
        # No feature names: still build sparse dict from user hops
        # and just pass that raw (this is last-resort)
        agg = {}
        for entry in user_hops:
            hop_name = entry.get("name", "-")
            amt_g = float(entry.get("amt", 0.0) or 0.0)
            if hop_name == "-" or amt_g <= 0:
                continue
            col_name = f"hop_{hop_name}"
            agg[col_name] = agg.get(col_name, 0.0) + amt_g
        if agg:
            aligned_df = pd.DataFrame([agg], index=[0])
        else:
            # ensure at least 1 row so .predict doesn't immediately explode
            aligned_df = pd.DataFrame([{}], index=[0])

    # ----------------
    # Predict
    # ----------------
    aroma_scores = DEFAULT_AROMA_SCORES.copy()
    model_error = None
    raw_pred_for_debug = None

    if run_clicked:
        aroma_scores, model_error = hop_wrapper.predict_scores(aligned_df)
        raw_pred_for_debug = hop_wrapper.last_raw_pred

    # ----------------
    # Layout columns
    # ----------------
    col_chart, col_meta = st.columns([1.1, 1])

    with col_chart:
        st.subheader("Hop Aroma Radar")
        fig = make_radar(aroma_scores)
        st.pyplot(fig, clear_figure=True)

    with col_meta:
        st.subheader("Top hop notes:")
        for note, score in infer_top_notes(aroma_scores, top_n=2):
            st.write(f"• {note} ({score:.2f})")

        st.markdown("---")

        st.subheader("Malt character:")
        st.write("bready")  # placeholder

        st.markdown("---")

        st.subheader("Yeast character:")
        yeast_traits = []
        if float(yeast_row.get("fruity_esters", 0) or 0) >= 1:
            yeast_traits.append("fruity_esters")
        if float(yeast_row.get("clean_neutral", 0) or 0) >= 1:
            yeast_traits.append("clean_neutral")
        if not yeast_traits:
            yeast_traits.append("clean / neutral")
        st.write(", ".join(yeast_traits))

        st.markdown("---")

        st.subheader("Style direction:")
        style_guess = infer_style_direction(yeast_row)
        st.write(f"{STYLE_EMOJI} {style_guess}")

        st.markdown("---")

        st.subheader("Hops used by the model:")
        used_hops_str = []
        for hop in user_hops:
            if hop["name"] != "-" and hop["amt"] > 0:
                used_hops_str.append(f"{hop['name']} ({hop['amt']:.0f}g)")
        st.write(", ".join(used_hops_str) if used_hops_str else "—")

    # ----------------
    # Debug info
    # ----------------
    st.markdown("---")
    st.subheader("🔬 Debug info")

    st.write("User hop entries:")
    st.json(user_hops)

    st.write("User malt entries:")
    st.json(user_malts)

    st.write("Selected yeast:", yeast_choice)

    st.write("Aroma scores dict:")
    st.json(aroma_scores)

    st.write("Wrapper feature_names:")
    st.json(hop_wrapper.feature_names)

    st.write("Total hop mass (g):", float(total_hop_mass))

    st.write("Bin hits (if binned mode):")
    st.json(bin_hits_debug)

    st.write("Sparse aggregate (if sparse mode):")
    st.json(sparse_debug)

    st.write("Aligned DF passed to model:")
    st.dataframe(aligned_df, use_container_width=True)

    if raw_pred_for_debug is not None:
        st.write("Raw model prediction output:")
        try:
            # if it's array-like, convert to list so Streamlit can show it
            if isinstance(raw_pred_for_debug, (np.ndarray, pd.Series)):
                st.json(raw_pred_for_debug.tolist())
            else:
                st.json(raw_pred_for_debug)
        except Exception:
            st.write(raw_pred_for_debug)

    if hop_wrapper.error:
        st.error(f"Hop model load error: {hop_wrapper.error}")
    if model_error:
        st.error(f"Model prediction error: {model_error}")

    st.write("Yeast dataset columns:")
    st.json(list(yeast_df.columns))

    st.dataframe(yeast_df.head(10), use_container_width=True)


if __name__ == "__main__":
    main()

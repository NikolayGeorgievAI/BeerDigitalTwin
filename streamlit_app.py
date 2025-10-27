import math
import json
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

# friendly emoji for style direction, etc
STYLE_EMOJI = "🍻"

# ----------------------------
# 2. MODEL WRAPPER
# ----------------------------

class HopModelWrapper:
    """
    A thin safety wrapper around whatever we load from hop_aroma_model.joblib.

    We try to read:
      - model.feature_names_in_  OR
      - model["feature_names"]   (if saved manually)  OR
      - model["feature_names_in_"]
    We store those as .feature_names (list of col names the model expects).

    Then .predict(hop_feature_df) will:
      - align user df to model.feature_names,
      - call model.predict if possible,
      - or return zeros with an error message.
    """

    def __init__(self, raw_obj):
        self.raw = raw_obj      # can be Pipeline, dict, etc.
        self.model = None       # the thing with .predict()
        self.feature_names = None
        self.error = None

        # 1. If it's already a sklearn Pipeline / estimator
        if hasattr(raw_obj, "predict"):
            self.model = raw_obj
        # 2. If it's a dict-like saved structure
        elif isinstance(raw_obj, dict):
            # Try to find a subkey that looks like the estimator
            if "model" in raw_obj and hasattr(raw_obj["model"], "predict"):
                self.model = raw_obj["model"]
            else:
                # fallback: maybe they shoved pipeline directly in dict?
                for v in raw_obj.values():
                    if hasattr(v, "predict"):
                        self.model = v
                        break

        # figure out feature names
        self.feature_names = self._extract_feature_names(raw_obj)

    def _extract_feature_names(self, raw_obj):
        # Try common patterns
        # a) sklearn pipeline / estimator w/ feature_names_in_
        if hasattr(raw_obj, "feature_names_in_"):
            return list(raw_obj.feature_names_in_)
        if hasattr(raw_obj, "feature_names_in"):
            return list(raw_obj.feature_names_in)
        # b) dict-saved
        if isinstance(raw_obj, dict):
            if "feature_names" in raw_obj and isinstance(raw_obj["feature_names"], list):
                return raw_obj["feature_names"]
            if "feature_names_in_" in raw_obj and isinstance(raw_obj["feature_names_in_"], list):
                return raw_obj["feature_names_in_"]
            # sometimes nested "model"
            if "model" in raw_obj:
                sub = raw_obj["model"]
                if hasattr(sub, "feature_names_in_"):
                    return list(sub.feature_names_in_)
                if hasattr(sub, "feature_names_in"):
                    return list(sub.feature_names_in)

        # We also sometimes saw the warning "feature_names": ["hop_Adeena", ...]
        # If we still can't find them, return None -> means "unknown"
        return None

    def predict_scores(self, hop_vector_df: pd.DataFrame) -> Tuple[Dict[str, float], Optional[str]]:
        """
        Return ({axis->score}, error_msg_or_None)

        Will ALWAYS return *all 8* AROMA_AXES so that the radar can plot,
        even if model fails.
        """

        # default
        out_scores = {axis: 0.0 for axis in AROMA_AXES}

        # If we don't have a usable model, bail gracefully
        if self.model is None or not hasattr(self.model, "predict"):
            return out_scores, "No usable hop model (no .predict)."

        # If we don't have feature names, we can't align columns.
        if self.feature_names is None or len(self.feature_names) == 0:
            return out_scores, "Model has no known feature_names; can't align."

        # Align "hop_vector_df" to model.feature_names
        #   - We'll build aligned_df with exactly those columns
        #   - fill missing with 0.0
        try:
            aligned_cols = {}
            for col in self.feature_names:
                aligned_cols[col] = hop_vector_df[col] if col in hop_vector_df.columns else 0.0
            aligned_df = pd.DataFrame([aligned_cols])
        except Exception as e:
            return out_scores, f"Failed to align user hops -> model features: {e}"

        # Now run .predict(...)
        try:
            pred = self.model.predict(aligned_df)
            # We assume pred[0] is either:
            #   (a) array-like of length 8 (scores in our aroma axes order),
            #   (b) dict {axis->score},
            #   (c) scalar or nonsense.
            first = pred[0] if hasattr(pred, "__len__") and len(pred) > 0 else pred
        except Exception as e:
            return out_scores, f"Predict crashed: {e}"

        # interpret first
        if isinstance(first, dict):
            # pick known axes in our standard order
            for axis in AROMA_AXES:
                out_scores[axis] = float(first.get(axis, 0.0))
            return out_scores, None
        elif hasattr(first, "__len__") and len(first) == len(AROMA_AXES):
            # assume direct numeric vector [fruity,citrus,...]
            for axis, val in zip(AROMA_AXES, list(first)):
                out_scores[axis] = float(val)
            return out_scores, None
        else:
            # can't parse
            return out_scores, (
                "Model.predict() returned something unexpected. "
                f"type={type(first)}, value={first}"
            )


def load_hop_model_wrapper(joblib_path: str = "hop_aroma_model.joblib") -> HopModelWrapper:
    """
    Safely load your hop aroma model using joblib.
    If load fails, we create an "empty" wrapper that always returns zeros.
    """
    try:
        raw = joblib.load(joblib_path)
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
        # fallback empty
        return pd.DataFrame(
            {
                "MaltName": ["BEST ALE MALT", "BLACK MALT", "CARA GOLD MALT"],
            }
        )


@st.cache_data
def load_clean_yeast_df(path: str = "clean_yeast_df.pkl") -> pd.DataFrame:
    try:
        df = pd.read_pickle(path)
    except Exception:
        # Minimal fallback shape
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
    """Return a sorted list of unique (stringified) values, ignoring NaN."""
    vals = []
    for v in series_like:
        if pd.isna(v):
            continue
        s = str(v).strip()
        if s and s not in vals:
            vals.append(s)
    vals_sorted = sorted(vals, key=lambda x: x.lower())
    # prepend "-" as blank
    return ["-"] + vals_sorted


def build_hop_feature_vector(user_hops: List[Dict[str, float]]) -> pd.DataFrame:
    """
    Turn a list of hops entries like:
      [{"name":"Simcoe","amt":30}, {"name":"Citra","amt":50}, ...]
    into a single-row dataframe:
      {
        "hop_Simcoe": 30,
        "hop_Citra":  50,
        ...
      }

    This is BEFORE alignment to model.feature_names.
    """

    aggregate = {}
    for entry in user_hops:
        hop_name = entry.get("name", "-")
        amt_g = float(entry.get("amt", 0.0) or 0.0)
        if hop_name == "-" or amt_g <= 0:
            continue
        col_name = f"hop_{hop_name}"
        aggregate[col_name] = aggregate.get(col_name, 0.0) + amt_g

    if not aggregate:
        # empty row
        return pd.DataFrame([{}])
    return pd.DataFrame([aggregate])


def infer_style_direction(yeast_row: pd.Series) -> str:
    """
    Dumb heuristic for "style direction"
    using yeast traits. Can adjust later.
    """
    fruity = float(yeast_row.get("fruity_esters", 0) or 0)
    clean = float(yeast_row.get("clean_neutral", 0) or 0)

    if fruity >= 1 and clean <= 0:
        return "Fruit-forward Ale"
    if clean >= 1 and fruity <= 0:
        return "Clean / Neutral Ale"
    return "Experimental / Hybrid"


def infer_top_notes(aroma_scores: Dict[str, float], top_n: int = 2) -> List[Tuple[str, float]]:
    """
    Return top N aroma axes with score, descending.
    """
    pairs = [(k, float(v)) for k, v in aroma_scores.items()]
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs[:top_n]


# ----------------------------
# 4. SIDEBAR UI
# ----------------------------

def sidebar_inputs(malt_df: pd.DataFrame, yeast_df: pd.DataFrame) -> Tuple[List[Dict], List[Dict], str, bool]:
    """
    Returns:
      user_hops   : list of dicts {name, amt}
      user_malts  : list of dicts {name, pct}
      yeast_name  : str
      run_clicked : bool
    """

    st.sidebar.header("Hop Bill (g)")
    hop_options = unique_sorted(
        []  # We'll allow manual typed list? For now leave blank.
    )
    # We don't actually have the global hop list in pkl yet,
    # so we fall back to free text or do a short list:
    # Let's create a tiny manual list for now:
    manual_hops = [
        "-", "Simcoe", "Citra", "Amarillo", "Mosaic", "Astra", "Adeena", "Admiral"
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

    # create malt_options from malt_df if present
    malt_col_guess = None
    for cand in ["MaltName", "malt_name", "Malt", "Name"]:
        if cand in malt_df.columns:
            malt_col_guess = cand
            break
    if malt_col_guess is None:
        malt_col_guess = "MaltName"
        # fallback minimal
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
            value=0.0 if i > 1 else 50.0,  # default first 50, others 0
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
# 5. RADAR CHART
# ----------------------------

def make_radar(aroma_scores: Dict[str, float]) -> plt.Figure:
    """
    Build a classic 'spider' radar chart from a dict of 8 axes -> score.
    Ensures angles and values arrays line up for Matplotlib's polar plot.
    """

    # guarantee we have all 8 axes in correct order
    vals = [float(aroma_scores.get(axis, 0.0)) for axis in AROMA_AXES]

    # number of axes
    N = len(AROMA_AXES)
    # angles around the circle for each axis
    angles = np.linspace(0, 2 * math.pi, N, endpoint=False)

    fig, ax = plt.subplots(subplot_kw=dict(polar=True), figsize=(6, 6))

    # close the loop for plotting (repeat first)
    vals_loop = vals + [vals[0]]
    angles_loop = list(angles) + [angles[0]]

    # draw polygon
    ax.plot(angles_loop, vals_loop, color="#1f2a44", linewidth=2)
    ax.fill(angles_loop, vals_loop, color="#1f2a44", alpha=0.25)

    # radial grid styling
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylim(0.0, 1.0)

    # label each axis
    ax.set_xticks(angles)
    ax.set_xticklabels(AROMA_AXES, fontsize=12)

    # annotate center value? we'll annotate mean for a quick visual "intensity"
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

    # lighter grid style
    ax.grid(color="gray", linestyle="--", linewidth=0.7, alpha=0.7)
    return fig


# ----------------------------
# 6. MAIN APP LOGIC
# ----------------------------

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

    # load static data
    malt_df = load_clean_malt_df()
    yeast_df = load_clean_yeast_df()
    hop_wrapper = load_hop_model_wrapper()  # from hop_aroma_model.joblib

    # collect inputs
    user_hops, user_malts, yeast_choice, run_clicked = sidebar_inputs(malt_df, yeast_df)

    # pick yeast row
    yeast_row = yeast_df[yeast_df["Name"] == yeast_choice]
    if yeast_row.empty:
        yeast_row = yeast_df.iloc[[0]].copy()
    yeast_row = yeast_row.iloc[0]

    # ----------------
    # Predict hop aroma if requested
    # ----------------
    aroma_scores = DEFAULT_AROMA_SCORES.copy()
    model_error = None

    if run_clicked:
        # build hop vector frame (un-aligned)
        raw_hop_df = build_hop_feature_vector(user_hops)
        aroma_scores, model_error = hop_wrapper.predict_scores(raw_hop_df)

    # ----------------
    # right column: textual predictions
    # ----------------
    col_chart, col_meta = st.columns([1.1, 1])

    with col_chart:
        st.subheader("Hop Aroma Radar")

        # show radar (always show something, even if run_clicked False -> zeros)
        fig = make_radar(aroma_scores)
        st.pyplot(fig, clear_figure=True)

    with col_meta:
        # Top hop notes
        st.subheader("Top hop notes:")
        top_notes = infer_top_notes(aroma_scores, top_n=2)
        for note, score in top_notes:
            st.write(f"• {note} ({score:.2f})")

        st.markdown("---")

        # Malt character - naive heuristic: pick the "bready/sweet_malt" style
        st.subheader("Malt character:")
        # we can do very naive descriptor from malts used
        # for now just "bready"
        st.write("bready")

        st.markdown("---")

        # Yeast character
        st.subheader("Yeast character:")
        # let's pick the main descriptors from the yeast row
        yeast_traits = []
        if float(yeast_row.get("fruity_esters", 0) or 0) >= 1:
            yeast_traits.append("fruity_esters")
        if float(yeast_row.get("clean_neutral", 0) or 0) >= 1:
            yeast_traits.append("clean_neutral")
        if not yeast_traits:
            yeast_traits.append("clean / neutral")
        st.write(", ".join(yeast_traits))

        st.markdown("---")

        # Style direction
        st.subheader("Style direction:")
        style_guess = infer_style_direction(yeast_row)
        st.write(f"{STYLE_EMOJI} {style_guess}")

        st.markdown("---")

        # Hops used
        st.subheader("Hops used by the model:")
        used_hops_str = []
        for hop in user_hops:
            if hop["name"] != "-" and hop["amt"] > 0:
                used_hops_str.append(f"{hop['name']} ({hop['amt']:.0f}g)")
        if used_hops_str:
            st.write(", ".join(used_hops_str))
        else:
            st.write("—")

    # ----------------
    # Debug panel
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

    st.write("Wrapper feature_names:", hop_wrapper.feature_names)

    if hop_wrapper.error:
        st.error(f"Hop model load error: {hop_wrapper.error}")

    if model_error:
        st.error(f"Model prediction error: {model_error}")

    st.write("Yeast dataset columns:")
    st.json(list(yeast_df.columns))

    # show first few yeast rows
    st.dataframe(yeast_df.head(10), use_container_width=True)


# ----------------------------
# 7. RUN
# ----------------------------

if __name__ == "__main__":
    main()

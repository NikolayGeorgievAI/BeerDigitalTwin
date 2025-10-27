import os
import re
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# -------------------------------------------------
# TEXT NORMALIZATION / MATCHING HELPERS
# -------------------------------------------------

def _clean_name(name: str) -> str:
    """
    Normalize user-entered or display names to compare against model feature cols.
    """
    if name is None:
        return ""
    s = str(name).lower()
    s = s.replace("®", "").replace("™", "")
    s = re.sub(r"[^a-z0-9]", "", s)
    return s


def _best_feature_match(user_name: str, feature_cols: list, prefix: str):
    """
    Map a human-facing name like "Citra" or "Maris Otter"
    back to the model's feature column (e.g. 'hop_Citra®').

    We do crude fuzzy matching based on character overlap,
    restricted to columns that start with the given prefix.
    """
    cleaned_user = _clean_name(user_name)
    best_match = None
    best_score = -1

    for col in feature_cols:
        if not col.startswith(prefix):
            continue

        raw_label = col[len(prefix):]  # e.g. "Citra®" from "hop_Citra®"
        cleaned_label = _clean_name(raw_label)

        # quick filter: first 3 chars should appear somewhere
        if len(cleaned_user) >= 3 and cleaned_user[:3] not in cleaned_label:
            continue

        # score by char-overlap, cheap fuzzy
        common = set(cleaned_user) & set(cleaned_label)
        score = len(common)

        if score > best_score:
            best_score = score
            best_match = col

    return best_match


def _choices_from_features(feature_cols, prefix):
    """
    Turn model feature columns into nice dropdown display names.
    Example:
      ['hop_Citra®', 'hop_Mosaic', 'hop_Amarillo®'] ->
      ['Amarillo', 'Citra', 'Mosaic']
    """
    names = []
    for col in feature_cols:
        if not col.startswith(prefix):
            continue
        raw_label = col[len(prefix):]  # remove "hop_" / "malt_" / "yeast_"
        pretty = raw_label.replace("®", "").replace("™", "")
        pretty = pretty.replace("_", " ").strip()
        if pretty and pretty not in names:
            names.append(pretty)

    names = sorted(names, key=lambda s: s.lower())
    return names


# -------------------------------------------------
# LOAD MODELS
# -------------------------------------------------

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
# We discovered structure is:
# { "model": ..., "feature_cols": [...], "flavor_cols": [...] }
MALT_MODEL_PATH = os.path.join(ROOT_DIR, "malt_sensory_model.joblib")
malt_bundle = joblib.load(MALT_MODEL_PATH)

malt_model = malt_bundle["model"]
malt_feature_cols = malt_bundle["feature_cols"]      # inputs like 'malt_Maris Otter', etc.
malt_dims = malt_bundle["flavor_cols"]               # outputs like body/caramel/color/etc.

# --- Yeast model bundle ---
# We discovered similar structure:
# { "model": ..., "feature_cols": [...], "flavor_cols": [...] }
YEAST_MODEL_PATH = os.path.join(ROOT_DIR, "yeast_sensory_model.joblib")
yeast_bundle = joblib.load(YEAST_MODEL_PATH)

yeast_model = yeast_bundle["model"]
yeast_feature_cols = yeast_bundle["feature_cols"]    # inputs like 'yeast_London Ale III', etc.
yeast_dims = yeast_bundle["flavor_cols"]             # outputs like esters/attenuation/etc.

# Build dropdown option lists from the model feature columns
HOP_CHOICES = _choices_from_features(hop_feature_cols, prefix="hop_")
MALT_CHOICES = _choices_from_features(malt_feature_cols, prefix="malt_")
YEAST_CHOICES = _choices_from_features(yeast_feature_cols, prefix="yeast_")


# -------------------------------------------------
# FEATURE BUILDERS + PREDICTORS
# -------------------------------------------------

# ---------- HOPS ----------

def build_hop_features(user_hops):
    """
    user_hops: [ {"name": "Citra", "amt": 50}, {"name": "Mosaic", "amt": 30}, ... ]
    amt in grams.
    Returns 1-row DataFrame with columns hop_feature_cols.
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
    Returns dict of {dimension -> score} for hop aroma
    """
    X = build_hop_features(user_hops)
    y_pred = hop_model.predict(X)[0]
    return {dim: float(val) for dim, val in zip(hop_dims, y_pred)}


def advise_hops(user_hops, target_dim, trial_amt=20.0):
    """
    Brute force:
    1. Calculate base predicted profile.
    2. For each possible hop in the model, "add trial_amt grams".
    3. Check which hop boosts the chosen target_dim the most.
    """
    base_vec = predict_hop_profile(user_hops)
    base_score = base_vec.get(target_dim, 0.0)

    best_choice = None
    best_delta = -999.0
    best_new_profile = None

    for col in hop_feature_cols:
        if not col.startswith("hop_"):
            continue
       

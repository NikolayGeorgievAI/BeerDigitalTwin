import os
import re
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# -------------------------------------------------
# 0. Setup / load model
# -------------------------------------------------

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Load model bundle (model, feature_cols, aroma_dims) from same folder as this file
MODEL_PATH = os.path.join(os.path.dirname(__file__), "hop_aroma_model.joblib")

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
feature_cols = bundle["feature_cols"]
aroma_dims = [a for a in bundle["aroma_dims"] if str(a).lower() not in ("nan", "", "none")]

# -------------------------------------------------
# 1. Helper functions
# -------------------------------------------------

def _clean_hop_name(name: str) -> str:
    """
    Make hop names comparable:
    - lowercase
    - strip ®, ™ and punctuation
    - keep only a-z0-9
    """
    if name is None:
        return ""
    s = str(name)
    s = s.lower()
    s = s.replace("®", "").replace("™", "")
    s = re.sub(r"[^a-z0-9]", "", s)
    return s

def _map_user_hop_to_model_col(user_name: str, feature_cols: list):
    """
    Map user input hop name (e.g. 'Citra') to the proper model column
    (e.g. 'hop_Citra®' or 'hop_Citra').

    Basic fuzzy match:
    - normalize both sides
    - require first 3 chars overlap
    - pick max char-overlap score
    """
    cleaned_user = _clean_hop_name(user_name)
    best_match = None
    best_score = -1

    for col in feature_cols:
        if not col.startswith("hop_"):
            continue
        model_hop_raw = col[len("hop_"):]  # "Citra®"
        cleaned_model = _clean_hop_name(model_hop_raw)

        # skip if first 3 chars of user don't even appear
        if len(cleaned_user) >= 3 and cleaned_user[:3] not in cleaned_model:
            continue

        # tiny similarity score = size of char set intersection
        common = set(cleaned_user) & set(cleaned_model)
        score = len(common)

        if score > best_score:
            best_score = score
            best_match = col

    return best_match

def build_aligned_features(user_hops, feature_cols):
    """
    Convert a hop bill into a single-row DataFrame aligned with model's expected columns.
    user_hops = [ {"name": "Citra", "amt": 50}, {"name": "Mosaic", "amt": 30}, ... ]
    """
    totals = {col: 0.0 for col in feature_cols}

    for entry in user_hops:
        hop_name = entry.get("name", "")
        amt = float(entry.get("amt", 0.0))

        if amt <= 0:
            continue
        if not hop_name or str(hop_name).strip() in ["", "-"]:
            continue

        model_col = _map_user_hop_to_model_col(hop_name, feature_cols)
        if model_col is not None:
            totals[model_col] += amt
        # if model_col is None: unknown hop, ignore

    row_df = pd.DataFrame([totals], columns=feature_cols)
    return row_df

def predict_hop_profile(user_hops):
    """
    Use the trained RandomForest model to predict aroma vector.
    Returns dict: {aroma_dimension: score}
    """
    X = build_aligned_features(user_hops, feature_cols)
    y_pred = model.predict(X)[0]  # first row
    return {dim: float(val) for dim, val in zip(aroma_dims, y_pred)}

def advise_recipe(user_hops, target_dim, trial_amt=20.0):
    """
    Brute-force check: for each hop in the training set,
    pretend we add `trial_amt` grams of it and see which hop
    increases target_dim the most.

    Returns a dict with the suggestion.
    """
    if target_dim not in aroma_dims:
        raise ValueError(f"'{target_dim}' not in aroma_dims: {aroma_dims}")

    base_vec = predict_hop_profile(user_hops)
    base_score = base_vec.get(target_dim, 0.0)

    best_choice = None
    best_delta = -999.0
    best_new_profile = None

    for col in feature_cols:
        if not col.startswith("hop_"):
            continue

        candidate_hop_display = col[len("hop_"):]  # "Citra®", etc.

        trial_bill = user_hops + [{"name": candidate_hop_display, "amt": trial_amt}]
        trial_vec = predict_hop_profile(trial_bill)
        trial_score = trial_vec.get(target_dim, 0.0)

        delta = trial_score - base_score
        if delta > best_delta:
            best_delta = delta
            best_choice = candidate_hop_display
            best_new_profile = trial_vec

    return {
        "target_dim": target_dim,
        "current_score": base_score,
        "recommended_hop": best_choice,
        "addition_grams": trial_amt,
        "expected_improvement": best_delta,
        "new_profile": best_new_profile,
    }

def plot_aroma_radar(aroma_profile, title="Aroma Profile"):
    """
    Radar chart of the aroma vector.
    aroma_profile = dict {dimension -> value}
    """
    dims = list(aroma_profile.keys())
    vals = [aroma_profile[d] for d in dims]

    # close polygon
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


# -------------------------------------------------
# 2. Streamlit UI
# -------------------------------------------------

st.set_page_config(page_title="Beer Recipe Digital Twin", page_icon="🍺", layout="centered")

st.title("🍺 Beer Recipe Digital Twin")
st.caption("Predict hop aroma and get targeted hop addition advice. Prototype co-brewer.")
st.write(
    "1. Enter your hop bill (variety + grams).\n"
    "2. The model predicts flavor dimensions like citrus, pine, tropical fruit.\n"
    "3. Pick what you want more of.\n"
    "4. It recommends a single hop addition (and grams) to steer the beer."
)

st.markdown("---")

st.header("🧪 Input your hop bill")

col1, col2, col3 = st.columns([1,1,1])

with col1:
    hop1 = st.text_input("Hop 1 name", "Citra")
    hop2 = st.text_input("Hop 2 name", "Mosaic")

with col2:
    amt1 = st.number_input(
        "Hop 1 (g)",
        min_value=0.0,
        max_value=500.0,
        value=50.0,
        step=5.0,
    )
    amt2 = st.number_input(
        "Hop 2 (g)",
        min_value=0.0,
        max_value=500.0,
        value=30.0,
        step=5.0,
    )

with col3:
    st.write("")
    st.write("We'll predict the aroma balance of this bill, then tell you how to push it.")

user_hops = []
if hop1 and amt1 > 0:
    user_hops.append({"name": hop1, "amt": amt1})
if hop2 and amt2 > 0:
    user_hops.append({"name": hop2, "amt": amt2})

st.markdown("---")

predict_clicked = st.button("🔍 Predict Aroma Profile")

if predict_clicked:
    if not user_hops:
        st.error("Please enter at least one hop with a non-zero amount.")
    else:
        # 1. Predict
        profile = predict_hop_profile(user_hops)

        st.subheader("Predicted Aroma Profile")
        st.json(profile)

        fig = plot_aroma_radar(profile, title="Current Hop Bill")
        st.pyplot(fig)

        st.markdown("---")
        st.header("🎯 Get AI Advice")

        target = st.selectbox(
            "Which direction do you want to push the recipe?",
            aroma_dims,
            help="For example: more 'tropical fruit', more 'pine', more 'citrus'.",
        )

        trial_amt = st.slider(
            "Simulate late-addition/whirlpool hop (g):",
            min_value=5,
            max_value=60,
            value=20,
            step=5,
        )

        advise_clicked = st.button("🧠 Advise me")

        if advise_clicked:
            suggestion = advise_recipe(
                user_hops,
                target_dim=target,
                trial_amt=trial_amt,
            )

            st.success(
                f"To boost **{suggestion['target_dim']}**, "
                f"add {suggestion['addition_grams']} g of **{suggestion['recommended_hop']}**.\n\n"
                f"Expected improvement: +{suggestion['expected_improvement']:.3f}"
            )

            st.subheader("New predicted profile after that change")
            st.json(suggestion["new_profile"])

            fig2 = plot_aroma_radar(suggestion["new_profile"], title="Revised Hop Bill")
            st.pyplot(fig2)

            brewer_sentence = (
                f"Digital Twin Suggestion: To push this beer toward more "
                f"'{suggestion['target_dim']}', add {suggestion['addition_grams']} g of "
                f"{suggestion['recommended_hop']}. This addition is predicted to increase "
                f"'{suggestion['target_dim']}' by {suggestion['expected_improvement']:.3f} "
                f"while keeping the rest of the profile balanced."
            )

            st.markdown("**Brewer Note / LinkedIn Caption Draft:**")
            st.code(brewer_sentence, language="text")

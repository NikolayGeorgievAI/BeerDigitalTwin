import re
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import os

# -------------------------------------------------
# 0. Setup / load model
# -------------------------------------------------

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Resolve model path relative to this file so it works no matter how you launch Streamlit
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
    Normalize hop names so 'Amarillo', 'Amarillo®', 'amarillo ' all match.
    Remove symbols, lowercase, strip spaces.
    """
    if name is None:
        return ""
    s = str(name)
    s = s.lower()
    s = s.replace("®", "").replace("™", "")
    s = re.sub(r"[^a-z0-9]", "", s)  # keep only alphanumerics
    return s

def _map_user_hop_to_model_col(user_name: str, feature_cols: list):
    """
    Map the user-entered hop name ('Citra') to the closest known training column
    in feature_cols (which look like 'hop_Citra®', 'hop_Ahtanum™', etc.).

    Returns the exact feature column name to increment, or None if no good match.
    """
    cleaned_user = _clean_hop_name(user_name)
    best_match = None
    best_score = -1

    for col in feature_cols:
        if not col.startswith("hop_"):
            continue
        model_hop_raw = col[len("hop_"):]  # e.g. "Citra®"
        cleaned_model = _clean_hop_name(model_hop_raw)

        # quick similarity heuristic:
        # score = overlap of character sets
        common = set(cleaned_user) & set(cleaned_model)
        score = len(common)

        # tiny sanity check so we don't match random junk:
        # require first 3-4 chars of user to appear in model hop
        if len(cleaned_user) >= 3 and cleaned_user[:3] not in cleaned_model:
            continue

        if score > best_score:
            best_score = score
            best_match = col

    return best_match

def build_aligned_features(user_hops, feature_cols):
    """
    Turn user's hop bill (list of {name, amt}) into a 1-row DataFrame with columns
    exactly matching the model training columns.

    Unknown hops are silently ignored (for now).
    """
    totals = {col: 0.0 for col in feature_cols}

    for entry in user_hops:
        hop_name = entry.get("name", "")
        amt = float(entry.get("amt", 0.0))

        if amt <= 0:
            continue
        if hop_name is None:
            continue
        if str(hop_name).strip() in ["", "-"]:
            continue

        model_col = _map_user_hop_to_model_col(hop_name, feature_cols)
        if model_col is not None:
            totals[model_col] += amt
        # else: hop not recognized by model, ignore gracefully

    row_df = pd.DataFrame([totals], columns=feature_cols)
    return row_df

def predict_hop_profile(user_hops):
    """
    Predict aroma profile from a hop bill.

    Returns dict: {aroma_dim -> score ~0..1}
    """
    X = build_aligned_features(user_hops, feature_cols)
    y_pred = model.predict(X)[0]  # first row of prediction

    result = {dim: float(val) for dim, val in zip(aroma_dims, y_pred)}
    return result

def advise_recipe(user_hops, target_dim, trial_amt=20.0):
    """
    Recommend a hop addition (trial_amt grams) that best increases target_dim.

    Returns dict with:
    - target_dim
    - current_score
    - recommended_hop
    - addition_grams
    - expected_improvement
    - new_profile
    """
    if target_dim not in aroma_dims:
        raise ValueError(
            f"'{target_dim}' not in aroma_dims: {aroma_dims}"
        )

    base_vec = predict_hop_profile(user_hops)
    base_score = base_vec.get(target_dim, 0.0)

    best_choice = None
    best_delta = -999.0
    best_new_profile = None

    for col in feature_cols:
        if not col.startswith("hop_"):
            continue
        hop_display_name = col[len("hop_"):]  # e.g. "Citra®"

        trial_bill = user_hops + [{"name": hop_display_name, "amt": trial_amt}]
        trial_vec = predict_hop_profile(trial_bill)
        trial_score = trial_vec.get(target_dim, 0.0)

        delta = trial_score - base_score

        if delta > best_delta:
            best_delta = delta
            best_choice = hop_display_name
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
    Build a radar chart from a {dimension -> value} dictionary.
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
st.caption("AI-powered hop aroma prediction and recipe optimization")
st.write(
    "It predicts the aroma balance from your hop bill, then suggests which hop "
    "to add (and how many grams) to push you toward a target like citrus, pine, "
    "tropical fruit, etc."
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
    st.write("")  # spacing
    st.write("")  # spacing
    st.write("")  # spacing
    st.write("Enter the hops you used so far.\nWe'll predict the aroma balance.")

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
        # 1. Predict current aroma
        profile = predict_hop_profile(user_hops)

        st.subheader("Predicted Aroma Profile")
        st.json(profile)

        fig = plot_aroma_radar(profile, title="Current Hop Bill")
        st.pyplot(fig)

        # 2. Ask goal + simulate advice
        st.markdown("---")
        st.header("🎯 Get AI Advice")

        target = st.selectbox(
            "Which direction do you want to push the recipe?",
            aroma_dims,
            help="For example: more 'tropical fruit' or more 'pine'.",
        )

        trial_amt = st.slider(
            "Simulate adding (g) of ONE new hop:",
            min_value=5,
            max_value=60,
            value=20,
            step=5,
        )

        advise_clicked = st.button("🧠 Advise me")

        if advise_clicked:
            advice = advise_recipe(
                user_hops,
                target_dim=target,
                trial_amt=trial_amt,
            )

            st.success(
                f"To boost **{advice['target_dim']}**, "
                f"add {advice['addition_grams']} g of **{advice['recommended_hop']}**.\n\n"
                f"Expected improvement: +{advice['expected_improvement']:.3f}"
            )

            st.subheader("New predicted profile after that change")
            st.json(advice["new_profile"])

            fig2 = plot_aroma_radar(advice["new_profile"], title="Revised Hop Bill")
            st.pyplot(fig2)

            # also generate a marketing-style summary for screenshots / LinkedIn
            # (Not rendered unless you want to show it)
            brewer_sentence = (
                f"Digital Twin Suggestion: To push this beer toward more '{advice['target_dim']}', "
                f"add {advice['addition_grams']} g of {advice['recommended_hop']}. "
                f"This addition is predicted to increase '{advice['target_dim']}' "
                f"by {advice['expected_improvement']:.3f}, while keeping the rest of the profile balanced."
            )

            st.markdown("**Brewer Note / LinkedIn Caption Draft:**")
            st.code(brewer_sentence, language="text")

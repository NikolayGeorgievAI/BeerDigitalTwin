#########################
# Beer Recipe Digital Twin (fixed bin parsing version)
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
    Load any reference dataframes used for dropdowns, yeast info, etc.
    You mentioned having something like:
    - clean_yeast_df.pkl
    - clean_malt_df.pkl
    This function tries to load them.
    If filenames differ, update here.
    """
    try:
        yeast_df = pd.read_pickle("clean_yeast_df.pkl")
    except Exception as e:
        yeast_df = pd.DataFrame(columns=[
            "Name","Lab","Type","Form","Temp","Attenuation","Flocculation","Notes",
            "Temp_avg_C","Flocculation_num","Attenuation_pct",
            "fruity_esters","phenolic_spicy","clean_neutral",
            "dry_finish","malty","sulfur_note"
        ])
    try:
        malt_df = pd.read_pickle("clean_malt_df.pkl")
    except Exception as e:
        malt_df = pd.DataFrame(columns=["MaltName","SomeOtherCols"])

    return yeast_df, malt_df


@st.cache_resource(show_spinner=False)
def load_hop_model():
    """
    Load your hop aroma model from hop_aroma_model.joblib.
    The object may be:
      - a pipeline with .predict(...)
      - OR a dict/wrapper we created
    We'll wrap it in a HopModelWrapper to unify usage.
    """
    try:
        raw_obj = joblib.load("hop_aroma_model.joblib")
    except Exception as e:
        raw_obj = None
    return HopModelWrapper(raw_obj)


class HopModelWrapper:
    """
    We wrap whatever was loaded so we always have:
      - self.model: something with .predict(...)
      - self.feature_names: names of features it expects
    """

    def __init__(self, raw_obj):
        self.model = None
        self.feature_names = None

        if raw_obj is None:
            return

        # Case 1: raw_obj already is a pipeline / estimator with predict
        if hasattr(raw_obj, "predict"):
            self.model = raw_obj
            # scikit-learn >= 1.0 tends to have feature_names_in_
            fn = getattr(raw_obj, "feature_names_in_", None)
            if fn is not None:
                self.feature_names = list(fn)
            else:
                # fallback: None -> we'll guess later
                self.feature_names = None

        # Case 2: dict wrapper with 'model' and maybe 'feature_names'
        elif isinstance(raw_obj, dict):
            # attempt to extract
            mdl = raw_obj.get("model", None)
            feat = raw_obj.get("feature_names", None)

            if mdl is not None and hasattr(mdl, "predict"):
                self.model = mdl
            else:
                # not usable
                self.model = None

            # feature names might be in dict
            if feat is not None:
                self.feature_names = list(feat)
            else:
                # or the nested model might have feature_names_in_
                fn = getattr(mdl, "feature_names_in_", None)
                if fn is not None:
                    self.feature_names = list(fn)
                else:
                    self.feature_names = None

        # else unknown object type -> leave as None

    def is_ready(self) -> bool:
        return (self.model is not None) and hasattr(self.model, "predict")


#########################
# --- HOP FEATURE BIN LOGIC
#########################

def parse_feature_bin(bin_label: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Convert strings like "[0 - 100]" or "0 - 100" → (0.0, 100.0).
    Returns (None, None) if we can't parse.
    """
    if bin_label is None:
        return (None, None)

    txt = str(bin_label).strip()

    # Strip leading/trailing square brackets if they exist
    # e.g. "[0 - 100]" -> "0 - 100"
    if txt.startswith("[") and txt.endswith("]"):
        txt = txt[1:-1].strip()

    # Normalize weird dashes
    txt = txt.replace("–", "-").replace("—", "-")

    # Now we expect something like "0 - 100"
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
    True if EVERY feature name parses as a '[lo - hi]' style numeric bin.
    """
    if not feat_list:
        return False

    good = 0
    for f in feat_list:
        lo, hi = parse_feature_bin(f)
        if lo is not None and hi is not None:
            good += 1

    return good == len(feat_list)


def build_aligned_df_for_model(
    user_hops: List[Dict[str, float]],
    model_feature_names: Optional[List[str]]
) -> Tuple[pd.DataFrame, float, Dict[str, float], Dict[str, float]]:
    """
    Build the single-row DataFrame that we'll feed into the hop model.
    We support two modes:

    1. BIN MODE (model_feature_names look like "[0 - 100]", ...)
       We figure out total hop mass, pick which bin that mass falls in,
       and set that bin column=1.0 (others=0.0).

    2. SPARSE MODE (model_feature_names like "hop_Adeena", "hop_Amarillo", ...)
       We sum grams per hop name and align them to those feature columns.

    Returns:
      aligned_df, total_hop_mass, bin_debug, sparse_debug
    """

    total_hop_mass = sum(float(h.get("amt", 0.0) or 0.0) for h in user_hops)

    bin_hits_debug = {}
    sparse_debug = {}

    # --- BIN MODE ---
    if model_feature_names and feature_names_look_like_bins(model_feature_names):
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

    # --- SPARSE MODE ---
    else:
        aggregate = {}
        for entry in user_hops:
            hop_name = entry.get("name", "-")
            amt_g = float(entry.get("amt", 0.0) or 0.0)
            if hop_name == "-" or amt_g <= 0:
                continue
            col_name = f"hop_{hop_name}"
            aggregate[col_name] = aggregate.get(col_name, 0.0) + amt_g

        sparse_debug = aggregate.copy()

        # If the model gave us a known set of feature names, align to them:
        if model_feature_names:
            row_dict = {}
            for feat in model_feature_names:
                row_dict[feat] = float(aggregate.get(feat, 0.0))
        else:
            # fallback: just use whatever we saw
            row_dict = aggregate

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
    hop_wrapper: HopModelWrapper,
    aligned_df: pd.DataFrame
) -> Tuple[Dict[str, float], np.ndarray]:
    """
    Call hop_wrapper.model.predict() on aligned_df.
    Return a dict keyed by AROMA_COLUMNS, plus raw prediction array.
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
        # pred expected shape (1, 8)
        if isinstance(pred, (list, tuple)):
            pred = np.array(pred)
        if pred.ndim == 1:
            pred = pred.reshape(1, -1)
        raw_pred = pred

        # map into aroma_scores dict
        for i, aroma in enumerate(AROMA_COLUMNS):
            if i < pred.shape[1]:
                aroma_scores[aroma] = float(pred[0, i])

    except Exception:
        # just keep defaults
        pass

    return aroma_scores, raw_pred


#########################
# --- RADAR CHART
#########################

def make_radar(aroma_scores: Dict[str, float]):
    """
    Create a radar/spider chart figure from aroma_scores dict.
    We'll order them in AROMA_COLUMNS.
    """
    values = [aroma_scores[a] for a in AROMA_COLUMNS]
    labels = AROMA_COLUMNS

    # close the loop
    values += values[:1]
    labels_circ = labels + [labels[0]]

    angles = np.linspace(0, 2 * np.pi, len(values), endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])  # ensure closed

    fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))

    ax.plot(angles, values, color="#1f2a44", linewidth=2)
    ax.fill(angles, values, color="#1f2a44", alpha=0.2)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12)

    # radial grid styling
    ax.set_yticklabels([])  # hide radial labels if you want
    ax.grid(color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.spines["polar"].set_color("#222")
    ax.spines["polar"].set_linewidth(1.5)

    # show center value box = average intensity
    avg_val = float(np.mean(values[:-1])) if len(values) > 1 else 0.0
    ax.text(
        0.5 * np.pi,  # angle pointing "down" (pi/2 is straight up in polar coords)
        0,            # radius
        f"{avg_val:.2f}",
        ha="center",
        va="center",
        fontsize=14,
        bbox=dict(facecolor="#e6ebf5", edgecolor="#1f2a44", boxstyle="round,pad=0.4"),
        color="#1f2a44",
    )

    return fig


#########################
# --- STYLE/CHARACTER TEXT
#########################

def summarize_top_hop_notes(aroma_scores: Dict[str,float], top_n:int=2) -> List[Tuple[str,float]]:
    """
    Return top N aroma dimensions by score.
    """
    pairs = [(k, v) for k,v in aroma_scores.items()]
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs[:top_n]


def guess_malt_character(malt_entries: List[Dict[str,float]]) -> str:
    """
    Very placeholder: looks at chosen malts and returns a text descriptor.
    You can replace with a real malt model.
    """
    # naive: if any "BLACK" -> roasty, else bready
    chosen = [m["name"].upper() for m in malt_entries if m.get("name","-") != "-"]
    if any("BLACK" in x for x in chosen):
        return "roasty / dark malt"
    if any("CARA" in x or "CARAMEL" in x for x in chosen):
        return "sweet_caramel"
    return "bready"


def guess_yeast_character(yeast_name: str, yeast_df: pd.DataFrame) -> str:
    """
    Placeholder. We'll try to read columns in yeast_df that match the yeast_name.
    We'll produce something like "fruity_esters, clean_neutral".
    """
    if not yeast_name or yeast_name.strip() == "-":
        return "clean / neutral"

    row = yeast_df.loc[yeast_df["Name"]==yeast_name]
    if row.empty:
        return "clean / neutral"

    descs = []
    if "fruity_esters" in row.columns and row["fruity_esters"].iloc[0] == 1:
        descs.append("fruity_esters")
    if "clean_neutral" in row.columns and row["clean_neutral"].iloc[0] == 1:
        descs.append("clean_neutral")
    if "phenolic_spicy" in row.columns and row["phenolic_spicy"].iloc[0] == 1:
        descs.append("phenolic_spicy")

    if not descs:
        return "clean / neutral"
    return ", ".join(descs)


def guess_style_direction(aroma_scores: Dict[str,float], yeast_desc: str) -> str:
    """
    Toy logic: if fruity high + fruity_esters in yeast -> "Fruit-forward Ale"
    else "Experimental / Hybrid"
    """
    fruity_val = aroma_scores.get("fruity", 0.0)
    if fruity_val > 0.3 and "fruity" in yeast_desc:
        return "🍑 Fruit-forward Ale"
    return "🍻 Experimental / Hybrid"


#########################
# --- SIDEBAR INPUTS
#########################

def sidebar_inputs(yeast_df: pd.DataFrame, malt_df: pd.DataFrame):
    """
    Build sidebar with up to 4 hops (name + grams),
    3 malts (name + pct),
    and a yeast strain dropdown.
    Return them in structured form.
    """

    st.sidebar.header("Model Inputs")
    st.sidebar.markdown("### Hop Bill (g)")

    # We don't have a canonical full hop list in your shared code, so let's mock a short list.
    # You can replace this with the real hop list you've been using.
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

    # We'll build malt options from malt_df if a column with names is present,
    # else fallback to a short list.
    if "MaltName" in malt_df.columns and len(malt_df["MaltName"].dropna().unique())>0:
        malt_options = ["-"] + sorted(malt_df["MaltName"].dropna().unique().tolist())
    else:
        # fallback
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
        malt_pct = st.sidebar.number_input(
            f"Malt {i} %",
            min_value=0.0,
            max_value=100.0,
            value=0.0 if i>2 else (50.0 if i<=2 else 0.0),
            step=5.0,
            key=f"malt{i}_pct"
        )
        malt_entries.append({"name": malt_name, "pct": malt_pct})

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Yeast Strain")

    # Yeast dropdown from yeast_df["Name"] or fallback:
    yeast_options = ["-"]
    if "Name" in yeast_df.columns:
        yeast_names = yeast_df["Name"].dropna().unique().tolist()
        yeast_options += yeast_names

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
# --- MAIN APP LAYOUT
#########################

def main():
    st.title("🍺 Beer Recipe Digital Twin")
    st.write(
        "Predict hop aroma, malt character, and fermentation profile using trained ML models (work in progress)."
    )

    yeast_df, malt_df = load_reference_data()
    hop_wrapper = load_hop_model()

    # 1. Sidebar inputs
    hop_entries, malt_entries, yeast_choice, run_button = sidebar_inputs(yeast_df, malt_df)

    # 2. Build aligned_df for the hop model
    aligned_df, total_hop_mass, bin_debug, sparse_debug = build_aligned_df_for_model(
        hop_entries,
        hop_wrapper.feature_names
    )

    # 3. Predict aroma
    aroma_scores, raw_pred = predict_hop_aroma(hop_wrapper, aligned_df)

    # 4. Make radar chart
    fig = make_radar(aroma_scores)

    # 5. Build side descriptors
    top_notes = summarize_top_hop_notes(aroma_scores, top_n=2)
    malt_desc = guess_malt_character(malt_entries)
    yeast_desc = guess_yeast_character(yeast_choice, yeast_df)
    style_dir  = guess_style_direction(aroma_scores, yeast_desc)

    # 6. Display main chart + right panel
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
            if h["name"] != "-" and h["amt"]>0
        ]
        if used_hops:
            st.write(", ".join(used_hops))
        else:
            st.write("—")

    # 7. Debug info
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

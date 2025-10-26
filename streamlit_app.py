import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import streamlit as st


# ---------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Beer Recipe Digital Twin",
    page_icon="🍺",
    layout="wide"
)

# Nice clean typography spacing
st.markdown(
    """
    <style>
    /* tighten sidebar widgets vertically */
    section[data-testid="stSidebar"] .stSlider, 
    section[data-testid="stSidebar"] .stNumberInput,
    section[data-testid="stSidebar"] .stSelectbox,
    section[data-testid="stSidebar"] button {
        margin-bottom: .5rem !important;
    }

    /* make headers in main area tighter */
    h1, h2, h3, h4 {
        margin-top: .5rem;
        margin-bottom: .5rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------------------------
# LOAD MODELS + DATA (CACHED)
# ---------------------------------------------------------------------
@st.cache_resource
def load_models_and_data():
    """
    Load trained models and reference data exactly once per session.
    We assume the repo includes:
      - hop_aroma_model.joblib        (dict with keys model, feature_cols, aroma_dims)
      - malt_sensory_model.joblib     (dict with keys model, feature_cols, flavor_cols)
      - yeast_sensory_model.joblib    (dict with keys model, feature_cols, flavor_cols)
      - clean_malt_df.pkl             (malt reference table)
      - clean_yeast_df.pkl            (yeast reference table)
    """
    hop_bundle   = joblib.load("hop_aroma_model.joblib")
    malt_bundle  = joblib.load("malt_sensory_model.joblib")
    yeast_bundle = joblib.load("yeast_sensory_model.joblib")

    hop_model      = hop_bundle["model"]
    hop_features   = hop_bundle["feature_cols"]   # e.g. ["hop_Citra","hop_Mosaic",...]
    hop_dims       = hop_bundle["aroma_dims"]     # e.g. ["tropical","citrus","resinous",...]

    malt_model     = malt_bundle["model"]
    malt_features  = malt_bundle["feature_cols"]  # numeric chemistry columns
    malt_dims      = malt_bundle["flavor_cols"]   # predicted binary traits, e.g. "bready","caramel"

    yeast_model    = yeast_bundle["model"]
    yeast_features = yeast_bundle["feature_cols"] # e.g. ["Temp_avg_C","Flocculation_num","Attenuation_pct"]
    yeast_dims     = yeast_bundle["flavor_cols"]  # predicted yeast traits, e.g. "fruity_esters"

    malt_df  = pd.read_pickle("clean_malt_df.pkl")
    yeast_df = pd.read_pickle("clean_yeast

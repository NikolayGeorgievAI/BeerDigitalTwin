import streamlit as st
import matplotlib.pyplot as plt

# -------------------------------------------------
# TEMP STUBS / PLACEHOLDERS
# -------------------------------------------------

def get_all_hop_names_stub():
    # Eventually this will come from hop model feature_cols.
    # For now, hardcode a few so dropdowns work.
    return ["Citra", "Mosaic", "Simcoe", "Galaxy", "Sabro", "Idaho 7"]

def get_all_malts_stub():
    # Eventually from malt_df["PRODUCT NAME"]
    return [
        "FINEST MARIS OTTER® ALE MALT",
        "EXTRA PALE MARIS OTTER® MALT",
        "AMBER MALT",
        "CARAMEL/CRYSTAL MALT 60L",
        "FLOKED OATS",
    ]

def get_all_yeasts_stub():
    # Eventually from yeast_df["Name"]
    return [
        "Cooper Ale",
        "WLP001 California Ale",
        "Nottingham Ale",
        "US-05 Chico / American Ale",
    ]


def fake_prediction_summary(hop_bill, malt_selections, yeast_choice):
    """
    This TEMP function mimics the shape of the real prediction output
    without loading models. It just returns hard-coded values.
    """
    hop_out = {
        "citrus": 0.72,
        "tropical": 0.65,
        "resinous": 0.40,
        "floral": 0.32,
        "herbal": 0.28,
        "spicy": 0.20,
        "earthy": 0.18,
    }

    hop_top_notes = ["citrus (0.72)", "tropical (0.65)"]
    malt_traits = ["bready", "caramel"]
    yeast_traits = ["clean_neutral", "fruity_esters"]
    style_guess = "Hazy / NEIPA leaning"

    return {
        "hop_out": hop_out,
        "hop_top_notes": hop_top_notes,
        "malt_traits": malt_traits,
        "yeast_traits": yeast_traits,
        "style_guess": style_guess,
    }


def plot_hop_radar_stub(hop_profile, title="Hop Aroma Radar"):
    labels = list(hop_profile.keys())
    values = list(hop_profile.values())

    # close loop
    labels += [labels[0]]
    values += [values[0]]

    import numpy as np  # small import **inside** function for now
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)

    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, values, linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels)
    ax.set_title(title)
    ax.set_rlabel_position(0)
    return fig


# -------------------------------------------------
# STREAMLIT PAGE LAYOUT
# -------------------------------------------------

st.set_page_config(
    page_title="Beer Recipe Digital Twin",
    page_icon="🍺",
    layout="wide"
)

st.title("🍺 Beer Recipe Digital Twin")
st.caption("Predict hop aroma, malt character, and fermentation profile (demo mode).")

st.sidebar.header("Hop Bill")

all_hops = get_all_hop_names_stub()

hop1_name = st.sidebar.selectbox("Hop 1", all_hops, index=0, key="hop1_name")
hop1_amt  = st.sidebar.slider(f"{hop1_name} (g)", 0, 120, 40, 5, key="hop1_amt")

hop2_name = st.sidebar.selectbox("Hop 2", all_hops, index=min(1, len(all_hops)-1), key="hop2_name")
hop2_amt  = st.sidebar.slider(f"{hop2_name} (g)", 0, 120, 40, 5, key="hop2_amt")

hop3_name = st.sidebar.selectbox("Hop 3", all_hops, index=min(2, len(all_hops)-1), key="hop3_name")
hop3_amt  = st.sidebar.slider(f"{hop3_name} (g)", 0, 120, 40, 5, key="hop3_amt")

hop4_name = st.sidebar.selectbox("Hop 4", all_hops, index=min(3, len(all_hops)-1), key="hop4_name")
hop4_amt  = st.sidebar.slider(f"{hop4_name} (g)", 0, 120, 40, 5, key="hop4_amt")

hop_bill = {
    hop1_name: hop1_amt,
    hop2_name: hop2_amt,
    hop3_name: hop3_amt,
    hop4_name: hop4_amt,
}


st.sidebar.header("Malt Bill")

malt_options = get_all_malts_stub()

malt1_name = st.sidebar.selectbox("Malt 1", malt_options, index=0, key="malt1_name")
malt1_pct  = st.sidebar.number_input("Malt 1 %", min_value=0.0, max_value=100.0, value=70.0, step=1.0, key="malt1_pct")

malt2_name = st.sidebar.selectbox("Malt 2", malt_options, index=1 if len(malt_options) > 1 else 0, key="malt2_name")
malt2_pct  = st.sidebar.number_input("Malt 2 %", min_value=0.0, max_value=100.0, value=20.0, step=1.0, key="malt2_pct")

malt3_name = st.sidebar.selectbox("Malt 3", malt_options, index=2 if len(malt_options) > 2 else 0, key="malt3_name")
malt3_pct  = st.sidebar.number_input("Malt 3 %", min_value=0.0, max_value=100.0, value=10.0, step=1.0, key="malt3_pct")

malt_selections = [
    {"name": malt1_name, "pct": malt1_pct},
    {"name": malt2_name, "pct": malt2_pct},
    {"name": malt3_name, "pct": malt3_pct},
]


st.sidebar.header("Yeast")
yeast_options = get_all_yeasts_stub()
chosen_yeast = st.sidebar.selectbox("Yeast Strain", yeast_options)

run_button = st.sidebar.button("Predict Flavor 🧪")


if run_button:
    summary = fake_prediction_summary(hop_bill, malt_selections, chosen_yeast)

    hop_profile   = summary["hop_out"]
    hop_notes     = summary["hop_top_notes"]
    malt_traits   = summary["malt_traits"]
    yeast_traits  = summary["yeast_traits"]
    style_guess   = summary["style_guess"]

    col1, col2 = st.columns([1.3, 1])

    with col1:
        st.subheader("Predicted Hop Aroma (demo)")
        fig = plot_hop_radar_stub(hop_profile, title="Hop Aroma Radar")
        st.pyplot(fig)

        st.markdown("**Top hop notes:**")
        if hop_notes:
            for n in hop_notes:
                st.write("- ", n)
        else:
            st.write("_No dominant hop note_")

    with col2:
        st.subheader("Beer Aroma Advisor (demo)")
        st.markdown(f"**Malt character:** {', '.join(malt_traits) if malt_traits else 'None detected'}")
        st.markdown(f"**Yeast character:** {', '.join(yeast_traits) if yeast_traits else 'None detected'}")

        st.markdown("**Style direction:**")
        st.markdown(f"🧭 {style_guess}")

    with st.expander("Debug / Inputs"):
        st.json({
            "hop_bill (grams)": hop_bill,
            "malt_bill (%)": malt_selections,
            "yeast": chosen_yeast,
        })
else:
    st.info("👈 Build your hop bill, malt bill, choose yeast, then click **Predict Flavor 🧪** (demo mode).")

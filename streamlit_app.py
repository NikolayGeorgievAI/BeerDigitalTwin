def sidebar_inputs():
    st.sidebar.header("🧪 Model Inputs")

    st.sidebar.subheader("Hop Bill (g)")

    # Hop 1
    hop1_name = st.sidebar.selectbox(
        "Hop 1",
        ["-"] + AVAILABLE_HOPS,
        index=0,
        key="hop1_name"
    )
    hop1_amt  = st.sidebar.number_input(
        f"{hop1_name} (g)",
        min_value=0.0,
        max_value=500.0,
        value=0.0,
        step=5.0,
        key="hop1_amt"
    )

    # Hop 2
    hop2_name = st.sidebar.selectbox(
        "Hop 2",
        ["-"] + AVAILABLE_HOPS,
        index=0,
        key="hop2_name"
    )
    hop2_amt  = st.sidebar.number_input(
        f"{hop2_name} (g)",
        min_value=0.0,
        max_value=500.0,
        value=0.0,
        step=5.0,
        key="hop2_amt"
    )

    # Hop 3
    hop3_name = st.sidebar.selectbox(
        "Hop 3",
        ["-"] + AVAILABLE_HOPS,
        index=0,
        key="hop3_name"
    )
    hop3_amt  = st.sidebar.number_input(
        f"{hop3_name} (g)",
        min_value=0.0,
        max_value=500.0,
        value=0.0,
        step=5.0,
        key="hop3_amt"
    )

    # Hop 4
    hop4_name = st.sidebar.selectbox(
        "Hop 4",
        ["-"] + AVAILABLE_HOPS,
        index=0,
        key="hop4_name"
    )
    hop4_amt  = st.sidebar.number_input(
        f"{hop4_name} (g)",
        min_value=0.0,
        max_value=500.0,
        value=0.0,
        step=5.0,
        key="hop4_amt"
    )

    st.sidebar.subheader("Malt Bill")

    malt1_type = st.sidebar.text_input(
        "Malt 1",
        "AMBER MALT",
        key="malt1_type"
    )
    malt1_pct  = st.sidebar.number_input(
        "Malt 1 %",
        min_value=0.0,
        max_value=100.0,
        value=70.0,
        step=1.0,
        key="malt1_pct"
    )

    malt2_type = st.sidebar.text_input(
        "Malt 2",
        "BEST ALE MALT",
        key="malt2_type"
    )
    malt2_pct  = st.sidebar.number_input(
        "Malt 2 %",
        min_value=0.0,
        max_value=100.0,
        value=20.0,
        step=1.0,
        key="malt2_pct"
    )

    malt3_type = st.sidebar.text_input(
        "Malt 3",
        "BLACK MALT",
        key="malt3_type"
    )
    malt3_pct  = st.sidebar.number_input(
        "Malt 3 %",
        min_value=0.0,
        max_value=100.0,
        value=10.0,
        step=1.0,
        key="malt3_pct"
    )

    st.sidebar.subheader("Yeast Strain")

    if "name" in clean_yeast_df.columns:
        yeast_options = ["-"] + sorted(clean_yeast_df["name"].unique().tolist())
    else:
        yeast_options = ["-"]

    yeast_choice = st.sidebar.selectbox(
        "Select yeast",
        yeast_options,
        index=0,
        key="yeast_choice"
    )

    run_button = st.sidebar.button("Predict Flavor 🧪", key="run_button")

    hop_inputs = [
        (hop1_name, hop1_amt),
        (hop2_name, hop2_amt),
        (hop3_name, hop3_amt),
        (hop4_name, hop4_amt),
    ]

    malt_inputs = [
        (malt1_type, malt1_pct),
        (malt2_type, malt2_pct),
        (malt3_type, malt3_pct),
    ]

    return hop_inputs, malt_inputs, yeast_choice, run_button

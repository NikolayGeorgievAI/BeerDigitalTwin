run_button = st.sidebar.button("Predict Flavor 🧪")

if run_button:
    summary = summarize_beer(
        hop_bill,
        malt_selections,
        chosen_yeast,
        hop_model, hop_features, hop_dims,
        malt_model, malt_df, malt_features, malt_dims,
        yeast_model, yeast_df, yeast_features, yeast_dims,
    )

    hop_out      = summary["hop_out"]
    hop_notes    = summary["hop_top_notes"]
    malt_traits  = summary["malt_traits"]
    yeast_traits = summary["yeast_traits"]
    style_guess  = summary["style_guess"]

    left_col, right_col = st.columns([0.55, 0.45], vertical_alignment="top")

    with left_col:
        fig = make_spider_plot(hop_out)
        st.pyplot(fig, use_container_width=True)

    with right_col:
        st.markdown("### Top hop notes:")
        ...
else:
    st.info("👉 Build your hop bill, malt bill, choose a yeast strain, then click **Predict Flavor 🧪**.")

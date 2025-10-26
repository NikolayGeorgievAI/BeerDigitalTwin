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

    left_col, right_col = st.columns([0.6, 0.4], vertical_alignment="top")

    with left_col:
        fig = make_spider_plot(hop_out)
        st.pyplot(fig, use_container_width=True)

    with right_col:
        st.markdown("### Top hop notes:")
        if hop_notes:
            for n in hop_notes:
                st.write(f"- {n}")
        else:
            st.write("_No dominant hop note_")

        st.markdown("### Malt character:")
        if malt_traits:
            st.write(", ".join(malt_traits))
        else:
            st.write("None")

        st.markdown("### Yeast character:")
        if yeast_traits:
            st.write(", ".join(yeast_traits))
        else:
            st.write("None")

        st.markdown("### Style direction:")
        st.write(f"🧭 {style_guess}")

else:
    st.info("👉 Build your hop bill, malt bill, choose a yeast strain, then click **Predict Flavor 🧪**.")

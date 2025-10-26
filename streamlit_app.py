st.sidebar.markdown("### Hop Bill (g)")
all_hops = get_all_hops_from_features(hop_features)

def hop_input(slot_idx: int, default_index: int):
    idx = default_index if default_index < len(all_hops) else 0

    hop_name = st.sidebar.selectbox(
        f"Hop {slot_idx}",
        all_hops,
        index=idx,
        key=f"hop_select_{slot_idx}"
    )

    amt = st.sidebar.number_input(
        f"{hop_name} (g)",
        min_value=0.0,
        max_value=500.0,
        value=0.0,
        step=5.0,
        key=f"hop_amount_{slot_idx}"
    )

    return hop_name, amt

hop1_name, hop1_amt = hop_input(slot_idx=1, default_index=0)
hop2_name, hop2_amt = hop_input(slot_idx=2, default_index=1)
hop3_name, hop3_amt = hop_input(slot_idx=3, default_index=2)
hop4_name, hop4_amt = hop_input(slot_idx=4, default_index=3)

hop_bill = {
    hop1_name: hop1_amt,
    hop2_name: hop2_amt,
    hop3_name: hop3_amt,
    hop4_name: hop4_amt,
}

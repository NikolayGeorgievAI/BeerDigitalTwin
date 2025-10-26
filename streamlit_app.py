def make_spider_plot(hop_out_dict):
    """
    Draw a filled radar/spider chart of hop aroma intensities.

    Axes are in a fixed order so the shape is stable between runs.
    """
    axes_order = [
        "tropical",
        "citrus",
        "fruity",
        "resinous",
        "floral",
        "herbal",
        "spicy",
        "earthy",
    ]

    # get model outputs in that order, fallback 0.0
    vals = [float(hop_out_dict.get(dim, 0.0)) for dim in axes_order]

    # close the polygon so it wraps back around
    vals_closed = vals + [vals[0]]

    # angles for each axis
    angles = np.linspace(0, 2 * np.pi, len(axes_order), endpoint=False)
    angles_closed = np.concatenate([angles, [angles[0]]])

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.set_facecolor("#fafafa")

    # fill + outline
    ax.plot(
        angles_closed,
        vals_closed,
        color="#1f77b4",
        linewidth=2,
    )
    ax.fill(
        angles_closed,
        vals_closed,
        color="#1f77b4",
        alpha=0.25,
    )

    # annotate each vertex with its numeric value
    for ang, val in zip(angles, vals):
        ax.text(
            ang,
            val,
            f"{val:.2f}",
            ha="center",
            va="center",
            fontsize=9,
            bbox=dict(
                boxstyle="round,pad=0.2",
                fc="white",
                ec="#1f77b4",
                lw=1
            ),
        )

    # category labels around the outside
    ax.set_xticks(angles)
    ax.set_xticklabels(axes_order, fontsize=10)

    # style for radial grid
    ax.set_rlabel_position(0)
    ax.yaxis.grid(color="gray", linestyle="--", alpha=0.4)
    ax.xaxis.grid(color="gray", linestyle="--", alpha=0.4)

    # remove the radial tick labels (0.1, 0.2, etc.) for a cleaner spider look
    ax.set_yticklabels([])

    # pick a nice radial max
    max_val = max(max(vals), 0.5)  # avoid collapsing to nothing
    ax.set_ylim(0, max_val * 1.2)

    ax.set_title("Hop Aroma Radar", fontsize=20, fontweight="bold", pad=20)
    fig.tight_layout()
    return fig

def plot_hop_radar(hop_profile, title="Hop Aroma Radar"):
    """
    Render a proper spider-web radar chart:
    - dashed radial grid (rings + spokes)
    - filled polygon of the hop aroma intensities
    - numeric label at each vertex
    """

    import numpy as np
    import matplotlib.pyplot as plt

    # 1. Define the axes order around the radar.
    #    You can reorder these if you want a different clockwise layout.
    axes_order = [
        "fruity",
        "citrus",
        "tropical",
        "earthy",
        "spicy",
        "herbal",
        "floral",
        "resinous",
    ]

    # 2. Pull values from hop_profile in that order (fallback 0.0 if missing).
    vals = []
    for dim in axes_order:
        vals.append(float(hop_profile.get(dim, 0.0)) if hop_profile else 0.0)

    # Close polygon by repeating the first value/angle
    vals_closed = vals + [vals[0]]

    n_axes = len(axes_order)
    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False)
    angles_closed = list(angles) + [angles[0]]

    # 3. Pick a sensible radial max.
    vmax_candidate = max(vals) if len(vals) else 1.0
    vmax = max(1.0, vmax_candidate * 1.2)

    # 4. Build figure/axes with polar projection.
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    # ======================
    # Draw the "spider web"
    # ======================

    # -- dashed concentric rings
    num_rings = 6
    ring_radii = np.linspace(0, vmax, num_rings + 1)[1:]  # skip radius=0
    for r in ring_radii:
        ax.plot(
            angles_closed,
            [r] * len(angles_closed),
            color="gray",
            linewidth=0.8,
            linestyle="--",
            alpha=0.6,
        )

    # -- dashed spokes
    for ang in angles:
        ax.plot(
            [ang, ang],
            [0, vmax],
            color="gray",
            linewidth=0.8,
            linestyle="--",
            alpha=0.6,
        )

    # =====================================
    # Draw the actual predicted aroma blob
    # =====================================

    # outline
    ax.plot(
        angles_closed,
        vals_closed,
        color="#1f77b4",
        linewidth=2,
    )

    # fill
    ax.fill(
        angles_closed,
        vals_closed,
        color="#1f77b4",
        alpha=0.25,
    )

    # numeric label at each vertex
    for ang, val in zip(angles, vals):
        ax.text(
            ang,
            val,
            f"{val:.2f}",
            ha="center",
            va="center",
            fontsize=10,
            bbox=dict(
                boxstyle="round,pad=0.25",
                fc="white",
                ec="#1f77b4",
                lw=1,
            ),
        )

    # ======================
    # Cosmetics / labels
    # ======================

    # axis labels around the outside
    ax.set_xticks(angles)
    ax.set_xticklabels(axes_order, fontsize=12)

    # hide radial tick labels (we draw our own rings so they clutter)
    ax.set_yticklabels([])
    ax.set_ylim(0, vmax)

    # turn off Matplotlib's default polar grid (we did our own)
    ax.grid(False)

    # hide outer frame circle
    if hasattr(ax, "spines") and "polar" in ax.spines:
        ax.spines["polar"].set_visible(False)

    # title
    ax.set_title(title, fontsize=22, fontweight="bold", pad=20)

    fig.tight_layout()
    return fig

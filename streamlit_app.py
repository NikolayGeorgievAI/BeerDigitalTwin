def plot_hop_radar(hop_profile, title="Hop Aroma Radar"):
    """
    Radar/spider chart with polygonal grid ("proper spider web").
    - one axis per aroma dimension
    - polygon fill
    - polygon concentric grid rings
    - axis spokes
    """

    # Fallback if hop_profile is empty
    if not hop_profile:
        hop_profile = {
            "tropical": 0.0,
            "citrus": 0.0,
            "fruity": 0.0,
            "resinous": 0.0,
            "floral": 0.0,
            "herbal": 0.0,
            "earthy": 0.0
        }

    labels = list(hop_profile.keys())
    values = [float(hop_profile[k]) for k in labels]

    # Close the polygon
    values_closed = values + [values[0]]

    import numpy as np
    num_axes = len(labels)

    # Angles for each axis in radians
    angles = np.linspace(0, 2 * np.pi, num_axes, endpoint=False)
    angles_closed = list(angles) + [angles[0]]

    # We'll size the radial limit to max value
    vmax = max(1.0, max(values) * 1.2 if values else 1.0)

    # Figure: a little smaller than before so it fits nicely
    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))

    # --- Draw spider web grid (polygon rings) manually ---
    # Choose how many rings (e.g. 5)
    num_rings = 5
    ring_radii = np.linspace(0, vmax, num_rings + 1)[1:]  # skip 0, just draw >0

    for r in ring_radii:
        # same radius for every angle -> polygon
        ax.plot(angles_closed, [r]*len(angles_closed),
                color="gray", linewidth=0.8, linestyle="--", alpha=0.5)

    # Axis spokes (one line from center to max for each category)
    for ang in angles:
        ax.plot([ang, ang], [0, vmax],
                color="gray", linewidth=0.8, linestyle="--", alpha=0.5)

    # --- Plot the actual aroma polygon ---
    ax.plot(angles_closed, values_closed, linewidth=2, color="#1f77b4")
    ax.fill(angles_closed, values_closed, color="#1f77b4", alpha=0.25)

    # Numeric annotation at each vertex
    for ang, val in zip(angles, values):
        ax.text(
            ang,
            val,
            f"{val:.2f}",
            ha="center",
            va="center",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#1f77b4", lw=0.8),
        )

    # Category labels at the outer edge
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=10)

    # Remove default circular radial grid & labels (we drew our own web)
    ax.set_yticklabels([])
    ax.set_ylim(0, vmax)
    ax.grid(False)

    # Clean up frame aesthetics
    ax.spines["polar"].set_visible(False)

    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    fig.tight_layout()
    return fig

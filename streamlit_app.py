def plot_hop_radar(hop_profile, title="Hop Aroma Radar"):
    """
    Custom spider web / radar chart with polygon grid rings instead of circles.
    hop_profile is a dict like:
      {"tropical":0.8,"citrus":0.5,"fruity":0.4,"resinous":0.3,...}
    """

    # Fallback if empty
    if not hop_profile:
        hop_profile = {
            "tropical": 0.0,
            "citrus": 0.0,
            "fruity": 0.0,
            "resinous": 0.0,
            "floral": 0.0,
            "herbal": 0.0,
            "earthy": 0.0,
            "spicy": 0.0,
        }

    labels = list(hop_profile.keys())
    values = list(hop_profile.values())

    n_axes = len(labels)

    # Convert each axis angle (in radians)
    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False)

    # Close the polygon by repeating the first point at the end
    values_closed = np.array(values + [values[0]], dtype=float)
    angles_closed = np.concatenate([angles, [angles[0]]])

    # Decide the radial max for scaling
    # We'll just pick max of 1.0 or your max value * 1.2 so it fits nicely
    max_val = max(1.0, float(np.max(values)) * 1.2 if len(values) else 1.0)

    # We'll create polygon rings at (20%,40%,60%,80%,100%) of max_val
    ring_fracs = [0.2, 0.4, 0.6, 0.8, 1.0]

    # Pre-compute XY coords for each point at each ring
    def polar_to_xy(r, ang):
        return r * np.cos(ang), r * np.sin(ang)

    # Build figure
    fig, ax = plt.subplots(figsize=(6,6))

    # --- draw polygon rings (the web) ---
    for frac in ring_fracs:
        r = frac * max_val
        ring_x = []
        ring_y = []
        for ang in angles:
            x, y = polar_to_xy(r, ang)
            ring_x.append(x)
            ring_y.append(y)
        # close ring
        ring_x.append(ring_x[0])
        ring_y.append(ring_y[0])

        ax.plot(ring_x, ring_y, linestyle="--", color="gray", alpha=0.5, linewidth=1)

    # --- draw radial spokes ---
    for i, ang in enumerate(angles):
        x0, y0 = polar_to_xy(0, ang)
        x1, y1 = polar_to_xy(max_val, ang)
        ax.plot([x0, x1], [y0, y1], linestyle="--", color="gray", alpha=0.5, linewidth=1)

        # add category label slightly beyond max_val
        lx, ly = polar_to_xy(max_val * 1.08, ang)
        ax.text(
            lx, ly,
            labels[i],
            ha="center",
            va="center",
            fontsize=12
        )

    # --- plot the predicted polygon ---
    poly_x = []
    poly_y = []
    for ang, val in zip(angles, values):
        # scale val if your max_val is > 1
        # (the model output is already 0-1-ish, so usually no scaling needed;
        #  but just in case, we'll clamp it to max_val)
        r = min(val, max_val)
        x, y = polar_to_xy(r, ang)
        poly_x.append(x)
        poly_y.append(y)

    # close polygon
    poly_x.append(poly_x[0])
    poly_y.append(poly_y[0])

    ax.fill(
        poly_x,
        poly_y,
        color="#1f77b4",
        alpha=0.15,
        zorder=3
    )
    ax.plot(
        poly_x,
        poly_y,
        color="#1f77b4",
        linewidth=2,
        zorder=4
    )

    # --- numeric value labels at each vertex ---
    for ang, val in zip(angles, values):
        r = min(val, max_val)
        x, y = polar_to_xy(r, ang)
        ax.text(
            x, y,
            f"{val:.2f}",
            color="black",
            ha="center",
            va="center",
            fontsize=10,
            bbox=dict(
                boxstyle="round,pad=0.25",
                fc="white",
                ec="#1f77b4",
                lw=1
            ),
            zorder=5
        )

    # Clean up axes: no default ticks, equal aspect, tight
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-max_val*1.3, max_val*1.3)
    ax.set_ylim(-max_val*1.3, max_val*1.3)

    ax.set_title(
        title,
        fontsize=20,
        fontweight="bold",
        pad=20
    )

    fig.tight_layout()
    return fig

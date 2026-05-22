import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from pathlib import Path

# ---------------------------------------------------------------------
# Geometry parameters from the mesh-generator __main__ block
# ---------------------------------------------------------------------
flake_diameter = 0.20
pin_pitch = 0.032
r_HP = 0.011
r_f = 0.0072

theta_hex = np.pi / 6.0
y_flat = flake_diameter / 2.0
x_flat = np.tan(theta_hex) * y_flat

# ---------------------------------------------------------------------
# OpenMC-like centres used in the mesh-generator code
# ---------------------------------------------------------------------
x_HP0, y_HP0 = 0.0, 0.0

x_F1, y_F1 = 0.0, 1.0 * pin_pitch
x_F2, y_F2 = 0.0, 2.0 * pin_pitch

x_HP2 = np.sqrt(3.0) / 2.0 * pin_pitch
y_HP2 = 1.5 * pin_pitch

x_F3 = np.sqrt(3.0) / 2.0 * pin_pitch
y_F3 = 2.5 * pin_pitch

# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------
def line_points(p_start, p_end, n=2):
    p_start = np.asarray(p_start, dtype=float)
    p_end = np.asarray(p_end, dtype=float)

    t = np.linspace(0.0, 1.0, n)

    return (1 - t)[:, None] * p_start + t[:, None] * p_end


def arc_points(center, radius, angle_start_deg, angle_end_deg, n=80):
    angles = np.deg2rad(np.linspace(angle_start_deg, angle_end_deg, n))
    center = np.asarray(center, dtype=float)

    return np.column_stack(
        [
            center[0] + radius * np.cos(angles),
            center[1] + radius * np.sin(angles),
        ]
    )


def plot_segment(ax, pts, color, lw=4.0, zorder=4):
    ax.plot(
        pts[:, 0],
        pts[:, 1],
        color=color,
        lw=lw,
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=zorder,
    )


# ---------------------------------------------------------------------
# Boundary points corresponding to the moderator loop in the gmsh code
# ---------------------------------------------------------------------
p1 = np.array([0.0, 0.0])
p11 = np.array([0.0, y_flat])
p12 = np.array([x_flat, y_flat])

p2 = np.array([x_HP0, y_HP0 + r_HP])
p17 = np.array(
    [
        x_HP0 + r_HP * np.sin(theta_hex),
        y_HP0 + r_HP * np.cos(theta_hex),
    ]
)

p3 = np.array([x_F1, y_F1 - r_f])
p4 = np.array([x_F1, y_F1])
p5 = np.array([x_F1 + r_f, y_F1])
p6 = np.array([x_F1, y_F1 + r_f])

p7 = np.array([x_F2, y_F2 - r_f])
p8 = np.array([x_F2, y_F2])
p9 = np.array([x_F2 + r_f, y_F2])
p10 = np.array([x_F2, y_F2 + r_f])

p14 = np.array([x_HP2, y_HP2])
p13 = np.array(
    [
        x_HP2 + r_HP * np.sin(theta_hex),
        y_HP2 + r_HP * np.cos(theta_hex),
    ]
)
p15 = np.array([x_HP2 - r_HP, y_HP2])
p16 = np.array(
    [
        x_HP2 - r_HP * np.sin(theta_hex),
        y_HP2 - r_HP * np.cos(theta_hex),
    ]
)

# ---------------------------------------------------------------------
# Boundary-condition segments
# ---------------------------------------------------------------------

# Remaining outer/symmetry boundary, marked in grey
grey_segments = [
    line_points(p2, p3),
    line_points(p6, p7),
    line_points(p10, p11),
    line_points(p11, p12),
    line_points(p12, p13),
    line_points(p16, p17),
]

# Fuel pin boundary conditions, marked in red
fuel_segments = [
    arc_points(p4, r_f, -90, 0),
    arc_points(p4, r_f, 0, 90),
    arc_points(p8, r_f, -90, 0),
    arc_points(p8, r_f, 0, 90),
]

# Heat pipe boundary conditions, marked in blue
hp_segments = [
    arc_points(p14, r_HP, 60, 180),
    arc_points(p14, r_HP, 180, 240),
    arc_points(p1, r_HP, 60, 90),
]

# Full internal fuel-pin boundary
fuel3_circle = arc_points(
    [x_F3, y_F3],
    r_f,
    0,
    360,
    n=240,
)

# ---------------------------------------------------------------------
# Filled moderator silhouette polygon
# ---------------------------------------------------------------------
boundary_parts = [
    grey_segments[0],
    fuel_segments[0],
    fuel_segments[1],
    grey_segments[1],
    fuel_segments[2],
    fuel_segments[3],
    grey_segments[2],
    grey_segments[3],
    grey_segments[4],
    hp_segments[0],
    hp_segments[1],
    grey_segments[5],
    hp_segments[2],
]

boundary_poly = np.vstack(
    [
        part if i == 0 else part[1:]
        for i, part in enumerate(boundary_parts)
    ]
)

# ---------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.2, 9.0))

# Moderator domain fill
ax.fill(
    boundary_poly[:, 0],
    boundary_poly[:, 1],
    facecolor="#f3f3f3",
    edgecolor="none",
    zorder=1,
)

# White mask for the internal fuel-pin hole
ax.add_patch(
    Circle(
        (x_F3, y_F3),
        r_f,
        facecolor="white",
        edgecolor="none",
        zorder=2,
    )
)

# Draw remaining boundary in grey
for pts in grey_segments:
    plot_segment(
        ax,
        pts,
        color="#7a7a7a",
        lw=4.2,
    )

# Draw fuel pin boundary conditions in red
for pts in fuel_segments:
    plot_segment(
        ax,
        pts,
        color="#c62828",
        lw=4.8,
    )

plot_segment(
    ax,
    fuel3_circle,
    color="#c62828",
    lw=4.8,
)

# Draw heat pipe boundary conditions in blue
for pts in hp_segments:
    plot_segment(
        ax,
        pts,
        color="#1565c0",
        lw=4.8,
    )

# ---------------------------------------------------------------------
# Optional text labels
# Remove this block as well if you want a completely clean figure.
# ---------------------------------------------------------------------
ax.text(
    x_F1 + 1.35 * r_f,
    y_F1,
    "Fuel pin",
    ha="left",
    va="center",
    fontsize=12,
)

ax.text(
    x_F2 + 1.35 * r_f,
    y_F2,
    "Fuel pin",
    ha="left",
    va="center",
    fontsize=12,
)

ax.text(
    x_F3,
    y_F3 + 1.65 * r_f,
    "Fuel pin",
    ha="center",
    va="bottom",
    fontsize=12,
)

ax.text(
    p14[0] - 1.75 * r_HP,
    p14[1] - 0.15 * r_HP,
    "Heat pipe",
    ha="right",
    va="center",
    fontsize=12,
)

ax.text(
    p17[0] + 0.8 * r_HP,
    p17[1] - 0.15 * r_HP,
    "Heat pipe",
    ha="left",
    va="center",
    fontsize=12,
)

# ---------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------
ax.set_aspect("equal", adjustable="box")

pad = 0.006
ax.set_xlim(-pad, x_flat + pad)
ax.set_ylim(r_HP - pad, y_flat + pad)

# Remove axes completely
ax.axis("off")

fig.tight_layout(pad=0.0)

# ---------------------------------------------------------------------
# Save figure
# ---------------------------------------------------------------------
output_dir = Path(".")

png_path = output_dir / "moderator_boundary_conditions_silhouette_no_axes.png"
pdf_path = output_dir / "moderator_boundary_conditions_silhouette_no_axes.pdf"
svg_path = output_dir / "moderator_boundary_conditions_silhouette_no_axes.svg"

fig.savefig(
    png_path,
    dpi=400,
    bbox_inches="tight",
    pad_inches=0.02,
)

fig.savefig(
    pdf_path,
    bbox_inches="tight",
    pad_inches=0.02,
)

fig.savefig(
    svg_path,
    bbox_inches="tight",
    pad_inches=0.02,
)

plt.show()
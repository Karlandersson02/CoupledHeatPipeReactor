import matplotlib.pyplot as plt

def draw_block(
    ax,
    x0,
    y0,
    width,
    height,
    *,
    nx=0,
    ny=0,
    title=None,
    boundary_color="black",
    grid_color="red",
    linewidth_boundary=2.0,
    linewidth_grid=1.0,
):
    ax.plot(
        [x0, x0 + width, x0 + width, x0, x0],
        [y0, y0, y0 + height, y0 + height, y0],
        color=boundary_color,
        linewidth=linewidth_boundary,
    )

    if nx > 0:
        for i in range(1, nx):
            x = x0 + i * width / nx
            ax.plot(
                [x, x],
                [y0, y0 + height],
                color=grid_color,
                linewidth=linewidth_grid,
            )

    if ny > 0:
        for i in range(1, ny):
            y = y0 + i * height / ny
            ax.plot(
                [x0, x0 + width],
                [y, y],
                color=grid_color,
                linewidth=linewidth_grid,
            )

    if title:
        ax.text(
            x0 + width / 2,
            y0 + height + 0.12,
            title,
            ha="center",
            va="bottom",
            fontsize=14,
            color="black",
        )


def draw_heatpipe(
    ax,
    x0,
    y0,
    width,
    h_evap,
    h_adiabatic,
    h_cond,
    *,
    nx,
    ny_evap,
    ny_adiabatic,
    ny_cond,
    title="Heat Pipe",
    boundary_color="black",
    grid_color="red",
):
    h_total = h_evap + h_adiabatic + h_cond

    ax.plot(
        [x0, x0 + width, x0 + width, x0, x0],
        [y0, y0, y0 + h_total, y0 + h_total, y0],
        color=boundary_color,
        linewidth=2.0,
    )

    y1 = y0 + h_cond
    y2 = y1 + h_adiabatic

    ax.plot([x0, x0 + width], [y1, y1], color=boundary_color, linewidth=2.0)
    ax.plot([x0, x0 + width], [y2, y2], color=boundary_color, linewidth=2.0)

    if nx > 0:
        for i in range(1, nx):
            x = x0 + i * width / nx
            ax.plot([x, x], [y0, y0 + h_total], color=grid_color, linewidth=1.0)

    if ny_cond > 0:
        for i in range(1, ny_cond):
            y = y0 + i * h_cond / ny_cond
            ax.plot([x0, x0 + width], [y, y], color=grid_color, linewidth=1.0)

    if ny_adiabatic > 0:
        for i in range(1, ny_adiabatic):
            y = y1 + i * h_adiabatic / ny_adiabatic
            ax.plot([x0, x0 + width], [y, y], color=grid_color, linewidth=1.0)

    if ny_evap > 0:
        for i in range(1, ny_evap):
            y = y2 + i * h_evap / ny_evap
            ax.plot([x0, x0 + width], [y, y], color=grid_color, linewidth=1.0)

    ax.text(
        x0 + width / 2,
        y0 + h_total + 0.12,
        title,
        ha="center",
        va="bottom",
        fontsize=14,
        color="black",
    )

    ax.text(x0 + width + 0.10, y0 + h_cond / 2, "Condenser", va="center", fontsize=11, color="black")
    ax.text(x0 + width + 0.10, y1 + h_adiabatic / 2, "Adiabatic", va="center", fontsize=11, color="black")
    ax.text(x0 + width + 0.10, y2 + h_evap / 2, "Evaporator", va="center", fontsize=11, color="black")

    return y1, y2


def add_mesh_labels(ax, x0, y0, width, height, *, N_R=None, N_Z=None, left=True):
    if N_Z is not None:
        ax.text(
            x0 - 0.10 if left else x0 + width + 0.10,
            y0 + height / 2,
            f"$N_Z = {N_Z}$",
            rotation=90,
            ha="center",
            va="center",
            fontsize=11,
            color="black",
        )

    if N_R is not None:
        ax.text(
            x0 + width / 2,
            y0 - 0.15,
            f"$N_R = {N_R}$",
            ha="center",
            va="top",
            fontsize=11,
            color="black",
        )


def add_length_mark_vertical(ax, x, y0, y1, label, text_dx=0.06, color="black"):
    ax.annotate(
        "",
        xy=(x, y1),
        xytext=(x, y0),
        arrowprops=dict(arrowstyle="<->", color=color, lw=1.2),
    )
    ax.text(
        x + text_dx,
        0.5 * (y0 + y1),
        label,
        rotation=90,
        ha="left",
        va="center",
        fontsize=10,
        color=color,
    )


def add_length_mark_horizontal(ax, x0, x1, y, label, text_dy=0.05, color="black"):
    ax.annotate(
        "",
        xy=(x1, y),
        xytext=(x0, y),
        arrowprops=dict(arrowstyle="<->", color=color, lw=1.2),
    )
    ax.text(
        0.5 * (x0 + x1),
        y + text_dy,
        label,
        ha="center",
        va="bottom",
        fontsize=10,
        color=color,
    )


def add_heatpipe_section_counts(ax, x, y0, h_cond, h_adi, h_evap, *, N_cond, N_adiabatic, N_evap):
    ax.text(x, y0 + 0.5 * h_cond,       f"$N_{{cond}} = {N_cond}$",        va="center", ha="left", fontsize=10, color="black")
    ax.text(x, y0 + h_cond + 0.5 * h_adi, f"$N_{{adi}} = {N_adiabatic}$",  va="center", ha="left", fontsize=10, color="black")
    ax.text(x, y0 + h_cond + h_adi + 0.5 * h_evap, f"$N_{{evap}} = {N_evap}$", va="center", ha="left", fontsize=10, color="black")


def plot_reactor_schematic(cfg):
    fig, ax = plt.subplots(figsize=(13, 7))

    x_neu, w_neu = 0.0, 1.0
    x_fp,  w_fp  = 2.0, 1.25
    x_hp,  w_hp  = 4.6, 1.45

    hp_geom = cfg.HP.geometry
    hp_mesh = cfg.HP.mesh
    fp_geom = cfg.FP.geometry
    fp_mesh = cfg.FP.mesh
    n_mesh = cfg.N.mesh

    # Draw HP to axial proportion
    hp_total_plot_height = 4.2
    h_cond = hp_total_plot_height * hp_geom.l_cond / hp_geom.l_tot
    h_adi  = hp_total_plot_height * hp_geom.l_adiabatic / hp_geom.l_tot
    h_evap = hp_total_plot_height * hp_geom.l_evap / hp_geom.l_tot

    y_hp = -1.0
    y_evap_bottom = y_hp + h_cond + h_adi

    # Draw FP and neutronics aligned with evaporator
    fp_plot_height = h_evap
    neu_plot_height = h_evap
    y_fp = y_evap_bottom
    y_neu = y_evap_bottom

    draw_block(
        ax,
        x_neu,
        y_neu,
        w_neu,
        neu_plot_height,
        nx=0,
        ny=n_mesh.N_Z,
        title="Neutronics",
        boundary_color="black",
        grid_color="red",
    )
    add_mesh_labels(ax, x_neu, y_neu, w_neu, neu_plot_height, N_R=n_mesh.N_R, N_Z=n_mesh.N_Z)

    draw_block(
        ax,
        x_fp,
        y_fp,
        w_fp,
        fp_plot_height,
        nx=fp_mesh.N_R,
        ny=fp_mesh.N_Z,
        title="Fuel Pin",
        boundary_color="black",
        grid_color="red",
    )
    add_mesh_labels(ax, x_fp, y_fp, w_fp, fp_plot_height, N_R=fp_mesh.N_R, N_Z=fp_mesh.N_Z)

    draw_heatpipe(
        ax,
        x_hp,
        y_hp,
        w_hp,
        h_evap,
        h_adi,
        h_cond,
        nx=hp_mesh.N_R,
        ny_evap=hp_mesh.N_evap,
        ny_adiabatic=hp_mesh.N_adiabatic,
        ny_cond=hp_mesh.N_cond,
        title="Heat Pipe",
        boundary_color="black",
        grid_color="red",
    )
    add_mesh_labels(
        ax,
        x_hp,
        y_hp,
        w_hp,
        h_cond + h_adi + h_evap,
        N_R=hp_mesh.N_R,
        N_Z=hp_mesh.N_Z,
        left=True,
    )

    # Extra HP axial section counts
    add_heatpipe_section_counts(
        ax,
        x_hp - 1.05,
        y_hp,
        h_cond,
        h_adi,
        h_evap,
        N_cond=hp_mesh.N_cond,
        N_adiabatic=hp_mesh.N_adiabatic,
        N_evap=hp_mesh.N_evap,
    )

    # Axial lengths
    add_length_mark_vertical(
        ax,
        x_neu - 0.42,
        y_neu,
        y_neu + neu_plot_height,
        fr"$l = {n_mesh.l:.3g}$",
    )

    add_length_mark_vertical(
        ax,
        x_fp - 0.42,
        y_fp,
        y_fp + fp_plot_height,
        fr"$l = {fp_geom.l:.3g}$",
    )

    add_length_mark_vertical(
        ax,
        x_hp + w_hp + 0.55,
        y_hp,
        y_hp + h_cond,
        fr"$l_{{cond}} = {hp_geom.l_cond:.3g}$",
    )
    add_length_mark_vertical(
        ax,
        x_hp + w_hp + 0.95,
        y_hp + h_cond,
        y_hp + h_cond + h_adi,
        fr"$l_{{adi}} = {hp_geom.l_adiabatic:.3g}$",
    )
    add_length_mark_vertical(
        ax,
        x_hp + w_hp + 1.35,
        y_hp + h_cond + h_adi,
        y_hp + h_cond + h_adi + h_evap,
        fr"$l_{{evap}} = {hp_geom.l_evap:.3g}$",
    )

    # Radial lengths
    add_length_mark_horizontal(
        ax,
        x_fp,
        x_fp + w_fp,
        y_fp - 0.42,
        fr"$r = {fp_geom.r:.3g}$",
    )

    add_length_mark_horizontal(
        ax,
        x_hp,
        x_hp + w_hp,
        y_hp - 0.42,
        fr"$r_{{outer}} - r_{{vap}} = {hp_geom.r_outer - hp_geom.r_vapour:.3g}$",
    )

    ax.set_xlim(-0.9, 7.7)
    ax.set_ylim(-1.7, y_hp + hp_total_plot_height + 0.8)
    ax.axis("off")

    plt.tight_layout()
    plt.show()
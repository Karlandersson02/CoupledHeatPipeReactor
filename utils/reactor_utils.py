import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def plot_reactor_temperature_schematic_with_vapour(solver, solution_idx: int = -1):
    reactor = solver.components[solution_idx]
    ((T_solid, u_v, T_v), T_FP, (phi_n_g, k)) = solver.solutions[solution_idx]

    cfg = reactor.cfg_R
    hp_geom, fp_geom = cfg.HP.geometry, cfg.FP.geometry
    hp_mesh, fp_mesh = cfg.HP.mesh, cfg.FP.mesh

    fig, ax = plt.subplots(figsize=(13, 6))

    # -----------------------------
    # Scaling
    # -----------------------------
    hp_total_plot_height = 4.2
    axial_scale = hp_total_plot_height / hp_geom.l_tot

    radial_exaggeration = 250.0 * 2
    radial_scale = axial_scale * radial_exaggeration

    # -----------------------------
    # Layout positions (no gap)
    # -----------------------------
    x_fp = 0

    # Compute fuel pin width first
    w_fp = fp_geom.r * radial_scale

    # Place heat pipe directly next to fuel pin
    x_hp = x_fp + w_fp

    y_hp = -1.0

    # Sizes
    h_hp = hp_geom.l_tot * axial_scale
    h_fp = fp_geom.l * axial_scale

    w_fp = fp_geom.r * radial_scale
    w_hp_solid = (hp_geom.r_outer - hp_geom.r_vapour) * radial_scale

    # Vapour strip width
    vapour_width_factor = 0.55
    w_hp_vap = vapour_width_factor * w_hp_solid

    # HP regions
    h_cond = hp_geom.l_cond * axial_scale
    h_adi  = hp_geom.l_adiabatic * axial_scale

    y_fp = y_hp + h_cond + h_adi

    # -----------------------------
    # Mesh edges
    # -----------------------------
    x_fp_edges = np.linspace(x_fp, x_fp + w_fp, fp_mesh.N_R + 1)
    y_fp_edges = np.linspace(y_fp, y_fp + h_fp, fp_mesh.N_Z + 1)

    # Heat pipe solid
    x_hp_solid_edges = np.linspace(x_hp, x_hp + w_hp_solid, hp_mesh.N_R + 1)
    y_hp_edges = np.linspace(y_hp, y_hp + h_hp, hp_mesh.N_Z + 1)

    # Vapour strip
    x_hp_vap_edges = np.linspace(
        x_hp + w_hp_solid,
        x_hp + w_hp_solid + w_hp_vap,
        2,  # one single radial cell
    )

    # -----------------------------
    # Prepare fields
    # -----------------------------
    # Flip solid so the plotted orientation matches your earlier schematic
    T_hp_plot = T_solid[::-1, ::-1]

    # Turn 1D vapour temperature into a thin 2D strip
    T_vap_plot = T_v[::-1].reshape(-1, 1)

    # Fuel pin field
    pcm_fp = ax.pcolormesh(
        x_fp_edges,
        y_fp_edges,
        T_FP,
        cmap="hot",
        shading="flat",
    )

    # -----------------------------
    # Shared normalization (HP + vapour)
    # -----------------------------
    T_min = min(T_hp_plot.min(), T_vap_plot.min())
    T_max = max(T_hp_plot.max(), T_vap_plot.max())

    norm = mpl.colors.Normalize(vmin=T_min, vmax=T_max) # type: ignore
    cmap_hp = "viridis"

    # -----------------------------
    # Heat pipe solid
    # -----------------------------
    pcm_hp = ax.pcolormesh(
        x_hp_solid_edges,
        y_hp_edges,
        T_hp_plot,
        cmap=cmap_hp,
        norm=norm,
        shading="flat",
    )

    # -----------------------------
    # Vapour (same cmap + norm)
    # -----------------------------
    pcm_vap = ax.pcolormesh(
        x_hp_vap_edges,
        y_hp_edges,
        T_vap_plot,
        cmap=cmap_hp,
        norm=norm,
        shading="flat",
    )

    # -----------------------------
    # Boundaries
    # -----------------------------
    # Fuel pin boundary
    ax.plot(
        [x_fp, x_fp + w_fp, x_fp + w_fp, x_fp, x_fp],
        [y_fp, y_fp, y_fp + h_fp, y_fp + h_fp, y_fp],
        color="black",
        linewidth=1.5,
    )

    # Heat pipe solid boundary
    ax.plot(
        [x_hp, x_hp + w_hp_solid, x_hp + w_hp_solid, x_hp, x_hp],
        [y_hp, y_hp, y_hp + h_hp, y_hp + h_hp, y_hp],
        color="black",
        linewidth=1.5,
    )

    # Vapour boundary
    x_v0 = x_hp + w_hp_solid
    x_v1 = x_hp + w_hp_solid + w_hp_vap
    ax.plot(
        [x_v0, x_v1, x_v1, x_v0, x_v0],
        [y_hp, y_hp, y_hp + h_hp, y_hp + h_hp, y_hp],
        color="black",
        linewidth=1.5,
    )

    # Heat pipe region separators across both solid and vapour
    for y_sep in [y_hp + h_cond, y_hp + h_cond + h_adi]:
        ax.plot([x_hp, x_v1], [y_sep, y_sep], color="black", linewidth=1.0)

    # Divider between solid and vapour
    ax.plot([x_v0, x_v0], [y_hp, y_hp + h_hp], color="black", linewidth=1.0)

    # -----------------------------
    # Titles
    # -----------------------------
    ax.text(x_fp + w_fp / 2, y_fp + h_fp + 0.08, "Fuel Pin", ha="center", fontsize=12)
    ax.text(x_hp + w_hp_solid / 2, y_hp + h_hp + 0.08, "Heat Pipe Solid", ha="center", fontsize=12)
    ax.text(x_v0 + w_hp_vap / 2, y_hp + h_hp + 0.08, "Vapour", ha="center", fontsize=12)

    # -----------------------------
    # Colorbars
    # -----------------------------
    # Single shared colorbar for solid + vapour
    cbar_hp = fig.colorbar(pcm_hp, ax=ax, fraction=0.03, pad=0.04)
    cbar_hp.set_label("Heat Pipe + Vapour [K]")

    cbar_fp = fig.colorbar(pcm_fp, ax=ax, fraction=0.03, pad=0.04)
    cbar_fp.set_label("Fuel Pin [K]")

    # -----------------------------
    # Final styling
    # -----------------------------
    ax.set_xlim(1.0, x_v1 + 0.8)
    ax.set_ylim(-1.4, y_hp + h_hp + 0.4)
    ax.set_aspect("equal")
    ax.axis("off")

    fig.suptitle(
        rf"2D Temperature Fields | $k_{{eff}} = {k:.4f}$",
        fontsize=14
    )

    plt.tight_layout()
    plt.show()


def plot_reactor_solutions(solver, labels=None):
    reactors = solver.components

    plt.rcParams["font.size"] = 12

    fig, axs = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)

    colors = plt.cm.viridis(np.linspace(0, 1, len(reactors)))  # type: ignore

    for i in range(len(reactors)):
        reactor = reactors[i]
        cfg_R = reactor.cfg_R

        label = labels[i] if labels else f"Solution {i+1}"

        if type(reactor).__name__ == "Reactor":
            ((T_solid, u_v, T_v), T_FP, (phi_n_g, k)) = solver.solutions[i]  # type: ignore

            u_full = np.r_[0, u_v, 0]
            u_bar = (u_full[:-1] + u_full[1:]) / 2

            z_full = reactor.heatpipe.solid.Z
            z_evap = reactor.heatpipe.solid.Z[:cfg_R.HP.mesh.N_evap]

            axs[0, 1].plot(z_full, u_bar, color=colors[i], label=label)
            axs[0, 2].plot(z_full, T_v, color=colors[i], label=label)

        if type(reactor).__name__ == "IsoReactor":
            ((T_solid, T_v), T_FP, (phi_n_g, k)) = solver.solutions[i]  # type: ignore

            T_v_vec = np.repeat(np.array([T_v]), cfg_R.HP.mesh.N_Z)

            z_full = reactor.heat_pipe_thermal_model.Z
            z_evap = reactor.heat_pipe_thermal_model.Z[:cfg_R.HP.mesh.N_evap]

            axs[0, 2].plot(z_full, T_v_vec, color=colors[i], label=label)

        axs[0, 0].plot(z_full, T_solid[:, 0], color=colors[i], label=label)
        axs[1, 0].plot(z_evap, T_FP[:cfg_R.HP.mesh.N_evap, -1], color=colors[i], label=label)
        axs[1, 1].plot(z_evap, phi_n_g[:cfg_R.HP.mesh.N_evap, 0], color=colors[i], label=label)
        axs[1, 2].plot(z_evap, phi_n_g[:cfg_R.HP.mesh.N_evap, -1], color=colors[i], label=label)

    # Titles
    axs[0, 0].set_title("Solid Axial Temperature")
    axs[0, 1].set_title("Vapour Velocity") 
    axs[0, 2].set_title("Vapour Temperature")
    axs[1, 0].set_title("Fuel Axial Pin Temperature")
    axs[1, 1].set_title("Neutron Flux (Group 0)")
    axs[1, 2].set_title("Neutron Flux (Group -1)")

    # Axis labels
    for ax in axs[0, :]:
        ax.set_xlabel("Axial position (z)")
    for ax in axs[1, :]:
        ax.set_xlabel("Axial position (z)")

    axs[0, 0].set_ylabel("Temperature [K]")
    axs[0, 1].set_ylabel("Velocity")
    axs[0, 2].set_ylabel("Temperature [K]")
    axs[1, 0].set_ylabel("Temperature [K]")
    axs[1, 1].set_ylabel("Flux")
    axs[1, 2].set_ylabel("Flux")

    # Legends per subplot (deduplicated)
    for ax in axs.flat:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), fontsize=10)

    plt.show()
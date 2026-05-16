import numpy as np
import matplotlib.pyplot as plt

import data.dataclass as d_class

from typing import Sequence

def generate_config(data, N_R_HP: int, N_R_FP: int, N_Z: int):
    # Heat pipe config
    geom   = d_class.HeatpipeGeometry(**data["HeatPipe"]["geometry"])
    mesh   = d_class.HeatpipeMesh(N_R=N_R_HP, N_Z=N_Z)
    mat    = d_class.HeatpipeMaterial(**data["HeatPipe"]["material"])
    wick   = d_class.HeatpipeWick(**data["HeatPipe"]["wick"])
    bc     = d_class.HeatpipeBC(**data["HeatPipe"]["bc"])
    cfg_HP = d_class.HeatpipeConfig(geom, mesh, mat, wick, bc)

    # Fuel pin config
    geom_FP   = d_class.FuelPinGeometry(**data["FuelPin"]["geometry"])
    mesh_FP   = d_class.FuelPinMesh(N_R=N_R_FP, N_Z=9*N_Z//20)
    energy_FP = d_class.FuelPinEnergy(**data["FuelPin"]["energy"])
    mat_FP    = d_class.FuelPinMaterial(**data["FuelPin"]["material"])
    cfg_FP    = d_class.FuelPinConfig(geom_FP, mesh_FP, energy_FP, mat_FP)

    # Neutronics config 
    mesh_N = d_class.NeutronicsMesh(
        N_R = N_R_FP,
        N_Z = 9*N_Z//20,
        l   = data["FuelPin"]["geometry"]["l"]
    )
    energy = d_class.NeutronicsEnergy(
        N_G   = cfg_FP.energy.N_G,
        power = data["Reactor"]["power"]["thermal"] / data["Reactor"]["components"]["N_FP"]
    )
    cfg_N = d_class.NeutronicsConfig(mesh_N, energy)

    # Reactor config
    cfg_R = d_class.ReactorConfig(cfg_HP, cfg_FP, cfg_N)
    cfg_R = cfg_R.resolve_mesh()

    return cfg_R

def generate_config_seq(data, Ns: Sequence[Sequence[int]]):

    cfgs = []

    for (N_R_HP, N_R_FP, N_Z) in Ns:
        cfg = generate_config(data, N_R_HP, N_R_FP, N_Z)
        cfgs.append(cfg)

    return cfgs

def plot_reactor_solutions(solver):
    reactors = solver.components

    plt.rcParams["font.size"] = 12
    plt.rcParams["font.family"] = "Computer Modern"
    plt.rcParams["text.usetex"] = True

    fig, axs = plt.subplots(3, 2, figsize=(11, 11))

    k_effs = []
    Q_fracs = []

    colors = plt.cm.viridis(np.linspace(0, 1, len(reactors))) # type: ignore

    for i, reactor in enumerate(reactors):
        ((T_solid, T_vap), T_FP, (phi_ng_hat, k)) = solver.solutions[i]

        cfg_R = reactor.cfg_R

        N_evap = int(cfg_R.HP.mesh.N_evap)
        N_cond_start = int(cfg_R.HP.mesh.N_evap + cfg_R.HP.mesh.N_adiabatic)
        hp_wall_slice = slice(cfg_R.HP.mesh.N_R - cfg_R.HP.mesh.N_wall, cfg_R.HP.mesh.N_R)
        fp_clad_slice = slice(cfg_R.FP.mesh.N_fuel + cfg_R.FP.mesh.N_gap, cfg_R.FP.mesh.N_R)

        N_R_FP = cfg_R.FP.mesh.N_R
        N_R_HP = cfg_R.HP.mesh.N_R
        N_Z    = cfg_R.HP.mesh.N_Z

        # Condenser heat rejection
        T_edge_cond_HP = T_solid[N_cond_start:, -1]
        Q_out = np.sum(
            (T_edge_cond_HP - 300.0)
            * cfg_R.HP.material.h_cond
            * cfg_R.HP.geometry.r_outer
            * 2 * np.pi
            * cfg_R.HP.geometry.l_cond
            / cfg_R.HP.mesh.N_cond
        )
        Q_in = cfg_R.N.energy.power

        k_effs.append(k)
        Q_fracs.append(Q_out / Q_in)

        z_flux = reactor.neutron_flux_model.Z
        z_hp = reactor.heat_pipe_thermal_model.Z[:N_evap]
        z_fp = reactor.fuel_pin_thermal_model.Z[:N_evap] if hasattr(reactor.fuel_pin_thermal_model, "Z") else z_hp

        r_hp = reactor.heat_pipe_thermal_model.R
        r_fp = reactor.fuel_pin_thermal_model.R
        T_hp_wall_axial = np.mean(T_solid[:N_evap, hp_wall_slice], axis=1)
        T_fp_clad_axial = np.mean(T_FP[:, fp_clad_slice], axis=1)
        T_hp_radial_avg = np.mean(T_solid[:N_evap], axis=0)
        T_fp_radial_avg = np.mean(T_FP, axis=0)

        # Top row: neutron flux
        axs[0, 0].plot(z_flux, phi_ng_hat[:, 0], linewidth=2, color=colors[i])

        # Keep top-right empty or use it for another group if desired
        axs[0, 1].plot(z_flux, phi_ng_hat[:, -1], linewidth=2, color=colors[i])

        # Middle row: axial temperature distributions
        axs[1, 0].plot(z_hp, T_hp_wall_axial, linewidth=2, color=colors[i])

        axs[1, 1].plot(z_fp, T_fp_clad_axial, linewidth=2, color=colors[i])

        # Bottom row: radial temperature distributions
        axs[2, 0].plot(r_hp, T_hp_radial_avg, linewidth=2, color=colors[i])

        axs[2, 1].plot(r_fp, T_fp_radial_avg, linewidth=2, color=colors[i])

    k_eff_string = ", ".join(f"{k_eff:.3f}" for k_eff in k_effs)
    Q_string = ", ".join(f"{100 * Q_frac:.2f}\\%" for Q_frac in Q_fracs)

    fig.suptitle(
        "Reactor Profiles\n"
        + rf"$k_{{eff}} = [{k_eff_string}]$" + "\n"
        + rf"$Q_{{\mathrm{{frac}}}} = [{Q_string}]$",
        fontsize=14,
    )

    # Top row
    axs[0, 0].set_title("Axial Neutron Flux Profile (Group 1)")
    axs[0, 0].set_xlabel("Length [m]")
    axs[0, 0].set_ylabel("Flux")
    axs[0, 0].grid(True, alpha=0.4)

    axs[0, 1].set_title("Axial Neutron Flux Profile (Last Group)")
    axs[0, 1].set_xlabel("Length [m]")
    axs[0, 1].set_ylabel("Flux")
    axs[0, 1].grid(True, alpha=0.4)

    # Middle row
    axs[1, 0].set_title("Heat Pipe Wall Axial Average Temperature")
    axs[1, 0].set_xlabel("Length [m]")
    axs[1, 0].set_ylabel("Temperature [K]")
    axs[1, 0].grid(True, alpha=0.4)

    axs[1, 1].set_title("Fuel Pin Cladding Axial Average Temperature")
    axs[1, 1].set_xlabel("Length [m]")
    axs[1, 1].set_ylabel("Temperature [K]")
    axs[1, 1].grid(True, alpha=0.4)

    # Bottom row
    axs[2, 0].set_title("Heat Pipe Evaporator Radial Average Temperature")
    axs[2, 0].set_xlabel("Radius [m]")
    axs[2, 0].set_ylabel("Temperature [K]")
    axs[2, 0].grid(True, alpha=0.4)

    axs[2, 1].set_title("Fuel Pin Radial Average Temperature")
    axs[2, 1].set_xlabel("Radius [m]")
    axs[2, 1].set_ylabel("Temperature [K]")
    axs[2, 1].grid(True, alpha=0.4)

    plt.tight_layout(rect=(0, 0, 1, 0.94))
    plt.show()

def plot_reactor_temperature_schematic(solver, solution_idx: int = -1):
    reactor = solver.components[solution_idx]
    ((T_solid, T_vap), T_FP, (phi_n_g, k)) = solver.solutions[solution_idx]

    cfg = reactor.cfg_R
    hp_geom, fp_geom = cfg.HP.geometry, cfg.FP.geometry
    hp_mesh, fp_mesh = cfg.HP.mesh, cfg.FP.mesh

    fig, ax = plt.subplots(figsize=(12, 6))

    # -----------------------------
    # Layout positions
    # -----------------------------
    x_fp, x_hp = 2.0, 6.0
    y_hp = -1.0

    # -----------------------------
    # Scaling
    # -----------------------------
    hp_total_plot_height = 4.2
    axial_scale = hp_total_plot_height / hp_geom.l_tot

    radial_exaggeration = 250.0 * 2
    radial_scale = axial_scale * radial_exaggeration

    # Sizes
    h_hp = hp_geom.l_tot * axial_scale
    h_fp = fp_geom.l * axial_scale

    w_fp = fp_geom.r * radial_scale
    w_hp = (hp_geom.r_outer - hp_geom.r_vapour) * radial_scale

    # HP regions
    h_cond = hp_geom.l_cond * axial_scale
    h_adi  = hp_geom.l_adiabatic * axial_scale

    y_fp = y_hp + h_cond + h_adi

    # -----------------------------
    # Mesh edges
    # -----------------------------
    x_fp_edges = np.linspace(x_fp, x_fp + w_fp, fp_mesh.N_R + 1)
    y_fp_edges = np.linspace(y_fp, y_fp + h_fp, fp_mesh.N_Z + 1)

    x_hp_edges = np.linspace(x_hp, x_hp + w_hp, hp_mesh.N_R + 1)
    y_hp_edges = np.linspace(y_hp, y_hp + h_hp, hp_mesh.N_Z + 1)

    # -----------------------------
    # Plot fields
    # -----------------------------
    pcm_fp = ax.pcolormesh(
        x_fp_edges, y_fp_edges, T_FP,
        cmap="hot",
        shading="flat",
    )

    pcm_hp = ax.pcolormesh(
        x_hp_edges,
        y_hp_edges,
        T_solid[::-1, ::-1],   # <-- FIX
        cmap="viridis",
        shading="flat",
    )

    # -----------------------------
    # Boundaries
    # -----------------------------
    for (x0, y0, w, h) in [(x_fp, y_fp, w_fp, h_fp), (x_hp, y_hp, w_hp, h_hp)]:
        ax.plot(
            [x0, x0 + w, x0 + w, x0, x0],
            [y0, y0, y0 + h, y0 + h, y0],
            color="black",
            linewidth=1.5,
        )

    # Heat pipe region separators
    ax.plot([x_hp, x_hp + w_hp], [y_hp + h_cond, y_hp + h_cond], color="black", linewidth=1.0)
    ax.plot([x_hp, x_hp + w_hp], [y_hp + h_cond + h_adi, y_hp + h_cond + h_adi], color="black", linewidth=1.0)

    # -----------------------------
    # Titles
    # -----------------------------
    ax.text(x_fp + w_fp / 2, y_fp + h_fp + 0.08, "Fuel Pin", ha="center", fontsize=12)
    ax.text(x_hp + w_hp / 2, y_hp + h_hp + 0.08, "Heat Pipe", ha="center", fontsize=12)

    # -----------------------------
    # Colorbars (clean placement)
    # -----------------------------

    cbar_hp = fig.colorbar(pcm_hp, ax=ax, fraction=0.035, pad=0.08)
    cbar_hp.set_label("Heat Pipe [K]")

    cbar_fp = fig.colorbar(pcm_fp, ax=ax, fraction=0.035, pad=0.08)
    cbar_fp.set_label("Fuel Pin [K]")

    # -----------------------------
    # Final styling
    # -----------------------------
    ax.set_xlim(1.0, 8.0)
    ax.set_ylim(-1.4, y_hp + h_hp + 0.4)
    ax.set_aspect("equal")
    ax.axis("off")

    fig.suptitle(
        rf"2D Temperature Fields | $k_{{eff}} = {k:.4f}$",
        fontsize=14
    )

    plt.tight_layout()
    plt.show()

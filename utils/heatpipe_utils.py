import numpy as np
import matplotlib.pyplot as plt

import data.dataclass as d_class
import utils.sodium_properties as s_props

def generate_cfg(data, N_R, N_Z, Q):

    geom = d_class.HeatpipeGeometry(**data["geometry"])
    mesh = d_class.HeatpipeMesh(N_R=N_R, N_Z=N_Z)
    mat  = d_class.HeatpipeMaterial(**data["material"])
    mat.k_gap = mat.k_wick
    wick = d_class.HeatpipeWick(**data["wick"])
    bc   = d_class.HeatpipeBC(**data["bc"])
    bc.Q = Q

    cfg = d_class.HeatpipeConfig(geom, mesh, mat, wick, bc)
    cfg = cfg.resolve_mesh()

    return cfg

def generate_cfgs_seq(data, Ns, Qs):

    cfgs = []

    for i, ((N_R, N_Z), Q) in enumerate(zip(Ns, Qs)):
        cfg = generate_cfg(data, N_R, N_Z, Q)
        cfgs.append(cfg)
    
    return cfgs


def calculate_m(HP, T_HP, T_v):
    T_interface = T_HP[:, 0]

    h_vap = HP.cfg.material.h_vap
    surface_areas, _, _, _ = HP._initialize_discretization()
    surface_interface = surface_areas[:, 0, 1]

    qp = h_vap * surface_interface * (T_interface - T_v)
    qp[HP.cfg.mesh.N_evap:-HP.cfg.mesh.N_cond] = 0          # No adiabatic heat transfer
    h_lat = s_props.calculate_Na_h_fg(T_v)
    # h_lat = np.full_like(T_v, 4.182e6)

    dmdz = qp / h_lat
    delta_z = HP.cfg.geometry.l_tot / HP.cfg.mesh.N_Z
    m = np.cumsum(dmdz)

    return m

def plot_heatpipe_temperature_2d(T_HP, l_z, l_r):
    T_HP = np.asarray(T_HP)

    if T_HP.ndim != 2:
        raise ValueError(f"T_HP must be 2D, got shape {T_HP.shape}")

    n_z, n_r = T_HP.shape

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

    extent = [0.0, l_r, 0.0, l_z]

    im = ax.imshow(
        T_HP[::-1, ::-1],
        origin="lower",
        aspect="equal",
        extent=extent, # type: ignore
        interpolation="nearest",
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Temperature [K]")

    ax.set_title("Heat Pipe Temperature")
    ax.set_xlabel("Radius [m]")
    ax.set_ylabel("Axial Position [m]")

    plt.show()


def plot_heatpipe_solutions(solver, labels=None):
    components = solver.components

    fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=False)
    fig.subplots_adjust(top=0.80)
    fig.suptitle("Heat Pipe–Vapour Coupled Solution", fontsize=16, y=0.98)

    colors = plt.cm.viridis(np.linspace(0, 1, len(components)))[::-1] # type: ignore

    if labels is None:
        labels = [f"Solution {i}" for i in range(len(components))]

    for i, component in enumerate(components):
        T_HP, mach, T_v = solver.solutions[i]

        z = component.solid.Z
        m = calculate_m(component.solid, T_HP, T_v)

        axs[0, 0].plot(z, T_HP[:, 0], color=colors[i], label=labels[i])
        axs[0, 1].plot(z, T_v       , color=colors[i], label=labels[i])
        axs[1, 0].plot(z, mach      , color=colors[i], label=labels[i])
        axs[1, 1].plot(z, m         , color=colors[i], label=labels[i])

    # Titles
    axs[0, 0].set_title("Heat Pipe Temperature (interface)")
    axs[0, 1].set_title("Vapour Temperature")
    axs[1, 0].set_title("Vapour Mach Number")
    axs[1, 1].set_title("Mass Flow Rate")

    # Axis labels
    axs[0, 0].set_ylabel("Temperature [K]")
    axs[1, 0].set_ylabel("Mach [-]")

    axs[1, 0].set_xlabel("Axial position z [m]")
    axs[1, 1].set_xlabel("Axial position z [m]")

    axs[0, 1].set_ylabel("Temperature [K]")
    axs[1, 1].set_ylabel("Mass flow rate [kg/s]")

    # Grid for readability
    for ax in axs.flat:
        ax.grid(True, linestyle="--", alpha=0.5)

    # Legend (avoid duplicates by only showing one entry per solution)
    handles, labels = axs[0, 0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(
        unique.values(),
        unique.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),  # lower than suptitle
        ncol=4,
        frameon=False
    )

    plt.show()
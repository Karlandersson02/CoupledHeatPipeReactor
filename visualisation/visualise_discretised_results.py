import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def display_temperature_distribution(T, data):

    # -------------------------
    # Extract discretisation
    # -------------------------
    N_evap = data["N_evap"]
    N_adiabatic = data["N_adiabatic"]
    N_cond = data["N_cond"]

    N_wick = data["N_wick"]
    N_wall = data["N_wall"]

    l_evap = data["l_evap"]
    l_adiabatic = data["l_adiabatic"]
    l_cond = data["l_cond"]

    delta_wick = data["delta_wick"]
    delta_wall = data["delta_wall"]
    r_outer = data["r_outer"]

    # -------------------------
    # Derived sizes
    # -------------------------
    N_Z = N_evap + N_adiabatic + N_cond
    N_R = N_wick + N_wall

    # -------------------------
    # Reconstruct solid field
    # -------------------------
    T_solid = T[:-1].reshape((N_Z, N_R))
    T_vap = T[-1]

    # -------------------------
    # Axial grid (non-uniform)
    # -------------------------
    z_edges = np.concatenate([
        np.linspace(0, l_evap, N_evap + 1),
        np.linspace(l_evap, l_evap + l_adiabatic, N_adiabatic + 1)[1:],
        np.linspace(l_evap + l_adiabatic,
                    l_evap + l_adiabatic + l_cond,
                    N_cond + 1)[1:]
    ])

    z_total = l_evap + l_adiabatic + l_cond

    # -------------------------
    # Radial grid
    # Order: vapour (bottom) -> wick -> wall (top)
    # -------------------------

    r_vap_top = r_outer - delta_wall - delta_wick
    r_wick_top = r_outer - delta_wall

    r_edges = np.concatenate([
        np.linspace(r_vap_top, r_wick_top, N_wick + 1),
        np.linspace(r_wick_top, r_outer, N_wall + 1)[1:]
    ])

    # -------------------------
    # Plot solid
    # -------------------------
    fig, ax = plt.subplots()

    cmap = mpl.cm.get_cmap("viridis")

    norm = mpl.colors.Normalize(
        vmin=min(T_solid.min(), T_vap),
        vmax=max(T_solid.max(), T_vap)
    )

    pcm = ax.pcolormesh(
        z_edges,
        r_edges,
        T_solid.T,
        shading="auto",
        cmap=cmap,
        norm=norm
    )

    vap_color = cmap(norm(T_vap))

    ax.fill_between(
        [0, z_total],
        0,
        r_vap_top,
        color=vap_color
    )

    # -------------------------
    # Formatting
    # -------------------------
    ax.set_xlabel("Axial position z")
    ax.set_ylabel("Radial position r")
    ax.set_xlim(0, z_total)
    ax.set_ylim(0, r_outer)

    fig.colorbar(pcm, ax=ax, label="Temperature")

    plt.show()

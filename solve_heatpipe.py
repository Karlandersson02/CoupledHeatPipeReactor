import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from models.heat_network_model import solve_heatpipe_network_model_static
from models.heat_discretised_model import heatpipe_discretised


def print_heat_pipe_temperatures(T):
    # Index mapping (from your code)
    T_ev_wall = T[0]
    T_ev_wick = T[1]
    T_co_wick = T[2]
    T_co_wall = T[3]
    T_ad_wick = T[4]
    T_ad_wall = T[5]

    print("\n   EVAPORATOR     ADIABATIC      CONDENSER")
    print("┌──────────────┬──────────────┬──────────────┐")
    print(f"│   WICK       │    WICK      │    WICK      │")
    print(f"│ {T_ev_wick:6.2f} K     │ {T_ad_wick:6.2f} K     │ {T_co_wick:6.2f} K     │")
    print(f"│ {T_ev_wick - 273.15:6.2f} C     │ {T_ad_wick - 273.15:6.2f} C     │ {T_co_wick - 273.15:6.2f} C     │")
    print("├──────────────┼──────────────┼──────────────┤")
    print(f"│   WALL       │    WALL      │    WALL      │")
    print(f"│ {T_ev_wall:6.2f} K     │ {T_ad_wall:6.2f} K     │ {T_co_wall:6.2f} K     │")
    print(f"│ {T_ev_wall - 273.15:6.2f} C     │ {T_ad_wall - 273.15:6.2f} C     │ {T_co_wall - 273.15:6.2f} C     │")
    print("└──────────────┴──────────────┴──────────────┘\n")

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

def main(args):
    D_v = 0.014
    delta_wick = 0.001
    delta_wall = 0.001

    l_evap = 0.105
    l_adiabatic = 0.0525
    l_cond = 0.5425

    T_infc = 300
    Q = 770

    k = np.array([
        21.7,  # evap wall (radial)
        45.0,  # evap wick (radial)
        45.0,  # cond wick (radial)
        21.7,  # cond wall (radial)
        45.0,  # adiabatic wick (axial)
        21.7,  # adiabatic wall (axial)
        39.0,  # convection boundary (heat transfer coefficient)
    ])

    A = np.array([
        np.pi*(D_v + 2 * delta_wick + 2 * delta_wall) * l_evap,
        np.pi*(D_v + 2 * delta_wick) * l_evap,
        np.pi*(D_v + 2 * delta_wick) * l_cond,
        np.pi*(D_v + 2 * delta_wick + 2 * delta_wall) * l_cond,
        np.pi*((D_v/2 + delta_wick)**2 - (D_v/2)**2), 
        np.pi*((D_v/2 + delta_wick + delta_wall)**2 - (D_v/2 + delta_wick)**2), 
        np.pi*(D_v + 2*delta_wick + 2*delta_wall) * l_cond 
    ])

    lamb = np.array([
        delta_wall,   # radial
        delta_wick,   # radial
        delta_wick,   # radial
        delta_wall,   # radial
        l_adiabatic,  # axial
        l_adiabatic,  # axial
        -1.0,     # dummy; convection handled via k[6]*A[6]
    ])

    N_evap = 15
    # ///
    data_discretised = {
        "r_outer": D_v/2 + delta_wall + delta_wick,
        "delta_wick": delta_wick,
        "delta_wall": delta_wall,
        "l_evap": l_evap,
        "l_adiabatic": l_adiabatic,
        "l_cond": l_cond,
        "N_wick": 15,
        "N_wall": 15,
        "N_evap": N_evap,
        "N_adiabatic": 15,
        "N_cond": 150,
        "h_vap": 1e10,       # högt tal bara
        "h_cond": 39,
        "T_cond": T_infc,
        "k_wall": 45.0,
        "k_wick": 21.7,
        "Q": np.repeat(np.array([Q/N_evap]), N_evap)
    }
    # \\\

    if len(args) > 0 and args[0] == "network":
        T = solve_heatpipe_network_model_static(k, A, lamb, T_infc, Q)
        print_heat_pipe_temperatures(T)
        return 0
    elif len(args) > 0 and args[0] == "discretised":
        heatpipe = heatpipe_discretised(data_discretised)
        T = heatpipe.solve_heatpipe_discretised()
        # print(T)
        display_temperature_distribution(T, data_discretised)
        return 0
    else:
        raise ValueError("Invalid function argument")

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

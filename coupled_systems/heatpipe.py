import numpy as np

from models.heatpipe.solid_discretised_vapour_model import HeatpipeDiscretisedVapour
from models.heatpipe.vapour_discretised_model import VapourDiscretised
from models.component import Component

from data.dataclass import HeatpipeConfigResolved

import utils.sodium_properties as s_props

class Heatpipe(Component):

    def __init__(self, cfg: HeatpipeConfigResolved):
        self.cfg = cfg

        self.HP  = HeatpipeDiscretisedVapour(cfg)
        self.Vap = VapourDiscretised(cfg, -1)

        # N_HP, N_u, N_v
        self.N_HP = self.cfg.mesh.N_Z * self.cfg.mesh.N_R
        self.N_u  = self.cfg.mesh.N_Z - 1
        self.N_v  = self.cfg.mesh.N_Z

    def assemble(self):
        self.HP.assemble()
        self.Vap.assemble()

    def initial_guess(self):
        X_HP_lin = self.HP.linear_solve()

        T_HP0 = X_HP_lin[:-1]
        T_v0_scalar = X_HP_lin[-1]

        self.Vap.set_T_HP(T_HP0)

        T_v0 = np.full(self.N_v, T_v0_scalar, dtype=float)
        u0 = self.Vap._build_initial_velocity(T_v0)

        return np.r_[T_HP0, u0, T_v0]

    def get_residuals(self, X):
        X_HP, u, T_v = self.unpack(X)
        X_Vap = self.Vap.pack((u, T_v))

        self.HP.set_T_v(T_v)
        self.Vap.set_T_HP(X_HP)

        res_HP  = self.HP.get_residuals(X_HP)
        res_Vap = self.Vap.get_residuals(X_Vap)

        return np.r_[res_HP, res_Vap]

    def post_process(self, X):
        T_HP, u, T_v = self.unpack(X)
        T_HP = T_HP.reshape((self.cfg.mesh.N_Z, self.cfg.mesh.N_R))

        u_bar = self._interpolate_u(u)
        c_s = self._calculate_c_s(T_v)
        mach = u_bar / c_s

        return T_HP, mach, T_v

    def unpack(self, X):
        T_HP = X[:self.N_HP]
        u    = X[self.N_HP:(self.N_HP + self.N_u)]
        T_v  = X[-self.N_v:]
        return T_HP, u, T_v
    
    def pack(self, X_tuple):
        return np.r_[*X_tuple]
    
    def _calculate_c_s(self, T_v):
        gamma = 5 / 3
        c_s = np.sqrt(gamma * s_props.calculate_Na_pressure_v(T_v) / s_props.calculate_rho_cc(T_v))
        return c_s
    
    def _interpolate_u(self, u):
        u_full = np.r_[0, u, 0]
        u_bar = (u_full[:-1] + u_full[1:]) / 2
        return u_bar


def generate_cfg(data, N_R, N_Z, Q):

    geom = HeatpipeGeometry(**data["geometry"])
    mesh = HeatpipeMesh(N_R=N_R, N_Z=N_Z)
    mat  = HeatpipeMaterial(**data["material"])
    mat.k_gap = mat.k_wick
    wick = HeatpipeWick(**data["wick"])
    bc   = HeatpipeBC(**data["bc"])
    bc.Q = Q

    cfg = HeatpipeConfig(geom, mesh, mat, wick, bc)
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
        extent=extent,
        interpolation="nearest",
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Temperature [K]")

    ax.set_title("Heat Pipe Temperature")
    ax.set_xlabel("Radius [m]")
    ax.set_ylabel("Axial Position [m]")

    plt.show()


def plot_heatpipe_solutions(solver):
    components = solver.components

    fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=False)
    fig.subplots_adjust(top=0.80)
    fig.suptitle("Heat Pipe–Vapour Coupled Solution", fontsize=16, y=0.98)

    colors = plt.cm.viridis(np.linspace(0, 1, len(components)))[::-1]

    for i, component in enumerate(components):
        T_HP_iterative, u_iterative, T_v_iterative = solver.solutions[i]

        z = component.HP.Z
        m_iterative = calculate_m(component.HP, T_HP_iterative, T_v_iterative)

        axs[0, 0].plot(z, T_HP_iterative[:, 0], color=colors[i], label=f"Solution {i}")
        axs[0, 1].plot(z, T_v_iterative       , color=colors[i], label=f"Solution {i}")
        axs[1, 0].plot(z, u_iterative         , color=colors[i], label=f"Solution {i}")
        axs[1, 1].plot(z, m_iterative         , color=colors[i], label=f"Solution {i}")

    # Titles
    axs[0, 0].set_title("Heat Pipe Temperature (interface)")
    axs[0, 1].set_title("Vapour Temperature")
    axs[1, 0].set_title("Vapour Velocity")
    axs[1, 1].set_title("Mass Flow Rate")

    # Axis labels
    axs[0, 0].set_ylabel("Temperature [K]")
    axs[1, 0].set_ylabel("Velocity [-]")

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
    

if __name__ == "__main__":
    import json
    import matplotlib.pyplot as plt
    from data.dataclass import *
    from utils.solver import Solver

    with open("./data/vapour_data.json", "r") as f:
        data_guoju = json.load(f)
        data = data_guoju["data_guoju_1000"]

    Qs = [0.97e3]
    Ns = [[28, 50] for i in range(len(Qs))]
    cfgs = generate_cfgs_seq(data, Ns, Qs)

    heatpipes = [Heatpipe(cfg) for cfg in cfgs]

    solver = Solver([heatpipes[-1]])
    solver.fsolve()


    plot_heatpipe_solutions(solver)
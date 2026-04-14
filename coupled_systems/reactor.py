import numpy as np
import matplotlib.pyplot as plt

import json

from models.heatpipe.solid_discretised_model import HeatpipeDiscretised
from models.neutronics.axial_neutron_model import NeutronicsModel
from models.fuel_pin.fuel_pin_model import FuelPin
from models.moderator.mesh_conduction_model import ModeratorDiscretisedMesh
from models.moderator.triangle_mesh import UnstructuredMesh

from models.component import Component
from data.dataclass import *

class Reactor(Component):
    def __init__(self, cfg_R: ReactorConfigResolved):

        self.cfg_HP: HeatpipeConfigResolved = cfg_R.HP
        self.cfg_FP: FuelPinConfigResolved = cfg_R.FP
        self.cfg_N: NeutronicsConfigResolved = cfg_R.N

        self.heat_pipe_thermal_model = HeatpipeDiscretised(self.cfg_HP)
        self.fuel_pin_thermal_model = FuelPin(self.cfg_FP)
        self.neutron_flux_model = NeutronicsModel(self.cfg_N)
        
        self.moderator_eff_res = 0.03
        self.T_cond = 300.

        # heat transfer HP variables: N_R * N_Z + 1, heat transfer FP variables: N_R * N_Z + 1, neutron flux variables: N_Z + 1
        self.N_HP = self.cfg_HP.mesh.N_R * self.cfg_HP.mesh.N_Z + 1
        self.N_FP = self.cfg_N.mesh.N_R  * self.cfg_N.mesh.N_Z
        self.N_N  = self.cfg_N.mesh.N_Z  * self.cfg_N.energy.N_G + 1

        self.N_var = self.N_HP + self.N_FP + self.N_N

    def assemble(self):
        return
    
    def post_process(self, X):
        T_HP, T_FP, phi_ng_hat_and_k = self.unpack(X)
        T_HP *= self.T_cond
        T_FP *= self.T_cond
        
        T_solid    = T_HP[:-1].reshape((self.cfg_HP.mesh.N_Z, self.cfg_HP.mesh.N_R))
        T_vap      = T_HP[-1]
        T_FP       = T_FP.reshape((self.cfg_FP.mesh.N_Z, self.cfg_FP.mesh.N_R))
        phi_ng_hat = phi_ng_hat_and_k[:-1].reshape((self.cfg_N.mesh.N_Z, self.cfg_N.energy.N_G))
        k          = phi_ng_hat_and_k[-1]

        return ((T_solid, T_vap), T_FP, (phi_ng_hat, k))
    
    def unpack(self, X):
        T_HP             = X[:self.N_HP]
        T_FP             = X[self.N_HP:(self.N_HP + self.N_FP)]
        phi_ng_hat_and_k = X[-self.N_N:]
        return T_HP, T_FP, phi_ng_hat_and_k

    def pack(self, X_tuple):
        X = np.r_[*X_tuple]
        return X

    def initial_guess(self):
        X_initial = np.ones(self.N_var)
        return X_initial
    
    def get_residuals(self, X):
        T_HP = self.T_cond * X[:(self.cfg_HP.mesh.N_R * self.cfg_HP.mesh.N_Z + 1)]
        T_FP = self.T_cond * X[(self.cfg_HP.mesh.N_R * self.cfg_HP.mesh.N_Z + 1):((self.cfg_HP.mesh.N_R * self.cfg_HP.mesh.N_Z + 1) + self.cfg_FP.mesh.N_R * self.cfg_FP.mesh.N_Z)]
        phi_ng_hat = X[((self.cfg_HP.mesh.N_R * self.cfg_HP.mesh.N_Z + 1) + self.cfg_N.mesh.N_R * self.cfg_N.mesh.N_Z):]

        # self.fuel_pin_thermal_model.initialize_discretization()

        T_FP_ave = np.mean(T_FP.reshape(self.cfg_N.mesh.N_Z, self.cfg_N.mesh.N_R), axis=1)
        qr = self.calculate_qr(T_FP, phi_ng_hat[:-1]) 
        Q_HP, T_mod = self.calculate_HP_FP_boundary_cond(T_FP, T_HP)

        self.heat_pipe_thermal_model.cfg.bc.Q = Q_HP
        # self.heat_pipe_thermal_model.assemble()
        res_cond_HP = self.heat_pipe_thermal_model.get_residuals(T_HP)
        
        self.fuel_pin_thermal_model.qr = qr
        self.fuel_pin_thermal_model.T_mod = T_mod
        res_cond_FP = self.fuel_pin_thermal_model.get_residuals(T_FP)
        
        self.neutron_flux_model.T_FP = T_FP
        self.neutron_flux_model.T_M  = np.mean(T_mod)
        self.neutron_flux_model.T_HP = np.mean(T_HP[:self.cfg_HP.mesh.N_R * self.cfg_HP.mesh.N_evap], dtype=float)
        res_flux = self.neutron_flux_model.get_residuals(phi_ng_hat)

        return np.r_[res_cond_HP, res_cond_FP, res_flux]
    

    def calculate_qr(self, T_FP_ave, phi_ng_hat):
        _, _, _, Sigma_f, _, _, kappa = self.neutron_flux_model.get_material_data(T_FP_ave)

        qr_rel = np.sum(phi_ng_hat.reshape(self.cfg_N.mesh.N_Z, self.cfg_N.energy.N_G) * Sigma_f * kappa * self.fuel_pin_thermal_model.Delta_V, axis=1) # W / m

        power_rel = np.sum(qr_rel)

        return qr_rel * self.cfg_N.energy.power / (power_rel * self.cfg_FP.mesh.N_fuel)
            

    def calculate_HP_FP_boundary_cond(self, T_FP, T_HP):

        T_edge_FP = T_FP[self.cfg_FP.mesh.N_R - 1::self.cfg_FP.mesh.N_R]
        T_edge_HP = T_HP[self.cfg_HP.mesh.N_R - 1:(self.cfg_HP.mesh.N_R * self.cfg_HP.mesh.N_evap):self.cfg_HP.mesh.N_R]

        Delta_z = self.cfg_HP.geometry.l_evap / self.cfg_HP.mesh.N_evap
        R_seg = self.moderator_eff_res / Delta_z
        Q_HP = (T_edge_FP - T_edge_HP) / R_seg

        T_mod = T_edge_FP - Q_HP / (self.cfg_FP.material.h_mod * 2 * self.cfg_FP.geometry.r * np.pi * self.cfg_FP.geometry.l / self.cfg_FP.mesh.N_Z)

        return Q_HP, T_mod

    
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


def plot_reactor_schematic(cfg: ReactorConfigResolved):
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


if __name__ == "__main__":

    with open("./data/test_data.json", "r") as f:
        data = json.load(f)
    
    # Mesh dimensions
    N_R_HP, N_R_FP, N_Z = 21, 21, 51

    # Heat pipe config
    geom   = HeatpipeGeometry(**data["HeatPipe"]["geometry"])
    mesh   = HeatpipeMesh(N_R=N_R_HP, N_Z=N_Z)
    mat    = HeatpipeMaterial(**data["HeatPipe"]["material"])
    wick   = HeatpipeWick(**data["HeatPipe"]["wick"])
    bc     = HeatpipeBC(**data["HeatPipe"]["bc"])
    cfg_HP = HeatpipeConfig(geom, mesh, mat, wick, bc)

    # Fuel pin config
    geom_FP   = FuelPinGeometry(**data["FuelPin"]["geometry"])
    mesh_FP   = FuelPinMesh(N_R=N_R_FP, N_Z=17)
    energy_FP = FuelPinEnergy(**data["FuelPin"]["energy"])
    mat_FP    = FuelPinMaterial(**data["FuelPin"]["material"])
    cfg_FP    = FuelPinConfig(geom_FP, mesh_FP, energy_FP, mat_FP)

    # Neutronics config
    mesh_N = NeutronicsMesh(
        N_R = N_R_FP,
        N_Z = 17,
        l   = data["FuelPin"]["geometry"]["l"]
    )
    energy = NeutronicsEnergy(
        N_G   = cfg_FP.energy.N_G,
        power = data["Reactor"]["power"]["thermal"] / data["Reactor"]["components"]["N_FP"]
    )
    cfg_N = NeutronicsConfig(mesh_N, energy)

    cfg_R1 = ReactorConfig(cfg_HP, cfg_FP, cfg_N)
    cfg_R1 = cfg_R1.resolve_geometry()



    # Mesh dimensions
    N_R_HP, N_R_FP, N_Z = 21, 21, 51

    # Heat pipe config
    geom   = HeatpipeGeometry(**data["HeatPipe"]["geometry"])
    mesh   = HeatpipeMesh(N_R=N_R_HP, N_Z=N_Z)
    mat    = HeatpipeMaterial(**data["HeatPipe"]["material"])
    wick   = HeatpipeWick(**data["HeatPipe"]["wick"])
    bc     = HeatpipeBC(**data["HeatPipe"]["bc"])
    cfg_HP = HeatpipeConfig(geom, mesh, mat, wick, bc)

    # Fuel pin config
    geom_FP   = FuelPinGeometry(**data["FuelPin"]["geometry"])
    mesh_FP   = FuelPinMesh(N_R=N_R_FP, N_Z=17)
    energy_FP = FuelPinEnergy(**data["FuelPin"]["energy"])
    mat_FP    = FuelPinMaterial(**data["FuelPin"]["material"])
    cfg_FP    = FuelPinConfig(geom_FP, mesh_FP, energy_FP, mat_FP)

    # Neutronics config
    mesh_N = NeutronicsMesh(
        N_R = N_R_FP,
        N_Z = 17,
        l   = data["FuelPin"]["geometry"]["l"]
    )
    energy = NeutronicsEnergy(
        N_G   = cfg_FP.energy.N_G,
        power = data["Reactor"]["power"]["thermal"] / data["Reactor"]["components"]["N_FP"]
    )
    cfg_N = NeutronicsConfig(mesh_N, energy)




    cfg_R2 = ReactorConfig(cfg_HP, cfg_FP, cfg_N)
    cfg_R2 = cfg_R2.resolve_geometry()

    # plot_reactor_schematic(cfg_R2)

    # Reactor ------------------
    reactor1 = Reactor(cfg_R1)
    reactor2 = Reactor(cfg_R2)

    from utils.solver import Solver

    solver = Solver([reactor1, reactor2], iterate=False)
    solver.fsolve()

    ((T_solid1, T_vap1), T_FP1, (phi_ng_hat1, k1)), ((T_solid2, T_vap2), T_FP2, (phi_ng_hat2, k2)) = solver.solutions

    # Calculating Q_out for both solutions
    cfg_HP = cfg_R2.HP
    cfg_FP = cfg_R2.FP
    cfg_N  = cfg_R2.N

    T_edge_cond_HP1 = T_solid1[(cfg_HP.mesh.N_evap + cfg_HP.mesh.N_adiabatic):, -1]
    Q_out1 = np.sum(
        (T_edge_cond_HP1 - 300.0)
        * cfg_HP.material.h_cond
        * cfg_HP.geometry.r_outer
        * 2 * np.pi
        * cfg_HP.geometry.l_cond
        / cfg_HP.mesh.N_cond
    )

    T_edge_cond_HP2 = T_solid2[(cfg_HP.mesh.N_evap + cfg_HP.mesh.N_adiabatic):, -1]
    Q_out2 = np.sum(
        (T_edge_cond_HP2 - 300.0)
        * cfg_HP.material.h_cond
        * cfg_HP.geometry.r_outer
        * 2 * np.pi
        * cfg_HP.geometry.l_cond
        / cfg_HP.mesh.N_cond
    )

    r_hp2 = reactor2.heat_pipe_thermal_model.R[1::2]
    r_fp2 = reactor2.fuel_pin_thermal_model.R[1::2]
    z_flux2 = reactor2.neutron_flux_model.Z

    r_fp1 = reactor1.fuel_pin_thermal_model.R[1::2]
    z_flux1 = reactor1.neutron_flux_model.Z
    r_hp1 = reactor1.heat_pipe_thermal_model.R[1::2]

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(
        f"Reactor Profiles\n"
        f"Solution 1: $k_{{eff}}={k1:.4f}$, $Q_{{in}}={cfg_N.energy.power:.0f}$ W, $Q_{{out}}={Q_out1:.0f}$ W\n"
        f"Solution 2: $k_{{eff}}={k2:.4f}$, $Q_{{in}}={cfg_N.energy.power:.0f}$ W, $Q_{{out}}={Q_out2:.0f}$ W",
        fontsize=14
    )

    # --- Top left: Neutronics ---
    axs[0, 0].plot(z_flux1, phi_ng_hat1[:, 0], linewidth=2, label="Solution 1")
    axs[0, 0].plot(z_flux2, phi_ng_hat2[:, 0], linewidth=2, label="Solution 2")
    axs[0, 0].set_title("Axial Neutron Flux Profile")
    axs[0, 0].set_xlabel("Length [m]")
    axs[0, 0].set_ylabel("Flux")
    axs[0, 0].grid(True)
    axs[0, 0].legend()

    # --- Top right: Heat pipe evaporator ---
    axs[0, 1].plot(r_hp1, T_solid1[0], linewidth=2, label="Solution 1")
    axs[0, 1].plot(r_hp2, T_solid2[0], linewidth=2, label="Solution 2")
    axs[0, 1].set_title("Heat Pipe Evaporator Radial Temperature")
    axs[0, 1].set_xlabel("Radius [m]")
    axs[0, 1].set_ylabel("Temperature [K]")
    axs[0, 1].grid(True)
    axs[0, 1].legend()

    # --- Bottom left: Fuel pin ---
    axs[1, 0].plot(r_fp1, T_FP1[0], linewidth=2, label="Solution 1")
    axs[1, 0].plot(r_fp2, T_FP2[0], linewidth=2, label="Solution 2")
    axs[1, 0].set_title("Fuel Pin Radial Temperature Profile")
    axs[1, 0].set_xlabel("Radius [m]")
    axs[1, 0].set_ylabel("Temperature [K]")
    axs[1, 0].grid(True)
    axs[1, 0].legend()

    # --- Bottom right: Heat pipe condenser ---
    axs[1, 1].plot(r_hp1, T_solid1[-1], linewidth=2, label="Solution 1")
    axs[1, 1].plot(r_hp2, T_solid2[-1], linewidth=2, label="Solution 2")
    axs[1, 1].set_title("Heat Pipe Condenser Radial Temperature")
    axs[1, 1].set_xlabel("Radius [m]")
    axs[1, 1].set_ylabel("Temperature [K]")
    axs[1, 1].grid(True)
    axs[1, 1].legend()

    plt.tight_layout(rect=(0, 0, 1, 0.92))
    plt.show()
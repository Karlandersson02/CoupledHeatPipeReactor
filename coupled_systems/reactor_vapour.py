import numpy as np

from models.component import Component

from coupled_systems.heatpipe import Heatpipe
from models.neutronics.axial_neutron_model import NeutronicsModel
from models.fuel_pin.fuel_pin_model import FuelPin

from data.dataclass import ReactorConfigResolved, HeatpipeConfigResolved, FuelPinConfigResolved, NeutronicsConfigResolved

class VapourReactor(Component):

    def __init__(self, cfg_R: ReactorConfigResolved):

        self.cfg_R = cfg_R

        self.cfg_HP: HeatpipeConfigResolved = cfg_R.HP
        self.cfg_FP: FuelPinConfigResolved = cfg_R.FP
        self.cfg_N: NeutronicsConfigResolved = cfg_R.N

        self.heatpipe = Heatpipe(self.cfg_HP)
        self.fuel_pin_thermal_model = FuelPin(self.cfg_FP)
        self.neutron_flux_model = NeutronicsModel(self.cfg_N)
        
        self.moderator_eff_res = 0.00807
        self.T_cond = 300.
        self.T_vap_ref = 2000

        # heat transfer HP variables: N_R * N_Z + 2 * N_Z - 1, heat transfer FP variables: N_R * N_Z + 1, neutron flux variables: N_Z + 1

        self.N_T_HP = self.cfg_HP.mesh.N_R * self.cfg_HP.mesh.N_Z
        self.N_u_v  = self.cfg_HP.mesh.N_Z - 1
        self.N_T_v  = self.cfg_HP.mesh.N_Z
        self.N_HP   = self.N_T_HP + self.N_u_v + self.N_T_v

        self.N_FP   = self.cfg_N.mesh.N_R  * self.cfg_N.mesh.N_Z
        self.N_N    = self.cfg_N.mesh.N_Z  * self.cfg_N.energy.N_G + 1

        self.N_var = self.N_HP + self.N_FP + self.N_N

    def set_variable_k(self, cond: bool):
        self.heatpipe.variable_k = cond
        self.fuel_pin_thermal_model.variable_k  = cond

    def initial_guess(self):
        X_initial = np.ones(self.N_var)
        return X_initial

    def assemble(self):
        return
    
    def post_process(self, X):
        X_HP, T_FP, phi_ng_hat_and_k = self.unpack(X)

        T_HP       = X_HP[:self.N_T_HP]
        u_v        = X_HP[self.N_T_HP:(self.N_T_HP + self.N_u_v)]
        T_v        = X_HP[(self.N_T_HP + self.N_u_v):]
        T_solid    = T_HP.reshape((self.cfg_HP.mesh.N_Z, self.cfg_HP.mesh.N_R))
        T_FP       = T_FP.reshape((self.cfg_FP.mesh.N_Z, self.cfg_FP.mesh.N_R))
        phi_ng_hat = phi_ng_hat_and_k[:-1]
        k          = phi_ng_hat_and_k[-1]

        T_HP *= self.T_cond
        T_v  *= self.T_vap_ref
        T_FP *= self.T_cond
        
        # power normalisation
        _, _, _, Sigma_f, _, _, kappa = self.neutron_flux_model.get_material_data(T_FP)
        phi_n_g_hat = phi_ng_hat.reshape((self.cfg_N.mesh.N_Z, self.cfg_N.energy.N_G))
        power_density = kappa * Sigma_f * phi_n_g_hat
        power = np.sum(power_density) * self.cfg_N.mesh.cross_sectional_area * self.cfg_N.mesh.delta_Z
        phi_n_g = phi_n_g_hat * self.cfg_N.energy.power / power

        return ((T_solid, u_v, T_v), T_FP, (phi_n_g, k))
    
    def unpack(self, X):
        X_HP             = X[:self.N_HP].copy()
        T_FP             = X[self.N_HP:(self.N_HP + self.N_FP)].copy()
        phi_ng_hat_and_k = X[-self.N_N:].copy()
        return X_HP, T_FP, phi_ng_hat_and_k

    def pack(self, X_tuple):
        X = np.r_[*X_tuple]
        return X
    
    def get_residuals(self, X):
        X_HP, T_FP, phi_ng_hat_and_k = self.unpack(X)

        T_HP        = X_HP[:self.N_T_HP]
        u_v         = X_HP[self.N_T_HP:(self.N_T_HP + self.N_u_v)]
        T_v         = X_HP[(self.N_T_HP + self.N_u_v):self.N_HP]
        phi_ng_hat  = phi_ng_hat_and_k[:-1]

        T_HP *= self.T_cond # modifies X_HP as well
        u_v  *= 10
        T_v  *= self.T_vap_ref
        T_FP *= self.T_cond

        Q_HP, T_mod = self.calculate_HP_FP_boundary_cond(T_FP, T_HP)
        qr = self.calculate_qr(T_FP, phi_ng_hat)
        
        T_HP_ave  = np.mean(T_HP[:self.cfg_HP.mesh.N_R * self.cfg_HP.mesh.N_evap], dtype=float)
        T_FP_ave  = np.mean(T_FP.reshape(self.cfg_N.mesh.N_Z, self.cfg_N.mesh.N_R), axis=1)
        T_mod_ave = np.mean(T_mod)

        self.heatpipe.cfg.bc.Q = Q_HP
        res_cond_HP = self.heatpipe.get_residuals(X_HP)
        
        self.fuel_pin_thermal_model.qr = qr
        self.fuel_pin_thermal_model.T_mod = T_mod
        res_cond_FP = self.fuel_pin_thermal_model.get_residuals(T_FP)
        
        self.neutron_flux_model.T_FP = T_FP
        self.neutron_flux_model.T_M  = np.mean(T_mod)
        self.neutron_flux_model.T_HP = T_HP_ave
        res_flux = self.neutron_flux_model.get_residuals(phi_ng_hat_and_k)

        return np.r_[res_cond_HP, res_cond_FP, res_flux]
    

    def calculate_qr(self, T_FP, phi_ng_hat):
        _, _, _, Sigma_f, _, _, kappa = self.neutron_flux_model.get_material_data(T_FP)

        qr_rel = np.sum(phi_ng_hat.reshape(self.cfg_N.mesh.N_Z, self.cfg_N.energy.N_G) * Sigma_f * kappa * self.fuel_pin_thermal_model.Delta_V, axis=1) # W
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

    norm = mpl.colors.Normalize(vmin=T_min, vmax=T_max)
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




if __name__ == "__main__":
    import json
    import matplotlib.pyplot as plt
    from data.dataclass import *
    from utils.solver import Solver

    with open("./data/reactor_data.json", "r") as f:
        data = json.load(f)

    N_R_HP, N_R_FP, N_Z = 15, 65, 20

    # Heat pipe config
    geom   = HeatpipeGeometry(**data["HeatPipe"]["geometry"])
    mesh   = HeatpipeMesh(N_R=N_R_HP, N_Z=N_Z)
    mat    = HeatpipeMaterial(**data["HeatPipe"]["material"])
    wick   = HeatpipeWick(**data["HeatPipe"]["wick"])
    bc     = HeatpipeBC(**data["HeatPipe"]["bc"])
    cfg_HP = HeatpipeConfig(geom, mesh, mat, wick, bc)

    # Fuel pin config
    geom_FP   = FuelPinGeometry(**data["FuelPin"]["geometry"])
    mesh_FP   = FuelPinMesh(N_R=N_R_FP, N_Z=9*N_Z//20)
    energy_FP = FuelPinEnergy(**data["FuelPin"]["energy"])
    mat_FP    = FuelPinMaterial(**data["FuelPin"]["material"])
    cfg_FP    = FuelPinConfig(geom_FP, mesh_FP, energy_FP, mat_FP)

    # Neutronics config
    mesh_N = NeutronicsMesh(
        N_R = N_R_FP,
        N_Z = 9*N_Z//20,
        l   = data["FuelPin"]["geometry"]["l"]
    )
    energy = NeutronicsEnergy(
        N_G   = cfg_FP.energy.N_G,
        power = data["Reactor"]["power"]["thermal"] / data["Reactor"]["components"]["N_FP"]
    )
    cfg_N = NeutronicsConfig(mesh_N, energy)

    # Reactor config
    cfg_R = ReactorConfig(cfg_HP, cfg_FP, cfg_N)
    cfg_R = cfg_R.resolve_geometry()

    vapour_reactor = VapourReactor(cfg_R)

    solver = Solver([vapour_reactor])
    solver.fsolve()

    # plot_reactor_temperature_schematic_with_vapour(solver)

    ((T_solid, u_v, T_v), T_FP, (phi_n_g, k)) = solver.solution

    u_full = np.r_[0, u_v, 0]
    u_bar  = (u_full[:-1] + u_full[1:]) / 2

    z_full = vapour_reactor.heatpipe.HP.Z
    z_evap = vapour_reactor.heatpipe.HP.Z[:cfg_R.HP.mesh.N_evap]

    fig, axs = plt.subplots(2, 3, figsize=(16, 9))

    axs[0, 0].plot(z_full, T_solid[:, 0])
    axs[0, 1].plot(z_full, u_bar)
    axs[0, 2].plot(z_full, T_v)

    axs[1, 0].plot(z_evap, T_FP[:cfg_R.HP.mesh.N_evap, -1])
    axs[1, 1].plot(z_evap, phi_n_g[:cfg_R.HP.mesh.N_evap, 0])
    axs[1, 2].plot(z_evap, phi_n_g[:cfg_R.HP.mesh.N_evap, -1])

    axs[0, 0].set_title("Solid Axial Temperature")
    axs[0, 1].set_title("Vapour Velocity")
    axs[0, 2].set_title("Vapour Temperature")
    axs[1, 0].set_title("Fuel Axial Pin Temperature")
    axs[1, 1].set_title("Neutron Flux (Group 0)")
    axs[1, 2].set_title("Neutron Flux (Group -1)")

    plt.show()
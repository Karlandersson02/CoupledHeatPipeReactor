import numpy as np
import matplotlib.pyplot as plt

import data.dataclass as d_class

from utils.sodium_properties import calculate_Na_rho_l, calculate_Na_viscosity_l, calculate_Na_h_fg


def make_liquid_input(T_HP, T_v):
    """
    LiquidDiscretised expects one flat array:

        [flattened solid temperature field, scalar vapour temperature]

    The coupled Heatpipe solver may return T_HP either as:
        - flat array of shape (N_Z * N_R,)
        - 2D array of shape (N_Z, N_R)

    Therefore, we explicitly flatten it before appending T_v.
    """

    T_HP_flat = np.asarray(T_HP, dtype=float).reshape(-1)

    T_v_arr = np.asarray(T_v, dtype=float)
    T_v_scalar = float(np.mean(T_v_arr))

    T_liquid_input = np.concatenate([
        T_HP_flat,
        np.array([T_v_scalar], dtype=float),
    ])

    return T_liquid_input, T_v_scalar


class LiquidDiscretised:
    def __init__(self, config: d_class.HeatpipeConfigResolved, T_HP_full):
        self.cfg = config

        self.T_HP_full = T_HP_full

        # Full reactor heat-pipe solution
        self.T_solid_full = np.asarray(T_HP_full[0], dtype=float)
        self.u_v_full = np.asarray(T_HP_full[1], dtype=float)
        self.T_v_full = np.asarray(T_HP_full[2], dtype=float).reshape(-1)

        # Old format expected by the rest of the liquid model
        self.T_HP, self.T_v_scalar = make_liquid_input(
            self.T_solid_full,
            self.T_v_full,
        )

        self.interface_index = 0


    def get_solid_temperature_2d(self):
        """
        Return solid heat-pipe temperature field as shape (N_Z, N_R).
        """
        T_solid = np.asarray(self.T_HP, dtype=float)[:-1]

        expected_size = self.cfg.mesh.N_Z * self.cfg.mesh.N_R

        if T_solid.size != expected_size:
            raise ValueError(
                "Temperature-size mismatch in LiquidDiscretised:\n"
                f"  T_solid.size = {T_solid.size}\n"
                f"  N_Z * N_R    = {expected_size}\n"
                f"  N_Z          = {self.cfg.mesh.N_Z}\n"
                f"  N_R          = {self.cfg.mesh.N_R}"
            )

        return T_solid.reshape(self.cfg.mesh.N_Z, self.cfg.mesh.N_R)

    def get_mdot(self):
        """
        Compute the liquid/vapour mass-flow profile using the full axial
        reactor temperature profile.

        The old version used

            T_interface(z) - mean(T_v)

        while this version uses

            T_interface(z) - T_v(z)

        locally.

        Positive local heat transfer means evaporation.
        Negative local heat transfer means condensation.
        """
        T_solid_2d = self.get_solid_temperature_2d()

        T_interface = T_solid_2d[:, self.interface_index]
        T_v = np.asarray(self.T_v_full, dtype=float).reshape(-1)

        if len(T_v) != self.cfg.mesh.N_Z:
            raise ValueError(
                "Vapour temperature profile length mismatch:\n"
                f"  len(T_v) = {len(T_v)}\n"
                f"  N_Z      = {self.cfg.mesh.N_Z}"
            )

        delta_z = np.concatenate([
            np.ones(self.cfg.mesh.N_evap)
            * self.cfg.geometry.l_evap
            / self.cfg.mesh.N_evap,

            np.ones(self.cfg.mesh.N_adiabatic)
            * self.cfg.geometry.l_adiabatic
            / self.cfg.mesh.N_adiabatic,

            np.ones(self.cfg.mesh.N_cond)
            * self.cfg.geometry.l_cond
            / self.cfg.mesh.N_cond,
        ])

        if len(delta_z) != self.cfg.mesh.N_Z:
            raise ValueError(
                "Axial cell-length mismatch:\n"
                f"  len(delta_z) = {len(delta_z)}\n"
                f"  N_Z          = {self.cfg.mesh.N_Z}"
            )

        # Local wick/vapour interface area per axial cell
        A_int = (
            2.0
            * np.pi
            * self.cfg.geometry.r_vapour
            * delta_z
        )

        # Local heat transfer into vapour
        q_lv = (
            self.cfg.material.h_vap
            * A_int
            * (T_interface - T_v)
        )

        h_fg = calculate_Na_h_fg(T_v)

        # Local evaporation/condensation mass source
        dm_cell = q_lv / h_fg

        # Accumulate axial vapour/liquid mass flow.
        #
        # mdot_faces[0] = 0 at start of evaporator.
        # mdot_faces[i+1] = mdot_faces[i] + dm_cell[i]
        mdot_faces = np.concatenate([
            np.array([0.0]),
            np.cumsum(dm_cell),
        ])

        # Cell-centred mass flow
        mdot_cell_signed = 0.5 * (
            mdot_faces[:-1]
            + mdot_faces[1:]
        )

        # For the Darcy pressure-drop magnitude, use absolute flow.
        mdot_cell = np.abs(mdot_cell_signed)

        # Store diagnostics
        self.T_interface = T_interface
        self.T_v_used = T_v
        self.q_lv = q_lv
        self.dm_cell = dm_cell
        self.mdot_faces = mdot_faces
        self.mdot_cell_signed = mdot_cell_signed
        self.mdot_cell = mdot_cell

        return mdot_cell


    def calculate_K_annular_wick(self):
        self.r_2 = self.cfg.geometry.r_wick
        self.r_1 = self.cfg.geometry.r_gap

        R_star = self.r_2 / self.r_1

        fRe_l = 16 * (1 - R_star)**2 / (1 + R_star**2 - (1 - R_star**2)/np.log(1/R_star))

        D_h = 2 * (self.r_1 - self.r_2)

        K = D_h**2 / (2 * fRe_l)

        return K

    def get_pressure_drop_profile(self):
        if self.cfg.wick.Is_annular == True:
            A_wick = np.pi * (self.cfg.geometry.r_gap**2 - self.cfg.geometry.r_wick**2) 
        else:
            A_wick = np.pi * (self.cfg.geometry.r_wick**2 - self.cfg.geometry.r_vapour**2)
        
        T_wick = np.mean(np.array(self.T_HP)[:-1].reshape(self.cfg.mesh.N_Z, self.cfg.mesh.N_R)[:, :self.cfg.mesh.N_wick], axis=1) 

        mu_l = calculate_Na_viscosity_l(T_wick)  
        rho_l = calculate_Na_rho_l(T_wick)

        if self.cfg.wick.Is_annular == True:
            K = self.calculate_K_annular_wick()
        else:
            K = self.cfg.wick.K

        delta_z = np.concatenate(
            [np.ones(self.cfg.mesh.N_evap) * self.cfg.geometry.l_evap / self.cfg.mesh.N_evap,
            np.ones(self.cfg.mesh.N_adiabatic) * self.cfg.geometry.l_adiabatic / self.cfg.mesh.N_adiabatic,
            np.ones(self.cfg.mesh.N_cond) * self.cfg.geometry.l_cond / self.cfg.mesh.N_cond]
            )
        
        mdot = self.get_mdot()
        
        P = -np.cumsum(mu_l * np.array(mdot) / (rho_l * A_wick * K) * delta_z)
        self.pressure = P

        return P
    
    def print_mdot_diagnostics(self, label=""):
        """
        Diagnostic printout for checking whether the local temperature-based
        mass-flow calculation is reasonable.
        """
        mdot = self.get_mdot()

        q_lv = self.q_lv
        dm_cell = self.dm_cell
        mdot_faces = self.mdot_faces
        mdot_cell_signed = self.mdot_cell_signed

        Q_evap = float(np.sum(q_lv[q_lv > 0.0]))
        Q_cond = float(np.sum(q_lv[q_lv < 0.0]))
        Q_net = float(np.sum(q_lv))

        print("\n" + "=" * 80)
        print(f"LIQUID MDOT DIAGNOSTIC {label}")
        print("=" * 80)

        print("\nTemperature fields:")
        print(f"  interface_index                 = {self.interface_index}")
        print(f"  T_interface min/max             = {np.nanmin(self.T_interface):.6e} / {np.nanmax(self.T_interface):.6e} K")
        print(f"  T_v full min/max                = {np.nanmin(self.T_v_used):.6e} / {np.nanmax(self.T_v_used):.6e} K")
        print(f"  T_v scalar used in old format   = {self.T_v_scalar:.6e} K")
        print(f"  dT local min/max                = {np.nanmin(self.T_interface - self.T_v_used):.6e} / {np.nanmax(self.T_interface - self.T_v_used):.6e} K")

        print("\nHeat exchange:")
        print(f"  sum positive q_lv               = {Q_evap:.6e} W")
        print(f"  sum negative q_lv               = {Q_cond:.6e} W")
        print(f"  net q_lv                        = {Q_net:.6e} W")
        print(f"  max abs q_lv cell               = {np.nanmax(np.abs(q_lv)):.6e} W")

        print("\nMass flow:")
        print(f"  dm_cell min/max                 = {np.nanmin(dm_cell):.6e} / {np.nanmax(dm_cell):.6e} kg/s")
        print(f"  mdot_faces min/max              = {np.nanmin(mdot_faces):.6e} / {np.nanmax(mdot_faces):.6e} kg/s")
        print(f"  mdot_cell_signed min/max        = {np.nanmin(mdot_cell_signed):.6e} / {np.nanmax(mdot_cell_signed):.6e} kg/s")
        print(f"  mdot_cell magnitude min/max     = {np.nanmin(mdot):.6e} / {np.nanmax(mdot):.6e} kg/s")

        print("=" * 80 + "\n")


    def print_liquid_diagnostics(self, label=""):
        """
        Diagnostic printout for the liquid/wick pressure-drop model.

        The pressure-gradient model is

            dP/dz = mu_l * mdot / (rho_l * A_wick * K)

        so this prints all quantities entering that expression.
        """

        T_HP_arr = np.asarray(self.T_HP, dtype=float)

        N_R = self.cfg.mesh.N_R
        N_Z = self.cfg.mesh.N_Z
        N_evap = self.cfg.mesh.N_evap
        N_adiabatic = self.cfg.mesh.N_adiabatic
        N_cond = self.cfg.mesh.N_cond
        N_wick = self.cfg.mesh.N_wick

        T_solid = T_HP_arr[:-1]
        T_v = float(T_HP_arr[-1])

        if T_solid.size != N_Z * N_R:
            print(
                "\nLIQUID DIAGNOSTIC WARNING:"
                f"\n  T_solid.size = {T_solid.size}"
                f"\n  N_Z * N_R    = {N_Z * N_R}"
                f"\n  N_Z          = {N_Z}"
                f"\n  N_R          = {N_R}"
            )
            return

        T_solid_2d = T_solid.reshape(N_Z, N_R)

        # Current code uses radial index 0 as wick/liquid-vapour interface.
        T_interface_current = T_solid_2d[:, 0]

        # Useful alternatives to inspect.
        T_interface_last_wick = T_solid_2d[:, N_wick - 1]
        T_wick_mean = np.mean(T_solid_2d[:, :N_wick], axis=1)

        if self.cfg.wick.Is_annular:
            A_wick = np.pi * (
                self.cfg.geometry.r_gap**2
                - self.cfg.geometry.r_wick**2
            )
            K = self.calculate_K_annular_wick()
            K_source = "calculate_K_annular_wick()"
        else:
            A_wick = np.pi * (
                self.cfg.geometry.r_wick**2
                - self.cfg.geometry.r_vapour**2
            )
            K = self.cfg.wick.K
            K_source = "cfg.wick.K"

        delta_z = np.concatenate([
            np.ones(N_evap) * self.cfg.geometry.l_evap / N_evap,
            np.ones(N_adiabatic) * self.cfg.geometry.l_adiabatic / N_adiabatic,
            np.ones(N_cond) * self.cfg.geometry.l_cond / N_cond,
        ])

        mdot = np.asarray(self.get_mdot(), dtype=float)
        mu_l = calculate_Na_viscosity_l(T_wick_mean)
        rho_l = calculate_Na_rho_l(T_wick_mean)
        h_fg = calculate_Na_h_fg(T_v)

        dP_dz = mu_l * mdot / (rho_l * A_wick * K)
        dP_cells = dP_dz * delta_z
        P = -np.cumsum(dP_cells)

        # Reconstruct Qevap used by get_mdot().
        A_int_cell = (
            2.0
            * np.pi
            * self.cfg.geometry.r_vapour
            * self.cfg.geometry.l_evap
            / N_evap
        )

        Qevap_current = (
            np.sum(
                self.cfg.material.h_vap
                * (T_interface_current[:N_evap] - T_v)
            )
            * A_int_cell
        )

        Qevap_last_wick = (
            np.sum(
                self.cfg.material.h_vap
                * (T_interface_last_wick[:N_evap] - T_v)
            )
            * A_int_cell
        )

        print("\n" + "=" * 80)
        print(f"LIQUID / WICK PRESSURE-DROP DIAGNOSTIC {label}")
        print("=" * 80)

        print("\nMesh:")
        print(f"  N_R                      = {N_R}")
        print(f"  N_Z                      = {N_Z}")
        print(f"  N_wick                   = {N_wick}")
        print(f"  N_evap                   = {N_evap}")
        print(f"  N_adiabatic              = {N_adiabatic}")
        print(f"  N_cond                   = {N_cond}")
        print(f"  N_evap + N_adiabatic + N_cond = {N_evap + N_adiabatic + N_cond}")

        print("\nGeometry:")
        print(f"  r_vapour                 = {self.cfg.geometry.r_vapour:.6e} m")
        print(f"  r_wick                   = {self.cfg.geometry.r_wick:.6e} m")
        print(f"  r_gap                    = {self.cfg.geometry.r_gap:.6e} m")
        print(f"  l_evap                   = {self.cfg.geometry.l_evap:.6e} m")
        print(f"  l_adiabatic              = {self.cfg.geometry.l_adiabatic:.6e} m")
        print(f"  l_cond                   = {self.cfg.geometry.l_cond:.6e} m")
        print(f"  l_tot from sections       = {(self.cfg.geometry.l_evap + self.cfg.geometry.l_adiabatic + self.cfg.geometry.l_cond):.6e} m")

        print("\nWick / flow constants:")
        print(f"  Is_annular               = {self.cfg.wick.Is_annular}")
        print(f"  A_wick                   = {A_wick:.6e} m^2")
        print(f"  K                        = {K:.6e} m^2")
        print(f"  K source                 = {K_source}")
        print(f"  A_wick * K               = {(A_wick * K):.6e} m^4")
        print(f"  wick r_pore              = {self.cfg.wick.r_pore:.6e} m")
        print(f"  h_vap                    = {self.cfg.material.h_vap:.6e} W/(m^2 K)")
        print(f"  A_int_cell               = {A_int_cell:.6e} m^2")

        print("\nTemperatures:")
        print(f"  T_v                      = {T_v:.6e} K")
        print(f"  T_wick_mean min/max      = {np.nanmin(T_wick_mean):.6e} / {np.nanmax(T_wick_mean):.6e} K")
        print(f"  T_interface_current min/max = {np.nanmin(T_interface_current):.6e} / {np.nanmax(T_interface_current):.6e} K")
        print(f"  T_interface_last_wick min/max = {np.nanmin(T_interface_last_wick):.6e} / {np.nanmax(T_interface_last_wick):.6e} K")
        print(f"  evap dT current min/max  = {np.nanmin(T_interface_current[:N_evap] - T_v):.6e} / {np.nanmax(T_interface_current[:N_evap] - T_v):.6e} K")
        print(f"  evap dT last-wick min/max = {np.nanmin(T_interface_last_wick[:N_evap] - T_v):.6e} / {np.nanmax(T_interface_last_wick[:N_evap] - T_v):.6e} K")

        print("\nSodium properties:")
        print(f"  h_fg(T_v)                = {h_fg:.6e} J/kg")
        print(f"  mu_l min/max             = {np.nanmin(mu_l):.6e} / {np.nanmax(mu_l):.6e} Pa s")
        print(f"  rho_l min/max            = {np.nanmin(rho_l):.6e} / {np.nanmax(rho_l):.6e} kg/m^3")

        print("\nEvaporation / mass flow:")
        print(f"  Qevap using current interface index 0      = {Qevap_current:.6e} W")
        print(f"  Qevap using last wick index N_wick-1       = {Qevap_last_wick:.6e} W")
        print(f"  mdot total estimate from get_mdot max      = {np.nanmax(mdot):.6e} kg/s")
        print(f"  mdot min/max profile                       = {np.nanmin(mdot):.6e} / {np.nanmax(mdot):.6e} kg/s")
        print(f"  superficial velocity max mdot/(rho A)      = {np.nanmax(np.abs(mdot / (rho_l * A_wick))):.6e} m/s")

        print("\nPressure-gradient contributions:")
        print(f"  delta_z min/max           = {np.nanmin(delta_z):.6e} / {np.nanmax(delta_z):.6e} m")
        print(f"  dP_dz min/max             = {np.nanmin(dP_dz):.6e} / {np.nanmax(dP_dz):.6e} Pa/m")
        print(f"  dP_cell min/max           = {np.nanmin(dP_cells):.6e} / {np.nanmax(dP_cells):.6e} Pa")
        print(f"  total signed liquid dP    = {(P[-1] - P[0]):.6e} Pa")
        print(f"  total abs liquid dP       = {abs(P[-1] - P[0]):.6e} Pa")
        print(f"  cumulative P min/max      = {np.nanmin(P):.6e} / {np.nanmax(P):.6e} Pa")

        print("=" * 80 + "\n")


if __name__ == "__main__":
    import json
    from models.heatpipe.solid_discretised_model import HeatpipeDiscretised
    from utils.solver import Solver

    with open("./data/vapour_data.json", "r") as f:
        data_guoju = json.load(f)
        data = data_guoju["data_guoju_560"]

    geom = d_class.HeatpipeGeometry(**data["geometry"])
    mesh = d_class.HeatpipeMesh(N_R=20, N_Z=30)
    mat = d_class.HeatpipeMaterial(**data["material"])
    wick = d_class.HeatpipeWick(**data["wick"])
    bc = d_class.HeatpipeBC(**data["bc"])
    cfg = d_class.HeatpipeConfig(geom, mesh, mat, wick, bc)
    cfg = cfg.resolve_geometry()

    heatpipe = HeatpipeDiscretised(cfg)
    solver = Solver([heatpipe])
    solver.fsolve()

    T_HP = heatpipe.pre_process(solver.solution)
    cfg = d_class.HeatpipeConfig(geom, mesh, mat, wick, bc)
    cfg = cfg.resolve_geometry()

    liquid = LiquidDiscretised(cfg, T_HP)
    P = liquid.get_pressure_drop_profile()

    plt.rcParams["font.size"] = 22
    plt.rcParams["font.family"] = "Computer modern"
    # plt.rcParams["text.usetex"] = True

    fig = plt.figure(figsize=(16,9))
    ax = fig.add_subplot(111)

    x = np.linspace(0, heatpipe.cfg.geometry.l_tot, len(P))
    ax.grid(alpha=0.4)
    ax.plot(x, P, color="black")
    ax.set_xlim((heatpipe.cfg.geometry.l_tot, 0.))

    ax.set_xlabel("l")
    ax.set_ylabel("P")

    plt.show()
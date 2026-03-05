import numpy as np
from scipy.optimize import root, newton_krylov
import matplotlib.pyplot as plt
import matplotlib as mpl

from models.sodium_properties import calculate_Na_h_fg, calculate_Na_viscosity_v, calculate_Na_rho_v, calculate_Na_pressure_v

class vapour_discretised:
    def __init__(self, data):
        self.r_outer  = data.get("r_outer")
        self.delta_wick  = data.get("delta_wick")
        self.delta_wall  = data.get("delta_wall")
        self.r_vapour = self.r_outer - self.delta_wick - self.delta_wall

        self.l_evap  = data.get("l_evap")
        self.l_adiabatic  = data.get("l_adiabatic")
        self.l_cond  = data.get("l_cond")
        self.l_tot = self.l_evap + self.l_adiabatic + self.l_cond

        self.N_evap = data.get("N_evap")
        self.N_adiabatic = data.get("N_adiabatic")
        self.N_cond = data.get("N_cond")
        self.N_Z = self.N_evap + self.N_adiabatic + self.N_cond

        self.N_wick = data.get("N_wick")
        self.N_wall = data.get("N_wall")
        self.N_R  = self.N_wick + self.N_wall

        # heat transfer coefficient
        self.h_vap = data.get("h_vap")

        self.T_HP = data.get("T_HP")

        self.T_C = data.get("T_C")
        self.P_C = data.get("P_C")

        # Molar gas constant
        self.R = 8.314472 

        # Specific gas constant, 0.022990 being the molar mass.
        self.R_Na = self.R / 0.022990

    def solve(self):
        Gamma, h_fg_Na = self.calculate_mass_flow_and_latent_heat()
        self.viscosity_Na = calculate_Na_viscosity_v(self.T_HP[-1])

        T_v, u_v = self.solve_vapour_heat_profile(Gamma, h_fg_Na)

        P_v = calculate_Na_pressure_v(T_v)

        # T_full = T_v

        # rhoim1 = calculate_Na_rho_v(T_full[:-1])
        # mdot = rhoim1*u_v

        # Rei = calculate_Na_rho_v((T_full[1:] + T_full[:-1])/2) * np.abs(u_v) * 2*self.r_vapour / self.viscosity_Na
        # lami = np.zeros_like(Rei)
        # lami[Rei <= 2200] = 64 / Rei[Rei <= 2200]
        # lami[Rei > 3000] = 0.316 / Rei[Rei > 3000]**0.25
        # lami[(Rei > 2200) & (Rei <= 3000)] = Rei[(Rei > 2200) & (Rei <= 3000)] * 1.70088e-5 - 0.00832838               # interpolation (behöver dubbelkollas)
        # lami[Rei == 0] = 1e10

        # dxi = np.zeros_like(T_full)
        # dxi[:self.N_evap] = self.l_evap / self.N_evap
        # dxi[self.N_evap:(self.N_evap + self.N_adiabatic)] = self.l_adiabatic / self.N_adiabatic
        # dxi[(self.N_evap + self.N_adiabatic):] = self.l_cond / self.N_cond

        self.temperature = T_v
        self.pressure = P_v

    #     # return T_v, P_v, Gamma*dxi, mdot, Rei, lami, u_v, P_analytic
    

    def calculate_mass_flow_and_latent_heat(self):
        T_wick_lv_interface = np.array(self.T_HP)[:-1:self.N_R]

        T_v = self.T_HP[-1]

        q_bis_surface = np.zeros(self.N_Z)
        q_bis_surface[0: self.N_evap]             =  self.h_vap * (T_wick_lv_interface[0: self.N_evap]           - T_v)
        q_bis_surface[self.N_Z - self.N_cond: -1] =  self.h_vap * (T_wick_lv_interface[self.N_Z - self.N_cond: -1] - T_v)
        
        # Heat transfer surface area density per unit volume.
        # a_W = 2 * np.pi * self.r_vapour * delta_Z / np.pi * self.r_vapour**2 * delta_Z
        a_W = 2 / self.r_vapour

        h_fg_Na = calculate_Na_h_fg(T_v)

        Gamma = a_W * q_bis_surface / h_fg_Na

        return Gamma, h_fg_Na

    def build_u_initial_guess(self):
        mdot = self.get_mdot()
        rho_v = calculate_Na_rho_v(self.T_HP[-1])
        A_v = np.pi * self.r_vapour**2
        u_guess = mdot / (rho_v * A_v)
        
        return u_guess[1:]
        # T_wick_lv_interface = np.array(self.T_HP)[:-1:self.N_R]
        # T_v = self.T_HP[-1]

        # A_int = 2 * np.pi * self.r_vapour * self.l_evap / self.N_evap
        # A_v = np.pi * self.r_vapour**2
        # Qevap = np.sum(self.h_vap * (T_wick_lv_interface[:self.N_evap] - T_v)) * A_int
        # h_fg = calculate_Na_h_fg(T_v)
        # rho_v = calculate_Na_rho_v(T_v)

        # u_shape = np.array([0.07810245, 0.15620481, 0.23430684, 0.31240859, 0.39051026, 0.46861212, 0.54671421, 0.62481634, 0.70291819, 0.7810199, 0.85912161, 0.93722368, 1.01532644, 1.09342929, 1.17153219, 1.24963485, 1.32773731, 1.40583935, 1.48393355, 1.56174227, 1.56174184, 1.56174126, 1.56174057, 1.56174046, 1.56174099, 1.56174192, 1.56174299, 1.561744,   1.56174489, 1.56174572, 1.54783865, 1.53364632, 1.51944598, 1.50524516, 1.49104431, 1.47684384, 1.4626438,  1.44844357, 1.4342428,  1.42004181, 1.40584097, 1.39164058, 1.37744032, 1.36324051, 1.34904079, 1.33484092, 1.32064108, 1.30644105, 1.29224103, 1.27804115, 1.26384077, 1.24964012, 1.23543916, 1.22123884, 1.20703852, 1.19283811, 1.17863743, 1.16443672, 1.15023611, 1.13603599, 1.12183574, 1.10763584, 1.09343659, 1.07923738, 1.0650379,  1.05083841, 1.0366393,  1.02243946, 1.00823923, 0.99403854, 0.97983745, 0.96563596, 0.95143382, 0.93723274, 0.92303139, 0.9088297,  0.89462788, 0.88042592, 0.86622391, 0.85202234, 0.837821,   0.8236198,  0.80941873, 0.79521783, 0.78101771, 0.76681744, 0.75261726, 0.73841769, 0.72421857, 0.71001976, 0.69582092, 0.68162228, 0.66742358, 0.65322438, 0.6390244,  0.62482395, 0.61062363, 0.59642297, 0.58222231, 0.56802169, 0.55382078, 0.53961971, 0.52541961, 0.51121881, 0.49701744, 0.48281564, 0.46861366, 0.45441181, 0.4402102,  0.42600836, 0.41180717, 0.39760609, 0.38340523, 0.36920421, 0.35500445, 0.34080469, 0.32660507, 0.31240588, 0.29820726, 0.2840084, 0.2698095,  0.25561013, 0.2414097,  0.22720912, 0.21300832, 0.19880726, 0.18460593, 0.1704047,  0.15620322, 0.14200221, 0.12780148, 0.11360088, 0.0994005,  0.0851985,  0.07099711, 0.05679684, 0.04259744, 0.02839852, 0.01420016])
        # u_normalised = u_shape / np.max(u_shape)
        # u_guess = u_normalised * Qevap / (h_fg * rho_v * A_v)

        # return u_guess
    
    def get_mdot(self):
        T_wick_lv_interface = np.array(self.T_HP)[:-1:self.N_R]
        T_v = self.T_HP[-1]

        A_int = 2 * np.pi * self.r_vapour * self.l_evap / self.N_evap
        Qevap = np.sum(self.h_vap * (T_wick_lv_interface[:self.N_evap] - T_v)) * A_int
        mdot_peak = Qevap / calculate_Na_h_fg(T_v)

        mdot = np.concatenate([
            np.repeat(np.array([mdot_peak/self.N_evap]), self.N_evap) * np.arange(self.N_evap),
            np.repeat(np.array([mdot_peak]), self.N_adiabatic),
            np.repeat(np.array([mdot_peak/self.N_cond]), self.N_cond) * np.arange(self.N_cond)[::-1]
        ])

        return mdot
    
        # T_wick_lv_interface = np.array(self.T_HP)[:-1:self.N_R]
        # T_v = self.T_HP[-1]
        # h_fg = calculate_Na_h_fg(T_v)

        # A_int = 2 * np.pi * self.r_vapour * self.l_evap / self.N_evap
        # Qevap = np.sum(self.h_vap * (T_wick_lv_interface[:self.N_evap] - T_v)) * A_int
        # mdot = Qevap / h_fg

        # return mdot


    def build_T_initial_guess(self) -> np.ndarray:
        # cell-centered z positions (same indexing as your Gamma and T arrays)
        dxi = np.zeros(self.N_Z)
        dxi[:self.N_evap] = self.l_evap / self.N_evap
        dxi[self.N_evap:self.N_evap + self.N_adiabatic] = self.l_adiabatic / self.N_adiabatic
        dxi[self.N_evap + self.N_adiabatic:] = self.l_cond / self.N_cond
        zc = np.cumsum(dxi) - 0.5 * dxi  # cell centers

        # --- pick reasonable temperature levels from what you already have ---
        # A good "hot end" estimate: average wick-LV interface temperature in evaporator
        # (Fix the extraction if needed; see note at the end.)
        T_wick_lv_interface = np.array(self.T_HP)[:-1:self.N_R]
        T_hot = float(np.mean(T_wick_lv_interface[:self.N_evap]))

        # A good "cold end" estimate: average wick-LV interface temperature in condenser
        T_cold_wall = float(np.mean(T_wick_lv_interface[-self.N_cond:]))

        # Use a drop magnitude similar to the paper (about 12–18 K). Tie it to your data:
        dT_drop = np.clip(T_hot - T_cold_wall, 8.0, 18.0)

        T0   = T_hot
        Tmin = T0 - dT_drop
        Tend = Tmin + 0.55 * dT_drop   # recovery in condenser (tune 0.45–0.70)

        # anchor locations: start, end of evap, end of adiabatic (min), end
        z0   = 0.0
        zE   = self.l_evap
        zEA  = self.l_evap + self.l_adiabatic
        zL   = self.l_tot

        # anchor temperatures: sharp drop in evaporator, slight further drop to min
        anchors_z = np.array([z0,  zE,          zEA,  zL])
        anchors_T = np.array([T0,  Tmin + 0.25*dT_drop, Tmin, Tend])

        T_guess = np.interp(zc, anchors_z, anchors_T)

        # Optional: enforce your Neumann ends in the initial guess
        T_guess[0]  = T_guess[1]
        T_guess[-1] = T_guess[-2]
        return T_guess

    
    def analytical_pressure_drop(self):
        T_v  = self.T_HP[-1]
        h_fg = calculate_Na_h_fg(T_v)
        rho_v = calculate_Na_rho_v(T_v)
        self.viscosity_Na = calculate_Na_viscosity_v(self.T_HP[-1])

        Rv  = self.r_vapour          # make sure this is in meters
        mu_v = self.viscosity_Na

        Av = np.pi * Rv**2
        mdot = self.build_u_initial_guess() * rho_v * Av

        dmdx_evap = mdot[self.N_evap] / self.l_evap
        dmdx_adia = 0.0
        dmdx_cond = -mdot[self.N_evap] / self.l_cond

        dpdx = np.zeros_like(mdot, dtype=float)

        # Evaporator: q>0 => s=1, a=0
        dpdx[:self.N_evap] = -(1.0) * (mdot[:self.N_evap] * dmdx_evap) / (4.0 * rho_v * Rv**4)

        # Adiabatic: q=0 => s=0, a=1
        i0 = self.N_evap
        i1 = self.N_evap + self.N_adiabatic
        dpdx[i0:i1] = -(8.0 * mu_v * mdot[i0:i1]) / (rho_v * np.pi * Rv**4)

        # Condenser: q<0 => s=4/pi^2, a=0
        j0 = self.N_Z - self.N_cond
        dpdx[j0:] = -(4.0 / np.pi**2) * (mdot[j0:] * dmdx_cond) / (4.0 * rho_v * Rv**4)

        # Integrate dp/dx over x
        dx = self.l_tot / (self.N_Z - 1)
        # return np.sum(dpdx) * dx
        return np.cumsum(dpdx) * dx


    def solve_vapour_heat_profile(self, Gamma: np.ndarray, h_fg_Na: float):
        print(self.N_Z)
        initial_guess = np.ones(self.N_Z - 1 + self.N_Z)
        initial_guess[self.N_Z-1:] = self.build_T_initial_guess()
        initial_guess[:self.N_Z-1] = self.build_u_initial_guess()

        coupled_system_lambda = lambda S: self.coupled_system(S[:(self.N_Z - 1)], S[(self.N_Z - 1):], Gamma, h_fg_Na) 

        self.train_history = np.zeros((1, len(initial_guess)))
        def history_append(x, f):
            self.train_history = np.append(self.train_history, x[None], axis=0)

        sol_krylov = newton_krylov(
            coupled_system_lambda,
            initial_guess,
            iter = 3000,
            verbose = True,
            method = "lgmres",
            callback = history_append
        )

        self.train_history = self.train_history[1:]

        T_v = sol_krylov[(self.N_Z - 1):]
        u_v = sol_krylov[:(self.N_Z - 1)]

        return T_v, u_v
    
    
    def update_Gamma(self, T_i: np.ndarray, Gami: np.ndarray, h_fg_Na: float):
        T_wick_lv_interface = np.array(self.T_HP)[:-1:self.N_R]

        a_W = 2 / self.r_vapour

        Gami[-1: ] = (a_W / h_fg_Na) * self.h_vap * (T_wick_lv_interface[-1: ] - T_i[-1: ])

        return Gami


    def coupled_system(self, u: np.ndarray, T: np.ndarray, Gami: np.ndarray, h_fg_Na: float):
        T_full = T
        Ti = T_full[1:]
        Tim1 = T_full[:-1]
        Tbar = (Ti + Tim1)/2

        u_full = np.zeros(len(u) + 2)
        u_full[1:-1] = u
        ui = u
        uim1 = u_full[:-2]
        uip1 = u_full[2:]

        rho_full = calculate_Na_rho_v(T_full)
        rhoi = rho_full[1:]
        rhoim1 = rho_full[:-1]
        rhobar = calculate_Na_rho_v(Tbar)

        dxi = np.zeros_like(T_full)
        dxi[:self.N_evap] = self.l_evap / self.N_evap
        dxi[self.N_evap:(self.N_evap + self.N_adiabatic)] = self.l_adiabatic / self.N_adiabatic
        dxi[(self.N_evap + self.N_adiabatic):] = self.l_cond / self.N_cond

        Rei = rhobar * np.abs(ui) * 2*self.r_vapour / self.viscosity_Na
        lami = np.zeros_like(Rei)
        lami[Rei <= 2200] = 64 / Rei[Rei <= 2200]
        lami[Rei > 3000] = 0.316 / Rei[Rei > 3000]**0.25
        lami[(Rei > 2200) & (Rei <= 3000)] = Rei[(Rei > 2200) & (Rei <= 3000)] * 1.70088e-5 - 0.00832838               # interpolation (behöver dubbelkollas)
        lami[Rei == 0] = 1e10

        Gami = self.update_Gamma(T_full, Gami, h_fg_Na)

        r1 = np.zeros_like(T_full)
        r1[1:-1] = (uip1[:-1]*rhoi[:-1] - ui[:-1]*rhoim1[:-1]) - dxi[1:-1] * Gami[1:-1]
        r1[0] = u_full[1]*rho_full[0] - dxi[0]*Gami[0]
        r1[-1] = -u_full[-2]*rho_full[-2] - dxi[-1]*Gami[-1]

        r2 = 1 * ((rhoi * ui**2 - rhoim1 * uim1**2)
               + (rhobar * h_fg_Na) / Tbar * (Ti - Tim1) 
               + dxi[1:] * lami / (2*2*self.r_vapour) * rhobar * ui * np.abs(ui))
        
        adaptive_r1r2_bias = np.sqrt(np.linalg.norm(r2) / np.linalg.norm(r1)) * 5.

        return np.concatenate([adaptive_r1r2_bias * r1 , r2])


if __name__ == "__main__":
    data = {
        "r_outer": .007 + 0.001 + 0.0005,
        "delta_wick": 0.0005,
        "delta_wall": 0.001,
        "l_evap": 0.1,
        "l_adiabatic": 0.05,
        "l_cond": 0.55,
        "N_wick": 15,
        "N_wall": 15,
        "N_evap": 20,
        "N_adiabatic": 10,
        "N_cond": 110,
        "h_vap": 1e6,
        "h_cond": 62.6,
        "T_cond": 300,
        "k_wick": 45.0,
        "k_wall": 21.7,
        "P_C": 2476,
        "T_C": 856,
        "viscosity_Na": 1.80e-5,
    }

    vapor = vapour_discretised(data) 

    vapor.solve()
    T_v = vapor.temperature
    P_v = vapor.pressure

    mpl.rcParams["font.size"] = 22
    # mpl.rcParams["font.family"] = "computer modern"
    #mpl.rcParams["text.usetex"] = True

    # fig = plt.figure(figsize=(16,9))

    # fig.subplots_adjust(
    #     hspace=0.0,
    #     wspace=0.5
    # )
    # ax1 = fig.add_subplot(221)
    # ax3 = fig.add_subplot(222)
    # ax4 = fig.add_subplot(223, sharex=ax1)
    # ax2 = fig.add_subplot(224, sharex=ax3)
    # ax1_twin = ax1.twinx()
    # ax4_twin = ax4.twinx()

    # ax1.grid(alpha=0.4)
    # ax2.grid(alpha=0.4)
    # ax3.grid(alpha=0.4)
    # ax4.grid(alpha=0.4)

    # ax1.plot(T_v, color="red", marker="o", label="Temperature")
    # ax1_twin.plot(P_v, color="blue", marker="o", markersize=5, label="Pressure")
    # ax4.plot(Rei, color="red", marker="o", label=r"Reynolds Re")
    # ax4_twin.plot(lami, color="blue", marker="o", markersize=5, label=r"Friction $\lambda$")

    # ax2.plot(Gamma, color='black', marker="o", label=rf"$\sum\Gamma=$ {np.sum(Gamma):.3g}")
    # ax3.plot(mdot, color="black", marker="o")

    # # --- Make axes colored ---
    # ax1.set_ylabel("Temperature", color="red")
    # ax1.tick_params(axis='y', colors="red")
    # ax1.spines["left"].set_color("red")

    # ax1_twin.set_ylabel("Pressure", color="blue")
    # ax1_twin.tick_params(axis='y', colors="blue")
    # ax1_twin.spines["right"].set_color("blue")

    # ax4.set_ylabel("Reynolds Re", color="red")
    # ax4.tick_params(axis='y', colors="red")
    # ax4.spines["left"].set_color("red")

    # ax4_twin.set_ylabel(r"Friction $\lambda$", color="blue")
    # ax4_twin.tick_params(axis='y', colors="blue")
    # ax4_twin.spines["right"].set_color("blue")
    # # --------------------------

    # ax1.tick_params(
    #     axis='x',
    #     which='both',
    #     bottom=False,
    #     top=False,
    #     labelbottom=False)
    # ax3.tick_params(
    #     axis='x',
    #     which='both',
    #     bottom=False,
    #     top=False,
    #     labelbottom=False)

    # lines1, labels1 = ax1.get_legend_handles_labels()
    # lines2, labels2 = ax1_twin.get_legend_handles_labels()
    # ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    # lines4, labels4 = ax4.get_legend_handles_labels()
    # lines3, labels3 = ax4_twin.get_legend_handles_labels()
    # ax4.legend(lines4 + lines3, labels4 + labels3, loc="best")

    # ax2.legend()

    # ax2.set_xlabel(r"Element number $n$")
    # ax4.set_xlabel(r"Element number $n$")

    # ax3.set_ylabel(r"Mass flow $\rho_{i-1}u_i$")
    # ax2.set_ylabel(r"Generated mass flow $\Gamma$")

    # ------------------------------

    # plt.savefig("./Figures/vapour_system_unconstrained.png", bbox_inches="tight")

    # fig = plt.figure(figsize=(16,9))
    # fig.subplots_adjust(hspace=0.0)
    # ax1 = fig.add_subplot(211)
    # ax1_twin = ax1.twinx()
    # ax2 = fig.add_subplot(212)

    # ax1.grid(alpha=0.4)
    # ax2.grid(alpha=0.4)

    # ax1.plot(T_v, marker="o", markersize=5, color="red", label="Temperature")
    # ax1_twin.plot(P_v, marker="o", markersize=5, color="blue", label=rf"$\Delta p =$ {np.round(P_v[0] - P_v[-1])} Pa")
    # ax2.plot(mdot, marker="o", markersize=5, color="black")

    # ax1.tick_params(
    #     axis='x',
    #     which='both',
    #     bottom=False,
    #     top=False,
    #     labelbottom=False)
    
    # ax1.set_ylabel(r"$T$", color="red", fontsize=32)
    # ax1.tick_params(axis='y', colors="red")
    # ax1.spines["left"].set_color("red")
    # ax1_twin.set_ylabel(r"$P$", color="blue", fontsize=32)
    # ax1_twin.tick_params(axis='y', colors="blue")
    # ax1_twin.spines["right"].set_color("blue")
    # ax2.set_xlabel(r"$n$", fontsize=32)
    # ax2.set_ylabel(r"$\dot m$", fontsize=32)

    # # lines1, labels1 = ax1.get_legend_handles_labels()
    # # lines2, labels2 = ax1_twin.get_legend_handles_labels()
    # # ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    # ax1_twin.legend()

    # --------------------------------

    # P_v_subtracted = P_v - np.max(P_v)

    # fig = plt.figure(figsize=(16,9))
    # ax = fig.add_subplot(111)
    # ax.grid(alpha=0.4)
    # ax.plot(P_v_subtracted, color='red', label=rf"solver $\Delta P$: {P_v_subtracted[-1]:.0f}")
    # ax.plot(P_analytic, color='blue', label=rf"analytic $\Delta P$: {P_analytic[-1]:.0f}")

    # ax.legend()
    # ax.set_xlabel(r"$n$")
    # ax.set_ylabel(r"$P$")
    # plt.show()

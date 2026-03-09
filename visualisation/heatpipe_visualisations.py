
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from CoupledSystems.Heatpipe import Heatpipe
from visualisation.visualise_discretised_results import display_temperature_distribution

class heatpipe_visualisations:
    def __init__(self, heatpipe: Heatpipe):
        self.heatpipe = heatpipe
        l_evap = self.heatpipe.data["l_evap"]
        l_adiabatic = self.heatpipe.data["l_adiabatic"]
        l_cond = self.heatpipe.data["l_cond"]
        self.l_tot = l_evap + l_adiabatic + l_cond

        self.fontsize = 26
        self.fontfamily = "Computer modern"
        self.usetex = True

        self.figsize = (16,9)

    def setup_figure(self):
        mpl.rcParams["font.size"] = self.fontsize
        mpl.rcParams["font.family"] = self.fontfamily
        mpl.rcParams["text.usetex"] = self.usetex
        fig = plt.figure(figsize=self.figsize)
        return fig

    def plot_heatpipe_temperature(self):
        T = self.heatpipe.get_heatpipe_temperature()
        data = self.heatpipe.data
        display_temperature_distribution(T, data)

    def plot_vapour_temperature(self, ax=None, **kwargs):
        T = self.heatpipe.get_vapour_temperature()
        l = np.linspace(0, self.l_tot, len(T))

        if ax is None:
            fig = self.setup_figure()        
            ax = fig.add_subplot(111)

            ax.grid(alpha=0.4)
            ax.set_title("Vapour")
            ax.set_xlabel("Length [m]")
            ax.set_ylabel(r"Temperature $T$ [K]")

        ax.plot(l, T, **kwargs)

        if ax is None:
            plt.show()
        else:
            return ax
        
    def plot_vapour_pressure_drop(self, ax=None, **kwargs):
        P = self.heatpipe.get_vapour_pressure_drop_profile_numeric()
        P -= P[0]
        l = np.linspace(0, self.l_tot, len(P))

        if ax is None:
            fig = self.setup_figure()        
            ax = fig.add_subplot(111)

            ax.grid(alpha=0.4)
            ax.set_title("Vapour")
            ax.set_xlabel("Length [m]")
            ax.set_ylabel(r"Pressure $\Delta P$ [Pa]")

        ax.plot(l, P, **kwargs)

        if ax is None:
            plt.show()
        else:
            return ax
        
    def plot_vapour_mach_number(self, ax=None, **kwargs):
        T = self.heatpipe.get_vapour_temperature()
        u = self.heatpipe.vapour_discretised.get_velocity()
        gamma = 5/3
        cs = np.sqrt(gamma * self.heatpipe.vapour_discretised.R_Na * T[:-1]) # ignore last temperature
        Mach = u / cs

        l = np.linspace(0, self.l_tot, len(T)-1)

        if ax is None:
            fig = self.setup_figure()        
            ax = fig.add_subplot(111)

            ax.grid(alpha=0.4)
            ax.set_title("Vapour")
            ax.set_xlabel("Length [m]")
            ax.set_ylabel(r"Mach [1]")

        ax.plot(l, Mach, **kwargs)

        if ax is None:
            plt.show()
        else:
            return ax
        
        


class heatpipe_comparisons:
    def __init__(
        self, 
        baseline_data,
        comparison_data: dict[str, list]
    ):
        self.fontsize = 26
        self.fontfamily = "Computer modern"
        self.usetex = True

        self.figsize = (16,9)
        
        self.baseline_data = baseline_data
        self.comparison_data = comparison_data

        self.heatpipes: list[Heatpipe] = []
        self.heatpipe_visualisations: list[heatpipe_visualisations] = []
        
        self.comparison_data_keys = list(comparison_data.keys())
        self.comparison_data_values = list(comparison_data.values())
        for i in range(len(self.comparison_data_values[0])):
            data = self.baseline_data
            for j in range(len(self.comparison_data_keys)):
                data[self.comparison_data_keys[j]] = self.comparison_data_values[j][i]

            heatpipe = Heatpipe(data)
            heatpipe.setup_fluid_models()
            heatpipe_vis = heatpipe_visualisations(heatpipe)

            self.heatpipes.append(heatpipe)
            self.heatpipe_visualisations.append(heatpipe_vis)

    def setup_figure(self):
        mpl.rcParams["font.size"] = self.fontsize
        mpl.rcParams["font.family"] = self.fontfamily
        mpl.rcParams["text.usetex"] = self.usetex
        fig = plt.figure(figsize=self.figsize)
        return fig

    def plot_vapour_temperature_comparison(self):
        fig = self.setup_figure()
        ax = fig.add_subplot(111)

        for i in range(len(self.heatpipes)):
            ax = self.heatpipe_visualisations[i].plot_vapour_temperature(ax=ax, label=f"{self.heatpipes[i].vapour_discretised.T_HP[-1]:.0f}")

        ax.grid(alpha=0.4)
        ax.set_title("Vapour")
        ax.set_xlabel("Length [m]")
        ax.set_ylabel(r"Temperature $T$ [K]")
        ax.legend()
        plt.show()

    def plot_vapour_pressure_drop_comparison(self):
        fig = self.setup_figure()
        ax = fig.add_subplot(111)

        for i in range(len(self.heatpipes)):
            ax = self.heatpipe_visualisations[i].plot_vapour_pressure_drop(ax=ax, label=i)

        ax.grid(alpha=0.4)
        ax.set_title("Vapour")
        ax.set_xlabel("Length [m]")
        ax.set_ylabel(r"Pressure $\Delta P$ [Pa]")
        ax.legend()
        plt.show()
        
    def plot_vapour_mach_number_comparison(self):
        fig = self.setup_figure()
        ax = fig.add_subplot(111)

        l_bar = np.linspace(0, self.heatpipes[0].vapour_discretised.l_tot, 10)

        for i in range(len(self.heatpipes)):
            ax = self.heatpipe_visualisations[i].plot_vapour_mach_number(ax=ax, label=i)

        ax.grid(alpha=0.4)
        ax.hlines(1, l_bar[0], l_bar[-1], color="black", linestyle="--")
        ax.set_title("Vapour")
        ax.set_xlabel("Length [m]")
        ax.set_ylabel(r"Mach [1]")
        ax.legend()
        plt.show()
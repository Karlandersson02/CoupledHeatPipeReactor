import numpy as np

from models.component import Component
from models.heatpipe.heat_discretised_model import HeatpipeDiscretised
from models.neutronics.axial_neutron_model import NeutronicsModel

from scipy.optimize import fsolve, newton_krylov
from scipy.interpolate import RegularGridInterpolator

from typing import Sequence

def interpolate_2D(x_coarse, y_coarse, x_fine, y_fine, U_coarse):
    # Build interpolator

    interp = RegularGridInterpolator(
        (x_coarse, y_coarse),
        U_coarse,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )

    Xf, Yf = np.meshgrid(x_fine, y_fine, indexing="ij")

    # Points where interpolation is evaluated
    points_fine = np.stack([Xf.ravel(), Yf.ravel()], axis=-1)

    # Interpolated fine-grid solution
    U_fine = interp(points_fine).reshape(len(x_fine), len(y_fine))
    return U_fine

class Solver:

    def __init__(self, components: Sequence[Component], iterate = True):
        self.components = components
        self.iterate = iterate

    def newton_krylov(self, verbose=True, **kwargs):
        X_initial = self.components[0].initial_guess()

        for i, component in enumerate(self.components):
            component.assemble()
            if not self.iterate and i > 0:
                X_initial = component.initial_guess()
            
            X_sol = newton_krylov(
                component.get_residuals,
                X_initial,
                verbose = True,
                **kwargs
            )

            X_out = component.post_process(np.array(X_sol))
            if self.iterate and i < len(self.components)-1:
            
                if type(component) is HeatpipeDiscretised:
                    z_coarse = np.linspace(1, component.cfg.geometry.l_tot, component.cfg.mesh.N_Z)
                    r_coarse = np.linspace(1, component.cfg.geometry.r_outer, component.cfg.mesh.N_R)

                    next_component = self.components[i+1]
                    z_fine = np.linspace(1, next_component.cfg.geometry.l_tot, next_component.cfg.mesh.N_Z)
                    r_fine = np.linspace(1, next_component.cfg.geometry.r_outer, next_component.cfg.mesh.N_R)

                    X_out_interpolated = interpolate_2D(
                        z_coarse, r_coarse,
                        z_fine, r_fine,
                        X_out[0]
                    )

                    X_initial = next_component.pack((X_out_interpolated, X_out[-1]))
                    # X_out = next_component.post_process(X_initial)
                    # self.solution = X_out
                    # return

                if type(component) is NeutronicsModel:
                    z_coarse = np.linspace(1, component.cfg.mesh.l, component.cfg.mesh.l)
                    g_coarse = np.linspace(1, component.cfg.mesh.N_G, component.cfg.mesh.N_G)

                    next_component = self.components[i+1]
                    z_fine = np.linspace(1, next_component.cfg.mesh.l, next_component.cfg.mesh.l)
                    g_fine = np.linspace(1, next_component.cfg.energy.N_G, next_component.cfg.energy.N_G)

                    X_out_interpolated = interpolate_2D(
                        z_coarse, g_coarse,
                        z_fine, g_fine,
                        X_out[0]
                    )

                    X_initial = next_component.pack((X_out_interpolated, X_out[-1]))
        
        self.solution = X_out       # Not implemented multiple X_out
    
    def fsolve(self, **kwargs):
        X_initial = self.components[0].initial_guess()

        for i, component in enumerate(self.components):
            component.assemble()
            if not self.iterate and i > 0:
                X_initial = component.initial_guess()
            
            X_sol = fsolve(
                component.get_residuals,
                X_initial,
                **kwargs
            )

            X_out = component.post_process(np.array(X_sol))
            if self.iterate and i < len(self.components)-1:
            
                if type(component) is HeatpipeDiscretised:
                    r_coarse = np.linspace(1, component.cfg.mesh.N_R, component.cfg.mesh.N_R)
                    z_coarse = np.linspace(1, component.cfg.mesh.N_Z, component.cfg.mesh.N_Z)

                    next_component = self.components[i+1]
                    r_fine = np.linspace(1, next_component.cfg.mesh.N_R, next_component.cfg.mesh.N_R)
                    z_fine = np.linspace(1, next_component.cfg.mesh.N_Z, next_component.cfg.mesh.N_Z)

                    X_out_interpolated = interpolate_2D(
                        r_coarse, z_coarse,
                        r_fine, z_fine,
                        X_out[0]
                    )

                    X_initial = next_component.pack((X_out_interpolated, X_out[-1]))

                if type(component) is NeutronicsModel:
                    r_coarse = np.linspace(1, component.cfg.mesh.N_G, component.cfg.mesh.N_G)
                    z_coarse = np.linspace(1, component.cfg.mesh.N_Z, component.cfg.mesh.N_Z)

                    next_component = self.components[i+1]
                    r_fine = np.linspace(1, next_component.cfg.geometry.r_outer, next_component.cfg.mesh.N_R)
                    z_fine = np.linspace(1, next_component.cfg.geometry.l_tot, next_component.cfg.mesh.N_Z)

                    X_out_interpolated = interpolate_2D(
                        r_coarse, z_coarse,
                        r_fine, z_fine,
                        X_out[0]
                    )

                    X_initial = next_component.pack((X_out_interpolated, X_out[-1]))
        
        self.solution = X_out
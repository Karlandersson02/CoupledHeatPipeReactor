import numpy as np
import meshio

from models.heatpipe.solid_discretised_model import heatpipe_discretised
from models.neutronics.axial_neutron_model import NeutronModel
from models.fuel_pin.fuel_pin_model import FuelPin
from models.mesh_heat_conduction.mesh_conduction_model import moderator_discretised_mesh
from models.mesh_heat_conduction.triangle_mesh import UnstructuredMesh


class Reactor:
    def __init__(self, data):

        # Models used
        self.heat_pipe_thermal_model = heatpipe_discretised(data)
        self.fuel_pin_thermal_model = FuelPin(data)
        self.neutron_flux_model = NeutronModel(data)
        
        if data.get("thermal_resistance") is None:
            mesh = meshio.read("meshHeatConduction/mesh.msh")

            points = mesh.points[:, :2]                
            triangles = mesh.cells_dict["triangle"]     
            mesh = UnstructuredMesh(points, triangles)

            self.moderator_thermal_model = moderator_discretised_mesh(mesh=mesh, data=data)

            self.moderator_eff_res = self.moderator_thermal_model.calculate_effective_resistance()
        else:
            self.moderator_eff_res = data.get("moderator_eff_res")

        # Discretisation
        self.N_Z_HP = data.get("N_evap_HP") + data.get("N_adiabatic_HP") + data.get("N_cond_HP")
        self.N_R_HP = data.get("N_wick_HP") + data.get("N_wall_HP")

        self.N_Z_FP = data.get("N_Z_FP")
        self.N_R_FP = data.get("N_R_FP")

        # Geometry

        # Solver settings
        self.max_iter = data.get("max_iter")

    
    def solve(self, monolithic=True):
        return 42. 
        
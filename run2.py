
import numpy as np
import matplotlib.pyplot as plt
import json

from models.neutronics.axial_neutron_model import NeutronicsModel
from models.heatpipe.solid_discretised_model import HeatpipeDiscretised

from project_data.heatpipe_dataclasses import *
from project_data.neutronics_dataclasses import *

from utils.solver import Solver

mesh = NeutronicsMesh(
    N_R = 50,
    N_Z = 100,
    l = 1
)
energy = NeutronicsEnergy(
    N_G = 8,
    power = 1000
)
cfg = NeutronicsConfig(mesh, energy)

neutronics_model = NeutronicsModel(cfg)

# ---------------

with open("./project_data/vapour_data.json", "r") as f:
    data_guoju = json.load(f)
    data = data_guoju["data_guoju_560"]





geom0 = HeatpipeGeometry(**data["geometry"])
mesh0 = HeatpipeMesh(N_R=25, N_Z=75)
mat0 = HeatpipeMaterial(**data["material"])
bc0 = HeatpipeBC(**data["bc"])
cfg0 = HeatpipeConfig(geom0, mesh0, mat0, bc0)
cfg0 = cfg0.resolve()

geom1 = HeatpipeGeometry(**data["geometry"])
mesh1 = HeatpipeMesh(N_R=50, N_Z=150)
mat1 = HeatpipeMaterial(**data["material"])
bc1 = HeatpipeBC(**data["bc"])
cfg1 = HeatpipeConfig(geom1, mesh1, mat1, bc1)
cfg1 = cfg1.resolve()

geom2 = HeatpipeGeometry(**data["geometry"])
mesh2 = HeatpipeMesh(N_R=100, N_Z=300)
mat2 = HeatpipeMaterial(**data["material"])
bc2 = HeatpipeBC(**data["bc"])
cfg2 = HeatpipeConfig(geom2, mesh2, mat2, bc2)
cfg2 = cfg2.resolve()


heatpipe0 = HeatpipeDiscretised(cfg0)
heatpipe1 = HeatpipeDiscretised(cfg1)
heatpipe2 = HeatpipeDiscretised(cfg2)

solver = Solver([heatpipe0, heatpipe1, heatpipe2])
solver.newton_krylov()

T_solid, T_vapour = solver.solution










N_Z = cfg2.mesh.N_Z
N_R = cfg2.mesh.N_R

T_linear = heatpipe2.linear_solve()

x_linear = np.linspace(0, heatpipe2.cfg.geometry.r_outer, len(T_linear[:-1].reshape(N_Z, N_R)[0]))
x_nonlinear = np.linspace(0, heatpipe1.cfg.geometry.r_outer, len(T_solid[0]))

plt.rcParams["font.size"] = 22
plt.rcParams["font.family"] = "Computer modern"
plt.rcParams["text.usetex"] = True

fig = plt.figure(figsize = (16, 9))

ax = fig.add_subplot(111)
ax.plot(x_nonlinear, T_solid[0], lw=4, label="non-linear")
ax.plot(x_linear, T_linear[:-1].reshape(N_Z, N_R)[0], ls="--", lw=4, label="linear")

plt.legend()
plt.show()
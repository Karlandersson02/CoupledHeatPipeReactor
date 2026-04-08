from project_data.heatpipe_dataclasses import *
import json

with open("./project_data/vapour_data.json", "r") as f:
    data_guoju = json.load(f)
data_guoju_1 = data_guoju["data_Guoju_560"]
data_guoju_1["bc"]["Q"] = 560
    
geom = HeatpipeGeometry(l_evap=0.1, l_adiabatic=0.5, l_tot=1, delta_wall=0.001, r_outer=0.1)
mesh = HeatpipeMesh(N_wick=1, N_R=10, N_evap=10, N_Z=100)
mat = HeatpipeMaterial(**data_guoju_1["material"])
bc = HeatpipeBC(**data_guoju_1["bc"])
cfg = HeatpipeConfig(geom, mesh, mat, bc)
cfg = cfg.resolve()

print(cfg.mesh.N_cond)
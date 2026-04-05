import numpy as np

import utils.sodium_properties as sodium_properties

from models.heatpipe.heat_discretised_model import HeatpipeDiscretised
from models.heatpipe.vapour_discretised_model import VapourDiscretised
from models.heatpipe.liquid_discretised_model import LiquidDiscretised
from utils.solver import Solver

from project_data.heatpipe_dataclasses import *

class Heatpipe:

    def __init__(self, data, config):
        self.cfg = config
        self.solid = HeatpipeDiscretised(self.cfg)

    def setup_fluid_models(self):
        T_HP = self.solid.linear_solve()

        self.vapour = VapourDiscretised(self.cfg, T_HP)
        self.liquid = LiquidDiscretised(self.cfg, T_HP)

    def get_heatpipe_temperature(self):
        if hasattr(self, "heatpipe_T"):
            T = self.heatpipe_T
        else:
            T = self.solid.linear_solve()
            self.heatpipe_T = T

        return T
    
    def get_vapour_temperature(self):
        if hasattr(self, "vapour_T"):
            T = self.vapour_T
        else:
            solver = Solver([self.vapour])
            solver.newton_krylov()
            T = solver.solution[1]
            P = sodium_properties.calculate_Na_pressure_v(T)

            self.vapour_T = T
            self.vapour_P_numeric = P

        return T
    
    def get_vapour_pressure_drop_profile_numeric(self):
        if hasattr(self, "vapour_P_numeric"):
            P = self.vapour_P_numeric
        else:
            solver = Solver([self.vapour])
            solver.newton_krylov()
            T = solver.solution[1]
            P = sodium_properties.calculate_Na_pressure_v(T)
            self.vapour_T = T
            self.vapour_P_numeric = P

        return P
    
    def get_liquid_pressure_drop_profile(self):
        if hasattr(self, "liquid_P"):
            P = self.liquid_P
        else:
            P = self.liquid.get_pressure_drop_profile()
            self.liquid_P = P

        return P
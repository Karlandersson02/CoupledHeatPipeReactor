
from models.heat_discretised_model import heatpipe_discretised
from models.vapor_discretised_model import vapour_discretised
from models.liquid_discretised_model import liquid_discretised

class Heatpipe:

    def __init__(self, data):
        self.data = data

        self.heatpipe_discretised = heatpipe_discretised(data)
        self.vapour_discretised = vapour_discretised(data)
        self.liquid_discretised = liquid_discretised(data)

        self.calculated_quantities = dict()

    def get_heatpipe_temperature(self):
        T = self.calculated_quantities.get("heatpipe_T")
        if T is None:
            self.heatpipe_discretised.solve()
            T = self.heatpipe_discretised.temperature
            self.calculated_quantities["heatpipe_T"] = T
        return T
    
    def get_vapour_temperature(self):
        T = self.calculated_quantities.get("vapour_T")
        if T is None:
            self.vapour_discretised.solve()
            T = self.vapour_discretised.temperature
            self.calculated_quantities["vapour_T"] = T
        return T
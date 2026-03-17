import numpy as np

class NeutronModel:

    def __init__(self):

        # geometry

        # system properties

        # material properties

        raise NotImplementedError

    def solve_neutron_flux(self):

        self.initialise_discretisation()

        self.calculate_abc()

        self.calculate_residuals()

    def initialise_discretisation(self):
        raise NotImplementedError
    
    def calculate_abc(self):
        raise NotImplementedError
    
    def calculate_residuals(self):
        raise NotImplementedError
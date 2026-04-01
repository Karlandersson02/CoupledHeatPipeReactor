from dataclasses import dataclass
import numpy as np

@dataclass(slots=True, kw_only=True)
class NeutronicsMesh:
    N_R: int
    N_Z: int
    l: float
    
    @property
    def delta_Z(self):
        return self.l / self.N_Z
    
    @property
    def cross_sectional_area(self):
        return 0.01**2 * np.pi  # temp

@dataclass(slots=True, kw_only=True)
class NeutronicsEnergy:
    N_G: int
    power: float | int

@dataclass(slots=True)
class NeutronicsConfig:
    mesh: NeutronicsMesh
    energy: NeutronicsEnergy
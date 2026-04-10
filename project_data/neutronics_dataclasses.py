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

# Fuel Pin ----------------------------

@dataclass(slots=True, kw_only=True)
class FuelPinGeometry:
    l: int | float
    r: int | float
    delta_gap : int | float
    delta_wall: int | float

    @property
    def r_fuel(self):
        return self.r - self.delta_gap - self.delta_wall

@dataclass(slots=True, kw_only=True)
class FuelPinMesh:
    N_R: int
    N_Z: int

@dataclass(slots=True, kw_only=True)
class FuelPinMeshResolved:
    N_fuel : int
    N_gap  : int
    N_wall : int
    N_R    : int
    N_Z    : int

@dataclass(slots=True, kw_only=True)
class FuelPinEnergy:
    N_G: int

@dataclass(slots=True, kw_only=True)
class FuelPinMaterial:
    k_fuel: int | float
    k_clad: int | float
    k_gap : int | float
    h_gap : int | float
    h_mod : int | float


@dataclass(slots=True)
class FuelPinConfig:
    geometry: FuelPinGeometry
    mesh    : FuelPinMesh
    energy  : FuelPinEnergy
    material: FuelPinMaterial

    def resolve(self):

        r_fuel = self.geometry.r - self.geometry.delta_gap - self.geometry.delta_wall
        
        lengths = np.array([
            r_fuel,
            self.geometry.delta_gap,
            self.geometry.delta_wall
        ], dtype=float)

        # Normalize to proportions
        proportions = lengths / lengths.sum()

        # Ideal (non-integer) counts
        raw = proportions * self.mesh.N_R

        # Base integer part
        counts = np.floor(raw).astype(int)

        # Remaining cells to distribute
        remainder = self.mesh.N_R - counts.sum()

        # Distribute to largest fractional parts
        fractional = raw - counts
        indices = np.argsort(fractional)[::-1]

        for i in range(remainder):
            counts[indices[i]] += 1

        N_fuel, N_gap, N_wall = counts

        delta_gap  = N_gap  / self.mesh.N_R * self.geometry.r
        delta_wall = N_wall / self.mesh.N_R * self.geometry.r
        r_fuel_new = N_fuel / self.mesh.N_R * self.geometry.r

        if not np.isclose(r_fuel, r_fuel_new):
            print("\033[33mGeometry warning: \033[0m", f"r_fuel changed from {r_fuel} to {r_fuel_new}")
        if not np.isclose(self.geometry.delta_gap, delta_gap):
            print("\033[33mGeometry warning: \033[0m", f"delta_gap changed from {self.geometry.delta_gap} to {delta_gap}")
        if not np.isclose(self.geometry.delta_wall, delta_wall):
            print("\033[33mGeometry warning: \033[0m", f"delta_wall changed from {self.geometry.delta_wall} to {delta_wall}")

        self.geometry.delta_gap  = delta_gap
        self.geometry.delta_wall = delta_wall
        self.geometry.r          = r_fuel + delta_gap + delta_wall

        mesh_resolved = FuelPinMeshResolved(
            N_fuel = N_fuel,
            N_gap  = N_gap,
            N_wall = N_wall,
            N_R    = self.mesh.N_R,
            N_Z    = self.mesh.N_Z
        )   

        return FuelPinConfigResolved(
            self.geometry,
            mesh_resolved,
            self.energy,
            self.material
        )
    
@dataclass(slots=True)
class FuelPinConfigResolved:
    geometry: FuelPinGeometry
    mesh    : FuelPinMeshResolved
    energy  : FuelPinEnergy
    material: FuelPinMaterial
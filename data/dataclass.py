import numpy as np
from dataclasses import dataclass

# Heatpipe ------------------------

@dataclass(slots=True, kw_only=True)
class HeatpipeGeometry:
    r_vapour: float
    delta_wick: float
    delta_gap: float
    delta_wall: float

    l_evap: float
    l_adiabatic: float
    l_cond: float

    @property
    def r_wick(self) -> float:
        return self.r_vapour + self.delta_wick
    
    @property
    def r_gap(self) -> float:
        return self.r_vapour + self.delta_wick + self.delta_gap

    @property
    def r_outer(self) -> float:
        return self.r_vapour + self.delta_wick + self.delta_gap + self.delta_wall

    @property
    def l_tot(self) -> float:
        return self.l_evap + self.l_adiabatic + self.l_cond

@dataclass(slots=True)
class HeatpipeMesh:
    N_R: int
    N_Z: int

    
@dataclass(slots=True)
class HeatpipeMeshResolved:
    N_wick: int
    N_gap: int
    N_wall: int

    N_evap: int
    N_adiabatic: int
    N_cond: int

    @property
    def N_R(self):
        return self.N_wick + self.N_gap + self.N_wall

    @property
    def N_Z(self):
        return self.N_evap + self.N_adiabatic + self.N_cond

@dataclass(slots=True, kw_only=True)
class HeatpipeMaterial:
    k_wall: float
    k_gap: float
    k_wick: float
    h_vap: float
    h_cond: float

@dataclass(slots=True, kw_only=True)
class HeatpipeBC:
    Temperature_BC: bool
    T_op: float
    T_cond: float
    Q: float

@dataclass(slots=True, kw_only=True)
class HeatpipeBCResolved:
    Temperature_BC: bool
    T_op: float
    T_cond: float
    Q: np.ndarray


@dataclass(slots=True, kw_only=True)
class HeatpipeWick:
    r_pore: float
    porosity: float
    K: float
    Is_annular: bool


@dataclass(slots=True)
class HeatpipeConfig:
    geometry: HeatpipeGeometry
    mesh: HeatpipeMesh
    material: HeatpipeMaterial
    wick: HeatpipeWick
    bc: HeatpipeBC

    def resolve(self):
        def allocate_counts(lengths: np.ndarray, total_cells: int) -> np.ndarray:
            lengths = np.asarray(lengths, dtype=float)

            if total_cells <= 0:
                raise ValueError("total_cells must be positive.")
            if np.any(lengths < 0):
                raise ValueError("All lengths must be non-negative.")
            if np.isclose(lengths.sum(), 0.0):
                raise ValueError("The provided lengths must not all be zero.")

            proportions = lengths / lengths.sum()
            raw = proportions * total_cells
            counts = np.floor(raw).astype(int)

            remainder = total_cells - counts.sum()
            fractional = raw - counts
            indices = np.argsort(fractional)[::-1]

            for i in range(remainder):
                counts[indices[i]] += 1

            return counts

        # -----------------------
        # Resolve radial mesh from radial geometry
        radial_lengths = np.array([
            self.geometry.delta_wick,
            self.geometry.delta_gap,
            self.geometry.delta_wall,
        ], dtype=float)

        N_wick, N_gap, N_wall = allocate_counts(radial_lengths, self.mesh.N_R)

        radial_total = self.geometry.r_outer - self.geometry.r_vapour
        delta_wick = N_wick / self.mesh.N_R * radial_total
        delta_gap  = N_gap  / self.mesh.N_R * radial_total
        delta_wall = N_wall / self.mesh.N_R * radial_total

        if not np.isclose(delta_wick, self.geometry.delta_wick):
            print("\033[33mGeometry warning:\033[0m",
                  f"delta_wick changed from {self.geometry.delta_wick} to {delta_wick}")
        if not np.isclose(delta_gap, self.geometry.delta_gap):
            print("\033[33mGeometry warning:\033[0m",
                  f"delta_gap changed from {self.geometry.delta_gap} to {delta_gap}")
        if not np.isclose(delta_wall, self.geometry.delta_wall):
            print("\033[33mGeometry warning:\033[0m",
                  f"delta_wall changed from {self.geometry.delta_wall} to {delta_wall}")

        self.geometry.delta_wick = delta_wick
        self.geometry.delta_gap  = delta_gap
        self.geometry.delta_wall = delta_wall

        # -----------------------
        # Resolve axial mesh from axial geometry
        axial_lengths = np.array([
            self.geometry.l_evap,
            self.geometry.l_adiabatic,
            self.geometry.l_cond,
        ], dtype=float)

        N_evap, N_adiabatic, N_cond = allocate_counts(axial_lengths, self.mesh.N_Z)

        axial_total = self.geometry.l_tot
        l_evap      = N_evap      / self.mesh.N_Z * axial_total
        l_adiabatic = N_adiabatic / self.mesh.N_Z * axial_total
        l_cond      = N_cond      / self.mesh.N_Z * axial_total

        if not np.isclose(l_evap, self.geometry.l_evap):
            print("\033[33mGeometry warning:\033[0m",
                  f"l_evap changed from {self.geometry.l_evap} to {l_evap}")
        if not np.isclose(l_adiabatic, self.geometry.l_adiabatic):
            print("\033[33mGeometry warning:\033[0m",
                  f"l_adiabatic changed from {self.geometry.l_adiabatic} to {l_adiabatic}")
        if not np.isclose(l_cond, self.geometry.l_cond):
            print("\033[33mGeometry warning:\033[0m",
                  f"l_cond changed from {self.geometry.l_cond} to {l_cond}")

        self.geometry.l_evap = l_evap
        self.geometry.l_adiabatic = l_adiabatic
        self.geometry.l_cond = l_cond

        # -----------------------
        # Resolve Q
        Q = self.bc.Q
        if isinstance(Q, (float, int)):
            Q = np.repeat(np.array([Q / N_evap], dtype=float), N_evap)

        if not self.bc.Temperature_BC:
            Qout = -np.ones(N_cond) * np.sum(Q) / N_cond

            Qnew = np.zeros(N_evap + N_adiabatic + N_cond)
            Qnew[:N_evap] = Q
            Qnew[(N_evap + N_adiabatic):] = Qout
            Q = Qnew

        mesh_resolved = HeatpipeMeshResolved(
            N_wick      = N_wick,
            N_gap       = N_gap,
            N_wall      = N_wall,
            N_evap      = N_evap,
            N_adiabatic = N_adiabatic,
            N_cond      = N_cond,
        )

        bc_resolved = HeatpipeBCResolved(
            Temperature_BC = self.bc.Temperature_BC,
            T_op           = self.bc.T_op,
            T_cond         = self.bc.T_cond,
            Q              = Q,
        )

        return HeatpipeConfigResolved(
            geometry = self.geometry,
            mesh     = mesh_resolved,
            material = self.material,
            wick     = self.wick,
            bc       = bc_resolved,
        )
    

@dataclass(slots=True)
class HeatpipeConfigResolved:
    geometry: HeatpipeGeometry
    mesh: HeatpipeMeshResolved
    material: HeatpipeMaterial
    wick: HeatpipeWick
    bc: HeatpipeBCResolved


# Neutronics ------------------------

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
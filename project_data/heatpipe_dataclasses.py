import numpy as np
from dataclasses import dataclass


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


# @dataclass(slots=True)
# class VapourConfig:
#     geometry: HeatpipeGeometry
#     mesh: HeatpipeMesh
#     material: HeatpipeMaterial

#     def resolve(self):
#         return VapourConfigResolved(
#             geometry = self.geometry,
#             mesh = self.mesh,
#             material = self.material,
#         )

# @dataclass(slots=True)
# class LiquidConfig:
#     geometry: HeatpipeGeometry
#     mesh: HeatpipeMesh
#     material: HeatpipeMaterial
#     wick: HeatpipeWick

#     def resolve(self):
#         geometry = self.geometry.resolve()
#         mesh = self.mesh.resolve()
#         material = self.material.resolve()
#         wick = self.wick.resolve()

#         return LiquidConfigResolved(
#             geometry = geometry,
#             mesh = mesh,
#             material = material,
#             wick = wick,
#         )


# @dataclass(slots=True)
# class VapourConfigResolved:
#     geometry: HeatpipeGeometry
#     mesh: HeatpipeMeshResolved
#     material: HeatpipeMaterial

# @dataclass(slots=True)
# class LiquidConfigResolved:
#     geometry: HeatpipeGeometry
#     mesh: HeatpipeMeshResolved
#     material: HeatpipeMaterial
#     wick: HeatpipeWick
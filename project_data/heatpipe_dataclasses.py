import numpy as np
from dataclasses import dataclass

# ---------------------------------------------
# Data classes used by the user
# ---------------------------------------------

@dataclass(slots=True, kw_only=True)
class HeatpipeGeometry:
    r_vapour: float | None = None
    delta_wick: float | None = None
    delta_gap: float | None = None
    delta_wall: float | None = None
    r_outer: float | None = None

    l_evap: float | None = None
    l_adiabatic: float | None = None
    l_cond: float | None = None
    l_tot: float | None = None

    def resolve(self):
        r_vapour, delta_wick, delta_gap, delta_wall = self._resolve_group(
            values=(self.r_vapour, self.delta_wick, self.delta_gap, self.delta_wall),
            total=self.r_outer,
            names=("r_vapour", "delta_wick", "delta_wall"),
            total_name="r_outer",
        )

        l_evap, l_adiabatic, l_cond = self._resolve_group(
            values=(self.l_evap, self.l_adiabatic, self.l_cond),
            total=self.l_tot,
            names=("l_evap", "l_adiabatic", "l_cond"),
            total_name="l_tot",
        )

        return HeatpipeGeometryResolved(
            r_vapour=r_vapour,
            delta_wick=delta_wick,
            delta_gap=delta_gap,
            delta_wall=delta_wall,
            l_evap=l_evap,
            l_adiabatic=l_adiabatic,
            l_cond=l_cond,
        )

    @staticmethod
    def _resolve_group(
        *,
        values: tuple[float | None, ...],
        total: float | None,
        names: tuple[str, ...],
        total_name: str,
    ) -> tuple[float, ...]:
        n_parts = len(values)

        for name, value in zip(names, values):
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive if provided.")

        if total is not None and total <= 0:
            raise ValueError(f"{total_name} must be positive if provided.")

        n_given = sum(v is not None for v in values)
        all_given = n_given == n_parts

        if all_given:
            if total is not None:
                raise ValueError(
                    f"Provide either all of {names} or {total_name}, not both."
                )
            return tuple(values)

        if total is None:
            raise ValueError(
                f"If not all of {names} are provided, you must provide {total_name}."
            )

        known_sum = sum(v for v in values if v is not None)
        n_missing = n_parts - n_given
        remainder = total - known_sum

        if remainder <= 0:
            raise ValueError(
                f"{total_name} is too small given the specified values in {names}."
            )

        fill_value = remainder / n_missing

        resolved = tuple(
            v if v is not None else fill_value
            for v in values
        )

        return resolved

@dataclass(slots=True, kw_only=True)
class HeatpipeMesh:
    N_wick: int | None = None
    N_wall: int | None = None
    N_gap: int | None = None
    N_evap: int | None = None
    N_adiabatic: int | None = None
    N_cond: int | None = None
    N_R: int | None = None
    N_Z: int | None = None

    def resolve(self):
        N_wick, N_gap, N_wall = self._resolve_group(
            values=(self.N_wick, self.N_gap, self.N_wall),
            total=self.N_R,
            names=("N_wick", "N_wall"),
            total_name="N_R",
        )

        N_evap, N_adiabatic, N_cond = self._resolve_group(
            values=(self.N_evap, self.N_adiabatic, self.N_cond),
            total=self.N_Z,
            names=("N_evap", "N_adiabatic", "N_cond"),
            total_name="N_Z",
        )

        return HeatpipeMeshResolved(
            N_wick=N_wick,
            N_gap=N_gap,
            N_wall=N_wall,
            N_evap=N_evap,
            N_adiabatic=N_adiabatic,
            N_cond=N_cond,
        )
    
    @staticmethod
    def _resolve_group(
        *,
        values: tuple[int | None, ...],
        total: int | None,
        names: tuple[str, ...],
        total_name: str,
    ) -> tuple[int, ...]:
        n_parts = len(values)

        # Basic validation of provided entries
        for name, value in zip(names, values):
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive if provided.")

        if total is not None and total <= 0:
            raise ValueError(f"{total_name} must be positive if provided.")

        n_given = sum(v is not None for v in values)
        all_given = n_given == n_parts
        none_given = n_given == 0

        # Case 1: all parts given
        if all_given:
            if total is not None:
                raise ValueError(
                    f"Provide either all of {names} or {total_name}, not both."
                )
            return tuple(values)

        # Case 2: total must be given if not all parts are given
        if total is None:
            raise ValueError(
                f"If not all of {names} are provided, you must provide {total_name}."
            )

        # Split remainder among missing parts
        known_sum = sum(v for v in values if v is not None)
        n_missing = n_parts - n_given
        remainder = total - known_sum

        if remainder < n_missing:
            raise ValueError(
                f"{total_name}={total} is too small given the specified values; "
                f"each unspecified entry in {names} must receive at least 1."
            )

        base = remainder // n_missing
        extra = remainder % n_missing

        resolved = []
        missing_seen = 0

        for value in values:
            if value is not None:
                resolved.append(value)
            else:
                fill = base + (1 if missing_seen < extra else 0)
                resolved.append(fill)
                missing_seen += 1

        return tuple(resolved)

@dataclass(slots=True, kw_only=True)
class HeatpipeMaterial:
    k_wall: float
    k_gap: float
    k_wick: float
    h_vap: float
    h_cond: float

    def resolve(self):
        return HeatpipeMaterialResolved(
            k_wick = self.k_wick,
            k_gap = self.k_gap,
            k_wall = self.k_wall,
            h_vap = self.h_vap,
            h_cond = self.h_cond,
        )

@dataclass(slots=True, kw_only=True)
class HeatpipeBC:
    Temperature_BC: bool
    T_op: float
    T_cond: float
    Q: np.ndarray | float | int

    def resolve(self):
        return
    

@dataclass(slots=True, kw_only=True)
class HeatpipeWick:
    r_pore: float
    porosity: float
    K: float
    Is_annular: bool

    def resolve(self):
        return HeatpipeWickResolved(
            r_pore = self.r_pore,
            porosity = self.porosity,
            K = self.K,
            Is_annular = self.Is_annular
        )
    
@dataclass(slots=True)
class HeatpipeConfig:
    geometry: HeatpipeGeometry
    mesh: HeatpipeMesh
    material: HeatpipeMaterial
    bc: HeatpipeBC
    wick: HeatpipeWick

    def resolve(self):
        geometry = self.geometry.resolve()
        mesh = self.mesh.resolve()
        material = self.material.resolve()
        wick = self.wick.resolve()

        Q = self.bc.Q
        if type(Q) is float or type(Q) is int:
            Qnew = np.repeat(np.array([Q / mesh.N_evap], dtype=float), mesh.N_evap)
            Q = Qnew

        if not self.bc.Temperature_BC:
            Qout = -np.ones(mesh.N_cond) * np.sum(Q) / mesh.N_cond

            Qnew = np.zeros(mesh.N_Z)
            Qnew[:mesh.N_evap] = Q
            Qnew[(mesh.N_evap + mesh.N_adiabatic):] = Qout
            Q = Qnew

        bc = HeatpipeBCResolved(
            Temperature_BC = self.bc.Temperature_BC,
            T_op = self.bc.T_op,
            T_cond = self.bc.T_cond,
            Q = Q,
        )

        return HeatpipeConfigResolved(
            geometry = geometry,
            mesh = mesh,
            material = material,
            bc = bc,
            wick = wick,
        )
    
@dataclass(slots=True, kw_only=True)
class VapourBC:
    T_HP: np.ndarray

    def resolve(self):
        return VapourBCResolved(T_HP = self.T_HP)
    
@dataclass(slots=True)
class VapourConfig:
    geometry: HeatpipeGeometry
    mesh: HeatpipeMesh
    material: HeatpipeMaterial
    vapour_bc: VapourBC

    def resolve(self):
        geometry = self.geometry.resolve()
        mesh = self.mesh.resolve()
        material = self.material.resolve()
        vapour_bc = self.vapour_bc.resolve()

        return VapourConfigResolved(
            geometry = geometry,
            mesh = mesh,
            material = material,
            vapour_bc = vapour_bc,
        )

@dataclass(slots=True)
class LiquidConfig:
    geometry: HeatpipeGeometry
    mesh: HeatpipeMesh
    material: HeatpipeMaterial
    wick: HeatpipeWick
    vapour_bc: VapourBC

    def resolve(self):
        geometry = self.geometry.resolve()
        mesh = self.mesh.resolve()
        material = self.material.resolve()
        wick = self.wick.resolve()
        vapour_bc = self.vapour_bc.resolve()

        return LiquidConfigResolved(
            geometry = geometry,
            mesh = mesh,
            material = material,
            wick = wick,
            vapour_bc = vapour_bc,
        )

# ---------------------------------------------
# Resolved data classes
# ---------------------------------------------

@dataclass(slots=True, kw_only=True)
class HeatpipeGeometryResolved:
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
class HeatpipeMaterialResolved:
    k_wall: float
    k_wick: float
    k_gap: float
    h_vap: float
    h_cond: float

@dataclass(slots=True, kw_only=True)
class HeatpipeBCResolved:
    Temperature_BC: bool
    T_op: float
    T_cond: float
    Q: np.ndarray


@dataclass(slots=True, kw_only=True)
class HeatpipeWickResolved:
    r_pore: float
    porosity: float
    K: float
    Is_annular: bool
    
@dataclass(slots=True)
class HeatpipeConfigResolved:
    geometry: HeatpipeGeometryResolved
    mesh: HeatpipeMeshResolved
    material: HeatpipeMaterialResolved
    bc: HeatpipeBCResolved
    wick: HeatpipeWickResolved

@dataclass(slots=True, kw_only=True)
class VapourBCResolved:
    T_HP: np.ndarray

@dataclass(slots=True)
class VapourConfigResolved:
    geometry: HeatpipeGeometryResolved
    mesh: HeatpipeMeshResolved
    material: HeatpipeMaterialResolved
    vapour_bc: VapourBCResolved

@dataclass(slots=True)
class LiquidConfigResolved:
    geometry: HeatpipeGeometryResolved
    mesh: HeatpipeMeshResolved
    material: HeatpipeMaterialResolved
    wick: HeatpipeWickResolved
    vapour_bc: VapourBCResolved
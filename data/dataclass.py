import math
import numpy as np
from dataclasses import dataclass
from fractions import Fraction

from copy import deepcopy
from typing import Union

def _lcm(a: int, b: int) -> int:
    return abs(a * b) // math.gcd(a, b)


def _lcm_many(values) -> int:
    out = 1
    for v in values:
        out = _lcm(out, int(v))
    return out


def _minimal_compatible_total(lengths: np.ndarray, ndigits: int = 12) -> int:
    lengths = np.asarray(lengths, dtype=float)

    if np.any(lengths < 0):
        raise ValueError("All lengths must be non-negative.")
    if np.isclose(lengths.sum(), 0.0):
        raise ValueError("The provided lengths must not all be zero.")

    lengths = np.round(lengths, ndigits)

    lengths_frac = [Fraction(str(x)).limit_denominator() for x in lengths]
    total_frac = sum(lengths_frac)

    proportions = [x / total_frac for x in lengths_frac]
    denominators = [p.denominator for p in proportions]

    return _lcm_many(denominators)


def _snap_to_compatible_total(requested_total: int, lengths: np.ndarray) -> int:
    """
    Snap requested_total to the nearest compatible total cell count.
    In a tie, prefer the higher value.
    """
    if requested_total <= 0:
        raise ValueError("requested_total must be positive.")

    base = _minimal_compatible_total(lengths)

    lower = (requested_total // base) * base
    upper = lower if lower == requested_total else lower + base

    if lower <= 0:
        return upper

    d_lower = requested_total - lower
    d_upper = upper - requested_total

    if d_upper <= d_lower:
        return upper
    return lower


def _allocate_counts(lengths: np.ndarray, total_cells: int) -> np.ndarray:
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


def _allocate_counts_quadratic(lengths: np.ndarray, total_cells: int) -> np.ndarray:
    """
    Allocate radial cell counts for a mesh that is uniform in r^2.

    This is intended for the radial discretisation used in FuelPin.initialize_discretization(),
    where the outer face radius after n cells is approximately

        R_n = R_outer * sqrt(n / total_cells)

    Therefore, a physical interface at radius R_i should be placed near

        n_i = total_cells * (R_i / R_outer)^2

    Parameters
    ----------
    lengths : np.ndarray
        Radial layer thicknesses, ordered from the centre outward.
        For the fuel pin this should be:

            [r_fuel, delta_gap, delta_wall]

    total_cells : int
        Total number of radial cells.

    Returns
    -------
    counts : np.ndarray
        Integer cell counts per layer.
    """

    lengths = np.asarray(lengths, dtype=float)

    if total_cells <= 0:
        raise ValueError("total_cells must be positive.")

    if np.any(lengths < 0.0):
        raise ValueError("All lengths must be non-negative.")

    if np.isclose(lengths.sum(), 0.0):
        raise ValueError("The provided lengths must not all be zero.")

    n_layers = len(lengths)

    if np.count_nonzero(lengths > 0.0) > total_cells:
        raise ValueError(
            "There are more non-zero radial layers than available radial cells."
        )

    # Radii of layer interfaces:
    # [0, R_fuel, R_gap_outer, R_clad_outer]
    radii = np.concatenate(([0.0], np.cumsum(lengths)))

    R_outer = radii[-1]

    # Target mesh-interface indices for internal layer boundaries.
    # Boundary index n means: after n radial cells.
    raw_boundary_indices = total_cells * (radii[1:-1] / R_outer) ** 2

    boundary_indices = []
    previous = 0

    for i, raw_idx in enumerate(raw_boundary_indices):
        layer_idx = i

        # Leave at least one cell for every remaining non-zero layer.
        remaining_nonzero_layers = np.count_nonzero(lengths[layer_idx + 1:] > 0.0)

        if lengths[layer_idx] > 0.0:
            lower = previous + 1
        else:
            lower = previous

        upper = total_cells - remaining_nonzero_layers

        idx = int(round(raw_idx))
        idx = max(lower, min(idx, upper))

        boundary_indices.append(idx)
        previous = idx

    boundary_indices = np.array(boundary_indices, dtype=int)

    counts = np.diff(
        np.concatenate(([0], boundary_indices, [total_cells]))
    )

    return counts


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

    def resolve_mesh(self):
        radial_lengths = np.array([
            self.geometry.delta_wick,
            self.geometry.delta_gap,
            self.geometry.delta_wall,
        ], dtype=float)

        axial_lengths = np.array([
            self.geometry.l_evap,
            self.geometry.l_adiabatic,
            self.geometry.l_cond,
        ], dtype=float)

        N_R_old = self.mesh.N_R
        N_Z_old = self.mesh.N_Z

        N_R_new = _snap_to_compatible_total(N_R_old, radial_lengths)
        N_Z_new = _snap_to_compatible_total(N_Z_old, axial_lengths)

        if N_R_new != N_R_old:
            print(
                "\033[33mMesh warning:\033[0m",
                f"N_R changed from {N_R_old} to {N_R_new}"
            )
        if N_Z_new != N_Z_old:
            print(
                "\033[33mMesh warning:\033[0m",
                f"N_Z changed from {N_Z_old} to {N_Z_new}"
            )

        N_wick, N_gap, N_wall = _allocate_counts(radial_lengths, N_R_new)
        N_evap, N_adiabatic, N_cond = _allocate_counts(axial_lengths, N_Z_new)

        self.mesh.N_R = N_R_new
        self.mesh.N_Z = N_Z_new

        mesh_resolved = HeatpipeMeshResolved(
            N_wick=N_wick,
            N_gap=N_gap,
            N_wall=N_wall,
            N_evap=N_evap,
            N_adiabatic=N_adiabatic,
            N_cond=N_cond,
        )

        Q = self.bc.Q
        if isinstance(Q, (float, int)):
            Q = np.repeat(np.array([Q / N_evap], dtype=float), N_evap)

        if not self.bc.Temperature_BC:
            Qout = -np.ones(N_cond) * np.sum(Q) / N_cond

            Qnew = np.zeros(N_evap + N_adiabatic + N_cond)
            Qnew[:N_evap] = Q
            Qnew[(N_evap + N_adiabatic):] = Qout
            Q = Qnew

        bc_resolved = HeatpipeBCResolved(
            Temperature_BC=self.bc.Temperature_BC,
            T_op=self.bc.T_op,
            T_cond=self.bc.T_cond,
            Q=Q,
        )

        return HeatpipeConfigResolved(
            geometry=self.geometry,
            mesh=mesh_resolved,
            material=self.material,
            wick=self.wick,
            bc=bc_resolved,
        )

    def resolve_geometry(self):
        # self.resolve_mesh()

        radial_lengths = np.array([
            self.geometry.delta_wick,
            self.geometry.delta_gap,
            self.geometry.delta_wall,
        ], dtype=float)

        N_wick, N_gap, N_wall = _allocate_counts(radial_lengths, self.mesh.N_R)

        radial_total = self.geometry.r_outer - self.geometry.r_vapour
        delta_wick = N_wick / self.mesh.N_R * radial_total
        delta_gap  = N_gap  / self.mesh.N_R * radial_total
        delta_wall = N_wall / self.mesh.N_R * radial_total

        if not np.isclose(delta_wick, self.geometry.delta_wick):
            print(
                "\033[33mGeometry warning:\033[0m",
                f"delta_wick changed from {self.geometry.delta_wick} to {delta_wick}"
            )
        if not np.isclose(delta_gap, self.geometry.delta_gap):
            print(
                "\033[33mGeometry warning:\033[0m",
                f"delta_gap changed from {self.geometry.delta_gap} to {delta_gap}"
            )
        if not np.isclose(delta_wall, self.geometry.delta_wall):
            print(
                "\033[33mGeometry warning:\033[0m",
                f"delta_wall changed from {self.geometry.delta_wall} to {delta_wall}"
            )

        self.geometry.delta_wick = delta_wick
        self.geometry.delta_gap  = delta_gap
        self.geometry.delta_wall = delta_wall

        axial_lengths = np.array([
            self.geometry.l_evap,
            self.geometry.l_adiabatic,
            self.geometry.l_cond,
        ], dtype=float)

        N_evap, N_adiabatic, N_cond = _allocate_counts(axial_lengths, self.mesh.N_Z)

        axial_total = self.geometry.l_tot
        l_evap      = N_evap      / self.mesh.N_Z * axial_total
        l_adiabatic = N_adiabatic / self.mesh.N_Z * axial_total
        l_cond      = N_cond      / self.mesh.N_Z * axial_total

        if not np.isclose(l_evap, self.geometry.l_evap):
            print(
                "\033[33mGeometry warning:\033[0m",
                f"l_evap changed from {self.geometry.l_evap} to {l_evap}"
            )
        if not np.isclose(l_adiabatic, self.geometry.l_adiabatic):
            print(
                "\033[33mGeometry warning:\033[0m",
                f"l_adiabatic changed from {self.geometry.l_adiabatic} to {l_adiabatic}"
            )
        if not np.isclose(l_cond, self.geometry.l_cond):
            print(
                "\033[33mGeometry warning:\033[0m",
                f"l_cond changed from {self.geometry.l_cond} to {l_cond}"
            )

        self.geometry.l_evap = l_evap
        self.geometry.l_adiabatic = l_adiabatic
        self.geometry.l_cond = l_cond

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
            geometry=self.geometry,
            mesh=mesh_resolved,
            material=self.material,
            wick=self.wick,
            bc=bc_resolved,
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
        return 0.01**2 * np.pi


@dataclass(slots=True, kw_only=True)
class NeutronicsEnergy:
    N_G: int
    power: float | int


@dataclass(slots=True)
class NeutronicsConfig:
    mesh: NeutronicsMesh
    energy: NeutronicsEnergy

@dataclass(slots=True)
class NeutronicsConfigResolved:
    mesh: NeutronicsMesh
    energy: NeutronicsEnergy


# Fuel Pin ----------------------------

@dataclass(slots=True, kw_only=True)
class FuelPinGeometry:
    l: int | float
    r: int | float
    delta_gap: int | float
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
    N_fuel: int
    N_gap: int
    N_wall: int
    N_R: int
    N_Z: int


@dataclass(slots=True, kw_only=True)
class FuelPinEnergy:
    N_G: int


@dataclass(slots=True, kw_only=True)
class FuelPinMaterial:
    k_fuel: int | float
    k_clad: int | float
    h_gap: int | float
    h_mod: int | float


@dataclass(slots=True)
class FuelPinConfig:
    geometry: FuelPinGeometry
    mesh: FuelPinMesh
    energy: FuelPinEnergy
    material: FuelPinMaterial

    def resolve_mesh(self):
        radial_lengths = np.array([
            self.geometry.r_fuel,
            self.geometry.delta_gap,
            self.geometry.delta_wall,
        ], dtype=float)

        N_R_old = self.mesh.N_R
        N_R_new = _snap_to_compatible_total(N_R_old, radial_lengths)

        if N_R_new != N_R_old:
            print(
                "\033[33mMesh warning:\033[0m",
                f"N_R changed from {N_R_old} to {N_R_new}"
            )

        N_fuel, N_gap, N_wall = _allocate_counts_quadratic(radial_lengths, N_R_new)

        self.mesh.N_R = N_R_new

        mesh_resolved = FuelPinMeshResolved(
            N_fuel=N_fuel,
            N_gap=N_gap,
            N_wall=N_wall,
            N_R=self.mesh.N_R,
            N_Z=self.mesh.N_Z,
        )

        return FuelPinConfigResolved(
            geometry=self.geometry,
            mesh=mesh_resolved,
            energy=self.energy,
            material=self.material,
        )

    def resolve_geometry(self):
        # self.resolve_mesh()

        r_fuel = self.geometry.r - self.geometry.delta_gap - self.geometry.delta_wall
        
        lengths = np.array([
            r_fuel,
            self.geometry.delta_gap,
            self.geometry.delta_wall
        ], dtype=float)

        N_fuel, N_gap, N_wall = _allocate_counts(lengths, self.mesh.N_R)

        delta_gap  = N_gap  / self.mesh.N_R * self.geometry.r
        delta_wall = N_wall / self.mesh.N_R * self.geometry.r
        r_fuel_new = N_fuel / self.mesh.N_R * self.geometry.r

        if not np.isclose(r_fuel, r_fuel_new):
            print(
                "\033[33mGeometry warning:\033[0m",
                f"r_fuel changed from {r_fuel} to {r_fuel_new}"
            )
        if not np.isclose(self.geometry.delta_gap, delta_gap):
            print(
                "\033[33mGeometry warning:\033[0m",
                f"delta_gap changed from {self.geometry.delta_gap} to {delta_gap}"
            )
        if not np.isclose(self.geometry.delta_wall, delta_wall):
            print(
                "\033[33mGeometry warning:\033[0m",
                f"delta_wall changed from {self.geometry.delta_wall} to {delta_wall}"
            )

        self.geometry.delta_gap  = delta_gap
        self.geometry.delta_wall = delta_wall
        self.geometry.r          = r_fuel_new + delta_gap + delta_wall

        mesh_resolved = FuelPinMeshResolved(
            N_fuel=N_fuel,
            N_gap=N_gap,
            N_wall=N_wall,
            N_R=self.mesh.N_R,
            N_Z=self.mesh.N_Z
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
    mesh: FuelPinMeshResolved
    energy: FuelPinEnergy
    material: FuelPinMaterial


@dataclass(slots=True)
class ReactorConfig:
    HP: HeatpipeConfig
    FP: FuelPinConfig
    N: NeutronicsConfig

    def resolve_mesh(self):
        hp = deepcopy(self.HP)
        fp = deepcopy(self.FP)
        n = deepcopy(self.N)

        hp_resolved = hp.resolve_mesh()

        N_evap = hp_resolved.mesh.N_evap
        l_evap = hp_resolved.geometry.l_evap

        if fp.mesh.N_Z != N_evap:
            print(
                "\033[33mMesh warning:\033[0m",
                f"Fuel pin N_Z changed from {fp.mesh.N_Z} to {N_evap}"
            )

        if n.mesh.N_Z != N_evap:
            print(
                "\033[33mMesh warning:\033[0m",
                f"Neutronics N_Z changed from {n.mesh.N_Z} to {N_evap}"
            )

        if not np.isclose(n.mesh.l, l_evap):
            print(
                "\033[33mGeometry warning:\033[0m",
                f"Neutronics l changed from {n.mesh.l} to {l_evap}"
            )

        fp.mesh.N_Z = N_evap
        fp.geometry.l = l_evap

        n.mesh.N_Z = N_evap
        n.mesh.l = l_evap

        fp_resolved = fp.resolve_mesh()
        n.mesh.N_R = fp.mesh.N_R
        n_resolved = NeutronicsConfigResolved(
            mesh=n.mesh,
            energy=n.energy,
        )

        return ReactorConfigResolved(
            HP=hp_resolved,
            FP=fp_resolved,
            N=n_resolved,
        )

    def resolve_geometry(self):
        hp = deepcopy(self.HP)
        fp = deepcopy(self.FP)
        n = deepcopy(self.N)

        hp_resolved = hp.resolve_geometry()

        N_evap = hp_resolved.mesh.N_evap
        l_evap = hp_resolved.geometry.l_evap

        if not np.isclose(fp.geometry.l, l_evap):
            print(
                "\033[33mGeometry warning:\033[0m",
                f"Fuel pin l changed from {fp.geometry.l} to {l_evap}"
            )

        if not np.isclose(n.mesh.l, l_evap):
            print(
                "\033[33mGeometry warning:\033[0m",
                f"Neutronics l changed from {n.mesh.l} to {l_evap}"
            )

        if fp.mesh.N_Z != N_evap:
            print(
                "\033[33mMesh warning:\033[0m",
                f"Fuel pin N_Z changed from {fp.mesh.N_Z} to {N_evap}"
            )

        if n.mesh.N_Z != N_evap:
            print(
                "\033[33mMesh warning:\033[0m",
                f"Neutronics N_Z changed from {n.mesh.N_Z} to {N_evap}"
            )

        fp.geometry.l = l_evap
        fp.mesh.N_Z = N_evap

        n.mesh.l = l_evap
        n.mesh.N_Z = N_evap

        fp_resolved = fp.resolve_geometry()
        n.mesh.N_R = fp.mesh.N_R
        n_resolved = NeutronicsConfigResolved(
            mesh=n.mesh,
            energy=n.energy,
        )

        return ReactorConfigResolved(
            HP=hp_resolved,
            FP=fp_resolved,
            N=n_resolved,
        )


@dataclass(slots=True)
class ReactorConfigResolved:
    HP: HeatpipeConfigResolved
    FP: FuelPinConfigResolved
    N: NeutronicsConfigResolved


ConfigLike = Union[HeatpipeConfig, FuelPinConfig, NeutronicsConfig]


def make_mesh_sequence(cfg: ConfigLike, n: int):
    if n <= 0:
        raise ValueError("n must be positive.")

    if isinstance(cfg, HeatpipeConfig):
        return _make_heatpipe_sequence(cfg, n)

    if isinstance(cfg, FuelPinConfig):
        return _make_fuelpin_sequence(cfg, n)

    if isinstance(cfg, NeutronicsConfig):
        return _make_neutronics_sequence(cfg, n)

    raise TypeError(
        "cfg must be a HeatpipeConfig, FuelPinConfig, or NeutronicsConfig."
    )


def _scaled_counts_2d(N_R: int, N_Z: int, n: int) -> list[tuple[int, int]]:
    """
    Build n meshes from coarse to fine, where the total number of cells is
    approximately equidistant, using the input mesh as the finest one.
    """
    if n == 1:
        return [(N_R, N_Z)]

    scales = np.sqrt(np.linspace(1 / n, 1.0, n))
    counts: list[tuple[int, int]] = []

    for s in scales:
        Nr = max(1, int(round(N_R * s)))
        Nz = max(1, int(round(N_Z * s)))
        pair = (Nr, Nz)

        if not counts or pair != counts[-1]:
            counts.append(pair)

    counts[-1] = (N_R, N_Z)
    return counts


def _make_heatpipe_sequence(cfg: HeatpipeConfig, n: int) -> list[HeatpipeConfigResolved]:
    meshes = _scaled_counts_2d(cfg.mesh.N_R, cfg.mesh.N_Z, n)
    out: list[HeatpipeConfigResolved] = []

    for N_R, N_Z in meshes:
        cfg_i = deepcopy(cfg)
        cfg_i.mesh.N_R = N_R
        cfg_i.mesh.N_Z = N_Z
        out.append(cfg_i.resolve_mesh())

    return out


def _make_fuelpin_sequence(cfg: FuelPinConfig, n: int) -> list[FuelPinConfigResolved]:
    meshes = _scaled_counts_2d(cfg.mesh.N_R, cfg.mesh.N_Z, n)
    out: list[FuelPinConfigResolved] = []

    for N_R, N_Z in meshes:
        cfg_i = deepcopy(cfg)
        cfg_i.mesh.N_R = N_R
        cfg_i.mesh.N_Z = N_Z
        out.append(cfg_i.resolve_mesh())

    return out


def _make_neutronics_sequence(cfg: NeutronicsConfig, n: int) -> list[NeutronicsConfig]:
    meshes = _scaled_counts_2d(cfg.mesh.N_R, cfg.mesh.N_Z, n)
    out: list[NeutronicsConfig] = []

    for N_R, N_Z in meshes:
        cfg_i = deepcopy(cfg)
        cfg_i.mesh.N_R = N_R
        cfg_i.mesh.N_Z = N_Z
        out.append(cfg_i)

    return out
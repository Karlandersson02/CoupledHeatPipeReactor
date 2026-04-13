from matplotlib.pyplot import scatter
import numpy as np
import openmc
import openmc.mgxs
import os

from utils.sodium_properties import calculate_Na_rho_l
from coupled_systems.heatpipe import Heatpipe


# =============================================================================
# User parameters
# =============================================================================

OUTPUT_DIR = "outputs/homogenised_cell_model"

# Change this to whatever number of groups you want
NUM_ENERGY_GROUPS = 8

# Upper/lower bounds for the group structure [eV]
E_MIN_eV = 1.0e-5
E_MAX_eV = 20.0e6

# If True, use logarithmically spaced groups from E_MIN_eV to E_MAX_eV.
# Replace make_energy_group_edges() if you want a custom structure.
USE_LOG_GROUPS = True

T_HEAT_PIPE = 900
T_MODERATOR = 900
T_FUEL_PIN  = 900


def make_energy_group_edges(num_groups, e_min=E_MIN_eV, e_max=E_MAX_eV):
    """Return monotonically increasing energy bin edges in eV."""
    if USE_LOG_GROUPS:
        return np.logspace(np.log10(e_min), np.log10(e_max), num_groups + 1)
    else:
        raise NotImplementedError("Provide your own energy-group edges here.")


# =============================================================================
# Geometry / materials
# =============================================================================

def create_heat_pipe_universe(HP, temperature):
    density_Na = 1e-3 * calculate_Na_rho_l(temperature)
    density_fecral = 7.15
    density_graphite = 2.0

    porosity   = HP.cfg.wick.porosity
    r_outer    = 1e2 * HP.cfg.geometry.r_outer
    delta_wall = 1e2 * HP.cfg.geometry.delta_wall
    delta_gap  = 1e2 * HP.cfg.geometry.delta_gap
    delta_wick = 1e2 * HP.cfg.geometry.delta_wick

    r_gap    = r_outer - delta_wall
    r_wick   = r_gap   - delta_gap
    r_vapour = r_wick  - delta_wick

    fecral_alloy = openmc.Material(1, name='fecral')
    fecral_alloy.add_element('Fe', 0.73)
    fecral_alloy.add_element('Cr', 0.22)
    fecral_alloy.add_element('Al', 0.05)
    fecral_alloy.set_density('g/cm3', density_fecral)
    fecral_alloy.temperature = temperature

    sodium = openmc.Material(2, name='sodium')
    sodium.add_element('Na', 1.0)
    sodium.set_density('g/cm3', density_Na)
    sodium.temperature = temperature

    wick_material = openmc.Material.mix_materials(
        [sodium, fecral_alloy], [porosity, 1 - porosity], 'vo'
    )
    wick_material.temperature = temperature

    graphite = openmc.Material(3, name='graphite')
    graphite.add_nuclide('C0', 2.0)
    graphite.set_density('g/cm3', density_graphite)
    graphite.add_s_alpha_beta('c_Graphite')
    graphite.temperature = temperature

    s_vapour = openmc.ZCylinder(r=r_vapour)
    s_wick   = openmc.ZCylinder(r=r_wick)
    s_gap    = openmc.ZCylinder(r=r_gap)
    s_wall   = openmc.ZCylinder(r=r_outer)

    vapour_cell = openmc.Cell(name='vapour',   fill=None,          region=-s_vapour)
    wick_cell   = openmc.Cell(name='wick',     fill=wick_material, region=+s_vapour & -s_wick)
    gap_cell    = openmc.Cell(name='gap',      fill=sodium,        region=+s_wick   & -s_gap)
    wall_cell   = openmc.Cell(name='wall',     fill=fecral_alloy,  region=+s_gap    & -s_wall)
    mod_cell    = openmc.Cell(name='graphite', fill=graphite,      region=+s_wall)

    return openmc.Universe(cells=(vapour_cell, wick_cell, gap_cell, wall_cell, mod_cell))

def create_fuel_pin_universe(temperature):
    density_uo2       = 10.0
    density_zirconium = 6.6
    density_graphite  = 2.0
    r_fuel       = 1.2
    r_clad_inner = 1.3
    r_clad_outer = 1.4

    uo2 = openmc.Material(11, 'uo2')
    uo2.add_nuclide('U235', 0.06)
    uo2.add_nuclide('U238', 0.94)
    uo2.add_nuclide('O16',  2.0)
    uo2.set_density('g/cm3', density_uo2)
    uo2.temperature = temperature

    zirconium = openmc.Material(12, name='zirconium')
    zirconium.add_element('Zr', 1.0)
    zirconium.set_density('g/cm3', density_zirconium)
    zirconium.temperature = temperature

    graphite = openmc.Material(13, name='graphite')
    graphite.add_nuclide('C0', 1.0)
    graphite.set_density('g/cm3', density_graphite)
    graphite.add_s_alpha_beta('c_Graphite')
    graphite.temperature = temperature

    s_fuel  = openmc.ZCylinder(r=r_fuel)
    s_inner = openmc.ZCylinder(r=r_clad_inner)
    s_outer = openmc.ZCylinder(r=r_clad_outer)

    fuel_cell = openmc.Cell(name='fuel',      fill=uo2,       region=-s_fuel)
    gap_cell  = openmc.Cell(name='air gap',   fill=None,      region=+s_fuel  & -s_inner)
    clad_cell = openmc.Cell(name='clad',      fill=zirconium, region=+s_inner & -s_outer)
    mod_cell  = openmc.Cell(name='graphite',  fill=graphite,  region=+s_outer)

    return openmc.Universe(cells=(fuel_cell, gap_cell, clad_cell, mod_cell))


def create_moderator_universe(temperature):
    density_graphite = 2.0

    graphite = openmc.Material(21, name='graphite')
    graphite.add_nuclide('C0', 1.0)
    graphite.set_density('g/cm3', density_graphite)
    graphite.add_s_alpha_beta('c_Graphite')
    graphite.temperature = temperature

    mod_cell = openmc.Cell(name='moderator', fill=graphite)
    return openmc.Universe(cells=[mod_cell])


# =============================================================================
# MGXS setup
# =============================================================================

def build_mgxs_objects(domain, energy_group_edges):
    energy_groups = openmc.mgxs.EnergyGroups(group_edges=energy_group_edges)

    mgxs_objects = {
        "total": openmc.mgxs.TotalXS(
            domain=domain, domain_type='cell',
            energy_groups=energy_groups, by_nuclide=False
        ),
        "absorption": openmc.mgxs.AbsorptionXS(
            domain=domain, domain_type='cell',
            energy_groups=energy_groups, by_nuclide=False
        ),
        "scattering": openmc.mgxs.ScatterXS(
            domain=domain, domain_type='cell',
            energy_groups=energy_groups, by_nuclide=False
        ),
        "scatter_matrix": openmc.mgxs.ScatterMatrixXS(
            domain=domain, domain_type='cell',
            energy_groups=energy_groups, by_nuclide=False
        ),
        "fission": openmc.mgxs.FissionXS(
            domain=domain, domain_type='cell',
            energy_groups=energy_groups, by_nuclide=False
        ),
        "nu_fission": openmc.mgxs.FissionXS(
            domain=domain, domain_type='cell',
            energy_groups=energy_groups, nu=True, by_nuclide=False
        ),
        "chi": openmc.mgxs.Chi(
            domain=domain, domain_type='cell',
            energy_groups=energy_groups, by_nuclide=False
        ),
        "kappa_fission": openmc.mgxs.KappaFissionXS(
            domain=domain, domain_type='cell',
            energy_groups=energy_groups, by_nuclide=False
        ),
        "diffusion_coefficient": openmc.mgxs.DiffusionCoefficient(
            domain=domain, domain_type="cell",
            energy_groups=energy_groups, by_nuclide=False
        )
    }

    mgxs_objects["scatter_matrix"].formulation = "consistent"
    mgxs_objects["scatter_matrix"].correction = None

    return mgxs_objects


def collect_tallies_from_mgxs(mgxs_objects):
    tallies = openmc.Tallies()
    for mgxs in mgxs_objects.values():
        tallies += mgxs.tallies.values()
    return tallies


# =============================================================================
# Model assembly
# =============================================================================

def create_openmc_model(
    HP,
    num_groups=NUM_ENERGY_GROUPS,
    T_heat_pipe=T_HEAT_PIPE,
    T_moderator=T_MODERATOR,
    T_fuel_pin=T_FUEL_PIN
):
    # openmc.Materials.cross_sections = '/home/felixpersson/MasterThesisProject/NuclearData/endfb71/endfb-vii.1-hdf5/cross_sections.xml'

    heat_pipe_universe = create_heat_pipe_universe(HP, T_heat_pipe)
    fuel_pin_universe  = create_fuel_pin_universe(T_fuel_pin)
    moderator_universe = create_moderator_universe(T_moderator)

    all_materials = {}
    for u in [heat_pipe_universe, fuel_pin_universe, moderator_universe]:
        for mat_id, mat in u.get_all_materials().items():
            if mat_id not in all_materials:
                all_materials[mat_id] = mat

    materials = openmc.Materials(list(all_materials.values()))
    materials.export_to_xml()

    lattice = openmc.HexLattice()
    lattice.center = (0., 0.)
    lattice.pitch  = (10.,)
    lattice.outer  = moderator_universe
    lattice.universes = [
        6 * [moderator_universe, fuel_pin_universe, fuel_pin_universe],
        6 * [fuel_pin_universe, heat_pipe_universe],
        6 * [fuel_pin_universe],
        [heat_pipe_universe],
    ]

    outer_surface = openmc.model.HexagonalPrism(
        edge_length=4.0 * lattice.pitch[0],
        boundary_type='reflective',
        orientation='x'
    )
    top    = openmc.ZPlane( 100., boundary_type='vacuum')
    bottom = openmc.ZPlane(-100., boundary_type='vacuum')

    # This is the spatial domain over which we homogenize
    main_cell = openmc.Cell(
        name='homogenized_assembly',
        fill=lattice,
        region=(-outer_surface & +bottom & -top)
    )
    root_universe = openmc.Universe(cells=[main_cell])

    geometry = openmc.Geometry(root_universe)
    geometry.export_to_xml()

    # -------------------------------------------------------------------------
    # Multigroup cross section setup
    # -------------------------------------------------------------------------
    energy_group_edges = make_energy_group_edges(num_groups)
    mgxs_objects = build_mgxs_objects(main_cell, energy_group_edges)
    tallies = collect_tallies_from_mgxs(mgxs_objects)
    tallies.export_to_xml()

    source = openmc.IndependentSource(
        space=openmc.stats.Box(
            lower_left=(-2 * lattice.pitch[0], -100., -2 * lattice.pitch[0]),
            upper_right=( 2 * lattice.pitch[0],  100.,  2 * lattice.pitch[0]),
            only_fissionable=False
        )
    )

    settings = openmc.Settings()
    settings.batches = 150
    settings.inactive = 75
    settings.particles = 2000
    settings.source = source
    settings.verbosity = 4

    settings.temperature = {
        # 'default': 850.0,              # fallback temperature [K]
        'method': 'interpolation',     # use interpolation between tabulated temps
        'range': (300.0, 1400.0),      # preload all XS temperatures in this range
        'tolerance': 100.0             # outside range of available data, snap to bound if close enough
    }

    settings.export_to_xml()

    model = openmc.model.Model(
        geometry=geometry,
        materials=materials,
        settings=settings,
        tallies=tallies
    )

    return model, mgxs_objects, energy_group_edges


# =============================================================================
# Post-processing
# =============================================================================

def _safe_divide(a, b):
    out = np.zeros_like(a, dtype=float)
    mask = np.abs(b) > 0.0
    out[mask] = a[mask] / b[mask]
    return out


def load_homogenized_xs_from_statepoint(sp_filename, mgxs_objects):
    with openmc.StatePoint(sp_filename) as sp:
        for mgxs in mgxs_objects.values():
            mgxs.load_from_statepoint(sp)

        # OpenMC typically returns groups in its MGXS ordering.
        # We flatten to simple numpy arrays here.
        total_xs = np.squeeze(mgxs_objects["total"].get_xs())
        absorption_xs = np.squeeze(mgxs_objects["absorption"].get_xs())
        scattering_xs = np.squeeze(mgxs_objects["scattering"].get_xs())
        scatter_matrix = np.squeeze(mgxs_objects["scatter_matrix"].get_xs(row_column="inout"))
        fission_xs = np.squeeze(mgxs_objects["fission"].get_xs())
        nu_fission_xs = np.squeeze(mgxs_objects["nu_fission"].get_xs())
        chi = np.squeeze(mgxs_objects["chi"].get_xs())
        kappa_fission_xs = np.squeeze(mgxs_objects["kappa_fission"].get_xs())
        diffusion_coefficient = np.squeeze(mgxs_objects["diffusion_coefficient"].get_xs())

        # Derived quantities
        nu = _safe_divide(nu_fission_xs, fission_xs)
        kappa = _safe_divide(kappa_fission_xs, fission_xs)

        return {
            "total_xs": total_xs,                      
            "absorption_xs": absorption_xs,                      
            "scattering_xs": scattering_xs,                      
            "scatter_matrix_xs": scatter_matrix,
            "fission_xs": fission_xs,                  
            "nu": nu,                                  
            "chi": chi,                                
            "kappa_fission_xs": kappa_fission_xs,      
            "kappa": kappa,                            
            "nu_fission_xs": nu_fission_xs,
            "diffusion_coefficient": diffusion_coefficient,
            "difference": total_xs - absorption_xs - scattering_xs,
        }


def _array_to_code(arr, name):
    return f"{name} = np.array({np.array2string(arr, separator=', ', precision=6, suppress_small=False)})"


def print_homogenized_xs(results, energy_group_edges):
    print("# ================= COPY-PASTE ARRAYS =================\n")

    print(_array_to_code(energy_group_edges, "energy_group_edges"))
    print()

    print(_array_to_code(results["total_xs"], "total_xs"))
    print()

    print(_array_to_code(results["scatter_matrix_xs"], "scatter_matrix_xs"))
    print()

    print(_array_to_code(results["fission_xs"], "fission_xs"))
    print()

    print(_array_to_code(results["nu_fission_xs"], "nu_fission_xs"))
    print()

    print(_array_to_code(results["nu"], "nu"))
    print()

    print(_array_to_code(results["chi"], "chi"))
    print()

    print(_array_to_code(results["kappa_fission_xs"], "kappa_fission_xs"))
    print()

    print(_array_to_code(results["kappa"], "kappa"))
    print()

    print(_array_to_code(results["diffusion_coefficient"], "diffusion_coefficient"))
    print()

    print("# =====================================================")


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    import json
    from project_data.heatpipe_dataclasses import *

    with open("./data/vapour_data.json", "r") as f:
        data_guoju = json.load(f)
        data = data_guoju["data_guoju_560"]

    geom = HeatpipeGeometry(**data["geometry"])
    mesh = HeatpipeMesh(**data["mesh"])
    mat  = HeatpipeMaterial(**data["material"])
    wick = HeatpipeWick(**data["wick"])
    bc   = HeatpipeBC(**data["bc"])
    cfg  = HeatpipeConfig(geom, mesh, mat, wick, bc)
    cfg  = cfg.resolve()

    heatpipe = Heatpipe(cfg)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.chdir(OUTPUT_DIR)
    os.system("rm -f summary.h5 statepoint.*.h5 tallies.xml geometry.xml materials.xml settings.xml")

    model, mgxs_objects, energy_group_edges = create_openmc_model(
        heatpipe,
        num_groups=NUM_ENERGY_GROUPS,
        T_heat_pipe=T_HEAT_PIPE,
        T_moderator=T_MODERATOR,
        T_fuel_pin=T_FUEL_PIN
    )

    statepoint_path = model.run()

    results = load_homogenized_xs_from_statepoint(statepoint_path, mgxs_objects)
    print_homogenized_xs(results, energy_group_edges)

    # statepoint = openmc.StatePoint("/home/karlandersson/MasterThesisProject/outputs/homogenised_cell_model/statepoint.150.h5")
    # print(statepoint.keff)
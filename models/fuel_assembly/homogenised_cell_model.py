from matplotlib.pyplot import scatter
import numpy as np
import openmc
import openmc.mgxs
import os
import glob

from utils.sodium_properties import calculate_Na_rho_l
from coupled_systems.heatpipe import Heatpipe

import matplotlib.pyplot as plt

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

def create_heat_pipe_universe(cfg_HP, temperature):
    density_Na = 1e-3 * calculate_Na_rho_l(temperature)
    density_fecral = 7.16                            
    density_graphite = 1.85   # https://www.osti.gov/servlets/purl/1330693, Average between the two samples at 600 deg C.

    porosity   = cfg_HP.wick.porosity
    r_outer    = 1e2 * cfg_HP.geometry.r_outer
    delta_wall = 1e2 * cfg_HP.geometry.delta_wall
    delta_gap  = 1e2 * cfg_HP.geometry.delta_gap
    delta_wick = 1e2 * cfg_HP.geometry.delta_wick

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

def create_fuel_pin_universe(cfg_FP, temperature):
    density_uo2       = 10.6       # https://www.researchgate.net/publication/341370951_PROCESSES_of_UO_2_fuel_cycle
    density_zirconium = 6.6
    density_graphite  = 1.85
    r_clad_outer = 1.0e2 * (cfg_FP.geometry.r)
    r_clad_inner = 1.0e2 * (cfg_FP.geometry.r - cfg_FP.geometry.delta_wall)
    r_fuel       = 1.0e2 * (cfg_FP.geometry.r - cfg_FP.geometry.delta_wall - cfg_FP.geometry.delta_gap)

    uo2 = openmc.Material(11, 'uo2')
    uo2.add_nuclide('U235', 0.10)
    uo2.add_nuclide('U238', 0.90)
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
    density_graphite = 1.85

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
    cfg_R,
    num_groups=NUM_ENERGY_GROUPS,
    T_heat_pipe=T_HEAT_PIPE,
    T_moderator=T_MODERATOR,
    T_fuel_pin=T_FUEL_PIN
):
    # openmc.Materials.cross_sections = '/home/felixpersson/MasterThesisProject/NuclearData/endfb71/endfb-vii.1-hdf5/cross_sections.xml'

    heat_pipe_universe = create_heat_pipe_universe(cfg_R.HP, T_heat_pipe)
    fuel_pin_universe  = create_fuel_pin_universe(cfg_R.FP, T_fuel_pin)
    moderator_universe = create_moderator_universe(T_moderator)

    all_materials = {}
    for u in [heat_pipe_universe, fuel_pin_universe, moderator_universe]:
        for mat_id, mat in u.get_all_materials().items():
            if mat_id not in all_materials:
                all_materials[mat_id] = mat

    materials = openmc.Materials(list(all_materials.values()))
    materials.export_to_xml()

    lattice_pitch = 2.86
    lattice = openmc.HexLattice()
    lattice.center = (0., 0.)
    lattice.pitch  = (lattice_pitch, )
    lattice.outer  = moderator_universe
    lattice.universes = [
        6 * [moderator_universe, fuel_pin_universe, fuel_pin_universe],
        6 * [fuel_pin_universe, heat_pipe_universe],
        6 * [fuel_pin_universe],
        [heat_pipe_universe],
    ]

    flake_diameter = 20
    outer_surface = openmc.model.HexagonalPrism(
        edge_length = flake_diameter / (2. * np.sin(np.pi/3)),
        boundary_type='reflective',
        orientation='x'
    )
    top    = openmc.ZPlane( 1e2*cfg_FP.geometry.l/2, boundary_type='vacuum')
    bottom = openmc.ZPlane(-1e2*cfg_FP.geometry.l/2, boundary_type='vacuum')

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

    z_half = 1e2 * cfg_R.FP.geometry.l / 2

    source = openmc.IndependentSource(
        space=openmc.stats.Box(
            lower_left=(-11.0, -11.0, -z_half),
            upper_right=(11.0, 11.0, z_half),
        ),
        constraints={'fissionable': True}
    )

    settings = openmc.Settings()
    settings.batches = 150
    settings.inactive = 20
    settings.particles = 2000
    settings.source = source
    settings.verbosity = 4

    settings.temperature = {
        # 'default': 850.0,              # fallback temperature [K]
        'method': 'interpolation',     # use interpolation between tabulated temps
        'range': (700.0, 1200.0),      # preload all XS temperatures in this range
        'tolerance': 100.0             # outside range of available data, snap to bound if close enough
    }

    entropy_mesh = openmc.RegularMesh()
    entropy_mesh.lower_left  = (-11, -11, -90)
    entropy_mesh.upper_right = ( 11,  11,  90)
    entropy_mesh.dimension   = ( 15,  15,  15)

    settings.entropy_mesh = entropy_mesh

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
# Geometry and Shannon entropy plotting
# =============================================================================

def plot_geometry_and_entropy(model, statepoint_path=None, z0=0.0):
    if statepoint_path is None:
        sps = sorted(glob.glob("statepoint.*.h5"))
        if not sps:
            raise FileNotFoundError("No statepoint file found in current directory.")
        statepoint_path = sps[-1]

    with openmc.StatePoint(statepoint_path) as sp:
        entropy = np.asarray(sp.entropy)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: geometry
    model.geometry.plot(
        basis='xy',
        origin=(0.0, 0.0, z0),
        width=(22.5, 22.5),
        pixels=(800, 800),
        color_by='material',
        axes=axes[0],          # <- important fix
        axis_units='cm'        # optional, but nice if your model is in cm
    )
    axes[0].set_title(f'Geometry in x-y plane at z={z0}')
    axes[0].set_xlabel('x [cm]')
    axes[0].set_ylabel('y [cm]')
    axes[0].set_aspect('equal')

    # Right: entropy history
    axes[1].plot(np.arange(1, len(entropy) + 1), entropy, marker='o', markersize=3)
    axes[1].set_title('Shannon entropy vs batch')
    axes[1].set_xlabel('Batch')
    axes[1].set_ylabel('Entropy')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    import json
    from data.dataclass import *

    with open("./data/reactor_data.json", "r") as f:
        data = json.load(f)
    
    # Mesh dimensions
    N_R_HP, N_R_FP, N_Z = 15, 15, 50

    # Heat pipe config
    geom   = HeatpipeGeometry(**data["HeatPipe"]["geometry"])
    mesh   = HeatpipeMesh(N_R=N_R_HP, N_Z=N_Z)
    mat    = HeatpipeMaterial(**data["HeatPipe"]["material"])
    wick   = HeatpipeWick(**data["HeatPipe"]["wick"])
    bc     = HeatpipeBC(**data["HeatPipe"]["bc"])
    cfg_HP = HeatpipeConfig(geom, mesh, mat, wick, bc)

    # Fuel pin config
    geom_FP   = FuelPinGeometry(**data["FuelPin"]["geometry"])
    mesh_FP   = FuelPinMesh(N_R=N_R_FP, N_Z=30)
    energy_FP = FuelPinEnergy(**data["FuelPin"]["energy"])
    mat_FP    = FuelPinMaterial(**data["FuelPin"]["material"])
    cfg_FP    = FuelPinConfig(geom_FP, mesh_FP, energy_FP, mat_FP)

    # Neutronics config
    mesh_N = NeutronicsMesh(
        N_R = N_R_FP,
        N_Z = 50,
        l   = data["FuelPin"]["geometry"]["l"]
    )
    energy = NeutronicsEnergy(
        N_G   = cfg_FP.energy.N_G,
        power = data["Reactor"]["power"]["thermal"] / data["Reactor"]["components"]["N_FP"]
    )
    cfg_N = NeutronicsConfig(mesh_N, energy)

    # Reactor config
    cfg_R = ReactorConfig(cfg_HP, cfg_FP, cfg_N)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.chdir(OUTPUT_DIR)
    os.system("rm -f summary.h5 statepoint.*.h5 tallies.xml geometry.xml materials.xml settings.xml")

    model, mgxs_objects, energy_group_edges = create_openmc_model(
        cfg_R,
        num_groups=NUM_ENERGY_GROUPS,
        T_heat_pipe=T_HEAT_PIPE,
        T_moderator=T_MODERATOR,
        T_fuel_pin=T_FUEL_PIN
    )

    statepoint_path = model.run()

    plot_geometry_and_entropy(model, statepoint_path)
    results = load_homogenized_xs_from_statepoint(statepoint_path, mgxs_objects)
    #print_homogenized_xs(results, energy_group_edges)

    # statepoint = openmc.StatePoint("/home/karlandersson/MasterThesisProject/outputs/homogenised_cell_model/statepoint.150.h5")
    # print(statepoint.keff)
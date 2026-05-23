from asyncio import threads

from matplotlib.pyplot import scatter
import numpy as np
import openmc
import openmc.mgxs
import os
import glob

from utils.sodium_properties import calculate_Na_rho_l
from coupled.heatpipe import Heatpipe

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator
from matplotlib.ticker import FuncFormatter

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

T_HEAT_PIPE = 900.0
T_MODERATOR = 900.0
T_FUEL_PIN  = 900.0


def make_energy_group_edges(num_groups, e_min=E_MIN_eV, e_max=E_MAX_eV):
    """Return monotonically increasing energy bin edges in eV."""

    if num_groups == 8:
        return np.asarray(openmc.mgxs.GROUP_STRUCTURES["CASMO-8"], dtype=float)

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

    fecral_alloy = openmc.Material(material_id=1, name='fecral')
    fecral_alloy.add_element('Fe', 0.73)
    fecral_alloy.add_element('Cr', 0.22)
    fecral_alloy.add_element('Al', 0.05)
    fecral_alloy.set_density('g/cm3', density_fecral)
    fecral_alloy.temperature = temperature

    sodium = openmc.Material(material_id=2, name='sodium')
    sodium.add_element('Na', 1.0)
    sodium.set_density('g/cm3', density_Na)
    sodium.temperature = temperature

    wick_material = openmc.Material.mix_materials(
        [sodium, fecral_alloy], [porosity, 1 - porosity], 'vo'
    )
    wick_material.temperature = temperature
    wick_material.name = 'wick_material'
    wick_material.id = 7

    graphite = openmc.Material(material_id=5, name='graphite_hp')
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

    uo2 = openmc.Material(material_id=11, name='uo2')
    uo2.add_nuclide('U235', 0.10)
    uo2.add_nuclide('U238', 0.90)
    uo2.add_nuclide('O16',  2.0)
    uo2.set_density('g/cm3', density_uo2)
    uo2.temperature = temperature

    zirconium = openmc.Material(material_id=12, name='zirconium')
    zirconium.add_element('Zr', 1.0)
    zirconium.set_density('g/cm3', density_zirconium)
    zirconium.temperature = temperature

    graphite = openmc.Material(material_id=13, name='graphite_fp')
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

    graphite = openmc.Material(material_id=21, name='graphite_mod')
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
    openmc.Materials.cross_sections = '/home/felixpersson/MasterThesisProject/NuclearData/endfb71/endfb-vii.1-hdf5/cross_sections.xml'

    openmc.reset_auto_ids()

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

    lattice_pitch = 3.2
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
    top    = openmc.ZPlane( 1e2*cfg_R.FP.geometry.l/2, boundary_type='vacuum')
    bottom = openmc.ZPlane(-1e2*cfg_R.FP.geometry.l/2, boundary_type='vacuum')

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
    settings.batches = 250
    settings.inactive = 20
    settings.particles = 100
    settings.source = source
    settings.verbosity = 7

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

        # Mean values
        total_xs = np.squeeze(mgxs_objects["total"].get_xs())
        absorption_xs = np.squeeze(mgxs_objects["absorption"].get_xs())
        scattering_xs = np.squeeze(mgxs_objects["scattering"].get_xs())
        scatter_matrix = np.squeeze(
            mgxs_objects["scatter_matrix"].get_xs(row_column="inout")
        )
        fission_xs = np.squeeze(mgxs_objects["fission"].get_xs())
        nu_fission_xs = np.squeeze(mgxs_objects["nu_fission"].get_xs())
        chi = np.squeeze(mgxs_objects["chi"].get_xs())
        kappa_fission_xs = np.squeeze(mgxs_objects["kappa_fission"].get_xs())
        diffusion_coefficient = np.squeeze(
            mgxs_objects["diffusion_coefficient"].get_xs()
        )

        # Standard deviations
        total_xs_std = np.squeeze(mgxs_objects["total"].get_xs(value='std_dev'))
        absorption_xs_std = np.squeeze(mgxs_objects["absorption"].get_xs(value='std_dev'))
        scattering_xs_std = np.squeeze(mgxs_objects["scattering"].get_xs(value='std_dev'))
        scatter_matrix_std = np.squeeze(
            mgxs_objects["scatter_matrix"].get_xs(value='std_dev', row_column="inout")
        )
        fission_xs_std = np.squeeze(mgxs_objects["fission"].get_xs(value='std_dev'))
        nu_fission_xs_std = np.squeeze(mgxs_objects["nu_fission"].get_xs(value='std_dev'))
        chi_std = np.squeeze(mgxs_objects["chi"].get_xs(value='std_dev'))
        kappa_fission_xs_std = np.squeeze(
            mgxs_objects["kappa_fission"].get_xs(value='std_dev')
        )
        diffusion_coefficient_std = np.squeeze(
            mgxs_objects["diffusion_coefficient"].get_xs(value='std_dev')
        )

        # Derived quantities
        nu = _safe_divide(nu_fission_xs, fission_xs)
        kappa = _safe_divide(kappa_fission_xs, fission_xs)

        return {
            # Means
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

            # Standard deviations
            "total_xs_std": total_xs_std,
            "absorption_xs_std": absorption_xs_std,
            "scattering_xs_std": scattering_xs_std,
            "scatter_matrix_xs_std": scatter_matrix_std,
            "fission_xs_std": fission_xs_std,
            "nu_fission_xs_std": nu_fission_xs_std,
            "chi_std": chi_std,
            "kappa_fission_xs_std": kappa_fission_xs_std,
            "diffusion_coefficient_std": diffusion_coefficient_std,
        }


def _array_to_code(arr, name):
    return f"{name} = np.array({np.array2string(arr, separator=', ', precision=6, suppress_small=False)})"


def print_homogenized_xs(results, energy_group_edges):
    print("# ================= COPY-PASTE ARRAYS =================\n")

    print(_array_to_code(energy_group_edges, "energy_group_edges"))
    print()

    print(_array_to_code(results["total_xs"], "total_xs"))
    print(_array_to_code(results["total_xs_std"], "total_xs_std"))
    print()

    print(_array_to_code(results["absorption_xs"], "absorption_xs"))
    print(_array_to_code(results["absorption_xs_std"], "absorption_xs_std"))
    print()

    print(_array_to_code(results["scattering_xs"], "scattering_xs"))
    print(_array_to_code(results["scattering_xs_std"], "scattering_xs_std"))
    print()

    print(_array_to_code(results["scatter_matrix_xs"], "scatter_matrix_xs"))
    print(_array_to_code(results["scatter_matrix_xs_std"], "scatter_matrix_xs_std"))
    print()

    print(_array_to_code(results["fission_xs"], "fission_xs"))
    print(_array_to_code(results["fission_xs_std"], "fission_xs_std"))
    print()

    print(_array_to_code(results["nu_fission_xs"], "nu_fission_xs"))
    print(_array_to_code(results["nu_fission_xs_std"], "nu_fission_xs_std"))
    print()

    print(_array_to_code(results["nu"], "nu"))
    print()

    print(_array_to_code(results["chi"], "chi"))
    print(_array_to_code(results["chi_std"], "chi_std"))
    print()

    print(_array_to_code(results["kappa_fission_xs"], "kappa_fission_xs"))
    print(_array_to_code(results["kappa_fission_xs_std"], "kappa_fission_xs_std"))
    print()

    print(_array_to_code(results["kappa"], "kappa"))
    print()

    print(_array_to_code(results["diffusion_coefficient"], "diffusion_coefficient"))
    print(_array_to_code(results["diffusion_coefficient_std"], "diffusion_coefficient_std"))
    print()

    print("# =====================================================")

# =============================================================================
# Geometry and Shannon entropy plotting
# =============================================================================

from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FixedFormatter

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "mathtext.fontset": "cm",
    "font.size": 30,
    "text.usetex": True,
})


def _canonical_material_name(material):
    """
    Map implementation-specific OpenMC material names to physical
    material names used for plotting.

    This makes graphite_hp, graphite_fp, and graphite_mod appear
    with the same colour, even though they are different OpenMC
    Material objects.
    """
    name = (material.name or "").lower()

    if name.startswith("graphite"):
        return "graphite"
    if name.startswith("fecral"):
        return "fecral"
    if name.startswith("sodium"):
        return "sodium"
    if name.startswith("wick"):
        return "wick_material"
    if name.startswith("uo2"):
        return "uo2"
    if name.startswith("zirconium"):
        return "zirconium"

    return name


def _hex_to_rgb(hex_color):
    """
    Convert '#RRGGBB' to an RGB tuple accepted by OpenMC.

    Example
    -------
    '#BBBBBB' -> (187, 187, 187)
    """
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


def _as_xy_pair(value):
    """
    Accept either a scalar width or an (x_width, y_width) pair.
    """
    if np.isscalar(value):
        return float(value), float(value)

    if len(value) != 2:
        raise ValueError("Expected either a scalar or a length-2 tuple/list.")

    return float(value[0]), float(value[1])


def _resolve_pixels(width, pixels):
    """
    Resolve pixels for OpenMC plotting.

    If pixels is a scalar, it is interpreted as the number of pixels along
    the longest side. The shorter side is scaled to preserve pixel density.

    Example
    -------
    width=(24, 21), pixels=4000 -> pixels=(4000, 3500)
    """
    wx, wy = _as_xy_pair(width)

    if np.isscalar(pixels):
        p_long = int(pixels)
        w_long = max(wx, wy)

        px = max(1, int(round(p_long * wx / w_long)))
        py = max(1, int(round(p_long * wy / w_long)))

        return px, py

    if len(pixels) != 2:
        raise ValueError("Expected pixels to be either a scalar or a length-2 tuple/list.")

    return int(pixels[0]), int(pixels[1])


def _figsize_from_width(width, long_side=8.0):
    """
    Choose a Matplotlib figure size with the same aspect ratio as the
    OpenMC plot width.
    """
    wx, wy = _as_xy_pair(width)
    w_long = max(wx, wy)

    return (
        long_side * wx / w_long,
        long_side * wy / w_long,
    )


def _nice_local_ticks(width, step=0.5):
    """
    Create symmetric local ticks around zero for a given plot width.
    """
    wx, wy = _as_xy_pair(width)

    half_x = 0.5 * wx
    half_y = 0.5 * wy

    x_max = step * np.floor(half_x / step)
    y_max = step * np.floor(half_y / step)

    x_ticks = np.arange(-x_max, x_max + 0.5 * step, step)
    y_ticks = np.arange(-y_max, y_max + 0.5 * step, step)

    return x_ticks, y_ticks


def _format_tick_labels(ticks):
    """
    Format tick labels without unnecessary trailing zeros.
    """
    labels = []

    for value in ticks:
        if abs(value) < 1.0e-12:
            value = 0.0

        labels.append(f"{value:.2f}".rstrip("0").rstrip("."))

    return labels


def _set_local_detail_ticks(ax, origin, width, step=0.5):
    """
    Show ticks relative to the local centre of a zoomed component.

    The OpenMC plot itself still uses global coordinates internally,
    but the displayed tick labels are shifted so that the chosen
    component centre appears as 0.
    """
    x0, y0 = origin

    local_x_ticks, local_y_ticks = _nice_local_ticks(width, step=step)

    global_x_ticks = x0 + local_x_ticks
    global_y_ticks = y0 + local_y_ticks

    ax.xaxis.set_major_locator(FixedLocator(global_x_ticks))
    ax.yaxis.set_major_locator(FixedLocator(global_y_ticks))

    ax.xaxis.set_major_formatter(FixedFormatter(_format_tick_labels(local_x_ticks)))
    ax.yaxis.set_major_formatter(FixedFormatter(_format_tick_labels(local_y_ticks)))


def _get_material_color_scheme():
    """
    Define the material colour scheme once.

    Returns
    -------
    material_colors_hex : dict
        Hex colours for Matplotlib legends.

    material_colors_rgb : dict
        RGB integer tuples for OpenMC geometry.plot().
    """
    colors = {
        "blue": "#0077BB",
        "cyan": "#33BBEE",
        "teal": "#009988",
        "orange": "#EE7733",
        "red": "#CC3311",
        "magenta": "#EE3377",
        "grey": "#BBBBBB",
        "shamrock": "#4DA167",
        "teagreen": "#CBEFB6",
        "vanillaCustard": "#D7D9B1",
        "clay": "#EDB88B",
        "desert": "#E1BD9E",
    }

    # Extra greys
    soft_grey = "#D6D6D6"
    grey = "#BBBBBB"
    dark_grey = "#7A7A7A"
    darker_grey = "#212121"

    material_colors_hex = {
        # Moderator / background
        "graphite": colors["grey"],

        # Fuel-pin materials
        "uo2": colors["red"],
        "zirconium": darker_grey,

        # Heat-pipe materials
        "sodium": colors["cyan"],
        "wick_material": colors["cyan"],
        "fecral": colors["blue"],
    }

    material_colors_rgb = {
        material_type: _hex_to_rgb(hex_color)
        for material_type, hex_color in material_colors_hex.items()
    }

    return material_colors_hex, material_colors_rgb


def _make_material_plot_colors(model):
    """
    Create an OpenMC-compatible material colour dictionary.

    The keys passed to OpenMC must be actual Material objects,
    but the colour choice is based on the physical material type.
    """
    _, material_colors_rgb = _get_material_color_scheme()

    material_colors = {}

    for material in model.geometry.get_all_materials().values():
        material_type = _canonical_material_name(material)

        if material_type in material_colors_rgb:
            material_colors[material] = material_colors_rgb[material_type]

    return material_colors


def _make_material_legend_handles():
    """
    Create legend handles matching the actual material colour scheme.
    """
    material_colors_hex, _ = _get_material_color_scheme()

    labels = {
        "graphite": "Graphite moderator",
        "zirconium": "Zirconium clad",
        "uo2": r"UO$_2$ fuel",
        "fecral": "FeCrAl wall",
        "wick_material": "Sodium / wick material",
    }

    order = [
        "graphite",
        "zirconium",
        "uo2",
        "fecral",
        "wick_material",
    ]

    handles = [
        Patch(
            facecolor=material_colors_hex[name],
            edgecolor="black",
            linewidth=0.5,
            label=labels[name],
        )
        for name in order
    ]

    return handles


def _plot_geometry_panel(
    model,
    ax,
    origin,
    width,
    pixels,
    material_colors,
    title=None,
    show_labels=True,
    local_ticks=False,
    tick_step=0.5,
    z0=0.0,
):
    """
    Helper for plotting one OpenMC geometry panel.

    width can be either:
      - scalar, e.g. 22.5
      - tuple, e.g. (24.0, 21.0)

    If local_ticks=True, tick labels are shown relative to the chosen
    origin, but the axis labels remain x [cm] and y [cm].
    """
    width_xy = _as_xy_pair(width)
    pixels_xy = _resolve_pixels(width_xy, pixels)

    model.geometry.plot(
        basis="xy",
        origin=(origin[0], origin[1], z0),
        width=width_xy,
        pixels=pixels_xy,
        color_by="material",
        colors=material_colors,
        axes=ax,
        axis_units="cm",
    )

    if title is not None:
        ax.set_title(title, pad=6)

    ax.set_aspect("equal")

    if show_labels:
        # Do not set per-axis labels here. They are too large for the
        # three-panel figure when using font.size = 30.
        if local_ticks:
            _set_local_detail_ticks(
                ax=ax,
                origin=origin,
                width=width_xy,
                step=tick_step,
            )
    else:
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks([])
        ax.set_yticks([])


def save_geometry_material_subfigures(
    model,
    z0=0.0,
    pdf_path="geometry_material_subfigures.pdf",
    png_path="geometry_material_subfigures.png",
    assembly_width=(24.0, 21.0),
    detail_width=4.2,
    assembly_pixels=4000,
    detail_pixels=2200,
    dpi=600,
    fuel_pin_origin=(0.0, 3.2),
    heat_pipe_origin=(0.0, 0.0),
):
    """
    Save a geometry-only figure with:
      1. full assembly,
      2. zoomed fuel pin,
      3. zoomed heat pipe.

    The representative fuel pin and heat pipe are selected by their
    x-y origins. With the current lattice setup, the central heat pipe
    is at approximately (0, 0), and a nearby fuel pin is approximately
    at (0, 3.2).
    """
    pdf_path = os.path.abspath(pdf_path)
    png_path = os.path.abspath(png_path)

    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
    os.makedirs(os.path.dirname(png_path), exist_ok=True)

    material_colors = _make_material_plot_colors(model)

    assembly_wx, assembly_wy = _as_xy_pair(assembly_width)
    assembly_panel_ratio = assembly_wx / assembly_wy

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(19, 8.5),
        gridspec_kw={
            "wspace": 0.20,
            "width_ratios": [assembly_panel_ratio, 1.0, 1.0],
        },
        facecolor="white",
    )

    ax_assembly, ax_fuel, ax_hp = axes

    _plot_geometry_panel(
        model=model,
        ax=ax_assembly,
        origin=(0.0, 0.0),
        width=assembly_width,
        pixels=assembly_pixels,
        material_colors=material_colors,
        title="Assembly",
        show_labels=True,
        local_ticks=False,
        z0=z0,
    )

    _plot_geometry_panel(
        model=model,
        ax=ax_fuel,
        origin=fuel_pin_origin,
        width=detail_width * 0.6,
        pixels=detail_pixels,
        material_colors=material_colors,
        title="Fuel pin",
        show_labels=True,
        local_ticks=True,
        tick_step=0.5,
        z0=z0,
    )

    _plot_geometry_panel(
        model=model,
        ax=ax_hp,
        origin=heat_pipe_origin,
        width=detail_width * 0.95,
        pixels=detail_pixels,
        material_colors=material_colors,
        title="Heat pipe",
        show_labels=True,
        local_ticks=True,
        tick_step=0.5,
        z0=z0,
    )

    material_handles = _make_material_legend_handles()

    fig.legend(
        handles=material_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=3,
        frameon=False,
        fontsize=24,
        handlelength=1.6,
        columnspacing=1.6,
        handletextpad=0.6,
    )

    # Shared axis labels, instead of one x/y label per subplot.
    fig.supxlabel(r"$x$ [cm]", fontsize=30, y=0.045)
    fig.supylabel(r"$y$ [cm]", fontsize=30, x=0.015)

    fig.subplots_adjust(
        top=0.76,
        bottom=0.16,
        left=0.07,
        right=0.98,
        wspace=0.20,
    )

    fig.savefig(pdf_path, format="pdf", dpi=dpi, bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=dpi, bbox_inches="tight")

    plt.close(fig)

    print(f"Saved geometry subfigure PDF to: {pdf_path}")
    print(f"Saved geometry subfigure PNG to: {png_path}")


def save_geometry_only(
    model,
    z0=0.0,
    pdf_path="geometry_plot.pdf",
    png_path="geometry_plot.png",
    width=(24.0, 21.0),
    pixels=4000,
    dpi=600,
):
    """
    Save only the full assembly geometry as both PDF and PNG.
    """
    pdf_path = os.path.abspath(pdf_path)
    png_path = os.path.abspath(png_path)

    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
    os.makedirs(os.path.dirname(png_path), exist_ok=True)

    material_colors = _make_material_plot_colors(model)

    fig, ax = plt.subplots(
        figsize=_figsize_from_width(width, long_side=8.0),
        facecolor="white",
    )

    _plot_geometry_panel(
        model=model,
        ax=ax,
        origin=(0.0, 0.0),
        width=width,
        pixels=pixels,
        material_colors=material_colors,
        title=None,
        show_labels=True,
        local_ticks=False,
        z0=z0,
    )

    fig.tight_layout()

    fig.savefig(pdf_path, format="pdf", dpi=dpi, bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=dpi, bbox_inches="tight")

    plt.close(fig)

    print(f"Saved geometry-only PDF to: {pdf_path}")
    print(f"Saved geometry-only PNG to: {png_path}")


def plot_geometry_and_entropy(
    model,
    statepoint_path=None,
    z0=0.0,
    save_geometry=True,
    save_geometry_subfigures=True,
    geometry_pdf_path="geometry_plot.pdf",
    geometry_png_path="geometry_plot.png",
    subfigures_pdf_path="geometry_material_subfigures.pdf",
    subfigures_png_path="geometry_material_subfigures.png",
    assembly_width=(24.0, 21.0),
    geometry_pixels=4000,
    display_pixels=1600,
    detail_pixels=2200,
    dpi=600,
    fuel_pin_origin=(0.0, 3.2),
    heat_pipe_origin=(0.0, 0.0),
):
    """
    Plot geometry and Shannon entropy.

    Also optionally saves:
      - geometry-only PDF/PNG,
      - three-panel geometry material figure PDF/PNG.
    """
    if statepoint_path is None:
        sps = sorted(glob.glob("statepoint.*.h5"))
        if not sps:
            raise FileNotFoundError("No statepoint file found in current directory.")
        statepoint_path = sps[-1]

    with openmc.StatePoint(statepoint_path) as sp:
        entropy = np.asarray(sp.entropy)

    material_colors = _make_material_plot_colors(model)

    # -------------------------------------------------------------------------
    # Save geometry-only figure
    # -------------------------------------------------------------------------
    if save_geometry:
        save_geometry_only(
            model=model,
            z0=z0,
            pdf_path=geometry_pdf_path,
            png_path=geometry_png_path,
            width=assembly_width,
            pixels=geometry_pixels,
            dpi=dpi,
        )

    # -------------------------------------------------------------------------
    # Save assembly + fuel-pin + heat-pipe subfigure
    # -------------------------------------------------------------------------
    if save_geometry_subfigures:
        save_geometry_material_subfigures(
            model=model,
            z0=z0,
            pdf_path=subfigures_pdf_path,
            png_path=subfigures_png_path,
            assembly_width=assembly_width,
            detail_width=4.2,
            assembly_pixels=geometry_pixels,
            detail_pixels=detail_pixels,
            dpi=dpi,
            fuel_pin_origin=fuel_pin_origin,
            heat_pipe_origin=heat_pipe_origin,
        )

    # -------------------------------------------------------------------------
    # Combined geometry + entropy figure for display
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor="white")

    _plot_geometry_panel(
        model=model,
        ax=axes[0],
        origin=(0.0, 0.0),
        width=assembly_width,
        pixels=display_pixels,
        material_colors=material_colors,
        title=f"Geometry in x-y plane at z={z0}",
        show_labels=True,
        local_ticks=False,
        z0=z0,
    )

    axes[1].plot(
        np.arange(1, len(entropy) + 1),
        entropy,
        marker="o",
        markersize=3,
    )
    axes[1].set_title("Shannon entropy vs batch")
    axes[1].set_xlabel("Batch")
    axes[1].set_ylabel("Entropy")
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

    statepoint_path = model.run(threads=16)

    plot_geometry_and_entropy(model, statepoint_path)
    results = load_homogenized_xs_from_statepoint(statepoint_path, mgxs_objects)
    print_homogenized_xs(results, energy_group_edges)

    # statepoint = openmc.StatePoint("/home/karlandersson/MasterThesisProject/outputs/homogenised_cell_model/statepoint.150.h5")
    # print(statepoint.keff)
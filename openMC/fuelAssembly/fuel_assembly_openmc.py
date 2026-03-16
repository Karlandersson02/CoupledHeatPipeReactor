import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import openmc
import os

os.environ["OPENMC_CROSS_SECTIONS"] = (
    "/home/felixpersson/MasterThesisProject/NuclearData/endfb71/"
    "endfb-vii.1-hdf5/cross_sections.xml"
)

from models.sodium_properties import calculate_Na_rho_l
from CoupledSystems.Heatpipe import Heatpipe


# Thermal/fast boundary: 0.625 eV (standard CASMO convention).
# Knott & Wehlage, "Lattice Physics Computations", 2010.
E_THERMAL_CUTOFF_eV = 0.625

# Cartesian mesh parameters — centred on origin, spans the full core XY extent.
# One z-bin integrates over the full axial height (geometry is axially reflective).
MESH_N_XY = 200    # bins per side in x and y
MESH_N_Z  = 1      # single z-bin; z-integrated flux
MESH_HALF = 35.0   # half-width [cm]; covers 3.5 * pitch = 35 cm

# All XML files, statepoints, and plots are written here.
OUTPUT_DIR = "openMC/fuelAssembly"

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _get_xy_slice(flux_raw, mesh_dims):
    """
    Reshape flat tally output to a 2-D image array (ny, nx) for imshow.

    OpenMC RegularMesh flattens values in (x, y, z) order — x is the
    slowest-varying index.  Reshaping to (nx, ny, nz) then transposing
    the first two axes gives (ny, nx), which imshow renders with
    x → horizontal and y → vertical when origin='lower'.

    Parameters
    ----------
    flux_raw : np.ndarray, shape (nx*ny*nz, 1, 1)
    mesh_dims : tuple, (nx, ny, nz)

    Returns
    -------
    np.ndarray, shape (ny, nx)
    """
    nx, ny, nz = mesh_dims
    return flux_raw.reshape((nx, ny, nz))[:, :, 0].T


def plot_flux_xy(flux_raw, mesh_dims, title, filename):
    """
    Imshow of z-integrated flux on the xy plane.

    Parameters
    ----------
    flux_raw  : np.ndarray  Raw values from tally.get_values(scores=['flux']).
    mesh_dims : tuple       (nx, ny, nz) from mesh.dimension.
    title     : str         Plot title.
    filename  : str         Output PDF path.
    """
    sns.set_theme(style="white")

    img    = _get_xy_slice(flux_raw, mesh_dims)
    extent = [-MESH_HALF, MESH_HALF, -MESH_HALF, MESH_HALF]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(img, origin='lower', extent=extent,
                   cmap='inferno', interpolation='nearest')
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r'$\phi$ [a.u.]', fontsize=15)
    cbar.ax.tick_params(labelsize=12)
    ax.set_xlabel(r'$x$ [cm]', fontsize=17)
    ax.set_ylabel(r'$y$ [cm]', fontsize=17)
    ax.set_title(title, fontsize=14)
    ax.tick_params(labelsize=12)
    plt.tight_layout()
    # plt.savefig(filename)
    plt.show()
    print(f"Saved: {filename}")


def plot_total_flux_xy(sp, mesh_dims):
    """Imshow of total (energy-integrated) flux."""
    flux = sp.get_tally(name='flux_total').get_values(scores=['flux'])
    plot_flux_xy(flux, mesh_dims,
                 title=r'Total flux',
                 filename='total_flux_xy.pdf')


def plot_thermal_flux_xy(sp, mesh_dims):
    """Imshow of thermal flux (E < 0.625 eV)."""
    flux = sp.get_tally(name='flux_thermal').get_values(scores=['flux'])
    plot_flux_xy(flux, mesh_dims,
                 title=r'Thermal flux ($E < 0.625\,\mathrm{eV}$)',
                 filename='thermal_flux_xy.pdf')


def plot_fast_flux_xy(sp, mesh_dims):
    """Imshow of fast flux (E > 0.625 eV)."""
    flux = sp.get_tally(name='flux_fast').get_values(scores=['flux'])
    plot_flux_xy(flux, mesh_dims,
                 title=r'Fast flux ($E > 0.625\,\mathrm{eV}$)',
                 filename='fast_flux_xy.pdf')


# ---------------------------------------------------------------------------
# Geometry plot
# ---------------------------------------------------------------------------

def plot_geometry(lattice_pitch):
    plot_xy = openmc.Plot()
    plot_xy.basis    = 'xy'
    plot_xy.origin   = (0., 0., 0.)
    plot_xy.width    = (2 * 5 * lattice_pitch, 2 * 5 * lattice_pitch)
    plot_xy.pixels   = (1000, 1000)
    plot_xy.color_by = 'material'

    plot_xz = openmc.Plot()
    plot_xz.basis    = 'xz'
    plot_xz.origin   = (0., 0., 0.)
    plot_xz.width    = (2 * 5 * lattice_pitch, 200.)
    plot_xz.pixels   = (1000, 1000)
    plot_xz.color_by = 'material'

    plots = openmc.Plots([plot_xy, plot_xz])
    plots.export_to_xml()
    openmc.plot_geometry()


# ---------------------------------------------------------------------------
# Universe builders
# ---------------------------------------------------------------------------

def create_heat_pipe_universe(HP):
    density_Na       = 1e-3 * calculate_Na_rho_l(HP.data["T_op"])
    density_fecral    = 7.15
    density_graphite = 2.0

    porosity   = HP.data.get("porosity")
    r_outer    = 1e2 * HP.data.get("r_outer")
    delta_wall = 1e2 * HP.data.get("delta_wall")
    delta_gap  = 1e2 * HP.data.get("delta_gap")
    delta_wick = 1e2 * HP.data.get("delta_wick")
    r_gap    = r_outer - delta_wall
    r_wick   = r_gap   - delta_gap
    r_vapour = r_wick  - delta_wick

    print(f"Heat pipe outer radius: {r_outer:.4f} cm")

    fecral_alloy = openmc.Material(1, name='stainlessSteel')
    fecral_alloy.add_element('Fe', 0.73)
    fecral_alloy.add_element('Cr', 0.22)
    fecral_alloy.add_element('Al', 0.05)
    fecral_alloy.set_density('g/cm3', density_fecral)

    sodium = openmc.Material(2, name='Sodium')
    sodium.add_element('Na', 1.0)
    sodium.set_density('g/cm3', density_Na)

    wick_material = openmc.Material.mix_materials(
        [sodium, fecral_alloy], [porosity, 1 - porosity], 'vo')

    graphite = openmc.Material(3, name='moderator')
    graphite.add_nuclide('C0', 2.0)
    graphite.set_density('g/cm3', density_graphite)
    graphite.add_s_alpha_beta('c_Graphite')

    s_vapour = openmc.ZCylinder(r=r_vapour)
    s_wick   = openmc.ZCylinder(r=r_wick)
    s_gap    = openmc.ZCylinder(r=r_gap)
    s_wall   = openmc.ZCylinder(r=r_outer)

    vapour_cell = openmc.Cell(name='vapour',   fill=None,            region=-s_vapour)
    wick_cell   = openmc.Cell(name='wick',     fill=wick_material,   region=+s_vapour & -s_wick)
    gap_cell    = openmc.Cell(name='gap',      fill=sodium,          region=+s_wick   & -s_gap)
    wall_cell   = openmc.Cell(name='wall',     fill=fecral_alloy,    region=+s_gap    & -s_wall)
    mod_cell    = openmc.Cell(name='graphite', fill=graphite,        region=+s_wall)

    return openmc.Universe(cells=(vapour_cell, wick_cell, gap_cell, wall_cell, mod_cell))


def create_fuel_pin_universe():
    density_uo2       = 10.0
    density_zirconium =  6.6
    density_graphite  =  2.0
    r_fuel       = 1.2
    r_clad_inner = 1.3
    r_clad_outer = 1.4

    uo2 = openmc.Material(11, 'uo2')
    uo2.add_nuclide('U235', 0.06)
    uo2.add_nuclide('U238', 0.94)
    uo2.add_nuclide('O16',  2.0)
    uo2.set_density('g/cm3', density_uo2)

    zirconium = openmc.Material(12, name='zirconium')
    zirconium.add_element('Zr', 1.0)
    zirconium.set_density('g/cm3', density_zirconium)

    graphite = openmc.Material(13, name='moderator')
    graphite.add_nuclide('C0', 1.0)
    graphite.set_density('g/cm3', density_graphite)
    graphite.add_s_alpha_beta('c_Graphite')

    s_fuel  = openmc.ZCylinder(r=r_fuel)
    s_inner = openmc.ZCylinder(r=r_clad_inner)
    s_outer = openmc.ZCylinder(r=r_clad_outer)

    fuel_cell = openmc.Cell(name='fuel',      fill=uo2,       region=-s_fuel)
    gap_cell  = openmc.Cell(name='air gap',   fill=None,      region=+s_fuel  & -s_inner)
    clad_cell = openmc.Cell(name='clad',      fill=zirconium, region=+s_inner & -s_outer)
    mod_cell  = openmc.Cell(name='moderator', fill=graphite,  region=+s_outer)

    return openmc.Universe(cells=(fuel_cell, gap_cell, clad_cell, mod_cell))


def create_moderator_universe():
    density_graphite = 2.0

    graphite = openmc.Material(21, name='moderator')
    graphite.add_nuclide('C0', 1.0)
    graphite.set_density('g/cm3', density_graphite)
    graphite.add_s_alpha_beta('c_Graphite')

    mod_cell = openmc.Cell(name='moderator', fill=graphite)
    return openmc.Universe(cells=[mod_cell])


# ---------------------------------------------------------------------------
# Model assembly
# ---------------------------------------------------------------------------

def create_openmc_model(HP):
    heat_pipe_universe = create_heat_pipe_universe(HP)
    fuel_pin_universe  = create_fuel_pin_universe()
    moderator_universe = create_moderator_universe()

    all_materials = {}
    for u in [heat_pipe_universe, fuel_pin_universe, moderator_universe]:
        for mat_id, mat in u.get_all_materials().items():
            if mat_id not in all_materials:
                all_materials[mat_id] = mat
            else:
                print(f"Warning: duplicate material ID {mat_id} ('{mat.name}') — skipping.")

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
        boundary_type='vacuum',
        orientation='x'
    )
    top    = openmc.ZPlane( 100., boundary_type='reflective')
    bottom = openmc.ZPlane(-100., boundary_type='reflective')

    main_cell     = openmc.Cell(fill=lattice, region=(-outer_surface & +bottom & -top))
    root_universe = openmc.Universe(cells=[main_cell])

    geometry = openmc.Geometry(root_universe)
    geometry.export_to_xml()

    # ------------------------------------------------------------------
    # Cartesian RegularMesh.
    # lower_left / upper_right span the full core; MESH_N_Z = 1 gives a
    # single z-bin so the tally is effectively z-integrated flux.
    # ------------------------------------------------------------------
    mesh = openmc.RegularMesh()
    mesh.lower_left  = (-MESH_HALF, -MESH_HALF, -100.)
    mesh.upper_right = ( MESH_HALF,  MESH_HALF,  100.)
    mesh.dimension   = (MESH_N_XY, MESH_N_XY, MESH_N_Z)

    mesh_filter           = openmc.MeshFilter(mesh)
    energy_filter_thermal = openmc.EnergyFilter([0.0,                E_THERMAL_CUTOFF_eV])
    energy_filter_fast    = openmc.EnergyFilter([E_THERMAL_CUTOFF_eV, 20.0e6])

    t_total = openmc.Tally(name='flux_total')
    t_total.filters = [mesh_filter]
    t_total.scores  = ['flux']

    t_thermal = openmc.Tally(name='flux_thermal')
    t_thermal.filters = [mesh_filter, energy_filter_thermal]
    t_thermal.scores  = ['flux']

    t_fast = openmc.Tally(name='flux_fast')
    t_fast.filters = [mesh_filter, energy_filter_fast]
    t_fast.scores  = ['flux']

    openmc.Tallies([t_total, t_thermal, t_fast]).export_to_xml()

    source = openmc.IndependentSource(
        space=openmc.stats.Box(
            lower_left =(-2 * lattice.pitch[0], -100., -2 * lattice.pitch[0]),
            upper_right=( 2 * lattice.pitch[0],  100.,  2 * lattice.pitch[0])
        )
    )

    settings = openmc.Settings()
    settings.batches   = 100
    settings.inactive  = 50
    settings.particles = 10000
    settings.source    = source
    settings.export_to_xml()

    return mesh.dimension, openmc.model.Model(geometry, materials, settings, tallies=openmc.Tallies([t_total, t_thermal, t_fast]))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.chdir(OUTPUT_DIR)
    os.system('rm -f summary.h5 statepoint.100.h5')

    data_Guoju_2 = {
        "r_outer":     0.03 + 0.001 + 0.0005 + 0.0005,
        "delta_wall":  0.001,
        "delta_gap":   0.0005,
        "delta_wick":  0.0005,
        "l_evap":      0.1,
        "l_adiabatic": 0.05,
        "l_cond":      0.55,
        "N_wick":      15,
        "N_wall":      15,
        "N_evap":      30,
        "N_adiabatic": 15,
        "N_cond":      165,
        "adiabatic_radial_flux": False,
        "Temperature_BC": True,
        "h_vap":    1e6,
        "h_cond":   62.6,
        "T_cond":   300,
        "T_op":     850,
        "k_wick":   66.2,
        "k_wall":   19.0,
        "P_C":      2476,
        "T_C":      856,
        "porosity": 0.7,
    }

    HP = Heatpipe(data_Guoju_2)
    mesh_dims, mod = create_openmc_model(HP)

    plot_geometry(10.)
    mod.run()

    with openmc.StatePoint('statepoint.100.h5') as sp:
        plot_total_flux_xy(sp, mesh_dims)
        plot_thermal_flux_xy(sp, mesh_dims)
        plot_fast_flux_xy(sp, mesh_dims)
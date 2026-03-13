import numpy as np
import matplotlib.pyplot as plt
import openmc
import os
import seaborn as sns
os.environ["OPENMC_CROSS_SECTIONS"] = "/home/felixpersson/MasterThesisProject/NuclearData/endfb71/endfb-vii.1-hdf5/cross_sections.xml"

from models.sodium_properties import calculate_Na_rho_l
from CoupledSystems.Heatpipe import Heatpipe


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
    plt.savefig(filename)
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


def create_heat_pipe_universe(HP):
    density_Na = 1e-3 * calculate_Na_rho_l(HP.data["T_op"]) # g/cm^3
    density_fecral_alloy = 7.15 # g/cm^3
    density_graphite = 2.0

    l_hp = HP.data.get("l_evap")

    porosity = HP.data.get("porosity")

    # Converting to cm.
    r_outer    = 1e2 * HP.data.get("r_outer")
    delta_wall = 1e2 * HP.data.get("delta_wall")
    delta_gap  = 1e2 * HP.data.get("delta_gap")
    delta_wick = 1e2 * HP.data.get("delta_wick")
    r_gap    = r_outer - delta_wall
    r_wick   = r_gap   - delta_gap
    r_vapour = r_wick  - delta_wick

    print(r_outer)

    # Creating the heat pipe materials
    fecral_alloy = openmc.Material(1, name = 'stainlessSteel')
    fecral_alloy.add_element('Fe', 0.73)
    fecral_alloy.add_element('Cr', 0.22)
    fecral_alloy.add_element('Al', 0.05)
    fecral_alloy.set_density('g/cm3', density_fecral_alloy)

    sodium = openmc.Material(2, name = 'Sodium')
    sodium.add_element('Na', 1.0)
    sodium.set_density('g/cm3', density_Na)

    wick_material = openmc.Material.mix_materials([sodium, fecral_alloy], [porosity, (1 - porosity)], 'vo')

    graphite = openmc.Material(3, name="moderator")
    graphite.add_nuclide('C0', 2.0)
    graphite.set_density('g/cm3', density_graphite)
    graphite.add_s_alpha_beta('c_Graphite')

    materials = openmc.Materials([fecral_alloy, sodium, wick_material, graphite])

    # Defining the regions
    vapour_surface = openmc.ZCylinder(r=r_vapour)
    wick_surface   = openmc.ZCylinder(r=r_wick)
    gap_surface    = openmc.ZCylinder(r=r_gap)
    wall_surface   = openmc.ZCylinder(r=r_outer)

    top    = openmc.ZPlane(100., boundary_type='reflective')
    bottom = openmc.ZPlane(-100., boundary_type='reflective')

    vapour_region = -vapour_surface
    wick_region = +vapour_surface & -wick_surface
    gap_region = +wick_surface & -gap_surface
    wall_region = +gap_surface & -wall_surface
    moderator_region = +wall_surface

    # Creating the cells
    vapour_cell = openmc.Cell(name='vapour')
    vapour_cell.fill = None
    vapour_cell.region = vapour_region

    wick_cell = openmc.Cell(name='wick')
    wick_cell.fill = wick_material
    wick_cell.region = wick_region
    
    gap_cell = openmc.Cell(name='gap')
    gap_cell.fill = sodium
    gap_cell.region = gap_region

    wall_cell = openmc.Cell(name='wall')
    wall_cell.fill = fecral_alloy
    wall_cell.region = wall_region

    moderator_cell = openmc.Cell(name='graphite')
    moderator_cell.fill = graphite
    moderator_cell.region = moderator_region

    # Adding all the cells to the heat pipe universe
    heat_pipe_universe = openmc.Universe(cells=(
        vapour_cell, 
        wick_cell, 
        gap_cell, 
        wall_cell, 
        moderator_cell))

    return heat_pipe_universe


def create_fuel_pin_universe():
    density_uo2 = 10.0
    density_zirconium = 6.6
    density_graphite = 2.0

    r_fuel = 2.2
    r_cladding_inner = 2.3
    r_cladding_outer = 2.4 

    # Creating the fuel pin materials
    uo2 = openmc.Material(11, "uo2")
    uo2.add_nuclide('U235', 0.06)
    uo2.add_nuclide('U238', 0.94)
    uo2.add_nuclide('O16', 2.0)
    uo2.set_density('g/cm3', density_uo2)

    zirconium = openmc.Material(12, name="zirconium")
    zirconium.add_element('Zr', 1.0)
    zirconium.set_density('g/cm3', density_zirconium)

    graphite = openmc.Material(13, name="moderator")
    graphite.add_nuclide('C0', 1.0)
    graphite.set_density('g/cm3', density_graphite)
    graphite.add_s_alpha_beta('c_Graphite')

    # Defining the regions
    fuel_surface           = openmc.ZCylinder(r=r_fuel)
    cladding_inner_surface = openmc.ZCylinder(r=r_cladding_inner)
    cladding_outer_surface = openmc.ZCylinder(r=r_cladding_outer)

    top    = openmc.ZPlane(100., boundary_type='reflective')
    bottom = openmc.ZPlane(-100., boundary_type='reflective')

    fuel_region      = -fuel_surface 
    gap_region       = +fuel_surface            & -cladding_inner_surface
    clad_region      = +cladding_inner_surface  & -cladding_outer_surface
    moderator_region = +cladding_outer_surface

    # Creating the cells
    fuel_cell = openmc.Cell(name='fuel')
    fuel_cell.fill = uo2
    fuel_cell.region = fuel_region

    gap_cell = openmc.Cell(name='air gap')
    gap_cell.region = gap_region

    clad_cell = openmc.Cell(name='clad')
    clad_cell.fill = zirconium
    clad_cell.region = clad_region

    moderator_cell = openmc.Cell(name='moderator')
    moderator_cell.fill = graphite
    moderator_cell.region = moderator_region

    # Creating the fuel pin universe
    fuel_pin_universe = openmc.Universe(cells=(fuel_cell, gap_cell, clad_cell, moderator_cell))

    return fuel_pin_universe


def create_moderator_universe():
    density_graphite = 2.0

    # Creating the fuel pin materials
    graphite = openmc.Material(21, name="moderator")
    graphite.add_nuclide('C0', 1.0)
    graphite.set_density('g/cm3', density_graphite)
    graphite.add_s_alpha_beta('c_Graphite')

    # Creating region
    top    = openmc.ZPlane(100., boundary_type='reflective')
    bottom = openmc.ZPlane(-100., boundary_type='reflective')

    # Creating the cells
    moderator_cell = openmc.Cell(name='moderator')
    moderator_cell.fill = graphite

    # Creating the fuel pin universe
    fuel_pin_universe = openmc.Universe(cells=[moderator_cell])

    return fuel_pin_universe


def create_openmc_model(HP):
    # Define inner universes 
    heat_pipe_universe = create_heat_pipe_universe(HP)
    fuel_pin_universe = create_fuel_pin_universe()
    moderator_universe = create_moderator_universe()

    # Create Materials collection
    universes = [heat_pipe_universe, fuel_pin_universe, moderator_universe]
    all_materials = {}
    for u in universes:
        for mat_id, mat in u.get_all_materials().items():
            if mat_id in all_materials:
                print(f"Warning: duplicate material ID {mat_id} ('{mat.name}') — skipping.")
            else:
                all_materials[mat_id] = mat

    materials = openmc.Materials(list(all_materials.values()))
    materials.export_to_xml()

    # Define the hexagonal geometry
    lattice = openmc.HexLattice()
    lattice.center = (0., 0.)
    lattice.pitch = (10,)
    lattice.outer = moderator_universe

    # Construct the lattice universe from the inner universes
    ring_0 = 6 * [moderator_universe, fuel_pin_universe, fuel_pin_universe]

    ring_1 = 6 * [fuel_pin_universe, heat_pipe_universe]

    ring_2 = 6 * [fuel_pin_universe]

    inner_ring = [heat_pipe_universe]

    lattice.universes = [ring_0, ring_1, ring_2, inner_ring]

    # Place the hexagonal lattice in a outer hexagonal shape.    
    outer_surface = openmc.model.HexagonalPrism(edge_length=3.8*lattice.pitch[0], boundary_type='reflective', orientation="x")
    top    = openmc.ZPlane(100., boundary_type='reflective')
    bottom = openmc.ZPlane(-100., boundary_type='reflective')

    main_cell = openmc.Cell(fill=lattice, region=(-outer_surface & +bottom & -top))

    # Create the root universe
    root_universe = openmc.Universe(cells=[main_cell])

    # Create and export the geometry
    geometry = openmc.Geometry(root_universe)
    geometry.export_to_xml()

    # Create mesh tallies for axial power and flux
    mesh = openmc.RegularMesh()

    MESH_N_XY        = 200    # bins per side in x and y
    MESH_N_Z         = 1      # single z-bin; z-integrated flux
    MESH_HALF        = 2.0 * lattice.pitch[0]
    mesh.lower_left  = (-MESH_HALF, -MESH_HALF, -100.)
    mesh.upper_right = ( MESH_HALF,  MESH_HALF,  100.)
    mesh.dimension   = (MESH_N_XY, MESH_N_XY, MESH_N_Z)
 
    E_THERMAL_CUTOFF_eV   = 0.625
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

    # Create a point source
    source = openmc.IndependentSource(
        space=openmc.stats.Box(
            lower_left=(-2*lattice.pitch[0], -100., -2*lattice.pitch[0]),
            upper_right=( 2*lattice.pitch[0],  100.,  2*lattice.pitch[0])
        )
    )

    # Set up OpenMC settings
    settings = openmc.Settings()
    settings.batches = 100
    settings.inactive = 50
    settings.particles = 10000
    settings.source = source
    settings.export_to_xml()
    
    # # Plot geometry
    # openmc.plot_geometry()
    
    return cylindrical_mesh.dimension, r_grid, phi_grid, openmc.model.Model(geometry, materials, settings, tallies)


if __name__ == "__main__":
    os.system('rm summary.h5')
    os.system('rm statepoint.100.h5')

    data_Guoju_2 = {
        "r_outer": .03 + 0.001 + 0.0005 + 0.0005,
        "delta_wall": 0.001,
        "delta_gap": 0.0005,
        "delta_wick": 0.0005,
        "l_evap": 0.1,
        "l_adiabatic": 0.05,
        "l_cond": 0.55,

        "N_wick": 15,
        "N_wall": 15,
        "N_evap": 30,
        "N_adiabatic": 15,
        "N_cond": 165,

        "adiabatic_radial_flux": False,
        "Temperature_BC": True,
        "h_vap": 1e6,
        "h_cond": 62.6,
        "T_cond": 300,
        "T_op": 850,

        "k_wick": 66.2,
        "k_wall": 19.0,

        "P_C": 2476,
        "T_C": 856,

        "porosity": 0.7,
    }


    HP = Heatpipe(data_Guoju_2)
    LATTICE_PITCH = 10.  # cm
 
    cylindrical_mesh_dims, r_grid, phi_grid, mod = create_openmc_model(HP)
    plot_geometry(LATTICE_PITCH)
    mod.run()
 
 
    with openmc.StatePoint('statepoint.100.h5') as sp:
        # --- Original radial flux plot ---
        tally_v = sp.get_tally(name='axial peaking')
        fission_rates = tally_v.get_values(scores=['flux'])
        fission_rates.shape = cylindrical_mesh_dims
 
        average_radial_power = np.zeros(cylindrical_mesh_dims[0])
        for i in range(cylindrical_mesh_dims[0]):
            average_radial_power[i] = np.mean(fission_rates[i, :, :])
 
        plt.plot(average_radial_power)
        plt.show()
 
        # --- Thermal and fast flux plots ---
        plot_thermal_flux(sp, cylindrical_mesh_dims, LATTICE_PITCH)
        plot_fast_flux(sp, cylindrical_mesh_dims, LATTICE_PITCH)

        # --- XY imshow plots at z=0 ---
        plot_thermal_flux_xy(sp, cylindrical_mesh_dims, r_grid, phi_grid)
        plot_fast_flux_xy(sp, cylindrical_mesh_dims, r_grid, phi_grid)
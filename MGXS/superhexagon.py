import math
import numpy as np
import matplotlib.pyplot as plt
import openmc
import openmc.mgxs


# =============================================================================
# PARAMETERS
# =============================================================================

# -------------------------
# Pin / rod geometry [cm]
# -------------------------
fuel_radius = 0.40
gap_radius = 0.41
clad_radius = 0.47
copper_radius = 0.35

# Thin slab thickness for effectively 2D model
height = 1.0

# -------------------------
# Inner assembly geometry
# -------------------------
r_center_to_inner_fuel = 1.35
r_center_to_outer_copper = 3.00
r_outer_copper_to_fuel = 1.35

# Hex edge length of one assembly
assembly_hex_edge = 5.8

# -------------------------
# Outer 7-assembly lattice
# -------------------------
# center-to-center pitch between neighboring hex assemblies
# For a regular hexagon with edge length a, the center-to-center distance
# between side-sharing neighbors is sqrt(3) * a
assembly_pitch = math.sqrt(3.0) * assembly_hex_edge

# Outer boundary of full 7-assembly system
# Must be large enough to contain the center assembly + the surrounding ring.
outer_hex_edge = 2.8 * assembly_hex_edge

# -------------------------
# Tallies / mesh
# -------------------------
mesh_nx = 320
mesh_ny = 320

# MGXS groups [eV]
group_edges = [0.0, 0.625, 20.0e6]


# =============================================================================
# MATERIALS
# =============================================================================
fuel = openmc.Material(name='UO2 fuel')
fuel.set_density('g/cm3', 10.4)
fuel.add_element('U', 1.0, enrichment=4.5)
fuel.add_element('O', 2.0)

gap = openmc.Material(name='He gap')
gap.set_density('g/cm3', 0.0016)
gap.add_element('He', 1.0)

clad = openmc.Material(name='Zircaloy')
clad.set_density('g/cm3', 6.55)
clad.add_element('Zr', 1.0)

graphite = openmc.Material(name='Graphite moderator')
graphite.set_density('g/cm3', 1.70)
graphite.add_element('C', 1.0)
# graphite.add_s_alpha_beta('c_Graphite')  # uncomment if supported by your library

copper = openmc.Material(name='Copper')
copper.set_density('g/cm3', 8.96)
copper.add_element('Cu', 1.0)

materials = openmc.Materials([fuel, gap, clad, graphite, copper])


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def hex_ring_positions(radius):
    """Six Cartesian positions on a hexagon of circumradius = radius."""
    pts = []
    for k in range(6):
        ang = math.radians(60.0 * k)
        x = radius * math.cos(ang)
        y = radius * math.sin(ang)
        pts.append((x, y))
    return pts


def shifted_hex_ring(center, radius):
    cx, cy = center
    return [(cx + x, cy + y) for x, y in hex_ring_positions(radius)]


def deduplicate_points(points, tol=1e-6):
    unique = []
    for p in points:
        if not any(abs(p[0] - q[0]) < tol and abs(p[1] - q[1]) < tol for q in unique):
            unique.append(p)
    return unique


def is_inside_hex_y_orientation(x, y, edge):
    """
    Conservative point-in-hex test for OpenMC HexagonalPrism(orientation='y').
    """
    sqrt3 = math.sqrt(3.0)
    return (
        abs(x) <= sqrt3 * edge / 2.0 and
        abs(y) <= edge and
        abs(sqrt3 * x + y) <= sqrt3 * edge and
        abs(-sqrt3 * x + y) <= sqrt3 * edge
    )


# =============================================================================
# BUILD ONE INNER ASSEMBLY UNIVERSE
# =============================================================================
def make_inner_assembly_universe():
    cells = []

    copper_positions = [(0.0, 0.0)] + hex_ring_positions(r_center_to_outer_copper)

    fuel_positions = []
    fuel_positions += shifted_hex_ring((0.0, 0.0), r_center_to_inner_fuel)
    for cp in hex_ring_positions(r_center_to_outer_copper):
        fuel_positions += shifted_hex_ring(cp, r_outer_copper_to_fuel)
    fuel_positions = deduplicate_points(fuel_positions)

    for i, (x, y) in enumerate(copper_positions):
        cyl = openmc.ZCylinder(x0=x, y0=y, r=copper_radius)
        cells.append(openmc.Cell(fill=copper, region=-cyl))

    for i, (x, y) in enumerate(fuel_positions):
        surf_fuel = openmc.ZCylinder(x0=x, y0=y, r=fuel_radius)
        surf_gap = openmc.ZCylinder(x0=x, y0=y, r=gap_radius)
        surf_clad = openmc.ZCylinder(x0=x, y0=y, r=clad_radius)

        cells.append(openmc.Cell(fill=fuel, region=-surf_fuel))
        cells.append(openmc.Cell(fill=gap, region=+surf_fuel & -surf_gap))
        cells.append(openmc.Cell(fill=clad, region=+surf_gap & -surf_clad))

    cells.append(openmc.Cell(fill=graphite))
    return openmc.Universe(cells=cells)


inner_assembly_universe = make_inner_assembly_universe()


# =============================================================================
# BUILD 7-ASSEMBLY SUPERLATTICE
# =============================================================================
super_lat = openmc.HexLattice(name='7-assembly lattice')
super_lat.center = (0.0, 0.0)
super_lat.pitch = [assembly_pitch]
super_lat.orientation = 'y'

# Outer ring of 6 assemblies, plus 1 center assembly
ring1 = [inner_assembly_universe] * 6
ring0 = [inner_assembly_universe]

# For HexLattice: universes = [outer_ring, ..., inner_ring]
super_lat.universes = [ring1, ring0]

# What fills outside the listed rings but inside the outer supercell?
# Graphite is a reasonable choice.
graphite_universe = openmc.Universe(cells=[openmc.Cell(fill=graphite)])
super_lat.outer = graphite_universe


# =============================================================================
# ROOT GEOMETRY
# =============================================================================
zmin = openmc.ZPlane(z0=-height/2, boundary_type='reflective')
zmax = openmc.ZPlane(z0= height/2, boundary_type='reflective')

outer_hex = openmc.model.HexagonalPrism(
    edge_length=outer_hex_edge,
    orientation='y',
    boundary_type='vacuum'
)

root_cell = openmc.Cell(fill=super_lat, region=-outer_hex & +zmin & -zmax)
root_universe = openmc.Universe(cells=[root_cell])
geometry = openmc.Geometry(root_universe)


# =============================================================================
# SETTINGS
# =============================================================================
settings = openmc.Settings()
settings.run_mode = 'eigenvalue'
settings.batches = 120
settings.inactive = 30
settings.particles = 30000

source = openmc.IndependentSource()
source.space = openmc.stats.Box(
    [-outer_hex_edge, -outer_hex_edge, -height/2],
    [ outer_hex_edge,  outer_hex_edge,  height/2],
    only_fissionable=False
)
settings.source = source


# =============================================================================
# PLOTS
# =============================================================================
plot = openmc.Plot()
plot.filename = 'seven_hex_assemblies'
plot.basis = 'xy'
plot.width = (2.1 * outer_hex_edge, 2.1 * outer_hex_edge)
plot.pixels = (1200, 1200)
plot.color_by = 'material'
plot.colors = {
    fuel: 'gold',
    gap: 'white',
    clad: 'gray',
    graphite: 'black',
    copper: 'orange',
}
plots = openmc.Plots([plot])


# =============================================================================
# TALLIES
# =============================================================================
tallies = openmc.Tallies()

# Mesh flux tally over whole 7-assembly system
mesh = openmc.RegularMesh(name='supercell_mesh')
mesh.lower_left = (-outer_hex_edge, -outer_hex_edge, -height/2)
mesh.upper_right = ( outer_hex_edge,  outer_hex_edge,  height/2)
mesh.dimension = (mesh_nx, mesh_ny, 1)

mesh_tally = openmc.Tally(name='mesh flux')
mesh_tally.filters = [openmc.MeshFilter(mesh)]
mesh_tally.scores = ['flux']
tallies.append(mesh_tally)

# MGXS by material
groups = openmc.mgxs.EnergyGroups(group_edges=group_edges)

mgxs_lib = openmc.mgxs.Library(geometry)
mgxs_lib.energy_groups = groups
mgxs_lib.domain_type = 'material'
mgxs_lib.domains = [fuel, graphite, copper]
mgxs_lib.by_nuclide = False
mgxs_lib.mgxs_types = ['total', 'absorption', 'nu-fission']
mgxs_lib.build_library()
mgxs_lib.add_to_tallies(tallies, merge=True)


# =============================================================================
# EXPORT + RUN
# =============================================================================
model = openmc.Model(
    geometry=geometry,
    materials=materials,
    settings=settings,
    tallies=tallies,
    plots=plots
)

model.export_to_xml()

# Optional preview:
# openmc.plot_geometry()

statepoint_path = model.run()


# =============================================================================
# POSTPROCESSING
# =============================================================================
sp = openmc.StatePoint(statepoint_path)
summary = openmc.Summary('summary.h5')
sp.link_with_summary(summary)

mgxs_lib.load_from_statepoint(sp)


# -----------------------------------------------------------------------------
# Plot mesh flux
# -----------------------------------------------------------------------------
flux_tally = sp.get_tally(name='mesh flux')
flux = flux_tally.mean.ravel().reshape((mesh_ny, mesh_nx))

fig1, ax1 = plt.subplots(figsize=(9, 8))
im = ax1.imshow(
    flux,
    origin='lower',
    extent=(-outer_hex_edge, outer_hex_edge, -outer_hex_edge, outer_hex_edge),
    interpolation='nearest',
    cmap="hot"
)
ax1.set_title('2D Mesh Flux: Central Hexagon + One Ring of Hexagons')
ax1.set_xlabel('x [cm]')
ax1.set_ylabel('y [cm]')
fig1.colorbar(im, ax=ax1, label='Flux')
fig1.tight_layout()
fig1.savefig('mesh_flux_7assemblies.png', dpi=200)


# -----------------------------------------------------------------------------
# Plot MGXS
# -----------------------------------------------------------------------------
group_labels = ['thermal', 'fast']
materials_to_plot = [fuel, graphite, copper]
xs_names = ['total', 'absorption', 'nu-fission']

fig2, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True)

for ax, xs_name in zip(axes, xs_names):
    for mat in materials_to_plot:
        mgxs = mgxs_lib.get_mgxs(mat, xs_name)
        xs = mgxs.get_xs(xs_type='macro', value='mean', order_groups='increasing')
        ax.plot(group_labels, xs, marker='o', label=mat.name)

    ax.set_title(f'Macro {xs_name}')
    ax.set_ylabel(r'$\Sigma$ [1/cm]')
    ax.grid(True, alpha=0.3)

axes[0].legend()
fig2.tight_layout()
fig2.savefig('mgxs_materials_7assemblies.png', dpi=200)

plt.show()
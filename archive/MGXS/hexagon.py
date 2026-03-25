import math
import numpy as np
import matplotlib.pyplot as plt
import openmc
import openmc.mgxs


class HexLatticeMGXSModel:
    def __init__(self):
        # =========================================================================
        # GEOMETRIC PARAMETERS [cm]
        # =========================================================================
        self.fuel_radius = 0.40
        self.gap_radius = 0.41
        self.clad_radius = 0.47
        self.copper_radius = 0.35

        self.height = 1.0  # thin axial slab for effectively 2D model

        # Distances between object centers
        self.r_center_to_inner_fuel = 1.35
        self.r_center_to_outer_copper = 3.00
        self.r_outer_copper_to_fuel = 1.35

        # Outer hex boundary
        self.hex_edge = 5.8

        # =========================================================================
        # TALLY / MESH / ENERGY GROUP PARAMETERS
        # =========================================================================
        self.mesh_nx = 220
        self.mesh_ny = 220
        self.group_edges = [0.0, 0.625, 20.0e6]

        # =========================================================================
        # MONTE CARLO SETTINGS
        # =========================================================================
        self.batches = 120
        self.inactive = 30
        self.particles = 20000

        # =========================================================================
        # OUTPUT FILE NAMES
        # =========================================================================
        self.geometry_plot_filename = "geometry_xy"
        self.mesh_flux_plot_filename = "mesh_flux.png"
        self.mgxs_plot_filename = "mgxs_materials.png"

        # =========================================================================
        # INTERNAL OBJECTS
        # =========================================================================
        self.fuel = None
        self.gap = None
        self.clad = None
        self.graphite = None
        self.copper = None
        self.materials = None

        self.copper_positions = None
        self.fuel_positions = None

        self.geometry = None
        self.settings = None
        self.plots = None
        self.tallies = None
        self.mgxs_lib = None
        self.model = None

    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    @staticmethod
    def hex_ring_positions(radius):
        """Return 6 Cartesian positions on a hexagon of circumradius = radius."""
        pts = []
        for k in range(6):
            ang = math.radians(60.0 * k)
            x = radius * math.cos(ang)
            y = radius * math.sin(ang)
            pts.append((x, y))
        return pts

    @classmethod
    def shifted_hex_ring(cls, center, radius):
        """Return 6 positions around a given center."""
        cx, cy = center
        pts = []
        for x, y in cls.hex_ring_positions(radius):
            pts.append((cx + x, cy + y))
        return pts

    @staticmethod
    def deduplicate_points(points, tol=1e-6):
        """Remove repeated / nearly repeated points."""
        unique = []
        for p in points:
            if not any(abs(p[0] - q[0]) < tol and abs(p[1] - q[1]) < tol for q in unique):
                unique.append(p)
        return unique

    @staticmethod
    def is_inside_hex_y_orientation(x, y, edge):
        """
        Check if point is inside a regular hex centered at origin,
        flat sides left/right ('y' orientation used by OpenMC HexagonalPrism).
        """
        sqrt3 = math.sqrt(3.0)
        return (
            abs(x) <= sqrt3 * edge / 2.0
            and abs(y) <= edge
            and abs(sqrt3 * x + y) <= sqrt3 * edge
            and abs(-sqrt3 * x + y) <= sqrt3 * edge
        )

    # =========================================================================
    # BUILD MATERIALS
    # =========================================================================
    def build_materials(self):
        self.fuel = openmc.Material(name="UO2 fuel")
        self.fuel.set_density("g/cm3", 10.4)
        self.fuel.add_element("U", 1.0, enrichment=4.5)
        self.fuel.add_element("O", 2.0)

        self.gap = openmc.Material(name="He gap")
        self.gap.set_density("g/cm3", 0.0016)
        self.gap.add_element("He", 1.0)

        self.clad = openmc.Material(name="Zircaloy")
        self.clad.set_density("g/cm3", 6.55)
        self.clad.add_element("Zr", 1.0)

        self.graphite = openmc.Material(name="Graphite moderator")
        self.graphite.set_density("g/cm3", 1.70)
        self.graphite.add_element("C", 1.0)
        # self.graphite.add_s_alpha_beta('c_Graphite')  # Uncomment if available

        # https://en.wikipedia.org/wiki/Kanthal_%28alloy%29?.com
        # https://www.electrokit.com/upload/product/41020/41020658/41020658.pdf

        self.fecral = openmc.Material(name="FeCrAl heat pipe alloy")
        self.fecral.set_density("g/cm3", 7.15)

        # Representative FeCrAl composition (wt%)
        self.fecral.add_element("Fe", 0.73, percent_type="wo")
        self.fecral.add_element("Cr", 0.22, percent_type="wo")
        self.fecral.add_element("Al", 0.05, percent_type="wo")

        self.materials = openmc.Materials([
            self.fuel,
            self.gap,
            self.clad,
            self.graphite,
            self.fecral,
        ])

    # =========================================================================
    # BUILD OBJECT POSITIONS
    # =========================================================================
    def build_positions(self):
        # Copper rods: one central + 6 outer
        copper_positions = [(0.0, 0.0)] + self.hex_ring_positions(self.r_center_to_outer_copper)

        # Fuel pins:
        # - 6 around center copper
        # - 6 around each outer copper
        fuel_positions = []
        fuel_positions += self.shifted_hex_ring((0.0, 0.0), self.r_center_to_inner_fuel)

        for cp in self.hex_ring_positions(self.r_center_to_outer_copper):
            fuel_positions += self.shifted_hex_ring(cp, self.r_outer_copper_to_fuel)

        fuel_positions = self.deduplicate_points(fuel_positions)

        # Keep only positions safely inside the outer hex
        fuel_positions = [
            p for p in fuel_positions
            if self.is_inside_hex_y_orientation(
                p[0], p[1], self.hex_edge - self.clad_radius - 0.05
            )
        ]

        copper_positions = [
            p for p in copper_positions
            if self.is_inside_hex_y_orientation(
                p[0], p[1], self.hex_edge - self.copper_radius - 0.05
            )
        ]

        self.fuel_positions = fuel_positions
        self.copper_positions = copper_positions

    # =========================================================================
    # BUILD GEOMETRY
    # =========================================================================
    def build_geometry(self):
        zmin = openmc.ZPlane(z0=-self.height / 2, boundary_type="reflective")
        zmax = openmc.ZPlane(z0= self.height / 2, boundary_type="reflective")

        outer_hex = openmc.model.HexagonalPrism(
            edge_length=self.hex_edge,
            orientation="y",
            boundary_type="vacuum"
        )

        cells = []

        # Graphite background first; holes are carved out as rods/pins are added
        background_region = -outer_hex & +zmin & -zmax

        # --- Copper rods ---
        for i, (x, y) in enumerate(self.copper_positions):
            cyl = openmc.ZCylinder(x0=x, y0=y, r=self.copper_radius)
            rod_region = -cyl & -outer_hex & +zmin & -zmax
            cells.append(
                openmc.Cell(
                    name=f"copper_{i}",
                    fill=self.copper,
                    region=rod_region
                )
            )
            background_region &= +cyl

        # --- Fuel pins ---
        for i, (x, y) in enumerate(self.fuel_positions):
            surf_fuel = openmc.ZCylinder(x0=x, y0=y, r=self.fuel_radius)
            surf_gap = openmc.ZCylinder(x0=x, y0=y, r=self.gap_radius)
            surf_clad = openmc.ZCylinder(x0=x, y0=y, r=self.clad_radius)

            fuel_region = -surf_fuel & -outer_hex & +zmin & -zmax
            gap_region = +surf_fuel & -surf_gap & -outer_hex & +zmin & -zmax
            clad_region = +surf_gap & -surf_clad & -outer_hex & +zmin & -zmax

            cells.append(openmc.Cell(name=f"fuel_{i}", fill=self.fuel, region=fuel_region))
            cells.append(openmc.Cell(name=f"gap_{i}", fill=self.gap, region=gap_region))
            cells.append(openmc.Cell(name=f"clad_{i}", fill=self.clad, region=clad_region))

            background_region &= +surf_clad

        # Graphite moderator everywhere else
        cells.append(
            openmc.Cell(
                name="graphite_background",
                fill=self.graphite,
                region=background_region
            )
        )

        root = openmc.Universe(cells=cells)
        self.geometry = openmc.Geometry(root)

    # =========================================================================
    # BUILD SETTINGS
    # =========================================================================
    def build_settings(self):
        settings = openmc.Settings()
        settings.run_mode = "eigenvalue"
        settings.batches = self.batches
        settings.inactive = self.inactive
        settings.particles = self.particles

        source = openmc.IndependentSource()
        source.space = openmc.stats.Box(
            [-self.hex_edge, -self.hex_edge, -self.height / 2],
            [ self.hex_edge,  self.hex_edge,  self.height / 2],
            only_fissionable=False
        )
        settings.source = source

        self.settings = settings

    # =========================================================================
    # BUILD PLOTS
    # =========================================================================
    def build_plots(self):
        plot = openmc.Plot()
        plot.filename = self.geometry_plot_filename
        plot.basis = "xy"
        plot.width = (2.1 * self.hex_edge, 2.1 * self.hex_edge)
        plot.pixels = (1000, 1000)
        plot.color_by = "material"
        plot.colors = {
            self.fuel: "gold",
            self.gap: "white",
            self.clad: "gray",
            self.graphite: "black",
            self.copper: "orange",
        }

        self.plots = openmc.Plots([plot])

    # =========================================================================
    # BUILD TALLIES AND MGXS
    # =========================================================================
    def build_tallies(self):
        tallies = openmc.Tallies()

        # Mesh flux tally
        mesh = openmc.RegularMesh(name="mesh")
        mesh.lower_left = (-self.hex_edge, -self.hex_edge, -self.height / 2)
        mesh.upper_right = ( self.hex_edge,  self.hex_edge,  self.height / 2)
        mesh.dimension = (self.mesh_nx, self.mesh_ny, 1)

        mesh_tally = openmc.Tally(name="mesh flux")
        mesh_tally.filters = [openmc.MeshFilter(mesh)]
        mesh_tally.scores = ["flux"]
        tallies.append(mesh_tally)

        # MGXS setup
        groups = openmc.mgxs.EnergyGroups(group_edges=self.group_edges)

        mgxs_lib = openmc.mgxs.Library(self.geometry)
        mgxs_lib.energy_groups = groups
        mgxs_lib.domain_type = "material"
        mgxs_lib.domains = [self.fuel, self.graphite, self.copper]
        mgxs_lib.by_nuclide = False
        mgxs_lib.mgxs_types = ["total", "absorption", "nu-fission"]
        mgxs_lib.build_library()
        mgxs_lib.add_to_tallies(tallies, merge=True)

        self.tallies = tallies
        self.mgxs_lib = mgxs_lib

    # =========================================================================
    # BUILD FULL MODEL
    # =========================================================================
    def build_model(self):
        self.build_materials()
        self.build_positions()
        self.build_geometry()
        self.build_settings()
        self.build_plots()
        self.build_tallies()

        self.model = openmc.Model(
            geometry=self.geometry,
            materials=self.materials,
            settings=self.settings,
            tallies=self.tallies,
            plots=self.plots
        )

    # =========================================================================
    # EXPORT XML
    # =========================================================================
    def export_to_xml(self):
        if self.model is None:
            self.build_model()
        self.model.export_to_xml()

    # =========================================================================
    # OPTIONAL GEOMETRY PLOT
    # =========================================================================
    def plot_geometry(self):
        if self.model is None:
            self.build_model()
        openmc.plot_geometry()

    # =========================================================================
    # RUN MODEL
    # =========================================================================
    def run(self):
        if self.model is None:
            self.build_model()

        self.model.export_to_xml()
        statepoint_path = self.model.run()
        return statepoint_path

    # =========================================================================
    # POSTPROCESSING
    # =========================================================================
    def load_statepoint_and_mgxs(self, statepoint_path, summary_path="summary.h5"):
        sp = openmc.StatePoint(statepoint_path)
        summary = openmc.Summary(summary_path)
        sp.link_with_summary(summary)

        self.mgxs_lib.load_from_statepoint(sp)
        return sp

    def plot_mesh_flux(self, sp):
        flux_tally = sp.get_tally(name="mesh flux")
        flux = flux_tally.mean.ravel().reshape((self.mesh_ny, self.mesh_nx))

        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(
            flux,
            origin="lower",
            extent=(-self.hex_edge, self.hex_edge, -self.hex_edge, self.hex_edge),
            interpolation="nearest",
            cmap="hot"
        )
        ax.set_title("2D Mesh Flux")
        ax.set_xlabel("x [cm]")
        ax.set_ylabel("y [cm]")
        fig.colorbar(im, ax=ax, label="Flux")
        fig.tight_layout()
        fig.savefig(self.mesh_flux_plot_filename, dpi=200)

        return fig, ax

    def plot_mgxs(self):
        group_labels = ["thermal", "fast"]
        materials_to_plot = [self.fuel, self.graphite, self.copper]
        xs_names = ["total", "absorption", "nu-fission"]

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True)

        for ax, xs_name in zip(axes, xs_names):
            for mat in materials_to_plot:
                mgxs = self.mgxs_lib.get_mgxs(mat, xs_name)
                xs = mgxs.get_xs(
                    xs_type="macro",
                    value="mean",
                    order_groups="increasing"
                )
                ax.plot(group_labels, xs, marker="o", label=mat.name)

            ax.set_title(f"Macro {xs_name}")
            ax.set_ylabel(r"$\Sigma$ [1/cm]")
            ax.grid(True, alpha=0.3)

        axes[0].legend()
        fig.tight_layout()
        fig.savefig(self.mgxs_plot_filename, dpi=200)

        return fig, axes

    # =========================================================================
    # CONVENIENCE METHOD
    # =========================================================================
    def run_all(self, summary_path="summary.h5", show=True):
        statepoint_path = self.run()
        sp = self.load_statepoint_and_mgxs(statepoint_path, summary_path=summary_path)

        self.plot_mesh_flux(sp)
        self.plot_mgxs()

        if show:
            plt.show()

        return sp


if __name__ == "__main__":
    model = HexLatticeMGXSModel()
    model.run_all()
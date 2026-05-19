import sys
import numpy as np
import gmsh # type: ignore
import meshio # type: ignore


def create_hexagonal_reactor_mesh_old(data, show_mesh=True):
    # Adding the points of a slice of a hexagonal fuel assembly 
    # Only a 1/12 of the fuel assembly geometry needs meshing due to symmetry in the model.
    theta_hex = (np.pi / 6)

    l_pitch     = data.get("l_pitch")
    r_HP        = data.get("r_HP")
    r_f         = data.get("r_fuel_pin")
    l_mesh_size = data.get("l_mesh_size")

    gmsh.initialize()
    gmsh.model.add("Cell of hex fuel assembly")


    p1  = gmsh.model.geo.addPoint(0,                                                          0,                            0, l_mesh_size)
    p2  = gmsh.model.geo.addPoint(0,                                                          r_HP,                         0, l_mesh_size)

    p3  = gmsh.model.geo.addPoint(0,                                                          l_pitch * 3./2. - r_f,        0, l_mesh_size)
    p4  = gmsh.model.geo.addPoint(0,                                                          l_pitch * 3./2.,              0, l_mesh_size)
    p5  = gmsh.model.geo.addPoint(r_f,                                                        l_pitch * 3./2.,              0, l_mesh_size)
    p6  = gmsh.model.geo.addPoint(0,                                                          l_pitch * 3./2. + r_f,        0, l_mesh_size)

    p7  = gmsh.model.geo.addPoint(0,                                                          l_pitch * 5./2. - r_f,        0, l_mesh_size)
    p8  = gmsh.model.geo.addPoint(0,                                                          l_pitch * 5./2.,              0, l_mesh_size)
    p9  = gmsh.model.geo.addPoint(r_f,                                                        l_pitch * 5./2.,              0, l_mesh_size)
    p10 = gmsh.model.geo.addPoint(0,                                                          l_pitch * 5./2. + r_f,        0, l_mesh_size)

    p11 = gmsh.model.geo.addPoint(0,                                                          l_pitch * 4,                0, l_mesh_size)
    p12 = gmsh.model.geo.addPoint(np.tan(theta_hex) * 4 * l_pitch,                            4 * l_pitch ,                0, l_mesh_size)

    p13 = gmsh.model.geo.addPoint(np.tan(theta_hex) * 2 * l_pitch + r_HP * np.sin(theta_hex), 2 * l_pitch + r_HP * np.cos(theta_hex), 0, l_mesh_size)
    p14 = gmsh.model.geo.addPoint(np.tan(theta_hex) * 2 * l_pitch,                            2 * l_pitch,                            0, l_mesh_size)
    p15 = gmsh.model.geo.addPoint(np.tan(theta_hex) * 2 * l_pitch - r_HP,                     2 * l_pitch,                            0, l_mesh_size)
    p16 = gmsh.model.geo.addPoint(np.tan(theta_hex) * 2 * l_pitch - r_HP * np.sin(theta_hex), 2 * l_pitch - r_HP * np.cos(theta_hex), 0, l_mesh_size)

    p17 = gmsh.model.geo.addPoint(r_HP * np.sin(theta_hex),                                   r_HP * np.cos(theta_hex),   0, l_mesh_size)

    p18 = gmsh.model.geo.addPoint(np.tan(theta_hex) * 2 * l_pitch,                            l_pitch * 7./2.,      0, l_mesh_size)   
    p19 = gmsh.model.geo.addPoint(np.tan(theta_hex) * 2 * l_pitch+r_f,                        l_pitch * 7./2.,      0, l_mesh_size)
    p20 = gmsh.model.geo.addPoint(np.tan(theta_hex) * 2 * l_pitch,                            l_pitch * 7./2.+r_f,  0, l_mesh_size)
    p21 = gmsh.model.geo.addPoint(np.tan(theta_hex) * 2 * l_pitch-r_f,                        l_pitch * 7./2.,      0, l_mesh_size)
    p22 = gmsh.model.geo.addPoint(np.tan(theta_hex) * 2 * l_pitch,                            l_pitch * 7./2.-r_f,  0, l_mesh_size)

    # Hex-slice outer loop
    Outer_line1 = gmsh.model.geo.addLine(p1, p11)
    Outer_line2 = gmsh.model.geo.addLine(p11, p12)
    Outer_line3 = gmsh.model.geo.addLine(p12, p1)
    gmsh.model.geo.addCurveLoop([Outer_line1, Outer_line2, Outer_line3], 1)

    # Innermost Heat pipe
    HP1_line1       = gmsh.model.geo.addLine(p1, p2)
    HP1_circle_arc1 = gmsh.model.geo.addCircleArc(p2, p1, p17)
    HP1_line2       = gmsh.model.geo.addLine(p17, p1)
    gmsh.model.geo.addCurveLoop([HP1_line1, HP1_circle_arc1, HP1_line2], 2)

    # fuel 1
    Fuel1_line1       = gmsh.model.geo.addLine(p3, p6)  # p3 -> p4 -> p6 collapsed
    Fuel1_circle_arc1 = gmsh.model.geo.addCircleArc(p6, p4, p5)
    Fuel1_circle_arc2 = gmsh.model.geo.addCircleArc(p5, p4, p3)
    gmsh.model.geo.addCurveLoop([Fuel1_line1, Fuel1_circle_arc1, Fuel1_circle_arc2], 3)

    # fuel 2
    Fuel2_line1       = gmsh.model.geo.addLine(p7, p10)
    Fuel2_circle_arc1 = gmsh.model.geo.addCircleArc(p10, p8, p9)
    Fuel2_circle_arc2 = gmsh.model.geo.addCircleArc(p9, p8, p7)
    gmsh.model.geo.addCurveLoop([Fuel2_line1, Fuel2_circle_arc1, Fuel2_circle_arc2], 4)

    # fuel 3
    Fuel3_circle_arc1 = gmsh.model.geo.addCircleArc(p19, p18, p20)
    Fuel3_circle_arc2 = gmsh.model.geo.addCircleArc(p20, p18, p21)
    Fuel3_circle_arc3 = gmsh.model.geo.addCircleArc(p21, p18, p22)
    Fuel3_circle_arc4 = gmsh.model.geo.addCircleArc(p22, p18, p19)
    gmsh.model.geo.addCurveLoop([Fuel3_circle_arc1, Fuel3_circle_arc2, Fuel3_circle_arc3, Fuel3_circle_arc4], 7)

    # Outer Heat pipe
    HP2_line1       = gmsh.model.geo.addLine(p13, p14)
    HP2_line2       = gmsh.model.geo.addLine(p14, p16)
    HP2_circle_arc1 = gmsh.model.geo.addCircleArc(p16, p14, p15)
    HP2_circle_arc2 = gmsh.model.geo.addCircleArc(p15, p14, p13)
    gmsh.model.geo.addCurveLoop([HP2_line1, HP2_line2, HP2_circle_arc1, HP2_circle_arc2], 5)

    # Moderator, thanks Claude (Not, all was wrong :( ))!
    # Continue up symmetry axis p2 -> p3
    Mod_line_p2_p3      = gmsh.model.geo.addLine(p2, p3)

    # Fuel1 detour
    Mod_Fuel1_arc1      = gmsh.model.geo.addCircleArc(p3, p4, p5)
    Mod_Fuel1_arc2      = gmsh.model.geo.addCircleArc(p5, p4, p6)

    # Continue up symmetry axis p6 -> p7
    Mod_line_p6_p7      = gmsh.model.geo.addLine(p6, p7)

    # Fuel2 detour
    Mod_Fuel2_arc1      = gmsh.model.geo.addCircleArc(p7, p8, p9)
    Mod_Fuel2_arc2      = gmsh.model.geo.addCircleArc(p9, p8, p10)

    # Continue up symmetry axis p10 -> p11
    Mod_line_p10_p11    = gmsh.model.geo.addLine(p10, p11)

    # Outer triangle top and hypotenuse
    Mod_line_p11_p12    = gmsh.model.geo.addLine(p11, p12)

    # Hypotenuse: p12 -> p13 (approach HP2), HP2 detour, continue to p17
    Mod_line_p12_p13    = gmsh.model.geo.addLine(p12, p13)
    Mod_HP2_arc1        = gmsh.model.geo.addCircleArc(p13, p14, p15)
    Mod_HP2_arc2        = gmsh.model.geo.addCircleArc(p15, p14, p16)
    Mod_line_p16_p1     = gmsh.model.geo.addLine(p16, p17)

    # Inner HP circle arc p17 -> p2 
    Mod_HP1_arc         = gmsh.model.geo.addCircleArc(p17, p1, p2)

    gmsh.model.geo.addCurveLoop([
        Mod_line_p2_p3, 
        Mod_Fuel1_arc1, 
        Mod_Fuel1_arc2,
        Mod_line_p6_p7, 
        Mod_Fuel2_arc1, 
        Mod_Fuel2_arc2, 
        Mod_line_p10_p11, 
        Mod_line_p11_p12, 
        Mod_line_p12_p13, 
        Mod_HP2_arc1, 
        Mod_HP2_arc2, 
        Mod_line_p16_p1, 
        Mod_HP1_arc
    ], 6)

    # Adding the corresponding surfaces
    # gmsh.model.geo.addPlaneSurface([2], 7)  # HP1
    # gmsh.model.geo.addPlaneSurface([3], 8)  # Fuel1
    # gmsh.model.geo.addPlaneSurface([4], 9)  # Fuel2
    # gmsh.model.geo.addPlaneSurface([7], 12)   # Fuel3
    # gmsh.model.geo.addPlaneSurface([5], 10)  # HP2
    gmsh.model.geo.addPlaneSurface([6, 7], 11) # Moderator with circular hole

    gmsh.model.geo.synchronize()

    gmsh.model.mesh.generate(2)

    if show_mesh == True:
        if '-nopopup' not in sys.argv:
            gmsh.fltk.run()
    
    gmsh.write("hex_mesh.msh")

    gmsh.finalize()


def create_rectangular_mesh(show_mesh=True):
    height = 10
    length = 50

    l_mesh_size = 0.25

    gmsh.initialize()
    gmsh.model.add("rectangular mesh")

    p1  = gmsh.model.geo.addPoint(0.    , 0.    , 0., l_mesh_size)
    p2  = gmsh.model.geo.addPoint(0.    , height, 0., l_mesh_size)
    p3  = gmsh.model.geo.addPoint(length, height, 0., l_mesh_size)
    p4  = gmsh.model.geo.addPoint(length, 0.    , 0., l_mesh_size)

    left   = gmsh.model.geo.addLine(p1, p2)
    top    = gmsh.model.geo.addLine(p2, p3)
    right  = gmsh.model.geo.addLine(p3, p4)
    bottom = gmsh.model.geo.addLine(p4, p1)
    gmsh.model.geo.addCurveLoop([left, top, right, bottom], 1)


    gmsh.model.geo.addPlaneSurface([1], 2) # Moderator with circular hole


    gmsh.model.geo.synchronize()


    gmsh.model.mesh.generate()


    if show_mesh == True:
        if '-nopopup' not in sys.argv:
            gmsh.fltk.run()
    
    gmsh.write("rect_mesh.msh")

    gmsh.finalize()


def create_hexagonal_reactor_mesh(data, show_mesh=True):
    """
    Create a 1/12 symmetry slice of a hexagonal reactor flake.

    This version mimics the OpenMC HexLattice placement more closely.

    Geometry convention:
        - `flake_diameter` is the flat-to-flat diameter of the outer hexagon [m].
        - `pin_cell_pitch` is the OpenMC-like hex-lattice pitch [m].
        - The 1/12 wedge is bounded by:
              x = 0
              y = flake_diameter / 2
              x = tan(pi/6) y

    Expected data keys:
        flake_diameter or flake_diamter
        pin_cell_pitch or l_pitch
        r_HP
        r_fuel_pin
        l_mesh_size optional
    """

    # ------------------------------------------------------------------
    # Inputs
    # ------------------------------------------------------------------
    theta_hex = np.pi / 6.0

    # Supports both correct spelling and your current typo.
    flake_diameter = data.get("flake_diameter", data.get("flake_diamter"))
    pin_pitch = data.get("pin_cell_pitch", data.get("l_pitch"))

    r_HP = data.get("r_HP")
    r_f = data.get("r_fuel_pin")

    if flake_diameter is None:
        raise ValueError("Missing `flake_diameter` or `flake_diamter` in data.")

    if pin_pitch is None:
        raise ValueError("Missing `pin_cell_pitch` or `l_pitch` in data.")

    if r_HP is None:
        raise ValueError("Missing `r_HP` in data.")

    if r_f is None:
        raise ValueError("Missing `r_fuel_pin` in data.")

    l_mesh_size = data.get("l_mesh_size", pin_pitch / 20.0)
    mesh_output_path = data.get("mesh_output_path", "hex_mesh.msh")

    # ------------------------------------------------------------------
    # Outer 1/12 flake boundary
    # ------------------------------------------------------------------
    y_flat = flake_diameter / 2.0
    x_flat = np.tan(theta_hex) * y_flat

    # ------------------------------------------------------------------
    # OpenMC-like hex-lattice centres
    # ------------------------------------------------------------------
    # Hex-lattice basis for OpenMC's default orientation:
    #
    #   a1 = (sqrt(3)/2 p, 1/2 p)
    #   a2 = (0, p)
    #
    # Relevant centres in the 1/12 wedge:
    #
    #   HP0    = (0, 0)
    #   Fuel1 = (0, p)
    #   Fuel2 = (0, 2p)
    #   HP2    = (sqrt(3)/2 p, 3/2 p)
    #   Fuel3 = (sqrt(3)/2 p, 5/2 p)
    #
    x_HP0, y_HP0 = 0.0, 0.0

    x_F1, y_F1 = 0.0, 1.0 * pin_pitch
    x_F2, y_F2 = 0.0, 2.0 * pin_pitch

    x_HP2 = np.sqrt(3.0) / 2.0 * pin_pitch
    y_HP2 = 1.5 * pin_pitch

    x_F3 = np.sqrt(3.0) / 2.0 * pin_pitch
    y_F3 = 2.5 * pin_pitch

    # ------------------------------------------------------------------
    # Geometry fit checks
    # ------------------------------------------------------------------
    def distance_to_slanted_boundary(x, y):
        """
        Distance to the slanted wedge boundary:
            x = tan(theta_hex) y
        """
        a = 1.0
        b = -np.tan(theta_hex)
        c = 0.0
        return abs(a * x + b * y + c) / np.sqrt(a**2 + b**2)

    # Vertical-axis half pins only need to fit upward.
    if y_F2 + r_f >= y_flat:
        raise ValueError(
            "Fuel pin 2 does not fit inside the flake. "
            f"Need y_F2 + r_f = {y_F2 + r_f:.6f} m < y_flat = {y_flat:.6f} m."
        )

    # Interior fuel pin must fit against all wedge boundaries.
    if y_F3 + r_f >= y_flat:
        raise ValueError(
            "Fuel pin 3 does not fit below the top flake boundary. "
            f"Need y_F3 + r_f = {y_F3 + r_f:.6f} m < y_flat = {y_flat:.6f} m."
        )

    if x_F3 - r_f <= 0.0:
        raise ValueError(
            "Fuel pin 3 intersects the vertical symmetry boundary x=0. "
            f"Need x_F3 - r_f = {x_F3 - r_f:.6f} m > 0."
        )

    d_F3_slanted = distance_to_slanted_boundary(x_F3, y_F3)
    if d_F3_slanted <= r_f:
        raise ValueError(
            "Fuel pin 3 intersects the slanted symmetry boundary. "
            f"Distance to boundary = {d_F3_slanted:.6f} m, "
            f"fuel radius = {r_f:.6f} m."
        )

    gmsh.initialize()

    try:
        gmsh.model.add("Cell of hex fuel assembly")

        # ==============================================================
        # Points
        # ==============================================================

        # --------------------------------------------------------------
        # Wedge corners
        # --------------------------------------------------------------
        p1 = gmsh.model.geo.addPoint(
            0.0,
            0.0,
            0.0,
            l_mesh_size,
        )

        p11 = gmsh.model.geo.addPoint(
            0.0,
            y_flat,
            0.0,
            l_mesh_size,
        )

        p12 = gmsh.model.geo.addPoint(
            x_flat,
            y_flat,
            0.0,
            l_mesh_size,
        )

        # --------------------------------------------------------------
        # Central heat pipe, centre HP0 = (0, 0)
        # --------------------------------------------------------------
        p2 = gmsh.model.geo.addPoint(
            x_HP0,
            y_HP0 + r_HP,
            0.0,
            l_mesh_size,
        )

        p17 = gmsh.model.geo.addPoint(
            x_HP0 + r_HP * np.sin(theta_hex),
            y_HP0 + r_HP * np.cos(theta_hex),
            0.0,
            l_mesh_size,
        )

        # --------------------------------------------------------------
        # Fuel pin 1, centre F1 = (0, p)
        # Half-circle because it lies on the vertical symmetry boundary
        # --------------------------------------------------------------
        p3 = gmsh.model.geo.addPoint(
            x_F1,
            y_F1 - r_f,
            0.0,
            l_mesh_size,
        )

        p4 = gmsh.model.geo.addPoint(
            x_F1,
            y_F1,
            0.0,
            l_mesh_size,
        )

        p5 = gmsh.model.geo.addPoint(
            x_F1 + r_f,
            y_F1,
            0.0,
            l_mesh_size,
        )

        p6 = gmsh.model.geo.addPoint(
            x_F1,
            y_F1 + r_f,
            0.0,
            l_mesh_size,
        )

        # --------------------------------------------------------------
        # Fuel pin 2, centre F2 = (0, 2p)
        # Half-circle because it lies on the vertical symmetry boundary
        # --------------------------------------------------------------
        p7 = gmsh.model.geo.addPoint(
            x_F2,
            y_F2 - r_f,
            0.0,
            l_mesh_size,
        )

        p8 = gmsh.model.geo.addPoint(
            x_F2,
            y_F2,
            0.0,
            l_mesh_size,
        )

        p9 = gmsh.model.geo.addPoint(
            x_F2 + r_f,
            y_F2,
            0.0,
            l_mesh_size,
        )

        p10 = gmsh.model.geo.addPoint(
            x_F2,
            y_F2 + r_f,
            0.0,
            l_mesh_size,
        )

        # --------------------------------------------------------------
        # Outer heat pipe, centre HP2 = (sqrt(3)/2 p, 3/2 p)
        # This centre lies on the slanted symmetry boundary
        # --------------------------------------------------------------
        p14 = gmsh.model.geo.addPoint(
            x_HP2,
            y_HP2,
            0.0,
            l_mesh_size,
        )

        p13 = gmsh.model.geo.addPoint(
            x_HP2 + r_HP * np.sin(theta_hex),
            y_HP2 + r_HP * np.cos(theta_hex),
            0.0,
            l_mesh_size,
        )

        p15 = gmsh.model.geo.addPoint(
            x_HP2 - r_HP,
            y_HP2,
            0.0,
            l_mesh_size,
        )

        p16 = gmsh.model.geo.addPoint(
            x_HP2 - r_HP * np.sin(theta_hex),
            y_HP2 - r_HP * np.cos(theta_hex),
            0.0,
            l_mesh_size,
        )

        # --------------------------------------------------------------
        # Fuel pin 3, centre F3 = (sqrt(3)/2 p, 5/2 p)
        # Full circular hole because it is inside the wedge
        # --------------------------------------------------------------
        p18 = gmsh.model.geo.addPoint(
            x_F3,
            y_F3,
            0.0,
            l_mesh_size,
        )

        p19 = gmsh.model.geo.addPoint(
            x_F3 + r_f,
            y_F3,
            0.0,
            l_mesh_size,
        )

        p20 = gmsh.model.geo.addPoint(
            x_F3,
            y_F3 + r_f,
            0.0,
            l_mesh_size,
        )

        p21 = gmsh.model.geo.addPoint(
            x_F3 - r_f,
            y_F3,
            0.0,
            l_mesh_size,
        )

        p22 = gmsh.model.geo.addPoint(
            x_F3,
            y_F3 - r_f,
            0.0,
            l_mesh_size,
        )

        # ==============================================================
        # Optional reference loops for individual components
        # ==============================================================

        # Outer triangular 1/12 slice
        Outer_line1 = gmsh.model.geo.addLine(p1, p11)
        Outer_line2 = gmsh.model.geo.addLine(p11, p12)
        Outer_line3 = gmsh.model.geo.addLine(p12, p1)
        gmsh.model.geo.addCurveLoop(
            [Outer_line1, Outer_line2, Outer_line3],
            1,
        )

        # Central heat pipe reference loop
        HP1_line1 = gmsh.model.geo.addLine(p1, p2)
        HP1_arc1 = gmsh.model.geo.addCircleArc(p2, p1, p17)
        HP1_line2 = gmsh.model.geo.addLine(p17, p1)
        gmsh.model.geo.addCurveLoop(
            [HP1_line1, HP1_arc1, HP1_line2],
            2,
        )

        # Fuel 1 reference loop
        Fuel1_line1 = gmsh.model.geo.addLine(p3, p6)
        Fuel1_arc1 = gmsh.model.geo.addCircleArc(p6, p4, p5)
        Fuel1_arc2 = gmsh.model.geo.addCircleArc(p5, p4, p3)
        gmsh.model.geo.addCurveLoop(
            [Fuel1_line1, Fuel1_arc1, Fuel1_arc2],
            3,
        )

        # Fuel 2 reference loop
        Fuel2_line1 = gmsh.model.geo.addLine(p7, p10)
        Fuel2_arc1 = gmsh.model.geo.addCircleArc(p10, p8, p9)
        Fuel2_arc2 = gmsh.model.geo.addCircleArc(p9, p8, p7)
        gmsh.model.geo.addCurveLoop(
            [Fuel2_line1, Fuel2_arc1, Fuel2_arc2],
            4,
        )

        # Outer heat pipe reference loop
        HP2_line1 = gmsh.model.geo.addLine(p13, p14)
        HP2_line2 = gmsh.model.geo.addLine(p14, p16)
        HP2_arc1 = gmsh.model.geo.addCircleArc(p16, p14, p15)
        HP2_arc2 = gmsh.model.geo.addCircleArc(p15, p14, p13)
        gmsh.model.geo.addCurveLoop(
            [HP2_line1, HP2_line2, HP2_arc1, HP2_arc2],
            5,
        )

        # Fuel 3 full circular hole
        Fuel3_arc1 = gmsh.model.geo.addCircleArc(p19, p18, p20)
        Fuel3_arc2 = gmsh.model.geo.addCircleArc(p20, p18, p21)
        Fuel3_arc3 = gmsh.model.geo.addCircleArc(p21, p18, p22)
        Fuel3_arc4 = gmsh.model.geo.addCircleArc(p22, p18, p19)
        gmsh.model.geo.addCurveLoop(
            [Fuel3_arc1, Fuel3_arc2, Fuel3_arc3, Fuel3_arc4],
            7,
        )

        # ==============================================================
        # Moderator boundary loop
        # ==============================================================

        # Continue up vertical symmetry axis from central HP to Fuel 1
        Mod_line_p2_p3 = gmsh.model.geo.addLine(p2, p3)

        # Detour around Fuel 1
        Mod_Fuel1_arc1 = gmsh.model.geo.addCircleArc(p3, p4, p5)
        Mod_Fuel1_arc2 = gmsh.model.geo.addCircleArc(p5, p4, p6)

        # Continue up vertical axis from Fuel 1 to Fuel 2
        Mod_line_p6_p7 = gmsh.model.geo.addLine(p6, p7)

        # Detour around Fuel 2
        Mod_Fuel2_arc1 = gmsh.model.geo.addCircleArc(p7, p8, p9)
        Mod_Fuel2_arc2 = gmsh.model.geo.addCircleArc(p9, p8, p10)

        # Continue up vertical axis to top flat boundary
        Mod_line_p10_p11 = gmsh.model.geo.addLine(p10, p11)

        # Top flat boundary
        Mod_line_p11_p12 = gmsh.model.geo.addLine(p11, p12)

        # Down slanted symmetry boundary to outer HP
        Mod_line_p12_p13 = gmsh.model.geo.addLine(p12, p13)

        # Detour around outer HP
        Mod_HP2_arc1 = gmsh.model.geo.addCircleArc(p13, p14, p15)
        Mod_HP2_arc2 = gmsh.model.geo.addCircleArc(p15, p14, p16)

        # Continue down slanted symmetry boundary to central HP
        Mod_line_p16_p17 = gmsh.model.geo.addLine(p16, p17)

        # Detour around central HP back to start
        Mod_HP1_arc = gmsh.model.geo.addCircleArc(p17, p1, p2)

        gmsh.model.geo.addCurveLoop(
            [
                Mod_line_p2_p3,
                Mod_Fuel1_arc1,
                Mod_Fuel1_arc2,
                Mod_line_p6_p7,
                Mod_Fuel2_arc1,
                Mod_Fuel2_arc2,
                Mod_line_p10_p11,
                Mod_line_p11_p12,
                Mod_line_p12_p13,
                Mod_HP2_arc1,
                Mod_HP2_arc2,
                Mod_line_p16_p17,
                Mod_HP1_arc,
            ],
            6,
        )

        # Moderator surface with Fuel 3 as an internal hole.
        # Fuel 1, Fuel 2, HP1, and HP2 are already excluded by the
        # outer moderator boundary loop because they lie on symmetry boundaries.
        gmsh.model.geo.addPlaneSurface([6, 7], 11)

        gmsh.model.geo.synchronize()

        gmsh.model.mesh.generate(2)

        if show_mesh:
            if "-nopopup" not in sys.argv:
                gmsh.fltk.run()

        gmsh.write(mesh_output_path)

        print("Created hexagonal reactor mesh")
        print(f"  mesh_output_path  = {mesh_output_path}")
        print(f"  flake_diameter    = {flake_diameter:.6f} m")
        print(f"  pin_cell_pitch    = {pin_pitch:.6f} m")
        print(f"  y_flat            = {y_flat:.6f} m")
        print(f"  x_flat            = {x_flat:.6f} m")
        print()
        print("OpenMC-like centres:")
        print(f"  HP0    = ({x_HP0:.6f}, {y_HP0:.6f}) m")
        print(f"  Fuel1 = ({x_F1:.6f}, {y_F1:.6f}) m")
        print(f"  Fuel2 = ({x_F2:.6f}, {y_F2:.6f}) m")
        print(f"  HP2    = ({x_HP2:.6f}, {y_HP2:.6f}) m")
        print(f"  Fuel3 = ({x_F3:.6f}, {y_F3:.6f}) m")

    finally:
        gmsh.finalize()


if __name__ == "__main__":
    data = {
        "flake_diameter": 0.20,
        "l_pitch": 0.032,
        "r_HP": 0.011,
        "r_fuel_pin": 0.0072,
        "l_mesh_size": 0.0286 / 35,
    }

    create_hexagonal_reactor_mesh(data)

    #create_rectangular_mesh()

    # points = mesh.points[:, :2]                 # x,y coordinates
    # triangles = mesh.cells_dict["triangle"]     # element connectivity

    # mesh = UnstructuredMesh(points, triangles)

    # tri = 16
    # neighbours, surfaces = mesh.get_faces_and_neighbours(tri)
    # print("Neighbours of triangle 0:", neighbours)
    # print("Center of triangle 0:", mesh.get_center_point(tri))
    # print("Boundary triangles:", mesh.get_boundary_triangle())
    # print("Surface normals of triangle 0:\n", mesh.get_triangle_surface_normals(tri))

    # for i, s in enumerate(surfaces):
    #     print(f"\nSurface {i}")
    #     print("  nodes:", s.nodes)
    #     print("  center:", s.center)
    #     print("  length:", s.length)
    #     print("  attached triangles:", s.triangles)
import sys
import numpy as np
import gmsh # type: ignore
import meshio # type: ignore


def create_hexagonal_reactor_mesh(data, show_mesh=True):
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


if __name__ == "__main__":
    data = {
        "l_pitch": 10.,
        "r_HP": 3.,
        "r_fuel_pin": 1.5,
        "l_mesh_size": 2.,
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
import sys
import numpy as np
import gmsh # type: ignore
import meshio # type: ignore


from meshHeatConduction.triangle_mesh import UnstructuredMesh, Surface


l_pitch = 10.
r_HP = 3.
r_f = 1.5

theta_hex = (np.pi / 6)

lc = 0.5

gmsh.initialize()
gmsh.model.add("Cell of hex fuel assembly")

# Adding the points of a slice of a hexagonal fuel assembly 
# Only a 1/12 of the fuel assembly needs modelling due to symmetry.


# p1  = gmsh.model.geo.addPoint(0, 0, 0, lc)
# p2  = gmsh.model.geo.addPoint(0, r_HP, 0, lc)

# p3  = gmsh.model.geo.addPoint(0, l_pitch * 3/2 - r_f, 0, lc)
# p4  = gmsh.model.geo.addPoint(0, l_pitch * 3/2, 0, lc)
# p5  = gmsh.model.geo.addPoint(r_f, l_pitch * 3/2, 0, lc)
# p6  = gmsh.model.geo.addPoint(0, l_pitch * 3/2 + r_f, 0, lc)

# p7  = gmsh.model.geo.addPoint(0, l_pitch * 5/2 - r_f, 0, lc)
# p8  = gmsh.model.geo.addPoint(0, l_pitch * 5/2, 0, lc)
# p9  = gmsh.model.geo.addPoint(r_f, l_pitch * 5/2, 0, lc)
# p10 = gmsh.model.geo.addPoint(0, l_pitch * 5/2 + r_f, 0, lc)

# p11 = gmsh.model.geo.addPoint(0, l_pitch * 4, 0, lc)
# p12 = gmsh.model.geo.addPoint(np.tan(theta_hex) * 4 * l_pitch, 4 * l_pitch, 0, lc)

# p13 = gmsh.model.geo.addPoint(np.tan(theta_hex) * 2 * l_pitch + r_HP * np.sin(theta_hex), 2 * l_pitch + r_HP * np.cos(theta_hex), 0, lc)
# p14 = gmsh.model.geo.addPoint(np.tan(theta_hex) * 2 * l_pitch, 2 * l_pitch, 0, lc)
# p15 = gmsh.model.geo.addPoint(np.tan(theta_hex) * 2 * l_pitch - r_HP, 2 * l_pitch, 0, lc)
# p16 = gmsh.model.geo.addPoint(np.tan(theta_hex) * 2 * l_pitch - r_HP * np.sin(theta_hex), 2 * l_pitch - r_HP * np.cos(theta_hex), 0, lc)

# p17 = gmsh.model.geo.addPoint(r_HP * np.sin(theta_hex), r_HP * np.cos(theta_hex), 0, lc)

p1  = gmsh.model.geo.addPoint(0,                                                          0,                          0, lc)
p2  = gmsh.model.geo.addPoint(0,                                                          r_HP,                       0, lc)

p3  = gmsh.model.geo.addPoint(0,                                                          l_pitch * 3./2. - r_f,        0, lc)
p4  = gmsh.model.geo.addPoint(0,                                                          l_pitch * 3./2.,              0, lc)
p5  = gmsh.model.geo.addPoint(r_f,                                                        l_pitch * 3./2.,              0, lc)
p6  = gmsh.model.geo.addPoint(0,                                                          l_pitch * 3./2. + r_f,        0, lc)

p7  = gmsh.model.geo.addPoint(0,                                                          l_pitch * 5./2. - r_f,        0, lc)
p8  = gmsh.model.geo.addPoint(0,                                                          l_pitch * 5./2.,              0, lc)
p9  = gmsh.model.geo.addPoint(r_f,                                                        l_pitch * 5./2.,              0, lc)
p10 = gmsh.model.geo.addPoint(0,                                                          l_pitch * 5./2. + r_f,        0, lc)

p11 = gmsh.model.geo.addPoint(0,                                                          l_pitch * 4,                0, lc)
p12 = gmsh.model.geo.addPoint(np.tan(theta_hex) * 4 * l_pitch,                            4 * l_pitch ,                0, lc)

p13 = gmsh.model.geo.addPoint(np.tan(theta_hex) * 2 * l_pitch + r_HP * np.sin(theta_hex), 2 * l_pitch + r_HP * np.cos(theta_hex), 0, lc)
p14 = gmsh.model.geo.addPoint(np.tan(theta_hex) * 2 * l_pitch,                            2 * l_pitch,                            0, lc)
p15 = gmsh.model.geo.addPoint(np.tan(theta_hex) * 2 * l_pitch - r_HP,                     2 * l_pitch,                            0, lc)
p16 = gmsh.model.geo.addPoint(np.tan(theta_hex) * 2 * l_pitch - r_HP * np.sin(theta_hex), 2 * l_pitch - r_HP * np.cos(theta_hex), 0, lc)

p17 = gmsh.model.geo.addPoint(r_HP * np.sin(theta_hex),                                  r_HP * np.cos(theta_hex),   0, lc)

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
# gmsh.model.geo.addPlaneSurface([5], 10)  # HP2
gmsh.model.geo.addPlaneSurface([6], 11) # Moderator

gmsh.model.geo.synchronize()

gmsh.model.mesh.generate(2)
if '-nopopup' not in sys.argv:
    gmsh.fltk.run()
gmsh.write("mesh.msh")
gmsh.finalize()

mesh = meshio.read("mesh.msh")

points = mesh.points[:, :2]                 # x,y coordinates
triangles = mesh.cells_dict["triangle"]     # element connectivity

mesh = UnstructuredMesh(points, triangles)

tri = 16
neighbours, surfaces = mesh.get_faces_and_neighbours(tri)
print("Neighbours of triangle 0:", neighbours)
print("Center of triangle 0:", mesh.get_center_point(tri))
print("Boundary triangles:", mesh.get_boundary_triangle())
print("Surface normals of triangle 0:\n", mesh.get_triangle_surface_normals(tri))

for i, s in enumerate(surfaces):
    print(f"\nSurface {i}")
    print("  nodes:", s.nodes)
    print("  center:", s.center)
    print("  length:", s.length)
    print("  attached triangles:", s.triangles)
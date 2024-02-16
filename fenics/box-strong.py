from fenics import *

# Optimization options for the form compiler
parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True

# Create mesh and define function space
mesh = BoxMesh(Point(0, 0, 0), Point(100, 100, 100), 16, 16, 16)
V = VectorFunctionSpace(mesh, "P", 1)

# Define Dirichlet boundary condition
def boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, Constant((0, 0, 0)), boundary)

# Define Neumann boundary condition for traction
Traction = Constant((0, 0, -0.018))
ds = Measure("ds")(domain=mesh, subdomain_data=mesh)
bc_traction = dot(Traction, Constant((0, 0, 1))) * ds(1)

# Define displacement
u = Function(V, name="Displacement")

# Define kinematics
d = len(u)
I = Identity(d)
F = variable(I + grad(u))
C = F.T * F
J = det(F)

# Elasticity parameters
E, nu = 0.05, 0.385
mu = Constant(E / (2 * (1 + nu)))
lmbda = Constant(E * nu / ((1 + nu) * (1 - 2 * nu)))

# Free energy function
psi = (mu / 2) * (tr(C) - 3) - mu * ln(J) + (lmbda / 2) * ln(J) ** 2

# First Piola-Kirchhoff stress
P = diff(psi, F)

# Define equilibrium equation
residual = div(P)

# Solve equations of motion with Neumann boundary condition
solve(residual == 0, u, bcs=[bc], bcs_constrained=[bc_traction])

# Save solution
file = File("fenics/strong/u.pvd")
file << u

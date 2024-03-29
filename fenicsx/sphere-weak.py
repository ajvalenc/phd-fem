from mpi4py import MPI
from dolfinx import log, fem, mesh
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.fem import locate_dofs_topological, locate_dofs_geometrical
from dolfinx.io import XDMFFile
from dolfinx.mesh import locate_entities_boundary

import pyvista as pv
import dolfinx.plot

import numpy as np
import ufl

# Load mesh from file
file = "/home/ajvalenc/Datasets/romado/models/objects/ball.xdmf"
with XDMFFile(MPI.COMM_WORLD, file, "r") as xdmf:
    domain = xdmf.read_mesh(name="Grid")

# Define function space
element = ufl.VectorElement("P", domain.ufl_cell(), 1, dim=3)
V = fem.FunctionSpace(domain, element)

# Find maximum and minimum values in the z dimension
max_z = np.max(domain.geometry.x[:, 2])
min_z = np.min(domain.geometry.x[:, 2])
tol_top = 2e-2
tol_bottom = 5.9e-4 # 1e-3
z_threshold = -0.049

# Define Dirichlet boundary
def bottom(x):
    return np.abs(x[2] - min_z) < tol_bottom

def top(x):
    return np.abs(x[2] - max_z) < tol_top

fdim = domain.topology.dim - 1
bottom_facets = locate_entities_boundary(domain, fdim, bottom)
top_facets = locate_entities_boundary(domain, fdim, top)
marked_facets = np.hstack([bottom_facets, top_facets])
marked_values = np.hstack([np.full_like(bottom_facets, 1), np.full_like(top_facets, 2)])
sorted_facets = np.argsort(marked_facets)
facet_tag = mesh.meshtags(domain, fdim, marked_facets[sorted_facets], marked_values[sorted_facets])

# Define boundary conditions
u_bc = np.array((0,) * domain.geometry.dim, dtype=np.float64)
bottom_dofs = locate_dofs_topological(V, facet_tag.dim, facet_tag.find(1))
#bcs = [fem.dirichletbc(u_bc, bottom_dofs, V)]

def update_bcs():
    below_threshold_dofs = locate_dofs_geometrical(V, lambda x: x[2] < z_threshold)
    u_bc_below_threshold = np.array((0, 0, 0), dtype=np.float64)
    return [fem.dirichletbc(u_bc, bottom_dofs, V), fem.dirichletbc(u_bc_below_threshold, below_threshold_dofs, V)], below_threshold_dofs

bcs, old_dofs = update_bcs()

# Define functions
v = ufl.TestFunction(V)
u = fem.Function(V)

B = fem.Constant(domain, np.zeros(3, dtype=np.float64))
#T = fem.Constant(domain, np.array([0.0, 0.0, 0.037], dtype=np.float64))
#T = fem.Constant(domain, np.array([0.0, 0.0, -0.019], dtype=np.float64))
#T = fem.Constant(domain, np.array([0.0, 0.0, 3.7e4], dtype=np.float64))
T = fem.Constant(domain, np.array([0.0, 0.0, -3e3], dtype=np.float64))

# Kinematics
d = len(u)
I = ufl.variable(ufl.Identity(d))
F = ufl.variable(I + ufl.grad(u))
C = ufl.variable(F.T * F)

# Invariants of deformation tensors
Ic = ufl.variable(ufl.tr(C))
J = ufl.variable(ufl.det(F))

# Elasticity parameters (similar to the FEniCS implementation)
E = 5e4
nu = 0.385
mu = fem.Constant(domain, E / (2 * (1 + nu)))
lmbda = fem.Constant(domain, E * nu / ((1 + nu) * (1 - 2 * nu)))

# Stored strain energy density (compressible neo-Hookean model)
psi = (mu / 2) * (Ic - 3) - mu * ufl.ln(J) + (lmbda / 2) * (ufl.ln(J))**2
#psi = (mu / 2) * (Ic-3) - mu * (J - 1) + (lmbda / 2) * (J - 1)**2 # snh

# Stress
P = ufl.diff(psi, F)

# Residual energy
dx = ufl.Measure("dx", domain=domain)
ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tag)
L = ufl.inner(ufl.grad(v), P) * dx - ufl.inner(v, B) * dx - ufl.inner(v, T) * ds(2)

# Compute Jacobian of L
J = ufl.derivative(L, u, ufl.TrialFunction(V))

# Solve variational problem
problem = NonlinearProblem(L, u, bcs, J)
solver = NewtonSolver(domain.comm, problem)
solver.atol = 1e-6
solver.rtol = 1e-6
solver.convergence_criterion = "residual"
solver.maximum_iterations = 1

log.set_log_level(log.LogLevel.INFO)

# Solve the problme until solution has converged and boundary conditions have not changed
converged = False
while not converged:
    num_its, converged = solver.solve(u)
    if num_its >= solver.maximum_iterations:
        print("Maximum number of iterations reached without convergence.")
        break  # Exit the loop if maximum iterations reached without convergence

    new_bcs, new_dofs = update_bcs()
    print("new dofs", new_dofs)
    if not np.array_equal(new_dofs, old_dofs):
        bcs = new_bcs
        old_dofs = new_dofs
        solver.set_bounds(bcs)
        converged = False

# Save solution to file
with XDMFFile(MPI.COMM_WORLD, "fenicsx/sphere.xdmf", "w") as file:
    file.write_mesh(domain)
    file.write_function(u, 0)
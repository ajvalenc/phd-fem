from mpi4py import MPI
from dolfinx import log, fem, mesh
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.fem import locate_dofs_topological
from dolfinx.io import XDMFFile
from dolfinx.mesh import locate_entities_boundary
import numpy as np
import ufl

# Create mesh and define function space
L = 100.0  # Size of the box in the x-direction
H = 100.0  # Size of the box in the y-direction
W = 100.0  # Size of the box in the z-direction
n = 16  # Number of elements in each direction
domain = mesh.create_box(MPI.COMM_WORLD, [np.array([0, 0, 0]), np.array([L, H, W])], [n, n, n], mesh.CellType.tetrahedron)
element = ufl.VectorElement("P", domain.ufl_cell(), 1, dim=3)
V = fem.FunctionSpace(domain, element)
#V = fem.VectorFunctionSpace(domain, ("Lagrange", 1))

# Define Dirichlet boundary
def bottom(x):
    return np.isclose(x[2], 0.0)

def top(x):
    return np.isclose(x[2], W)

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
bcs = [fem.dirichletbc(u_bc, bottom_dofs, V)]

# Define functions
v = ufl.TestFunction(V)
u = fem.Function(V)

B = fem.Constant(domain, np.zeros(3, dtype=np.float64))
#T = fem.Constant(domain, np.array([0.0, 0.0, 0.037], dtype=np.float64))
T = fem.Constant(domain, np.array([0.0, 0.0, -0.019], dtype=np.float64))

# Kinematics
d = len(u)
I = ufl.variable(ufl.Identity(d))
F = ufl.variable(I + ufl.grad(u))
C = ufl.variable(F.T * F)

# Invariants of deformation tensors
Ic = ufl.variable(ufl.tr(C))
J = ufl.variable(ufl.det(F))

# Elasticity parameters (similar to the FEniCS implementation)
E = 0.05
nu = 0.385
mu = fem.Constant(domain, E / (2 * (1 + nu)))
lmbda = fem.Constant(domain, E * nu / ((1 + nu) * (1 - 2 * nu))) + mu

# Stored strain energy density (compressible neo-Hookean model)
#psi = (mu / 2) * (Ic - 3) - mu * ufl.ln(J) + (lmbda / 2) * (ufl.ln(J))**2
psi = (mu / 2) * (Ic-3) - mu * (J - 1) + (lmbda / 2) * (J - 1)**2 # snh

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
solver.convergence_criterion = "incremental"
solver.maximum_iterations = 200

log.set_log_level(log.LogLevel.INFO)

num_its, converged = solver.solve(u)
assert converged

# Save solution to file
with XDMFFile(MPI.COMM_WORLD, "fenicsx/u.xdmf", "w") as file:
    file.write_mesh(domain)
    file.write_function(u, 0)
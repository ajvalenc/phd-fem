
from fenics import *

# Optimization options for the form compiler
parameters["form_compiler"]["cpp_optimize"] = True
ffc_options = {"optimize": True, \
               "eliminate_zeros": True, \
               "precompute_basis_const": True, \
               "precompute_ip_const": True}

# Create mesh and define function space
mesh = BoxMesh(Point(0, 0, 0), Point(100, 100, 100), 16, 16, 16)
V = VectorFunctionSpace(mesh, "CG", 1) # CG = Continuous Galerkin elements or Lagrange elements of degree 1

# Define Dirichlet boundary
bottom = CompiledSubDomain("near(x[2], value) && on_boundary", value = 0.0)
c = Expression(("0.0", "0.0", "0.0"), element=V.ufl_element())
bottom_bcs = DirichletBC(V, c, bottom)
bcs = [bottom_bcs]

# Define functions
du = TrialFunction(V)            # Incremental displacement
v  = TestFunction(V)             # Test function
u  = Function(V)                 # Displacement from previous iteration
B  = Constant((0.0, 0.0, 0.0))  # Body force per unit volume
T  = Constant((0.0,  0.0, 0.005))  # Traction force on the boundary

# Kinematics
d = u.geometric_dimension()
I = Identity(d)             # Identity tensor
F = I + grad(u)             # Deformation gradient
C = F.T*F                   # Right Cauchy-Green tensor

# Invariants of deformation tensors
Ic = tr(C)
J  = det(F)

# Elasticity parameters
E, nu = 0.05, 0.385
mu = Constant(E/(2*(1 + nu)))
lmbda  = Constant(E*nu/((1 + nu)*(1 - 2*nu)))

# Stored strain energy density (compressible neo-Hookean model)
psi = (mu/2)*(Ic-3) - mu*(J-1) + (lmbda/2)*(J-1)**2

# Total potential energy
Pi = psi*dx - dot(B, u)*dx - dot(T, u)*ds

# Compute first variation of Pi (directional derivative about u in the direction of v)
F = derivative(Pi, u, v)

# Compute Jacobian of F
J = derivative(F, u, du)

# Solve variational problem
solve(F==0, u, bcs, J=J,
      form_compiler_parameters=ffc_options)

# Save solution in VTK format
file = File("fenics/potential/u.pvd")
file << u
from fenics import *

# Optimization options for the form compiler
parameters["form_compiler"]["optimize"]     = True
parameters["form_compiler"]["cpp_optimize"] = True

# Load mesh from file
file = "/home/ajvalenc/Datasets/romado/models/objects/ball.xml"
mesh = Mesh(file)

# Create mesh and define function space
V = VectorFunctionSpace(mesh, "P", 1) # P = Piecewise linear elements

# Find maximum and minimum values in the z dimension
max_z = mesh.coordinates()[:, 2].max()
min_z = mesh.coordinates()[:, 2].min()
tol = 2e-2

# Create subdomains
class on_top(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and abs(x[2] - max_z) < tol
class on_bottom(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and abs(x[2] - min_z) < tol

# Mark boundary subdomains
mf = MeshFunction("size_t", mesh, 2) # facets in 2d (triangle)
mf.set_all(0) # Initialize to zero
top_boundary = on_top()
bottom_boundary = on_bottom()
top_boundary.mark(mf, 1)
bottom_boundary.mark(mf, 2)
ds = Measure("ds")[mf] # NOTE: masking [mf] may not be necessary as all facets are marked

# Define Dirichlet boundary
bc = DirichletBC(V, Constant((0, 0, 0)), mf, 2)
bcs = [bc]

# Define variational problem
du = TrialFunction(V)
v = TestFunction(V)
u = Function(V, name="Displacement")
#u.interpolate(Constant((0, 0, 0)))
B = Constant((0.0, 0.0, 0.0))
T = Constant((0.0, 0.0, -0.018)) #-0.018

# Kinematics
d = len(u)
I = Identity(d)
F = variable(I + grad(u))
C = F.T*F

# Invariants of deformation tensors
Ic = tr(C)
J = det(F)

# Elasticity parameters
E, nu = 0.05, 0.385
mu = Constant(E/(2*(1 + nu)))
lmbda = Constant(E*nu/((1 + nu)*(1- 2*nu)))

# Free Energy Function
#psi= (mu/2)*(Ic-3) - mu*ln(J) + (lmbda/2)*(J-1)**2
psi = (mu/2)*(Ic-3) - mu*ln(J) + (lmbda/2)*ln(J)**2 #bw08
#psi = (mu/2)*(Ic-3) - mu*(J-1) + (lmbda/2)*(J-1)**2 # snh

# First Piola-Kirchhoff stress
P = diff(psi, F)

# Residual energy
L = inner(P, grad(v))*dx + dot(B, v)*dx - dot(T, v)*ds(1)

# Compute Jacobian of L
J = derivative(L, u, du)

# Save solution in VTK format
solve(L==0, u, bcs, J=J,
      solver_parameters={"newton_solver":{"linear_solver":"lu",
                                          "relative_tolerance": 1e-6,
                                          #"preconditioner":"ilu",
                                          "convergence_criterion":"incremental",}})
#File("medium/u0.pvd", "compressed") << u
file = File("weak/u.pvd") 
file << u
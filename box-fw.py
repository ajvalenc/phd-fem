from fenics import *

# Optimization options for the form compiler
parameters["form_compiler"]["optimize"]     = True
parameters["form_compiler"]["cpp_optimize"] = True

# Create mesh and define function space
#mesh = BoxMesh(Point(0, 0, 0), Point(1, 1, 5), 16, 16, 80)
mesh = BoxMesh(Point(0, 0, 0), Point(100, 100, 100), 16, 16, 16)
V = VectorFunctionSpace(mesh, "P", 1) # P = Piecewise linear elements

File("weak/mesh.xml") << mesh

# Mark boundary subdomians
mf = MeshFunction("size_t", mesh, 2)
mf.set_all(0)
class top_boundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[2], 100)
right_boundary = top_boundary()
right_boundary.mark(mf, 1)
ds = Measure("ds")[mf]

# Define Dirichlet boundary
bottom = CompiledSubDomain("near(x[2], value) && on_boundary", value = 0.0)
c = Expression(("0.0", "0.0", "0.0"), element=V.ufl_element())
bottom_bcs = DirichletBC(V, c, bottom)
bcs = [bottom_bcs]

# Define functions
du = TrialFunction(V)
v = TestFunction(V)
u = Function(V)
u.interpolate(Constant((0, 0, 0)))
B = Constant((0.0, 0.0, 0.0))
#T = Constant((0.0, 0.0, 0.037))
T = Constant((0.0, 0.0, -0.019)) #-0.018

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
file = File("fenics/weak/u.pvd") 
file << u
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
tol = 9e-3#6e-3

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
bottom_boundary = on_bottom()
bottom_boundary.mark(mf, 2)

# Define Dirichlet boundary
bc = DirichletBC(V, Constant((0, 0, 0)), mf, 2)
bcs = [bc]

# Define variational problem
du = TrialFunction(V)
v = TestFunction(V)
u = Function(V, name="Displacement")
#u.interpolate(Constant((0, 0, 0)))
B = Constant((0.0, 0.0, 0.0))
#T = Constant((0.0, 0.0, -0.018)) #-0.018

# Kinematics
d = len(u)
I = Identity(d)
F = variable(I + grad(u))
C = F.T*F

# Invariants of deformation tensors
Ic = tr(C)
J = det(F)

# Elasticity parameters
E, nu = 5e4, 0.385
mu = Constant(E/(2*(1 + nu)))
lmbda = Constant(E*nu/((1 + nu)*(1- 2*nu)))

# Free Energy Function
#psi= (mu/2)*(Ic-3) - mu*ln(J) + (lmbda/2)*(J-1)**2
#psi = (mu/2)*(Ic-3) - mu*ln(J) + (lmbda/2)*ln(J)**2 #bw08
lmbda += mu
psi = (mu/2)*(Ic-3) - mu*(J-1) + (lmbda/2)*(J-1)**2 # snh

# First Piola-Kirchhoff stress
P = diff(psi, F)

# Residual energy
L = inner(P, grad(v))*dx + dot(B, v)*dx 
#L =- dot(T, v)*ds(1)

# Define nodal traction values and corresponding nodes
node_indices = [ 50,  76,  79,  82,  89,  93, 106, 112, 118, 119, 128, 129, 130, 145,
        153, 156, 158, 159, 166, 171, 186, 192, 198, 507, 518, 519, 528, 540] 
traction_values = [[ 1.230880981e+03,  1.741688232e+03, -2.952074890e+02],
        [ 1.562018311e+03,  4.003358398e+03,  1.395641846e+03],
        [ 1.695908203e+04,  4.757515234e+04,  5.555067383e+03],
        [ 2.783799072e+03, -7.734872070e+03, -9.177367554e+02],
        [ 7.521973633e+03,  2.375977148e+04, -2.452510010e+03],
        [ 2.021377441e+03,  9.137484375e+03, -2.764421631e+03],
        [ 3.113195068e+03, -1.753252734e+04, -1.007688843e+03],
        [ 5.206421387e+03, -4.010328125e+04,  5.473391113e+03],
        [ 1.134407578e+02,  2.578741211e+03, -3.654305649e+01],
        [-7.103817749e+01,  1.924553589e+03,  7.160610962e+02],
        [-1.640285492e+01, -2.114100781e+04,  7.045223633e+03],
        [ 1.204677368e+03, -2.379060352e+04, -8.314186523e+03],
        [-1.978640869e+03, -6.302108203e+04,  2.780467285e+03],
        [-1.154692871e+03, -5.890072266e+03, -1.584917725e+03],
        [-3.721277344e+03,  1.697386328e+04, -5.130552734e+03],
        [-1.789478760e+03, -7.041416992e+03,  2.470753174e+03],
        [-4.754909180e+03, -1.510268457e+04,  1.549460449e+03],
        [-9.887507935e+02,  2.952400146e+03, -3.184899902e+02],
        [-5.185876953e+03,  1.532769336e+04,  1.336391846e+03],
        [-3.661842811e-07,  9.318505363e-07,  3.246916265e-07],
        [-4.050482178e+03,  7.442309082e+03, -2.537936249e+02],
        [-4.926197266e+03,  8.251315430e+03,  1.329463501e+03],
        [-1.849211182e+03,  2.831256592e+03, -6.873321533e+02],
        [ 4.112359863e+03,  6.996581543e+03,  1.336546143e+03],
        [ 6.927833252e+02,  3.914920654e+03,  2.319002075e+02],
        [-4.079049377e+02, -2.406661377e+03, -2.065279541e+02],
        [-4.514149170e+02, -1.526393652e+04, -2.865106934e+03],
        [ 6.172259277e+03,  1.337629199e+04, -3.844353516e+03]]

# print node position
for node_index, traction_value in zip(node_indices, traction_values):
    node = mesh.coordinates()[node_index]
    print("Node", node_index, "at", node, "has traction", traction_value)

# Interpolate nodal traction values to facet markers
facet_markers = MeshFunction("size_t", mesh, 2)
for node_index, traction_value in zip(node_indices, traction_values):
    node = Point(mesh.coordinates()[node_index])
    cell_index = mesh.bounding_box_tree().compute_first_entity_collision(node)
    cell = Cell(mesh, cell_index)
    # Mark all boundary facets of the cell
    if cell_index != -1:
        for facet in facets(cell):
            if facet.exterior():
                print("Boundary facet found", facet)
                facet_markers[facet] = node_index  # Mark all facets for traction

# Define residual energy
ds = Measure("ds")(subdomain_data=facet_markers)
T = [Constant(traction_value) for traction_value in traction_values]
#L_traction = sum(dot(T[i], v)*ds(1) for i in range(len(traction_values)))
L_traction = sum(dot(T[i], v)*ds(node_indices[i]) for i in range(len(node_indices)))

L -= L_traction

# Compute Jacobian of L
J = derivative(L, u, du)

# Save solution in VTK format
solve(L==0, u, bcs, J=J,
      solver_parameters={"newton_solver":{"linear_solver":"lu",
                                          "relative_tolerance": 1e-6,
                                          #"preconditioner":"ilu",
                                          "convergence_criterion":"incremental",}})
#File("medium/u0.pvd", "compressed") << u
file = File("weak/u-{}.pvd".format(tol)) 
file << u
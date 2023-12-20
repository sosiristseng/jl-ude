#===
# Solving PDE with ModelingToolkit and NeuralPDE

Solving Poisson PDE Systems

$$
\partial^{2}_{x}u(x,y) + \partial^{2}_{y}u(x,y) = -\sin (\pi x) \sin (\pi y)
$$

with boundary conditions

$$
\begin{align}
u(0, y) &= 0 \\
u(1, y) &= 0 \\
u(x, 0) &= 0 \\
u(x, 1) &= 0 \\
\end{align}
$$
===#

using NeuralPDE
using Lux
using Plots
using Optimization
using OptimizationOptimJL
using ModelingToolkit
import ModelingToolkit: Interval

#---
@variables x y u(..)
Dxx = Differential(x)^2
Dyy = Differential(y)^2

# 2D PDE equations
eq  = Dxx(u(x, y)) + Dyy(u(x, y)) ~ -sinpi(x) * sinpi(y)

# Boundary conditions
bcs = [
    u(0, y) ~ 0.0, u(1, y) ~ 0.0,
    u(x, 0) ~ 0.0, u(x, 1) ~ 0.0
]

# Space and time domains
domains = [
    x ∈ Interval(0.0, 1.0),
    y ∈ Interval(0.0, 1.0)
]

# Build a Neural Network for the NPDE solver.
dim = 2
chain = Lux.Chain(Dense(dim, 16, Lux.σ), Dense(16, 16, Lux.σ), Dense(16, 1))

# Discretization method uses `PhysicsInformedNN()` (PINN).
dx = 0.05
discretization = PhysicsInformedNN(chain, GridTraining(dx))

# Next we build our PDE system and discretize it. Because this system is time-invariant, the corresponding problem is an `OptimizationProblem`.
@named pde_system = PDESystem(eq, bcs, domains, [x, y], [u(x, y)])
prob = discretize(pde_system, discretization)

# The callback function records the loss value.
opt = OptimizationOptimJL.BFGS()

larr = Vector{Float64}()

callback = function (p,l)
    push!(larr, l)
    return false
end

# Solve the problem

res = Optimization.solve(prob, opt, callback = callback, maxiters=1500)

#---
larr[end]

#---
plot(larr, xlabel="# Iterations", label="Loss", yscale=:log10, xlims=(0, 1500))

# Plot the predicted solution of the PDE and compare it with the analytical solution to see the relative error.

xs,ys = [infimum(d.domain):dx/10:supremum(d.domain) for d in domains]

analytic_sol_func(x,y) = (sinpi(x)*sinpi(y))/(2pi^2)

phi = discretization.phi
u_predict = reshape([first(phi([x, y], res.u)) for x in xs for y in ys], (length(xs), length(ys)))
u_real = reshape([analytic_sol_func(x, y) for x in xs for y in ys], (length(xs), length(ys)))
diff_u = abs.(u_predict .- u_real)

p1 = plot(xs, ys, u_real, linetype=:contourf, title = "analytic")
p2 = plot(xs, ys, u_predict, linetype=:contourf, title = "predict")
p3 = plot(xs, ys, diff_u, linetype=:contourf, title = "error")
plot(p1, p2, p3)

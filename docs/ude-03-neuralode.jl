#===
# Solving ODEs with NeuralPDE.jl

Solving ODEs with Physics-Informed Neural Networks: https://docs.sciml.ai/NeuralPDE/stable/tutorials/ode/
===#
using NeuralPDE
using Lux
using OptimizationOptimisers
using OrdinaryDiffEq
using LinearAlgebra
using Random
using Plots
rng = Random.default_rng()
Random.seed!(rng, 0)

# ## Solve ODEs
# The true function: $u^{\prime} = cos(2 \pi t)$
model(u, p, t) = cospi(2t)

# Prepare data
tspan = (0.0, 1.0)
u0 = 0.0
prob = ODEProblem(model, u0, tspan)

# Construct a neural network to solve the problem.
chain = Lux.Chain(Lux.Dense(1, 5, σ), Lux.Dense(5, 1))
ps, st = Lux.setup(rng, chain) |> Lux.f64

# Solve the ODE with `NeuralPDE.NNODE()`.
optimizer = OptimizationOptimisers.Adam(0.1)
alg = NeuralPDE.NNODE(chain, optimizer, init_params = ps)
# Turn `verbose=true` to see the fitting process
@time sol = solve(prob, alg, maxiters = 2000, saveat = 0.01, verbose = false)

# Comparing to the regular solver
sol2 = solve(prob, Tsit5(), saveat=sol.t)

plot(sol2, label = "Tsit5")
plot!(sol.t, sol.u, label = "NNODE")

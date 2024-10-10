#===
# Solving ODEs with NeuralPDE.jl

From https://docs.sciml.ai/NeuralPDE/stable/tutorials/ode/

For example, to solve the ODE

$$
u^{\prime} = cos(2 \pi t)
$$
===#
using NeuralPDE
using Lux
using OptimizationOptimisers
using OrdinaryDiffEq
using LinearAlgebra
using Random
using Plots
rng = Random.default_rng()

# The true function
model(u, p, t) = cospi(2t)

tspan = (0.0f0, 1.0f0)
u0 = 0.0f0
prob = ODEProblem(model, u0, tspan)

# Construct a neural network to solve the problem.
chain = Lux.Chain(Lux.Dense(1, 5, σ), Lux.Dense(5, 1))
ps, st = Lux.setup(rng, chain)

# Solve the ODE as in `DifferentialEquations.jl`, just change the solver algorithm to `NeuralPDE.NNODE()`.
optimizer = OptimizationOptimisers.Adam(0.1)
alg = NeuralPDE.NNODE(chain, optimizer, init_params = ps)

#---
sol = solve(prob, alg, maxiters=2000, saveat = 0.01f0)

# Comparing to the regular solver
sol2 = solve(prob, Tsit5(), saveat=sol.t)

plot(sol2, label = "Truth")
plot!(sol.t, sol.u, label = "Predicted")

#===

# Solving ODEs with NeuralPDE.jl

From https://neuralpde.sciml.ai/dev/tutorials/ode/

For example, solving the ODE

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
rng = Random.default_rng()

# True function.
model(u, p, t) = cospi(2t)

tspan = (0.0f0, 1.0f0)
u0 = 0.0f0
prob = ODEProblem(model, u0, tspan)

# Construct a neural network to solve the problem.
chain = Lux.Chain(Lux.Dense(1, 5, tanh), Lux.Dense(5, 1))
p, st = Lux.setup(rng, chain)

# We solve the ODE as before, just change the solver algorithm to `NeuralPDE.NNODE()`.
optimizer = OptimizationOptimisers.Adam(0.1)
alg = NeuralPDE.NNODE(chain, optimizer)
sol = solve(prob, alg, verbose=true, abstol=1f-6, maxiters=300)

# Comparing to the regular solver
sol2 = solve(prob, Tsit5(), abstol=1f-6, saveat=sol.t)

using LinearAlgebra
norm(sol.u .- sol2.u, Inf)

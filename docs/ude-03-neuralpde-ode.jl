#==
# Solving ODEs with NeuralPDE.jl

From https://neuralpde.sciml.ai/dev/tutorials/ode/

For example, solving the ODE

$$
u^{\prime} = cos(2 \pi t)
$$
==#

using NeuralPDE
using Flux
using OptimizationOptimisers
using OrdinaryDiffEq

#---

model(u, p, t) = cospi(2t)

tspan = (0.0f0, 1.0f0)
u0 = 0.0f0
prob = ODEProblem(model, u0, tspan)

# We need to construct a neural network to solve the problem.

chain = Flux.Chain(Dense(1, 5, Flux.σ), Dense(5, 1))
optimizer = OptimizationOptimisers.Adam(0.1)
alg = NeuralPDE.NNODE(chain, optimizer)

# And we solve the ODE as before, just replace the solver algorithm to `NeuralPDE` with common ones (e.g. `Tsit5()`).

sol = solve(prob, alg, verbose=true, abstol=1f-6, maxiters=200)

# Comparing to the regular solver
sol2 = solve(prob, Tsit5(), abstol=1f-6, saveat=sol.t)

# Showing the error
using LinearAlgebra
norm(sol.u .- sol2.u, Inf)

# ## Runtime information

import Pkg
Pkg.status()

#---

import InteractiveUtils
InteractiveUtils.versioninfo()

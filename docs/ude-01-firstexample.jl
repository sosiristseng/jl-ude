#===
# First N-ODE example

A neural ODE is an ODE where a neural network defines its derivative function. $\dot{u} = NN(u)$

Docs: https://docs.sciml.ai/DiffEqFlux/dev/examples/neural_ode/
===#

using Lux
using ComponentArrays
using DiffEqFlux
using OrdinaryDiffEq
using Optimization
using OptimizationOptimJL
using Random
using Plots

rng = Random.default_rng()

#---

u0 = Float32[2.0; 0.0]
datasize = 30
tspan = (0.0f0, 1.5f0)
tsteps = range(tspan[begin], tspan[end], length = datasize)

#---

true_A = Float32[-0.1 2.0; -2.0 -0.1]

function trueODEfunc!(du, u, p, t)
    du .= ((u.^3)'true_A)'
end

#---

prob_trueode = ODEProblem(trueODEfunc!, u0, tspan)
ode_data = Array(solve(prob_trueode, Tsit5(), saveat = tsteps))

#---

nodeFunc = Lux.Chain(
    ActivationFunction(x -> x.^3),
    Lux.Dense(2, 50, tanh),
    Lux.Dense(50, 2)
)

p, st = Lux.setup(rng, nodeFunc)

# Parameters for neural network
p

# Use NeuroODE to construct an DE problem object
prob_node = NeuralODE(nodeFunc, tspan, Tsit5(), saveat = tsteps)

function predict_neuralode(p)
    Array(prob_node(u0, p, st)[1])
end

function loss_neuralode(p)
    pred = predict_neuralode(p)
    loss = sum(abs2, ode_data .- pred)
    return loss, pred
end

# Callback function to observe training
anim = Animation()
callback = function (p, l, pred; doplot = true)
    ## display(l)
    if doplot
        plt = scatter(tsteps, ode_data[1,:], label = "data")
        scatter!(plt, tsteps, pred[1,:], label = "prediction")
        frame(anim)
        ## display(plot(plt))
    end
    return false
end

# Train using ADAM optimizer
adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x, p) -> loss_neuralode(x), adtype)

# Create an Optimizattion problem
optprob = Optimization.OptimizationProblem(optf, ComponentArray(p))

# Solve the problem
result_neuralode = Optimization.solve(
    optprob,
    ADAM(0.05),
    callback = callback,
    maxiters = 300
)

# Retrain using LBFGS optimizer
optprob2 = remake(optprob, u0 = result_neuralode.u)

result_neuralode2 = Optimization.solve(
    optprob2,
    LBFGS(),
    callback = callback,
    allow_f_increases = false
)

# ## Runtime information

import Pkg
Pkg.status()

#---

import InteractiveUtils
InteractiveUtils.versioninfo()

#===
# First Neural ODE example

A neural ODE is an ODE where a neural network defines its derivative function. $\dot{u} = NN(u)$

From: https://docs.sciml.ai/DiffEqFlux/stable/examples/neural_ode/
===#

using Lux, DiffEqFlux, DifferentialEquations, ComponentArrays
using Optimization, OptimizationOptimJL, OptimizationOptimisers
using Random, Plots
using DisplayAs: PNG

rng = Random.default_rng()

# True solution
function trueODEfunc(du, u, p, t)
    true_A = [-0.1 2.0; -2.0 -0.1]
    du .= ((u.^3)'true_A)'
end

# The data used for training
u0 = Float32[2.0; 0.0]
datasize = 30
tspan = (0.0f0, 1.5f0)
tsteps = range(tspan[begin], tspan[end], length = datasize)
prob_trueode = ODEProblem(trueODEfunc, u0, tspan)
ode_data = Array(solve(prob_trueode, Tsit5(), saveat = tsteps))

# Make a `NeuralODE` problem with a neural network defined by `Lux.jl`.
dudt2 = Lux.Chain(
    x -> x.^3,
    Lux.Dense(2, 50, tanh),
    Lux.Dense(50, 2)
)

p, st = Lux.setup(rng, dudt2)
prob_neuralode = NeuralODE(dudt2, tspan, Tsit5(), saveat = tsteps)

# Define output, loss, and callback functions.
function predict_neuralode(p)
    Array(prob_neuralode(u0, p, st)[1])
  end

function loss_neuralode(p)
    pred = predict_neuralode(p)
    loss = sum(abs2, ode_data .- pred)
    return loss, pred
end

# Do not generate plots by default. Users could change doplot=true to see the figures in the callback fuction.

callback = function (p, l, pred; doplot = false)
    println(l)
    ## plot current prediction against data
    if doplot
      plt = scatter(tsteps, ode_data[1,:], label = "data")
      scatter!(plt, tsteps, pred[1,:], label = "prediction")
      plot(plt) |> PNG
    end
    return false
end

# Try the callback function on the first iteration.
pinit = ComponentArray(p)
callback(pinit, loss_neuralode(pinit)...; doplot=true)

#===
Use Optimization.jl to solve the problem.
- `Zygote` for automatic differentiation (AD)
- `loss_neuralode` as the function to be optimized
- Make an `OptimizationProblem`
===#

adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x, p) -> loss_neuralode(x), adtype)
optprob = Optimization.OptimizationProblem(optf, pinit)

# Solve the `OptimizationProblem`.
result_neuralode = Optimization.solve(
    optprob,
    OptimizationOptimisers.ADAM(0.05),
    callback = callback,
    maxiters = 300
)

# Use another optimization algorithm `Optim.BFGS()` and start from where the `ADAM()` algorithm stopped.
optprob2 = remake(optprob, u0 = result_neuralode.u)

result_neuralode2 = Optimization.solve(
    optprob2,
    Optim.BFGS(initial_stepnorm=0.01),
    callback=callback,
    allow_f_increases = false
)

# Plot the solution to see if it matches the provided data.
callback(result_neuralode2.u, loss_neuralode(result_neuralode2.u)...; doplot=true)

# ## Animated solving process
# Let's reset the problem and visualize the training process.

rng = Random.default_rng()
u0 = Float32[2.0; 0.0]
datasize = 30
tspan = (0.0f0, 1.5f0)
tsteps = range(tspan[begin], tspan[end], length = datasize)

# Setup truth values for validation
true_A = Float32[-0.1 2.0; -2.0 -0.1]

function trueODEfunc!(du, u, p, t)
    du .= ((u.^3)'true_A)'
end

#---
prob_trueode = ODEProblem(trueODEfunc!, u0, tspan)
ode_data = Array(solve(prob_trueode, Tsit5(), saveat = tsteps))

#---
nodeFunc = Lux.Chain(
    x -> x.^3,
    Lux.Dense(2, 50, tanh),
    Lux.Dense(50, 2)
)

p, st = Lux.setup(rng, nodeFunc)

# Parameters in the neural network:
p

# Use `NeuroODE()` to construct the problem
prob_node = NeuralODE(nodeFunc, tspan, Tsit5(), saveat = tsteps)

# Predicted values.
function predict_neuralode(p)
    Array(prob_node(u0, p, st)[1])
end

# The loss function.
function loss_neuralode(p)
    pred = predict_neuralode(p)
    loss = sum(abs2, ode_data .- pred)
    return loss, pred
end

# Callback function to observe training process
anim = Animation()
callback = function (p, l, pred; doplot = true)
    if doplot
        plt = scatter(tsteps, ode_data[1,:], label = "data")
        scatter!(plt, tsteps, pred[1,:], label = "prediction")
        frame(anim)
    end
    return false
end

#---
adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x, p) -> loss_neuralode(x), adtype)
optprob = Optimization.OptimizationProblem(optf, ComponentArray(p))

# Solve the problem using the ADAM optimizer
result_neuralode = Optimization.solve(
    optprob,
    OptimizationOptimisers.ADAM(0.05),
    callback = callback,
    maxiters = 300
)

# And then solve the problem using the LBFGS optimizer
optprob2 = remake(optprob, u0 = result_neuralode.u)

result_neuralode2 = Optimization.solve(
    optprob2,
    Optim.LBFGS(),
    callback = callback,
    allow_f_increases = false
)

# Visualize fitting process
mp4(anim, fps=15)

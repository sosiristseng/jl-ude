#===
# Multiple Shooting

Docs: https://docs.sciml.ai/DiffEqFlux/dev/examples/multiple_shooting/

In Multiple Shooting, the training data is split into overlapping intervals. The solver is then trained on individual intervals.

The optimization is achieved by `OptimizationPolyalgorithms.PolyOpt()`.
===#

using Lux
using ComponentArrays
using DiffEqFlux
using Optimization
using OptimizationPolyalgorithms
using OrdinaryDiffEq
using DiffEqFlux: group_ranges
using Plots
using Random
using DisplayAs: PNG
rng = Random.default_rng()

# Define initial conditions and time steps
datasize = 30
u0 = Float32[2.0, 0.0]
tspan = (0.0f0, 5.0f0)
tsteps = range(tspan[begin], tspan[end], length = datasize)

# True values
true_A = Float32[-0.1 2.0; -2.0 -0.1]

# Generate data from the truth function.
function trueODEfunc!(du, u, p, t)
    du .= ((u.^3)'true_A)'
end

prob_trueode = ODEProblem(trueODEfunc!, u0, tspan)
ode_data = Array(solve(prob_trueode, Tsit5(), saveat = tsteps))

# Define the Neural Network
nn = Lux.Chain(
    x -> x.^3,
    Lux.Dense(2, 16, tanh),
    Lux.Dense(16, 2)
)
p_init, st = Lux.setup(rng, nn)

# Define the `NeuralODE` problem
neuralode = NeuralODE(nn, tspan, Tsit5(), saveat = tsteps)
prob_node = ODEProblem((u,p,t)->nn(u,p,st)[1], u0, tspan, ComponentArray(p_init))

#---
function plot_multiple_shoot(plt, preds, group_size)
	step = group_size-1
	ranges = group_ranges(datasize, group_size)

	for (i, rg) in enumerate(ranges)
		plot!(plt, tsteps[rg], preds[i][1,:], markershape=:circle, label="Group $(i)")
	end
end

# Animate training process by the callback function
anim = Animation()
callback = function (p, l, preds; doplot = true)
    ## display(l)
    if doplot
        ## plot the original data
        plt = scatter(tsteps, ode_data[1,:], label = "Data")
        ## plot the different predictions for individual shoot
        plot_multiple_shoot(plt, preds, group_size)
        frame(anim)
        ## display(plot(plt))
    end
    return false
end

# Define parameters for Multiple Shooting
group_size = 3
continuity_term = 200

function loss_function(data, pred)
	return sum(abs2, data - pred)
end

ps = ComponentArray(p_init)
pd, pax = getdata(ps), getaxes(ps)

function loss_multiple_shooting(p)
    ps = ComponentArray(p, pax)
    return multiple_shoot(ps, ode_data, tsteps, prob_node, loss_function, Tsit5(), group_size; continuity_term)
end

# Solve the problem
adtype = Optimization.AutoForwardDiff()
optf = Optimization.OptimizationFunction((x,p) -> loss_multiple_shooting(x), adtype)
optprob = Optimization.OptimizationProblem(optf, pd)

#---
res_ms = Optimization.solve(optprob, PolyOpt(), callback = callback)

# Visualize the fitting processes
mp4(anim, fps=15)

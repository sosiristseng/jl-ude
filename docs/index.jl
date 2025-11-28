#===
# UDEs

Universal Differential Equations (UDEs) are hybrids of differential equations and neural networks.

- https://github.com/SciML/DiffEqFlux.jl : fusing differential equations (`DifferentialEquations.jl`) and neural networks (`Lux.jl`).
- https://github.com/SciML/NeuralPDE.jl : physics-Informed Neural Networks (PINN) Solvers, learning and building the equations from the ground up. `NeuralPDE.jl` is slower than `DiffEqFlux.jl`.

## Runtime information
===#
using InteractiveUtils
InteractiveUtils.versioninfo()

#---
using Pkg
Pkg.status()

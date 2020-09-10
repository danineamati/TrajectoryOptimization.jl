#=

Simplified Rocket Soft-Landing Example (.jl version)
Please see the TrajectoryOptimization.jl github for the Jupyter Notebook
version.
=#

using RobotDynamics
import RobotDynamics: dynamics
using TrajectoryOptimization, Altro
using StaticArrays, LinearAlgebra

using Plots
pyplot()

#=
Here we build off of work found in the repository below to develop a
simplified rocket soft-landing scenario (i.e. no SOCP constraints and
no attitude dynamics).

https://github.com/danineamati/TrajOptSOCPs.jl

=#

struct Rocket{T} <: AbstractModel
    mass::T  # mass of the rocket
    g::SArray{Tuple{3},T,1,3}   # gravity
end

# Set-up the default constructor for ease of use
Rocket() = Rocket(10.0, SA[0.0; 0.0; -9.81])

@doc raw"""
    dynamics(model::Rocket, x, u)

We have a linear dynamics system, that when continuous is
```math
\frac{dx}{dt} = Ax + Bu + g
```

Where for a simple rocket, we have
```math
A = [0 \ I; 0 \ 0] \quad B = [0; \frac{1}{m} I] \quad G = [0; -g]
```

So, we can write it out explicitly to avoid multiplying by zero as
```math
\frac{dq}{dt} = \frac{dq}{dt}
\frac{d^2q}{dt^2} = \frac{1}{m} u - g
```
"""
function dynamics(model::Rocket, x, u)
    m = model.mass   # mass of the rocket in kg (1)
    g = model.g     # gravity m/s^2

    # q  = x[SA[1,2,3]] # [x, y, z, ...]
    qd = x[SA[4,5,6]] # [..., vx, vy, vz]

    nDim = size(u, 1) # i.e. a 3D system -> 3
    B = -(1/m) * I

    qdd = B * u - g
    return [qd; qdd]
end

RobotDynamics.state_dim(::Rocket) = 6
RobotDynamics.control_dim(::Rocket) = 3

# ----------------------
# Now we run with the model

model = Rocket()
n, m = size(model)
# n is the size of the states ([x, y, z, vx, vy, vz])
# m is the size of the control thrust ([Tx, Ty, Tz]])

# Trajectory Discretization
N = 151
tf = 5.
dt = tf/(N-1)

# Initial and Final Conditions
x0 = @SVector [1.0, 0.0, 20.0, -0.1, 0.0, -5.0] # Start at a 20 m altitude
xf = @SVector zeros(n)  # Swing pendulum up and end at rest

# LQR Objective Set Up
Q = 1.0e-2*Diagonal(@SVector ones(n))
Qf = 100.0*Diagonal(@SVector ones(n))
R = 1.0*Diagonal(@SVector ones(m))
obj = LQRObjective(Q,R,Qf,xf,N)

# Package the objective and constraints into a "problem" type
prob = Problem(model, obj, xf, tf, x0=x0)

# Now, initialize the trajectory
u0 = model.g # controls that would naminally hover
U0 = [u0 for k = 1:N-1] # vector of the small controls
initial_controls!(prob, U0)
rollout!(prob);

# The last step before solving is the solver options
opts = SolverOptions(
    cost_tolerance_intermediate=1e-2,
    penalty_scaling=10.,
    penalty_initial=1.0
)

altro = ALTROSolver(prob, opts)
set_options!(altro, show_summary=true)
solve!(altro)

X = states(altro)

# Now we want to plot the results
xs = [x[1] for x in X]
zs = [x[3] for x in X]

plt = plot(xs, zs, label = "Trajectory")
xlabel!("x")
ylabel!("z (height)")
display(plt)

xmin = minimum(xs)
xmax = maximum(xs)

zmin = minimum(zs)
zmax = maximum(zs)

@gif for i in 1:N
    plt = plot(xs[1:i], zs[1:i], label = "Trajectory")
    xlabel!("x")
    ylabel!("z (height)")
    xlims!(xmin,xmax)
    ylims!(zmin,zmax)
end

println("Complete!")

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
N = 251
tf = 10.
dt = tf/(N-1)

# Initial and Final Conditions
x0 = @SVector [1.0, 0.0, 20.0, -0.1, 0.0, -19.0] # Start at a 20 m altitude
xf = @SVector zeros(n)  # Swing pendulum up and end at rest

# LQR Objective Set Up
Q = 1.0e-2*Diagonal(@SVector ones(n))
Qf = 100.0*Diagonal(@SVector ones(n))
R = 1.0e-1*Diagonal(@SVector ones(m))
obj = LQRObjective(Q,R,Qf,xf,N)

# Now we get into the other constraints
# Create Empty ConstraintList
conSet = ConstraintList(n,m,N)

# Bounds
ground_level = 0.0 # Crash Prevention Constraint
umax = 11 * model.mass # Simplified Max Thrust Constraint
theta = 20 # deg. Max Angle Constraint
ulateral_max = umax * sind(theta)
# bnd = BoundConstraint(n,m, x_min=[-Inf, -Inf, ground_level,
#                                   -Inf, -Inf, -Inf])
bnd = BoundConstraint(n,m, x_min=[-Inf, -Inf, ground_level,
                                  -Inf, -Inf, -Inf],
                           u_min=[-ulateral_max, -ulateral_max, 0   ],
                           u_max=[ ulateral_max,  ulateral_max, umax])
add_constraint!(conSet, bnd, 1:N-1)

# Goal Constraint that the rocket must reach the landing site.
goal = GoalConstraint(xf)
add_constraint!(conSet, goal, N)

# Package the objective and constraints into a "problem" type
prob = Problem(model, obj, xf, tf, x0=x0, constraints=conSet)

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
U = controls(altro)

# Now we want to plot the results
println("Beginning Plotting")

xs = [x[1] for x in X]
zs = [x[3] for x in X]

uxs = [u[1] for u in U]
uzs = [u[3] for u in U]

titleText = "Rocket Soft-Landing Trajectory \n" *
        "starting at an initial $(x0[6]) m/s plummet"
filename = "RocketGroundThrustBound_BarelyFeasible3"

plt_x = plot(xs, zs, label = "Trajectory")
xlabel!("x")
ylabel!("z (height)")
title!(titleText)
display(plt_x)
savefig(filename * "_Trajectory")

plt_ux = plot(uxs, label = "Ux Control")
hline!([-ulateral_max, ulateral_max], linecolor = :grey, linestyle = :dash,
                label = "Max Lateral Thrust")
# xlabel!("time (s)")
ylabel!("control (N)")
title!("Controls over Time")

plt_uz = plot(uzs, label = "Uz Control")
hline!([umax], linecolor = :grey, linestyle = :dash, label = "Max Thrust")
xlabel!("time (s)")
ylabel!("control (N)")

plt_u = plot(plt_ux, plt_uz, layout = (2, 1))
display(plt_u)
savefig(filename * "_Controls")

println("Beginning Animation")

xmin = minimum(xs)
xmax = maximum(xs)

zmin = minimum(zs)
zmax = maximum(zs)

animSimple = @animate for i in 1:N
    plt = plot(xs[1:i], zs[1:i], label = "Trajectory", legend = :topleft)
    xlabel!("x")
    ylabel!("z (height)")
    xlims!(xmin,xmax)
    ylims!(zmin,zmax)
    title!(titleText)
end

gif(animSimple, filename * "_anim.gif")

println("Complete!")

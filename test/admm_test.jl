using PartedArrays
n1,m1 = 4,3
n2,m2 = 4,1
n = n1+n2
m = m1+m2

Q1 = Diagonal(1.0I,n1)
R1 = Diagonal(1.0I,m1)
Qf1 = Diagonal(0.0I,n1)
Q2 = Diagonal(1.0I,n2)
R2 = Diagonal(1.0I,m2)
Qf2 = Diagonal(10.0I,n2)

cost1 = QuadraticCost(Q1,R1,zeros(m1,n1),zeros(n1),zeros(m1),0,Qf1,zeros(n1),0)
cost2 = QuadraticCost(Q2,R2,zeros(m2,n2),zeros(n2),zeros(m2),0,Qf2,zeros(n2),0)
bodies = (:a1,:m)
costs = NamedTuple{bodies}((cost1,cost2))
costs.a1

part_x = create_partition((n1,n2),bodies)
part_u = create_partition((m1,m2),bodies)
y0 = [0.;1.;1.;0.]
v0 = zeros(m1)
z0 = [1.;0.;0.;0.]
w0 = zeros(m2)
x0 = [y0;z0]
d = 1
x = BlockArray(x0,part_x)
u = BlockArray(zeros(m1+m2),part_u)
ϕ(c,x::BlockArray,u::BlockArray) = copyto!(c, ϕ(c,x))
ϕ(c,x::BlockArray) = copyto!(c, norm(x.a1[1:2] - x.m[1:2]) - d^2)
ϕ(c,x::Vector,u::Vector) = ϕ(c,BlockArray(x,part_x),BlockArray(u,part_u))
ϕ(c,x::Vector) = ϕ(c,BlockArray(x,part_x))
function ∇ϕ(cx,cu,x::BlockArray,u::BlockArray)
    y = x.a1[1:2]
    z = x.m[1:2]
    cx[1:2] = 2(y-z)
    cx[5:6] = -2(y-z)
end
∇ϕ(cx,x) = begin y = x.a1[1:2];
                 z = x.m[1:2]; cx[1:2] = 2(y-z) end
part_cx = NamedTuple{bodies}([(1:1,rng) for rng in values(part_x)])
part_cu = NamedTuple{bodies}([(1:1,rng) for rng in values(part_u)])
cx = BlockArray(zeros(1,n),part_cx)
cu = BlockArray(zeros(1,m),part_cu)
∇ϕ(cx,cu,x,u)
∇ϕ(cx,x)


## Test joint solve
model = Dynamics.model_admm
tf = 1.0
y0 = [0.;1.]
ẏ0 = [0.;0.]
z0 = [0.;0.]
ż0 = [0.;0.]
x0 = [y0;ẏ0;z0;ż0]

yf = [10.;1.]
ẏf = ẏ0
zf = [10.;0.]
żf = ż0
xf = [yf;ẏf;zf;żf]

Q1 = Diagonal(1.0I,n1)
R1 = Diagonal(1.0I,m1)
Qf1 = Diagonal(0.0I,n1)
Q2 = Diagonal(1.0I,n2)
R2 = Diagonal(1.0I,m2)
Qf2 = Diagonal(10.0I,n2)

cost1 = LQRCost(Q1,R1,Qf1,[yf;ẏf])#QuadraticCost(Q1,R1,zeros(m1,n1),zeros(n1),zeros(m1),0,Qf1,zeros(n1),0)
cost2 = LQRCost(Q2,R2,Qf2,[zf;żf])#QuadraticCost(Q2,R2,zeros(m2,n2),zeros(n2),zeros(m2),0,Qf2,zeros(n2),0)#LQRCost(Q2,R2,Qf2,[zf;żf])
costs = NamedTuple{bodies}((cost1,cost2))
acost = ADMMCost(costs,ϕ,∇ϕ,2,[:a1],n1+n2,m1+m2,part_x,part_u)

bodies = (:a1,:m)
ns = (n1,n2)
ms = (m1,m2)
N = 11

# Q = Diagonal(0.0001I,model.n)
# R = Diagonal(0.0001I,model.m)
# Qf = Diagonal(100.0I,model.n)

# function cE(c,x::AbstractArray,u::AbstractArray)
#     c[1] = norm(x[1:2] - x[5:6])^2 - d^2
#     c[2] = u[3] - u[4]
# end
# function cE(c,x::AbstractArray)
#     c[1] = norm(x[1:2] - x[5:6])^2 - d^2
# end

obj = UnconstrainedObjective(acost,tf,x0,xf)
obj = ConstrainedObjective(obj,cE=ϕ,cE_N=ϕ,∇cE=∇ϕ,use_xf_equality_constraint=false)
p = obj.p
p_N = obj.p_N

solver = Solver(model,obj,integration=:none,dt=0.1)
res = ADMMResults(bodies,ns,ms,p,N,p_N);
U0 = ones(model.m,solver.N-1)

J0 = initial_admm_rollout!(solver,res,U0);
J0
J = ilqr_loop(solver,res)
update_constraints!(res,solver)
λ_update_default!(res,solver)
μ_update_default!(res,solver)

J = admm_solve(solver,res,U0)
plot(res.X,label="")

admm_plot(res)
function admm_solve(solver,results,U0)
    J0 = initial_admm_rollout!(solver,res,U0);
    println("J0 = $J0")
    for i = 1:10
        J = ilqr_loop(solver,res)
        println("iter: $i, J = $J")
        update_constraints!(res,solver)
        λ_update_default!(res,solver)
        μ_update_default!(res,solver)
    end
    return J
end

function ilqr_loop(solver::Solver,res::ADMMResults)
    J = Inf
    for b in res.bodies
        J = ilqr_solve(solver::Solver,res::ADMMResults,b::Symbol)
    end
    return J
end

function initial_admm_rollout!(solver::Solver,res::ADMMResults,U0)
    for k = 1:N-1
        res.U[k] .= U0[:,k]
    end

    for b in res.bodies
        rollout!(res,solver,1.0,b)
    end
    copyto!(res.X,res.X_);
    copyto!(res.U,res.U_);

    J = cost(solver,res)
    return J
end

function ilqr_solve(solver::Solver,res::ADMMResults,b::Symbol)
    X = res.X; U = res.U; X_ = res.X_; U_ = res.U_
    J0 = cost(solver,res)
    update_jacobians!(res,solver)
    Δv = _backwardpass_admm!(res,solver,b)
    J = forwardpass!(res, solver, Δv, J0,b)

    for ii = 1:solver.opts.iterations_innerloop
        iter_inner = ii

        ### BACKWARD PASS ###
        update_jacobians!(res, solver)
        Δv = _backwardpass_admm!(res, solver, b)

        ### FORWARDS PASS ###
        J = forwardpass!(res, solver, Δv, J0, b)
        c_max = max_violation(res)

        ### UPDATE RESULTS ###
        copyto!(X,X_)
        copyto!(U,U_)

        dJ = copy(abs(J-J0)) # change in cost
        J0 = copy(J)

        evaluate_convergence(solver,:inner,dJ,c_max,Inf,ii,0,0) ? break : nothing
    end
    return J
end

function rollout!(res::ADMMResults,solver::Solver,alpha::Float64,b::Symbol)
    n,m,N = get_sizes(solver)
    m̄,mm = get_num_controls(solver)
    n̄,nn = get_num_states(solver)

    dt = solver.dt

    X = res.X; U = res.U;
    X_ = res.X_; U_ = res.U_

    K = res.K[b]; d = res.d[b]

    X_[1] .= solver.obj.x0;

    for k = 2:N
        # Calculate state trajectory difference
        δx = X_[k-1][b] - X[k-1][b]

        # Calculate updated control
        copyto!(U_[k-1][b], U[k-1][b] + K[k-1]*δx + alpha*d[k-1])

        # Propagate dynamics
        solver.fd(X_[k], X_[k-1], U_[k-1], dt)

        # Check that rollout has not diverged
        if ~(norm(X_[k],Inf) < solver.opts.max_state_value && norm(U_[k-1],Inf) < solver.opts.max_control_value)
            return false
        end
    end

    # Update constraints
    update_constraints!(res,solver,X_,U_)

    return true
end
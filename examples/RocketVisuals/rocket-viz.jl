using Colors # Handle RGB colors
using CoordinateTransformations # Translations and rotations
using FileIO # Save and load files
using GeometryTypes # Define geometric shape
using LinearAlgebra
using MeshCat # Visualize 3D animations
using MeshIO # Load meshes in MeshCat
using Meshing
include(joinpath(pwd(),"dynamics/visualize.jl"))


vis = Visualizer()
open(vis);

obj_rocket = "/home/taylor/Research/direct_policy_optimization/dynamics/rocket/space_x_booster.obj"
mtl_rocket = "/home/taylor/Research/direct_policy_optimization/dynamics/rocket/space_x_booster.mtl"
q = deepcopy(X_nom_sample)
rkt_offset = [3.9,-6.35,0.2]
ctm = ModifiedMeshFileObject(obj_rocket,mtl_rocket,scale=1.0)
t = 1
setobject!(vis["rocket1"],ctm)
settransform!(vis["rocket1"], compose(Translation(([q[t][1];0.0;q[t][2]] + rkt_offset)...),LinearMap(RotY(-1.0*q[t][3])*RotZ(pi)*RotX(pi/2.0))))
t = 7
setobject!(vis["rocket2"],ctm)
settransform!(vis["rocket2"], compose(Translation(([q[t][1];0.0;q[t][2]] + rkt_offset)...),LinearMap(RotY(-1.0*q[t][3])*RotZ(pi)*RotX(pi/2.0))))
t = 15
setobject!(vis["rocket3"],ctm)
settransform!(vis["rocket3"], compose(Translation(([q[t][1];0.0;q[t][2]] + rkt_offset)...),LinearMap(RotY(-1.0*q[t][3])*RotZ(pi)*RotX(pi/2.0))))
t = 29
setobject!(vis["rocket4"],ctm)
settransform!(vis["rocket4"], compose(Translation(([q[t][1];0.0;q[t][2]] + rkt_offset)...),LinearMap(RotY(-1.0*q[t][3])*RotZ(pi)*RotX(pi/2.0))))
t = T
setobject!(vis["rocket5"],ctm)
settransform!(vis["rocket5"], compose(Translation(([q[t][1];0.0;q[t][2]] + rkt_offset)...),LinearMap(RotY(-1.0*q[t][3])*RotZ(pi)*RotX(pi/2.0))))

obj_platform = "/home/taylor/Research/direct_policy_optimization/dynamics/rocket/space_x_platform.obj"
mtl_platform = "/home/taylor/Research/direct_policy_optimization/dynamics/rocket/space_x_platform.mtl"

ctm_platform = ModifiedMeshFileObject(obj_platform,mtl_platform,scale=1.0)
setobject!(vis["platform"],ctm_platform)
settransform!(vis["platform"], compose(Translation(0.0,0.0,0.0),LinearMap(RotZ(pi)*RotX(pi/2))))

setvisible!(vis["rocket1"],true)
setvisible!(vis["rocket2"],true)
setvisible!(vis["rocket3"],true)
setvisible!(vis["rocket4"],true)
setvisible!(vis["rocket5"],true)

anim = MeshCat.Animation(convert(Int,floor(1/dt_sim_sample)))

q = [z_sample...,[z_sample[end] for t = 1:length(z_sample)]...]
q = [z_tvlqr...,[z_tvlqr[end] for t = 1:length(z_tvlqr)]...]
for t = 1:length(q)
	MeshCat.atframe(anim,t) do
		settransform!(vis["rocket1"], compose(Translation(q[t][1]+rkt_offset[1],0.0+rkt_offset[2],q[t][2]+rkt_offset[3]),LinearMap(RotY(-1.0*q[t][3])*RotZ(pi)*RotX(pi/2.0))))
	end
end
# settransform!(vis["/Cameras/default"], compose(Translation(-1, -1, 0),LinearMap(RotZ(pi/2))))
MeshCat.setanimation!(vis,anim)

q_to = deepcopy(X_nom)
for t = 1:T
	setobject!(vis["traj_to$t"], HyperSphere(Point3f0(0),
		convert(Float32,0.25)),
		MeshPhongMaterial(color=RGBA(0.0,255.0/255.0,255.0/255.0,1.0)))
	settransform!(vis["traj_to$t"], Translation((q_to[t][1],-2.35-0.5,q_to[t][2]+3.0)))
	setvisible!(vis["traj_to$t"],true)
end

q_dpo = deepcopy(X_nom_sample)
for t = 1:T
	setobject!(vis["traj_dpo$t"], HyperSphere(Point3f0(0),
		convert(Float32,0.25)),
		MeshPhongMaterial(color=RGBA(255.0/255.0,127.0/255.0,0.0,1.0)))
	settransform!(vis["traj_dpo$t"], Translation((q_dpo[t][1],-2.35-1.0,q_dpo[t][2]+3.0)))
	setvisible!(vis["traj_dpo$t"],true)
end

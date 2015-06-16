from control3.common import *
mdp = get_mdp("mjc:hopper4ball")
world = mdp.world
x0 = mdp.default_state()
u = np.random.randn(mdp.ctrl_dim())

T=5
ys,fs,dcoms,dists,kins = world.StepMulti2( np.repeat(x0[None,:],T,axis=0).astype('float64'), np.repeat(u[None,:], T, axis=0).astype('float64'), np.zeros(T,'uint8') )
assert (ys==ys[0:1]).all() and  (fs==fs[0:1]).all() and  (dcoms==dcoms[0:1]).all() and  (dists==dists[0:1]).all()

u = np.random.randn(5,mdp.ctrl_dim())
x = np.repeat(x0[None,:],T,axis=0).astype('float64')
ys = world.StepMulti( x, u)
ys1,_,_,_,_ = world.StepMulti2( x,u,np.zeros(T,'uint8'))
assert (ys==ys1).all()
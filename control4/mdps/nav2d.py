from control4.core.mdp import MDP
from control4.config import floatX
import numpy as np

def idx2onehot(i,n):
    out = np.zeros(n,floatX)
    out[i] = 1
    return out


class Nav2dState(object):
    def __init__(self,pos,targpos,t):
        self.pos = pos
        self.targpos = targpos
        self.t = t
        self.path = [pos.copy()]

class Nav2D(MDP):
    def __init__(self, halfsize=3.0, obs_efference=True, obs_cur_pos=False, target_mode="four"):
        self.halfsize = halfsize
        self.thresh_dist = 0.8
        self.sample_frac = 0.2
        self.t_max = 50
        self.viewer = None
        self.target_mode = target_mode
        self.obs_efference=obs_efference
        self.obs_cur_pos=obs_cur_pos

        self._obs_dim = 4
        if self.target_mode == "four":
            self._obs_dim += 4
        if self.obs_efference:
            self._obs_dim += 2
        if self.obs_cur_pos:
            self._obs_dim += 2

    def call(self, input_arrs):
        state = input_arrs["x"]
        u = input_arrs["u"]
        assert u.shape[0]==1
        u = u[0]
        u = np.clip(u,-1,1)
        ytarg = state.pos + u
        halfsize = self.halfsize

        components = [ytarg > halfsize, ytarg < -halfsize]
        if self.target_mode == "four":
            components.append(np.zeros(4,floatX))
        ytarg = np.clip(ytarg, -halfsize, halfsize)
        state.path.append(ytarg.copy())
        state.pos = ytarg

        targ_pos = state.targpos
        state.t += 1
        done = int(state.t == self.t_max or np.square(targ_pos - state.pos).sum() < self.thresh_dist**2)
        cost = np.array([done*(state.t - self.t_max)],floatX)

        ################################
        # Observation
        if self.obs_efference: components.append(u)
        if self.obs_cur_pos: components.append(targ_pos) # XXX
        o = np.concatenate(components)

        return {
            "x" : state,
            "o" : o.reshape(1,-1),
            "c" : cost.reshape(1,-1),
            "done" : done
        }

    def initialize_mdp_arrays(self):

        frac = self.sample_frac
        pos = (2*np.random.rand(2)-.1)*self.halfsize*frac
        pos = pos.astype(floatX)

        halfsize = self.halfsize
        if self.target_mode == "four":
            targs_42 = np.array([[halfsize,0],[0,halfsize],[-halfsize,0],[0,-halfsize]],floatX)        
            targidx = np.random.randint(4)
            targpos = targs_42[targidx]
        elif self.target_mode == "unif":
            targpos = np.random.uniform(low=-halfsize,high=halfsize,size=(2,)).astype(floatX)
            targidx = None

        x_init = Nav2dState(pos, targpos, 0)
        if targidx is None: 
            components = [np.zeros(4,floatX)]
        else:            
            components = [np.zeros(4,floatX),idx2onehot(targidx,4)]
        if self.obs_efference: components.append(np.zeros(2,floatX))
        if self.obs_cur_pos: components.append(pos)
        o_init = np.concatenate(components)
        c_init = np.array([0],floatX)

        return {
            "x" : x_init,
            "o" : o_init.reshape(1,-1),
        }

    def input_info(self):
        return {
            "x" : None,
            "u" : (2,floatX)        
        }

    def output_info(self):
        return {
            "x" : None,
            "o" : (self._obs_dim,floatX),
            "c" : (1,floatX),
            "done" : (None,'uint8')
        }

    def plot(self,input_arrs):
        import pygame
        x = input_arrs["x"]
        targpos = x.targpos
        from control4.viz.pygameviewer import PygameViewer, pygame
        if self.viewer is None:
            self.viewer = PygameViewer(size=(400,400))
        screen = self.viewer.screen
        screen.fill((255,255,255))

        tgt_x,tgt_y = targpos
        target_radius_pix = int(self.thresh_dist * 400/(2*self.halfsize))
        pygame.draw.circle(screen, (255,0,0), (self._to_pix(tgt_x),self._to_pix(tgt_y)), target_radius_pix)
        # pygame.draw.circle(screen, (0,0,0), (self._to_pix(pos_x),self._to_pix(pos_y)), 8)        
        pygame.draw.lines(screen, (0,255,0), False, [self._to_pix(pt) for pt in x.path])
        for (i,pt) in enumerate(x.path): 
            if i==0: color=(0,0,255)
            elif i==len(x.path)-1: color=(0,0,0)
            else: color = (0,255,0)
            pygame.draw.circle(screen, color,  self._to_pix(pt), 6)
        pygame.display.flip()

    def cost_names(self):
        return ["targethit"]

    def ctrl_bounds(self):
        return np.array([[-1,-1],[1,1]],floatX)

    def obs_bounds(self):
        h = self.halfsize
        return np.array([[-h,-h],[h,h]],floatX)

    def _to_pix(self,z):
        return np.int32((z+self.halfsize)*400/(2*self.halfsize))


if __name__ == "__main__":
    mdp = Nav2D()
    mdp.validate()
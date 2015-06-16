import theano.tensor as TT
from control4.config import floatX
import numpy as np
from control4.maths.numeric import uniform
from control4.core.symbolic_mdp import SymbolicMDP

class CartpoleBase(SymbolicMDP):
    def __init__(self, max_cart_pos, max_cart_speed, max_pole_speed, max_force, dt):
        self.max_cart_pos = max_cart_pos
        self.max_cart_speed = max_cart_speed
        self.max_pole_speed = max_pole_speed
        self.max_force = max_force
        self.dt = dt
        self.viewer = None   

        SymbolicMDP.__init__(self)

    def symbolic_call(self,x,u):
        raise NotImplementedError

    def output_names(self):
        return ["x","o","c","done"]

    def initialize_mdp_arrays(self):
        raise NotImplementedError

    def input_info(self):
        return {
            "x":(4,floatX),
            "u":(1,floatX)
        }

    def output_info(self):
        return {
            "x":(4,floatX),
            "o":(4,floatX),
            "c":(self.num_costs(),floatX),
            "done":(None,'uint8')
        }

    def plot(self,arrs):
        x = arrs["x"][0]
        u = arrs["u"][0]
        from control4.viz.pygameviewer import PygameViewer, pygame
        if self.viewer is None:
            self.viewer = PygameViewer()
        screen = self.viewer.screen
        screen.fill((255,255,255))
        
        world_width = 10
        screen_width = screen.get_width()
        screen_height = screen.get_height()
        scale = screen_width/world_width
        
        cartpos = x[0]*scale+screen_width/2
        

        poleang = x[2]
        force = u[0]
        
        poleheight = 100
        polewidth=10
                
        cartwidth=50
        cartheight=30

        cartleftx = cartpos - cartwidth/2
        carttopy  = screen_height/2 - cartheight/2


        pygame.draw.rect(screen, (0,0,0), pygame.Rect(cartleftx, carttopy, cartwidth,cartheight))
        
        poleorigin = np.array([cartpos, carttopy])
        polelocalpoints = np.array([[-polewidth/2, 0],[polewidth/2,0],[polewidth/2,-poleheight],[-polewidth/2,-poleheight]])
        polerotmat = np.array([[np.cos(poleang),np.sin(poleang)],[-np.sin(poleang),np.cos(poleang)]])
        poleworldpoints = poleorigin[None,:] + polelocalpoints.dot(polerotmat)
        
        pygame.draw.polygon(screen, (0,255,0), poleworldpoints)
        
        pygame.draw.line(screen, (255,0,0), (cartpos, screen_height/2), (cartpos + force*100, screen_height/2),5)
        pygame.display.flip()

    def cost_names(self):
        return ["state","ctrl"]

    def ctrl_bounds(self):
        hi = np.r_[self.max_force]
        return np.array([-hi,hi],floatX)

    def obs_bounds(self):
        raise NotImplementedError

    def obs_wraps(self):
        raise NotImplementedError


class CartpoleBarto(CartpoleBase):
    def __init__(self):
        self.max_pole_angle = .2
        CartpoleBase.__init__(self, 2.4, 4., 4., 10, .05)

    def symbolic_call(self,x,u):

        u = TT.clip(u, -self.max_force, self.max_force) #pylint: disable=E1111

        dt = self.dt

        z = TT.take(x,0,axis=x.ndim-1)
        zdot = TT.take(x,1,axis=x.ndim-1)    
        th = TT.take(x,2,axis=x.ndim-1)
        thdot = TT.take(x,3,axis=x.ndim-1)
        u0 = TT.take(u,0,axis=u.ndim-1)

        th1 = np.pi - th

        g = 10.
        mc = 1. # mass of cart
        mp = .1 # mass of pole
        muc = .0005 # coeff friction of cart
        mup = .000002 # coeff friction of pole
        l = 1. # length of pole

        def sign(x):
            return TT.switch(x>0, 1, -1)

        thddot = -(-g*TT.sin(th1)
         + TT.cos(th1) * (-u0 - mp * l *thdot**2 * TT.sin(th1) + muc*sign(zdot))/(mc+mp)
          - mup*thdot / (mp*l))  \
        / (l*(4/3. - mp*TT.cos(th1)**2 / (mc + mp)))
        zddot = (u0 + mp*l*(thdot**2 * TT.sin(th1) - thddot * TT.cos(th1)) - muc*sign(zdot))  \
            / (mc+mp)

        newzdot = zdot + dt*zddot
        newz = z + dt*newzdot
        newthdot = thdot + dt*thddot
        newth = th + dt*newthdot

        done = (z > self.max_cart_pos) | (z < -self.max_cart_pos) | (th > self.max_pole_angle) | (th < -self.max_pole_angle) 

        ucost = 1e-5*(u**2).sum(axis=u.ndim-1)
        xcost = 1-TT.cos(th)
        # notdone = TT.neg(done) #pylint: disable=W0612,E1111
        notdone = 1-done
        costs = TT.stack((done-1)*10., notdone*xcost, notdone*ucost).T #pylint: disable=E1103


        newx = TT.stack(newz, newzdot, newth, newthdot).T #pylint: disable=E1103

        return [newx,newx,costs,done]

    def initialize_mdp_arrays(self):
        lo, hi = -0.05*np.ones(4,floatX),0.05*np.ones(4,floatX)
        x = uniform(lo,hi).astype(floatX)
        o = x
        return {"x":o.reshape(1,-1),"o":o.reshape(1,-1)}

    def cost_names(self):
        return ["done", "state","ctrl"]

    def obs_bounds(self):
        hi = np.r_[self.max_cart_pos, self.max_cart_speed, self.max_pole_angle, self.max_pole_speed]
        return np.array([-hi,hi],floatX)

    def obs_ranges(self):
        return self.obs_bounds()        

    def obs_wraps(self):
        return [False,False,False,False]


if __name__ == "__main__":
    mdp = CartpoleBarto()
    mdp.validate()
import theano.tensor as TT
from control4.config import floatX
import numpy as np
from control4.maths.numeric import angle_normalize,uniform
from control4.misc.func_utils import once
from control4.core.symbolic_mdp import SymbolicMDP

class Pendulum(SymbolicMDP):

    def __init__(self, max_speed=8, max_torque=2., dt=.05):        
        self.max_speed = max_speed
        self.max_torque = max_torque
        self.dt = dt

        self.viewer = None

        SymbolicMDP.__init__(self)

    def symbolic_call(self,x,u):
        dt = self.dt

        a = TT.take(x,0,axis=x.ndim-1)
        adot = TT.take(x,1,axis=x.ndim-1)
        g = 10.
        m = 1.
        l = 1.

        u = TT.clip(u, -self.max_torque, self.max_torque) #pylint: disable=E1111

        newadot = adot + (-3*g/(2*l) * TT.sin(a + np.pi) + 3./(m*l**2)*TT.take(u,0,axis=u.ndim-1)) * dt
        newa = a + newadot*dt
        newadot = TT.clip(newadot, -self.max_speed, self.max_speed) #pylint: disable=E1111

        newx = TT.stack(newa, newadot).T #pylint: disable=E1103


        x0 = TT.take(x,0,axis=x.ndim-1)
        x1 = TT.take(x,1,axis=x.ndim-1)
        costs = TT.stack(angle_normalize(x0)**2 + .1*x1**2, .001*(u**2).sum(axis=u.ndim-1)).T #pylint: disable=E1103

        return [newx, newx, costs]

    def output_names(self):
        return ["x","o","c"]

    def initialize_mdp_arrays(self):
        lo, hi = self.obs_bounds()
        x = uniform(lo,hi).astype(floatX)
        o = x
        return {"x":o.reshape(1,-1),"o":o.reshape(1,-1)}

    def input_info(self):
        return {
            "x":(2,floatX),
            "u":(1,floatX)
            }

    def output_info(self):
        return {
            "x":(2,floatX),
            "o":(2,floatX),
            "c":(self.num_costs(),floatX),        
        }

    def plot(self, arrs):
        x = arrs["x"].ravel()
        u = arrs["u"].ravel()

        from control4.viz.pygameviewer import PygameViewer, pygame
        if self.viewer is None:
            self.viewer = PygameViewer()
        
        screen = self.viewer.screen
        screen.fill((255,255,255))
        screen_width = screen.get_width()
        screen_height = screen.get_height()
        
        cartpos = screen_width/2
        

        poleang = x[0]
        
        
        poleheight = 100
        polewidth=10
                
        cartheight=30

        carttopy  = screen_height/2 - cartheight/2
        
        poleorigin = np.array([cartpos, carttopy])
        polelocalpoints = np.array([[-polewidth/2, 0],[polewidth/2,0],[polewidth/2,-poleheight],[-polewidth/2,-poleheight]])
        polerotmat = np.array([[np.cos(poleang),-np.sin(poleang)],[np.sin(poleang),np.cos(poleang)]])
        poleworldpoints = poleorigin[None,:] + polelocalpoints.dot(polerotmat)
        
        pygame.draw.polygon(screen, (0,255,0), poleworldpoints)
        pygame.draw.circle(screen, (255,0,0), poleorigin, 10)

        if u[0] != 0: 
            screen.blit(get_counterclockwise_pendulum_img() if u[0]>0 else get_clockwise_pendulum_img(), pygame.rect.Rect(screen_width/2-50, screen_height/2-50,100,100))

        pygame.display.flip()
    
    def cost_names(self):
        return ["state","ctrl"]

    def ctrl_bounds(self):
        hi = np.r_[self.max_torque]
        return np.array([-hi,hi],floatX)

    ################################

    def obs_bounds(self):
        hi = np.r_[np.pi, self.max_speed]
        return np.array([-hi,hi],dtype=floatX)

    def obs_wraps(self):
        return [True,False]

    ################################


@once
def get_counterclockwise_pendulum_img():
    import pygame
    from control4.config import CTRL_ROOT
    import os.path as osp
    img =  pygame.image.load( osp.join(CTRL_ROOT,"domain_data/misc/clockwise.png" ))
    img = pygame.transform.flip(img, False,True)
    img = pygame.transform.scale(img, (100,100))
    return img

@once
def get_clockwise_pendulum_img():
    import pygame
    return pygame.transform.flip(get_counterclockwise_pendulum_img(), True,False)    


if __name__ == "__main__":
    mdp = Pendulum()
    mdp.validate()
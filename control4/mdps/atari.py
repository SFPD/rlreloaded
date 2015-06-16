from control4.core.mdp import MDP
import ctypes as ct, sys
from control4.config import CTRL_ROOT,floatX
import os.path as osp
import numpy as np
import cv2 #pylint: disable=F0401

LIBRARY_LOADED = False
def load_library():
    import cycontrol
    libfilename = osp.join(osp.dirname(cycontrol.__file__),"libatariemu.so")
    lib=ct.cdll.LoadLibrary(libfilename)
    global emuNewState, emuDeleteState, emuReset, emuStep, emuGetImage, emuGetRam #pylint: disable=W0601
    emuNewState=lib.emuNewState
    emuNewState.restype = ct.c_void_p
    emuDeleteState=lib.emuDeleteState
    emuReset=lib.emuReset
    emuStep=lib.emuStep
    emuGetImage=lib.emuGetImage
    emuGetRam=lib.emuGetRam

class Action(ct.Structure):
    _fields_ = [("horiz", ct.c_int),
                ("vert", ct.c_int),
                ("button", ct.c_int),
                ]


class AtariMDP(MDP):
    def __init__(self,game,obs_mode="ram",frame_skip=4,use_color=False):
        self.game = game
        self.obs_mode = obs_mode
        self.frame_skip = frame_skip
        self.use_color = use_color
        
        global LIBRARY_LOADED
        if not LIBRARY_LOADED: 
            load_library()
            LIBRARY_LOADED=True
        rom_dir = osp.join(CTRL_ROOT,"domain_data/atari_roms")
        p = emuNewState(ct.c_char_p(game), ct.c_char_p(rom_dir))
        if p is None: sys.exit(1)
        self.e = ct.c_void_p(p)

        # Need image stuff either way for possibly plotting        
        self.raw_img_arr = np.zeros((210,160,3),'uint8')
        self.raw_img_buf = self.raw_img_arr.ctypes.data_as(ct.POINTER(ct.c_char))

        if self.obs_mode == "ram": 
            self.ram_arr = np.zeros((1,128),'uint8')
            self.ram_buf = self.ram_arr.ctypes.data_as(ct.POINTER(ct.c_char))    
            self._obs_dim = 128    
        elif self.obs_mode == "image": 
            self.ds = 3
            self.row_start = (210%self.ds)//2
            self.row_end = 210-self.row_start
            self.col_start = (160%self.ds)//2
            self.col_end = 160-self.col_start
            self.obs_height = 210//self.ds
            self.obs_width = 160//self.ds
            self._obs_dim = self.obs_width*self.obs_height*self.frame_skip
        else:
            raise NotImplementedError

    def _get_ram_obs(self):
        emuGetRam(self.e,self.ram_buf)
        return self.ram_arr.copy()

    def _get_preprocessed_image(self):
        emuGetImage(self.e,self.raw_img_buf)
        return self.preproc(self.raw_img_arr)

    def preproc(self,img):
        if not self.use_color:
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img[self.row_start:self.row_end,self.col_start:self.col_end], 
            (self.obs_width,self.obs_height),interpolation=cv2.INTER_AREA)
        if self.use_color: 
            img = img.transpose(2,0,1)
        return img

    def call(self, input_arrs):
        
        reward = ct.c_int()
        gameOver = ct.c_bool()
        roundOver = ct.c_bool()

        u = input_arrs["u"]
        assert len(u)==1
        h,v,b = u[0]
        a = Action(h,v,b)

        ctotal = 0
        roundEnded = False

        if self.obs_mode == "ram":
            for _ in xrange(self.frame_skip):
                emuStep(self.e,ct.byref(a),ct.byref(reward),ct.byref(gameOver),ct.byref(roundOver))
                ctotal -= reward.value
                roundEnded |= roundOver.value        
            obs = self._get_ram_obs()
        elif self.obs_mode == "image":
            imgs = []
            for _ in xrange(self.frame_skip):
                emuStep(self.e,ct.byref(a),ct.byref(reward),ct.byref(gameOver),ct.byref(roundOver))
                imgs.append(self._get_preprocessed_image())
                ctotal -= reward.value
                roundEnded |= roundOver.value        
            obs = np.array(imgs).reshape(1,-1)
        else:
            raise NotImplementedError

        return {
            "o" : obs,
            "c" : np.array([[ctotal]],floatX),
            "done" : gameOver.value,
            "ro" : roundEnded
        }

    def initialize_mdp_arrays(self):
        emuReset(self.e)
        return {
            "o" : np.array([self._get_preprocessed_image() for _ in xrange(self.frame_skip)]).reshape(1,-1)
                if self.obs_mode == "image"
                else self._get_ram_obs()
        }

    def input_info(self):
        return {
            "x" : None,
            "u" : (3,'int64')
        }

    def output_info(self):
        return {
            "o" : (self._obs_dim, 'uint8'),
            "c" : (1, floatX),
            "done" : (None,'uint8'),
            "ro" : (None,'uint8')
        }

    def plot(self, _arrs):
        emuGetImage(self.e,self.raw_img_buf)
        cv2.imshow('hi',self.raw_img_arr.reshape(210,160,3))
        cv2.waitKey(5)

    def cost_names(self):
        return ["dscore"]

    def img_shape(self):
        return (self.frame_skip*(3 if self.use_color else 1), self.obs_height, self.obs_width)

def obs2img(mdp,o):
    frame_skip,height,width = mdp.img_shape()
    o = o.reshape(frame_skip,height,width).transpose(1,2,0)
    if o.shape[2] == 2:
        o = np.repeat(o,2,axis=2)[:,:,:3]
    elif o.shape[2] == 3:
        o = o
    elif o.shape[2] > 3:
        o = o[:,:,:3]
    else:
        raise NotImplementedError
    o = cv2.resize(o, (width*4,height*4),interpolation=cv2.INTER_NEAREST)
    return o

if __name__ == "__main__":


    mdp = AtariMDP("space_invaders",obs_mode='image',frame_skip=2)
    mdparrs = mdp.initialize_mdp_arrays()
    t = 0
    while True:
        t += 1
        horiz=np.random.randint(-1,2)
        vert=np.random.randint(-1,2)
        button=np.random.randint(0,2)
        u = np.array([[horiz,vert,button]])
        out = mdp.call({"u":u})
        print out["ro"]
        frame_skip,height,width = mdp.img_shape()
        cv2.imshow('hi',obs2img(mdp,out["o"]))
        cv2.imshow('raw',mdp.raw_img_arr.reshape(210,160,3))
        cv2.waitKey(-1)        
        if out['done']: break
    print "done after %i timesteps"%t
    # img = img.reshape(mdp.img_shape())
    mdp.validate()
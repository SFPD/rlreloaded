"""
Toy example of navigating through text to find the answer to a query.
This is the simplest possible version of the problem.
"""

from control4.core.mdp import MDP
from control4.config import floatX
import numpy as np

def idx2onehot(i,n):
    out = np.zeros(n,floatX)
    out[i] = 1
    return out


class TextNavState(object):
    def __init__(self,textarr,answer,pos):
        self.textarr = textarr
        self.binarr = textarr2binarr(self.textarr)
        self.answer = answer
        self.pos = pos

    def printme(self):
        textarr = self.textarr.copy()
        r,c = self.pos
        textarr[r,c] = "&"
        print
        print "\n".join([" ".join(row) for row in textarr])
        print

def gen_textarr(height,width,xblocksize):
    xblockrow = np.random.randint(low=0,high=height-xblocksize)
    xblockcol = np.random.randint(low=0,high=width-xblocksize)
    arow = np.random.randint(low=xblockrow,high=xblockrow+xblocksize)
    acol = np.random.randint(low=xblockcol,high=xblockcol+xblocksize)
    textarr = np.zeros((height,width),dtype="S1")
    textarr[:] = "."
    textarr[xblockrow:xblockrow+xblocksize,xblockcol:xblockcol+xblocksize]="*"
    ans = np.random.randint(low=0,high=2)
    textarr[arow,acol] = str(ans)
    return textarr,ans

def textarr2binarr(textarr):
    chars = [".","*","0","1"]    
    height,width = textarr.shape
    char2bin = {c:idx2onehot(chars.index(c),len(chars)) for c in chars}
    binarr = np.empty((height,width,len(chars)),'uint8')
    for row in xrange(height):
        for col in xrange(width):
            binarr[row,col] = char2bin[textarr[row,col]]
    return binarr

class TextNav(MDP):
    def __init__(self, width=10,height=10,xblocksize=3):
        self.width = width
        self.height = height
        self.xblocksize = xblocksize

    def call(self, input_arrs):
        state = input_arrs["x"]
        u = input_arrs["u"]
        assert u.shape[0]==1
        u = u[0]

        # NESW + A0 A1
        height,width = state.textarr.shape
        row,col = state.pos
        done = False
        cost = 0
        if u == 0: # NORTH
            row = max(row-1,0)
        elif u == 1: # EAST
            col = min(col+1,width-1)
        elif u == 2: # SOUTH
            row = min(row+1,height-1)
        elif u == 3: # WEST
            col = max(col-1,0)
        else: # answer 0,1
            cost = (u-4==state.answer) - .4
            done = True


        state.pos = (row,col)
        o = np.concatenate([state.binarr[row,col], state.pos / np.array([self.width,self.height],floatX)-.5]).astype(floatX)

        return {
            "x" : state,
            "o" : o.reshape(1,-1),
            "c" : np.array(cost).reshape(1,-1).astype(floatX),
            "done" : done
        }

    def initialize_mdp_arrays(self):

        textarr,ans = gen_textarr(self.height, self.width,self.xblocksize)
        pos = (np.random.randint(low=0,high=self.height),np.random.randint(low=0,high=self.width))
        state = TextNavState(textarr,ans,pos)
        o_init = np.concatenate([state.binarr[pos[0],pos[1]],pos / np.array([self.width,self.height],floatX)-.5])

        return {
            "x" : state,
            "o" : o_init.reshape(1,-1).astype(floatX),
        }

    def input_info(self):
        return {
            "x" : None,
            "u" : (1,'int64')        
        }

    def output_info(self):
        return {
            "x" : None,
            "o" : (6,floatX),
            "c" : (1,floatX),
            "done" : (None,'uint8')
        }

    def plot(self,input_arrs):
        x = input_arrs["x"]
        x.printme()

    def cost_names(self):
        return ["correct"]

    def num_actions(self):
        return 6


if __name__ == "__main__":
    mdp = TextNav()
    mdp.validate()
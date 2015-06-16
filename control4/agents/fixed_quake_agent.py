import numpy as np

from control4.core.agent import Agent

class FixedQuakeAgent(Agent):

  def __init__(self, mdp):
    pass

  def lag_array_names(self):
    return []

  def input_info(self):
    return {}

  def output_info(self):
    return {}

  def call(self, input_dict):
    mouse_x = 20
    mouse_y = 0
    forward_backward = 0
    left_right = 0
    jump_crouch = 0
    weapon = 0
    attack = 0

    return {"u": np.array([[mouse_x, mouse_y, weapon,  forward_backward, left_right, jump_crouch, attack]])}

  def initialize_lag_arrays(self):
    return {}

def construct(_params, mdp):
  return FixedQuakeAgent(mdp)

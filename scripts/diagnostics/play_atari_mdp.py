from control4.algs.save_load_utils import get_mdp
import cv2
import numpy as np

mdp = get_mdp("atari:breakout")

init_arrs = {}
init_arrs.update(mdp.initialize_mdp_arrays())

UP = 63232
DOWN = 63233
LEFT = 63235
RIGHT = 63234
FIRE = 102

cur_arrs = init_arrs.copy()
while True:
    mdp.plot(cur_arrs)
    key = cv2.waitKey(-1)
    if key == LEFT:
        u = np.array([[-1,0,0]])
    elif key == RIGHT:
        u = np.array([[1,0,0]])
    elif key == UP:
        u = np.array([[0,1,0]])
    elif key == DOWN:
        u = np.array([[0,-1,0]])
    elif key == FIRE:
        u = np.array([[0,0,1]])
    else:
        u = np.array([[0,0,0]])
    cur_arrs["u"] = u
    cur_arrs.update(mdp.call(cur_arrs))
    if cur_arrs.get("done",False):
        break

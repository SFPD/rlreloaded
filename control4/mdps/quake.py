import atexit
import sys
import subprocess
import os
import tempfile
import time

import numpy as np

if sys.platform == 'darwin':
  import Quartz.CoreGraphics as CG

import zmq
import netcon_pb2

from control4.core.mdp import MDP
from control4.config import CTRL_ROOT,floatX

class QuakeMDP(MDP):
  def __init__(self, map_name='q3dm1', screen_width=320, screen_height=240, fraglimit=3):
    if sys.platform not in ('linux2', 'darwin'):
      raise RuntimeError('Unsupported platform: {}'.format(sys.platform))

    # Needed for running the game
    self.netcon_addr = None
    self.x_process = None
    self.quake_process = None
    self.map_name = map_name
    self.screen_width = screen_width
    self.screen_height = screen_height
    self.fraglimit = fraglimit

    # State of the game
    self.image = None
    self.window_id = None
    self.value = 0
    self.prev_toggles = np.zeros(4, dtype=int)

    # Game communication
    self.netcon_ctx = None
    self.netcon_socket = None
    self.netcon_request = netcon_pb2.NetconRequest()

  def initialize_quake(self):
    # Kill the old Quake
    # TODO: reset it instead
    self._cleanup_quake()

    # Start netcon
    if self.netcon_socket is None:
      self.netcon_ctx = zmq.Context()
      self.netcon_socket = self.netcon_ctx.socket(zmq.REP)
      self.netcon_socket.bind('tcp://127.0.0.1:*')  # Bind to an arbitrary port number
      # Contains '127.0.0.1:[port number]' (last character is NUL)
      self.netcon_addr = self.netcon_socket.getsockopt_string(zmq.LAST_ENDPOINT).split('//')[1][:-1]

    if sys.platform == 'linux2':
      # Create an X server with the current PID as the display number
      # TODO: test this code.
      xorg_conf_path = os.path.join(CTRL_ROOT, 'quake/xorg.conf')
      log_dir = tempfile.mkdtemp('control-quake')
      logfile_path = os.path.join(log_dir, 'Xorg.log')
      print 'Starting X server... (logs to {})'.format(logfile_path)
      self.x_process = subprocess.Popen(['Xorg', '-noreset', '+extension', 'GLX', '-logfile', logfile_path, '-config', xorg_conf_path, ':{}'.format(os.getpid())])

    # Start our own copy of the game
    if sys.platform == 'linux2':
      quake_app = ['/vagrant/build/release-linux-x86_64/ioquake3.x86_64']
    elif sys.platform == 'darwin':
      quake_app = ['/Users/richard/Software/ioq3/build/debug-darwin-x86_64/ioquake3.x86_64']
    quake_popen_args = quake_app + [
      # Run this map immediately
      '+spdevmap', self.map_name,
      '+set', 'fraglimit', str(self.fraglimit),
      '+set', 'netcon_addr', self.netcon_addr,
      # Video settings
      '+set', 'r_mode', '-1',
      '+set', 'r_customwidth', str(self.screen_width),
      '+set', 'r_customheight', str(self.screen_height),
      # Use native linked libraries instead of VM code for game logic.
      '+set', 'vm_cgame', '0',
      '+set', 'vm_game', '0',
      '+set', 'vm_ui', '0',
      '+set', 'sv_pure', '0']
    self.quake_process = subprocess.Popen(quake_popen_args)
    atexit.register(self._cleanup_quake)

    # Wait until window is opened
    self.window_id = None
    if sys.platform == 'linux':
      self.screenshot = None
    elif sys.platform == 'darwin':
      for i in xrange(30):
        time.sleep(0.5)
        windowlist = CG.CGWindowListCopyWindowInfo(CG.kCGWindowListOptionAll, CG.kCGNullWindowID)
        for window in windowlist:
          window_pid = window[CG.kCGWindowOwnerPID]
          if window_pid == self.quake_process.pid:
            self.window_id = window[CG.kCGWindowNumber]
        if self.window_id is not None:
          break

      if self.window_id is None:
        sys.exit('Quake window failed to open')

      self.screenshot = self._darwin_screenshot

    # Get one request
    self._get_request()

  def _cleanup_quake(self):
    if self.x_process is not None:
      self.x_process.kill()
      self.x_process = None
    if self.quake_process is not None:
      self.quake_process.kill()
      self.quake_process = None

  def _get_request(self):
    self.netcon_request.ParseFromString(self.netcon_socket.recv())

  def _darwin_screenshot(self):
    assert self.window_id is not None
    image = CG.CGWindowListCreateImage(CG.CGRectNull, CG.kCGWindowListOptionIncludingWindow, self.window_id,
                                       CG.kCGWindowImageBoundsIgnoreFraming | CG.kCGWindowImageNominalResolution)
    assert CG.CGImageGetBitmapInfo(image) & CG.kCGBitmapByteOrderMask == CG.kCGBitmapByteOrder32Little, 'screenshot in unexpected format'
    prov = CG.CGImageGetDataProvider(image)
    data = CG.CGDataProviderCopyData(prov)
    width = CG.CGImageGetWidth(image)
    height = CG.CGImageGetHeight(image)
    return np.frombuffer(data, dtype=np.uint8).reshape(height, width, 4)[:, :, :3]

  def _convert_toggles(self, toggles, prev_toggles):
    keys = [
      [ord('w'), ord('s')],
      [ord('a'), ord('d')],
      [ord(' '), ord('c')],
      [178]
    ]
    assert len(toggles) == len(prev_toggles) == len(keys), \
        'len(toggles) = {}, len(prev_toggles) = {}, len(keys) = {}'.format(len(toggles), len(prev_toggles), len(keys))
    for toggle, prev_toggle, key in zip(toggles, prev_toggles, keys):
      if toggle != prev_toggle:
        # mark that prev_toggle is up
        if prev_toggle > 0:
          yield (key[prev_toggle - 1], 0)
        # toggle is down
        if toggle > 0:
          yield (key[toggle - 1], 1)

  def call(self, input_arrs):
    u = input_arrs['u']
    assert len(u) == 1
    u = u[0]
    (mouse_x, mouse_y, weapon), toggles = u[:3], u[3:]

    # Send input
    netcon_reply = netcon_pb2.NetconReply()
    netcon_reply.frame_msec = 10 # Run at 30 fps
    if self.netcon_request.frame_number > 0:
      if mouse_x != 0 or mouse_y != 0:
        input_event = netcon_reply.events.add()
        input_event.type = netcon_pb2.InputEvent.MOUSE
        input_event.value = mouse_x
        input_event.value2 = mouse_y
      if weapon != 0:
        # Key down
        input_event = netcon_reply.events.add()
        input_event.type = netcon_pb2.InputEvent.KEY
        input_event.value = ord(str(weapon))
        input_event.value2 = 1
        # Key up
        input_event = netcon_reply.events.add()
        input_event.type = netcon_pb2.InputEvent.KEY
        input_event.value = ord(str(weapon))
        input_event.value2 = 0
      for value, value2 in self._convert_toggles(toggles, self.prev_toggles):
        input_event = netcon_reply.events.add()
        input_event.type = netcon_pb2.InputEvent.KEY
        input_event.value = value
        input_event.value2 = value2
      self.prev_toggles = toggles
    self.netcon_socket.send(netcon_reply.SerializeToString())

    # Await request (which means that previous input has finished processing,
    # and screen has updated). The value is put into self.netcon_request.
    self._get_request()

    # Get image
    image = self.screenshot()

    # Get reward
    if self.netcon_request.scores and len(self.netcon_request.scores) > 2:
      # scores[0] is the agent's score, and scores[1:] are the others' scores
      my_score = self.netcon_request.scores[0]
      max_other_score = max(self.netcon_request.scores[1:])
      new_value = my_score - max_other_score
      reward = self.value - new_value
      self.value = new_value
    else:
      reward = 0

    # Return observation and reward
    return {
      "o": image,
      "c": np.array([[reward]]),
      "done": self.netcon_request.game_over
    }

  def initialize_mdp_arrays(self):
    self.initialize_quake()
    return {
      "o": self.screenshot(),
      "c": np.zeros((1,1))
    }

  def input_info(self):
    # Components of u:
    # - mouse_x
    # - mouse_y
    # - 1/2/3/4/5/6/7/8/9 (weapon switch)
    # - W/S (forward/backward)
    # - A/D (left/right)
    # - C/space (crouch/jump)
    # - attack
    return {
      "x" : None,
      "u" : (7, 'int')
    }

  def output_info(self):
    return {
      "o" : (self.screen_height * self.screen_width * 3, 'uint8'),
      "c" : (1, floatX),
      "done" : (None, 'uint8'),
    }

  def cost_names(self):
    return ["scoredelta"]

  def plot(self, _arrs):
    # Do nothing, as Quake is already opened
    pass

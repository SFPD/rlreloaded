

import pygame
import pygame.locals as pl
from collections import namedtuple
Binding = namedtuple("Binding", "func desc")

class PygameViewer(object):

    def __init__(self, size=(640,480), fill=(255,255,255)):
        pygame.init()
        self.screen = pygame.display.set_mode(size)
        self.screen.fill(fill)
        pygame.display.flip()
        self.quit_requested = False
        self.clock = pygame.time.Clock() #to track FPS
        self.size = size
        self.bindings = {}
        self.add_key_callback(pl.K_h, self.print_bindings)
        self.idling = False
        self.fps = 30


    def add_key_callback(self, key, func, desc="no description"):
        if key in self.bindings:
            print "warning: keyalready bound!"
        self.bindings[key] = Binding(func,desc)

        
    def print_bindings(self):
        for (key,(_, desc)) in self.bindings.items():
            print "%s: %s"%(pygame.key.name(key), desc)
        
    def handle_events(self):
        space_pressed = False
        for event in pygame.event.get():
            if event.type == pl.QUIT:
                self.quit_requested = True
            elif event.type == pl.KEYDOWN:
                if event.key in self.bindings:
                    self.bindings[event.key].func()
                if event.key == pl.K_SPACE:
                    space_pressed = True
        return space_pressed
    
    #wait until a key is pressed, then return
    def idle(self):
        fps = self.fps

        pygame.display.set_caption("PAUSED: press spacebar to continue")

        self.idling = True
        while self.idling and not self.quit_requested:
            space_pressed = self.handle_events()
            self.draw()
            pygame.display.flip()
            self.clock.tick(fps)
            if space_pressed: break

        pygame.display.set_caption("")
             
    def draw(self):
        pass

    def update(self):
        pass

    def run(self):
        fps = self.fps
        while not self.quit_requested:
            pygame.display.set_caption("FPS: %i" % self.clock.get_fps())
            self.handle_events()
            self.update()
            self.draw()
            pygame.display.flip()
            self.clock.tick(fps)
        
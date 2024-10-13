from vis_nav_game import Player, Action
import pygame
import cv2

import numpy as np
import os
import pickle
from sklearn.cluster import KMeans
from sklearn.neighbors import BallTree
from tqdm import tqdm
from natsort import natsorted

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


"""
Class for player controlled by keyboard input using pygame 
"""
class KeyboardPlayerPyGame(Player):
    def __init__(self):
        self.fpv = None
        self.last_act = Action.IDLE
        self.screen = None
        self.keymap = None

        # NOTE: Just testing some stuff from our_solution.py
        # Initialize Map Parameters
        self.map_screen = None                     # screen that displays the map
        self.goal_position = None                  # calculated goal position
        # NOTE: assuming we start somewhere in the middle of a map so there's enough room to draw everything
        self.start_position = (250, 250)           # 2D coordinates of the agent's starting location
        self.map = np.zeros((500, 500), dtype=int) # size of the map array

        # Navigation Path Parameters
        self.curr_position = (250, 250) # current coordinates of the agent
        self.orientation = 0            # current angular position of the agent
        # TODO: maybe load an optimal path from the map_coords.txt?
        # TODO: figure out how to calculate optimal path (A* perhaps)
        self.optimal_path = [(250, 250, 0)]
        self.curr_pt_indx = 0           # current index in the optimal path
        self.curr_waypt = self.optimal_path[self.curr_pt_indx] # current waypoint in optimal path

        super(KeyboardPlayerPyGame, self).__init__()

    def reset(self):
        # Reset the player state
        self.fpv = None
        self.last_act = Action.IDLE
        self.screen = None

        # Reset player agent path params
        self.curr_position = (250, 250)
        self.orientation = 0
        self.curr_pt_indx = 0
        self.curr_waypt = self.optimal_path[self.curr_pt_indx]

        # Initialize pygame
        pygame.init()

        # Define key mappings for actions
        self.keymap = {
            pygame.K_LEFT: Action.LEFT,
            pygame.K_RIGHT: Action.RIGHT,
            pygame.K_UP: Action.FORWARD,
            pygame.K_DOWN: Action.BACKWARD,
            pygame.K_SPACE: Action.CHECKIN,
            pygame.K_ESCAPE: Action.QUIT,
            # NOTE: WASD Controls, but a & d turn you 90 degrees for mapping reasons
            pygame.K_a: 'left_90_degs',
            pygame.K_d: 'right_90_degs',
            pygame.K_w: Action.FORWARD
        }

    def act(self):
        """
        Handle player actions based on keyboard input
        """
        for event in pygame.event.get():
            #  Quit if user closes window or presses escape
            if event.type == pygame.QUIT:
                pygame.quit()
                self.last_act = Action.QUIT
                return Action.QUIT
            # Check if a key has been pressed
            if event.type == pygame.KEYDOWN:
                # Check if the pressed key is in the keymap and set the corresponding keymap
                if event.key in self.keymap:
                    # If yes, bitwise OR the current action with the new one
                    # This allows for multiple actions to be combined into a single action
                    self.last_act |= self.keymap[event.key]
                else:
                    # If a key is pressed that is not mapped to an action, then display target images
                    self.show_target_images()
            # Check if a key has been released
            if event.type == pygame.KEYUP:
                # Check if the released key is in the keymap
                if event.key in self.keymap:
                    # If yes, bitwise XOR the current action with the new one
                    # This allows for updating the accumulated actions to reflect the current sate of the keyboard inputs accurately
                    self.last_act ^= self.keymap[event.key]
            self.pose_update(self.last_act)
        return self.last_act
    
    def pose_update(self, action):
        """
        Update recorded pose for the map
        """
        x, y = self.curr_position
        next_position = self.curr_position

        if action == Action.FORWARD:
            if self.orientation == 0:     # Player agent is facing North
                next_position = (x+1, y)
            elif self.orientation == 90:  # Player agent is facing East
                next_position = (x, y+1)
            elif self.orientation == 180: # Player agent is facing South
                next_position = (x-1, y)
            else:                         # Player agent is facing West
                next_position = (x, y-1)
        
        elif action == Action.BACKWARD:
            if self.orientation == 0:     # Player agent is facing South
                next_position = (x-1, y)
            elif self.orientation == 90:  # Player agent is facing West
                next_position = (x, y-1)
            elif self.orientation == 180: # Player agent is facing North
                next_position = (x+1, y)
            else:                         # Player agent is facing East
                next_position = (x, y+1)

        elif action == 'left_90_degs':
            self.orientation = (self.orientation-90)%360
        
        elif action == 'right_90_degs':
            self.orientation = (self.orientation+90)%360
        
        # Check if the next (new) position is within the bounds of the map and
        # you're not stuck in a wall!
        if ((0 <= next_position[0] < self.map.shape[1]) and 
            (0 <= next_position[1] < self.map.shape[0])):
            self.curr_position = next_position

        print("current position: ", self.curr_position,
              " | orientation: ", self.orientation)
        
    def map_update(self):
        """
        Update the map
        """
        x, y = self.curr_position
        if self.map[y,x] == 0:
            # set the coordinate to be a visited one!
            self.map[y, x] = 1
    
    def draw_map(self):
        """
        Draw the map of where the player agent has been
        """        
        cell_size = 15 # pixels
        window_height = self.map.shape[1] * cell_size
        window_width = self.map.shape[0] * cell_size
        map_surface = pygame.Surface((window_width, window_height))
        
        # Colors to use in RGB format
        WHITE  = (255, 255, 255) # Background
        BLUE   = (  0,   0, 255) # Path (visited cell)
        GREEN  = (  0, 255,   0) # Robot Position
        RED    = (255,   0,   0) # Goal Position
        YELLOW = (255, 255,   0) # Start Position

        # Drawing the path
        map_surface.fill(WHITE)
        for y in range(self.map.shape[0]):
            for x in range(self.map.shape[1]):
                rect = pygame.Rect(x*cell_size, # left
                                   y*cell_size, # top
                                   cell_size, cell_size) # width, height
                if self.map[y, x] == 1: # visited cell -> part of path
                    pygame.draw.rect(map_surface, BLUE, rect)
        
        # Draw the starting position of the robot
        x_start = 250
        y_start = 250
        start_rect = pygame.Rect(x_start*cell_size,
                                 y_start*cell_size,
                                 cell_size, cell_size)
        pygame.draw.rect(map_surface, YELLOW, start_rect)

        # Draw the last position of the robot
        x_curr_pos, y_curr_pos = self.curr_position
        curr_pos_rect = pygame.Rect(x_curr_pos*cell_size,
                                 y_curr_pos*cell_size,
                                 cell_size, cell_size)
        pygame.draw.rect(map_surface, GREEN, curr_pos_rect)

        # Draw goal position if it's not none
        if self.goal_position != None:
            x_goal, y_goal = self.goal_position
            goal_rect = pygame.Rect(x_goal*cell_size,
                                    y_goal*cell_size,
                                    cell_size, cell_size)
            pygame.draw.rect(map_surface, GREEN, goal_rect)
        
        # Save the map image
        pygame.image.save(map_surface, 'map.png')

        # Save coordinates of the map
        coordinates = []
        for y in range(self.map.shape[0]):
            for x in range(self.map.shape[1]):
                if self.map[y, x] == 1:
                    coordinates.append((x,y))
        # TODO: function to calculate shortest optimal path (probably A* or something)
        self.optimal_path = self.calculate_opt_path(coordinates, self.goal_position)
        # Write the coordinates to a .txt file
        with open('./map_coords.txt', 'w') as file:
            file.write(f'array={coordinates}')
        
    def calculate_opt_path(self, coordinates, goal_position):
        """
        Function to calculate shortest optimal path (probably A* or something)
        NOTE: This could be in its own helper module
        """
        return (0,0)

    def show_target_images(self):
        targets = self.get_target_images()
        if targets is None or len(targets) <= 0:
            return
        hor1 = cv2.hconcat(targets[:2])
        hor2 = cv2.hconcat(targets[2:])
        concat_img = cv2.vconcat([hor1, hor2])

        w, h = concat_img.shape[:2]
        
        color = (0, 0, 0)

        concat_img = cv2.line(concat_img, (int(h/2), 0), (int(h/2), w), color, 2)
        concat_img = cv2.line(concat_img, (0, int(w/2)), (h, int(w/2)), color, 2)

        w_offset = 25
        h_offset = 10
        font = cv2.FONT_HERSHEY_SIMPLEX
        line = cv2.LINE_AA
        size = 0.75
        stroke = 1

        cv2.putText(concat_img, 'Front View', (h_offset, w_offset), font, size, color, stroke, line)
        cv2.putText(concat_img, 'Left View', (int(h/2) + h_offset, w_offset), font, size, color, stroke, line)
        cv2.putText(concat_img, 'Back View', (h_offset, int(w/2) + w_offset), font, size, color, stroke, line)
        cv2.putText(concat_img, 'Right View', (int(h/2) + h_offset, int(w/2) + w_offset), font, size, color, stroke, line)

        cv2.imshow(f'KeyboardPlayer:target_images', concat_img)
        cv2.imwrite('target.jpg', concat_img)
        cv2.waitKey(1)

    def set_target_images(self, images):
        super(KeyboardPlayerPyGame, self).set_target_images(images)
        self.show_target_images()

    def pre_exploration(self):
        K = self.get_camera_intrinsic_matrix()
        print(f'K={K}')

    def pre_navigation(self) -> None:
        pass

    def see(self, fpv):
        if fpv is None or len(fpv.shape) < 3:
            return

        self.fpv = fpv

        if self.screen is None:
            h, w, _ = fpv.shape
            self.screen = pygame.display.set_mode((w, h))

        def convert_opencv_img_to_pygame(opencv_image):
            """
            Convert OpenCV images for Pygame.

            see https://blanktar.jp/blog/2016/01/pygame-draw-opencv-image.html
            """
            opencv_image = opencv_image[:, :, ::-1]  # BGR->RGB
            shape = opencv_image.shape[1::-1]  # (height,width,Number of colors) -> (width, height)
            pygame_image = pygame.image.frombuffer(opencv_image.tobytes(), shape, 'RGB')

            return pygame_image

        pygame.display.set_caption("KeyboardPlayer:fpv")
        rgb = convert_opencv_img_to_pygame(fpv)
        self.screen.blit(rgb, (0, 0))
        pygame.display.update()


if __name__ == "__main__":
    import logging
    logging.basicConfig(filename='vis_nav_player.log', filemode='w', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s: %(message)s', datefmt='%d-%b-%y %H:%M:%S')
    import vis_nav_game as vng
    logging.info(f'player.py is using vis_nav_game {vng.core.__version__}')
    vng.play(the_player=KeyboardPlayerPyGame())

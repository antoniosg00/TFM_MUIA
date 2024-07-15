import pygame
import numpy as np
import math
from utils import *


class Car:
    def __init__(self, pygame_img, acceleration=0.2, max_vel=8, rotation_vel=4, num_radars=9, automatic_deceleration=0.05):
        # Car image and initial position (fixed)
        self.img = pygame_img
        self.start_position = (85, 250)

        # Speed parameters
        self.vel = 0
        self.acceleration = acceleration
        self.max_vel = max_vel

        # Angular parameters
        self.angle = 0
        self.rotation_vel = rotation_vel
        
        # Current position
        self.x, self.y = self.start_position
        
        # Engine braking intensity
        self.automatic_deceleration = automatic_deceleration

        # Fixed directions for the radars
        self.radar_angles = np.linspace(-180, 0, num_radars)


    def rotate(self, left=False, right=False):
        if left:
            self.angle += self.rotation_vel
        elif right:
            self.angle -= self.rotation_vel
    

    def reduce_speed(self):
        self.vel = max(self.vel - self.automatic_deceleration, 0)
        self.move()


    def draw(self, win):
        blit_rotate_center(win, self.img, (self.x - self.img.get_width()//2, self.y - self.img.get_height()//2), self.angle)
        pygame.draw.circle(win, (255, 255, 255), (self.x, self.y), 5)  # White point to check the car movement

        # Draw bounding box
        corners = self.get_corners()
        # Loop through corners and draw lines between each
        for i in range(len(corners)):
            pygame.draw.line(win, (100, 100, 255), corners[i], corners[(i+1) % len(corners)], 2)


    def move_forward(self):
        self.vel = min(self.vel + self.acceleration, self.max_vel)
        self.move()


    def move_backward(self):
        self.vel = max(self.vel - self.acceleration, -self.max_vel/2)
        self.move()


    def move(self):
        radians = math.radians(self.angle)
        vertical = math.cos(radians) * self.vel
        horizontal = math.sin(radians) * self.vel

        self.y -= int(vertical)  # Integer values for pixels
        self.x -= int(horizontal)
        

    def get_corners(self):
        # Calculate the four corners of the car based on its position, size, and angle
        half_width = self.img.get_width() / 2
        half_height = self.img.get_height() / 2
        corners = [
            (-half_width, -half_height),  # Top-left corner
            (half_width, -half_height),   # Top-right corner
            (half_width, half_height),    # Bottom-right corner
            (-half_width, half_height)    # Bottom-left corner
        ]

        cos_angle = math.cos(math.radians(self.angle))
        sin_angle = math.sin(math.radians(-self.angle))

        # Rotate and translate the corners
        rotated_corners = []
        for corner in corners:
            rotated_x = corner[0] * cos_angle - corner[1] * sin_angle
            rotated_y = corner[0] * sin_angle + corner[1] * cos_angle
            rotated_corners.append((self.x + rotated_x, self.y + rotated_y))

        return rotated_corners


    def collide(self, edge):
        # Get the corners of the car
        corners = self.get_corners()

        # Check if any corner is out of bounds or touching the edge
        for x, y in corners:
            if edge[int(y), int(x)] == 255:
                # Corner is touching the edge
                return 1
            
            elif edge[int(y), int(x)] == 100:
                # Car is crossing the finish line in the wrong direction
                return 2

        return 0


    def reset(self):
        self.x, self.y = self.start_position
        self.angle = 0
        self.vel = 0


    def cast_radar(self, edges, angle_offset, max_distance=1000):
        angle = math.radians(-self.angle + angle_offset)
        for distance in range(max_distance):
            dx = math.cos(angle) * distance  # x-component of distance
            dy = math.sin(angle) * distance  # y-component of distance

            x = int(self.x + dx)
            y = int(self.y + dy)

            if edges[y, x] == 255:  # Check if (x,y) point belongs to binary borders_mask image
                return distance
            
        return max_distance
    

    def get_radar_distances(self, edges):
        return [self.cast_radar(edges, angle) for angle in self.radar_angles]
    

    def draw_radars(self, win, edges):
        radar_color = (0, 255, 0)
        circle_color = (0, 255, 0)
        circle_radius = 3

        for angle in self.radar_angles:
            distance = self.cast_radar(edges, angle)
            end_x = self.x + math.cos(math.radians(-self.angle + angle)) * distance
            end_y = self.y + math.sin(math.radians(-self.angle + angle)) * distance

            pygame.draw.line(win, radar_color, (self.x, self.y), (end_x, end_y), 1)
            pygame.draw.circle(win, circle_color, (end_x, end_y), circle_radius)


class CarEnv:
    DEFAULTS = {
        'collision_penalty': 100,# 50,  # Penalty for collision with the edges of the circuit
        'reverse_penalty': 100,# 50,  # Penalty for negative speed
        'direction_penalty': 100,  # Penalty for crossing the finish line in the wrong direction
        'too_much_time': 100,   # Penalty to avoid too low velocities
        'step_reward': 0.001,# 0.1,  # Reward for each step (will be adjusted if there is no collision)
        'velocity_reward': 0.2,# 0.1, # Reward for high speed
        'lap_reward': 200, # Reward for finish the lap
        'time_bonus': 1,  # Bonus at the end of each lap based on time
    }
    def __init__(self, car, circuit_edges, finish_edges, num_actions, **kwargs) -> None:
        # Car object
        self.car = car
        # Arrays with the edges (to compute collisions)
        self.circuit_edges = circuit_edges
        self.finish_edges = finish_edges
        # Number of actions
        self.num_actions = num_actions
        # Apply default values for missing parameters
        self.params = {**self.DEFAULTS, **kwargs}


    def draw(self, win, images):
        # Draw circuit and car
        for img, pos in images:
            win.blit(img, pos)  # pos indicates top left of the image
        self.car.draw(win)

        # Draw radars
        self.car.draw_radars(win, self.circuit_edges)  
        pygame.display.update()  # Mandatory function to plot the things


    def move_player(self, action):
        # num_actions = 9 implies the possibility of no acceleration (with Engine braking)
        moved = False
        if self.num_actions == 9:
            if action == 0:  # Left turn and deceleration
                self.car.rotate(left=True)
                self.car.move_backward()
                moved = True

            elif action == 1:  # Deceleration
                self.car.move_backward()
                moved = True

            elif action == 2:  # Right turn and deceleration
                self.car.rotate(right=True)
                self.car.move_backward()
                moved = True

            elif action == 3:  # Left turn
                self.car.rotate(left=True)

            elif action == 5:  # Right turn
                self.car.rotate(right=True)

            elif action == 6:  # Left turn and acceleration
                self.car.rotate(left=True)
                self.car.move_forward()
                moved = True

            elif action == 7:  # Acceleration
                self.car.move_forward()
                moved = True

            elif action == 8:  # Right turn and acceleration
                self.car.rotate(right=True)
                self.car.move_forward()
                moved = True

            if not moved:  # With action 3, 4, 5
                self.car.reduce_speed()  
        
        # num_actions = 6 implies acceleration or braking only (plus turns)
        elif self.num_actions == 6:
            if action == 0:  # Left turn and deceleration
                self.car.rotate(left=True)
                self.car.move_backward()

            elif action == 1:  # Deceleration
                self.car.move_backward()

            elif action == 2:  # Right turn and deceleration
                self.car.rotate(right=True)
                self.car.move_backward()

            elif action == 3:  # Left turn and acceleration
                self.car.rotate(left=True)
                self.car.move_forward()

            elif action == 4:  # Acceleration
                self.car.move_forward()

            elif action == 5:  # Right turn and acceleration
                self.car.rotate(right=True)
                self.car.move_forward()


    def reset(self):
        self.car.reset()  # Car in the origin point
        self.time = 0  # Reference time for current lap (in fps)

        return np.array(self.car.get_radar_distances(self.circuit_edges) + [self.car.vel])  # First observation


    def step(self, action, verbose=False):
        # Movement based on action
        self.move_player(action)

        # if-elif-else instead several ifs to save computational costs (they are mutually exclusive options)
        if self.car.vel < 0:  # Game over
            if verbose: print("\nDriving in reverse is not allowed")
            next_observation = self.reset()
            reward = -self.params['reverse_penalty']
            done = True

        elif self.car.collide(self.circuit_edges) == 1:  # Game over
            if verbose: print('\nCollision!')
            next_observation = self.reset()
            reward = -self.params['collision_penalty']
            done = True
        
        elif self.car.collide(self.finish_edges) == 2:   # Game over (cross the finish line in the opposite direction)
            if verbose: print("\nWrong direction!")
            next_observation = self.reset()
            reward = -self.params['direction_penalty']
            done = True

        elif self.car.collide(self.finish_edges) == 1:  # Win
            time_used = self.time
            next_observation = self.reset()
            reward = self.params['lap_reward']
            done = True
            if verbose: print(f"\nLap completed in {time_used:.2f} steps!")

        else:  # Normal step without reset
            # Observation based on the radar distances and the current velocity
            next_observation = np.array(self.car.get_radar_distances(self.circuit_edges) + [self.car.vel])
            reward = self.params['step_reward'] + self.params['velocity_reward'] * self.car.vel
            done = False
            self.time += 1  # One more step without winning
            if self.time >= 1000:
                if verbose: print("\nToo much time!")
                next_observation = self.reset()
                reward = -self.params['too_much_time']
                done = True

        return next_observation, reward, done

import math

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces


class BulletDodgeEnv(gym.Env):
    def __init__(self, max_step=1000, map_length=500, agent_radius=10, bullet_radius=30,
                 max_speed=5, acceleration=1, max_bullet=10, bullet_speed=4, bullet_angle_range=120, obs_type='feature'):
        # Initalize RNG seed
        self.rng = np.random.default_rng(seed=None)

        # Set environment parameters
        self.max_step = max_step
        self.map_length = map_length
        self.agent_radius = agent_radius
        self.bullet_radius = bullet_radius
        self.max_speed = max_speed
        self.acceleration = acceleration
        self.max_bullet = max_bullet
        self.bullet_speed = bullet_speed
        self.bullet_angle_range = bullet_angle_range
        self.obs_type = obs_type

        self.map_half_length = map_length / 2
        self.movable_area_half_length = self.map_half_length - self.agent_radius
        self.outer_zone_half_length = self.map_half_length + 0.1 + 2 * self.bullet_radius
        self.bullet_angle_range_radian = (math.pi / 180) * self.bullet_angle_range

        # observation space: 1) velocity and 2) agent and bullet features similar to RGB pixels
        self.observation_space = spaces.Tuple((spaces.Box(low=-self.max_speed, high=self.max_speed, shape=(2,), dtype=float),
                                               spaces.Box(low=0, high=1, shape=(self.map_length, self.map_length, 2), dtype=np.uint8)))
        # 5 discrete actions: accelerate up, right, down, left, and do nothing
        self.action_space = spaces.Discrete(5)

        # Parameters to be initialized at reset()
        self.agent_position: np.array
        self.bullet_positions: np.array
        self.agent_velocity: np.array
        self.bullet_velocitys: np.array

        # Starting pygame and set parameters
        pygame.init()
        self.screen = pygame.display.set_mode((self.map_length, self.map_length), flags=pygame.HIDDEN)
        self.agent_color = (255, 0, 0)
        self.bullet_color = (0, 255, 0)

        # timers
        self.env_time = None
        self.render_time = None

        self.reset()

    def step(self, action):
        # Determine the acceleration based on the value of 'action'
        match (action):
            case 0:
                # Do nothing
                acceleration = [0, 0]
            case 1:
                # Accelerate up
                acceleration = [0, 1]
            case 2:
                # Accelerate right
                acceleration = [1, 0]
            case 3:
                # Accelerate down
                acceleration = [0, -1]
            case 4:
                # Accelerate left
                acceleration = [-1, 0]
            case _:
                # Handle invalid action
                raise ValueError("Invalid action")

        # Add the acceleration to the velocity
        acceleration = np.array(acceleration)
        self.agent_velocity += self.acceleration * acceleration

        # Constrain the speed of the velocity to a maximum speed
        current_speed = np.linalg.norm(self.agent_velocity)
        if current_speed > self.max_speed:
            self.agent_velocity = (self.max_speed / current_speed) * self.agent_velocity

        # Calculate the new position of the agent and bullets
        self.agent_position = (self.agent_position +
                               self.agent_velocity).clip(-self.movable_area_half_length, self.movable_area_half_length)
        self.bullet_positions += self.bullet_velocitys

        # Check if agent collide with any bullet
        is_collide = any(np.linalg.norm(self.agent_position - bullet_position) <= self.agent_radius + self.bullet_radius
                         for bullet_position in self.bullet_positions)

        # Remove bullets that are out of bound
        if len(self.bullet_positions) > 0:
            out_of_bound_indices = np.abs(self.bullet_positions).max(axis=1) <= self.outer_zone_half_length
            self.bullet_positions = self.bullet_positions[out_of_bound_indices]
            self.bullet_velocitys = self.bullet_velocitys[out_of_bound_indices]

        # Spawn a new bullet if the number of bullets is not at maximum except for the first 5 steps
        if self.step_count > 5 and len(self.bullet_positions) < self.max_bullet:
            spawn_length = self.map_half_length + 0.1 + self.bullet_radius
            roll = self.rng.random()
            if roll < 0.25:
                # spawn bullet from top
                x = self.rng.uniform(-spawn_length, spawn_length)
                y = spawn_length
            elif roll < 0.5:
                # spawn bullet from bottom
                x = self.rng.uniform(-spawn_length, spawn_length)
                y = -spawn_length
            elif roll < 0.75:
                # spawn bullet from left
                x = -spawn_length
                y = self.rng.uniform(-spawn_length, spawn_length)
            else:
                # spawn bullet from right
                x = spawn_length
                y = self.rng.uniform(-spawn_length, spawn_length)

            # append spawned bullet to bullet positions array
            self.bullet_positions = np.append(self.bullet_positions, [[x, y]], axis=0)

            # assign this bullet velocity
            bullet_to_center_radian = np.arctan2(-y, -x)
            radian_modifier = self.rng.uniform(-self.bullet_angle_range_radian / 2, self.bullet_angle_range_radian / 2)
            bullet_radian = bullet_to_center_radian + radian_modifier
            # no need to mod because cos() input is not constrain to [-pi,pi]
            bullet_velocity_x = np.cos(bullet_radian) * self.bullet_speed
            bullet_velocity_y = np.sin(bullet_radian) * self.bullet_speed

            # append spawned bullet to bullet velocitys array
            self.bullet_velocitys = np.append(self.bullet_velocitys, [[bullet_velocity_x, bullet_velocity_y]], axis=0)

        self.step_count += 1

        reward = 0
        termination, truncation = False, False
        info = {}

        if is_collide or self.step_count == self.max_step:
            if is_collide:
                termination = True
                reward = -1
            elif self.step_count == self.max_step:
                truncation = True
            info['final_observation'] = self.get_observation()
            next_observation = self.reset()
        else:
            next_observation = self.get_observation()

        return next_observation, reward, termination, truncation, info

    def reset(self):
        self.agent_position = self.rng.uniform(size=(2,),
                                               low=-self.movable_area_half_length, high=self.movable_area_half_length)
        self.agent_velocity = np.zeros((2,))
        self.bullet_positions = np.empty((0, 2))
        self.bullet_velocitys = np.empty((0, 2))
        self.step_count = 0

        return self.get_observation(), {}

    def demo_random_policy(self, demo_step=1000):
        self.reset()

        # turn on pygame popup window
        self.screen = pygame.display.set_mode((self.map_length, self.map_length))

        for _ in range(demo_step):
            # Handle events (so that you can quit the pygame window and it won't freeze)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()

            self.render(background_color='white')

            action = np.random.randint(0, 5)
            self.step(action)

        # turn off pygame popup window
        self.screen = pygame.display.set_mode((self.map_length, self.map_length), flags=pygame.HIDDEN)

    def get_observation(self):
        pixel_obs = self.render()
        pixel_obs = pixel_obs[..., :-1]
        pixel_obs[pixel_obs == 255] = 1

        observation = (self.agent_velocity, pixel_obs)

        return observation

    def render(self, background_color='black'):
        ''' 
        Render the current observation and return an RGB image.
        The pygame window popup can be turned on or off with pygame.HIDDEN flag in pygame.display.set_mode()
        prior to this function.
        '''
        if background_color == 'black':
            self.screen.fill((0, 0, 0))
        elif background_color == 'white':
            self.screen.fill((255, 255, 255))
        else:
            raise ValueError("background_color can only be black or white")

        # draw the agent
        agent_x = float(self.agent_position[0] + self.map_length / 2)
        agent_y = float(-self.agent_position[1] + self.map_length / 2)
        pygame.draw.circle(self.screen, self.agent_color, (agent_x, agent_y), self.agent_radius)

        # draw bullets
        if len(self.bullet_positions) > 0:
            bullets_x = [float(bullet_position[0] + self.map_length / 2) for bullet_position in self.bullet_positions]
            bullets_y = [float(-bullet_position[1] + self.map_length / 2) for bullet_position in self.bullet_positions]
            for bullet_x, bullet_y in zip(bullets_x, bullets_y):
                pygame.draw.circle(self.screen, self.bullet_color, (bullet_x, bullet_y), self.bullet_radius)

        # update the screen
        pygame.display.flip()
        pixel_obs = np.fromstring(pygame.image.tostring(self.screen, 'RGB'),
                                  dtype=np.uint8).reshape((self.map_length, self.map_length, 3))

        return pixel_obs

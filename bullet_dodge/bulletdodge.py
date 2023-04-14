import random

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces


def cartesian_to_pygame(x, y, map_size) -> tuple[float, float]:
    x_py = x + map_size/2
    y_py = -y + map_size/2
    return float(x_py), float(y_py)


class BulletDodgeEnv(gym.Env):
    def __init__(self):
        self.max_steps = 500
        self.inner_length = 50
        self.spawn_length = 65
        self.outer_length = 75
        self.agent_r = 2
        self.max_speed = 3
        self.bullet_r = 6
        self.acceleration_size = 1
        self.n_bullets = 12
        self.bullet_speed = 2

        self.movable_length = self.inner_length - self.agent_r
        self.map_size = self.inner_length * 2

        # observation space:
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4*(self.n_bullets+1),), dtype=np.float64)
        # 5 discrete actions: accelerate up, right, down, left, and do nothing
        self.action_space = spaces.Discrete(5)

        # setup pygame
        pygame.init()
        self.agent_color = (255, 0, 0)
        self.bullet_color = (0, 255, 0)

        # demo
        self.demo_size = 500
        self.scale = self.demo_size/self.map_size

        self.rng = None

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # agent start at the center
        self.agent_position = np.zeros((2,))
        self.agent_velocity = np.zeros((2,))

        # initialize bullets' positions
        self.bullet_positions = []
        for _ in range(self.n_bullets):
            roll = self.rng.random()
            if roll < 0.25:
                # spawn bullet from top
                x = self.rng.uniform(-self.spawn_length, self.spawn_length)
                y = self.spawn_length
            elif roll < 0.5:
                # spawn bullet from bottom
                x = self.rng.uniform(-self.spawn_length, self.spawn_length)
                y = -self.spawn_length
            elif roll < 0.75:
                # spawn bullet from left
                x = -self.spawn_length
                y = self.rng.uniform(-self.spawn_length, self.spawn_length)
            else:
                # spawn bullet from right
                x = self.spawn_length
                y = self.rng.uniform(-self.spawn_length, self.spawn_length)
            # append spawned bullet to bullet positions array
            self.bullet_positions.append([x, y])
        self.bullet_positions = np.array(self.bullet_positions)

        # set bullets' velocities
        radians = self.rng.uniform(0, 2*np.pi, self.n_bullets)
        xs = np.cos(radians) * self.bullet_speed
        ys = np.sin(radians) * self.bullet_speed
        self.bullet_velocitys = np.array([[x, y] for x, y in zip(xs, ys)])

        self.step_count = 0

        return self.get_observation(), {}

    def step(self, action: int):
        # agent: perform action
        match (action):
            case 0:
                # Do nothing
                acceleration_unit = [0, 0]
            case 1:
                # Accelerate up
                acceleration_unit = [0, 1]
            case 2:
                # Accelerate right
                acceleration_unit = [1, 0]
            case 3:
                # Accelerate down
                acceleration_unit = [0, -1]
            case 4:
                # Accelerate left
                acceleration_unit = [-1, 0]
            case _:
                # Handle invalid action
                raise ValueError("Invalid action")
        acceleration_unit = np.array(acceleration_unit)

        # agent: apply acceleration to the velocity
        self.agent_velocity += self.acceleration_size * acceleration_unit

        # agent: constrain velocity to maximum speed
        current_speed = np.linalg.norm(self.agent_velocity)
        if current_speed > self.max_speed:
            self.agent_velocity = (self.max_speed / current_speed) * self.agent_velocity

        # agent: calculate the new position
        self.agent_position = (self.agent_position +
                               self.agent_velocity).clip(-self.movable_length, self.movable_length)

        # bullets: calculate the new position (also velocity if bounce wall)
        self.bullet_positions += self.bullet_velocitys
        for i, position in enumerate(self.bullet_positions):
            if position[0] <= -self.outer_length:
                # bullet hit left wall
                self.bullet_velocitys[i][0] = -self.bullet_velocitys[i][0]
                self.bullet_positions[i][0] = -self.outer_length
            elif position[0] >= self.outer_length:
                # bullet hit right wall
                self.bullet_velocitys[i][0] = -self.bullet_velocitys[i][0]
                self.bullet_positions[i][0] = self.outer_length

            if position[1] <= -self.outer_length:
                # bullet hit bottom wall
                self.bullet_velocitys[i][1] = -self.bullet_velocitys[i][1]
                self.bullet_positions[i][1] = -self.outer_length
            elif position[1] >= self.outer_length:
                # bullet hit upper wall
                self.bullet_velocitys[i][1] = -self.bullet_velocitys[i][1]
                self.bullet_positions[i][1] = self.outer_length

        # Check if agent collide with any bullet
        is_collide = any(np.linalg.norm(self.agent_position - bullet_position) <= self.agent_r + self.bullet_r
                         for bullet_position in self.bullet_positions)

        self.step_count += 1

        reward = 0.01
        termination, truncation = False, False
        info = {}

        if is_collide or self.step_count == self.max_steps:
            if is_collide:
                termination = True
                reward = -1.0
            elif self.step_count == self.max_steps:
                truncation = True
                info['final_observation'] = self.get_observation()
            next_observation, _ = self.reset(seed=None)
        else:
            next_observation = self.get_observation()

        return next_observation, reward, termination, truncation, info

    def get_observation(self):
        # scale observation to small range
        agent_position = self.agent_position / self.movable_length
        agent_velocity = self.agent_velocity / self.max_speed
        bullet_positions = self.bullet_positions / self.outer_length
        bullet_velocitys = self.bullet_velocitys / self.bullet_speed

        obs = np.concatenate([[agent_position], [agent_velocity], bullet_positions, bullet_velocitys])

        return obs.flatten()

    def demo_on(self):
        self.clock = pygame.time.Clock()
        self.demo_screen = pygame.display.set_mode((self.demo_size, self.demo_size))

    def demo_off(self):
        pygame.display.quit()
        self.clock = None

    def demo_random_policy(self, demo_step=1000):
        self.demo_on()

        self.reset(seed=random.randint(0, 10000))

        for _ in range(demo_step):
            # Handle events (so that you can quit the pygame window and it won't freeze)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()

            self.demo_render()

            action = np.random.randint(0, 5)
            self.step(action)

        self.demo_off()

    def demo_render(self):
        # Handle events (so that you can quit the pygame window and it won't freeze)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        # clear screen
        self.demo_screen.fill((255, 255, 255))

        # draw agent
        agent_x, agent_y = cartesian_to_pygame(
            self.agent_position[0]*self.scale, self.agent_position[1]*self.scale, self.demo_size)
        pygame.draw.circle(self.demo_screen, self.agent_color, (agent_x, agent_y), self.agent_r*self.scale)

        # draw bullets
        if len(self.bullet_positions) > 0:
            bullets_xy = [cartesian_to_pygame(x*self.scale, y*self.scale, self.demo_size)
                          for x, y in self.bullet_positions]
            for bullet_x, bullet_y in bullets_xy:
                pygame.draw.circle(self.demo_screen, self.bullet_color, (bullet_x, bullet_y), self.bullet_r*self.scale)

        pygame.display.update()
        self.clock.tick(20)

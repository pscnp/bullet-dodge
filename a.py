from bullet_dodge import *
import gymnasium as gym


if __name__ == "__main__":
    env = gym.vector.make('BulletDodge-v0', num_envs=4)

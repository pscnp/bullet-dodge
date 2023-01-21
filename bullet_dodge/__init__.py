from gymnasium.envs.registration import register

from .environment import BulletDodgeEnv

register(
    id="bullet_dodge/BulletDodge-v0",
    entry_point="bullet_dodge:BulletDodgeEnv",
    max_episode_steps=1000,
)

from gymnasium.envs.registration import register

from .bulletdodge import BulletDodgeEnv
from .bulletdodge_pixel import BulletDodgeEnvPixel

register(
    id="BulletDodge",
    entry_point="envs:BulletDodgeEnv"
)

register(
    id="BulletDodgePixel",
    entry_point="envs:BulletDodgeEnvPixel"
)

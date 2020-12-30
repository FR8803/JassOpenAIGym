from gym.envs.registration import registry, register, make, spec
from gym import envs
register(
    id='Jass-v0',
    entry_point='gym_Jass.envs:JassEnv',
)
print(envs.registry.all())
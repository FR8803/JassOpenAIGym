from gym.envs.registration import registry, register, make, spec
register(
    id='Jass-v0',
    entry_point='gym_Jass.envs:JassEnv',
)
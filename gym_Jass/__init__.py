from gym.envs.registration import register

register(
    id='Jass-v0',
    entry_point='gym_Jass.envs:JassEnv',
)
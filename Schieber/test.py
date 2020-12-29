import gym
from gym import error, spaces, utils
from Schieber.round import JassRound

observation_space = spaces.Box(low=0, high=1, shape=(48, 13,), dtype=int)
observation_space2 = spaces.Box(low=0, high=600, shape=(22,), dtype=int)
print(observation_space)
print(observation_space2)

def step(self, action):
    if not self.round.played_cards:
        print(True)
import gym
from gym import error, spaces, utils
from gym.utils import seeding

from Schieber.game import JassGame
from Schieber.card import JassCard, JassSuits
from Schieber.DEBUG import DEBUG
from Schieber.dealer import JassDealer
from Schieber.player import JassPlayer
from Schieber.round import JassRound, Trumps


class FooEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self):
  self.observation_space =
  self.action_space =

#copied from Atari gym
  self.viewer = None

  def step(self, action):
    ...
  def reset(self):
    ...

  #copied from atari gym
  def render(self, mode='human'):
    return None

  #copied from Atari gym
  def close(self):
    if self.viewer is not None:
      self.viewer.close()
      self.viewer = None

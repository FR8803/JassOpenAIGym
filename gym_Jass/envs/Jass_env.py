import gym
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding
import gym_Jass.Schieber as Schieber
from gym_Jass.Schieber.game import JassGame
from gym_Jass.Schieber.card import JassCard, JassSuits
from gym_Jass.Schieber.DEBUG import DEBUG
from gym_Jass.Schieber.dealer import JassDealer as dealer
from gym_Jass.Schieber.player import JassPlayer
from gym_Jass.Schieber.round import JassRound, Trumps


class JassEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self):
    round = JassRound(dealer, np_random, round_counter, team_scores=None)
    players = JassGame().players
    player_id = round.current_player
    self.observation_space = spaces.Box(low=0, high=1, shape=(52, 13,), dtype=int)
  #returns all legal actions a player has
    self._action_set = JassRound.get_legal_actions(players, player_id)
  #returns the space of legal action, e.g. Discrete (8) returns a set with 8 elements {0, 1, 2, ..., 7}
    self.action_space =  spaces.Discrete(len(self._action_set))

  '''From the OpenAI Gym doc (https://gym.openai.com/docs/#environments):
  what our actions are doing to the environment, step return four values (implementation of the action-environment loop):
  1. observation (object) -> an environment-specific object representing your observation of the environment. e.g. the board state in a board game.
  2. reward (float) -> reward achieved trough action
  3. done (bool) -> whether environment has to be reseted again (end of an episode, when done == True)
  4. info (dict) -> diagnostic information useful for debugging. It can sometimes be useful for learning (for example, it might contain the raw probabilities behind the environment’s last state change). 
  '''
  #reward after each stich
  def step(self, a):
    reward = 0.0
    action = self._action_set[a]

    #change this!
    state = self.round.get_state(players, player_id)
    observation = self.round.get_observation(state)

    #reward only if agent receives it
    if not self.round.played_cards:
      self.stich_winner = self.round.stich_winner
      reward += self.round.calculate_stich_points
      done = True
    else:
      self.stich_winner = None
      done = False

    info = {}
    return observation, reward, done, info




  def reset(self):
    #resetting the environment and returning initial observation
    self.done = False
    state = self.round.get_state(players, player_id)
    return self.round.get_observation(state)

  def render(self, mode='human'):
    return None


  def close(self):
    return none


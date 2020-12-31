import gym
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding
import gym_Jass.Schieber as Schieber
from gym_Jass.Schieber.game import JassGame
from gym_Jass.Schieber.card import JassCard, JassSuits, init_swiss_deck
from gym_Jass.Schieber.DEBUG import DEBUG
from gym_Jass.Schieber.dealer import JassDealer as dealer
from gym_Jass.Schieber.player import JassPlayer
from gym_Jass.Schieber.round import JassRound, Trumps

def get_card_encodings():
  """
  Returns the encoding of cards in Jass.
  :return: A mapping of cards to their action ID, and a mapping of action ID to the card.
  """
  c = 0
  out = {}
  inverse = {}
  for card in init_swiss_deck():
    out[str(card)] = c
    inverse[c] = str(card)
    c += 1
  for trump in Trumps:
    out[f"STICH-{trump.value}"] = c
    inverse[c] = f"STICH-{trump.value}"
    c += 1
  return out, inverse

ACTION_SPACE, INVERSE_ACTION_SPACE = get_card_encodings()

class JassEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self):
    self.game = JassGame()
    #is there an initial reset when gym is made?
    self.game.init_game()
    self.players = self.game.players
    self.current_player = self.game.get_player_id()
    self.player_id = self.game.round.current_player


    self.observation = {}
    #1-9 players hand, 1-3 played cards, 4-36 history played cards, 1-7 Trumps
    self.observation_space = spaces.Box(low=0, high=1, shape=(54, 13,), dtype=int)

    self.action = None
  #index of card in players hand, e.g. Discrete (9) returns a set with 8 elements {0, 1, 2, ..., 8}
    self._action_set = self._get_legal_actions()
    self.action_space = spaces.Discrete(len(self._action_set))


  '''From the OpenAI Gym doc (https://gym.openai.com/docs/#environments):
  what our actions are doing to the environment, step return four values (implementation of the action-environment loop):
  1. observation (object) -> an environment-specific object representing your observation of the environment. e.g. the board state in a board game.
  2. reward (float) -> reward achieved trough action
  3. done (bool) -> whether environment has to be reseted again (end of an episode, when done == True)
  4. info (dict) -> diagnostic information useful for debugging. It can sometimes be useful for learning (for example, it might contain the raw probabilities behind the environment’s last state change). 
  '''
  #reward after each stich
  def step(self, a):
    reward = 0
    done = False
    self.action = self._action_set[a]

    state = self.game.get_state(self.player_id)
    observation = self.game.round.get_observation(state)

    reward = self.get_payoffs()

    info = {}
    return observation, reward, done, info

  def _get_legal_actions(self):
    legal_actions = self.game.get_legal_actions()
    legal_ids = [ACTION_SPACE[action] for action in legal_actions]
    return legal_ids

  def get_payoffs(self):
    payoffs, _scores = self.game.get_payoffs()
    return np.array(payoffs)

  def _decode_action(self, action_id):
    legal_ids = self._get_legal_actions()
    if action_id in legal_ids:
      return INVERSE_ACTION_SPACE[action_id]
    return INVERSE_ACTION_SPACE[np.random.choice(legal_ids)]

  def reset(self):
    #resetting the environment and returning initial observation
    #self.done = False
    #self.game.init_game()
    state = self.game.get_state(self.player_id)
    return self.game.round.get_observation(state)

  def render(self, mode='human'):
    return None


  def close(self):
    return None


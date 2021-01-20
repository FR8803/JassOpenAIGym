import gym
import numpy as np
from gym import error, spaces, utils

from gym_Jass.Schieber.game import JassGame
from gym_Jass.Schieber.card import init_swiss_deck
from gym_Jass.Schieber.round import Trumps
from gym_Jass.Schieber.util import encode_cards

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
    # self.current_player = self.game.get_player_id()
    self.player_id = self.game.round.current_player
    # print(self.player_id)

    self.observation = []
    #1-9 players hand, 1-3 played cards in the current Stich, 4-36 history played cards
    self.observation_space = spaces.Box(low=0, high=1, shape=(8, 4, 9), dtype=int)

    action_set = self._get_legal_actions()

    #0-8 for each card in hand
    self.action_space = spaces.Discrete(8)





    self.action = None



  '''From the OpenAI Gym doc (https://gym.openai.com/do cs/#environments):
  what our actions are doing to the environment, step return four values (implementation of the action-environment loop):
  1. observation (object) -> an environment-specific object representing your observation of the environment. e.g. the board state in a board game.
  2. reward (float) -> reward achieved trough action
  3. done (bool) -> whether environment has to be reseted again (end of an episode, when done == True)
  4. info (dict) -> diagnostic information useful for debugging. It can sometimes be useful for learning (for example, it might contain the raw probabilities behind the environment’s last state change). 
  '''
  #reward after each stich
  def step(self, a):
    assert self.action_space.contains(a)

    done = None

    #action space is set to a length of 8, however if the number of legal actions is less and the agent chooses an action which isn't part of the legal actions, a random action is being chosen. This might have an effect on the learning process
    #to be checked
    action, legal_action = self._decode_action(a)

    #this should encourage the agent to take legal actions
    if legal_action == True:
      self.reward += 0.001

    else:
      self.reward -= 0.001

    self.state, self.observation, player_id = self.game.step(action)

    self.state = self._extract_state(self.state)


    #after a complete game
    if self.game.is_over():
      self.reward += self.get_payoffs()[self.player_id]
      done = True

    info = {}
    print(self.reward)
    return np.array(self.state), np.array(self.reward), done, info


  def _get_legal_actions(self):
    legal_actions = self.game.get_legal_actions()
    legal_ids = [ACTION_SPACE[action] for action in legal_actions]
    return legal_ids

  def get_payoffs(self):
    payoffs, _scores = self.game.get_payoffs()
    return np.array(payoffs)

  def _decode_action(self, action_id):
    legal_action = None
    legal_ids = self._get_legal_actions()
    if action_id in legal_ids:
      legal_action = True
      return INVERSE_ACTION_SPACE[action_id], legal_action
    legal_action = False
    return INVERSE_ACTION_SPACE[np.random.choice(legal_ids)], legal_action

  def _extract_state(self, state):
    # returns a players hand, in first array a 1 if he has a card and a zero if he doesn't have it and in then the second array the opposite
    obs = np.zeros((8, 4, 9), dtype=int)
    encode_cards(obs[:2], state["hand"], "hand")
    encode_cards(obs[2:4], [str(x[1]) for x in state["played_cards"]], "hand")
    encode_cards(obs[4:6], [str(x[1]) for x in state["history_played_cards"]])
    legal_actions = self.game.get_legal_actions()
    legal_ids = self._get_legal_actions()
    if legal_ids[0] < 36:
      encode_cards(obs[6:8], [str(x) for x in legal_actions])
    else:
      pass
    obs = obs.astype("int64")
    return obs


  def reset(self):
    #resetting the environment and returning initial observation
    self.game.init_game()
    self.reward = 0.0
    #self.state =np.zeros((4, 4, 9), dtype=int)
    self.state = self.game.get_state(self.player_id)
    self.state = self._extract_state(self.state)
    self.state = np.array(self.state)
    return self.state


  def close(self):
    return None


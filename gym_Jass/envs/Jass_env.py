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
    self.game.init_game()

    self.players = self.game.players

    #id of agent set to zero
    self.player_id = 0

    self.observation = []
    #1-9 players hand, 1-3 played cards in the current Stich, 4-36 history played cards, 1-9 legal actions
    self.observation_space = spaces.Box(low=0, high=1, shape=(4, 4, 9), dtype=int)

    action_set = self._get_legal_actions()

    #0-8 for each card in hand
    self.action_space = spaces.Discrete(8)

    '''Three types of reward systems:
    Game-> a reward of either 0 or 1 is given, if the game is either lost or won. This is probably the most sparse reward system of the three
    Hybrid-> the reward is also being given at the end of the game, however the end reward depens on the performance of each Stich (by dividing a teams points by the total points that could have been reached)
    Round-> Points are being given after each Round (36 cards played), in order to keep the reward between 0 and 1 it is being divided by the sum of the points the two teams achieved
    Stich-> Points are being given after each Stich (4 cards played), which is then being divided by 157 (max points after 9 Stiche / one round) to keep the points within 0 and 1
    
    '''

    self.reward_type = "Stich "

    if self.reward_type not in ["Game", "Hybrid", "Round", "Stich"]:
      raise ValueError("Invalid reward type")

    self.action = None
    self.players_list = []

  '''From the OpenAI Gym doc (https://gym.openai.com/do cs/#environments):
  what our actions are doing to the environment, step returns four values (implementation of the action-environment loop):
  1. observation (object) -> an environment-specific object representing your observation of the environment. e.g. the board state in a board game.
  2. reward (float) -> reward achieved trough action
  3. done (bool) -> whether environment has to be reseted again (end of an episode, when done == True)
  4. info (dict) -> diagnostic information useful for debugging. It can sometimes be useful for learning (for example, it might contain the raw probabilities behind the environment’s last state change). 
  '''
  #reward after each stich
  def step(self, a):
    assert self.action_space.contains(a)

    done = None

    info = {}
    self.state = self.game.get_state(self.player_id)

    #tracks the players who have played cards so far if all four players have played a card: cycle = True
    if self.reward_type == "Stich":
      cycle = self.check_player_cycle(self.state)

    #player 0 is being trained, while the other players take random actions
    current_player = self.game.round.current_player
    if current_player in [1, 2, 3]:
      action = self.opponent_or_team_member_play()
      self.game.step(action)


    else:
      #action space is set to a length of 8, however if the number of legal actions is less and the agent chooses an action which isn't part of the legal actions, a random action is being chosen. This might have an effect on the learning process
      action, legal_action = self._decode_action(a)

      #this should encourage the agent to take legal actions
      #if legal_action == True:
      #  self.rule_reward += 0.0001

      #else:
       # self.rule_reward -= 0.0001

      self.game.step(action)
      self.state = self.game.get_state(self.player_id)
      self.observation = self._extract_observation(self.state)
      self.observation = np.array(self.observation).astype(float)

    # played cards is empty which means, that a Stich is over
    if not self.game.round.played_cards:
      if self.reward_type == "Round":
        #history_played_cards is being reset to 0 after every round
        self.state = self.game.get_state(self.player_id)
        if len(self.state["history_played_cards"]) == 0:
          #returns the difference in points between the current and last round
          _, _, diff = self.get_payoffs()
          if diff[0, 2] != 0 or diff[1, 3] != 0:
            #to keep the reward within 0 and 1
            self.reward = diff[0, 2] / (diff[0, 2] + diff[1, 3])
            if self.player_id == 1 or self.player_id == 3:
              self.reward = 1 - self.reward
            #self.reward += self.rule_reward
            done = True

      elif self.reward_type == "Stich":
        #returns the difference in points between the current and last stich
        _, _, diff = self.get_payoffs()

        if cycle == True:
          if self.player_id == 0 or self.player_id == 2:
            self.reward = diff[0, 2] / 157
          else:
            self.reward = diff[1, 3] / 157
            #fix: reward can become slightly negative due to rule reward
          #self.reward += self.rule_reward
          done = True


    #after a complete game
    if self.game.is_over():
      done = True
      payoffs, _, _ = self.get_payoffs()
      if self.reward_type == "Hybrid":
        self.reward += payoffs[self.player_id]
        #self.reward += self.rule_reward

      elif self.reward_type == "Game":
        #if the payoff of a player is bigger than 0.5 this means that he won more than 50% of the points and is therefore a winner
        if payoffs[self.player_id] > 0.5:
          self.reward += 1
          #self.reward += self.rule_reward
        else:
          self.reward = 0
          #self.reward += self.rule_reward

      else:
        pass
    return self.observation, np.array(self.reward), done, info



  def _get_legal_actions(self):
    legal_actions = self.game.get_legal_actions()
    legal_ids = [ACTION_SPACE[action] for action in legal_actions]
    return legal_ids

  #returns a random legal action for the teammate and for the opponents
  def opponent_or_team_member_play(self):
    legal_ids = self._get_legal_actions()
    return INVERSE_ACTION_SPACE[np.random.choice(legal_ids)]


  def get_payoffs(self):
    payoffs, _scores, diff = self.game.get_payoffs()
    return np.array(payoffs), _scores, diff


  def _decode_action(self, action_id):
    legal_action = None
    legal_ids = self._get_legal_actions()
    if action_id in legal_ids:
      legal_action = True
      return INVERSE_ACTION_SPACE[action_id], legal_action
    legal_action = False
    return INVERSE_ACTION_SPACE[np.random.choice(legal_ids)], legal_action

  def _extract_observation(self, state):
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
    # dropping 50% of the state, so that 1 signifies card "is in hand", "card has been played" or "is a legal action"
    obs = np.delete(obs, [0, 2, 4, 6], 0)
    return obs

  def observation_and_action_constraint_splitter(self, observation):
    legal_action = observation[3]
    return observation, legal_action

  #checks whether each player has done an action in a round
  def check_player_cycle(self, state):
    cycle = False
    if state["legal_actions"]:
      if not state["legal_actions"][0].startswith("STICH-"):
        self.players_list.append([state["current_player"]])
    else:
      self.players_list.append([state["current_player"]])
    self.players_list.sort()
    if self.players_list == [[0], [1], [2], [3]]:
      cycle = True
      self.players_list = []
    return cycle



  def reset(self):
    #resetting the environment and returning initial observation
    self.reward = 0.0
    #self.rule_reward = 0.0
    self.state = self.game.get_state(self.player_id)
    if self.reward_type == "Hybrid" or self.reward_type == "Game":
      self.game.init_game()
    else:
      if len(self.state["history_played_cards"]) == 0:
        self.game.init_game()
    self.observation = self._extract_observation(self.state)
    self.observation = np.array(self.observation)
    return self.observation


  def close(self):
    return None


import gym
import numpy as np
from gym import error, spaces, utils

from gym_Jass.Schieber.game import JassGame
from gym_Jass.Schieber.card import init_swiss_deck
from gym_Jass.Schieber.round import Trumps
from gym_Jass.Schieber.util import encode_cards, encode_stich

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
    #1-9 players hand, 1-3 played cards in the current Stich, 4-36 history played cards, 1-9 legal actions, 1 current stich
    self.observation_space = spaces.Box(low=0, high=1, shape=(5, 4, 9), dtype=int)

    action_set = self._get_legal_actions()

    #0-8 for each card in hand
    self.action_space = spaces.Discrete(8)

    '''Five types of reward systems:
    Hybrid-> Reward after each round and a reward of 0.5 for winning the game
    Game 0/1 -> a reward of either 0 or 1 is given, if the game is either lost or won. This is probably the most sparse reward system of the three
    Game-> the reward is also being given at the end of the game, however the end reward depens on the performance of each Stich (by dividing a teams points by the total points that could have been reached)
    Round-> Points are being given after each Round (36 cards played), in order to keep the reward between 0 and 1 in a game, the points are being divided by the goal points (1000)
    Stich-> Points are being given after each Stich (4 cards played), which is then being divided by 157 (max points after 9 Stiche / one round) to keep the points within 0 and 1
    
    '''

    self.reward_type = "Round"

    if self.reward_type not in ["Game 0/1", "Game", "Hybrid", "Round", "Stich"]:
      raise ValueError("Invalid reward type")

    '''Three different types of strategies:
    Best-> Teammate and two opponents always play their best card (measured in the value of points a card has)
    Random-> Teammate and two opponents always play a random card
    Mixed->Teammate and one opponent play a random card, one opponent plays the best card
    '''
    self.strategy = "Mixed"
    if self.strategy not in ["Random", "Best", "Mixed"]:
      raise ValueError("Invalid strategy")

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
    self.reward = 0.0
    done = None
    info = {}
    #take the action chosen by the DQN agent
    action, legal_action = self._decode_action(a)
    self.game.step(action)


    #if first action was a Stich, do another action
    if action.startswith("STICH-"):
      self.state = self.game.get_state(self.player_id)
      self.observation = self._extract_observation(self.state)
      self.observation = np.array(self.observation).astype(float)
      return self.observation, np.array(self.reward), done, info

    self.state = self.game.get_state(self.player_id)
    #all the other players take actions until it's player 0 turn again
    while self.game.round.current_player != 0:
      #this means that the Stich is not over yet and opponents have to make moves
      if len(self.state["played_cards"]) != 0:
        action = self.opponent_or_team_member_play(self.game.round.current_player)
        self.game.step(action)
        self.state = self.game.get_state(self.player_id)
        #rewards are being returned and first player takes an action
      else:
        if self.reward_type in ["Round", "Stich", "Hybrid"]:
          self.reward = self.get_rewards()
        action = self.opponent_or_team_member_play(self.game.round.current_player)
        self.game.step(action)
        self.state = self.game.get_state(self.player_id)
        # if action is Stich, same player has to do another action
        if action.startswith("STICH-"):
          action = self.opponent_or_team_member_play(self.game.round.current_player)
          self.game.step(action)

    #checking whether it is the agents turn again (if so the current state is being returned), otherwise random actions are being taken
    if self.game.round.current_player == 0:
      self.state = self.game.get_state(self.player_id)
      self.observation = self._extract_observation(self.state)
      self.observation = np.array(self.observation).astype(float)
      if len(self.state["played_cards"]) == 0:
        if self.reward_type in ["Round", "Stich"]:
          self.reward = self.get_rewards()

    #after a complete game
    if self.game.is_over():
      done = True
      self.reward = self.get_rewards()

    return self.observation, np.array(self.reward), done, info



  def _get_legal_actions(self):
    legal_actions = self.game.get_legal_actions()
    legal_ids = [ACTION_SPACE[action] for action in legal_actions]
    return legal_ids, legal_actions

  #returns a random legal action for the teammate and for the opponents
  def opponent_or_team_member_play(self, current_player):
    current_player = int(current_player)
    legal_ids, legal_actions = self._get_legal_actions()
    if not legal_actions[0].startswith("STICH-"):
      if self.strategy == "Random":
        return INVERSE_ACTION_SPACE[np.random.choice(legal_ids)]
      elif self.strategy == "Best":
        legal_actions_values = self.game.round.calculate_card_value(legal_actions)
        card_max_value = max(legal_actions_values, key=legal_actions_values.get)
        return card_max_value
      elif self.strategy == "Mixed":
        if current_player == 1 or current_player == 2:
          return INVERSE_ACTION_SPACE[np.random.choice(legal_ids)]
        elif current_player == 3:
          legal_actions_values = self.game.round.calculate_card_value(legal_actions)
          card_max_value = max(legal_actions_values, key=legal_actions_values.get)
          return card_max_value
    else:
      return INVERSE_ACTION_SPACE[np.random.choice(legal_ids)]



  def get_payoffs(self):
    payoffs, _scores, diff = self.game.get_payoffs()
    return np.array(payoffs), _scores, diff

  def get_rewards(self):
    if self.reward_type == "Round":
      # history_played_cards is being reset to 0 after every round
      self.state = self.game.get_state(self.player_id)
      if len(self.state["history_played_cards"]) == 0:
        # returns the difference in points between the current and last round
        _, _, diff = self.get_payoffs()
        if diff[0, 2] != 0 or diff[1, 3] != 0:
          # to keep the reward within 0 and 1
          if self.player_id == 0 or self.player_id == 2:
            self.reward = diff[0, 2] / 1000
          else:
            self.reward = diff[1, 3] / 1000
          # self.reward += self.rule_reward
      if self.game.is_over():
        _, _, diff = self.get_payoffs()
        if diff[0, 2] != 0 or diff[1, 3] != 0:
          # to keep the reward within 0 and 1
          if self.player_id == 0 or self.player_id == 2:
            self.reward = diff[0, 2] / 1000
          else:
            self.reward = diff[1, 3] / 1000

    elif self.reward_type == "Stich":
      # returns the difference in points between the current and last stich
      _, _, diff = self.get_payoffs()
      if self.player_id == 0 or self.player_id == 2:
        self.reward = diff[0, 2] / 157
      else:
        self.reward = diff[1, 3] / 157

    elif self.reward_type == "Game":
      payoffs, _, _ = self.get_payoffs()
      self.reward += payoffs[self.player_id]

    elif self.reward_type == "Game 0/1":
      payoffs, _, _ = self.get_payoffs()
      # if the payoff of a player is bigger than 0.5 this means that he won more than 50% of the points and is therefore a winner
      if payoffs[self.player_id] > 0.5:
        self.reward += 1
      else:
        self.reward = 0

    elif self.reward_type == "Hybrid":
      self.state = self.game.get_state(self.player_id)
      if len(self.state["history_played_cards"]) == 0:
        # returns the difference in points between the current and last round
        _, _, diff = self.get_payoffs()
        if diff[0, 2] != 0 or diff[1, 3] != 0:
          # to keep the reward within 0 and 1
          if self.player_id == 0 or self.player_id == 2:
            self.reward = diff[0, 2] / 2000
          else:
            self.reward = diff[1, 3] / 2000
      if self.game.is_over():
        payoffs, _, _ = self.get_payoffs()
        #if player won more than 50% of all rounds
        if payoffs[self.player_id] > 0.5:
          self.reward = 0.5
    return self.reward


  def _decode_action(self, action_id):
    legal_action = None
    legal_ids, legal_actions = self._get_legal_actions()
    if action_id in legal_ids:
      legal_action = True
      return INVERSE_ACTION_SPACE[action_id], legal_action
    legal_action = False
    return INVERSE_ACTION_SPACE[np.random.choice(legal_ids)], legal_action

  def _extract_observation(self, state):
    # returns a players hand, in first array a 1 if he has a card and a zero if he doesn't have it and in then the second array the opposite
    obs = np.zeros((9, 4, 9), dtype=int)
    encode_cards(obs[:2], state["hand"], "hand")
    encode_cards(obs[2:4], [str(x[1]) for x in state["played_cards"]], "hand")
    encode_cards(obs[4:6], [str(x[1]) for x in state["history_played_cards"]])
    legal_ids, legal_actions = self._get_legal_actions()
    if legal_ids[0] < 36:
      encode_cards(obs[6:8], [str(x) for x in legal_actions])
    else:
      pass
    if state["trump"] != None:
      encode_stich(obs[8], str(state["trump"]))
    # dropping 50% of the state, so that 1 signifies card "is in hand", "card has been played" or "is a legal action"
    obs = np.delete(obs, [0, 2, 4, 6], 0)
    return obs

  def observation_and_action_constraint_splitter(self, observation):
    legal_action = observation[3]
    return observation, legal_action





  def reset(self):
    #resetting the environment and returning initial observation after the whole game
    self.reward = 0.0
    self.game.init_game()
    self.state = self.game.get_state(self.player_id)
    self.observation = np.array(self._extract_observation(self.state))
    return self.observation

  def close(self):
    return None


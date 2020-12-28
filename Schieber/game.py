from copy import deepcopy

import numpy as np

from rlcard.games.jass.dealer import JassDealer
from rlcard.games.jass.player import JassPlayer
from rlcard.games.jass.round import JassRound

MAX_STICH_SCORE = 157  # max score one can win in a stich


class JassGame(object):
    def __init__(self, allow_step_back=False):
        self.allow_step_back = allow_step_back
        self.np_random = np.random.RandomState()
        self.num_players = 4
        self.payoffs = [-1 for _ in range(self.num_players)]
        self.players: [JassPlayer] = []
        self.round: JassRound = None
        self.dealer: JassDealer = None
        self.history: [(JassDealer, [JassPlayer], JassRound)] = []
        # tracks when a new round started - is set back to False when the value is checked
        self.new_round = False
        self.point_history = []
        self.stich_history = []
        self.stich_winner = None

    def init_game(self, round_counter=1, team_scores=None):
        """
        Sets up a fresh round for the game.
        :param round_counter: Optional. Default: 1. The number of the current round being played. Used for debugging,
            otherwise should always be left as 1.
        :param team_scores: Optional. Default: None. The scores of each team. Used for debugging, otherwise should
            always be left as None.
        :return: The state of the game and the ID of the current player.
        """
        if team_scores is None:
            team_scores = {(0, 2): 0, (1, 3): 0}
        self.payoffs = [-1 for _ in range(self.num_players)]
        self.dealer = JassDealer(self.np_random)
        self.players = [JassPlayer(i, self.np_random) for i in range(self.num_players)]
        for player in self.players:
            self.dealer.deal_cards(player, 9)
        self.round = JassRound(self.dealer, self.np_random, round_counter, team_scores)
        player_id = self.round.current_player
        state = self.get_state(player_id)
        return state, player_id

    def get_state(self, player_id):
        """
        Returns the current state of the game for a given player.
        :param player_id: The ID of the player.
        :return: A dictionary containing the state of the game.
        """
        state = self.round.get_state(self.players, player_id)
        state["player_num"] = self.num_players
        state["current_player"] = self.round.current_player
        return state

    def step(self, action):
        """
        Performs a single step (action) in the game.
        :param action: The action being taken.
        :return: The state of the game AFTER the action, and the player whose turn it is AFTER the action.
        """
        # store previous state in history
        if self.allow_step_back:
            his_dealer = deepcopy(self.dealer)
            his_round = deepcopy(self.round)
            his_players = deepcopy(self.players)
            self.history.append((his_dealer, his_round, his_players))
        self.round.proceed_round(self.players, action)
        player_id = self.round.current_player
        state = self.get_state(player_id)

        # get stich_winner if played cards is empty so it can be passed to frontend
        if not self.round.played_cards:
            self.stich_winner = self.round.stich_winner
        else:
            self.stich_winner = None
        # if round is over, calculate points for this round
        if self.is_round_over():
            self.new_round = True
            team_scores = self.round.round_points
            if self.point_history:
                totals = {(0, 2): 0, (1, 3): 0}
                for _prev in self.point_history:
                    for team, score in _prev.items():
                        totals[team] += score
                current_points = {k: team_scores[k] - totals[k] for k in totals.keys()}
                self.point_history.append(deepcopy(current_points))
            else:
                self.point_history.append(deepcopy(team_scores))
            counter = (sum(team_scores.values()) / 157) + 1
            self.dealer.take_deck_back()
            for player in self.players:
                self.dealer.deal_cards(player, 9)
            self.stich_history.append(self.round.player2stack)
            # remember prev player from before
            prev_player = self.round.prev_player
            self.round = JassRound(self.dealer, self.np_random, counter, team_scores)
            # self.round.prev_player = prev_player
            # self.round.current_player = self.stich_winner
        return state, player_id

    def step_back(self):
        """
        Undo the last step, revert the game back to the previous state.
        :return: True if the game was stepped back, False if not, i.e. when there is no history.
        """
        if not self.history:
            return False
        self.dealer, self.players, self.round = self.history.pop()
        return True

    def get_payoffs(self):
        """
        Calculates the payoffs / rewards for the current state of the game.
        :return: The payoffs and the points for the current round.
        """
        score = self.round.round_points
        score = sorted(score.items(), key=lambda k: k[1], reverse=True)
        winner, w_score = score[0]
        loser, l_score = score[1]
        w_reward = w_score / (157 * self.round.round_counter)
        l_reward = 1 - (w_score / (157 * self.round.round_counter))
        self.payoffs[winner[0]] = w_reward
        self.payoffs[winner[1]] = w_reward
        self.payoffs[loser[0]] = l_reward
        self.payoffs[loser[1]] = l_reward
        return self.payoffs, self.round.round_points

    def get_legal_actions(self):
        return self.round.get_legal_actions(self.players, self.round.current_player)

    def is_round_over(self):
        return self.round.round_over

    def is_game_over(self):
        return self.round.game_over

    def get_player_num(self):
        # TODO: make this not ugly
        return self.num_players

    def get_action_num(self):
        return 43

    def is_over(self):
        # WORKS:
        return self.is_game_over()
        # EXPERIMENTAL, COULD BREAK THE UNIVERSE:
        # return self.is_round_over()

    def get_player_id(self):
        return self.round.current_player

    def check_new_round(self):
        if self.new_round:
            self.new_round = False
            return True
        return False

    def get_stich_history_full(self):
        """
        Returns full Stich history for all players.
        Maps the player ID to a list of won Stiches.
        Each won Stich is represented by a list of tuples, where each tuple has the format (player ID, card).
        The ID in the tuples is the ID of the player who played this particular card.
        :return: The Stich history.
        """
        history = {k: [] for k in range(self.round.NUM_PLAYERS)}
        for stiche in self.stich_history:
            for k, v in stiche.items():
                history[k].extend(v)
        return history

from enum import Enum

from gym_Jass.Schieber.card import JassSuits, JassCard
from gym_Jass.Schieber.util import cards2list


class Trumps(Enum):
    EICHEL = JassSuits.EICHEL
    ROSE = JassSuits.ROSEN
    SCHELLE = JassSuits.SCHELLEN
    SCHILTE = JassSuits.SCHILTEN
    OBENABE = "Obenabe"
    UNDENUFE = "Undenufe"
    GSCHOBE = "Gschobe"


OBENABE = {x: i for i, x in enumerate(["Ass", "König", "Ober", "Under", "Banner", "9", "8", "7", "6"])}
UNDENUFE = {x: i for i, x in
            enumerate(reversed(["Ass", "König", "Ober", "Under", "Banner", "9", "8", "7", "6"]))}
TRUMPF = {x: i for i, x in enumerate(["Under", "9", "Ass", "König", "Ober", "Banner", "8", "7", "6"])}
NEBENFARBE = {x: i for i, x in enumerate(["Ass", "König", "Ober", "Under", "Banner", "9", "8", "7", "6"])}

POINTS_OBENABE = {i: points for i, points in zip(OBENABE.values(), [11, 4, 3, 2, 10, 0, 8, 0, 0])}
POINTS_UNDENUFE = {i: points for i, points in zip(UNDENUFE.values(), [11, 0, 8, 0, 10, 2, 3, 4, 0])}
POINTS_TRUMPF = {i: points for i, points in zip(TRUMPF.values(), [20, 14, 11, 4, 3, 10, 0, 0, 0])}

# card order for nebenfarbe is same as obenabe, but points are different
POINTS_NEBENFARBE = {i: points for i, points in zip(NEBENFARBE.values(), [11, 4, 3, 2, 10, 0, 0, 0, 0])}


class JassRound(object):
    """
    One round of Jass.
    Begins with setting the Stich.
    """

    def __init__(self, dealer, np_random, round_counter, team_scores=None):
        self.NUM_PLAYERS = 4
        self.FINAL_STICH_BONUS = 5
        self.SCORE_GOAL = 1000
        self.np_random = np_random
        self.round_counter = round_counter
        self.dealer = dealer
        self.trump = None
        self.played_cards: [JassCard] = []
        self.history_played_cards = []
        self.card2player = {}
        # keeps track of the players 'winning cards'
        self.player2stack = {player: [] for player in range(self.NUM_PLAYERS)}
        self.current_player = round_counter - 1  # player whose turn it is - start with 0 by default
        if self.current_player >= self.NUM_PLAYERS:
            # round 5: current_player must be 0
            # round 7:
            # ich - christina - matthias - thomas - ich - christina - matthias - thomas - ich - christina - matthias
            #  1      2           3          4       5       6           7          8      9          10         11
            self.current_player = int((round_counter - 1) % self.NUM_PLAYERS)
            # print("PLAAAAAYER: ", self.current_player)
        self.prev_player = 0 if round_counter == 1 else self.current_player
        self.stich_counter = 0  # goes up to 9 - every player has played all cards in that case
        self.gschobe = False
        self.gschobe_player = None
        self.stich_winner = None  # tuple of winning players for stich
        self.round_winner = None  # tuple of winning players for round
        self.stich_over = False
        self.round_over = False
        self.game_over = False
        self.game_winner = None
        if team_scores is None:
            self.round_points = {(0, 2): 0, (1, 3): 0}
        else:
            self.round_points = team_scores

    def set_trump(self, trump: Trumps):
        self.trump = trump

    def get_legal_actions(self, players, player_id):
        """
        Returns a list of legal actions that a player can take.

        :param players: The players in the current game.
        :param player_id: The ID of the player whose legal actions are being checked.
        :return: A list of legal actions that a player can take.
        """
        # first action: pick trump for this round
        if not self.played_cards and not self.trump:
            # first card, must set the Stich
            if self.gschobe:
                # can do it again
                legal_actions = [f"STICH-{e.value}" for e in
                                 [Trumps.UNDENUFE, Trumps.OBENABE, Trumps.EICHEL, Trumps.ROSE, Trumps.SCHELLE,
                                  Trumps.SCHILTE]]
            else:
                # pick any trumpf option
                legal_actions = [f"STICH-{e.value}" for e in Trumps]
            return legal_actions
        hand = players[player_id].hand
        # check if a card has already been played
        if self.played_cards:
            # get card, list of tuples in format [(player, card), ...]
            current_suit = self.played_cards[0][1].suit
            same_in_hand = [card for card in hand if card.suit == current_suit or card.suit == self.trump.value]
            # check case for unter in trumpffarbe: dont have to play it!
            if len(same_in_hand) == 1 and same_in_hand[0].suit == self.trump.value and same_in_hand[0].rank == "Under":
                # only playable card in hand is trumpf unter: don't have to play it
                legal_actions = hand  # can play anything in hand
            if not same_in_hand:
                legal_actions = hand  # can play anything in hand
            else:  # have same color in hand, must play it
                legal_actions = same_in_hand
        else:
            # can play any card at first
            legal_actions = hand
        legal_actions = [str(x) for x in legal_actions]
        return legal_actions

    def get_state(self, players, player_id):
        state = {}
        if isinstance(player_id, float):
            player_id = int(player_id)
        try:
            player = players[player_id]
        except IndexError:
            print(player_id)
        state["hand"] = cards2list(player.hand)
        state["played_cards"] = self.played_cards
        state["card2player"] = self.card2player
        others_hand = []
        for player in players:
            if player.player_id != player_id:
                others_hand.extend(player.hand)
        state["others_hand"] = cards2list(others_hand)
        state["history_played_cards"] = self.history_played_cards
        state["legal_actions"] = self.get_legal_actions(players, player_id)
        state["trump"] = self.trump
        return state

    def get_observation(self, state):
        observation = {}
        observation["hand"] = state["hand"]
        observation["played_cards"] = state["played_cards"]
        observation["history_played_cards"] = state["history_played_cards"]
        observation["trump"] = state["trump"]
        return observation


    def proceed_round(self, players, action):
        if isinstance(self.current_player, float):
            self.current_player = int(self.current_player)
        player = players[self.current_player]

        # check if action is a trump action
        if action.startswith("STICH-"):
            action = action.split("STICH-")[1]
            if action == Trumps.GSCHOBE.value:
                self.gschobe_player = self.current_player  # go back to this player later
                self.prev_player = self.current_player
                self.current_player = self.current_player + 2
                self.gschobe = True
                if self.current_player >= len(players):
                    self.current_player -= len(players)
                return  # dont set this as trump, other player must decide

            # trump is being chosen
            trump = [trump for trump in Trumps if str(trump.value) == action][0]
            self.set_trump(trump)
            if self.gschobe:
                self.prev_player = self.current_player
                self.current_player = self.gschobe_player
                self.gschobe = False
            # next player is player AFTER the player picking the trump
            # self.current_player += 1
            # if self.current_player >= len(players):
            #     self.current_player = 0  # loop back to first
            return  # trump is set

        # not trump setting, gotta play a card
        # format is "{rank}-{suit}"
        card_info = action.split("-")
        rank = card_info[0]
        suit = card_info[1]
        ri = None
        for i, card in enumerate(player.hand):
            if str(card.suit) == suit and str(card.rank) == rank:
                ri = i
                break
        # card object of card that is being played
        card = player.hand.pop(ri)
        self.played_cards.append((self.current_player, card))
        self.card2player[player] = card

        # check if all players made their play
        if len(self.played_cards) == len(players):
            if len(player.hand) == 0:  # empty hand, all cards were played: end round!
                self.round_over = True
            winner = self.calculate_stich_winner()
            if self.round_over:
                points_awarded = sum(self.round_points.values())
                if points_awarded != self.round_counter * 157:  # should always be this exact value
                    raise Exception(">>> Point total awarded in a round does not add up to 157 <<<")
            self.prev_player = self.current_player
            self.current_player = winner  # winner opens next stich
        else:
            # advance player
            self.prev_player = self.current_player
            self.current_player += 1
            if self.current_player == len(players):
                self.current_player = 0  # loop back to first player
        # check if anyone has reached 1000 points
        self.check_end()

    def check_end(self):
        if any(x >= self.SCORE_GOAL for x in self.round_points.values()):
            self.game_over = True
            winner = sorted(self.round_points.items(), key=lambda k: k[1], reverse=True)[0][0]
            self.game_winner = winner
            return self.game_over

    def sort_cards(self, cards):
        if self.trump == Trumps.OBENABE:
            cards = sorted(cards, key=lambda k: (
                0 if k[1].suit == self.trump.value else 1 if k[1].suit == cards[0][1].suit else 2, OBENABE[k[1].rank]))
        elif self.trump == Trumps.UNDENUFE:
            cards = sorted(cards, key=lambda k: (
                0 if k[1].suit == self.trump.value else 1 if k[1].suit == cards[0][1].suit else 2, UNDENUFE[k[1].rank]))
        else:
            cards = sorted(cards, key=lambda k: (
                0 if k[1].suit == self.trump.value else 1 if k[1].suit == cards[0][1].suit else 2, TRUMPF[k[1].rank]))
        return cards

    def calculate_stich_winner(self):
        cards = self.sort_cards(self.played_cards)
        winner = cards[0][0]
        partner = winner + 2
        if partner >= self.NUM_PLAYERS:
            partner -= self.NUM_PLAYERS
        stich_winner = tuple(sorted([winner, partner]))
        # store winner cards
        if winner not in self.player2stack:
            self.player2stack[winner] = [self.played_cards]
        else:
            self.player2stack[winner].append(self.played_cards)
        score = self.calculate_stich_points([card for _player, card in self.played_cards])
        if self.round_over:
            score += self.FINAL_STICH_BONUS  # assign bonus points for final stich
        self.round_points[stich_winner] += score
        self.stich_winner = winner  # only track winning player, not team
        # self.current_player = winner  # winner opens the next stich
        #create history of cards played
        self.history_played_cards += self.played_cards
        self.played_cards = []  # clear played cards
        return winner

    def calculate_stich_points(self, cards):
        score = 0
        for card in cards:
            # color was chosen as trump, card matches it
            if card.suit == self.trump.value:
                score += POINTS_TRUMPF[TRUMPF[card.rank]]
            # color was chosen as trump, card is nebenfarbe
            elif self.trump in [Trumps.SCHILTE, Trumps.EICHEL, Trumps.ROSE, Trumps.SCHELLE]:
                score += POINTS_NEBENFARBE[NEBENFARBE[card.rank]]
            elif self.trump == Trumps.OBENABE:
                score += POINTS_OBENABE[OBENABE[card.rank]]
            elif self.trump == Trumps.UNDENUFE:
                score += POINTS_UNDENUFE[UNDENUFE[card.rank]]
            else:
                # PANIC
                raise Exception(">>> Trump was not properly set to valid value <<<")
        return score

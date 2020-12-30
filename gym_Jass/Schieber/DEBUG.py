from gym_Jass.Schieber.card import JassCard, JassSuits
from gym_Jass.Schieber.player import JassPlayer
from gym_Jass.Schieber.round import Trumps, TRUMPF, UNDENUFE


class DEBUG:
    played_cards = []
    trump = None
    gschobe = False

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
            if len(same_in_hand) == 1 and same_in_hand[0].suit == self.trump and same_in_hand[0].rank == "Unter":
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


def check_trump_play_legal():
    d = DEBUG()
    d.played_cards = [(0, JassCard(JassSuits.EICHEL, "10"))]
    p = JassPlayer(1, 42)
    p.hand = [JassCard(JassSuits.SCHELLEN, "König"), JassCard(JassSuits.EICHEL, "9"),
              JassCard(JassSuits.EICHEL, "König")]
    d.trump = Trumps.SCHELLE
    legal = d.get_legal_actions([0, p], 1)
    print(legal)


def check_priority():
    _cards = [
        (2, JassCard(JassSuits.SCHILTEN, "8")),
        (3, JassCard(JassSuits.SCHELLEN, "7")),
        (1, JassCard(JassSuits.ROSEN, "Under")),
        (0, JassCard(JassSuits.ROSEN, "Ass"))
    ]
    trump = Trumps.UNDENUFE
    _cards = sorted(_cards, key=lambda k: (
        0 if k[1].suit == trump.value else 1 if k[1].suit == _cards[0][1].suit else 2, UNDENUFE[k[1].rank]))
    print(_cards)


def check_all_trump():
    _cards = [
        (2, JassCard(JassSuits.SCHILTEN, "Under")),
        (3, JassCard(JassSuits.SCHILTEN, "Ass")),
        (1, JassCard(JassSuits.SCHILTEN, "König")),
        (0, JassCard(JassSuits.SCHILTEN, "Ober"))
    ]
    trump = Trumps.SCHILTE
    _cards = sorted(_cards, key=lambda k: (
        0 if k[1].suit == trump else 1 if k[1].suit == _cards[0][1].suit else 2, TRUMPF[k[1].rank]))
    print(_cards)


if __name__ == "__main__":
    # check_trump_play_legal()
    # check_priority()
    check_all_trump()

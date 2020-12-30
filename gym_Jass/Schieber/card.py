from enum import Enum


class JassSuits(Enum):
    SCHELLEN = "Schellen"
    ROSEN = "Rosen"
    SCHILTEN = "Schilten"
    EICHEL = "Eichel"


RANKS = ["Ass", "König", "Ober", "Under", "Banner", "9", "8", "7", "6"]


def init_swiss_deck():
    suit_list = [suit for suit in JassSuits]
    res = [JassCard(suit, rank) for suit in suit_list for rank in RANKS]
    return res


class JassCard(object):
    '''
    Card stores the suit and rank of a single card

    Note:
        The suit variable in a standard card game should be one of [S, H, D, C, BJ, RJ] meaning [Spades, Hearts, Diamonds, Clubs, Black Joker, Red Joker]
        Similarly the rank variable should be one of [A, 2, 3, 4, 5, 6, 7, 8, 9, T, J, Q, K]
    '''

    suit = None
    rank = None
    valid_suit = [suit for suit in JassSuits]
    valid_rank = ["6", "7", "8", "9", "Banner", "Under", "Ober", "König", "Ass"]

    def __init__(self, suit, rank):
        ''' Initialize the suit and rank of a card

        Args:
            suit: string, suit of the card, should be one of valid_suit
            rank: string, rank of the card, should be one of valid_rank
        '''
        self.suit = suit
        self.rank = rank

    def __eq__(self, other):
        if isinstance(other, JassCard):
            return self.rank == other.rank and self.suit == other.suit
        else:
            # don't attempt to compare against unrelated types
            return NotImplemented

    def __hash__(self):
        suit_index = JassCard.valid_suit.index(self.suit)
        rank_index = JassCard.valid_rank.index(self.rank)
        return rank_index + 100 * suit_index

    def __str__(self):
        ''' Get string representation of a card.

        Returns:
            string: the combination of rank and suit of a card. Eg: AS, 5H, JD, 3C, ...
        '''
        return f"{self.rank}-{self.suit}"

    def __repr__(self):
        return self.__str__()

    def __lt__(self, other):
        if self.suit != other.suit:
            return self.valid_suit.index(self.suit) < self.valid_suit.index(other.suit)
        return self.valid_rank.index(self.rank) < self.valid_rank.index(other.rank)

    def get_index(self):
        ''' Get index of a card.

        Returns:
            string: the combination of suit and rank of a card. Eg: 1S, 2H, AD, BJ, RJ...
        '''
        return self.suit + self.rank

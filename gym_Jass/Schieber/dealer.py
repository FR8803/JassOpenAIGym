from gym_Jass.Schieber.card import init_swiss_deck


class JassDealer(object):
    def __init__(self, np_random):
        self.np_random = np_random
        self.deck = init_swiss_deck()
        self.shuffle()

    def shuffle(self):
        self.np_random.shuffle(self.deck)

    def deal_cards(self, player, num):
        for _ in range(num):
            player.hand.append(self.deck.pop())
        player.hand = sorted(player.hand)

    def take_deck_back(self):
        self.deck = init_swiss_deck()
        self.shuffle()

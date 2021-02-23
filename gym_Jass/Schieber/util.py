import numpy as np

from gym_Jass.Schieber.card import JassSuits, JassCard

valid_rank = ["Ass", "König", "Ober", "Under", "Banner", "9", "8", "7", "6"]
valid_suit = [suit for suit in JassSuits]

RANK_MAP = {rank: i for rank, i in zip(valid_rank, range(len(valid_rank)))}
SUIT_MAP = {str(suit): i for suit, i in zip(valid_suit, range(len(valid_suit)))}


def cards2list(cards: [JassCard]):
    cards_list = []
    for card in cards:
        cards_list.append(str(card))
    return cards_list

def cards2dict(cards: [JassCard]):
    cards_dict = {}
    for card in cards:
        if card not in cards_dict:
            cards_dict[card] = 1
        else:
            cards_dict[card] += 1
    return cards_dict



def hand2dict(hand):
    hand_dict = {}
    for card in hand:
        if card not in hand_dict:
            hand_dict[card] = 1
        else:
            hand_dict[card] += 1
    return hand_dict


def encode_cards(plane, cards, type = "other"):
    # shape: (count, suit, rank)
    # must be ones (NOT zeros) to mark the cards that the player DOES NOT have
    # example: plane[0]["könig"]["rosen"] = 1
    # player has a count of "0" für "könig" of "rosen"
    plane[0] = np.ones((4, 9), dtype = int)
    if type == "hand":
        cards = hand2dict(cards)
        for card, count in cards.items():
            card_info = card.split("-")
            rank = card_info[0]
            suit = card_info[1]
            rank = RANK_MAP[rank]
            suit = SUIT_MAP[suit]
            plane[0][suit][rank] = 0
            plane[count][suit][rank] = 1
    else:
        cards = cards2dict(cards)
        for card, count in cards.items():
            card_info = card.split("-")
            rank = card_info[0]
            suit = card_info[1]
            rank = RANK_MAP[rank]
            suit = SUIT_MAP[suit]
            plane[0][suit][rank] = 0
            plane[count][suit][rank] = 1
    return plane

def encode_stich(plane, stich):
    Trumpf_pos = {x: i for i, x in enumerate(["Trumps.EICHEL", "Trumps.ROSE", "Trumps.SCHELLE", "Trumps.SCHILTE", "Trumps.OBENABE", "Trumps.UNDENUFE", "Trumps.GSCHOBE"])}
    plane[0][Trumpf_pos[stich]] = 1
    return plane


def tournament(env, num):
    payoffs = [0] * env.player_num
    counter = 0
    while counter < num:
        _trajectories, _payoffs = env.run(is_training=False)
        for i, _val in enumerate(payoffs):
            payoffs[i] += _payoffs[i]
        counter += 1
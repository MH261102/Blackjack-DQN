import random
import numpy as np

CARD_VALUES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]

def draw_card(deck):
    card = random.choice(deck)
    deck.remove(card)
    return card

def calc_hand_value(hand):
    total = sum(hand)
    num_aces = hand.count(1)
    while total <= 11 and num_aces > 0:
        total += 10
        num_aces -= 1
    is_soft = (1 in hand) and total <= 21
    return total, is_soft

class BlackjackEnv:
    def __init__(self):
        self.action_space = [0, 1, 2, 3]  # Hit, Stand, Double, Split
        self.reset()

    def reset(self):
        self.deck = CARD_VALUES * 4
        random.shuffle(self.deck)

        self.player = [draw_card(self.deck), draw_card(self.deck)]
        self.dealer = [draw_card(self.deck), draw_card(self.deck)]

        self.done = False
        self.split = False
        self.has_doubled = False

        # Handle natural blackjack
        player_total, _ = calc_hand_value(self.player)
        dealer_total, _ = calc_hand_value(self.dealer)

        if player_total == 21:
            self.done = True
            reward = 1.5 if dealer_total != 21 else 0.0
            return self.get_state(), reward, True

        return self.get_state(), 0.0, False

    def get_state(self):
        total, is_soft = calc_hand_value(self.player)
        dealer_up = self.dealer[0]
        card_counts = [self.deck.count(i) / 4.0 for i in range(1, 11)]  # normalized frequency
        return np.array([total / 21.0, int(is_soft), dealer_up / 10.0] + card_counts + [int(self.split)], dtype=np.float32)

    def valid_actions(self):
        actions = [0, 1]  # Hit, Stand
        if len(self.player) == 2:
            actions.append(2)  # Double
            if self.player[0] == self.player[1]:
                actions.append(3)  # Split
        return actions

    def step(self, action):
        assert action in self.valid_actions()

        if action == 0:  # Hit
            self.player.append(draw_card(self.deck))
            total, _ = calc_hand_value(self.player)
            if total > 21:
                self.done = True
                return self.get_state(), -1.0, True
            return self.get_state(), 0.0, False

        elif action == 1:  # Stand
            return self.dealer_turn()

        elif action == 2:  # Double
            self.player.append(draw_card(self.deck))
            self.has_doubled = True
            return self.dealer_turn()

        elif action == 3:  # Split
            self.split = True
            self.player = [self.player[0], draw_card(self.deck)]
            return self.get_state(), 0.0, False

    def dealer_turn(self):
        self.done = True
        while True:
            value, is_soft = calc_hand_value(self.dealer)
            if value < 17 or (value == 17 and is_soft):
                self.dealer.append(draw_card(self.deck))
            else:
                break

        player_total, _ = calc_hand_value(self.player)
        dealer_total, _ = calc_hand_value(self.dealer)

        blackjack = len(self.player) == 2 and player_total == 21

        if player_total > 21:
            reward = -1.0
        elif dealer_total > 21 or player_total > dealer_total:
            reward = 1.5 if blackjack else 1.0
        elif player_total == dealer_total:
            reward = 0.0
        else:
            reward = -0.8

        if self.has_doubled:
            reward *= 2

        return self.get_state(), reward, True

import random
import numpy as np
from conippets import json

__magic_square__ = np.array([
    2, 9, 4,
    7, 5, 3,
    6, 1, 8,
])
__init_state__ = np.array([
    '-', '-', '-',
    '-', '-', '-',
    '-', '-', '-',
], dtype='<U1')

class State:
    def __init__(self):
        self.X = []
        self.O = []
        self.possible_moves = list(range(9))

    def __str__(self):
        cur_state = __init_state__.copy()
        cur_state[self.X] = 'X'
        cur_state[self.O] = 'O'
        state = ''
        for i in range(0, 9, 3):
            state += ' '.join(cur_state[i:i+3]) + '\n'
        return state

    def copy(self):
        state = State()
        state.X = self.X.copy()
        state.O = self.O.copy()
        state.possible_moves = self.possible_moves.copy()
        return state

    def hash(self):
        cur_state = __init_state__.copy()
        cur_state[self.X] = 'X'
        cur_state[self.O] = 'O'
        return ''.join(cur_state)

def any_n_sum_to_k(n, k, lst):
    if n == 0:
        return k == 0
    if k < 0 or not lst:
        return False
    return any_n_sum_to_k(n-1, k - lst[0], lst[1:]) or any_n_sum_to_k(n, k, lst[1:])

class Game:
    def __init__(self, *, alpha, epsilon):
        self.value = {}
        self.alpha = alpha
        self.epsilon = epsilon

    def init_value(self, state, value):
        c = 'X' if value == 1 else 'O'
        for i in range(9):
            if state[i] == '-':
                state[i] = c
                key = ''.join(state)
                self.value[key] = 0.5
                self.init_value(state, -value)
                state[i] = '-'

    def step(self, cur_state, player, move):
        state = cur_state.copy()
        X_moves = state.X
        O_moves = state.O

        target = X_moves if player == 'X' else O_moves
        target.append(move)
        state.possible_moves.remove(move)
        return state

    def next_state(self, cur_state, player, move):
        state = self.step(cur_state, player, move)

        x_win = any_n_sum_to_k(3, 15, list(__magic_square__[state.X]))
        o_win = any_n_sum_to_k(3, 15, list(__magic_square__[state.O]))
        total = len(state.X) + len(state.O)

        if x_win or total >= 9:
            val = 0.0
        elif o_win:
            val = 1.0
        else:
            val = 0.5

        self.value[state.hash()] = val
        return state

    def update(self, old_state, new_state):
        old_val = self.value[old_state.hash()]
        new_val = self.value[new_state.hash()]
        updated = old_val + self.alpha * (new_val - old_val)
        self.value[old_state.hash()] = updated

    def random_move(self, state):
        moves = state.possible_moves
        return random.choice(moves) if moves else None

    def greedy_move(self, player, state):
        moves = state.possible_moves
        if not moves:
            return None

        best_val = -1
        best_move = None
        for move in moves:
            new_state = self.next_state(state, player, move)
            move_val = self.value[new_state.hash()]
            if move_val > best_val:
                best_val = move_val
                best_move = move
        return best_move

    def is_terminal_state(self, state):
        v = self.value[state.hash()]
        return v == 1.0 or v == 0.0

    def run(self):
        state = State()

        while True:
            x_move = self.random_move(state)
            if x_move is None:
                break

            new_state = self.next_state(state, 'X', x_move)
            self.update(state, new_state)
            if self.is_terminal_state(new_state):
                break

            exploratory = random.random() < self.epsilon
            o_move = self.random_move(new_state) if exploratory else self.greedy_move('O', new_state)
            if o_move is None:
                break

            o_new_state = self.next_state(new_state, 'O', o_move)
            if not exploratory:
                self.update(new_state, o_new_state)

            if self.is_terminal_state(o_new_state):
                return self.value[o_new_state.hash()]

            state = o_new_state
        return self.value[state.hash()]

def train(game):
    game.value[State().hash()] = 0.5
    print('init state value...')
    game.init_value(__init_state__.copy(), 1)
    print(f'state value size: {len(game.value)}')
    window = 5000
    cache_vals = [0] * window
    mean_val = 0
    for i in range(1000000):
        cur_val = game.run()
        last_val = cache_vals.pop(0)
        mean_val = mean_val + (cur_val - last_val) / window
        cache_vals.append(cur_val)
        if i % window == 0:
            print(mean_val)
    json.write('./state_value.json', game.value)

def play(game):
    game.value.update(json.read('./state_value.json'))

    state = State()
    while True:
        print(state)
        x_move = int(input('X move: '))
        state = game.step(state, 'X', x_move)
        x_win = any_n_sum_to_k(3, 15, list(__magic_square__[state.X]))
        if x_win:
            print(f'{state}Win')
            break

        o_move = game.greedy_move('O', state)
        state = game.step(state, 'O', o_move)
        o_win = any_n_sum_to_k(3, 15, list(__magic_square__[state.O]))
        if o_win:
            print(f'{state}Lose')
            break

        if len(state.X) + len(state.O) == 9:
            print(f'{state}Tie')
            break

if __name__ == '__main__':
    game = Game(alpha=0.5, epsilon=0.5)
    train(game)
    play(game)

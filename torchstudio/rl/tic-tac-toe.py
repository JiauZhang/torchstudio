import random
import numpy as np

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
        return ''.join(cur_state.reshape)

class Value(dict):
    def __init__(self):
        super().__init__()

    def __contains__(self, state):
        return state.hash() in self

    def __getattr__(self, item):
        hash = item.hash()
        if hash in self:
            return self[hash]
        else:
            return None

    def __setattr__(self, item, value):
        self[item.hash()] = value

def any_n_sum_to_k(n, k, lst):
    if n == 0:
        return k == 0
    if k < 0 or not lst:
        return False
    return any_n_sum_to_k(n-1, k - lst[0], lst[1:]) or any_n_sum_to_k(n, k, lst[1:])

class Game:
    def __init__(self, *, alpha, epsilon):
        self.value = Value()
        self.alpha = alpha
        self.epsilon = epsilon

    def next_state(self, cur_state, player, move):
        state = cur_state.copy()
        X_moves = state.X
        O_moves = state.O

        target = X_moves if player == 'X' else O_moves
        target.append(move)
        state.possible_moves.remove(move)

        x_win = any_n_sum_to_k(3, 15, list(__magic_square__[X_moves]))
        o_win = any_n_sum_to_k(3, 15, list(__magic_square__[O_moves]))
        total = len(X_moves) + len(O_moves)

        if x_win or total >= 9:
            val = 0.0
        elif o_win:
            val = 1.0
        else:
            val = 0.5

        self.value[state] = val
        return state

    def update(self, old_state, new_state):
        old_val = self.value[old_state]
        new_val = self.value[new_state]
        updated = old_val + self.alpha * (new_val - old_val)
        self.value[old_state] = updated

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
            move_val = self.value[new_state]
            if move_val > best_val:
                best_val = move_val
                best_move = move
        return best_move

    def is_terminal_state(self, state):
        v = self.value[state]
        return v == 1.0 or v == 0.0

    def run(self):
        state = State()

        while True:
            x_move = self.random_move(state)
            if x_move is None:
                break

            new_state = self.next_state(state, 'X', x_move)
            if self.is_terminal_state(new_state):
                self.update(state, new_state)
                return self.value[new_state]

            exploratory = random.random() < self.epsilon
            o_move = self.random_move(new_state) if exploratory else self.greedy_move('O', new_state)
            if o_move is None:
                break

            o_new_state = self.next_state(new_state, 'O', o_move)
            if not exploratory:
                self.update(new_state, o_new_state)

            if self.is_terminal_state(o_new_state):
                return self.value[o_new_state]

            state = o_new_state
        return self.value[state]

if __name__ == '__main__':
    game = Game(alpha=0.5, epsilon=0.01)
    window = 5000
    cache_vals = [0] * window
    mean_val = 0
    for i in range(100000):
        cur_val = game.run()
        last_val = cache_vals.pop(0)
        mean_val = mean_val + (cur_val - last_val) / window
        cache_vals.append(cur_val)
        if i % window == 0:
            print(mean_val)

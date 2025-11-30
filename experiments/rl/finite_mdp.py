import random, sys
import numpy as np

class FiniteMDP:
    ACTION_NORTH = 0
    ACTION_SOUTH = 1
    ACTION_WEST = 2
    ACTION_EAST = 3

    def __init__(self):
        self.grid = 5
        self.actions = [self.ACTION_NORTH, self.ACTION_SOUTH, self.ACTION_WEST, self.ACTION_EAST]
        self.gamma = 0.9
        self.A = np.array([0, 1], dtype=np.int32)
        self.B = np.array([0, 3], dtype=np.int32)

    def environment(self, state, action):
        state = state.copy()
        # state: [x, y] --> [row, col]
        if (state == self.A).all():
            return np.array([4, 1], dtype=np.int32), 10
        if (state == self.B).all():
            return np.array([2, 3], dtype=np.int32), 5

        match action:
            case self.ACTION_NORTH:
                state[0] -= 1
            case self.ACTION_SOUTH:
                state[0] += 1
            case self.ACTION_EAST:
                state[1] += 1
            case self.ACTION_WEST:
                state[1] -= 1
            case _:
                raise ValueError(f'unsupported action type: {action}')

        new_state = np.clip(state, 0, self.grid - 1)
        if not (new_state == state).all():
            return new_state, -1
        return state, 0

    def random_policy(self, state):
        return random.choice(self.actions)

    def figure_3_2(self):
        state_value = np.zeros((self.grid, self.grid), dtype=np.float32)
        diff = np.inf
        loop = 0
        np.set_printoptions(precision=3, suppress=True)
        print(f'loop: {loop}, diff: {diff}\n{state_value}')
        while diff > 1e-3:
            new_value = np.zeros_like(state_value)
            loop += 1
            for x in range(self.grid):
                for y in range(self.grid):
                    for action in self.actions:
                        state = np.array([x, y], dtype=np.int32)
                        new_state, r = self.environment(state, action)
                        new_value[state[0], state[1]] += 0.25 * (r + self.gamma * state_value[new_state[0], new_state[1]])
            diff = np.abs(state_value - new_value).mean()
            state_value = new_value
            sys.stdout.write(f'\033[{self.grid+1}F\033[0J')
            print(f'loop: {loop}, diff: {diff:.6f}\n{state_value}\033[?25l')
            sys.stdout.flush()

if __name__ == '__main__':
    mdp = FiniteMDP()
    mdp.figure_3_2()
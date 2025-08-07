import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class Bandit:
    def __init__(self):
        self.mean_action_value = np.array([0.3, -0.8, 1.6, 0.5, 1.2, -1.5, -0.2, -1.1, 0.9, -0.7])

    def figure_2_1(self):
        samples = np.random.randn(2000, 10) + self.mean_action_value
        plt.violinplot(samples)
        plt.xlabel('Action')
        plt.ylabel('Reward distribution')
        plt.savefig('./figure_2_1.png')
        plt.close()

    def reward(self, action):
        mean = self.mean_action_value[action]
        return np.random.randn(1) + mean

    def update_q(self, action, reward, lr=None):
        q_reward = self.q_action_value[action]
        self.action_count[action] += 1
        step_size = lr if lr else 1 / self.action_count[action]
        q_reward = q_reward + step_size * (reward - q_reward)
        self.q_action_value[action] = q_reward[0]

    def greedy_policy(self):
        action = 0
        value = self.q_action_value[action]
        for i in range(len(self.q_action_value)):
            if self.q_action_value[i] > value:
                value = self.q_action_value[i]
                action = i
        return action

    def random_policy(self):
        return random.choice(range(len(self.q_action_value)))

    def greedy_train(self, steps, epsilon=0, init_value=0, lr=None):
        self.q_action_value = np.ones_like(self.mean_action_value) * init_value
        self.action_count = np.zeros_like(self.mean_action_value)
        mean_reward = np.zeros((steps + 1, 1))
        mean_optimal_action = np.zeros(steps + 1)
        optimal_action = np.argmax(self.mean_action_value)

        for n in range(1, steps+1):
            if random.random() > epsilon:
                action = self.greedy_policy()
            else:
                action = self.random_policy()

            if action == optimal_action:
                mean_optimal_action[n] = 1

            reward = self.reward(action)
            mean_reward[n] = mean_reward[n - 1] + (reward - mean_reward[n - 1]) / n
            self.update_q(action, reward, lr=lr)

        return np.array(mean_reward), mean_optimal_action

    def figure_2_2(self, runs=500, steps=1000):
        mean_reward = []
        mean_optimal_action = []
        epsilon = [0.0, 0.01, 0.1]
        color = ['g', 'r', 'b']
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for eps, c in zip(epsilon, color):
            for _ in tqdm(range(runs)):
                reward, optimal_action = self.greedy_train(steps, epsilon=eps)
                mean_reward.append(reward)
                mean_optimal_action.append(optimal_action)
            mean_reward = np.array(mean_reward).mean(axis=0)
            mean_optimal_action = np.array(mean_optimal_action).mean(axis=0)
            axes[0].plot(mean_reward, color=c, label=f'$\\epsilon = {eps:.02f}$')
            axes[1].plot(mean_optimal_action, color=c, label=f'$\\epsilon = {eps:.02f}$')
            mean_reward = []
            mean_optimal_action = []

        axes[0].set_xlabel('steps')
        axes[0].set_ylabel('average reward')
        axes[0].legend()
        axes[1].set_xlabel('steps')
        axes[1].set_ylabel('optimal action')
        axes[1].legend()
        plt.savefig('./figure_2_2.png')
        plt.close()

    def figure_2_3(self, runs=500, steps=1000):
        mean_optimal_action = []
        epsilon = [0.0, 0.0, 0.1]
        # -2 < all true mean value
        init_value = [-2, 5, 0]
        color = ['g', 'r', 'b']
        _, axes = plt.subplots(1, 1, figsize=(12, 5))
        for eps, c, value in zip(epsilon, color, init_value):
            for _ in tqdm(range(runs)):
                _, optimal_action = self.greedy_train(steps, epsilon=eps, init_value=value, lr=0.1)
                mean_optimal_action.append(optimal_action)
            mean_optimal_action = np.array(mean_optimal_action).mean(axis=0)
            axes.plot(mean_optimal_action, color=c, label=f'$Q_1 = {value}, \\epsilon = {eps:.02f}$')
            mean_optimal_action = []

        axes.set_xlabel('steps')
        axes.set_ylabel('optimal action')
        axes.legend()
        plt.savefig('./figure_2_3.png')
        plt.close()

    def ucb_policy(self, c, t):
        action = 0
        max_action_value = self.q_action_value[0]
        for i in range(len(self.q_action_value)):
            if self.action_count[i] == 0:
                action = i
                break
            else:
                action_value = self.q_action_value[i] + c * np.sqrt(np.log(t) / self.action_count[i])
                if action_value > max_action_value:
                    action = i
                    max_action_value = action_value
        return action

    def ucb_train(self, steps, c, init_value=0, lr=None):
        self.q_action_value = np.ones_like(self.mean_action_value) * init_value
        self.action_count = np.zeros_like(self.mean_action_value)
        mean_reward = np.zeros((steps + 1, 1))

        for n in range(1, steps+1):
            action = self.ucb_policy(c, n)
            reward = self.reward(action)
            mean_reward[n] = mean_reward[n - 1] + (reward - mean_reward[n - 1]) / n
            self.update_q(action, reward, lr=lr)

        return np.array(mean_reward)

    def figure_2_4(self, runs=500, steps=1000):
        mean_reward = []
        train_fn = [self.greedy_train, self.ucb_train, self.ucb_train]
        train_kwargs = [{'epsilon': 0.1}, {'c': 2}, {'c': 1}]
        label_template = [
            '$\\epsilon$-greedy $\\epsilon = {epsilon}$',
            'UCB $c = {c}$',
            'UCB $c = {c}$',
        ]
        color = ['g', 'r', 'b']
        _, axes = plt.subplots(1, 1, figsize=(12, 5))
        for kwargs, c, train, label in zip(train_kwargs, color, train_fn, label_template):
            for _ in tqdm(range(runs)):
                reward = train(steps, **kwargs)
                if isinstance(reward, tuple): reward = reward[0]
                mean_reward.append(reward)
            mean_reward = np.array(mean_reward).mean(axis=0)
            axes.plot(mean_reward, color=c, label=label.format(**kwargs))
            mean_reward = []

        axes.set_xlabel('steps')
        axes.set_ylabel('average reward')
        axes.legend()
        plt.savefig('./figure_2_4.png')
        plt.close()

if __name__ == '__main__':
    bandit = Bandit()
    bandit.figure_2_1()
    bandit.figure_2_2()
    bandit.figure_2_3()
    bandit.figure_2_4()
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

    def update_q(self, action, reward):
        q_reward = self.q_action_value[action]
        self.action_count[action] += 1
        q_reward = q_reward + (reward - q_reward) / self.action_count[action]
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

    def train(self, steps, epsilon=0):
        self.q_action_value = np.zeros_like(self.mean_action_value)
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
            self.update_q(action, reward)

        return np.array(mean_reward), mean_optimal_action

    def figure_2_2(self, runs=500, steps=1000):
        mean_reward = []
        mean_optimal_action = []
        epsilon = [0.0, 0.01, 0.1]
        color = ['g', 'r', 'b']
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for eps, c in zip(epsilon, color):
            for _ in tqdm(range(runs)):
                reward, optimal_action = self.train(steps, epsilon=eps)
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

if __name__ == '__main__':
    bandit = Bandit()
    bandit.figure_2_1()
    bandit.figure_2_2()
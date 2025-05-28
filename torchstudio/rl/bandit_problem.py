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
        mean_reward = [np.array([0.0])]

        for n in tqdm(range(1, steps+1)):
            if random.random() > epsilon:
                action = self.greedy_policy()
            else:
                action = self.random_policy()
            reward = self.reward(action)
            mean_reward.append(mean_reward[-1] + (reward - mean_reward[-1]) / n)
            self.update_q(action, reward)

        return np.array(mean_reward)

    def figure_2_2(self, runs=500):
        mean_reward = []
        optimal_action = []
        epsilon = [0.0, 0.01, 0.1]
        color = ['g', 'r', 'b']
        for eps, c in zip(epsilon, color):
            for _ in range(runs):
                mean_reward.append(self.train(1000, epsilon=eps))
            mean_reward = np.array(mean_reward).mean(axis=0)
            plt.plot(mean_reward, color=c, label=f'$\\epsilon = {eps:.02f}$')
            mean_reward = []

        plt.xlabel('steps')
        plt.ylabel('average reward')
        plt.legend()
        plt.savefig('./mean_reward.png')
        plt.close()

if __name__ == '__main__':
    bandit = Bandit()
    bandit.figure_2_1()
    bandit.figure_2_2()
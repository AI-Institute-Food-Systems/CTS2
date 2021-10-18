from rllab.envs.base import Env
from rllab.spaces import Box
from rllab.envs.base import Step
import numpy as np

data_num = 200
after_current = 2
start_index = 2
window_size = after_current+start_index+1

# mobility: by adjusting x, we can get different curve shapes
x1 = np.linspace(0, np.pi, data_num)
x2_1 = np.linspace(0, np.pi/2, int(data_num/2))
x2_2 = np.linspace(0, 2*np.pi, int(data_num/2))
x3 = np.linspace(0, 4 * np.pi, data_num)
lowest = -1-0.5*0.2
highest = 1+0.5*0.2
constraint = 0.1
d3_constaint = 0.3

class SmoothEnv(Env):
    @property
    def observation_space(self):
        return Box(low=lowest, high=highest, shape=(window_size*3,))

    @property
    def action_space(self):
        return Box(low=-constraint, high=constraint, shape=(3, ))

    def reset(self):
        # generate noise curve with uniform noise
        # self.y1 = np.sin(x1) + (np.random.random(data_num) - 0.5) * 0.2
        # self.y2_1 = np.sin(x2_1) + (np.random.random(int(data_num/2)) - 0.5) * 0.2
        # self.y2_2 = np.cos(x2_2) + (np.random.random(int(data_num/2)) - 0.5) * 0.2
        # self.y2 = np.append(self.y2_1, self.y2_2)
        # self.y3 = np.sin(x3) + (np.random.random(data_num) - 0.5) * 0.2

        # generate noise with gaussian noise
        mu = 0
        sigma = 0.05
        self.y1 = np.sin(x1) + np.clip(np.random.normal(mu, sigma, data_num), -0.1, 0.1)
        self.y2_1 = np.sin(x2_1) + np.clip(np.random.normal(mu, sigma, int(data_num / 2)), -0.1, 0.1)
        self.y2_2 = np.cos(x2_2) + np.clip(np.random.normal(mu, sigma, int(data_num / 2)), -0.1, 0.1)
        self.y2 = np.append(self.y2_1, self.y2_2)
        self.y3 = np.sin(x3) + np.clip(np.random.normal(mu, sigma, data_num), -0.1, 0.1)

        self.y1_state = np.copy(self.y1)
        self.y2_state = np.copy(self.y2)
        self.y3_state = np.copy(self.y3)

        self.index = start_index
        observation = np.copy(np.concatenate((self.y1_state[0:window_size],
                                              self.y2_state[0:window_size],
                                              self.y3_state[0:window_size])))
        # print('=====reset observation:',observation)
        return observation

    def three_points_angle(self, three_points):
        a = np.array([0, three_points[0]])
        b = np.array([1, three_points[1]])
        c = np.array([2, three_points[2]])

        ab = b - a
        bc = c - b

        cosine_angle = np.dot(ab, bc) / (np.linalg.norm(ab) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)
        return -angle

    def step_reward(self, estimation):
        angle_sum = 0
        for i in range(len(estimation)-2):
            angle_sum = angle_sum + self.three_points_angle(estimation[i:i+3])
        return angle_sum

    def step(self, action):
        # if action is infeasible, do the projection
        if 2 * abs(action[0]) + abs(action[1]) + abs(action[2]) > d3_constaint:
            action = d3_constaint / (2 * abs(action[0]) + abs(action[1]) + abs(action[2])) * action  # linear projection
            # print('=====projected action:', action)

        self.y1_state[self.index] = self.y1_state[self.index] + action[0]
        self.y2_state[self.index] = self.y2_state[self.index] + action[1]
        self.y3_state[self.index] = self.y3_state[self.index] + action[2]

        estimation1 = np.copy(self.y1_state[self.index - start_index:self.index + 1])
        reward = self.step_reward(estimation1)
        estimation2 = np.copy(self.y2_state[self.index - start_index:self.index + 1])
        reward = reward + self.step_reward(estimation2)
        estimation3 = np.copy(self.y3_state[self.index - start_index:self.index + 1])
        reward = reward + self.step_reward(estimation3)

        if self.index - start_index + window_size < data_num:
            done = False
        else:
            done = True

        self.index = self.index + 1
        next_observation = np.copy(
            np.concatenate((self.y1_state[self.index - start_index:self.index - start_index + window_size],
                      self.y2_state[self.index - start_index:self.index - start_index + window_size],
                      self.y3_state[self.index - start_index:self.index - start_index + window_size])))
        # print('=====next observation:', next_observation)

        return Step(observation=next_observation, reward=reward, done=done,
                    env_y1 = np.copy(self.y1), env_y1_estimation = np.copy(self.y1_state),
                    env_y2 = np.copy(self.y2), env_y2_estimation = np.copy(self.y2_state),
                    env_y3 = np.copy(self.y3), env_y3_estimation = np.copy(self.y3_state),
                    env_action = np.copy(action), env_reward = reward
                    )

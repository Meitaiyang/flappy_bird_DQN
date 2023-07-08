import random
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ple import PLE
from ple.games import FlappyBird

device = torch.device("cuda:3")

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__(self, n_states, n_actions, n_hidden)
        
        self.fc1 = nn.Linear(3, 64*4)



class Agent():
    def __init__(self, action_set):
        self.gamma = 1
        self.model = self.init_netWork()
        self.batch_size = 128
        self.memory = deque(maxlen=2000000)
        self.greedy = 1
        self.action_set = action_set

    def get_state(self, state):
        """
        提取遊戲state中我們需要的資料
        :param state: 遊戲state
        :return: 返回提取好的資料
        """
        return_state = np.zeros((3,))
        dist_to_pipe_horz = state["next_pipe_dist_to_player"]
        dist_to_pipe_bottom = state["player_y"] - state["next_pipe_top_y"]
        velocity = state['player_vel']
        return_state[0] = dist_to_pipe_horz
        return_state[1] = dist_to_pipe_bottom
        return_state[2] = velocity
        return return_state

    def init_netWork(self):
        """
        構建模型
        :return:
        """
        model = Sequential()
        model.add(Dense(64 * 4, activation="tanh", input_shape=(3,)))
        model.add(Dense(64 * 4, activation="tanh"))
        model.add(Dense(2, activation="linear"))

        model.compile(loss=keras.losses.mean_squared_error,
                      optimizer=keras.optimizers.RMSprop(lr=0.001))
        return model

    def train_model(self):
        if len(self.memory) < 2500:
            return

        train_sample = random.sample(self.memory, k=self.batch_size)
        train_states = []
        next_states = []

        for sample in train_sample:
            cur_state, action, r, next_state, done = sample
            next_states.append(next_state)
            train_states.append(cur_state)
        # 轉成np陣列
        next_states = np.array(next_states)
        train_states = np.array(train_states)

        # 得到下一個state的q值
        next_states_q = self.model.predict(next_states)
        # 得到預測值
        state_q = self.model.predict_on_batch(train_states)

        for index, sample in enumerate(train_sample):
            cur_state, action, r, next_state, done = sample
            # 計算Q現實
            if not done:
                state_q[index][action] = r + self.gamma * np.max(next_states_q[index])
            else:
                state_q[index][action] = r
        self.model.train_on_batch(train_states, state_q)

    def add_memory(self, sample):
        self.memory.append(sample)

    def update_greedy(self):
        if self.greedy > 0.01:
            self.greedy *= 0.995

    def get_best_action(self, state):
        if random.random() < self.greedy:
            return random.randint(0, 1)
        else:
            return np.argmax(self.model.predict(state.reshape(-1, 3)))

    def act(self, p, action):
        """
        執行動作
        :param p: 通過p來向遊戲發出動作命令
        :param action: 動作
        :return: 獎勵
        """
        r = p.act(self.action_set[action])
        if r == 0:
            r = 1
        if r == 1:
            r = 100
        else:
            r = -1000
        return r


if __name__ == "__main__":
    # 訓練次數
    episodes = 20000
    # 例項化遊戲物件
    game = FlappyBird()
    # 類似遊戲的一個介面，可以為我們提供一些功能
    p = PLE(game, fps=30, display_screen=False)
    # 初始化
    p.init()
    # 例項化Agent，將動作集傳進去
    agent = Agent(p.getActionSet())
    max_score = 0
    scores = deque(maxlen=10)

    for episode in range(episodes):
        # 重置遊戲
        p.reset_game()
        # 獲得狀態
        state = agent.get_state(game.getGameState())
        if episode > 150:
            agent.update_greedy()
        while True:
            # 獲得最佳動作
            action = agent.get_best_action(state)
            # 然後執行動作獲得獎勵
            reward = agent.act(p, action)
            # 獲得執行動作之後的狀態
            next_state = agent.get_state(game.getGameState())
            agent.add_memory((state, action, reward, next_state, p.game_over()))
            agent.train_model()
            state = next_state
            if p.game_over():
                # 獲得當前分數
                current_score = p.score()
                max_score = max(max_score, current_score)
                scores.append(current_score)
                print('第%s次遊戲，得分為: %s,最大得分為: %s' % (episode, current_score, max_score))
                if len(scores) == 10 and np.mean(scores) > 150:
                    agent.model.save("bird_model.h5")
                    exit(0)
                break

import numpy as np
from ple import PLE
from ple.games.flappybird import FlappyBird
import random
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

device = torch.device("cuda:2")

class Net(nn.Module):
    def __init__(self, n_states, n_actions, n_hidden):
        super(Net, self).__init__()

        # 輸入層 (state) 到隱藏層，隱藏層到輸出層 (action)
        self.fc1 = nn.Linear(n_states, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.out = nn.Linear(n_hidden, n_actions)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x) # ReLU activation
        for layer in range(5):
            x = self.fc2(x)
            x = F.relu(x)
        actions_value = self.out(x)
        return actions_value

class DQN(object):
    def __init__(self, n_states, n_actions, n_hidden, batch_size, lr, epsilon, gamma, target_replace_iter, memory_capacity):
        self.eval_net, self.target_net = Net(n_states, n_actions, n_hidden).to(device), Net(n_states, n_actions, n_hidden).to(device)

        self.memory = np.zeros((memory_capacity, n_states * 2 + 2)) # 每個 memory 中的 experience 大小為 (state + next state + reward + action)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=lr)
        self.loss_func = nn.MSELoss()
        self.memory_counter = 0
        self.learn_step_counter = 0 # 讓 target network 知道什麼時候要更新

        self.n_states = n_states
        self.n_actions = n_actions
        self.n_hidden = n_hidden
        self.batch_size = batch_size
        self.lr = lr
        self.epsilon = epsilon
        self.gamma = gamma
        self.target_replace_iter = target_replace_iter
        self.memory_capacity = memory_capacity

    def choose_action(self, state):
        x = torch.unsqueeze(torch.FloatTensor(state).to(device), 0)
        # epsilon-greedy
        if np.random.uniform() < self.epsilon: # 隨機
            action = np.random.randint(0, self.n_actions)
        else: # 根據現有 policy 做最好的選擇
            actions_value = self.eval_net(x) # 以現有 eval net 得出各個 action 的分數
            action = torch.max(actions_value, 1)[1].cpu().data.numpy()[0] # 挑選最高分的 action
            # print(actions_value)
        return action
    
    def store_transition(self, state, action, reward, next_state):
        # 打包 experience
        transition = np.hstack((state, [action, reward], next_state))

        # 存進 memory；舊 memory 可能會被覆蓋
        index = self.memory_counter % self.memory_capacity
        self.memory[index, :] = transition
        self.memory_counter += 1
    
    def update_epsilon(self):
        if self.epsilon > 0.01:
            self.epsilon *= 0.905

    def learn(self):
        # 隨機取樣 batch_size 個 experience
        sample_index = np.random.choice(self.memory_capacity, self.batch_size)
        b_memory = self.memory[sample_index, :]
        b_state = torch.FloatTensor(b_memory[:, :self.n_states]).to(device)
        b_action = torch.LongTensor(b_memory[:, self.n_states:self.n_states+1].astype(int)).to(device)
        b_reward = torch.FloatTensor(b_memory[:, self.n_states+1:self.n_states+2]).to(device)
        b_next_state = torch.FloatTensor(b_memory[:, -self.n_states:]).to(device)

        # 計算現有 eval net 和 target net 得出 Q value 的落差
        q_eval = self.eval_net(b_state).gather(1, b_action) # 重新計算這些 experience 當下 eval net 所得出的 Q value
        q_next = self.target_net(b_next_state).detach() # detach 才不會訓練到 target net
        q_target = b_reward + self.gamma * q_next.max(1)[0].view(self.batch_size, 1) # 計算這些 experience 當下 target net 所得出的 Q value
        loss = self.loss_func(q_eval, q_target)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 每隔一段時間 (target_replace_iter), 更新 target net，即複製 eval net 到 target net
        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_replace_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())

    def saving_model(self):
        
        torch.save(self.eval_net, "save/eval_net")
        torch.save(self.target_net, "save/target_net")
        np.save("save/memory", self.memory)


    def loading_model(self):
        self.eval_net = torch.load("save/eval_net")
        self.target_net = torch.load("save/target_net")
        self.memory = np.load("save/memory.npy")

def get_reward(r):
    if r == 0:
        r = 1
    if r == 1:
        r = 10
    else:
        r = -1000
    return r


if __name__ == "__main__":
    
    display = False
    
    game = FlappyBird()
    p = PLE(game, fps=30, display_screen=display, force_fps=not(display))
    p.init()
    
    n_actions = len(p.getActionSet())
    n_states = len(game.getGameState())

    action_set = p.getActionSet()
    n_hidden = 64 * 4
    batch_size = 128
    lr = 0.001                 # learning rate
    epsilon = 0.5             # epsilon-greedy
    gamma = 0.9               # reward discount factor
    target_replace_iter = 100 # target network 更新間隔
    memory_capacity = 2500
    # n_episodes = 4000
    
    dqn = DQN(n_states, n_actions, n_hidden, batch_size, lr, epsilon, gamma, target_replace_iter, memory_capacity)
    # dqn.loading_model()
    # for i_episode in range(n_episodes):
    while True:
        
        for die in range(100):
            t = 0
            rewards = 0
            state = list(game.getGameState().values())
            while True:
                if t > 5:
                    dqn.update_epsilon()
                action = dqn.choose_action(state)
                reward = get_reward(p.act(action_set[action]))
                next_state =  list(game.getGameState().values())
                dqn.store_transition(state, action, reward, next_state)
                rewards += reward


                if dqn.memory_counter > memory_capacity:
                    dqn.learn()
                # dqn.learn()
                current_score = p.score()+5
                state = next_state
            
                if p.game_over():
                    print('Episode finished after {} timesteps, total rewards {}, current_score = {}'.format(t+1, rewards, current_score))
                    p.reset_game()
                    break

                t += 1
        dqn.saving_model()

"""
    for episode in range(episodes):
        p.reset_game()
        state = agent.get_state(game.getGameState())
        agent.update_greedy()
        while True:
            action = agent.get_best_action(state)
            reward = agent.act(p, action)
            next_state = agent.get_state(game.getGameState())
            agent.update_q_table(state, action, next_state, reward)
            current_score = p.score()
            state = next_state
            if p.game_over():
                max_score = max(current_score, max_score)
                print('Episodes: %s, Current score: %s, Max score: %s' % (episode, current_score, max_score))
                if current_score > 300:
                    np.save("{}_{}.npy".format(current_score, episode), agent.q_table)
                break
"""

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import pandas as pd
from ple import PLE
from ple.games.flappybird import FlappyBird

device = torch.device("cuda:2")

class Net(nn.Module):
    def __init__(self, n_states, n_actions, n_hidden):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_states, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.out = nn.Linear(n_hidden, n_actions)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        for _ in range(5):
            x = self.fc2(x)
            x = F.relu(x)
        actions_value = self.out(x)
        return actions_value

class DQN(object):
    def __init__(self, n_states, n_actions, n_hidden, batch_size, lr, epsilon, gamma, target_replace_iter, memory_capacity):
        self.eval_net = Net(n_states, n_actions, n_hidden).to(device)
        self.target_net = Net(n_states, n_actions, n_hidden).to(device)
        self.memory = np.zeros((memory_capacity, n_states * 2 + 2))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=lr)
        self.loss_func = nn.MSELoss()
        self.memory_counter = 0
        self.learn_step_counter = 0

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
        if np.random.uniform() < self.epsilon:
            action = np.random.randint(0, self.n_actions)
        else:
            actions_value = self.eval_net(x)
            action = torch.max(actions_value, 1)[1].cpu().data.numpy()[0]
        return action
    
    def store_transition(self, state, action, reward, next_state):
        transition = np.hstack((state, [action, reward], next_state))
        index = self.memory_counter % self.memory_capacity
        self.memory[index, :] = transition
        self.memory_counter += 1
        
    def learn(self):
        sample_index = np.random.choice(self.memory_capacity, self.batch_size)
        b_memory = self.memory[sample_index, :]
        b_state = torch.FloatTensor(b_memory[:, :self.n_states]).to(device)
        b_action = torch.LongTensor(b_memory[:, self.n_states:self.n_states+1].astype(int)).to(device)
        b_reward = torch.FloatTensor(b_memory[:, self.n_states+1:self.n_states+2]).to(device)
        b_next_state = torch.FloatTensor(b_memory[:, -self.n_states:]).to(device)
        
        q_eval = self.eval_net(b_state).gather(1, b_action)
        q_next = self.target_net(b_next_state).detach()
        q_target = b_reward + self.gamma * q_next.max(1)[0].view(self.batch_size, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

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
    lr = 0.001
    epsilon = 0.05
    gamma = 0.9
    target_replace_iter = 100
    memory_capacity = 2500
    n_episodes = 100

    recordResult = dict()
    
    for round in range(5):
        dqn = DQN(n_states, n_actions, n_hidden, batch_size, lr, epsilon, gamma, target_replace_iter, memory_capacity)
        recordlist = list()
        for i_episode in range(n_episodes):
            for die in range(100):
                t = 0
                rewards = 0
                state = list(game.getGameState().values())
                while True:
                    action = dqn.choose_action(state)
                    reward = get_reward(p.act(action_set[action]))
                    next_state = list(game.getGameState().values())
                    dqn.store_transition(state, action, reward, next_state)
                    rewards += reward
                    if dqn.memory_counter > memory_capacity:
                        dqn.learn()
                    current_score = int(p.score()) + 5
                    state = next_state
                
                    if p.game_over():
                        recordlist.append(current_score)
                        print('Episode finished after {} timesteps, total rewards {}, current_score = {}'.format(t+1, rewards, current_score))
                        p.reset_game()
                        break

                    t += 1
        recordResult[f'round_{round+1}'] = recordlist
    save = pd.DataFrame(recordResult)
    save.to_csv('save/result.csv', index=False)

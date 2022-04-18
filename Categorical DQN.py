import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Replay_Buffer:
    def __init__(self, input_dims, buffer_size, batch_size):
        self.counter = 0
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.state_memory = np.zeros((buffer_size, *input_dims), dtype = np.float32)
        self.action_memory = np.zeros(buffer_size, dtype = np.int8)
        self.reward_memory = np.zeros(buffer_size, dtype = np.float32)
        self.next_state_memory = np.zeros((buffer_size, *input_dims), dtype = np.float32)
        self.terminal_state_memory = np.zeros(buffer_size, dtype = np.bool8)
    
    def store(self, state, action, reward, next_state, is_done):
        if self.is_full():
            return None
        self.state_memory[self.counter] = state
        self.action_memory[self.counter] = action
        self.reward_memory[self.counter] = reward
        self.next_state_memory[self.counter] = next_state
        self.terminal_state_memory[self.counter] = is_done
        self.counter += 1
    
    def sample_batch(self):
        batch_index = np.random.choice(self.buffer_size, self.batch_size, False)
        return dict(
            state_batch = torch.tensor(self.state_memory[batch_index], dtype = torch.float32),
            action_batch = torch.tensor(self.action_memory[batch_index], dtype = torch.int64),
            reward_batch = torch.tensor(self.reward_memory[batch_index], dtype = torch.float32),
            next_state_batch = torch.tensor(self.next_state_memory[batch_index], dtype = torch.float32),
            terminal_state_batch = torch.tensor(self.terminal_state_memory[batch_index], dtype = torch.bool)
        )
    
    def reset_buffer(self):
        self.counter = 0
        self.state_memory.fill(0)
        self.action_memory.fill(0)
        self.reward_memory.fill(0)
        self.next_state_memory.fill(0)
        self.terminal_state_memory.fill(0)
    
    def is_full(self):
        return self.counter >= self.buffer_size


class Epsilon_Controller:
    def __init__(self, epsilon, epsilon_decay_rate, minimum_epsilon, reward_target, reward_target_grow_rate, confidence):
        self.epsilon = epsilon
        self.epsilon_decay_rate = epsilon_decay_rate
        self.minimum_epsilon = minimum_epsilon
        self.reward_target = reward_target
        self.reward_target_grow_rate = reward_target_grow_rate
        self.confidence = confidence
        self.confidence_count = 0
        self.deci_place = self._get_deci_place()
    
    def decay(self, last_reward):
        if self.epsilon > self.minimum_epsilon and last_reward >= self.reward_target:
            if self.confidence_count < self.confidence:
                self.confidence_count += 1
            else:
                self.epsilon = round(self.epsilon - self.epsilon_decay_rate, self.deci_place) if self.epsilon > self.minimum_epsilon else self.minimum_epsilon
                self.reward_target += self.reward_target_grow_rate
                self.confidence_count = 0
        else:
            self.confidence_count = 0
    
    def _get_deci_place(self) -> int:
        count = 0
        after_dot = False
        for i in self.epsilon_decay_rate:
            if after_dot:
                count += 1
            if i == ".":
                after_dot = True
        self.epsilon_decay_rate = float(self.epsilon_decay_rate)
        return count


class Network(nn.Module):
    def __init__(self, input_dims, output_dims, atom_size, max_score, learning_rate):
        super(Network, self).__init__()
        self.support = torch.linspace(0.0, max_score, atom_size)
        self.output_dims = output_dims
        self.max_score = max_score
        self.atom_size = atom_size
        self.loss = nn.MSELoss()
        self.layers = nn.Sequential(
            nn.Linear(*input_dims, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, output_dims * atom_size)
        )
        self.optimizer = optim.Adam(self.parameters(), learning_rate)
    
    def dist(self, x):
        q_atoms = self.layers.forward(x).view(-1, self.output_dims, self.atom_size)
        return F.softmax(q_atoms, dim = -1).clamp(min = 1e-3)
    
    def forward(self, x):
        return torch.sum(self.dist(x) * self.support, dim = 2)


class Agent:
    def __init__(self, input_dims, output_dims, atom_size, max_score, min_score, learning_rate, gamma, tau, buffer_size, batch_size, epsilon, epsilon_decay_rate, minimum_epsilon, reward_target, reward_target_grow_rate, confidence):
        self.Q_Network = Network(input_dims, output_dims, atom_size, max_score, learning_rate)
        self.Q_Target_Network = Network(input_dims, output_dims, atom_size, max_score, learning_rate)
        self.gamma = gamma
        self.tau = tau
        self.output_dims = output_dims
        self.max_score = max_score
        self.min_score = min_score
        self.Replay_Buffer = Replay_Buffer(input_dims, buffer_size, batch_size)
        self.Epsilon_Controller = Epsilon_Controller(epsilon, epsilon_decay_rate, minimum_epsilon, reward_target, reward_target_grow_rate, confidence)

        self.Q_Target_Network.load_state_dict(self.Q_Network.state_dict())
    
    def choose_action(self, state):
        if np.random.random() > self.Epsilon_Controller.epsilon:
            return self.Q_Network.forward(torch.tensor(state)).argmax().item()
        else:
            return np.random.choice(self.output_dims)
    
    def SHIN_choose_action(self, state):
        return self.Q_Network.forward(torch.tensor(state)).argmax().item()
    
    def update_network(self):
        for target_network_param, network_param in zip(self.Q_Target_Network.parameters(), self.Q_Network.parameters()):
            target_network_param.data.copy_(self.tau * network_param + (1 - self.tau) * target_network_param)

    def learn(self):
        batch = self.Replay_Buffer.sample_batch()
        states = batch.get("state_batch")
        actions = batch.get("action_batch")
        rewards = batch.get("reward_batch")
        next_states = batch.get("next_state_batch")
        terminal_states = batch.get("terminal_state_batch")

        d_z = float(self.max_score - self.min_score) / (self.Q_Network.atom_size - 1)

        with torch.no_grad():
            next_action = self.Q_Target_Network.forward(next_states).argmax(1)
            next_dist = self.Q_Target_Network.dist(next_states)[np.arange(agent.Replay_Buffer.batch_size), next_action]

            T_z = (rewards + self.gamma * self.Q_Network.support * ~terminal_states).clamp(min = self.min_score, max = self.max_score)
            B = (T_z - self.min_score) / d_z
            lower_bound = B.floor().long()
            upper_bound = B.ceil().long()

            offset = (
                torch.linspace(
                    0, (self.Replay_Buffer.batch_size - 1) * self.Q_Network.atom_size, self.Replay_Buffer.batch_size
                    ).long()
                    .unsqueeze(1)
                    .expand(self.Replay_Buffer.batch_size, self.Q_Network.atom_size)
            )

            proj_dist = torch.zeros(next_dist.size())
            proj_dist.view(-1).index_add_(
                0, (lower_bound + offset).view(-1), (next_dist * (upper_bound.float() - B)).view(-1)
            )
            proj_dist.view(-1).index_add_(
                0, (upper_bound + offset).view(-1), (next_dist * (B - lower_bound.float())).view(-1)
            )
        dist = self.Q_Network.dist(states)
        log_p = torch.log(dist[np.arange(self.Replay_Buffer.batch_size), actions])
        loss = -(proj_dist * log_p).sum(1).mean()
        
        self.Q_Network.optimizer.zero_grad()
        loss.backward()
        self.Q_Network.optimizer.step()

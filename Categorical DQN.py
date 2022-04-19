import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pickle
import time
import gym


class Network(nn.Module):
    def __init__(self, input_dims, output_dims, atom_size, learning_rate, support) -> None:
        super(Network, self).__init__()
        self.support = support
        self.output_dims = output_dims
        self.atom_size = atom_size

        self.layers = nn.Sequential(
            nn.Linear(input_dims, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dims * atom_size)
        )
        self.optimizer = optim.Adam(self.parameters(), learning_rate)
    
    def forward(self, x):
        dist = self.dist(x)
        q = torch.sum(dist * self.support, dim = 2)
        return q
    
    def dist(self, x):
        q_atoms = self.layers.forward(x).view(-1, self.output_dims, self.atom_size)
        dist = F.softmax(q_atoms, dim = -1)
        dist = dist.clamp(min = 1e-3)
        return dist


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


class Agent:
    def __init__(self, input_dims, output_dims, atom_size, v_min, v_max, learning_rate, gamma, tau, buffer_size, batch_size, epsilon, epsilon_decay_rate, minimum_epsilon, reward_target, reward_target_grow_rate, confidence):
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau
        self.v_min = v_min
        self.v_max = v_max
        self.network = Network(input_dims, output_dims, learning_rate, atom_size)
        self.target_network = Network(input_dims, output_dims, learning_rate, atom_size)
        self.buffer = Replay_Buffer(input_dims, buffer_size, batch_size)
        self.epsilon_controller = Epsilon_Controller(epsilon, epsilon_decay_rate, minimum_epsilon, reward_target, reward_target_grow_rate, confidence)
        self.support = torch.linspace(v_min, v_max, atom_size)
        self.target_network.load_state_dict(self.network.state_dict())

    def choose_action(self, state):
        if np.random.random() > self.epsilon_controller.epsilon:
            return torch.argmax(self.network.forward(torch.tensor(state))).item()
        else:
            return np.random.choice(self.output_dims)
    
    def SHIN_choose_action(self, state):
        return torch.argmax(self.network.forward(torch.tensor(state))).item()
    
    def update_network(self):
        for target_network_param, network_param in zip(self.target_network.parameters(), self.network.parameters()):
            target_network_param.data.copy_(self.tau * network_param + (1 - self.tau) * target_network_param)

    def learn(self, env):
        batch = self.buffer.sample_batch()
        state_batch = batch.get("state_batch")
        action_batch = batch.get("action_batch")
        reward_batch = batch.get("reward_batch")
        next_state_batch = batch.get("next_state_batch")
        terminal_state_batch = batch.get("terminal_state_batch")

        self.network.zero_grad()
        delta_z = (self.v_max - self.v_min) / (self.network.atom_size - 1)
        with torch.no_grad():
            next_action = self.target_network.forward(next_state_batch).argmax(1)
            next_dist = self.target_network.dist(next_state_batch)
            next_dist = next_dist[range(self.buffer.batch_size), next_action]

            t_z = reward_batch + ~terminal_state_batch * self.gamma * self.support
            t_z.clamp(min = self.v_min, max = self.v_max)
            b = (t_z - self.v_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()

            offset = (
                torch.linspace(
                    0, (self.buffer.batch_size - 1) * self.network.atom_size, self.buffer.batch_size
                ).long()
                .unsqueeze(1)
                .expand(self.buffer.batch_size, self.network.atom_size)
            )

            proj_dist = torch.zeros(next_dist.size())
            proj_dist.view(-1).index_add_(
                0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
            )
            proj_dist.view(-1).index_add_(
                0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
            )
        dist = self.network.dist(state_batch)
        log_p = torch.log(dist[range(self.buffer.batch_size), action_batch])
        loss = -(proj_dist * log_p).sum(dim = 1).mean()
        loss.backward()
        self.network.optimizer.step()
    
        score = 0
        is_done = False
        state = env.reset()
        while not is_done:
            action = self.SHIN_choose_action(state)
            state, reward, is_done, _ = env.step(action)
            score += reward
        self.epsilon_controller.decay(score)
        return score

    def test(self, env, n_games):
        for i in range(n_games):
            score = 0
            is_done = False
            state = env.reset()
            while not is_done:
                time.sleep(0.01)
                env.render()
                action = self.SHIN_choose_action(state)
                state, reward, is_done, _ = env.step(action)
                score += reward
            print(f"Game: {i + 1}, Score: {score}")
        env.close()

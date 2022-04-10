import gym
import time
import math
import torch
import pickle
import numpy as np
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


class NoisyLinear(nn.Module):
    def __init__(self, input_dim, output_dim, std_init = 0.5):
        super(NoisyLinear, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.std_init = std_init

        self.weight_mean = nn.Parameter(torch.Tensor(output_dim, input_dim))
        self.weight_std = nn.Parameter(torch.Tensor(output_dim, input_dim))
        self.register_buffer("weight_epsilon", torch.Tensor(output_dim, input_dim))

        self.bias_mean = nn.Parameter(torch.Tensor(output_dim))
        self.bias_std = nn.Parameter(torch.Tensor(output_dim))
        self.register_buffer("bias_epsilon", torch.Tensor(output_dim))

        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        mean_range = 1 / math.sqrt(self.input_dim)
        self.weight_mean.data.uniform_(-mean_range, mean_range)
        self.weight_std.data.fill_(self.std_init / math.sqrt(self.input_dim))
        self.bias_mean.data.uniform_(-mean_range, mean_range)
        self.bias_std.data.fill_(self.std_init / math.sqrt(self.input_dim))
    
    def reset_noise(self):
        epsilon_in = self.scale_noise(self.input_dim)
        epsilon_out = self.scale_noise(self.output_dim)

        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def forward(self, x):
        return F.linear(
            x, 
            self.weight_mean + self.weight_std * self.weight_epsilon, 
            self.bias_mean + self.bias_std * self.bias_epsilon
        )
    
    @staticmethod
    def scale_noise(size):
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())


class NoisyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, learning_rate):
        super(NoisyNetwork, self).__init__()
        self.feature = nn.Linear(*input_dim, 64)
        self.noisy_layer1 = NoisyLinear(64, 64)
        self.noisy_layer2 = NoisyLinear(64, output_dim)
        self.optimizer = optim.Adam(self.parameters(), lr = learning_rate)
        self.loss = nn.SmoothL1Loss()
    
    def forward(self, x):
        return self.noisy_layer2(F.mish(self.noisy_layer1(F.mish(self.feature(x.float())))))
    
    def reset_noise(self):
        self.noisy_layer1.reset_noise()
        self.noisy_layer2.reset_noise()


class Agent:
    def __init__(self, input_dims, output_dims, learning_rate, gamma, buffer_size, batch_size, epsilon, epsilon_decay_rate, minimum_epsilon, reward_target, reward_target_grow_rate, confidence):
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.network = NoisyNetwork(input_dims, output_dims, learning_rate)
        self.target_network = NoisyNetwork(input_dims, output_dims, learning_rate)
        self.buffer = Replay_Buffer(input_dims, buffer_size, batch_size)
        self.epsilon_controller = Epsilon_Controller(epsilon, epsilon_decay_rate, minimum_epsilon, reward_target, reward_target_grow_rate, confidence)
        self.update_network()

    def choose_action(self, state):
        if np.random.random() > self.epsilon_controller.epsilon:
            return torch.argmax(self.network.forward(torch.tensor(state))).item()
        else:
            return np.random.choice(self.output_dims)
    
    def SHIN_choose_action(self, state):
        return torch.argmax(self.network.forward(torch.tensor(state))).item()
    
    def update_network(self):
        self.target_network.load_state_dict(self.network.state_dict())

    def learn(self, env):
        batch = self.buffer.sample_batch()
        batch_index = np.arange(self.buffer.batch_size, dtype = np.longlong)
        state_batch = batch.get("state_batch")
        action_batch = batch.get("action_batch")
        reward_batch = batch.get("reward_batch")
        next_state_batch = batch.get("next_state_batch")
        terminal_state_batch = batch.get("terminal_state_batch")

        self.network.zero_grad()
        Q_Pred = self.network.forward(state_batch)[batch_index, action_batch]
        Q_Next = self.target_network.forward(next_state_batch)
        Argmax_action = torch.argmax(self.network.forward(next_state_batch), dim = 1)
        Q_Target = reward_batch + self.gamma * Q_Next[batch_index, Argmax_action] * ~terminal_state_batch
        loss = self.network.loss(Q_Pred, Q_Target)
        loss.backward()
        self.network.optimizer.step()
        self.network.reset_noise()
        self.target_network.reset_noise()

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


if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    agent = Agent(env.observation_space.shape, env.action_space.n, learning_rate = 0.001, gamma = 0.99, buffer_size = 2048, batch_size = 1024, epsilon = 1.0, epsilon_decay_rate = "0.05", minimum_epsilon = 0.0, reward_target = 25, reward_target_grow_rate = 25, confidence = 2)
    iteration = 100
    epoch_to_learn_from_buffer = 128
    score = 0
    stop_limit = 2
    max_count = 0
    for i in range(iteration):
        while not agent.buffer.is_full():
            is_done = False
            state = env.reset()
            while not is_done:
                action = agent.SHIN_choose_action(state)
                next_state, reward, is_done, _ = env.step(action)
                agent.buffer.store(state, action, reward, next_state, is_done)
                state = next_state
        for _ in range(epoch_to_learn_from_buffer):
            score = agent.learn(env)
            if score == 500:
                max_count += 1
                if max_count == stop_limit:
                    break
            else:
                max_count = 0
        if max_count == stop_limit:
          break
        agent.update_network()
        agent.buffer.reset_buffer()
        print(f"Iteration: {i + 1}, Epsilon: {agent.epsilon_controller.epsilon}, Current Target: {agent.epsilon_controller.reward_target}, Last Game Score: {score}")
    with open("Noisy_Net_DDQN_Agent.pickle", "wb") as f:
        pickle.dump(agent, f)
    with open("Noisy_Net_DDQN_Agent.pickle", "rb") as f:
        agent = pickle.load(f)
    agent.test(env, 3)
from typing import Dict, Tuple
from collections import deque
import numpy as np
import pickle
import gym
import torch
import torch.nn as nn
import torch.optim as optim


class ReplayBuffer:
    def __init__(self, input_dim: tuple, buffer_size: int, batch_size: int, n_step: int, gamma: float) -> None:
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.state_memory = np.zeros((buffer_size, *input_dim), dtype = np.float32)
        self.action_memory = np.zeros(buffer_size, dtype = np.int8)
        self.reward_memory = np.zeros(buffer_size, dtype = np.float32)
        self.next_state_memory = np.zeros((buffer_size, *input_dim), dtype = np.float32)
        self.terminal_state_memory = np.zeros(buffer_size, dtype = np.bool8)
        self.buffer_size, self.batch_size = buffer_size, batch_size
        self.counter, self.cur_size = 0, 0
        self.n_step_buffer = deque(maxlen = n_step)
        self.n_step = n_step
        self.gamma = gamma

    def store(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]:
        transition = (state, action, next_state, done)
        self.n_step_buffer.append(transition)
        if len(self.n_step_buffer) < self.n_step:
            return ()
        
        reward, next_state, done = self._get_n_step_info()
        state, action = self.n_step_buffer[0][:2]

        self.state_memory[self.counter] = state
        self.action_memory[self.counter] = action
        self.reward_memory[self.counter] = reward
        self.next_state_memory[self.counter] = next_state
        self.terminal_state_memory[self.counter] = done
        self.counter = (self.counter + 1) % self.buffer_size
        self.cur_size = min(self.cur_size + 1, self.buffer_size)

        return self.n_step_buffer[0]

    def _get_n_step_info(self) -> Tuple[np.float32, np.ndarray, bool]:
        reward, next_state, done = self.n_step_buffer[-1][-3:]
        for transition in reversed(list(self.n_step_buffer)[:-1]):
            r, n_s, d = transition[-3:]
            reward = r + self.gamma * reward * ~d
            next_state, done = (n_s, d) if d else (next_state, done)
        
        return reward, next_state, done

    def sample_batch(self) -> Dict[str, torch.Tensor]:
        batch_index = np.random.choice(self.buffer_size, self.batch_size, False)
        return dict(
            state_batch = torch.tensor(self.state_memory[batch_index], dtype = torch.float32),
            action_batch = torch.tensor(self.action_memory[batch_index], dtype = torch.int64),
            reward_batch = torch.tensor(self.reward_memory[batch_index], dtype = torch.float32),
            next_state_batch = torch.tensor(self.next_state_memory[batch_index], dtype = torch.float32),
            terminal_state_batch = torch.tensor(self.terminal_state_memory[batch_index], dtype = torch.bool),
            batch_index = batch_index
        )
    
    def sample_batch_from_indexes(self, indexes: np.ndarray) -> Dict[str, torch.Tensor]:
        return dict(
            state_batch = torch.tensor(self.state_memory[indexes], dtype = torch.float32),
            action_batch = torch.tensor(self.action_memory[indexes], dtype = torch.int64),
            reward_batch = torch.tensor(self.reward_memory[indexes], dtype = torch.float32),
            next_state_batch = torch.tensor(self.next_state_memory[indexes], dtype = torch.float32),
            terminal_state_batch = torch.tensor(self.terminal_state_memory[indexes], dtype = torch.bool)
        )
    
    def is_ready(self) -> bool:
        return self.cur_size >= self.batch_size


class EpsilonController:
    def __init__(self, epsilon: float, epsilon_decay_rate: str, minimum_epsilon: float) -> None:
        self.epsilon = epsilon
        self.epsilon_decay_rate = epsilon_decay_rate
        self.minimum_epsilon = minimum_epsilon
        self.deci_place = self._get_deci_place()
    
    def decay(self) -> None:
        self.epsilon = round(self.epsilon - self.epsilon_decay_rate, self.deci_place) if self.epsilon > self.minimum_epsilon else self.minimum_epsilon
    
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
    def __init__(self, input_dim: tuple, output_dim: int, learning_rate: float) -> None:
        super(Network, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(*input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )
        self.optimizer = optim.RMSprop(self.parameters(), learning_rate)
        self.loss = nn.SmoothL1Loss()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x.float())


class Agent:
    def __init__(
        self,
        input_dim: tuple,
        output_dim: int,
        learning_rate: float,
        gamma: float,
        update_target: int,
        buffer_size: int,
        batch_size: int,
        n_step: int,
        epsilon: float,
        epsilon_decay_rate: str,
        minimum_epsilon: float
    ) -> None:
        assert n_step > 1
        self.replaybuffer = ReplayBuffer(input_dim, buffer_size, batch_size, n_step, gamma)
        self.epsiloncontroller = EpsilonController(epsilon, epsilon_decay_rate, minimum_epsilon)
        self.network = Network(input_dim, output_dim, learning_rate)
        self.target_network = Network(input_dim, output_dim, learning_rate)
        self.update_target = update_target
        self.update_count = 0
        self.output_dim = output_dim
        self.n_step = n_step

        self.update_target_network()

    def update_target_network(self) -> None:
        self.target_network.load_state_dict(self.network.state_dict())
    
    def choose_action_train(self, state: np.ndarray) -> int:
        if self.epsiloncontroller.epsilon > np.random.random():
            return np.random.choice(self.output_dim)
        else:
            return self.network.forward(torch.tensor(state)).argmax().item()
    
    def choose_action_test(self, state: np.ndarray) -> int:
        return self.network.forward(torch.tensor(state)).argmax().item()

    def _compute_loss(self, batch: Dict[str, np.ndarray], gamma: float) -> torch.Tensor:
        states = batch.get("state_batch")
        actions = batch.get("action_batch")
        rewards = batch.get("reward_batch")
        next_states = batch.get("next_state_batch")
        terminal_states = batch.get("terminal_state_batch")
        batch_index = range(self.replaybuffer.batch_size)

        q_pred = self.network.forward(states)[batch_index, actions]
        q_next = self.target_network.forward(next_states)[batch_index, self.network.forward(next_states).argmax(1)]
        q_target = rewards + gamma * q_next * ~terminal_states

        return self.network.loss(q_pred, q_target)

    def update_model(self) -> torch.Tensor:
        samples = self.replaybuffer.sample_batch()
        indexes = samples.get("batch_index")
        loss = self._compute_loss(samples, self.replaybuffer.gamma)

        samples = self.replaybuffer.sample_batch_from_indexes(indexes)
        gamma = self.gamma ** self.n_step
        n_loss = self._compute_loss(samples, gamma)
        loss += n_loss

        self.network.optimizer.zero_grad()
        loss.backward()
        self.network.optimizer.step()
        return loss

    def train(self) -> torch.Tensor:
        loss = self.update_model()
        self.update_count += 1
        if self.update_count % self.update_target == 0:
            self.update_target_network()
        self.epsiloncontroller.decay()
        return loss

    def test(self, env: gym.Env, n_game: int) -> None:
        for i in range(n_game):
            state = env.reset()
            done = False
            score = 0
            while not done:
                action = self.choose_action_train(state)
                state, reward, done, _ = env.step(action)
                score += reward
            print(f"Game: {i + 1}, Score: {score}")

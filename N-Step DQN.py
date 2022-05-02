from collections import deque
from typing import Deque, Dict, Tuple
import numpy as np
import pickle
import gym
import torch
import torch.nn as nn
import torch.optim as optim


class ReplayBuffer:
    def __init__(self, input_dim: tuple, buffer_size: int, batch_size: int, n_step: int, gamma: float) -> None:
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

    def _get_n_step_info(self, n_step_buffer: Deque, gamma: float) -> Tuple[np.int8, np.ndarray, bool]:
        reward, next_state, done = n_step_buffer[-1][-3:]
        for transition in reversed(list(n_step_buffer)[:-1]):
            r, n_s, d = transition[-3:]
            reward = r + gamma * reward * ~d
            next_state, done = (n_s, d) if d else (next_state, done)
        return reward, next_state, done

    def store(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        transition = (state, action, reward, next_state, done)
        self.n_step_buffer.append(transition)
        if len(self.n_step_buffer) < self.n_step:
            return ()
        reward, next_state, done = self._get_n_step_info(self.n_step_buffer, self.gamma)
        state, action = self.n_step_buffer[0][:2]

        self.state_memory[self.counter] = state
        self.action_memory[self.counter] = action
        self.reward_memory[self.counter] = reward
        self.next_state_memory[self.counter] = next_state
        self.terminal_state_memory[self.counter] = done
        self.counter = (self.counter + 1) % self.buffer_size
        self.cur_size = min(self.cur_size + 1, self.buffer_size)
        return self.n_step_buffer[0]
    
    def sample_batch(self) -> Dict[str, np.ndarray]:
        indexes = np.random.choice(self.cur_size, self.batch_size, False)
        return dict(
            states = torch.tensor(self.state_memory[indexes], dtype = torch.float32),
            actions = torch.tensor(self.action_memory[indexes], dtype = torch.int64),
            rewards = torch.tensor(self.reward_memory[indexes], dtype = torch.float32),
            next_states = torch.tensor(self.next_state_memory[indexes], dtype = torch.float32),
            terminal_states = torch.tensor(self.terminal_state_memory[indexes], dtype = torch.bool),
            indexes = torch.tensor(indexes)
        )
    
    def sample_batch_indexes(self, indexes: np.ndarray) -> Dict[str, np.ndarray]:
        return dict(
            states = torch.tensor(self.state_memory[indexes], dtype = torch.float32),
            actions = torch.tensor(self.action_memory[indexes], dtype = torch.int64),
            rewards = torch.tensor(self.reward_memory[indexes], dtype = torch.float32),
            next_states = torch.tensor(self.next_state_memory[indexes], dtype = torch.float32),
            terminal_states = torch.tensor(self.terminal_state_memory[indexes], dtype = torch.bool)
        )

    def is_ready(self) -> bool:
        return self.cur_size >= self.batch_size


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
        self.optimizer = optim.RMSprop(self.parameters(), lr = learning_rate)
        self.loss = nn.SmoothL1Loss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x.float())


class EpsilonController:
    def __init__(self, init_eps: float, eps_dec_rate: str, min_eps: float) -> None:
        assert 0 <= init_eps <= 1
        assert 0 <= min_eps <= 1
        self.eps = init_eps
        self._deci_place, self.eps_dec_rate = self._get_deci_place(eps_dec_rate)
        self.min_eps = min_eps

    def _get_deci_place(self, eps_dec_rate: str) -> Tuple[int, float]:
        after_dot = False
        count = 0
        for char in eps_dec_rate:
            if char == ".":
                after_dot = True
            if after_dot:
                count += 1
        return count, float(eps_dec_rate)
    
    def decay(self) -> None:
        self.eps = round(self.eps - self.eps_dec_rate, self._deci_place) if self.eps > self.min_eps else self.min_eps


class Agent:
    def __init__(
        self, 
        input_dim: tuple, 
        output_dim: int, 
        learning_rate: float, 
        target_update: int,
        buffer_size: int, 
        batch_size: int, 
        n_step: int, 
        gamma: float, 
        init_eps: float, 
        eps_dec_rate: str, 
        min_eps: float
    ) -> None:
        assert n_step > 1
        self.epsilon_controller = EpsilonController(init_eps, eps_dec_rate, min_eps)
        self.replay_buffer = ReplayBuffer(input_dim, buffer_size, batch_size, 1, gamma)
        self.n_replay_buffer = ReplayBuffer(input_dim, buffer_size, batch_size, n_step, gamma)
        self.network = Network(input_dim, output_dim, learning_rate)
        self.target_network = Network(input_dim, output_dim, learning_rate)
        self.gamma, self.target_update, self.n_step, self.output_dim = gamma, target_update, n_step, output_dim
        self.update_count = 0
        self.update_target_network()

    def update_target_network(self) -> None:
        self.target_network.load_state_dict(self.network.state_dict())

    def choose_action_train(self, state: np.ndarray) -> int:
        if self.epsilon_controller.eps > np.random.random():
            return np.random.choice(self.output_dim)
        else:
            return self.network.forward(torch.tensor(state)).argmax().item()
    
    def choose_action_test(self, state: np.ndarray) -> int:
        return self.network.forward(torch.tensor(state)).argmax().item()
 
    def _compute_loss(self, batch: Dict[str, np.ndarray], gamma: float) -> torch.Tensor:
        states = batch.get("states")
        actions = batch.get("actions")
        rewards = batch.get("rewards")
        next_states = batch.get("next_states")
        terminal_states = batch.get("terminal_states")
        batch_index = np.arange(self.replay_buffer.batch_size, dtype = np.longlong)

        q_pred = self.network.forward(states)[batch_index, actions]
        q_next = self.target_network.forward(next_states)[batch_index, self.network.forward(next_states).argmax(1)]
        q_target = rewards + gamma * q_next * ~terminal_states
        return self.network.loss(q_pred, q_target)

    def train(self, iteration: int, env: gym.Env) -> torch.Tensor:
        for i in range(iteration):
            state = env.reset()
            done = False
            score = 0
            loss = 0
            while not done:
                action = self.choose_action_train(state)
                next_state, reward, done, _ = env.step(action)
                one_step_transition = self.n_replay_buffer.store(state, action, reward, next_state, done)
                if one_step_transition:
                    self.replay_buffer.store(*one_step_transition)
                state = next_state
                score += reward
                if self.replay_buffer.is_ready():
                    batch = self.replay_buffer.sample_batch()
                    indexes = batch["indexes"]
                    loss = self._compute_loss(batch, self.gamma)

                    batch = self.n_replay_buffer.sample_batch_indexes(indexes)
                    gamma = self.gamma ** self.n_step
                    n_loss = self._compute_loss(batch, gamma)
                    loss += n_loss

                    self.network.optimizer.zero_grad()
                    loss.backward()
                    self.network.optimizer.step()
                    self.update_count += 1
                    if self.update_count % self.target_update == 0:
                        self.update_target_network()
                    self.epsilon_controller.decay()
            if (i + 1) % 50:
                print(f"Iteration: {i + 1}, Epsilon: {self.epsilon_controller.eps}, Loss: {loss}, Last Game Score: {score}")

    def test(self, env: gym.Env) -> None:
        state = env.reset()
        done = False
        score = 0
        while not done:
            env.render()
            state, reward, done, _ = env.step(self.choose_action_train(state))
            score += reward
        print(f"Score: {score}")


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    agent = Agent(
        env.observation_space.shape,
        env.action_space.n, 0.001, 100,
        2000, 32, 3, 0.99,
        1.0, "0.0005", 0.0
    )
    # with open("N_Step_DDQN.pickle", "rb") as f:
    #     agent = pickle.load(f)
    # agent.test(env)
    with open("N_Step_DDQN.pickle", "wb") as f:
        pickle.dump(agent, f)
from typing import Dict
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
import gym


class Categorical_DQN(nn.Module):
    def __init__(self, input_dim: tuple, output_dim: int, learning_rate: float, atom_size: int, support: torch.Tnesor) -> None:
        super(Categorical_DQN, self).__init__()
        self.support = support
        self.output_dim = output_dim
        self.atom_size = atom_size
        self.layers = nn.Sequential(
            nn.Linear(*input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim * atom_size)
        )
        self.optimizer = optim.Adam(self.parameters(), learning_rate)
    
    def dist(self, x: torch.Tensor) -> torch.Tensor:
        q_atoms = self.layers.forward(x.float()).view(-1, self.output_dim, self.atom_size)
        dist = F.softmax(q_atoms, dim = -1).clamp(min = 1e-3)
        return dist

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sum(self.dist(x) * self.support, dim = 2)


class Epsilon_Controller:
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


class Replay_Buffer:
    def __init__(self, input_dim: tuple, buffer_size: int, batch_size: int) -> None:
        self.counter = 0
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.state_memory = np.zeros((buffer_size, *input_dim), dtype = np.float32)
        self.action_memory = np.zeros(buffer_size, dtype = np.int8)
        self.reward_memory = np.zeros(buffer_size, dtype = np.float32)
        self.next_state_memory = np.zeros((buffer_size, *input_dim), dtype = np.float32)
        self.terminal_state_memory = np.zeros(buffer_size, dtype = np.bool8)
    
    def store(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.state_memory[self.counter] = state
        self.action_memory[self.counter] = action
        self.reward_memory[self.counter] = reward
        self.next_state_memory[self.counter] = next_state
        self.terminal_state_memory[self.counter] = done
        self.counter = (self.counter + 1) % self.buffer_size
    
    def sample_batch(self) -> Dict[str, torch.Tensor]:
        batch_index = np.random.choice(self.buffer_size, self.batch_size, False)
        return dict(
            state_batch = torch.tensor(self.state_memory[batch_index], dtype = torch.float32),
            action_batch = torch.tensor(self.action_memory[batch_index], dtype = torch.int64),
            reward_batch = torch.tensor(self.reward_memory[batch_index], dtype = torch.float32),
            next_state_batch = torch.tensor(self.next_state_memory[batch_index], dtype = torch.float32),
            terminal_state_batch = torch.tensor(self.terminal_state_memory[batch_index], dtype = torch.bool)
        )
    
    def reset_buffer(self) -> None:
        self.counter = 0
        self.state_memory.fill(0)
        self.action_memory.fill(0)
        self.reward_memory.fill(0)
        self.next_state_memory.fill(0)
        self.terminal_state_memory.fill(0)
    
    def is_full(self) -> bool:
        return self.counter >= self.buffer_size
    
    def is_ready(self) -> bool:
        return self.counter >= self.batch_size


class Agent:
    def __init__(self, input_dim: tuple, output_dim: int, learning_rate: float, atom_size: int, v_min: float, v_max: float, gamma: float, buffer_size: int, batch_size: int, epsilon: float, epsilon_decay_rate: str, minimum_epsilon: float, update_target: int) -> None:
        self.v_min = v_min
        self.v_max = v_max
        self.support = torch.linspace(
            v_min, v_max, atom_size
        )
        self.network = Categorical_DQN(input_dim, output_dim, learning_rate, atom_size, self.support)
        self.target_network = Categorical_DQN(input_dim, output_dim, learning_rate, atom_size, self.support)
        self.gamma = gamma
        self.output_dim = output_dim
        self.update_target = update_target
        self.update_count = 0
        self.replay_buffer = Replay_Buffer(input_dim, buffer_size, batch_size)
        self.epsilon_controller = Epsilon_Controller(epsilon, epsilon_decay_rate, minimum_epsilon)

        self.update_target_network()
    
    def update_target_network(self) -> None:
        self.target_network.load_state_dict(self.network.state_dict())
    
    def choose_action_train(self, state: np.ndarray) -> int:
        if np.random.random() < self.epsilon_controller.epsilon:
            return np.random.choice(self.output_dim)
        else:
            return self.network.forward(torch.tensor(state)).argmax().item()
    
    def choose_action_test(self, state: np.ndarray) -> int:
        return self.network.forward(torch.tensor(state)).argmax().item()

    def train(self) -> float:
        batch = self.replay_buffer.sample_batch()
        batch_index = range(self.replay_buffer.batch_size)
        states = batch.get("state_batch")
        actions = batch.get("action_batch")
        rewards = batch.get("reward_batch").reshape(-1, 1)
        next_states = batch.get("next_state_batch")
        terminal_states = batch.get("terminal_state_batch").reshape(-1, 1)

        self.network.optimizer.zero_grad()
        delta_z = float(self.v_max - self.v_min) / (self.network.atom_size - 1)
        with torch.no_grad():
            next_action = self.target_network.forward(next_states).argmax(1)
            next_dist = self.target_network.dist(next_states)[batch_index, next_action]

            t_z = (rewards + self.gamma * self.support * ~terminal_states).clamp(min = self.v_min, max = self.v_max)
            b = (t_z - self.v_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()

            offset = (
                torch.linspace(
                    0, (self.replay_buffer.batch_size - 1) * self.network.atom_size, self.replay_buffer.batch_size
                ).long()
                .unsqueeze(1)
                .expand(self.replay_buffer.batch_size, self.network.atom_size)
            )

            proj_dist = torch.zeros(next_dist.size())
            proj_dist.view(-1).index_add_(
                0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
            )
            proj_dist.view(-1).index_add_(
                0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
            )
        dist = self.network.dist(states)
        log_p = torch.log(dist[batch_index, actions])
        loss = -(proj_dist * log_p).sum(1).mean()
        loss.backward()
        self.network.optimizer.step()
        self.update_count += 1
        if self.update_count % self.update_target == 0:
            self.update_target_network()
        self.epsilon_controller.decay()
        return loss
    
    def test(self, n_game: int, env: gym.Env) -> None:
        for i in range(n_game):
            state = env.reset()
            done = False
            score = 0
            while not done:
                env.render()
                action = self.choose_action_test(state)
                state, reward, done, _ = env.step(action)
                score += reward
            print(f"Game: {i + 1}, Score: {score}") 


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    agent = Agent(env.observation_space.shape, env.action_space.n, 0.001, 51, 0.0, 500.0, 0.99, 10000, 512, 1.0, "0.0001", 0.001, 5000)
    iteration = 100000
    max_score = 500
    stop_limit = 5
    max_score_count = 0

    for i in range(iteration):
        state = env.reset()
        done = False
        loss = 0
        score = 0
        while not done:
            action = agent.choose_action_train(state)
            next_state, reward, done, _ = env.step(action)
            score += reward
            agent.replay_buffer.store(state, action, reward, next_state, done)
            state = next_state
            if agent.replay_buffer.is_ready():
                loss = agent.train()
        if score == max_score:
            max_score_count += 1
            if max_score_count == stop_limit:
                break
        else:
            max_score_count = 0
        if (i + 1) % 50 == 0:
          print(f"Iteration: {i + 1}, Epsilon: {agent.epsilon_controller.epsilon}, Last Game Score: {score}, Loss: {loss}")
    with open("Categorical_DQN_Agent.pickle", "wb") as f:
        pickle.dump(agent, f)
    # with open("Categorical_DQN_Agent.pickle", "rb") as f:
    #     agent = pickle.load(f)
    # agent.test(3, env)

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pickle
import time
import gym


class DQN(nn.Module):
    def __init__(self, input_dim: tuple, output_dim: int, learning_rate: float
                 ) -> None:
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(*input_dim, 16)
        self.fc2 = nn.Linear(16, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, output_dim)
        self.optimizer = optim.RMSprop(self.parameters(), learning_rate)
        self.loss = nn.MSELoss()

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.fc4(F.relu(self.fc3(F.relu(self.fc2(F.relu(self.fc1(
            state.float())))))))


class Replay_Buffer:
    def __init__(self, input_dim: tuple, buffer_size: int, batch_size: int
                 ) -> None:
        self.counter = 0
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.state_memory = np.zeros((buffer_size, *input_dim),
                                     dtype=np.float32)
        self.action_memory = np.zeros(buffer_size, dtype=np.int8)
        self.reward_memory = np.zeros(buffer_size, dtype=np.float32)
        self.next_state_memory = np.zeros((buffer_size, *input_dim),
                                          dtype=np.float32)
        self.terminal_state_memory = np.zeros(buffer_size, dtype=np.bool8)

    def store(self, state: np.ndarray, action: int, reward: float,
              next_state: np.ndarray, done: bool) -> None:
        self.state_memory[self.counter] = state
        self.action_memory[self.counter] = action
        self.reward_memory[self.counter] = reward
        self.next_state_memory[self.counter] = next_state
        self.terminal_state_memory[self.counter] = done
        self.counter += 1

    def sample_batch(self) -> Dict[str, torch.Tensor]:
        batch_index = np.random.choice(self.buffer_size, self.batch_size,
                                       False)
        return dict(
            state_batch=torch.tensor(self.state_memory[batch_index],
                                     dtype=torch.float32),
            action_batch=torch.tensor(self.action_memory[batch_index],
                                      dtype=torch.int64),
            reward_batch=torch.tensor(self.reward_memory[batch_index],
                                      dtype=torch.float32),
            next_state_batch=torch.tensor(self.next_state_memory[batch_index],
                                          dtype=torch.float32),
            terminal_state_batch=torch.tensor(
                self.terminal_state_memory[batch_index], dtype=torch.bool)
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


class Epsilon_Controller:
    def __init__(
        self,
        epsilon: float,
        epsilon_decay_rate: str,
        minimum_epsilon: float,
        reward_target: float,
        reward_target_grow_rate: float,
        confidence: int
    ) -> None:
        self.epsilon = epsilon
        self.epsilon_decay_rate = epsilon_decay_rate
        self.minimum_epsilon = minimum_epsilon
        self.reward_target = reward_target
        self.reward_target_grow_rate = reward_target_grow_rate
        self.confidence = confidence
        self.confidence_count = 0
        self.deci_place = self._get_deci_place()

    def decay(self, last_reward: float) -> None:
        if (self.epsilon > self.minimum_epsilon and
           last_reward >= self.reward_target):
            if self.confidence_count < self.confidence:
                self.confidence_count += 1
            else:
                self.epsilon = round(self.epsilon - self.epsilon_decay_rate,
                                     self.deci_place) if self.epsilon > \
                                     self.minimum_epsilon else \
                                     self.minimum_epsilon
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
    def __init__(
        self,
        input_dim: tuple,
        output_dim: int,
        learning_rate: float,
        gamma: float,
        tau: float,
        buffer_size: int,
        batch_size: int,
        epsilon: float,
        epsilon_decay_rate: str,
        minimum_epsilon: float,
        reward_target: float,
        reward_target_grow_rate: float,
        confidence: int
    ) -> None:
        self.input_dims = input_dim
        self.output_dims = output_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau
        self.network = DQN(input_dim, output_dim, learning_rate)
        self.target_network = DQN(input_dim, output_dim, learning_rate)
        self.buffer = Replay_Buffer(input_dim, buffer_size, batch_size)
        self.epsilon_controller = Epsilon_Controller(
            epsilon, epsilon_decay_rate, minimum_epsilon, reward_target,
            reward_target_grow_rate, confidence)

        self.target_network.load_state_dict(self.network.state_dict())

    def choose_action(self, state: np.ndarray) -> int:
        if np.random.random() > self.epsilon_controller.epsilon:
            return torch.argmax(
                self.network.forward(torch.tensor(state))).item()
        else:
            return np.random.choice(self.output_dims)

    def SHIN_choose_action(self, state: np.ndarray) -> int:
        return torch.argmax(self.network.forward(torch.tensor(state))).item()

    def update_network(self) -> None:
        for target_network_param, network_param in zip(
            self.target_network.parameters(), self.network.parameters()
        ):
            target_network_param.data.copy_(self.tau * network_param +
                                            (1 - self.tau) *
                                            target_network_param)

    def learn(self, env: gym.Env) -> float:
        batch = self.buffer.sample_batch()
        batch_index = np.arange(self.buffer.batch_size, dtype=np.longlong)
        state_batch = batch.get("state_batch")
        action_batch = batch.get("action_batch")
        reward_batch = batch.get("reward_batch")
        next_state_batch = batch.get("next_state_batch")
        terminal_state_batch = batch.get("terminal_state_batch")

        self.network.zero_grad()
        Q_Pred = self.network.forward(state_batch)[batch_index, action_batch]
        Q_Next = self.target_network.forward(next_state_batch).max(dim=1)[0]
        Q_Target = reward_batch + self.gamma * Q_Next * ~terminal_state_batch
        loss = self.network.loss(Q_Pred, Q_Target.detach())
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

    def test(self, env: gym.Env, n_games: int) -> None:
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
    agent = Agent(
        env.observation_space.shape, env.action_space.n,
        learning_rate=0.001, gamma=0.99, tau=0.05,
        buffer_size=2048, batch_size=1024, epsilon=1.0,
        epsilon_decay_rate="0.05", minimum_epsilon=0.0,
        reward_target=25, reward_target_grow_rate=25,
        confidence=3
    )
    iteration = 100
    epoch_to_learn_from_buffer = 128
    score = 0
    stop_limit = 5
    max_count = 0
    for i in range(iteration):
        while not agent.buffer.is_full():
            is_done = False
            state = env.reset()
            while not is_done:
                action = agent.choose_action(state)
                next_state, reward, is_done, _ = env.step(action)
                agent.buffer.store(state, action, reward, next_state, is_done)
                state = next_state
        for _ in range(epoch_to_learn_from_buffer):
            score = agent.learn(env)
            agent.update_network()
            if score == 500:
                max_count += 1
                if max_count == stop_limit:
                    break
            else:
                max_count = 0
        if max_count == stop_limit:
            break
        agent.buffer.reset_buffer()
        print(f"Iteration: {i + 1}, Epsilon: "
              f"{agent.epsilon_controller.epsilon},"
              f" Current Target: "
              f"{agent.epsilon_controller.reward_target},"
              f" Last Game Score: {score}")
    with open("DQN_Agent.pickle", "wb") as f:
        pickle.dump(agent, f)
    # with open("DQN_Agent.pickle", "rb") as f:
    #     agent = pickle.load(f)
    # agent.test(env, 3)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pickle
import time
import gym


class Network(nn.Module):
    def __init__(self, input_dims, output_dims, learning_rate):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(*input_dims, 16)
        self.fc2 = nn.Linear(16, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, output_dims)
        self.optimizer = optim.Adam(self.parameters(), learning_rate)
        self.loss = nn.SmoothL1Loss()
    
    def forward(self, state):
        return self.fc4(F.relu(self.fc3(F.relu(self.fc2(F.relu(self.fc1(state)))))))


class Replay_Buffer:
    def __init__(self, input_dims, buffer_size, batch_size):
        self.counter = 0
        self.cur_size = 0
        self.input_dims = input_dims
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.state_memory = np.zeros((buffer_size, *input_dims), dtype = np.float32)
        self.action_memory = np.zeros(buffer_size, dtype = np.int8)
        self.reward_memory = np.zeros(buffer_size, dtype = np.float32)
        self.next_state_memory = np.zeros((buffer_size, *input_dims), dtype = np.float32)
        self.terminal_state_memory = np.zeros(buffer_size, dtype = np.bool8)
    
    def store(self, state, action, reward, next_state, is_done):
        self.state_memory[self.counter] = state
        self.action_memory[self.counter] = action
        self.reward_memory[self.counter] = reward
        self.next_state_memory[self.counter] = next_state
        self.terminal_state_memory[self.counter] = is_done
        self.counter = (self.counter + 1) % self.buffer_size
        self.cur_size = min(self.cur_size + 1, self.buffer_size)
    
    def sample_batch(self):
        batch_index = np.random.choice(self.buffer_size, self.batch_size, False)
        batch_dict = dict()
        batch_dict["state_batch"] = self.state_memory[batch_index]
        batch_dict["action_batch"] = self.action_memory[batch_index]
        batch_dict["reward_batch"] = self.reward_memory[batch_index]
        batch_dict["next_state_batch"] = self.next_state_memory[batch_index]
        batch_dict["terminal_state_batch"] = self.terminal_state_memory[batch_index]
        return batch_dict
    
    def reset_buffer(self):
        self.counter = 0
        self.cur_size = 0
        self.state_memory = np.zeros((self.buffer_size, *self.input_dims), dtype = np.float32)
        self.action_memory = np.zeros(self.buffer_size, dtype = np.int8)
        self.reward_memory = np.zeros(self.buffer_size, dtype = np.float32)
        self.next_state_memory = np.zeros((self.buffer_size, *self.input_dims), dtype = np.float32)
        self.terminal_state_memory = np.zeros(self.buffer_size, dtype = np.bool8)
    
    def is_full(self):
        return self.cur_size == self.buffer_size


class Epsilon_Controller:
    def __init__(self, epsilon, epsilon_decay_rate, minimum_epsilon, reward_target, reward_target_grow_rate, confidence):
        self.confidence = confidence
        self.confidence_stack = 0
        self.epsilon = epsilon
        self.epsilon_decay_rate = epsilon_decay_rate
        self.minimum_epsilon = minimum_epsilon
        self.reward_target = reward_target
        self.reward_target_grow_rate = reward_target_grow_rate
        self.deci_place = self._get_deci_place()
    
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
    
    def decay(self, last_game_score: float) -> None:
        if last_game_score >= self.reward_target:
            self.confidence_stack += 1
            if self.confidence_stack == self.confidence:
                self.epsilon = round(self.epsilon - self.epsilon_decay_rate, self.deci_place) if self.epsilon > self.minimum_epsilon else self.minimum_epsilon
                self.reward_target += self.reward_target_grow_rate
                self.confidence_stack = 0
        else:
            self.confidence_stack = 0


class Agent:
    def __init__(self, input_dims, output_dims, learning_rate, gamma, buffer_size, batch_size, epsilon, epsilon_decay_rate, minimum_epsilon, reward_target, reward_target_grow_rate, confidence):
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.network = Network(input_dims, output_dims, learning_rate)
        self.target_network = Network(input_dims, output_dims, learning_rate)
        self.buffer = Replay_Buffer(input_dims, buffer_size, batch_size)
        self.epsilon_controller = Epsilon_Controller(epsilon, epsilon_decay_rate, minimum_epsilon, reward_target, reward_target_grow_rate, confidence)
        self.update_network()
    
    def choose_action(self, state):
        if np.random.random() > self.epsilon_controller.epsilon:
            return self.network.forward(torch.tensor(np.array(state))).argmax().item()
        else:
            return np.random.randint(self.output_dims)
    
    def SHIN_choose_action(self, state):
        return self.network.forward(torch.tensor(np.array(state))).argmax().item()
    
    def update_network(self):
        self.target_network.load_state_dict(self.network.state_dict())

    def learn(self, env):
        batch = self.buffer.sample_batch()
        state_batch = torch.tensor(batch.get("state_batch"), dtype = torch.float32)
        action_batch = torch.tensor(batch.get("action_batch"), dtype = torch.int64)
        reward_batch = torch.tensor(batch.get("reward_batch"), dtype = torch.float32)
        next_state_batch = torch.tensor(batch.get("next_state_batch"), dtype = torch.float32)
        terminal_state_batch = torch.tensor(batch.get("terminal_state_batch"), dtype = torch.bool)
        self.network.zero_grad()
        Q_Pred = self.network.forward(state_batch)[np.arange(self.buffer.batch_size), action_batch]
        Q_Next_State = self.target_network.forward(next_state_batch).max(dim = 1)[0]
        Q_Target = reward_batch + self.gamma * Q_Next_State * ~terminal_state_batch
        loss = self.network.loss(Q_Pred, Q_Target)
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


if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    agent = Agent(env.observation_space.shape, env.action_space.n, learning_rate = 0.001, gamma = 0.99, buffer_size = 4096, batch_size = 1024, epsilon = 1.0, epsilon_decay_rate = "0.05", minimum_epsilon = 0.0, reward_target = 25, reward_target_grow_rate = 25, confidence = 3)
    iteration = 100
    epoch_to_learn_from_buffer = 128
    max_score_count = 0
    stop_limit = 5
    for i in range(iteration):
        while not agent.buffer.is_full():
            is_done = False
            state = env.reset()
            while not is_done:
                action = agent.choose_action(state)
                next_state, reward, is_done, _ = env.step(action)
                agent.buffer.store(state, action, reward, next_state, is_done)
                state = next_state
                if agent.buffer.is_full():
                    break
        for _ in range(epoch_to_learn_from_buffer):
            score = agent.learn(env)
        if score == 500:
            max_score_count += 1
            if max_score_count == stop_limit:
                break
        print(f"Iteration: {i + 1}, Epsilon: {agent.epsilon_controller.epsilon}, Last Game Score: {score} Current Target: {agent.epsilon_controller.reward_target}")
        agent.buffer.reset_buffer()
        agent.update_network()
    with open("DQN_Agent.pickle", "wb") as f:
        pickle.dump(agent, f)
    agent.test(env, 3)

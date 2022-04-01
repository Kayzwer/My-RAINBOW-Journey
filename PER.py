from typing import Callable, Dict, List
import numpy as np
import operator
import pickle
import time
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class SegmentTree:
    def __init__(self, max_size: int, operation: Callable, init_value: float) -> None:
        self.max_size = max_size
        self.tree = [init_value for _ in range(2 * max_size)]
        self.operation = operation
    
    def _operate_helper(self, start: int, end: int, node: int, node_start: int, node_end: int) -> float:
        if start == node_start and end == node_end:
            return self.tree[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._operate_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._operate_helper(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                return self.operation(
                    self._operate_helper(start, mid, 2 * node, node_start, mid),
                    self._operate_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end)
                )
    
    def operate(self, start: int = 0, end: int = 0) -> float:
        if end <= 0:
            end += self.max_size
        end -= 1
        return self._operate_helper(start, end, 1, 0, self.max_size - 1)
    
    def __setitem__(self, index: int, value: float) -> None:
        index += self.max_size
        self.tree[index] = value

        index //= 2
        while index >= 1:
            self.tree[index] = self.operation(self.tree[2 * index], self.tree[2 * index + 1])
            index //= 2
    
    def __getitem__(self, index: int) -> float:
        return self.tree[self.max_size + index]
    
    def __str__(self) -> str:
        return self.tree.__str__()


class SumSegmentTree(SegmentTree):
    def __init__(self, max_size: int) -> None:
        super().__init__(max_size, operator.add, 0.0)
    
    def sum(self, start: int = 0, end: int = 0) -> float:
        return super(SumSegmentTree, self).operate(start, end)
    
    def retrieve(self, upperbound: float) -> int:
        index = 1
        while index < self.max_size:
            left = 2 * index
            right = left + 1
            if self.tree[left] > upperbound:
                index = 2 * index
            else:
                upperbound -= self.tree[left]
                index = right
        return index - self.max_size
    

class MinSegmentTree(SegmentTree):
    def __init__(self, max_size: int) -> None:
        super().__init__(max_size, min, float("inf"))
    
    def min(self, start:int = 0, end: int = 0) -> float:
        return super(MinSegmentTree, self).operate(start, end)


class ReplayBuffer:
    def __init__(self, input_dim: tuple, buffer_size: int, batch_size: int) -> None:
        self.state_memory = np.zeros((buffer_size, *input_dim), dtype = np.float32)
        self.action_memory = np.zeros(buffer_size, dtype = np.int32)
        self.reward_memory = np.zeros(buffer_size, dtype = np.float32)
        self.next_state_memory = np.zeros((buffer_size, *input_dim), dtype = np.float32)
        self.terminal_state_memory = np.zeros(buffer_size, dtype = np.bool8)
        self.buffer_size, self.batch_size, self.input_dim = buffer_size, batch_size, input_dim
        self.pointer, self.cur_size = 0, 0
    
    def store(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.state_memory[self.pointer] = state
        self.action_memory[self.pointer] = action
        self.reward_memory[self.pointer] = reward
        self.next_state_memory[self.pointer] = next_state
        self.terminal_state_memory[self.pointer] = done
        self.pointer = (self.pointer + 1) % self.buffer_size
        self.cur_size = min(self.cur_size + 1, self.buffer_size)
    
    def sample_batch(self) -> Dict[str, np.ndarray]:
        indexes = np.random.choice(self.cur_size, self.batch_size, False)
        return dict(
            states = self.state_memory[indexes],
            actions = self.action_memory[indexes],
            rewards = self.reward_memory[indexes],
            next_states = self.next_state_memory[indexes],
            terminal_states = self.terminal_state_memory[indexes]
        )

    def __len__(self) -> int:
        return self.cur_size
    
    def is_full(self) -> bool:
        return self.cur_size == self.buffer_size
    
    def is_ready(self) -> bool:
        return self.cur_size >= self.batch_size


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, input_dim: tuple, buffer_size: int, batch_size: int, prior_eps: float, alpha: float = 0.6, beta: float = 0.4) -> None:
        super(PrioritizedReplayBuffer, self).__init__(input_dim, buffer_size, batch_size)
        self.max_priority, self.tree_ptr = 1.0, 0
        self.alpha, self.beta, self.prior_eps = alpha, beta, prior_eps

        tree_size = 1
        while tree_size < self.buffer_size:
            tree_size *= 2
        self.tree_size = tree_size

        self.sum_tree = SumSegmentTree(tree_size)
        self.min_tree = MinSegmentTree(tree_size)
    
    def store(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        super().store(state, action, reward, next_state, done)
        self.sum_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.min_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.tree_ptr = (self.tree_ptr + 1) % self.buffer_size
    
    def _sample_proportional(self) -> List[int]:
        indexes = []
        p_total = self.sum_tree.sum(0, len(self) - 1)
        segment = p_total / self.batch_size

        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = np.random.uniform(a, b)
            index = self.sum_tree.retrieve(upperbound)
            indexes.append(index)
        
        return indexes
    
    def _calculate_weight(self, index: int) -> float:
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * len(self)) ** (-self.beta)
        p_sample = self.sum_tree[index] / self.sum_tree.sum()
        return ((p_sample * len(self)) ** -self.beta) / max_weight
    
    def sample_batch(self) -> Dict[str, np.ndarray]:
        assert len(self) >= self.batch_size
        assert self.beta > 0

        indexes = self._sample_proportional()
        return dict(
            states = self.state_memory[indexes],
            actions = self.action_memory[indexes],
            rewards = self.reward_memory[indexes],
            next_states = self.next_state_memory[indexes],
            terminal_states = self.terminal_state_memory[indexes],
            weights = np.array([self._calculate_weight(i) for i in indexes]),
            indexes = indexes
        )

    def update_priorities(self, indexes: List[int], priorities: np.ndarray) -> None:
        assert len(indexes) == len(priorities)

        for index, priority in zip(indexes, priorities):
            assert priority > 0
            assert 0 <= index < len(self)

            self.sum_tree[index] = priority ** self.alpha
            self.min_tree[index] = priority ** self.alpha

            self.max_priority = max(self.max_priority, priority)
    
    def inc_beta(self, iteration: int, cur_iteration: int) -> None:
        self.beta += (min(cur_iteration / iteration, 1.0) * (1.0 - self.beta))

    def reset_buffer(self) -> None:
        self.state_memory = np.zeros((self.buffer_size, *self.input_dim), dtype = np.float32)
        self.action_memory = np.zeros(self.buffer_size, dtype = np.int32)
        self.reward_memory = np.zeros(self.buffer_size, dtype = np.float32)
        self.next_state_memory = np.zeros((self.buffer_size, *self.input_dim), dtype = np.float32)
        self.terminal_state_memory = np.zeros(self.buffer_size, dtype = np.bool8)
        self.pointer, self.cur_size = 0, 0
        self.sum_tree = SumSegmentTree(self.tree_size)
        self.min_tree = MinSegmentTree(self.tree_size)


class Network(nn.Module):
    def __init__(self, input_dim: tuple, output_dim: int, learning_rate: float) -> None:
        super(Network, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(*input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        self.optimizer = optim.Adam(self.parameters(), learning_rate)
        self.loss = nn.SmoothL1Loss()

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.layers.forward(state)


class EpsilonController:
    def __init__(self, epsilon: float, epsilon_decay_rate: str, minimum_epsilon: float, reward_target: int, reward_target_grow_rate: int, confidence: int) -> None:
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
        # if last_game_score >= self.reward_target:
        #     self.confidence_stack += 1
        #     if self.confidence_stack == self.confidence:
        #         self.epsilon = round(self.epsilon - self.epsilon_decay_rate, self.deci_place) if self.epsilon > self.minimum_epsilon else self.minimum_epsilon
        #         self.reward_target += self.reward_target_grow_rate
        #         self.confidence_stack = 0
        # else:
        #     self.confidence_stack = 0
        self.epsilon = round(self.epsilon - self.epsilon_decay_rate, self.deci_place) if self.epsilon > self.minimum_epsilon else self.minimum_epsilon


class Agent:
    def __init__(
        self, 
        input_dim: tuple, 
        output_dim: int, 
        learning_rate: float, 
        gamma: float, 
        target_update: int, 
        buffer_size: int, 
        batch_size: int, 
        alpha: float, 
        beta: float, 
        prior_eps: float, 
        epsilon: float, 
        epsilon_decay_rate: str, 
        minimum_epsilon: float, 
        reward_target: int, 
        reward_target_grow_rate: int, 
        confidence: int) -> None:
        self.gamma = gamma
        self.target_update = target_update
        self.update_counter = 0
        self.n_actions = output_dim
        self.Q_Network = Network(input_dim, output_dim, learning_rate)
        self.Q_Target_Network = Network(input_dim, output_dim, learning_rate)
        self.EpsilonController = EpsilonController(epsilon, epsilon_decay_rate, minimum_epsilon, reward_target, reward_target_grow_rate, confidence)
        self.PrioritizedReplayBuffer = PrioritizedReplayBuffer(input_dim, buffer_size, batch_size, prior_eps, alpha, beta)
    
    def update_Target_Network(self) -> None:
        self.Q_Target_Network.load_state_dict(self.Q_Network.state_dict())

    def choose_action(self, state: np.ndarray) -> int:
        if np.random.random() > self.EpsilonController.epsilon:
            return self.Q_Network.forward(torch.tensor(state)).argmax().item()
        else:
            return np.random.randint(self.n_actions)

    def SHIN_choose_action(self, state: np.ndarray) -> int:
        return self.Q_Network.forward(torch.tensor(np.array(state))).argmax().item()

    def learn(self, env: gym.Env) -> int:
        batch = self.PrioritizedReplayBuffer.sample_batch()
        weights = torch.tensor(batch.get("weights"), dtype = torch.float32)
        indexes = batch.get("indexes")

        self.Q_Network.zero_grad()
        states = torch.tensor(batch.get("states"), dtype = torch.float32)
        actions = torch.tensor(batch.get("actions"), dtype = torch.int64)
        rewards = torch.tensor(batch.get("rewards"), dtype = torch.float32)
        next_states = torch.tensor(batch.get("next_states"), dtype = torch.float32)
        terminal_states = torch.tensor(batch.get("terminal_states"), dtype = torch.bool)

        Q_Pred = self.Q_Network.forward(states)[np.arange(self.PrioritizedReplayBuffer.batch_size), actions]
        Q_Next_Pred = self.Q_Target_Network.forward(next_states).max(dim = 1)[0]
        Q_Target = rewards + self.gamma * Q_Next_Pred * ~terminal_states
        elementwise_loss = F.smooth_l1_loss(Q_Pred, Q_Target, reduction = "none")
        loss = torch.mean(elementwise_loss * weights)
        loss.backward()
        self.Q_Network.optimizer.step()
        self.update_counter += 1

        loss_for_prior = elementwise_loss.detach().numpy()
        new_priorities = loss_for_prior + self.PrioritizedReplayBuffer.prior_eps
        self.PrioritizedReplayBuffer.update_priorities(indexes, new_priorities)

        state = env.reset()
        score = 0
        done = False
        while not done:
            action = self.SHIN_choose_action(state)
            state, reward, done, _ = env.step(action)
            score += reward
        self.EpsilonController.decay(score)

        if self.update_counter % self.target_update == 0:
            self.update_Target_Network()

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
    env_test_score = gym.make('CartPole-v1')
    agent = Agent(env.observation_space.shape, env.action_space.n, 0.0001, 0.99, 100, 2000, 32, 0.2, 0.6, 1e-6, 1.0, "0.0005", 0.1, 5, 5, 1)
    iteration = 20000
    score = 0

    state = env.reset()
    for i in range(iteration):
        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.PrioritizedReplayBuffer.store(state, action, reward, next_state, done)
        agent.PrioritizedReplayBuffer.inc_beta(iteration, i + 1)

        if done:
            state = env.reset()
        
        if agent.PrioritizedReplayBuffer.is_ready():
            score = agent.learn(env_test_score)
        print(f"Iteration: {i + 1}, Epsilon: {agent.EpsilonController.epsilon}, Current Target: {agent.EpsilonController.reward_target}, Last Game Score: {score}")
    with open("PER_DQN_Agent.pickle", "wb") as f:
        pickle.dump(agent, f)
    agent.test(env, 3)

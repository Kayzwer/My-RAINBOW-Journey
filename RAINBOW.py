from collections import deque
from torch.nn.utils import clip_grad_norm_
from typing import Deque, Dict, Tuple, List
from segment_tree import MinSegmentTree, SumSegmentTree
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym


class ReplayBuffer:
    def __init__(
        self,
        state_dim: tuple,
        buffer_size: int,
        batch_size: int,
        n_step: int,
        gamma: float
    ) -> None:
        self.state_memory = np.zeros((buffer_size, *state_dim), dtype = np.float32)
        self.action_memory = np.zeros((buffer_size), dtype = np.int8)
        self.reward_memory = np.zeros((buffer_size), dtype = np.float32)
        self.next_state_memory = np.zeros((buffer_size, *state_dim), dtype = np.float32)
        self.terminal_state_memory = np.zeros((buffer_size), dtype = np.bool8)
        self.buffer_size, self.batch_size = buffer_size, batch_size
        self.ptr, self.cur_size = 0, 0

        self.n_step_buffer = deque(maxlen = n_step)
        self.n_step = n_step
        self.gamma = gamma

    def _get_n_step_info(
        self,
        n_step_buffer: Deque,
        gamma: float
    ) -> Tuple[np.int32, np.ndarray, bool]:
        reward, next_state, done = n_step_buffer[-1][-3:]

        for transition in reversed(list(n_step_buffer)[:-1]):
            r, n_s, d = transition[-3:]
            reward = r + gamma * reward * ~d
            next_state, done = (n_s, d) if d else (next_state, done)
        
        return reward, next_state, done

    def store(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]:
        transition = (state, action, reward, next_state, done)
        self.n_step_buffer.append(transition)

        if len(self.n_step_buffer) < self.n_step:
            return ()
        
        reward, next_state, done = self._get_n_step_info(self.n_step_buffer, self.gamma)
        self.state_memory[self.ptr] = state
        self.action_memory[self.ptr] = action
        self.reward_memory[self.ptr] = reward
        self.next_state_memory[self.ptr] = next_state
        self.terminal_state_memory[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.cur_size = min(self.cur_size + 1, self.buffer_size)

        return self.n_step_buffer[0]

    def sample_batch(self) -> Dict[str, np.ndarray]:
        indexes = np.random.choice(self.cur_size, self.batch_size, False)
        return dict(
            states = self.state_memory[indexes],
            actions = self.action_memory[indexes],
            rewards = self.reward_memory[indexes],
            next_states = self.next_state_memory[indexes],
            terminal_states = self.terminal_state_memory[indexes],
            indexes = indexes
        )

    def sample_batch_from_indexes(self, indexes: np.ndarray) -> Dict[str, np.ndarray]:
        return dict(
            states = self.state_memory[indexes],
            actions = self.action_memory[indexes],
            rewards = self.reward_memory[indexes],
            next_states = self.next_state_memory[indexes],
            terminal_states = self.terminal_state_memory[indexes]
        )
    
    def __len__(self) -> int:
        return self.cur_size


class PrioritiedReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        state_dim: tuple,
        buffer_size: int,
        batch_size: int,
        n_step: int,
        gamma: float,
        alpha: float
    ) -> None:
        super(PrioritiedReplayBuffer, self).__init__(
            state_dim, buffer_size, batch_size, n_step, gamma
        )
        self.max_priority, self.tree_ptr = 1.0, 0
        self.alpha = alpha
        tree_capacity = 1
        while tree_capacity < self.buffer_size:
            tree_capacity *= 2
        
        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)

    def store(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]:
        transition = super().store(state, action, reward, next_state, done)

        if transition:
            temp = self.max_priority ** self.alpha
            self.sum_tree[self.tree_ptr] = temp
            self.min_tree[self.tree_ptr] = temp
            self.tree_ptr = (self.tree_ptr + 1) % self.buffer_size
        
        return transition
    
    def sample_batch(self, beta: float) -> Dict[str, np.ndarray]:
        indexes = self._sample_proportional()
        states = self.state_memory[indexes]
        actions = self.action_memory[indexes]
        rewards = self.reward_memory[indexes]
        next_states = self.next_state_memory[indexes]
        terminal_states = self.terminal_state_memory[indexes]
        weights = np.array([self._calculate_weight(i, beta) for i in indexes])
    
        return dict(
            states = states,
            actions = actions,
            rewards = rewards,
            next_states = next_states,
            terminal_states = terminal_states,
            weights = weights,
            indexes = indexes
        )

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

    def _calculate_weight(self, index: int, beta: float) -> float:
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * len(self)) ** (-beta)

        p_sample = self.sum_tree[index] / self.sum_tree.sum()
        weight = (p_sample * len(self)) ** (-beta)
        weight /= max_weight
        return weight

    def update_priorities(self, indexes: List[int], priorities: np.ndarray) -> None:
        for index, priority in zip(indexes, priorities):
            temp = priority ** self.alpha
            self.sum_tree[index] = temp
            self.min_tree[index] = temp
            self.max_priority = max(self.max_priority, priority)


class NoisyLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        std_init: float
    ) -> None:
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.weight_mean = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_std = nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer(
            "weight_epsilon", torch.Tensor(out_features, in_features)
        )
        self.bias_mean = nn.Parameter(torch.Tensor(out_features))
        self.bias_std = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer(
            "bias_epsilon", torch.Tensor(out_features)
        )

        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        mean_range = 1 / math.sqrt(self.in_features)
        self.weight_mean.data.uniform_(-mean_range, mean_range)
        self.weight_std.data.fill_(
            self.std_init / math.sqrt(self.in_features)
        )
        self.bias_mean.data.uniform_(-mean_range, mean_range)
        self.bias_std.data.fill_(
            self.std_init / math.sqrt(self.out_features)
        )
    
    def reset_noise(self):
        """Make new noise."""
        epsilon_in = self.scale_noise(self.in_features)
        epsilon_out = self.scale_noise(self.out_features)

        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(
            x,
            self.weight_mu + self.weight_sigma * self.weight_epsilon,
            self.bias_mu + self.bias_sigma * self.bias_epsilon,
        )

    @staticmethod
    def scale_noise(size: int) -> torch.Tensor:
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())


class Network(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        atom_size: int,
        support: torch.Tensor
    ) -> None:
        super(Network, self).__init__()
        self.support = support
        self.out_dim = out_dim
        self.atom_size = atom_size

        self.feature_layer = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU()
        )

        self.advantage_hidden_layer = NoisyLinear(128, 128)
        self.advantage_layer = NoisyLinear(128, out_dim * atom_size)

        self.value_hidden_layer = NoisyLinear(128, 128)
        self.value_layer = NoisyLinear(128, atom_size)

    def dist(self, x: torch.Tensor) -> torch.Tensor:
        feature = self.feature_layer(x)
        adv_hid = F.relu(self.advantage_hidden_layer(feature))
        val_hid = F.relu(self.value_hidden_layer(feature))

        advantage = self.advantage_layer(adv_hid).view(-1, self.out_dim, self.atom_size)
        value = self.value_layer(val_hid).view(-1, 1, self.atom_size)
        q_atoms = value + advantage - advantage.mean(dim = 1, keepdim = True)

        dist = F.softmax(q_atoms, dim = -1)
        dist = dist.clamp(min = 1e-3)

        return dist

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dist = self.dist(x)
        q = torch.sum(dist * self.support, dim = 2)
        return q

    def reset_noise(self) -> None:
        self.advantage_hidden_layer.reset_noise()
        self.advantage_layer.reset_noise()
        self.value_hidden_layer.reset_noise()
        self.value_layer.reset_noise()


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
        env: gym.Env,
        buffer_size: int,
        batch_size: int,
        target_update: int,
        gamma: float,
        alpha: float,
        beta: float,
        prior_eps: float,
        v_min: float,
        v_max: float,
        atom_size: int,
        n_step: int,
        init_eps: float,
        eps_dec_rate: str,
        min_eps: float
    ) -> None:
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        self.env = env
        self.batch_size = batch_size
        self.target_update = target_update
        self.gamma = gamma
        self.beta = beta
        self.prior_eps = prior_eps
        self.memory = PrioritiedReplayBuffer(
            state_dim, buffer_size, batch_size, n_step, gamma, alpha
        )
        self.use_n_step = True if n_step > 1 else False
        if self.use_n_step:
            self.n_step = n_step
            self.memory_n = ReplayBuffer(
                state_dim, buffer_size, batch_size, n_step, gamma
            )
        self.v_min = v_min
        self.v_max = v_max
        self.atom_size = atom_size
        self.support = torch.linspace(
            self.v_min, self.v_max, self.atom_size
        )
        self.dqn = Network(
            state_dim, action_dim, self.atom_size, self.support
        )
        self.target_dqn = Network(
            state_dim, action_dim, self.atom_size, self.support
        )
        self.target_dqn.load_state_dict(self.dqn)
        self.target_dqn.eval()
        self.optimizer = optim.Adam(self.dqn.parameters())
        self.transition = list()
        self.is_test = False
    
    def choose_action(self, state: np.ndarray) -> int:
        return self.dqn(
            torch.as_tensor(state, dtype = torch.float32)
        ).argmax().item()

    def step(self, action: int) -> Tuple[np.ndarray, np.float64, bool]:
        next_state, reward, done, _ = self.env.step(action)
        if not self.is_test:
            self.transition += [reward, next_state, done]
            if self.use_n_step:
                one_step_transition = self.memory_n.store(*self.transition)
            else:
                one_step_transition = self.transition
            
            if one_step_transition:
                self.memory.store(*one_step_transition)
        return next_state, reward, done

    def update_model(self) -> torch.Tensor:
        samples = self.memory.sample_batch(self.beta)
        weights = torch.as_tensor(samples.get("weights").reshape(-1, 1))
        indexes = samples.get("indexes")
        elementwise_loss = self._compute_dqn_loss(samples, self.gamma)
        loss = torch.mean(elementwise_loss * weights)

        if self.use_n_step:
            gamma = self.gamma ** self.n_step
            samples = self.memory_n.sample_batch_from_indexes(indexes)
            elementwise_loss_n_loss = self._compute_dqn_loss(samples, gamma)
            elementwise_loss += elementwise_loss_n_loss

            loss = torch.mean(elementwise_loss * weights)
        self.optimizer.zero_grad(set_to_none = True)
        loss.backward()
        clip_grad_norm_(self.dqn.parameters(), 10.0)
        self.optimizer.step()
        
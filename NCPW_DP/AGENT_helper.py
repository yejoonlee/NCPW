import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import pandas as pd
import matplotlib as plt

b_skills = [
    {'idx':0,
     'name':'kick',
     'damage':25,
     'cool':2,
     'hit_rate':0.9
    },
    {'idx':1,
     'name':'punch',
     'damage':10,
     'cool':1,
     'hit_rate':1.0
    }
]

p_skills = [
    {'idx':0,
     'name':'kick',
     'damage':25,
     'cool':2,
     'hit_rate':0.9
    },
    {'idx':1,
     'name':'punch',
     'damage':10,
     'cool':1,
     'hit_rate':1.0
    },
    {'idx':2,
     'name':'heal',
     'damage':0,
     'cool':5,
     'hit_rate':1.0
    },
    {'idx':3,
     'name':'jump',
     'damage':0,
     'cool':1,
     'hit_rate':1.0
    },
]


class boss_raid_simulater():
    def __init__(self, b_skills, p_skills, reward_rate):

        self.reward_rate = reward_rate

        self.full_hp = 100

        self.num_bs = len(b_skills)
        self.num_ps = len(p_skills)

        self.state_dict = dict()
        self.state_dict['b_hp'] = self.full_hp
        for i in range(len(b_skills)):
            self.state_dict[f'b_cool_{i}'] = 0
        self.b_skills = b_skills

        self.state_dict['p_hp'] = self.full_hp
        for i in range(len(p_skills)):
            self.state_dict[f'p_cool_{i}'] = 0
        self.p_skills = p_skills

        self.state = np.array(list(self.state_dict.values()))

        self.boss_action = []
        self.done = 0
        self.len_t = 0

        self.reward = 0

    def observe(self):
        return self.state

    def step(self, p_action):
        self.len_t += 1
        len_t = self.len_t
        p_skill = self.p_skills[int(p_action)]

        b_action = int(np.random.uniform(0.0, 1.0, 1).round())
        self.boss_action.append(b_action)
        b_skill = self.b_skills[b_action]

        #         self.battle(p_skill, b_skill)
        self.battle_with_cool(p_skill, b_skill)

        self.reward = (- self.state_dict['b_hp'] * self.reward_rate[0] + self.state_dict['p_hp'] * self.reward_rate[
            1] - len_t * self.reward_rate[2])

        if self.state_dict['b_hp'] <= 0 or self.state_dict['p_hp'] <= 0:
            self.done = 1

            if self.state_dict['b_hp'] <= 0:
                self.reward += self.full_hp * max([self.reward_rate[0], self.reward_rate[1]]) * self.reward_rate[3]

            if self.state_dict['p_hp'] <= 0:
                self.reward -= self.full_hp * max([self.reward_rate[0], self.reward_rate[1]]) * self.reward_rate[3]

        ns = self.observe()
        r = self.reward
        done = self.done

        return ns, r, done

    def battle_with_cool(self, p_skill, b_skill):
        p_d = p_skill['damage']
        b_d = b_skill['damage']

        p_c = p_skill['cool']
        b_c = b_skill['cool']

        p_i = p_skill['idx']
        b_i = b_skill['idx']

        p_n = p_skill['name']

        p_sc = self.state_dict[f'p_cool_{p_i}']
        b_sc = self.state_dict[f'b_cool_{b_i}']

        # print(f"b:{b_skill['name']} / p:{p_skill['name']}")

        if b_sc == 0:
            if p_n == 'jump':
                if np.random.uniform(0.0, 1.0, 1) < 0.7:
                    # print('miss')
                    pass
                else:
                    self.state_dict['p_hp'] -= b_d
            else:
                self.state_dict['p_hp'] -= b_d
            self.state_dict[f'b_cool_{b_i}'] = b_c + 1
        for i in range(self.num_bs):
            self.state_dict[f'b_cool_{i}'] = max([self.state_dict[f'b_cool_{i}'] - 1, 0])

        if p_sc == 0:
            if p_n == 'heal':
                # print('healed')
                self.state_dict['p_hp'] += 20
            else:
                self.state_dict['b_hp'] -= p_d
            self.state_dict[f'p_cool_{p_i}'] = p_c + 1
        for i in range(self.num_ps):
            self.state_dict[f'p_cool_{i}'] = max([self.state_dict[f'p_cool_{i}'] - 1, 0])

        self.state = np.array(list(self.state_dict.values()))


class MLP(nn.Module):

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 num_neurons: list = [64, 32],
                 hidden_act: str = 'ReLU',
                 out_act: str = 'Identity'):
        super(MLP, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_neurons = num_neurons
        self.hidden_act = getattr(nn, hidden_act)()
        self.out_act = getattr(nn, out_act)()

        input_dims = [input_dim] + num_neurons
        output_dims = num_neurons + [output_dim]

        self.layers = nn.ModuleList()
        for i, (in_dim, out_dim) in enumerate(zip(input_dims, output_dims)):
            is_last = True if i == len(input_dims) - 1 else False
            self.layers.append(nn.Linear(in_dim, out_dim))
            if is_last:
                self.layers.append(self.out_act)
            else:
                self.layers.append(self.hidden_act)

    def forward(self, xs):
        for layer in self.layers:
            xs = layer(xs)
        return xs


class DQN(nn.Module):

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 qnet: nn.Module,
                 qnet_target: nn.Module,
                 lr: float,
                 gamma: float,
                 epsilon: float):
        """
        :param state_dim: input state dimension
        :param action_dim: action dimension
        :param qnet: main q network
        :param qnet_target: target q network
        :param lr: learning rate
        :param gamma: discount factor of MDP
        :param epsilon: E-greedy factor
        """

        super(DQN, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.qnet = qnet
        self.lr = lr
        self.gamma = gamma
        self.opt = torch.optim.Adam(params=self.qnet.parameters(), lr=lr)
        self.register_buffer('epsilon', torch.ones(1) * epsilon)

        # target network related
        qnet_target.load_state_dict(qnet.state_dict())
        self.qnet_target = qnet_target
        self.criteria = nn.SmoothL1Loss()

    def get_action(self, state):
        qs = self.qnet(state)
        prob = np.random.uniform(0.0, 1.0, 1)
        if torch.from_numpy(prob).float() <= self.epsilon:  # random
            action = np.random.choice(range(self.action_dim))
        else:  # greedy
            action = qs.argmax(dim=-1)
        return int(action)

    def update(self, state, action, reward, next_state, done):
        s, a, r, ns = state, action, reward, next_state

        # compute Q-Learning target with 'target network'
        with torch.no_grad():
            q_max, _ = self.qnet_target(ns).max(dim=-1, keepdims=True)
            q_target = r + self.gamma * q_max * (1 - done)

        q_val = self.qnet(s).gather(1, a)
        loss = self.criteria(q_val, q_target)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()


from random import sample
class ReplayMemory:
    def __init__(self, max_size):
        # deque object that we've used for 'episodic_memory' is not suitable for random sampling
        # here, we instead use a fix-size array to implement 'buffer'
        self.buffer = [None] * max_size
        self.max_size = max_size
        self.index = 0
        self.size = 0

    def push(self, obj):
        self.buffer[self.index] = obj
        self.size = min(self.size + 1, self.max_size)
        self.index = (self.index + 1) % self.max_size

    def sample(self, batch_size):
        indices = sample(range(self.size), batch_size)
        return [self.buffer[index] for index in indices]

    def __len__(self):
        return self.size


from types import prepare_class
def to_tensor(np_array: np.array, size=None) -> torch.tensor:
    torch_tensor = torch.from_numpy(np_array).float()
    if size is not None:
        torch_tensor = torch_tensor.view(size)
    return torch_tensor


def to_numpy(torch_tensor: torch.tensor) -> np.array:
    return torch_tensor.cpu().detach().numpy()


def prepare_training_inputs(sampled_exps, device='cpu'):
    states = []
    actions = []
    rewards = []
    next_states = []
    dones = []
    for sampled_exp in sampled_exps:
        states.append(sampled_exp[0])
        actions.append(sampled_exp[1])
        rewards.append(sampled_exp[2])
        next_states.append(sampled_exp[3])
        dones.append(sampled_exp[4])

    states = torch.cat(states, dim=0).float().to(device)
    actions = torch.cat(actions, dim=0).to(device)
    rewards = torch.cat(rewards, dim=0).float().to(device)
    next_states = torch.cat(next_states, dim=0).float().to(device)
    dones = torch.cat(dones, dim=0).float().to(device)
    return states, actions, rewards, next_states, dones


def get_agent_to_load(reward_rate, print_all=False, memory_size=1, batch_size=1, ):
    env = boss_raid_simulater(b_skills, p_skills, reward_rate)
    s_dim = env.state.shape[0]
    a_dim = len(p_skills)

    qnet = MLP(s_dim, a_dim, num_neurons=[64,64])
    qnet_target = MLP(s_dim, a_dim, num_neurons=[64,64])

    # initialize target network same as the main network.
    qnet_target.load_state_dict(qnet.state_dict())
    agent = DQN(s_dim, 1, qnet=qnet, qnet_target=qnet_target, lr=1e-4 * 5, gamma=0.88, epsilon=1.0)
    memory = ReplayMemory(memory_size)

    # epsilon scheduling
    # slowly decaying_epsilon
    epsilon = 0.8
    agent.epsilon = torch.tensor(epsilon)
    env = boss_raid_simulater(b_skills, p_skills, reward_rate)
    s = env.observe()
    cum_r = 0

    while True:
        s = to_tensor(s, size=(1, s_dim))
        a = agent.get_action(s)
        ns, r, done = env.step(a)

        experience = (s,
                      torch.tensor(a).view(1, 1),
                      torch.tensor(r / 100.0).view(1, 1),
                      torch.tensor(ns).view(1, s_dim),
                      torch.tensor(done).view(1, 1))
        memory.push(experience)

        s = ns
        cum_r += r
        if done:
            break

    # train agent
    sampled_exps = memory.sample(batch_size)
    sampled_exps = prepare_training_inputs(sampled_exps)
    agent.update(*sampled_exps)

    qnet_target.load_state_dict(qnet.state_dict())

    return agent


def load_agent(agent_name):
    reward_rate = [0, 0, 0, 0]
    agent_load = get_agent_to_load(reward_rate)
    agent_load.load_state_dict(torch.load(agent_name))

    return agent_load


def get_single_play_log(agent):
    reward_rate = [0, 0, 0, 0]
    sum_wr = 0

    log = {'states': [],
           'p_actions': [],
           'b_actions': [], }

    env = boss_raid_simulater(b_skills, p_skills, reward_rate)
    s_dim = env.state.shape[0]
    a_dim = len(p_skills)
    s = env.observe()
    cum_r = 0

    states = []
    actions = []
    rewards = []

    while True:
        s = to_tensor(s, size=(1, s_dim))
        a = agent.get_action(s)
        ns, r, done = env.step(a)

        states.append(list(to_numpy(s)[0]))
        actions.append(a)
        rewards.append(r)

        s = ns
        cum_r += r
        if done:
            break

    log['states'] = str(states)
    log['p_actions'] = str(actions)
    log['b_actions'] = str(env.boss_action)

    return log
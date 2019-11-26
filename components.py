import random, math, matplotlib, matplotlib.pyplot as plt, numpy as np, torch.nn.functional as F
import torch, torch.nn as nn, torch.optim as optim, torch.autograd, torchvision.transforms, torch.nn.functional as F
from collections import namedtuple
from itertools import count
from PIL import Image
from utils import conv2d_size_out
from baselines.common.atari_wrappers import LazyFrames
from torch.nn.parameter import Parameter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RL_AGENT(torch.nn.Module):
    def __init__(self, env, gamma, seed):
        super(RL_AGENT, self).__init__()
        self.env_name = env.spec._env_name
        self.gamma = gamma
        self.seed = seed
        self.observation_space, self.action_space = env.observation_space, env.action_space
    
    @staticmethod
    def sample_action(probs):
        """
        Select a discrete action based on a generated distribution (probs) over the actions
        """
        return int(torch.distributions.Categorical(probs=probs).sample())

    @staticmethod
    def epsilon_greedy(values, epsilon): # not used!
        """
        generating a distribution of the actions using epsilon greedy
        """
        probs = epsilon / values.size()[-1] * torch.ones_like(values)
        probs[values.argmax(-1)] += (1 - epsilon)
        return probs

class FEATURE_EXTRACTOR(nn.Module):
    """
    extracting features (states) from observations
    inputs an observation from the environment and outputs a vector representation of the states
    should be using some kind of RNN to deal with partial observability
    """
    def __init__(self, shape_input, func_activation=F.relu):
        super(FEATURE_EXTRACTOR, self).__init__()
        h, w, channels_input = shape_input
        self.func_activation = func_activation
        self.layer_conv1 = nn.Conv2d(channels_input, 32, 8, stride=4, padding=0) # input: (batchsize, c, h, w)
        self.layer_conv2 = nn.Conv2d(32, 64, 4, stride=2, padding=0)
        self.layer_conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.convw, self.convh = 7, 7
        self.reset_parameters()

    def get_wh(self):
        return self.convw, self.convh

    def forward(self, x):
        x = self.layer_conv1(x) # [, 4, 84, 84] -> [, 32, 20, 20]
        x = self.func_activation(x)
        x = self.layer_conv2(x) # [, 32, 20, 20] -> [, 64, 9, 9]
        x = self.func_activation(x)
        x = self.layer_conv3(x) # [, 64, 9, 9] -> [1, 64, 7, 7]
        x = self.func_activation(x)
        return x

    def reset_parameters(self):
        self.layer_conv1.weight.data.normal_(0, 0.1)
        self.layer_conv2.weight.data.normal_(0, 0.1)
        self.layer_conv3.weight.data.normal_(0, 0.1)

class DQN_NETWORK(nn.Module):
    def __init__(self, feature_extractor, num_actions, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(DQN_NETWORK, self).__init__()
        self.feature_extractor, self.num_actions = feature_extractor, num_actions
        convw, convh = self.feature_extractor.get_wh()
        self.before_head = nn.Linear(convw * convh * 64, 512)
        self.head_action_values = nn.Linear(512, num_actions)
        self.reset_parameters()
        self.device = device
    
    def forward(self, obs):
        if isinstance(obs, LazyFrames):
            obs = np.array(obs, dtype=np.float32, copy=False)
            obs = torch.tensor(obs / 255.0, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        elif isinstance(obs, np.ndarray):
            obs = torch.tensor(obs / 255.0, dtype=torch.float32).permute(0, 3, 1, 2)
        u = self.feature_extractor(obs).view(-1, self.before_head.in_features) # input: (batchsize, c, h, w)
        u = self.before_head(u)
        u = self.head_action_values(u.view(u.size(0), -1))
        return u
    
    def reset_parameters(self):
        self.feature_extractor.reset_parameters()
        self.head_action_values.weight.data.normal_(0, 0.1)
        self.before_head.weight.data.normal_(0, 0.1)
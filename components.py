import torch
from utils import init_weights

class RL_AGENT(torch.nn.Module):
    def __init__(self, env, gamma, seed):
        super(RL_AGENT, self).__init__()
        self.gamma = gamma
        self.seed = seed
        self.observation_space, self.action_space = env.observation_space, env.action_space

class ENCODER_ATARI(torch.nn.Module):
    """
    extracting features (states) from observations
    inputs an observation from the environment and outputs a vector representation of the states
    """
    def __init__(self, shape_input):
        super(ENCODER_ATARI, self).__init__()
        self.channels_in = shape_input[-1]
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(self.channels_in, 32, 8, stride=4, padding=0), # input: (batchsize, c, h, w), be careful!
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 4, stride=2, padding=0),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            torch.nn.Flatten(), # no relu on state representation, move relu to the start of the Q estimator
        )
        init_weights(self.layers)
        self.len_output = 7 * 7 * 64

    def forward(self, x):
        return self.layers(x)

class ESTIMATOR_Q(torch.nn.Module):
    def __init__(self, num_actions, len_input, width=512):
        super(ESTIMATOR_Q, self).__init__()
        self.len_input = len_input        
        self.layers = torch.nn.Sequential(
            torch.nn.ReLU(), # don't use a relu'ed representation
            torch.nn.Linear(self.len_input, width),
            torch.nn.ReLU(),
            torch.nn.Linear(width, num_actions),
        )
        init_weights(self.layers)

    def forward(self, x):
        return self.layers(x)

class DQN_NETWORK(torch.nn.Module):
    def __init__(self, encoder, estimator_Q, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(DQN_NETWORK, self).__init__()
        self.encoder, self.estimator_Q = encoder, estimator_Q
        self.device = device
    
    def forward(self, obs):
        state = self.encoder(obs)
        value_Q = self.estimator_Q(state)
        return value_Q
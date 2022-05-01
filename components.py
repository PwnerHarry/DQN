import torch

class RL_AGENT(torch.nn.Module):
    def __init__(self, env, gamma, seed):
        super(RL_AGENT, self).__init__()
        self.env_name = env.spec._env_name
        self.gamma = gamma
        self.seed = seed
        self.observation_space, self.action_space = env.observation_space, env.action_space
    
    # @staticmethod
    # def sample_action(probs):
    #     """
    #     Select a discrete action based on a generated distribution (probs) over the actions
    #     """
    #     return int(torch.distributions.Categorical(probs=probs).sample())

    # @staticmethod
    # def epsilon_greedy(values, epsilon): # not used!
    #     """
    #     generating a distribution of the actions using epsilon greedy
    #     """
    #     probs = epsilon / values.size()[-1] * torch.ones_like(values)
    #     probs[values.argmax(-1)] += (1 - epsilon)
    #     return probs

class ENCODER_ATARI(torch.nn.Module):
    """
    extracting features (states) from observations
    inputs an observation from the environment and outputs a vector representation of the states
    """
    def __init__(self, shape_input, device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")):
        super(ENCODER_ATARI, self).__init__()
        self.channels_in = shape_input[-1]
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(self.channels_in, 32, 8, stride=4, padding=0), # input: (batchsize, c, h, w), be careful!
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 4, stride=2, padding=0),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            torch.nn.Flatten(), # no relu on state representation, move relu to the start of the Q estimator
            # TODO: which are the DQN parts that we should initialize with Xavier normal?
        )
        # TODO: if device is cuda send this to cuda
        self.len_output = 7 * 7 * 64

    def forward(self, x):
        return self.layers(x)

class ENCODER_BABYAIBOW(torch.nn.Module):
    def __init__(self, shape_input, len_object=32, value_max=15, device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")):
        super(ENCODER_BABYAIBOW, self).__init__()
        self.h, self.w, self.channels_in = shape_input[-3], shape_input[-2], shape_input[-1]
        self.len_object, self.value_max = len_object, value_max
        # TODO: use a L2-normalized Normal(0, 1) embedding with size 3 * (value_max + 1)? I asked David and I expect some responses very soon
        self.embedder = torch.nn.Embedding(num_embeddings=3 * (value_max + 1), embedding_dim=len_object, device=device, dtype=torch.float32)
        self.offsets = torch.tensor([0, value_max + 1, 2 * (value_max + 1)], requires_grad=False, dtype=torch.int32, device=device).reshape([1, 1, 1, 3])
        self.len_output = self.h * self.w * self.len_object

    def forward(self, x):
        embeddings_per_dim = self.embedder(x + self.offsets.detach())
        embeddings = embeddings_per_dim.sum(dim=-2)
        return embeddings.flatten()

    """
        embedding_matrix = tf.random.normal([3 * (value_max + 1), len_output], mean=0.0, stddev=1.0)
        embedding_matrix = embedding_matrix / tf.norm(embedding_matrix, ord='euclidean', axis=-1, keepdims=True)
    """

class ESTIMATOR_Q(torch.nn.Module):
    def __init__(self, num_actions, len_input, width=512):
        super(ESTIMATOR_Q, self).__init__()
        self.len_input = len_input        
        self.layers = torch.nn.Sequential(
            torch.nn.ReLU(), # don't use a relu'ed representation
            torch.nn.Linear(self.len_input, width),
            torch.nn.ReLU(),
            torch.nn.Linear(width, num_actions), # TODO: which are the DQN parts that we should initialize with Xavier normal?
        )

    def forward(self, x):
        return self.layers(x)

class DQN_NETWORK(torch.nn.Module):
    def __init__(self, encoder, estimator_Q, num_actions, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(DQN_NETWORK, self).__init__()
        self.encoder, self.estimator_Q = encoder, estimator_Q
        self.device = device
    
    def forward(self, obs):
        state = self.encoder(obs)
        value_Q = self.estimator_Q(state)
        return value_Q
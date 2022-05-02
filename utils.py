import gym, torch, numpy as np, random
from collections import deque
from cpprb import PrioritizedReplayBuffer, ReplayBuffer
from gym import spaces
import cv2


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84, grayscale=True, dict_space_key=None):
        """
        Warp frames to height x width
        If the environment uses dictionary observations, `dict_space_key` can be specified which indicates which observation should be warped.
        """
        super().__init__(env)
        self._width = width
        self._height = height
        self._grayscale = grayscale
        self._key = dict_space_key
        if self._grayscale:
            num_colors = 1
        else:
            num_colors = 3

        new_space = gym.spaces.Box(low=0, high=255, shape=(self._height, self._width, num_colors), dtype=np.uint8)
        if self._key is None:
            original_space = self.observation_space
            self.observation_space = new_space
        else:
            original_space = self.observation_space.spaces[self._key]
            self.observation_space.spaces[self._key] = new_space
        assert original_space.dtype == np.uint8 and len(original_space.shape) == 3

    def observation(self, obs):
        if self._key is None:
            frame = obs
        else:
            frame = obs[self._key]

        if self._grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        if frame.shape[0] != self._height and frame.shape[1] != self._width:
            frame = cv2.resize(frame, (self._width, self._height), interpolation=cv2.INTER_AREA)
        if self._grayscale: frame = np.expand_dims(frame, -1)

        if self._key is None:
            obs = frame
        else:
            obs = obs.copy()
            obs[self._key] = frame
        return obs

class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """ Stack k last frames. Returns lazy array, which is memory efficient. """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[:-1] + (shp[-1] * k,)), dtype=env.observation_space.dtype)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))

class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        """Take action on reset for environments that are fixed until firing."""
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)

class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done  = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0: # for Qbert sometimes we stay in lives == 0 condition for a few frames. so it's important to keep lives > 0, so that we only reset once the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs

def wrap_deepmind(env, size=(84, 84), grayscale=True, episode_life=True, frame_stack=True):
    height, width = size
    if episode_life: env = EpisodicLifeEnv(env)
    if 'FIRE' in env.unwrapped.get_action_meanings(): env = FireResetEnv(env)
    env = WarpFrame(env, width=width, height=height, grayscale=grayscale)
    if frame_stack: env = FrameStack(env, 4)
    return env

class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay buffers.
        This object should only be converted to numpy array before being passed to the model."""
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            return np.concatenate(self._frames, axis=-1)
        else:
            return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

def atariobs2tensor(obs, divide=255, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    if isinstance(obs, LazyFrames):
        obs = obs._force()
    if isinstance(obs, np.ndarray):
        if len(obs.shape) == 1 or len(obs.shape) == 3:
            obs = np.expand_dims(obs, 0)
    if divide is None:
        tensor = torch.tensor(obs, device=device, dtype=torch.float32).permute(0, 3, 1, 2)
    else:
        tensor = torch.tensor(obs, device=device).permute(0, 3, 1, 2) / 255
    return tensor

class LinearSchedule(object):
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, t):
        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)

def get_cpprb(env, size_buffer, prioritized=False):
    env_dict = get_cpprb_env_dict(env)
    if 'atari' in env.spec.entry_point:
        if prioritized:
            return PrioritizedReplayBuffer(size_buffer, env_dict, next_of=("obs"), stack_compress="obs")
        else:
            return ReplayBuffer(size_buffer, env_dict, next_of=("obs"), stack_compress="obs")
    else:
        if prioritized:
            return PrioritizedReplayBuffer(size_buffer, env_dict, next_of=("obs"))
        else:
            return ReplayBuffer(size_buffer, env_dict, next_of=("obs"))
    

def get_space_size(space):
        if isinstance(space, gym.spaces.box.Box):
            return space.shape
        elif isinstance(space, gym.spaces.discrete.Discrete):
            return [1, ]  # space.n
        else:
            raise NotImplementedError("Assuming to use Box or Discrete, not {}".format(type(space)))

def get_default_rb_dict(size, env):
    return {"size": size, "default_dtype": np.float32,
            "env_dict": {
            "obs": {"shape": get_space_size(env.observation_space)},
            "next_obs": {"shape": get_space_size(env.observation_space)},
            "act": {"shape": get_space_size(env.action_space)},
            "rew": {},
            "done": {}}}

def get_cpprb_env_dict(env):
    shape_obs = get_space_size(env.observation_space)
    env_dict = {"obs": {"shape": shape_obs}, "act": {}, "rew": {"shape": 1}, "done": {}} # "dtype", np.bool
    if isinstance(env.action_space, gym.spaces.discrete.Discrete):
        env_dict["act"]["shape"] = 1
        env_dict["act"]["dtype"] = np.uint8
    elif isinstance(env.action_space, gym.spaces.box.Box):
        env_dict["act"]["shape"] = env.action_space.shape
        env_dict["act"]["dtype"] = np.float32
    obs = env.reset()
    if isinstance(obs, np.ndarray):
        env_dict["obs"]["dtype"] = obs.dtype
    elif isinstance(obs, LazyFrames):
        env_dict["obs"]["dtype"] = obs._frames[0].dtype
    return env_dict

def evaluate_agent(env, agent, num_episodes=5):
    return_cum, episode, returns = 0, 0, []
    while episode < num_episodes:
        obs_curr, done = env.reset(), False
        while not done:
            action = agent.decide(obs_curr, eval=True)
            obs_next, reward, done, _ = env.step(action) # take a computed action
            return_cum += reward
            obs_curr = obs_next
        if env.was_real_done:
            episode += 1
            returns.append(return_cum)
            return_cum = 0
    print('evaluated returns:')
    print(returns)
    return np.mean(returns)

def filter_nickname(name_env):
    if name_env == 'pacman':
        name_env = 'MsPacmanDeterministic-v4'
    elif name_env == 'pong':
        name_env = 'PongDeterministic-v4'
    elif name_env == 'pitfall':
        name_env = 'PitfallDeterministic-v4'
    elif name_env == 'breakout':
        name_env = 'BreakoutDeterministic-v4'
    return name_env

def get_env(name_env):
    env = wrap_deepmind(gym.make(name_env), episode_life=True, frame_stack=True)
    env.spec.max_episode_steps = 27000 # 108K frames cap
    return env

def get_set_seed(seed, env):
    if len(seed):
        seed = int(seed)
    else:
        seed = random.randint(0, 1000000) # 488815 for 2, 147039 for 5
    random.seed(seed)
    np.random.seed(seed)
    env.seed(seed)
    torch.manual_seed(seed)
    return seed

def get_set_device():
    flag_cuda = torch.cuda.is_available()
    if flag_cuda:
        device = torch.device("cuda")
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    else:
        device = torch.device("cpu")
        torch.set_default_tensor_type(torch.FloatTensor)
    return device, flag_cuda

@torch.no_grad()
def init_weights(architecture):
    for layer in architecture:
        if type(layer) == torch.nn.Conv2d:
            torch.nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
        elif type(layer) == torch.nn.Linear:
            torch.nn.init.xavier_normal_(layer.weight)
            torch.nn.init.uniform_(layer.bias, -np.sqrt(1.0 / layer.in_features), np.sqrt(1.0 / layer.in_features))
        elif type(layer) == torch.nn.Conv1d:
            torch.nn.init.xavier_normal_(layer.weight)
            torch.nn.init.uniform_(layer.bias, -np.sqrt(1.0 / layer.in_channels), np.sqrt(1.0 / layer.in_channels))
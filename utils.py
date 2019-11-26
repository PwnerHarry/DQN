import gym, imageio, math, torch, numpy as np, random, copy
from torch.autograd import Variable
from baselines.common.atari_wrappers import wrap_deepmind
def conv2d_size_out(size, kernel_size=5, stride=2):
    """
    compute size of output after a conv2d layer
    provided by pytorch documentation
    """
    return (size - kernel_size - 2) // stride + 1
    # return math.ceil((size - kernel_size - 2) / stride) + 1

def evaluate_agent(env, agent, seed, num_episodes=5, render=False):
    return_cum, episode, returns = 0, 0, []
    if render:
        pass
        filename = '%s_%d_%d.mp4' % (env.spec._env_name, seed, agent.t)
        video = imageio.get_writer(filename, fps=30)
    while episode < num_episodes:
        obs_curr, done = env.reset(), False
        while not done:
            if render: video.append_data(env.render(mode='rgb_array'))
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

def save_checkpoint(env, agent, episode_elapsed):
    filename = 'DQN_%s_%d.pt' % (agent.env_name, agent.t)
    torch.save({
        'env': env,
        'episode_elapsed': episode_elapsed,
        'env_game': agent.env_name,
        'gamma': agent.gamma,
        'seed': agent.seed,
        'observation_space': agent.observation_space,
        'action_space': agent.action_space,
        't': agent.t,
        'exploration': agent.exploration,
        'beta_schedule': agent.beta_schedule,
        'agent_state_dict': agent.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'replay_buffer': agent.replay_buffer,
        'size_batch': agent.size_batch,
        'time_learning_starts': agent.time_learning_starts,
        'freq_train': agent.freq_train,
        'freq_targetnet_update': agent.freq_targetnet_update,
        'steps_total': agent.steps_total,
        'device': agent.device,
        'step_last_print': agent.step_last_print,
        'time_last_print': agent.time_last_print,
        }, filename)
    print('checkpoint saved as %s' % (filename))

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
    env = wrap_deepmind(gym.make(name_env), episode_life=True, clip_rewards=False, frame_stack=True, scale=False)
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

def load_checkpoint(path_checkpoint, agent, env, episode_elapsed, step_elapsed):
    try:
        checkpoint = torch.load(path_checkpoint)
        agent.load_checkpoint(checkpoint)
        env = checkpoint['env']
        episode_elapsed = checkpoint['episode_elapsed']
        step_elapsed = checkpoint['t']
    except FileNotFoundError:
        print('cannot load specified checkpoint: %s' % (path_checkpoint))
    return agent, env, episode_elapsed, step_elapsed
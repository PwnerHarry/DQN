"""
This file is meant to contain all the assistive functions
"""

import imageio, math, torch, numpy as np, random, copy

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
    return np.mean(returns)

def save_checkpoint(env, agent, episode):
    filename = 'DQN_%s_%d.pt' % (env.spec._env_name, agent.t)
    torch.save({
        'env': env,
        'episode': episode,
        'env_game': agent.env_name,
        'gamma': agent.gamma,
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
        'freq_print': agent.freq_print,
        'steps_total': agent.steps_total,
        'device': agent.device,
        'step_last_print': agent.step_last_print,
        'time_last_print': agent.time_last_print,
        }, filename)
    print('checkpoint saved as %s' % (filename))
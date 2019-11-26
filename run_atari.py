import time, warnings, argparse, scipy.io, gym, random, datetime, torch
from utils import *
from DQN import DQN
from components import FEATURE_EXTRACTOR, DQN_NETWORK
from torch.utils.tensorboard import SummaryWriter
import os, psutil
process = psutil.Process(os.getpid())

parser = argparse.ArgumentParser(description='')
parser.add_argument('--game', type=str, default='breakout', help='')
parser.add_argument('--lr', type=float, default=0.0000625, help='')
parser.add_argument('--gamma', type=float, default=0.99, help='')
parser.add_argument('--steps', type=int, default=50000000, help='')
parser.add_argument('--episodes', type=int, default=50000000, help='')
parser.add_argument('--method', type=str, default='DQN', help='')
parser.add_argument('--freq_eval', type=int, default=500, help='')
parser.add_argument('--freq_checkpoint', type=int, default=1000000, help='')
parser.add_argument('--path_checkpoint', type=str, default='', help='')
parser.add_argument('--size_buffer', type=int, default=1000000, help='')
parser.add_argument('--prioritized_replay', type=int, default=0, help='')
parser.add_argument('--seed', type=str, default='', help='')

args = parser.parse_args()
device, args.cuda = get_set_device()

args.game = filter_nickname(args.game)
env = get_env(args.game)
seed = get_set_seed(args.seed, env)
writer = SummaryWriter("%s/%s/%d" % (env.spec._env_name, args.method, seed))

if args.method == 'DQN':
    feature_extractor = FEATURE_EXTRACTOR(shape_input=env.observation_space.shape)
    network = DQN_NETWORK(feature_extractor, num_actions=env.action_space.n)
    if args.cuda: network = network.cuda()
    agent = DQN(env, network, steps_total=args.steps, prioritized_replay=bool(args.prioritized_replay), size_buffer=args.size_buffer, gamma=args.gamma, lr=args.lr, seed=seed)

step_elapsed, episode_elapsed = 0, 0
agent, env, episode_elapsed, step_elapsed = load_checkpoint(args.path_checkpoint, agent, env, episode_elapsed, step_elapsed)

return_cum, step_episode = 0, 0
print('initialization completed')
time_start, time_episode_start = time.time(), time.time()
while step_elapsed <= args.steps and episode_elapsed <= args.episodes:
    obs_curr, done = env.reset(), False
    while not done and step_elapsed <= args.steps:
        if args.method == 'random':
            obs_next, reward, done, _ = env.step(env.action_space.sample()) # take a random action
        else:
            action = agent.decide(obs_curr)
            obs_next, reward, done, _ = env.step(action) # take a computed action
            agent.step(obs_curr, action, reward, obs_next, done)
        return_cum += reward
        step_episode += 1
        obs_curr = obs_next
    if env.was_real_done:
        time_episode_end = time.time()
        writer.add_scalar('Return/train', return_cum, step_elapsed)
        if episode_elapsed % args.freq_eval == 0 and args.method != 'random':
            return_eval_avg = evaluate_agent(env, agent, seed, num_episodes=5, render=True)
            print('--steps: %d, return_eval_avg: %.2f' % (step_elapsed, return_eval_avg))
            writer.add_scalar('Return/eval', return_eval_avg, step_elapsed)
        step_elapsed += step_episode
        episode_elapsed += 1
        if args.method == 'random':
            print('episode: %d, return: %.2f, steps: %d, fps_episode: %.2f, fps_overall: %.2f' % (episode_elapsed, return_cum, step_episode, step_episode / (time_episode_end - time_episode_start), step_elapsed / (time_episode_end - time_start)))
        elif args.method == 'DQN':
            epsilon = agent.exploration.value(agent.t)
            fps_episode = 4.0 * step_episode / (time_episode_end - time_episode_start) # 4 frames per agent step
            fps_overall = 4.0 * step_elapsed / (time_episode_end - time_start)
            eta = str(datetime.timedelta(seconds=int(4 * (args.steps - step_elapsed) / fps_overall)))
            writer.add_scalar('Other/epsilon', epsilon, step_elapsed)
            writer.add_scalar('Other/fps_episode', fps_episode, step_elapsed)
            writer.add_scalar('Other/trans_buffer', len(agent.replay_buffer), step_elapsed)
            writer.add_scalar('Other/usage_memory', process.memory_info().rss / (1024 ** 2), step_elapsed)
            print('episode: %d, epsilon: %.2f, return: %.2f, steps: %d, fps_episode: %.2f, fps_overall: %.2f, eta: %s' % (episode_elapsed, epsilon, return_cum, step_episode, fps_episode, fps_overall, eta))
        return_cum, step_episode, time_episode_start = 0, 0, time.time()
        if episode_elapsed % args.freq_checkpoint == 0:
            save_checkpoint(env, agent, episode_elapsed)
    else:
        continue
time_end = time.time()
env.close()
time_duration = time_end - time_start
print('total time elasped %.1fs' % (time_duration))
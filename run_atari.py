import time, argparse, datetime
from utils import get_env, evaluate_agent, get_set_device, filter_nickname, get_set_seed
from models import DQN
from components import ENCODER_ATARI, ESTIMATOR_Q, DQN_NETWORK
from tensorboardX import SummaryWriter
import os, psutil
process = psutil.Process(os.getpid())

parser = argparse.ArgumentParser(description='')
parser.add_argument('--game', type=str, default='pacman', help='')
parser.add_argument('--lr', type=float, default=0.0000625, help='')
parser.add_argument('--gamma', type=float, default=0.99, help='')
parser.add_argument('--steps_total', type=int, default=50000000, help='')
parser.add_argument('--episodes', type=int, default=50000000, help='')
parser.add_argument('--method', type=str, default='DQN', help='')
parser.add_argument('--freq_eval', type=int, default=500, help='')
parser.add_argument('--freq_checkpoint', type=int, default=1000000, help='')
parser.add_argument('--path_checkpoint', type=str, default='', help='')
parser.add_argument('--size_buffer', type=int, default=1000000, help='')
parser.add_argument('--prioritized_replay', type=int, default=1, help='')
parser.add_argument('--size_batch', type=int, default=32, help='')
parser.add_argument('--seed', type=str, default='', help='')

args = parser.parse_args()
device, args.cuda = get_set_device()

args.game = filter_nickname(args.game)
env = get_env(args.game)
seed = get_set_seed(args.seed, env)
writer = SummaryWriter("%s/%s/%d" % (env.spec.id, args.method, seed))

if args.method == 'DQN':
    encoder = ENCODER_ATARI(shape_input=env.observation_space.shape)
    estimator_Q = ESTIMATOR_Q(num_actions=env.action_space.n, len_input=encoder.len_output, width=512)
    network = DQN_NETWORK(encoder, estimator_Q, device=device)
    agent = DQN(env, network,
        gamma=args.gamma,
        steps_total=args.steps_total,
        prioritized_replay=bool(args.prioritized_replay),
        type_optimizer='Adam', lr=args.lr,
        size_batch=args.size_batch,
        device=device,
        seed=42,
    )
    if args.cuda: agent = agent.cuda()
else:
    raise NotImplementedError('only DQN agents are implemented')

step_elapsed, episode_elapsed = 0, 0
return_cum, step_episode = 0, 0

print('initialization completed')

time_start, time_episode_start = time.time(), time.time()
while step_elapsed <= args.steps_total and episode_elapsed <= args.episodes:
    obs_curr, done = env.reset(), False
    while not done and step_elapsed <= args.steps_total:
        if args.method == 'random':
            obs_next, reward, done, _ = env.step(env.action_space.sample()) # take a random action
        else:
            action = agent.decide(obs_curr)
            obs_next, reward, done, _ = env.step(action) # take a computed action
            agent.step(obs_curr, action, reward, obs_next, env.was_real_done)
        return_cum += reward
        step_episode += 1
        obs_curr = obs_next
    if env.was_real_done:
        time_episode_end = time.time()
        writer.add_scalar('Return/train', return_cum, step_elapsed)
        if episode_elapsed % args.freq_eval == 0 and args.method != 'random':
            return_eval_avg = evaluate_agent(env, agent, num_episodes=5)
            print('--steps: %d, return_eval_avg: %.2f' % (step_elapsed, return_eval_avg))
            writer.add_scalar('Return/eval', return_eval_avg, step_elapsed)
        step_elapsed += step_episode
        episode_elapsed += 1
        if args.method == 'random':
            print('episode: %d, return: %.2f, steps: %d, fps_episode: %.2f, fps_overall: %.2f' % (episode_elapsed, return_cum, step_episode, step_episode / (time_episode_end - time_episode_start), step_elapsed / (time_episode_end - time_start)))
        elif args.method == 'DQN':
            epsilon = agent.schedule_epsilon.value(agent.t)
            fps_episode = 4.0 * step_episode / (time_episode_end - time_episode_start) # 4 frames per agent step
            fps_overall = 4.0 * step_elapsed / (time_episode_end - time_start)
            eta = str(datetime.timedelta(seconds=int(4 * (args.steps_total - step_elapsed) / fps_overall)))
            writer.add_scalar('Other/epsilon', epsilon, step_elapsed)
            writer.add_scalar('Other/fps_episode', fps_episode, step_elapsed)
            writer.add_scalar('Other/transitions_stored', agent.replay_buffer.get_stored_size(), step_elapsed)
            writer.add_scalar('Other/usage_memory', process.memory_info().rss / (1024 ** 2), step_elapsed)
            print('episode: %d, epsilon: %.2f, return: %.2f, steps: %d, fps_episode: %.2f, fps_overall: %.2f, eta: %s' % (episode_elapsed, epsilon, return_cum, step_episode, fps_episode, fps_overall, eta))
        return_cum, step_episode, time_episode_start = 0, 0, time.time()
    else:
        continue
time_end = time.time()
env.close()
time_duration = time_end - time_start
print('total time elasped %.1fs' % (time_duration))
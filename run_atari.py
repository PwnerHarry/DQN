import time, warnings, argparse, scipy.io, gym, random, datetime, torch
from utils import *
from baselines.common.atari_wrappers import wrap_deepmind
from DQN import DQN
from components import FEATURE_EXTRACTOR, DQN_NETWORK
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description='')
parser.add_argument('--game', type=str, default='breakout', help='')
parser.add_argument('--lr', type=float, default=0.0000625, help='')
parser.add_argument('--gamma', type=float, default=0.99, help='')
parser.add_argument('--steps', type=int, default=50000000, help='')
parser.add_argument('--episodes', type=int, default=50000000, help='')
parser.add_argument('--runtimes', type=int, default=8, help='')
parser.add_argument('--learner_type', type=str, default='togtd', help='')
parser.add_argument('--evaluate_others', type=int, default=1, help='')
parser.add_argument('--evaluate_ours', type=int, default=1, help='')
parser.add_argument('--method', type=str, default='DQN', help='')
parser.add_argument('--freq_eval', type=int, default=1000, help='')
parser.add_argument('--freq_checkpoint', type=int, default=1000000, help='')
parser.add_argument('--path_checkpoint', type=str, default='', help='')

args = parser.parse_args()
args.cuda = torch.cuda.is_available()
if args.cuda:
    device = torch.device("cuda")
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    device = torch.device("cpu")
    torch.set_default_tensor_type(torch.FloatTensor)

if args.game == 'pacman':
    args.game = 'MsPacmanDeterministic-v4'
elif args.game == 'pong':
    args.game = 'PongDeterministic-v4'
elif args.game == 'pitfall':
    args.game = 'PitfallDeterministic-v4'
elif args.game == 'breakout':
    args.game = 'BreakoutDeterministic-v4'

env = wrap_deepmind(gym.make(args.game), episode_life=True, clip_rewards=False, frame_stack=True, scale=False)
env.spec.max_episode_steps = 27000 # 108K frames cap

if args.method == 'DQN':
    feature_extractor = FEATURE_EXTRACTOR(shape_input=env.observation_space.shape)
    network = DQN_NETWORK(feature_extractor, num_actions=env.action_space.n)
    if args.cuda: network = network.cuda()
    agent = DQN(env, network, steps_total=args.steps, gamma=args.gamma, lr=args.lr)

step_total = 0
episode = 0

try:
    checkpoint = torch.load(args.path_checkpoint)
    agent.load_checkpoint(checkpoint)
    env = checkpoint['env']
    episode = checkpoint['episode']
    step_total = checkpoint['t']
except FileNotFoundError:
    pass

seed = random.randint(0, 1000000)
random.seed(seed)
np.random.seed(seed)
env.seed(seed)
torch.manual_seed(seed)
writer = SummaryWriter("%s_%s/%d" % (args.method, agent.env_name, seed))

return_cum, step_episode = 0, 0
print('initialization completed')
time_start = time.time()
time_episode_start = time.time()
while step_total <= args.steps and episode <= args.episodes:
    obs_curr, done = env.reset(), False
    while not done and step_total <= args.steps:
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
        writer.add_scalar('Return/train', return_cum, step_total)
        if episode % args.freq_eval == 0 and args.method != 'random':
            return_eval_avg = evaluate_agent(env, agent, seed, num_episodes=5, render=True)
            print('--steps: %d, return_eval_avg: %.2f' % (step_total, return_eval_avg))
            writer.add_scalar('Return/eval', return_eval_avg, step_total)
        step_total += step_episode
        episode += 1
        if args.method == 'random':
            print('episode: %d, return: %.2f, steps: %d, fps_episode: %.2f, fps_overall: %.2f' % (episode, return_cum, step_episode, step_episode / (time_episode_end - time_episode_start), step_total / (time_episode_end - time_start)))
        elif args.method == 'DQN':
            epsilon = agent.exploration.value(agent.t)
            fps_episode = 4.0 * step_episode / (time_episode_end - time_episode_start) # 4 frames per agent step
            fps_overall = 4.0 * step_total / (time_episode_end - time_start)
            eta = str(datetime.timedelta(seconds=int(4 * (args.steps - step_total) / fps_overall)))
            writer.add_scalar('Other/epsilon', epsilon, step_total)
            writer.add_scalar('Other/fps_episode', fps_episode, step_total)
            print('episode: %d, epsilon: %.2f, return: %.2f, steps: %d, fps_episode: %.2f, fps_overall: %.2f, eta: %s' % (episode, epsilon, return_cum, step_episode, fps_episode, fps_overall, eta))
        return_cum, step_episode, time_episode_start = 0, 0, time.time()
        if episode % args.freq_checkpoint == 0:
            save_checkpoint(env, agent, episode)
    else:
        continue
time_end = time.time()
env.close()
time_duration = time_end - time_start
print('total time elasped %.1fs' % (time_duration))
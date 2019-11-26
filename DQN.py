import time, math, matplotlib, matplotlib.pyplot as plt, numpy as np, copy, random
import torch, torch.nn as nn, torch.optim as optim, torch.autograd, torchvision.transforms, torch.nn.functional as F
from components import *
from replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from baselines.common.schedules import LinearSchedule

class DQN(RL_AGENT):
    def __init__(self,
        env, 
        network_policy,
        gamma=1.0,
        exploration_fraction=0.02, exploration_final_eps=0.01, steps_total=50000000,
        size_buffer=1000000, prioritized_replay=True, alpha_prioritized_replay=0.6, prioritized_replay_beta0=0.4, prioritized_replay_beta_iters=None, prioritized_replay_eps=1e-6,
        type_optimizer='Adam', lr=5e-4, eps=1.5e-4,
        time_learning_starts=20000, freq_targetnet_update=8000, freq_train=4, size_batch=32,
        callback=None, load_path=None, # for debugging
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        seed=42,
        **network_kwargs):

        super(DQN, self).__init__(env, gamma, seed)
        self.create_replay_buffer(prioritized_replay, prioritized_replay_eps, size_buffer, alpha_prioritized_replay, prioritized_replay_beta0, prioritized_replay_beta_iters, steps_total)
        self.exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * steps_total), initial_p=1.0, final_p=exploration_final_eps)
        
        self.network_policy = network_policy # an instance of DQN_NETWORK, which contains an instance of FEATURE_EXTRACTOR and 1 additional head
        self.optimizer = eval('optim.%s' % type_optimizer)(self.network_policy.parameters(), lr=lr, eps=eps)

        # initialize target network
        self.network_target = copy.deepcopy(self.network_policy)
        for param in self.network_target.parameters():
            param.requires_grad = False
        self.network_target.eval()

        self.size_batch = size_batch
        self.time_learning_starts = time_learning_starts
        self.freq_train = freq_train
        self.freq_targetnet_update = freq_targetnet_update
        self.t, self.steps_total = 0, steps_total
        self.device = device
        self.step_last_print, self.time_last_print = 0, None
    
    def load_checkpoint(self, checkpoint):
        """
        loads checkpoint saved by utils/save_checkpoint
        """
        self.load_state_dict(checkpoint['agent_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.t = checkpoint['t']
        self.gamma = checkpoint['gamma']
        self.seed = checkpoint['seed']
        self.exploration = checkpoint['exploration']
        self.observation_space = checkpoint['observation_space']
        self.action_space = checkpoint['action_space']
        self.beta_schedule = checkpoint['beta_schedule']
        self.replay_buffer = checkpoint['replay_buffer']
        self.size_batch = checkpoint['size_batch']
        self.time_learning_starts = checkpoint['time_learning_starts']
        self.freq_train = checkpoint['freq_train']
        self.freq_targetnet_update = checkpoint['freq_targetnet_update']
        self.steps_total = checkpoint['steps_total']
        self.device = checkpoint['device']
        self.step_last_print = checkpoint['step_last_print']
        self.time_last_print = checkpoint['time_last_print']
        print('checkpoint loaded with replay buffer of size %d' % (len(self.replay_buffer)))

    def create_replay_buffer(self, prioritized_replay, prioritized_replay_eps, size_buffer, alpha_prioritized_replay, prioritized_replay_beta0, prioritized_replay_beta_iters, steps_total):
        self.prioritized_replay = prioritized_replay
        self.prioritized_replay_eps = prioritized_replay_eps
        if prioritized_replay:
            self.replay_buffer = PrioritizedReplayBuffer(size_buffer, alpha=alpha_prioritized_replay)
            if prioritized_replay_beta_iters is None:
                prioritized_replay_beta_iters = steps_total
            self.beta_schedule = LinearSchedule(prioritized_replay_beta_iters, initial_p=prioritized_replay_beta0, final_p=1.0)
        else:
            self.replay_buffer = ReplayBuffer(size_buffer)
            self.beta_schedule = None
        pass

    def decide(self, obs, eval=False): # Validated by Harry 17h45 23-11-2019
        """
        input observation and output action
        some through the computations of the policy network
        """
        if eval or random.random() > self.exploration.value(self.t):
            with torch.no_grad():
                return int(torch.argmax(self.network_policy(obs)))
        else: # explore
            return self.action_space.sample()
    
    def step(self, obs_curr, action, reward, obs_next, done, eval=False):
        """
        an agent step: in this step the agent does whatever it needs
        """
        if obs_next is not None:
            self.replay_buffer.add(obs_curr, action, np.sign(reward), obs_next, done) # clip rewards, done is the flag for whether obs_next is terminal
        if self.t >= self.time_learning_starts:
            if len(self.replay_buffer) >= self.size_batch and self.t % self.freq_train == 0:
                self.update()
            if self.t % self.freq_targetnet_update == 0:
                self.sync_parameters()
        self.t += 1

    def update(self):
        """
        update the parameters of the DQN model using the weighted sampled Bellman error
        """
        # sample a batch
        if self.prioritized_replay:
            experience = self.replay_buffer.sample(self.size_batch, beta=self.beta_schedule.value(self.t))
            (batch_obs_curr, batch_action, batch_reward, batch_obs_next, batch_done, weights, batch_idxes) = experience
        else:
            batch_obs_curr, batch_action, batch_reward, batch_obs_next, batch_done = self.replay_buffer.sample(self.size_batch)
            weights, batch_idxes = np.ones_like(batch_reward), None
        batch_action, batch_reward = torch.tensor(batch_action, dtype=torch.int64, device=self.device).view(-1, 1), torch.tensor(batch_reward, dtype=torch.float32, device=self.device)
        weights = torch.tensor(weights, dtype=torch.float32, device=self.device)
        # calculate the weighted Bellman error
        index_nonterm_trans = np.argwhere(batch_done == False).reshape(-1)
        values_next = torch.zeros_like(batch_reward, dtype=torch.float32)
        values_next[index_nonterm_trans] = self.network_target(batch_obs_next[index_nonterm_trans]).max(1)[0].detach()
        values_curr = self.network_policy(batch_obs_curr).gather(1, index=batch_action).view(-1)
        error_bellman = F.smooth_l1_loss(values_curr, batch_reward + self.gamma * values_next, reduction='none') # Huber loss
        error_bellman_weighted = torch.dot(error_bellman, weights)
        # calculate gradient w.r.t. to the weighted Bellman error
        self.optimizer.zero_grad()
        error_bellman_weighted.backward()
        # gradient clipping
        for param in self.network_policy.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        # update prioritized replay, if used
        if self.prioritized_replay:
            new_priorities = np.abs(error_bellman.detach().cpu().numpy()) + self.prioritized_replay_eps
            self.replay_buffer.update_priorities(batch_idxes, new_priorities)

    def sync_parameters(self):
        """
        synchronize the parameters of self.network_policy and self.network_target
        """
        self.network_target.load_state_dict(self.network_policy.state_dict())
        for param in self.network_target.parameters():
            param.requires_grad = False
        self.network_target.eval()

    def reset_parameters(self):
        self.network_policy.reset_parameters()
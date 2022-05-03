import torch, numpy as np, copy, random
from components import *
from utils import get_cpprb, LinearSchedule, atariobs2tensor

class DQN_BASE(RL_AGENT):
    def __init__(self,
        env, 
        network_policy,
        gamma=0.99, clip_reward=True,
        exploration_fraction=0.02, epsilon_final_train=0.01, epsilon_eval=0.001, steps_total=50000000,
        size_buffer=1000000, prioritized_replay=True,
        func_obs2tensor=atariobs2tensor,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        seed=42,
        ):

        super(DQN_BASE, self).__init__(env, gamma, seed)

        self.clip_reward = clip_reward        
        self.schedule_epsilon = LinearSchedule(schedule_timesteps=int(exploration_fraction * steps_total), initial_p=1.0, final_p=epsilon_final_train)
        self.epsilon_eval = epsilon_eval
        
        self.network_policy = network_policy # an instance of DQN_NETWORK, which contains an instance of FEATURE_EXTRACTOR and 1 additional head

        self.t, self.steps_total = 0, steps_total
        self.device = device
        self.step_last_print, self.time_last_print = 0, None

        self.obs2tensor = func_obs2tensor

        self.prioritized_replay = prioritized_replay
        self.replay_buffer = get_cpprb(env, size_buffer, prioritized=self.prioritized_replay)
        if self.prioritized_replay:
            self.schedule_beta_sample_priorities = LinearSchedule(steps_total, initial_p=0.4, final_p=1.0)
        self.func_loss_TD = lambda predicted, target: torch.nn.functional.smooth_l1_loss(predicted, target.detach(), reduction='none') # Huber loss

    def add_to_buffer(self, batch):
        (batch_obs_curr, batch_action, batch_reward, batch_obs_next, batch_done, weights, batch_idxes) = self.process_batch(batch, prioritized=False)
        if self.prioritized_replay:
            self.replay_buffer.add(**batch, priorities=self.calculate_priorities(batch_obs_curr, batch_action, batch_reward, batch_obs_next, batch_done, error_TD=None))
        else:
            self.replay_buffer.add(**batch)

    @torch.no_grad()
    def calculate_priorities(self, batch_obs_curr, batch_action, batch_reward, batch_obs_next, batch_done, error_TD=None):
        if error_TD is None:
            error_TD = self.calculate_TD_error(batch_obs_curr, batch_action, batch_reward, batch_obs_next, batch_done)
        else:
            assert error_TD.shape[0] == batch_reward.shape[0]
        new_priorities = np.abs(error_TD.detach().cpu().numpy()) + 1e-6
        return new_priorities

    @torch.no_grad()
    def process_batch(self, batch, prioritized=False):
        # even with prioritized replay, one would still want to process a batch without the priorities
        if prioritized:
            batch_obs_curr, batch_action, batch_reward, batch_done, batch_obs_next, weights, batch_idxes = batch.values()
            weights = torch.tensor(weights, dtype=torch.float32, device=self.device).reshape(-1, 1)
        else:
            batch_obs_curr, batch_action, batch_reward, batch_done, batch_obs_next = batch.values()
            weights, batch_idxes = None, None

        batch_reward = torch.tensor(batch_reward, dtype=torch.float32, device=self.device).reshape(-1, 1)
        batch_done = torch.tensor(batch_done, dtype=torch.bool, device=self.device).reshape(-1, 1)
        batch_action = torch.tensor(batch_action, dtype=torch.int64, device=self.device).reshape(-1, 1)

        batch_obs_curr, batch_obs_next = self.obs2tensor(batch_obs_curr, device=self.device), self.obs2tensor(batch_obs_next, device=self.device)
        if self.clip_reward: # this is a DQN-specific thing
            batch_reward = torch.sign(batch_reward)
        return (batch_obs_curr, batch_action, batch_reward, batch_obs_next, batch_done, weights, batch_idxes)

    def decide(self, obs, eval=False):
        """
        input observation and output action
        some through the computations of the policy network
        """
        if random.random() > float(eval) * self.epsilon_eval + (1 - float(eval)) * self.schedule_epsilon.value(self.t):
            with torch.no_grad():
                return int(torch.argmax(self.network_policy(self.obs2tensor(obs, device=self.device))))
        else: # explore
            return self.action_space.sample()
    
    def step(self, obs_curr, action, reward, obs_next, done, eval=False):
        if obs_next is not None:
            sample = {'obs': np.array(obs_curr), 'act': action, 'rew': reward, 'done': done, 'next_obs': np.array(obs_next)}
            self.add_to_buffer(sample)
        self.t += 1

    def calculate_TD_error(self, batch_obs_curr, batch_action, batch_reward, batch_obs_next, batch_done):
        with torch.no_grad():
            values_next = self.network_target(batch_obs_next).max(1)[0].reshape(-1, 1)
            values_next = torch.where(batch_done, torch.tensor(0.0, dtype=torch.float32, device=self.device), values_next)
        values_curr = self.network_policy(batch_obs_curr).gather(1, index=batch_action)
        error_TD = self.func_loss_TD(values_curr, (batch_reward + self.gamma * values_next).detach())
        return error_TD


class DQN(DQN_BASE):
    def __init__(self,
        env, 
        network_policy,
        gamma=0.99, clip_reward=True,
        exploration_fraction=0.02, epsilon_final_train=0.01, epsilon_eval=0.001, steps_total=50000000,
        size_buffer=1000000, prioritized_replay=True,
        type_optimizer='Adam', lr=5e-4, eps=1.5e-4,
        time_learning_starts=20000, freq_targetnet_update=8000, freq_train=4, size_batch=32,
        func_obs2tensor=atariobs2tensor,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        seed=42,
        ):

        super(DQN, self).__init__(
            env, 
            network_policy,
            gamma=gamma, clip_reward=clip_reward,
            exploration_fraction=exploration_fraction, epsilon_final_train=epsilon_final_train, epsilon_eval=epsilon_eval, steps_total=steps_total,
            size_buffer=size_buffer, prioritized_replay=prioritized_replay,
            func_obs2tensor=func_obs2tensor,
            device=device,
            seed=seed,
        )
        
        self.optimizer = eval('torch.optim.%s' % type_optimizer)(self.network_policy.parameters(), lr=lr, eps=eps)

        # initialize target network
        self.network_target = copy.deepcopy(self.network_policy)
        for param in self.network_target.parameters():
            param.requires_grad = False
        self.network_target.eval()

        self.size_batch = size_batch
        self.time_learning_starts = time_learning_starts
        self.freq_train = freq_train
        self.freq_targetnet_update = freq_targetnet_update
    
    def step(self, obs_curr, action, reward, obs_next, done, eval=False):
        """
        an agent step: in this step the agent does whatever it needs
        """
        if obs_next is not None:
            sample = {'obs': np.array(obs_curr), 'act': action, 'rew': reward, 'done': done, 'next_obs': np.array(obs_next)}
            self.add_to_buffer(sample)
        if self.t >= self.time_learning_starts:
            if self.replay_buffer.get_stored_size() >= self.size_batch and self.t % self.freq_train == 0:
                self.update()
            if self.t % self.freq_targetnet_update == 0:
                self.sync_parameters()
        self.t += 1

    def update(self, batch=None):
        """
        update the parameters of the DQN model using the weighted sampled Bellman error
        """
        if batch is None:
            if self.prioritized_replay:
                batch = self.replay_buffer.sample(self.size_batch, beta=self.schedule_beta_sample_priorities.value(self.t))
            else:
                batch = self.replay_buffer.sample(self.size_batch)
        (batch_obs_curr, batch_action, batch_reward, batch_obs_next, batch_done, weights, batch_idxes) = self.process_batch(batch, prioritized=self.prioritized_replay)
        # calculate the weighted Bellman error
        
        error_TD = self.calculate_TD_error(batch_obs_curr, batch_action, batch_reward, batch_obs_next, batch_done)

        if self.prioritized_replay:
            assert weights is not None
            error_TD_weighted = (error_TD * weights).mean() # kaixhin's rainbow implementation used mean()
        else:
            error_TD_weighted = error_TD.mean()

        self.optimizer.zero_grad()
        error_TD_weighted.backward()
        # gradient clipping
        for param in self.network_policy.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        # update prioritized replay, if used
        if self.prioritized_replay:
            new_priorities = self.calculate_priorities(batch_obs_curr, batch_action, batch_reward, batch_obs_next, batch_done, error_TD=error_TD)
            self.replay_buffer.update_priorities(batch_idxes, new_priorities.squeeze())

    def sync_parameters(self):
        """
        synchronize the parameters of self.network_policy and self.network_target
        """
        self.network_target.load_state_dict(self.network_policy.state_dict())
        for param in self.network_target.parameters():
            param.requires_grad = False
        self.network_target.eval()
        print('policy-target parameters synced')
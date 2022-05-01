import torch, numpy as np, copy, random
from components import *
from utils import get_cpprb, LinearSchedule, obs2tensor

# TODO: decouple the agent into 2-levels

class DQN(RL_AGENT):
    def __init__(self,
        env, 
        network_policy,
        gamma=1.0,
        exploration_fraction=0.02, epsilon_final_train=0.01, epsilon_eval=0.001, steps_total=50000000,
        size_buffer=1000000, prioritized_replay=True,
        type_optimizer='Adam', lr=5e-4, eps=1.5e-4,
        time_learning_starts=20000, freq_targetnet_update=8000, freq_train=4, size_batch=32,
        load_path=None, # for debugging
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        seed=42,
        ):

        super(DQN, self).__init__(env, gamma, seed)
        
        # self.create_replay_buffer(prioritized_replay, prioritized_replay_eps, size_buffer, alpha_prioritized_replay, prioritized_replay_beta0, prioritized_replay_beta_iters, steps_total)
        self.schedule_epsilon = LinearSchedule(schedule_timesteps=int(exploration_fraction * steps_total), initial_p=1.0, final_p=epsilon_final_train)
        self.epsilon_eval = epsilon_eval
        
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

        self.prioritized_replay = prioritized_replay
        self.replay_buffer = get_cpprb(env, size_buffer, prioritized=self.prioritized_replay)
        if self.prioritized_replay:
            self.schedule_beta_sample_priorities = LinearSchedule(steps_total, initial_p=0.4, final_p=1.0)

        self.lossfun_TD = lambda predicted, target: torch.nn.functional.smooth_l1_loss(predicted, target.detach(), reduction='none') # Huber loss
    
    def load_checkpoint(self, checkpoint):
        """
        loads checkpoint saved by utils/save_checkpoint
        """
        # self.load_state_dict(checkpoint['agent_state_dict'])
        # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # self.t = checkpoint['t']
        # self.gamma = checkpoint['gamma']
        # self.seed = checkpoint['seed']
        # self.schedule_epsilon = checkpoint['exploration']
        # self.observation_space = checkpoint['observation_space']
        # self.action_space = checkpoint['action_space']
        # self.schedule_beta_sample_priorities = checkpoint['schedule_beta_sample_priorities']
        # self.replay_buffer = checkpoint['replay_buffer']
        # self.size_batch = checkpoint['size_batch']
        # self.time_learning_starts = checkpoint['time_learning_starts']
        # self.freq_train = checkpoint['freq_train']
        # self.freq_targetnet_update = checkpoint['freq_targetnet_update']
        # self.steps_total = checkpoint['steps_total']
        # self.device = checkpoint['device']
        # self.step_last_print = checkpoint['step_last_print']
        # self.time_last_print = checkpoint['time_last_print']
        # print('checkpoint loaded with replay buffer of size %d' % (len(self.replay_buffer)))
        raise NotImplementedError('not implemented with cpprb')
        

    def add_batch_to_buffer(self, batch):
        # size_local_rb = local_rb.get_stored_size()
        # samples_local = local_rb.get_all_transitions()
        # local_rb.clear()
        if self.prioritized_replay:
            self.replay_buffer.add(**batch, priorities=self.calculate_priorities(batch))
        else:
            self.replay_buffer.add(**batch)
        pass

    def calculate_priorities(self, batch, error_TD=None):
        (batch_obs_curr, batch_action, batch_reward, batch_obs_next, batch_done, batch_not_done, weights, batch_idxes) = self.process_batch(batch)
        if error_TD is None:
            # calculate the weighted Bellman error
            with torch.no_grad():
                index_nonterm_trans = np.argwhere(batch_not_done).reshape(-1)
                values_next = self.network_target(batch_obs_next).max(1)[0]
                values_next = torch.where(index_nonterm_trans, values_next, 0) # TODO: validate this line
                values_curr = self.network_policy(batch_obs_curr).gather(1, index=batch_action).view(-1)
                error_TD = self.lossfun_TD(values_curr, batch_reward + self.gamma * values_next)
        else:
            assert error_TD.shape[0] == batch['rew'].size() 
        new_priorities = np.abs(error_TD.detach().cpu().numpy()) + 1e-6 # TODO: validate the update of priorities
        return new_priorities
        # self.replay_buffer.update_priorities(batch_idxes, new_priorities)

    def process_batch(self, batch):
        if self.prioritized_replay:
            (batch_obs_curr, batch_action, batch_reward, batch_done, batch_obs_next, weights, batch_idxes) = batch.values()
        else:
            (batch_obs_curr, batch_action, batch_reward, batch_done, batch_obs_next) = batch.values()
            weights, batch_idxes = None, None

        batch_reward = torch.tensor(batch_reward, dtype=torch.float32, device=self.device)
        batch_done = torch.tensor(batch_done, dtype=torch.bool, device=self.device)
        batch_action = torch.tensor(batch_action, dtype=torch.int32, device=self.device)

        batch_obs_curr, batch_obs_next = obs2tensor(batch_obs_curr), obs2tensor(batch_obs_next)
        batch_action, batch_reward, batch_done = torch.squeeze(batch_action), torch.squeeze(batch_reward), torch.squeeze(batch_done)
        # TODO: put everything about process_samples here. its not tf!
        batch_reward, batch_done, batch_not_done = self._process_samples(batch_reward, batch_done)
        if self.clip_reward:
            batch_reward = torch.sign(batch_reward)
        # else: # implement this when distributional output is provided
        #     batch_reward = tf.clip_by_value(batch_reward, clip_value_min=self.reward_min, clip_value_max=self.reward_max)
        batch_not_done = torch.logical_not(batch_done)

        return (batch_obs_curr, batch_action, batch_reward, batch_obs_next, batch_done, batch_not_done, weights, batch_idxes)

    def decide(self, obs, eval=False):
        """
        input observation and output action
        some through the computations of the policy network
        """
        if random.random() > float(eval) * self.epsilon_eval + (1 - float(eval)) * self.schedule_epsilon.value(self.t):
            with torch.no_grad():
                return int(torch.argmax(self.network_policy(obs2tensor(obs))))
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

    def update(self, batch=None):
        """
        update the parameters of the DQN model using the weighted sampled Bellman error
        """
        if batch is None:
            if self.prioritized_replay:
                batch = self.replay_buffer.sample(self.size_batch, beta=self.schedule_beta_sample_priorities.value(self.t))
            else:
                batch = self.replay_buffer.sample(self.size_batch)
        (batch_obs_curr, batch_action, batch_reward, batch_obs_next, batch_done, batch_not_done, weights, batch_idxes) = self.process_batch(batch)
        # calculate the weighted Bellman error
        with torch.no_grad():
            index_nonterm_trans = np.argwhere(batch_not_done).reshape(-1)
            values_next = self.network_target(batch_obs_next).max(1)[0]
            values_next = torch.where(index_nonterm_trans, values_next, 0) # TODO: validate this line

        values_curr = self.network_policy(batch_obs_curr).gather(1, index=batch_action).view(-1)
        error_TD = self.lossfun_TD(values_curr, batch_reward + self.gamma * values_next)
        if self.prioritized_replay:
            error_TD_weighted = error_TD.mean()
        else:
            assert weights is not None
            error_TD_weighted = torch.dot(error_TD, weights) # TODO: check if its perfect here
        self.optimizer.zero_grad()
        error_TD_weighted.backward()
        # gradient clipping
        for param in self.network_policy.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        # update prioritized replay, if used
        if self.prioritized_replay:
            new_priorities = self.calculate_priorities(batch, error_TD=None)
            self.replay_buffer.update_priorities(batch_idxes, new_priorities)

    def sync_parameters(self):
        """
        synchronize the parameters of self.network_policy and self.network_target
        """
        self.network_target.load_state_dict(self.network_policy.state_dict())
        for param in self.network_target.parameters():
            param.requires_grad = False
        self.network_target.eval()
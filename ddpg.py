import torch
import torch.autograd
import torch.optim as optim
import torch.nn as nn
from ddpg_network import *
from utils import *
from replaybuffer.prioritized import PriorityBuffer

class DDPGagent(nn.Module):
    def __init__(self, env, params, n_insize, n_outsize, device="cpu"):
        super().__init__()
        # Params
        self.num_states = n_insize
        self.num_actions = n_outsize
        self.gamma = params.gamma
        self.tau = params.tau
        self.device = device
        self.beta_init = 0.4
        self.beta = self.beta_init
        self.beta_steps = 2000
        self.ix = 0

        self.hidden_size = 256
        # Networks
        self.actor = Actor(self.num_states, self.hidden_size, self.num_actions, device=self.device).to(self.device)
        self.actor_target = Actor(self.num_states, self.hidden_size, self.num_actions, device=self.device).to(self.device)
        self.critic_1 = Critic(self.num_states + self.num_actions, self.hidden_size, self.num_actions, device=self.device).to(self.device)
        self.critic_target_1 = Critic(self.num_states + self.num_actions, self.hidden_size, self.num_actions, device=self.device).to(self.device)

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target_1.parameters(), self.critic_1.parameters()):
            target_param.data.copy_(param.data)

            # Training
            state_shape = n_insize
            state_dtype = "float32"
            self.memory = PriorityBuffer(params.buffersize, state_shape, state_dtype, params.alpha, epsilon=0.1)
            self.critic_criterion = nn.MSELoss()
            self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=params.lrvalue)
            self.critic_optimizer = optim.Adam(self.critic_1.parameters(), lr=params.lrpolicy)

    def get_action(self, state):
        state = Variable(torch.from_numpy(state).float().unsqueeze(0)).to(self.device)
        with torch.no_grad():
            action = self.actor.forward(state)
        action = action.data.cpu().numpy()[0]

        return action

    def get_enemy_actions(self, states):
        states = Variable(torch.from_numpy(states).float()).to(self.device)
        with torch.no_grad():
            action = self.actor.forward(states)
        action = action.data.cpu().numpy()

        return action

    def get_td_error(self, states, actions, rewards, next_states, dones):

        Qvals = self.critic_1.forward(states, actions)
        next_actions = self.actor_target.forward(next_states)
        next_Q = self.critic_1.forward(next_states, next_actions.detach())
        Qprime = (rewards + self.gamma * next_Q) * (1 - dones)
        critic_loss = np.abs((Qvals - Qprime))

        return critic_loss

    def update(self, batch_size):

        # Prioritized sampling from replay buffer
        self.beta = min(1.0,
                        self.beta_init + self.ix * (1.0 - self.beta_init) / self.beta_steps)
        self.ix += 1
        # batch = self.memory.sample(batch_size, beta=0.6)
        batch = None
        if not batch_size * 10 > self.memory.__len__():
            batch = self.memory.sample(batch_size, beta=0.6)
        if batch is not None:
            samples, indices, weights = batch[0], batch[1], batch[2]

            states = samples[0]
            actions = samples[1]
            rewards = samples[2]
            next_states = samples[3]
            dones = samples[4]

            states = torch.FloatTensor(states).to(self.device)
            actions = torch.FloatTensor(actions).to(self.device)
            rewards = torch.FloatTensor(rewards).to(self.device)
            next_states = torch.FloatTensor(next_states).to(self.device)
            dones = torch.FloatTensor(dones).to(self.device)

            # Critic & actor loss
            Qvals = self.critic_1.forward(states, actions.squeeze(dim=1))
            next_actions = self.actor_target.forward(next_states)
            next_Q = self.critic_target_1.forward(next_states, next_actions.detach())
            Qprime = (rewards + self.gamma * next_Q) * (1 - dones)
            critic_loss = self.critic_criterion(Qvals, Qprime)
            self.memory.update_priority(indices, abs((Qvals.detach() - Qprime.detach())).mean(dim=1))
            weighted_loss = torch.from_numpy(weights).unsqueeze(dim=1).float().to("cpu") * abs((Qvals - Qprime))
            critic_loss = weighted_loss.mean()
            policy_loss = -self.critic_1.forward(states, self.actor.forward(states)).mean()

            # update networks
            self.actor_optimizer.zero_grad()
            policy_loss.backward()
            self.actor_optimizer.step()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

        # update target networks
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

        for target_param, param in zip(self.critic_target_1.parameters(), self.critic_1.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

    def _totorch(self, container, dtype):
        if isinstance(container[0], torch.Tensor):
            tensor = torch.stack(container)
        else:
            tensor = torch.tensor(container, dtype=dtype)
        return tensor.to(self.device)

    def to(self, device):
        self.device = device
        super().to(device)

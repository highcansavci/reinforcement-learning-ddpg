import numpy as np
import torch
import torch.nn.functional as F
from replay_buffer.replay_buffer import ReplayBuffer
from model.actor import Actor
from model.critic import Critic
from model.ornstein_uhlenback_noise import OUNoise


class Agent:
    def __init__(self, actor_lr, critic_lr, input_dims, tau, gamma=0.99, n_actions=2, max_size=int(1e6), fc1_dim=400, fc2_dim=300, batch_size=64):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.actor = Actor(actor_lr, input_dims, fc1_dim, fc2_dim, n_actions, "actor")
        self.actor_target = Actor(actor_lr, input_dims, fc1_dim, fc2_dim, n_actions, "actor_target")
        self.critic = Critic(critic_lr, input_dims, fc1_dim, fc2_dim, n_actions, "critic")
        self.critic_target = Critic(critic_lr, input_dims, fc1_dim, fc2_dim, n_actions, "critic_target")
        self.noise = OUNoise(mu=np.zeros(n_actions))
        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        self.actor.eval()
        observation = torch.tensor(observation, dtype=torch.float32, device=self.actor.device).reshape(1, -1)
        mu = self.actor(observation).to(self.actor.device)
        mu_prime = mu + torch.tensor(self.noise(), dtype=torch.float32, device=self.actor.device)
        self.actor.train()
        return mu_prime.detach().cpu().numpy()

    def remember(self, state, action, reward, next_state, done):
        self.memory.store_transition(state, action, reward, next_state, done)

    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )

    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def learn(self):
        if self.memory.mem_ctr < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.memory.sample_buffer(self.batch_size)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.critic.device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.critic.device)
        actions = torch.tensor(actions, dtype=torch.float32, device=self.critic.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.critic.device)
        states = torch.tensor(states, dtype=torch.float32, device=self.critic.device)

        self.critic.eval()
        self.critic_target.eval()
        self.actor_target.eval()

        target_actions = self.actor_target(next_states)
        critic_target_value = self.critic_target(next_states, target_actions)
        critic_value = self.critic(states, actions)

        target = rewards.reshape(self.batch_size, 1) + self.gamma * critic_target_value * dones.reshape(self.batch_size, 1)

        self.critic.train()
        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target, critic_value)
        for param in self.critic.parameters():
            critic_loss += 0.5 * torch.norm(param, p=2) ** 2  # L2 regularization term
        critic_loss.backward()
        self.critic.optimizer.step()

        self.critic.eval()
        self.actor.optimizer.zero_grad()
        mu = self.actor(states)
        self.actor.train()
        actor_loss = -self.critic(states, mu).mean()
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        self.soft_update(self.critic_target, self.critic, tau)
        self.soft_update(self.actor_target, self.actor, tau)

    def save_models(self):
        print("Saving Models...")
        self.actor.save_checkpoint()
        self.actor_target.save_checkpoint()
        self.critic.save_checkpoint()
        self.critic_target.save_checkpoint()

    def load_models(self):
        print("Loading Models...")
        self.actor.load_checkpoint()
        self.actor_target.load_checkpoint()
        self.critic.load_checkpoint()
        self.critic_target.load_checkpoint()

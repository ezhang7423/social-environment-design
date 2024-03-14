import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.network = nn.Sequential(
            self.layer_init(nn.Conv2d(envs.single_observation_space.shape[2], 32, 8, stride=4)),
            nn.ReLU(),
            self.layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            self.layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            self.layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.actor = self.layer_init(nn.Linear(512, envs.single_action_space.n), std=0.01)
        self.critic = self.layer_init(nn.Linear(512, 1), std=1)

    def get_value(self, x):
        return self.critic(self.network(x.permute((0, 3, 1, 2))))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x.permute((0, 3, 1, 2)))
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)

    def layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer


class PrincipalAgent(nn.Module):
    def __init__(self, num_agents):
        super().__init__()
        self.conv_net = nn.Sequential(
            self.layer_init(nn.Conv2d(3, 32, 8, stride=4)),
            nn.ReLU(),
            self.layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            self.layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            self.layer_init(nn.Linear(64 * 14 * 20, 512)),
            nn.ReLU(),
        )
        self.fully_connected = nn.Sequential(
            self.layer_init(nn.Linear(512 + num_agents, 512)),
            nn.ReLU(),
            self.layer_init(nn.Linear(512, 512)),
            nn.ReLU(),
        )

        self.actor_head1 = self.layer_init(nn.Linear(512, 12), std=0.01)
        self.actor_head2 = self.layer_init(nn.Linear(512, 12), std=0.01)
        self.actor_head3 = self.layer_init(nn.Linear(512, 12), std=0.01)
        self.critic = self.layer_init(nn.Linear(512, 1), std=1)

    def get_value(self, world_obs, cumulative_reward):
        world_obs = world_obs.clone()
        world_obs /= 255.0
        conv_out = self.conv_net(world_obs.permute((0, 3, 1, 2)))
        with_rewards = torch.cat(
            (conv_out, cumulative_reward), dim=1
        )  # shape num_games x (512+num_agents)
        hidden = self.fully_connected(with_rewards)
        return self.critic(hidden)

    def get_action_and_value(self, world_obs, cumulative_reward, action=None):
        world_obs = world_obs.clone()
        world_obs /= 255.0
        conv_out = self.conv_net(world_obs.permute((0, 3, 1, 2)))
        with_rewards = torch.cat(
            (conv_out, cumulative_reward), dim=1
        )  # shape num_games x (512+num_agents)
        hidden = self.fully_connected(with_rewards)
        logits1 = self.actor_head1(hidden)
        logits2 = self.actor_head2(hidden)
        logits3 = self.actor_head3(hidden)
        probs1 = Categorical(logits=logits1)
        probs2 = Categorical(logits=logits2)
        probs3 = Categorical(logits=logits3)
        if action is None:
            action = torch.stack([probs1.sample(), probs2.sample(), probs3.sample()], dim=1)
        log_prob = (
            probs1.log_prob(action[:, 0])
            + probs2.log_prob(action[:, 1])
            + probs3.log_prob(action[:, 2])
        )
        entropy = probs1.entropy() + probs2.entropy() + probs3.entropy()
        return action, log_prob, entropy, self.critic(hidden)

    def layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

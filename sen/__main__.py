import dataclasses
from dataclasses import dataclass

import typer
from rich import print
from typer_config.decorators import dump_json_config, use_json_config

from sen import setup_experiment

setup_experiment()
import argparse
import copy
import os
import random
import shutil
import time
import warnings
from dataclasses import dataclass
from distutils.util import strtobool

import gymnasium as gym
import numpy as np
import supersuit as ss
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import typer
from eztils import datestr
from eztils.typer import dataclass_option
from meltingpot import substrate
from torch.utils.tensorboard import SummaryWriter
from typer_config.decorators import dump_json_config, use_json_config

from sen import LOG_DIR, huggingface_upload, utils, version
from sen.neural.agent_architectures import Agent, PrincipalAgent
from sen.principal import Principal
from sen.principal.utils import vote
from sen.vector_constructors import pettingzoo_env_to_vec_env_v1, sb3_concat_vec_envs_v1

app = typer.Typer(
    name="sen",
    help="project_tag",
    add_completion=False,
)


@dataclass
class Config:
    exp_name: str = "apple_picking_game"  # the name of this experiment
    seed: int = 1  # seed of the experiment
    torch_deterministic: bool = True  # if toggled, `torch.backends.cudnn.deterministic=False`
    cuda: bool = True  # if toggled, cuda will be enabled by default
    track: bool = False  # if toggled, this experiment will be tracked with Weights and Biases
    wandb_project_name: str = "apple-picking-game"  # the wandb's project name
    wandb_entity: str = None  # the entity (team) of wandb's project
    capture_video: bool = True  # whether to capture videos of the agent performances
    video_freq: int = 20  # capture video every how many episodes?
    save_model: bool = True  # whether to save model parameters
    save_model_freq: int = 100  # save model parameters every how many episodes?
    learning_rate: float = 2.5e-4  # the learning rate of the optimizer
    adam_eps: float = 1e-5  # eps value for the optimizer
    num_parallel_games: int = 1  # the number of parallel game environments
    num_frames: int = 4  # the number of game frames to stack together
    num_episodes: int = 100000  # the number of steps in an episode
    episode_length: int = 1000  # the number of steps in an episode
    tax_annealment_proportion: float = (
        0.02  # proportion of episodes over which to linearly anneal tax cap multiplier
    )
    sampling_horizon: int = 200  # the number of timesteps between policy update iterations
    tax_period: int = 50  # the number of timesteps tax periods last (at end of period tax vals updated and taxes applied)
    anneal_tax: bool = True  # Toggle tax cap annealing over an initial proportion of episodes
    anneal_lr: bool = True  # Toggle learning rate annealing for policy and value networks
    gamma: float = 0.99  # the discount factor gamma
    gae_lambda: float = 0.95  # the lambda for the general advantage estimation
    minibatch_size: int = 128  # size of minibatches when training policy network
    update_epochs: int = 4  # the K epochs to update the policy
    norm_adv: bool = True  # Toggles advantages normalization
    clip_coef: float = 0.2  # the surrogate clipping coefficient
    clip_vloss: bool = True  # Toggles whether or not to use a clipped loss for the value function, as per the paper.
    ent_coef: float = 0.01  # coefficient of the entropy
    vf_coef: float = 0.5  # coefficient of the value function
    max_grad_norm: float = 0.5  # the maximum norm for the gradient clipping
    target_kl: float = None  # the target KL divergence threshold


@app.command(name="")
@use_json_config()
@dump_json_config(str(LOG_DIR / "config.json"))
def main(
    conf: dataclass_option(Config) = "{}",  # type: ignore
) -> None:
    args: Config = conf  # for type hinting
    run_name = f"apple_picking__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print("device:", device)

    env_name = "commons_harvest__open"
    env_config = substrate.get_config(env_name)

    num_players = len(env_config.default_player_roles)
    principal = Principal(num_players, args.num_parallel_games, "egalitarian")

    env = utils.parallel_env(
        max_cycles=args.sampling_horizon, env_config=env_config, principal=principal
    )
    num_agents = env.max_num_agents
    num_envs = args.num_parallel_games * num_agents
    env.render_mode = "rgb_array"

    env = ss.observation_lambda_v0(env, lambda x, _: x["RGB"], lambda s: s["RGB"])
    env = ss.frame_stack_v1(env, args.num_frames)
    env = ss.agent_indicator_v0(env, type_only=False)
    env = pettingzoo_env_to_vec_env_v1(env)
    envs = sb3_concat_vec_envs_v1(  # need our own as need reset to pass up world obs and nearby in info
        env, num_vec_envs=args.num_parallel_games
    )

    envs.single_observation_space = envs.observation_space
    envs.single_action_space = envs.action_space
    envs.is_vector_env = True
    assert isinstance(
        envs.single_action_space, gym.spaces.Discrete
    ), "only discrete action space is supported"

    voting_values = np.random.uniform(size=[num_agents])
    selfishness = np.random.uniform(size=[num_agents])
    trust = np.random.uniform(size=[num_agents])

    agent = Agent(envs).to(device)
    principal_agent = PrincipalAgent(num_agents).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=args.adam_eps)
    principal_optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=args.adam_eps)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.sampling_horizon, num_envs) + envs.single_observation_space.shape).to(
        device
    )
    actions = torch.zeros((args.sampling_horizon, num_envs) + envs.single_action_space.shape).to(
        device
    )
    logprobs = torch.zeros((args.sampling_horizon, num_envs)).to(device)
    rewards = torch.zeros((args.sampling_horizon, num_envs)).to(device)
    dones = torch.zeros((args.sampling_horizon, num_envs)).to(device)
    values = torch.zeros((args.sampling_horizon, num_envs)).to(device)

    principal_obs = torch.zeros(
        (args.sampling_horizon, args.num_parallel_games) + (144, 192, 3)
    ).to(device)
    cumulative_rewards = torch.zeros(
        (args.sampling_horizon, args.num_parallel_games, num_agents)
    ).to(device)
    principal_actions = torch.zeros((args.sampling_horizon, args.num_parallel_games, 3)).to(device)
    principal_logprobs = torch.zeros((args.sampling_horizon, args.num_parallel_games)).to(device)
    principal_rewards = torch.zeros((args.sampling_horizon, args.num_parallel_games)).to(device)
    principal_dones = torch.zeros((args.sampling_horizon, args.num_parallel_games)).to(device)
    principal_values = torch.zeros((args.sampling_horizon, args.num_parallel_games)).to(device)

    next_obs = torch.Tensor(envs.reset()).to(device)
    next_done = torch.zeros(num_envs).to(device)
    next_cumulative_reward = torch.zeros(args.num_parallel_games, num_agents).to(device)

    principal_next_obs = torch.stack(
        [torch.Tensor(envs.reset_infos[i][1]) for i in range(0, num_envs, num_agents)]
    ).to(device)
    principal_next_done = torch.zeros(args.num_parallel_games).to(device)

    num_policy_updates_per_ep = args.episode_length // args.sampling_horizon
    num_policy_updates_total = args.num_episodes * num_policy_updates_per_ep
    num_updates_for_this_ep = 0
    current_episode = 1
    episode_step = 0
    episode_rewards = torch.zeros(num_envs).to(device)
    principal_episode_rewards = torch.zeros(args.num_parallel_games).to(device)
    start_time = time.time()

    prev_objective_val = 0
    tax_values = []
    tax_frac = 1

    # fill this with sampling horizon chunks for recording if needed
    episode_world_obs = [0] * (args.episode_length // args.sampling_horizon)

    """
    load a pre-trained agent if needed:
    """
    # warnings.warn("loading pretrained agents")
    # agent.load_state_dict(torch.load("./model9399.pth"))

    for update in range(1, num_policy_updates_total + 1):
        # annealing the rate if instructed to do so
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_policy_updates_total
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # annealing tax controlling multiplier
        if args.anneal_tax:
            tax_frac = 0.1 + 0.9 * min(
                (current_episode - 1.0) / (args.num_episodes * args.tax_annealment_proportion),
                1,
            )

        # collect data for policy update
        start_step = episode_step
        for step in range(0, args.sampling_horizon):
            if next_obs.shape[3] != 19:
                warnings.warn(
                    "hardcoded value of 12 RGB channels - check RBG/indicator channel division here"
                )
            num_rgb_channels = 12
            # we only divide the 4 stack frames x 3 RGB channels - NOT the agent indicators
            next_obs[:, :, :, :num_rgb_channels] /= 255.0
            obs[step] = next_obs
            dones[step] = next_done
            principal_obs[step] = principal_next_obs
            principal_dones[step] = principal_next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            with torch.no_grad():
                (
                    principal_action,
                    principal_logprob,
                    _,
                    principal_value,
                ) = principal_agent.get_action_and_value(principal_next_obs, next_cumulative_reward)
                principal_values[step] = principal_value.flatten()

            if episode_step % args.tax_period == 0:
                # this `principal_action` is the one that was fed cumulative reward of last step of previous tax period
                # so it is acting on the full wealth accumulated last tax period and an observation of the last frame
                principal_actions[step] = principal_action
                principal_logprobs[step] = principal_logprob
                principal.update_tax_vals(principal_action)
                tax_values.append(copy.deepcopy(principal.tax_vals))
            else:
                principal_actions[step] = torch.tensor([11] * 3)
                principal_actions[step] = torch.full((args.num_parallel_games, 3), 11)
                principal_logprobs[step] = torch.zeros(args.num_parallel_games)

            """
            NOTE: info has been changed to return a list of entries for each
                  environment (over num_agents and num_parallel_games), with
                  each entry being a tuple of the old info dict (asfaik always
                  empty until last step when it gets a 'terminal_observation'),
                  the world observation numpy array and the nearby player array.
                  IMPORTANT:
                  info is a list of environment, not agents.
                  If you are playing 2 simultaneous games of seven players, info
                  will be a list of length 14. In this, the first seven entries
                  will have the same info[i][1] world observation, and so will the
                  next seven - but the two will differ between each other.
            """
            next_obs, extrinsic_reward, done, info = envs.step(action.cpu().numpy())
            principal.report_reward(extrinsic_reward)

            # mix personal and nearby rewards
            intrinsic_reward = np.zeros_like(extrinsic_reward)
            nearby = torch.stack([torch.Tensor(info[i][2]) for i in range(0, num_envs)]).to(device)
            for game_id in range(args.num_parallel_games):
                game_reward = extrinsic_reward[game_id * num_agents : (game_id + 1) * num_agents]
                for player_id in range(num_agents):
                    env_id = player_id + game_id * num_agents
                    w = selfishness[player_id]
                    #! TODO fix nearby_reward = sum(nearby[env_id] * game_reward)
                    nearby_reward = 0
                    intrinsic_reward[env_id] = (
                        w * extrinsic_reward[env_id] + (1 - w) * nearby_reward
                    )

            # make sure tax is applied after extrinsic reward is used for intrinsic reward calculation
            if (episode_step + 1) % args.tax_period == 0:
                # last step of tax period
                taxes = principal.end_of_tax_period()
                extrinsic_reward -= tax_frac * np.array(list(taxes.values())).flatten()

            reward = np.zeros_like(extrinsic_reward)
            for env_id in range(len(reward)):
                player_id = env_id % num_agents
                v = trust[player_id]
                reward[env_id] = v * extrinsic_reward[env_id] + (1 - v) * intrinsic_reward[env_id]

            principal_next_obs = torch.stack(
                [torch.Tensor(info[i][1]) for i in range(0, num_envs, num_agents)]
            ).to(device)
            principal_reward = principal.objective(reward) - prev_objective_val
            prev_objective_val = principal.objective(reward)
            principal_next_done = torch.zeros(args.num_parallel_games).to(
                device
            )  # for now saying principal never done

            prev_cumulative_reward = (
                torch.zeros(args.num_parallel_games, num_agents)
                if (episode_step % args.tax_period) == 0
                else cumulative_rewards[step - 1]
            )
            next_cumulative_reward = prev_cumulative_reward.to(device) + torch.tensor(
                extrinsic_reward
            ).to(device).view(
                -1, num_agents
            )  # split reward into dimensions by game
            next_cumulative_reward = next_cumulative_reward.to(device)
            cumulative_rewards[step] = next_cumulative_reward.to(device)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

            principal_rewards[step] = torch.tensor(principal_reward).to(device).view(-1)

            episode_step += 1

        principal_episode_rewards += torch.sum(principal_rewards, 0)
        episode_rewards += torch.sum(rewards, 0)
        end_step = episode_step - 1

        episode_world_obs[num_updates_for_this_ep - 1] = principal_obs[:, 0, :, :, :].clone()
        if args.save_model and current_episode % args.save_model_freq == 0:
            try:
                os.mkdir(f"./saved_params_{run_name}")
            except FileExistsError:
                pass
            try:
                os.mkdir(f"./saved_params_{run_name}/ep{current_episode}")
            except FileExistsError:
                pass
            torch.save(
                obs,
                f"./saved_params_{run_name}/ep{current_episode}/obs_samplerun{num_updates_for_this_ep}_ep{current_episode}.pt",
            )
            torch.save(
                actions,
                f"./saved_params_{run_name}/ep{current_episode}/actions_samplerun{num_updates_for_this_ep}_ep{current_episode}.pt",
            )
            torch.save(
                logprobs,
                f"./saved_params_{run_name}/ep{current_episode}/logprobs_samplerun{num_updates_for_this_ep}_ep{current_episode}.pt",
            )
            torch.save(
                rewards,
                f"./saved_params_{run_name}/ep{current_episode}/rewards_samplerun{num_updates_for_this_ep}_ep{current_episode}.pt",
            )
            torch.save(
                dones,
                f"./saved_params_{run_name}/ep{current_episode}/dones_samplerun{num_updates_for_this_ep}_ep{current_episode}.pt",
            )
            torch.save(
                values,
                f"./saved_params_{run_name}/ep{current_episode}/values_samplerun{num_updates_for_this_ep}_ep{current_episode}.pt",
            )

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.sampling_horizon)):
                if t == args.sampling_horizon - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values

        # bootstrap principal value if not done
        with torch.no_grad():
            principal_next_value = principal_agent.get_value(
                principal_next_obs, next_cumulative_reward
            ).reshape(1, -1)
            principal_advantages = torch.zeros_like(principal_rewards).to(device)
            principal_lastgaelam = 0
            for t in reversed(range(args.sampling_horizon)):
                if t == args.sampling_horizon - 1:
                    principal_nextnonterminal = 1.0 - principal_next_done
                    principal_nextvalues = principal_next_value
                else:
                    principal_nextnonterminal = 1.0 - principal_dones[t + 1]
                    principal_nextvalues = principal_values[t + 1]
                principal_delta = (
                    principal_rewards[t]
                    + args.gamma * principal_nextvalues * principal_nextnonterminal
                    - principal_values[t]
                )
                principal_advantages[t] = principal_lastgaelam = (
                    principal_delta
                    + args.gamma
                    * args.gae_lambda
                    * principal_nextnonterminal
                    * principal_lastgaelam
                )
            principal_returns = principal_advantages + principal_values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the agent policy and value network
        b_inds = np.arange(len(b_obs))
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, len(b_obs), args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions.long()[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        # flatten batch for principal
        principal_b_obs = principal_obs.reshape((-1,) + (144, 192, 3))
        principal_b_logprobs = principal_logprobs.reshape(-1)
        b_cumulative_rewards = cumulative_rewards.reshape(
            -1, num_agents
        )  # from sampling_horizon x num_games x num_agents to (sampling_horizon*num_games) x num_agents
        principal_b_actions = principal_actions.reshape((-1, 3))
        principal_b_advantages = principal_advantages.reshape(-1)
        principal_b_returns = principal_returns.reshape(-1)
        principal_b_values = principal_values.reshape(-1)

        # Optimizing the principal policy and value network
        b_inds = np.arange(len(principal_b_obs))
        principal_clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(
                0, len(principal_b_obs), args.minibatch_size // num_agents
            ):  # principal has batch size num_games not num_envs(=num_games*num_agents) so divide to ensure same number of minibatches as agents
                end = start + args.minibatch_size // num_agents
                mb_inds = b_inds[start:end]

                (
                    _,
                    principal_newlogprob,
                    principal_entropy,
                    principal_newvalue,
                ) = principal_agent.get_action_and_value(
                    principal_b_obs[mb_inds],
                    b_cumulative_rewards[mb_inds],
                    principal_b_actions.long()[mb_inds],
                )
                principal_logratio = principal_newlogprob - principal_b_logprobs[mb_inds]
                principal_ratio = principal_logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    principal_old_approx_kl = (-principal_logratio).mean()
                    principal_approx_kl = ((principal_ratio - 1) - principal_logratio).mean()
                    principal_clipfracs += [
                        ((principal_ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    ]

                principal_mb_advantages = principal_b_advantages[mb_inds]
                if args.norm_adv:
                    principal_mb_advantages = (
                        principal_mb_advantages - principal_mb_advantages.mean()
                    ) / (principal_mb_advantages.std() + 1e-8)

                # Policy loss
                principal_pg_loss1 = -principal_mb_advantages * principal_ratio
                principal_pg_loss2 = -principal_mb_advantages * torch.clamp(
                    principal_ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                principal_pg_loss = torch.max(principal_pg_loss1, principal_pg_loss2).mean()

                # Value loss
                principal_newvalue = principal_newvalue.view(-1)
                if args.clip_vloss:
                    principal_v_loss_unclipped = (
                        principal_newvalue - principal_b_returns[mb_inds]
                    ) ** 2
                    principal_v_clipped = principal_b_values[mb_inds] + torch.clamp(
                        principal_newvalue - principal_b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    principal_v_loss_clipped = (
                        principal_v_clipped - principal_b_returns[mb_inds]
                    ) ** 2
                    principal_v_loss_max = torch.max(
                        principal_v_loss_unclipped, principal_v_loss_clipped
                    )
                    principal_v_loss = 0.5 * principal_v_loss_max.mean()
                else:
                    principal_v_loss = (
                        0.5 * ((principal_newvalue - principal_b_returns[mb_inds]) ** 2).mean()
                    )

                principal_entropy_loss = principal_entropy.mean()
                principal_loss = (
                    principal_pg_loss
                    - args.ent_coef * principal_entropy_loss
                    + principal_v_loss * args.vf_coef
                )

                principal_optimizer.zero_grad()
                principal_loss.backward()
                nn.utils.clip_grad_norm_(principal_agent.parameters(), args.max_grad_norm)
                principal_optimizer.step()

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        principal_y_pred, principal_y_true = (
            principal_b_values.cpu().numpy(),
            principal_b_returns.cpu().numpy(),
        )
        principal_var_y = np.var(principal_y_true)
        principal_explained_var = (
            np.nan
            if principal_var_y == 0
            else 1 - np.var(principal_y_true - principal_y_pred) / principal_var_y
        )

        # one more policy update done
        num_updates_for_this_ep += 1
        print(
            f"Completed policy update {num_updates_for_this_ep} for episode {current_episode} - used steps {start_step} through {end_step}"
        )

        if num_updates_for_this_ep == num_policy_updates_per_ep:
            # episode finished

            if args.capture_video and current_episode % args.video_freq == 0:
                # currently only records first of any parallel games running but
                # this is easily changed at the point where we add to episode_world_obs
                video = torch.cat(episode_world_obs, dim=0).cpu()
                try:
                    os.mkdir(f"./videos_{run_name}")
                except FileExistsError:
                    pass
                torchvision.io.write_video(
                    f"./videos_{run_name}/episode_{current_episode}.mp4", video, fps=20
                )
                huggingface_upload.upload(f"./videos_{run_name}", run_name)
                if args.track:
                    wandb.log(
                        {"video": wandb.Video(f"./videos_{run_name}/episode_{current_episode}.mp4")}
                    )
                os.remove(f"./videos_{run_name}/episode_{current_episode}.mp4")

            writer.add_scalar(
                "charts/learning_rate", optimizer.param_groups[0]["lr"], current_episode
            )
            writer.add_scalar("losses/value_loss", v_loss.item(), current_episode)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), current_episode)
            writer.add_scalar("losses/entropy", entropy_loss.item(), current_episode)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), current_episode)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), current_episode)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), current_episode)
            writer.add_scalar("losses/explained_variance", explained_var, current_episode)
            writer.add_scalar(
                "charts/mean_episodic_return",
                torch.mean(episode_rewards),
                current_episode,
            )
            writer.add_scalar("charts/episode", current_episode, current_episode)
            writer.add_scalar("charts/tax_frac", tax_frac, current_episode)
            mean_rewards_across_envs = {player_idx: 0 for player_idx in range(0, num_agents)}
            for idx in range(len(episode_rewards)):
                mean_rewards_across_envs[idx % num_agents] += episode_rewards[idx].item()
            mean_rewards_across_envs = list(
                map(
                    lambda x: x / args.num_parallel_games,
                    mean_rewards_across_envs.values(),
                )
            )

            for player_idx in range(num_agents):
                writer.add_scalar(
                    f"charts/episodic_return-player{player_idx}",
                    mean_rewards_across_envs[player_idx],
                    current_episode,
                )
            print(
                f"Finished episode {current_episode}, with {num_policy_updates_per_ep} policy updates"
            )
            print(f"Mean episodic return: {torch.mean(episode_rewards)}")
            print(f"Episode returns: {mean_rewards_across_envs}")
            print(f"Principal returns: {principal_episode_rewards.tolist()}")
            for game_id in range(args.num_parallel_games):
                writer.add_scalar(
                    f"charts/principal_return_game{game_id}",
                    principal_episode_rewards[game_id].item(),
                    current_episode,
                )
                for tax_period in range(len(tax_values)):
                    tax_step = (
                        current_episode - 1
                    ) * args.episode_length // args.tax_period + tax_period
                    for bracket in range(0, 3):
                        writer.add_scalar(
                            f"charts/tax_value_game{game_id}_bracket_{bracket+1}",
                            np.array(tax_values[tax_period][f"game_{game_id}"][bracket]),
                            tax_step,
                        )

            print(
                f"Tax values this episode (for each period): {tax_values}, capped by multiplier {tax_frac}"
            )
            print("*******************************")

            if args.save_model and current_episode % args.save_model_freq == 0:
                try:
                    os.mkdir(f"./models_{run_name}")
                except FileExistsError:
                    pass
                torch.save(
                    agent.state_dict(),
                    f"./models_{run_name}/agent_{current_episode}.pth",
                )
                torch.save(
                    principal_agent.state_dict(),
                    f"./models_{run_name}/principal_{current_episode}.pth",
                )
                huggingface_upload.upload(f"./models_{run_name}", run_name)
                os.remove(f"./models_{run_name}/agent_{current_episode}.pth")
                os.remove(f"./models_{run_name}/principal_{current_episode}.pth")

                huggingface_upload.upload(f"./saved_params_{run_name}", run_name)
                shutil.rmtree(f"./saved_params_{run_name}/ep{current_episode}")
                print("model saved")

            # vote on principal objective
            principal_objective = vote(voting_values)
            principal.set_objective(principal_objective)

            # start a new episode:
            next_obs = torch.Tensor(envs.reset()).to(device)
            next_done = torch.zeros(num_envs).to(device)
            # no need to reset obs,actions,logprobs,etc as they have length args.sampling_horizon so will be overwritten

            current_episode += 1
            num_updates_for_this_ep = 0
            episode_step = 0
            prev_objective_val = 0
            episode_rewards = torch.zeros(num_envs).to(device)
            principal_episode_rewards = torch.zeros(args.num_parallel_games).to(device)
            tax_values = []

    envs.close()
    writer.close()


app()

"""Evaluate a saved PPO agent on a given env config.

Usage:
    python eval.py --agent-path runs/<run_name>/agent.pt \
                   --goal-room random --yellow-room off \
                   --num-episodes 100
"""
from dataclasses import dataclass

import custom_envs  # noqa: F401  (registers SpuriousFourRooms-v0)
import gymnasium as gym
import minigrid  # noqa: F401
import numpy as np
import torch
import tyro
from minigrid.wrappers import FlatObsWrapper

from ppo import Agent, _parse_goal_room, _parse_yellow_room


@dataclass
class Args:
    agent_path: str
    """path to the saved agent .pt file"""
    env_id: str = "SpuriousFourRooms-v0"
    """env to evaluate in"""
    goal_room: str = "random"
    """goal placement: 0/1/2/3 or 'random'"""
    yellow_room: str = "off"
    """yellow placement: 0/1/2/3, 'follow', 'random', or 'off'"""
    num_episodes: int = 100
    """number of evaluation episodes"""
    seed: int = 0
    """base seed; episode i uses seed+i"""
    deterministic: bool = False
    """if True, take argmax of policy logits instead of sampling"""
    cuda: bool = True


def make_eval_env(env_id, goal_room, yellow_room):
    extra = {}
    if env_id.startswith("SpuriousFourRooms"):
        extra["goal_room"] = _parse_goal_room(goal_room)
        extra["yellow_room"] = _parse_yellow_room(yellow_room)
    env = gym.make(env_id, **extra)
    env = FlatObsWrapper(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    return env


if __name__ == "__main__":
    args = tyro.cli(Args)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Agent's __init__ uses envs.single_observation_space / single_action_space.
    # Wrap one env in SyncVectorEnv so those attributes exist.
    dummy_vec = gym.vector.SyncVectorEnv(
        [lambda: make_eval_env(args.env_id, args.goal_room, args.yellow_room)]
    )
    agent = Agent(dummy_vec).to(device)
    agent.load_state_dict(torch.load(args.agent_path, map_location=device))
    agent.eval()
    dummy_vec.close()

    env = make_eval_env(args.env_id, args.goal_room, args.yellow_room)
    returns, lengths = [], []
    for ep in range(args.num_episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        ep_return, ep_length, done = 0.0, 0, False
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                if args.deterministic:
                    action = int(agent.actor(obs_t).argmax(dim=-1).item())
                else:
                    a, _, _, _ = agent.get_action_and_value(obs_t)
                    action = int(a.item())
            obs, reward, term, trunc, _ = env.step(action)
            ep_return += float(reward)
            ep_length += 1
            done = term or trunc
        returns.append(ep_return)
        lengths.append(ep_length)
    env.close()

    returns = np.array(returns)
    lengths = np.array(lengths)
    success = float((returns > 0).mean())

    print()
    print(f"Agent:        {args.agent_path}")
    print(f"Eval env:     {args.env_id}  goal_room={args.goal_room}  yellow_room={args.yellow_room}")
    print(f"Episodes:     {args.num_episodes}  (base_seed={args.seed}, deterministic={args.deterministic})")
    print(f"Success rate: {success:.3f}  ({int(success * args.num_episodes)}/{args.num_episodes})")
    print(f"Mean return:  {returns.mean():.4f}  (std {returns.std():.4f})")
    print(f"Mean length:  {lengths.mean():.1f}  (std {lengths.std():.1f})")

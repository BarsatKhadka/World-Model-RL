"""Method 2 — The Intervention Test.

Mid-episode, forcibly teleport the goal to a new cell, then measure
how the agent responds.

Two protocol variants (see plan.md):
  --teleport-mode visible    -> new goal lands in agent's current 7x7 view
  --teleport-mode invisible  -> new goal lands far from agent, outside view

For each valid episode we record:
  - reached_new_goal:        did the agent reach the teleported goal at all?
  - steps_to_reach:          how many post-teleport steps until reward
  - min_dist_to_new_goal:    closest the agent got to the new goal
  - dist_drop_first_5_steps: change in distance to new goal in the
                             first 5 steps after teleport. Negative
                             values = moving toward the new goal.

The fastest distinguishing signal between causal and non-causal agents
is `dist_drop_first_5_steps` in --teleport-mode visible. A causal agent
sees the new goal and immediately heads toward it (large negative drop).
A trajectory-memorizer or wall-follower continues its prior plan
(drop near zero or even positive).

Usage:
    python intervention.py \
        --agent-path runs/cnn__grandom_yrandom__cond_D__1__.../agent.pt \
        --policy cnn \
        --teleport-mode visible \
        --num-episodes 200
"""
from dataclasses import dataclass

import custom_envs  # noqa: F401  (registers SpuriousFourRooms-v0)
import gymnasium as gym
import minigrid  # noqa: F401
import numpy as np
import torch
import tyro
from minigrid.core.world_object import Goal
from minigrid.wrappers import FlatObsWrapper, ImgObsWrapper

from ppo import build_agent, _parse_goal_room, _parse_yellow_room


@dataclass
class Args:
    agent_path: str
    """path to the trained agent .pt file"""
    policy: str = "mlp"
    """policy architecture used at training time: 'mlp' or 'cnn'"""
    teleport_mode: str = "visible"
    """'visible' (goal moves into agent's view) or 'invisible' (moves out of view)"""
    intervention_step: int = 30
    """step within the episode at which to teleport the goal"""
    num_episodes: int = 200
    """target number of valid intervention episodes to collect"""
    max_steps: int = 200
    """hard upper bound on steps per episode"""
    seed: int = 0
    """base seed for RNG"""
    # Env config to construct the eval distribution. Defaults to held-out env.
    env_id: str = "SpuriousFourRooms-v0"
    goal_room: str = "random"
    yellow_room: str = "off"
    cuda: bool = True


def make_eval_env(env_id, goal_room, yellow_room, policy):
    extra = {}
    if env_id.startswith("SpuriousFourRooms"):
        extra["goal_room"] = _parse_goal_room(goal_room)
        extra["yellow_room"] = _parse_yellow_room(yellow_room)
    env = gym.make(env_id, **extra)
    env = ImgObsWrapper(env) if policy == "cnn" else FlatObsWrapper(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    return env


def find_goal_pos(env):
    """Scan the grid for the (single) Goal object."""
    base = env.unwrapped
    for x in range(base.width):
        for y in range(base.height):
            cell = base.grid.get(x, y)
            if cell is not None and cell.type == "goal":
                return (x, y)
    return None


def teleport_goal(env, mode, rng, min_invisible_dist=5):
    """Move the goal to a new empty cell consistent with `mode`.

    Returns (old_pos, new_pos), or (old_pos, None) if no valid cell
    could be found (in which case the episode should be discarded).
    """
    base = env.unwrapped
    old_pos = find_goal_pos(env)
    ax, ay = int(base.agent_pos[0]), int(base.agent_pos[1])

    candidates = []
    for x in range(1, base.width - 1):
        for y in range(1, base.height - 1):
            if (x, y) == (ax, ay) or (x, y) == old_pos:
                continue
            cell = base.grid.get(x, y)
            # accept empty cells and yellow Floor tiles (which we'll paint over)
            if cell is not None and cell.type not in ("floor",):
                continue
            in_view = base.in_view(x, y)
            if mode == "visible" and not in_view:
                continue
            if mode == "invisible":
                if in_view:
                    continue
                if abs(x - ax) + abs(y - ay) < min_invisible_dist:
                    continue
            candidates.append((x, y))

    if not candidates:
        return old_pos, None

    new_pos = candidates[rng.integers(len(candidates))]
    if old_pos is not None:
        base.grid.set(old_pos[0], old_pos[1], None)
    base.grid.set(new_pos[0], new_pos[1], Goal())
    return old_pos, new_pos


def manhattan(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def run_intervention_episode(env, agent, device, args, rng):
    base = env.unwrapped
    obs, _ = env.reset(seed=int(rng.integers(0, 1_000_000)))

    teleport_happened = False
    new_goal_pos = None
    distances = []          # distance to new goal at each post-teleport step
    post_teleport_steps = 0
    reached_new_goal = False

    for step in range(args.max_steps):
        # Perform the intervention right before this step's action.
        if step == args.intervention_step and not teleport_happened:
            _, new_pos = teleport_goal(env, args.teleport_mode, rng)
            if new_pos is None:
                return None  # discard
            new_goal_pos = new_pos
            teleport_happened = True
            # Refresh the agent's observation to reflect the modified grid.
            obs_dict = base.gen_obs()
            if args.policy == "cnn":
                obs = obs_dict["image"]
            else:
                # FlatObsWrapper flattens (image + direction + mission encoding).
                # Easiest: re-apply the wrapper's observation transform.
                obs = env.observation(obs_dict)

        obs_t = torch.tensor(np.asarray(obs), dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            a, _, _, _ = agent.get_action_and_value(obs_t)
            action = int(a.item())

        obs, reward, term, trunc, _ = env.step(action)

        if teleport_happened:
            post_teleport_steps += 1
            distances.append(manhattan(tuple(int(c) for c in base.agent_pos), new_goal_pos))
            if reward > 0:
                reached_new_goal = True
                break

        if term or trunc:
            if not teleport_happened:
                return None  # episode ended before we could intervene
            break

    if not distances:
        return None

    # First 5 post-teleport steps: did the agent move toward or away from the new goal?
    n_early = min(5, len(distances))
    drop_first_5 = distances[n_early - 1] - distances[0]   # negative = closer

    return {
        "reached_new_goal": reached_new_goal,
        "post_teleport_steps": post_teleport_steps,
        "initial_dist": distances[0],
        "min_dist": min(distances),
        "final_dist": distances[-1],
        "drop_first_5_steps": drop_first_5,
    }


if __name__ == "__main__":
    args = tyro.cli(Args)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    rng = np.random.default_rng(args.seed)

    dummy_vec = gym.vector.SyncVectorEnv(
        [lambda: make_eval_env(args.env_id, args.goal_room, args.yellow_room, args.policy)]
    )
    agent = build_agent(args.policy, dummy_vec).to(device)
    agent.load_state_dict(torch.load(args.agent_path, map_location=device))
    agent.eval()
    dummy_vec.close()

    env = make_eval_env(args.env_id, args.goal_room, args.yellow_room, args.policy)
    results = []
    skipped = 0
    max_attempts = args.num_episodes * 3
    while len(results) < args.num_episodes and (len(results) + skipped) < max_attempts:
        r = run_intervention_episode(env, agent, device, args, rng)
        if r is None:
            skipped += 1
        else:
            results.append(r)
    env.close()

    if not results:
        print("No valid intervention episodes collected.")
        raise SystemExit(1)

    reached = np.array([r["reached_new_goal"] for r in results])
    steps_when_reached = np.array([r["post_teleport_steps"] for r in results if r["reached_new_goal"]])
    init_dists = np.array([r["initial_dist"] for r in results])
    min_dists = np.array([r["min_dist"] for r in results])
    drops = np.array([r["drop_first_5_steps"] for r in results])

    print()
    print(f"Agent:               {args.agent_path}")
    print(f"Policy:              {args.policy}")
    print(f"Eval distribution:   goal_room={args.goal_room}  yellow_room={args.yellow_room}")
    print(f"Teleport mode:       {args.teleport_mode}")
    print(f"Intervention step:   {args.intervention_step}")
    print(f"Valid episodes:      {len(results)}  (skipped {skipped})")
    print()
    print(f"Reach-new-goal rate:                 {reached.mean():.3f}  ({int(reached.sum())}/{len(reached)})")
    if len(steps_when_reached):
        print(f"Steps to reach (when reached):       mean={steps_when_reached.mean():.1f}  median={np.median(steps_when_reached):.1f}")
    print(f"Initial dist to new goal:            mean={init_dists.mean():.2f}  median={np.median(init_dists):.1f}")
    print(f"Closest approach (min dist after):   mean={min_dists.mean():.2f}  median={np.median(min_dists):.1f}")
    print(f"Distance drop in first 5 steps:      mean={drops.mean():+.2f}  median={np.median(drops):+.1f}")
    print("  (negative = moved toward new goal,  positive = moved away,  ~0 = no response)")

"""FourRooms with a yellow-floor spurious signal.

Room indexing (used by `goal_room` and `yellow_room`):

    +-------+-------+
    |   0   |   1   |     0 = top-left,  1 = top-right
    +-------+-------+
    |   2   |   3   |     2 = bot-left,  3 = bot-right
    +-------+-------+

Training conditions (all expressible as configs of this one class):

    Baselines (no spurious signal at all — used to anchor the comparison):
        0a: SpuriousFourRooms(goal_room=1, yellow_room="off")
            Fixed goal, no yellow. Tests pure location memorization.
        0b: SpuriousFourRooms(goal_room="random", yellow_room="off")
            Varying goal, no yellow. The causal-only ceiling for Condition D.

    The 2x2 spurious-signal matrix from plan.md:
        A:  SpuriousFourRooms(goal_room=1, yellow_room="follow")
            Both shortcuts available (location + yellow). Ambiguous baseline.
        B:  SpuriousFourRooms(goal_room=1, yellow_room="random")
            Location shortcut only — yellow is noise.
        C:  SpuriousFourRooms(goal_room="random", yellow_room="follow")
            Yellow shortcut only — location is useless.
        D:  SpuriousFourRooms(goal_room="random", yellow_room="random")
            No shortcuts. Maximum pressure toward causal learning.

    Held-out test env (used by Method 1 to probe trained agents — note this
    is identical to Baseline-0b; that's intentional):
        SpuriousFourRooms(goal_room="random", yellow_room="off")
"""

from __future__ import annotations

from typing import Literal, Union

from minigrid.core.world_object import Floor
from minigrid.envs import FourRoomsEnv

RoomChoice = Union[int, Literal["random"]]
YellowMode = Union[int, Literal["follow", "random", "off"]]


class SpuriousFourRooms(FourRoomsEnv):
    def __init__(
        self,
        goal_room: RoomChoice = "random",
        yellow_room: YellowMode = "follow",
        max_steps: int = 100,
        **kwargs,
    ):
        self.goal_room = goal_room
        self.yellow_room = yellow_room
        self._chosen_goal_room: int | None = None
        self._chosen_yellow_room: int | None = None
        super().__init__(agent_pos=None, goal_pos=None, max_steps=max_steps, **kwargs)

    def _room_bounds(self, room: int) -> tuple[int, int, int, int]:
        """Inclusive interior bounds (x_min, x_max, y_min, y_max) of a room."""
        room_w = self.width // 2
        room_h = self.height // 2
        col = room % 2       # 0 = left,   1 = right
        row = room // 2      # 0 = top,    1 = bottom
        x_min = col * room_w + 1
        x_max = (col + 1) * room_w - 1
        y_min = row * room_h + 1
        y_max = (row + 1) * room_h - 1
        return x_min, x_max, y_min, y_max

    def _resolve_goal_room(self) -> int:
        if isinstance(self.goal_room, int):
            return self.goal_room
        return int(self._rand_int(0, 4))

    def _resolve_yellow_room(self, goal_room: int) -> int | None:
        mode = self.yellow_room
        if mode == "off":
            return None
        if mode == "follow":
            return goal_room
        if mode == "random":
            return int(self._rand_int(0, 4))
        return int(mode)

    def _gen_grid(self, width, height):
        # Decide goal placement BEFORE calling super so FourRoomsEnv uses it.
        goal_room = self._resolve_goal_room()
        gx_min, gx_max, gy_min, gy_max = self._room_bounds(goal_room)
        gx = int(self._rand_int(gx_min, gx_max + 1))
        gy = int(self._rand_int(gy_min, gy_max + 1))
        self._goal_default_pos = (gx, gy)
        self._chosen_goal_room = goal_room

        # Decide yellow room for this episode.
        self._chosen_yellow_room = self._resolve_yellow_room(goal_room)

        # Let FourRoomsEnv build the walls, doors, agent, and goal.
        super()._gen_grid(width, height)

        # Paint the yellow room. Skip walls and the goal cell.
        if self._chosen_yellow_room is not None:
            yx_min, yx_max, yy_min, yy_max = self._room_bounds(self._chosen_yellow_room)
            for x in range(yx_min, yx_max + 1):
                for y in range(yy_min, yy_max + 1):
                    if self.grid.get(x, y) is None:
                        self.grid.set(x, y, Floor("yellow"))


if __name__ == "__main__":
    # Quick verification: instantiate each condition, dump a rendered PNG,
    # and print the chosen rooms across a few resets.
    import os

    from PIL import Image

    out_dir = os.path.join(os.path.dirname(__file__), "_verify")
    os.makedirs(out_dir, exist_ok=True)

    configs = {
        "0a_baseline_fixed":    dict(goal_room=1, yellow_room="off"),
        "0b_baseline_random":   dict(goal_room="random", yellow_room="off"),
        "A_fixed_goal_follow":  dict(goal_room=1, yellow_room="follow"),
        "B_fixed_goal_random":  dict(goal_room=1, yellow_room="random"),
        "C_random_goal_follow": dict(goal_room="random", yellow_room="follow"),
        "D_random_goal_random": dict(goal_room="random", yellow_room="random"),
        "test_held_out":        dict(goal_room="random", yellow_room="off"),
    }

    for name, cfg in configs.items():
        env = SpuriousFourRooms(render_mode="rgb_array", **cfg)
        print(f"\n[{name}] cfg={cfg}")
        for ep in range(5):
            env.reset(seed=ep)
            print(f"  ep={ep}  goal_room={env._chosen_goal_room}  "
                  f"yellow_room={env._chosen_yellow_room}  "
                  f"goal_pos={env._goal_default_pos}")
            if ep == 0:
                frame = env.render()
                Image.fromarray(frame).save(os.path.join(out_dir, f"{name}.png"))
        env.close()

    print(f"\nWrote PNGs to {out_dir}")

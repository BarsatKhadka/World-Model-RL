from gymnasium.envs.registration import register

from custom_envs.spurious_fourrooms import SpuriousFourRooms

register(
    id="SpuriousFourRooms-v0",
    entry_point="custom_envs.spurious_fourrooms:SpuriousFourRooms",
)

__all__ = ["SpuriousFourRooms"]

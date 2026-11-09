#!/user/bin/env python3

from elements import create_fire, create_water
from alchemy.elements import create_air, create_earth


def healing_potion() -> str:
    string: str = "Healing potion brewed with '"
    string = string + create_earth() + "' and '"
    string = string + create_air() + "'"
    return string


def strength_potion() -> str:
    string: str = "Strength potion brewed with '"
    string = string + create_fire() + "' and '"
    string = string + create_water() + "'"
    return string

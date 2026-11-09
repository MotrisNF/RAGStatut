from elements import create_fire as fire
from alchemy.potions import strength_potion as potion
from ..elements import create_air as air


def lead_to_gold() -> str:
    string: str = "Recipe transmuting Lead to Gold: brew '"
    string = string + air() + "' and '" + potion() + "' mixed with '"
    string = string + fire() + "'"
    return string

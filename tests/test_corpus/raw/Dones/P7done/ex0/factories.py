from .creatures import Creature, Flameling, Pyrodon, Aquabub, Torragon
from abc import ABC, abstractmethod


class CreatureFactory(ABC):
    @abstractmethod
    def create_base(self) -> Creature:
        pass

    @abstractmethod
    def create_evolved(self) -> Creature:
        pass

    def __repr__(self) -> str:
        return str(self)


class FlameFactory(CreatureFactory):
    def create_base(self) -> Flameling:
        return Flameling("Flameling", "Fire")

    def create_evolved(self) -> Pyrodon:
        return Pyrodon("Pyrodon", "Fire/Flying")

    def __str__(self) -> str:
        return "Flame"


class AquaFactory(CreatureFactory):
    def create_base(self) -> Aquabub:
        return Aquabub("Aquabub", "Water")

    def create_evolved(self) -> Torragon:
        return Torragon("Torragon", "Water")

    def __str__(self) -> str:
        return "Aqua"

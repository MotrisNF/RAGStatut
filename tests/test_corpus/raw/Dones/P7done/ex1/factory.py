from ex0 import CreatureFactory
from .creatures import Shiftling, Morphagon
from .creatures import Sproutling, Bloomelle


class HealingCreatureFactory(CreatureFactory):
    def create_base(self) -> Sproutling:
        return Sproutling("Sproutling", "Grass")

    def create_evolved(self) -> Bloomelle:
        return Bloomelle("Bloomelle", "Grass/Fairy")

    def __str__(self) -> str:
        return "Healing"


class TransformCreatureFactory(CreatureFactory):
    def create_base(self) -> Shiftling:
        return Shiftling("Shiftling", "Normal")

    def create_evolved(self) -> Morphagon:
        return Morphagon("Morphagon", "Normal/Dragon")

    def __str__(self) -> str:
        return "Transform"

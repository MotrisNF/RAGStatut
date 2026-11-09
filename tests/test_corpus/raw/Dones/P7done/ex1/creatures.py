from .capabilities import HealCapability, TransformCapability
from ex0 import creatures


class Sproutling(creatures.Creature, HealCapability):
    def attack(self) -> str:
        return str(self.name) + " uses Vine Whip"

    def heal(self, target: str = "None") -> str:
        string: str
        if target == "None":
            string = self.name + " heals it self for a small amount"
        else:
            string = self.name + " heals " + target + " for a small amount"

        return string

    def __str__(self) -> str:
        return "Sproutling"


class Bloomelle(creatures.Creature, HealCapability):
    def attack(self) -> str:
        return str(self.name) + " uses Petal Dance"

    def heal(self, target: str = "None") -> str:
        string: str
        if target == "None":
            string = self.name + " heals it self for a large amount"
        else:
            string = self.name + " heals " + target + " for a large amount"

        return string

    def __str__(self) -> str:
        return "Bloomelle"


class Shiftling(creatures.Creature, TransformCapability):
    def transform(self) -> str:
        if not self.transformed:
            self.transformed = True
            return self.name + " shifts into a sharper form!"
        else:
            return self.name + " can't transform"

    def revert(self) -> str:
        if self.transformed is True:
            self.transformed = False
            return self.name + " returns to normal"
        else:
            return self.name + " can't revert"

    def attack(self) -> str:
        string: str = self.name
        if self.transformed is True:
            string = string + " perform boosted strike!"
        else:
            string = string + " attacks normally."
        return string

    def __str__(self) -> str:
        return "Shiftling"


class Morphagon(creatures.Creature, TransformCapability):
    def transform(self) -> str:
        if not self.transformed:
            self.transformed = True
            return self.name + " shifts into a dragonic battle form!"
        else:
            return self.name + " can't transform."

    def revert(self) -> str:
        if self.transformed is True:
            self.transformed = False
            return self.name + " returns to normal."
        else:
            return self.name + " can't revert."

    def attack(self) -> str:
        string: str = self.name
        if self.transformed is True:
            string = string + " unleashes a devastating morph strike!"
        else:
            string = string + " attacks normally."
        return string

    def __str__(self) -> str:
        return "Morphagon"

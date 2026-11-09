from abc import ABC, abstractmethod


class Creature(ABC):
    def __init__(self, name: str, type: str):
        super().__init__()
        self.name: str = name
        self.type: str = type

    @abstractmethod
    def attack(self) -> str:
        pass

    def describe(self) -> str:
        return (f"{self.name} is a {self.type} type Creature")

    def __repr__(self) -> str:
        return str(self)


class Flameling(Creature):
    def attack(self) -> str:
        return str(self.name) + " uses Ember"

    def __str__(self) -> str:
        return "Flameling"


class Pyrodon(Creature):
    def attack(self) -> str:
        return str(self.name) + " uses Flamethrower"

    def __str__(self) -> str:
        return "Pyrodon"


class Aquabub(Creature):
    def attack(self) -> str:
        return str(self.name) + " uses Water Gun"

    def __str__(self) -> str:
        return "Aquabub"


class Torragon(Creature):
    def attack(self) -> str:
        return str(self.name) + " uses Hydro Pump"

    def __str__(self) -> str:
        return "Torragon"

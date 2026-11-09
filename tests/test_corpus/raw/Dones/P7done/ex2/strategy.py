from abc import ABC, abstractmethod
from ex0.creatures import Creature
from ex1.capabilities import TransformCapability, HealCapability


class CreatureError(Exception):
    pass


class BattleStrategy(ABC):
    @abstractmethod
    def is_valid(self, creature: Creature) -> bool:
        pass

    @abstractmethod
    def act(self, creature: Creature) -> None:
        pass

    def __repr__(self) -> str:
        return str(self)


class NormalStrategy(BattleStrategy):
    def is_valid(self, creature: Creature) -> bool:
        if creature is not None:
            return True
        return False

    def act(self, creature: Creature) -> None:
        print(creature.attack())

    def __str__(self) -> str:
        return "Normal"


class AggresiveStrategy(BattleStrategy):
    def is_valid(self, creature: Creature | TransformCapability) -> bool:
        return isinstance(creature, TransformCapability)

    def act(self, creature: Creature) -> None:
        if self.is_valid(creature):
            if isinstance(creature, TransformCapability):
                print(creature.transform())
            print(creature.attack())
            if isinstance(creature, TransformCapability):
                print(creature.revert())
        else:
            raise CreatureError(
                f"Invalid creature '{creature.name}'"
                " for this aggressive strategy"
                )

    def __str__(self) -> str:
        return "Aggresive"


class DefensiveStrategy(BattleStrategy):
    def is_valid(self, creature: Creature | HealCapability) -> bool:
        return isinstance(creature, HealCapability)

    def act(self,
            creature: Creature,
            target: str = "None"
            ) -> None:
        if self.is_valid(creature):
            print(creature.attack())
            if isinstance(creature, HealCapability):
                print(creature.heal(target))
        else:
            raise CreatureError(
                f"Invalid creature '{creature.name}'"
                " for this defensive strategy"
                )

    def __str__(self) -> str:
        return "Defensive"

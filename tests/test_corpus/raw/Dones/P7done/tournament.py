from ex2 import BattleStrategy, NormalStrategy
from ex2 import AggresiveStrategy, DefensiveStrategy
from ex2 import CreatureError
from ex1 import HealingCreatureFactory, TransformCreatureFactory
from ex0 import AquaFactory, FlameFactory
from ex0.factories import CreatureFactory


def battle(opponents: list[tuple[CreatureFactory, BattleStrategy]]) -> None:
    i: int = 1
    for creature in opponents:
        actual_creature = creature[0].create_base()
        for enemy in range(i, len(opponents)):
            enemy_factory = opponents[enemy]
            actual_enemy = enemy_factory[0].create_base()
            print("* Battle *")
            print(actual_creature.describe())
            print("vs.")
            print(actual_enemy.describe())
            print("now figth!")
            try:
                creature[1].act(actual_creature)
            except CreatureError as e:
                print(f"Battle error, aborting tournament {e}")
                break
            try:
                enemy_factory[1].act(actual_enemy)
            except CreatureError as e:
                print(f"Battle error, aborting tournament {e}")
                break
            print()
        i += 1


if __name__ == "__main__":
    aqua_factory = AquaFactory()
    flame_factory = FlameFactory()
    healing_factory = HealingCreatureFactory()
    transform_factory = TransformCreatureFactory()
    normal = NormalStrategy()
    aggresive = AggresiveStrategy()
    defensive = DefensiveStrategy()

    battle1: list[tuple[CreatureFactory, BattleStrategy]] = [
        (flame_factory, normal),
        (healing_factory, defensive)
    ]
    print("Tournamet 0 (basic)")
    print(battle1)
    print("*** Tournamet ***\n2 opponents involved\n")
    battle(battle1)

    battle2: list[tuple[CreatureFactory, BattleStrategy]] = [
        (flame_factory, aggresive),
        (aqua_factory, defensive)
    ]
    print("Tournament 1 (error)")
    print(battle2)
    print("*** Tournament ***\n2 opponents unvolved\n")
    battle(battle2)

    battle3: list[tuple[CreatureFactory, BattleStrategy]] = [
        (aqua_factory, normal),
        (healing_factory, defensive),
        (transform_factory, aggresive)
    ]
    print("Tournament 2 (error)")
    print(battle3)
    print("*** Tournament ***\n3 opponents unvolved\n")
    battle(battle3)

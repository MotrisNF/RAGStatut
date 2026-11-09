import ex0


def test_battle(
        factory1: ex0.CreatureFactory,
        factory2: ex0.CreatureFactory
        ) -> None:
    print("Testing battle")
    flame = factory1.create_base()
    water = factory2.create_base()
    print(flame.describe())
    print("vs")
    print(water.describe())
    print("figth!")
    print(flame.attack())
    print(water.attack())


def test_factory(factory: ex0.CreatureFactory) -> None:
    print("Testing factory")
    base_creature = factory.create_base()
    print(base_creature.describe())
    print(base_creature.attack())
    evol_creature = factory.create_evolved()
    print(evol_creature.describe())
    print(evol_creature.attack())
    print()


if __name__ == "__main__":
    flame_factory = ex0.FlameFactory()
    aqua_factory = ex0.AquaFactory()
    test_factory(flame_factory)
    test_factory(aqua_factory)
    test_battle(flame_factory, aqua_factory)

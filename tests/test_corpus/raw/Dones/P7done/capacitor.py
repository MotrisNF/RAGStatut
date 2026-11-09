from ex1 import HealingCreatureFactory, TransformCreatureFactory


def test_heal(factory: HealingCreatureFactory) -> None:
    base = factory.create_base()
    evol = factory.create_evolved()
    print("Testing Creature with Healing capability")
    print("base:")
    print(base.describe())
    print(base.attack())
    print(base.heal())
    print("evolved:")
    print(evol.describe())
    print(evol.attack())
    print(evol.heal(base.name))
    print()


def test_trans(factory: TransformCreatureFactory) -> None:
    base = factory.create_base()
    evol = factory.create_evolved()
    print("Testing Creature whit trabsform capability")
    print("base:")
    print(base.describe())
    print(base.attack())
    print(base.transform())
    print(base.attack())
    print(base.revert())
    print("evol:")
    print(evol.describe())
    print(evol.attack())
    print(evol.transform())
    print(evol.attack())
    print(evol.revert())


if __name__ == "__main__":
    heal_factory: HealingCreatureFactory = HealingCreatureFactory()
    trans_factory: TransformCreatureFactory = TransformCreatureFactory()
    test_heal(heal_factory)
    test_trans(trans_factory)

import operator
import functools
from typing import Callable, Any


def spell_reducer(spells: list[int], operation: str) -> Any:
    if spells == []:
        return 0
    operations = {
        "add": operator.add,
        "multiply": operator.mul,
        "max": lambda a, b: max(a, b),
        "min": lambda a, b: min(a, b)
        }
    try:
        reduce = functools.reduce(operations[operation], spells)
    except KeyError:
        print(f"Format error: unknown operation '{operation}'")
        return 0
    except TypeError as e:
        print(f"Format error: {e}")
        return 0
    return reduce


def partial_enchanter(
        base_enchantment: Callable[..., str]
        ) -> dict[str, Callable[..., str]]:
    return {
        "fire": functools.partial(base_enchantment, power=50, element="fire"),
        "ice": functools.partial(base_enchantment, power=50, element="ice"),
        "lightning": functools.partial(
            base_enchantment, power=50, element="lightning"
            ),
        }


@functools.lru_cache
def memoized_fibonacci(n: int) -> int:
    if n == 1:
        return 1
    if n <= 0:
        return 0
    return memoized_fibonacci(n - 1) + memoized_fibonacci(n - 2)


def spell_dispatcher() -> Callable[[Any], str]:
    @functools.singledispatch
    def cast(spell: Any) -> str:
        return "Unknown spell type"

    @cast.register
    def _(spell: int) -> str:
        return f"Damage spell: {spell} damage"

    @cast.register
    def _(spell: str) -> str:
        return f"Enchantment: {spell}"

    @cast.register(list)
    def _(spell: list[Any]) -> str:
        return f"Multi-cast: {len(spell)} spells"

    return cast


if __name__ == "__main__":
    print("Testing spell reducer...")
    spell_powers = [42, 38, 39, 14, 48, 41]
    print("Sum:", spell_reducer(spell_powers, "add"))
    print("Product:", spell_reducer(spell_powers, "multiply"))
    print("Max:", spell_reducer(spell_powers, "max"))
    print("Unknown op:", spell_reducer(spell_powers, "unknown_op"))

    print("Testing partial enchanter...")

    def enchant(power: int, element: str, target: str) -> str:
        return (
            f"{element.capitalize()} enchantment "
            f"({power} power) on {target}"
            )

    enchanters = partial_enchanter(enchant)
    print(enchanters["fire"](target="Sword"))
    print(enchanters["ice"](target="Shield"))
    print(enchanters["lightning"](target="Staff"))

    print("Testing memoized fibonacci...")
    # print(memoized_fibonacci.cache_info())
    print("Fib(0):", memoized_fibonacci(0))
    print("Fib(1):", memoized_fibonacci(1))
    print("Fib(10):", memoized_fibonacci(10))
    print("Fib(15):", memoized_fibonacci(15))
    # print(memoized_fibonacci.cache_info())
    print("Testing spell dispatcher...")
    print(spell_dispatcher()(42))
    print(spell_dispatcher()("fireball"))
    print(spell_dispatcher()([1, 2, 3]))
    print(spell_dispatcher()({"key": 42}))

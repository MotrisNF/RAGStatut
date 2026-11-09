from typing import Callable, Any


def mage_counter() -> Callable[[], int]:
    total: int = 0

    def grow() -> int:
        nonlocal total
        total += 1
        return total

    return grow


def spell_accumulator(initial_power: int) -> Callable[[int], int]:
    power: int = initial_power

    def grow(amount: int) -> int:
        nonlocal power
        power += amount
        return power

    return grow


def enchantment_factory(enchantment_type: str) -> Callable[..., Any]:
    def fusion(item: str) -> str:
        return enchantment_type + " " + item
    return fusion


def memory_vault() -> dict[str, Callable[..., Any]]:
    storage: dict[str, str] = {}

    def store(key: str, value: str) -> None:
        storage[key] = value

    def recall(
            key: str
            ) -> Any:
        retuned = storage.get(key, "Memory not found")
        return retuned

    return {
        "store": store,
        "recall": recall
    }


if __name__ == "__main__":
    print("Testing mage counter...")
    counter_a = mage_counter()
    print("counter_a call 1:", counter_a())
    print("counter_a call 2:", counter_a())
    counter_b = mage_counter()
    print("counter_b call 1:", counter_b())
    print()
    print("Testing spell acumulator...")
    base: int = 100
    acumulator = spell_accumulator(base)
    print(f"Base {base}, add 20: {acumulator(20)}")
    print(f"Base {base}, add 30: {acumulator(30)}")
    print()
    print("Testing enchantment factory...")
    flaming = enchantment_factory("Flaming")
    frozen = enchantment_factory("Frozen")
    print(flaming("Sword"))
    print(frozen("Shield"))
    print()
    print("Testing memory vault...")
    vault = memory_vault()
    print("Store 'secret' = 42")
    vault["store"]("secret", "42")
    print("Recall 'secret':", vault["recall"]("secret"))
    print("Recall 'unknown':", vault["recall"]("unknown"))

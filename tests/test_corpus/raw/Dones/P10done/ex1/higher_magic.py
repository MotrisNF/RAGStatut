from typing import Callable


def heal(target: str, power: int) -> str:
    return f"Heal restores {target} for {power} HP"


def fireball(target: str, power: int) -> str:
    return f"Fireball make to {target} {power} of damage"


def spell_combiner(
        spell1: Callable[[str, int], str],
        spell2: Callable[[str, int], str]
        ) -> Callable[[str, int], tuple[str, str]]:
    if not callable(spell1) or not callable(spell2):
        text: str = "The funtion 'spell_combiner' need two callable arguments"
        raise TypeError(text)

    def temp_function(target: str, power: int) -> tuple[str, str]:
        return (spell1(target, power), spell2(target, power))
    return temp_function


def power_amplifier(
        base_spell: Callable[[str, int], str],
        multipliet: int
        ) -> Callable[[str, int], str]:
    if not callable(base_spell):
        text: str = "The funtion 'spell_amplifier' need a callable argument"
        raise TypeError(text)

    def temp_function(target: str, power: int) -> str:
        return (base_spell(target, power * multipliet))
    return temp_function


def conditional_caster(
        condition: Callable[[str, int], bool],
        spell: Callable[[str, int], str]
        ) -> Callable[[str, int], str]:
    if not callable(condition) or not callable(spell):
        text: str = (
            "The funtion 'conditional_caster' need two callable arguments"
            )
        raise TypeError(text)

    def temp_function(target: str, power: int) -> str:
        if condition(target, power):
            return spell(target, power)
        return "Spell fizzled"
    return temp_function


def spell_sequence(
        spells: list[Callable[[str, int], str]]
        ) -> Callable[[str, int], list[str]]:
    def execute_sequence(target: str, power: int) -> list[str]:
        result_list: list[str] = []

        for spell in spells:
            result_list.append(spell(target, power))
        return result_list

    return execute_sequence


if __name__ == "__main__":
    print()
    test_values = [14, 8, 14]
    test_targets = [
        'Dragon',
        'Goblin',
        'Wizard',
        'Knight'
    ]

    print("Testing spell combiner...")
    combined = spell_combiner(fireball, heal)
    res_fb, res_hl = combined(test_targets[0], test_values[0])

    print(
        f"Combined spell result: Fireball hits {test_targets[0]},"
        f"Heals {test_targets[0]}"
        )

    print("Testing power amplifier...")
    base_power = 10
    multiplier = 3

    amplified_spell = power_amplifier(fireball, multiplier)

    res_amplified_str = amplified_spell("Target", base_power)
    amplified_power = [
        int(s) for s in res_amplified_str.split() if s.isdigit()
        ][0]

    print(f"Original: {base_power}, Amplified: {amplified_power}")

    print("\nTesting conditional caster...")

    def is_wizard(target: str, power: int) -> bool:
        return target == 'Wizard'

    caster = conditional_caster(is_wizard, fireball)

    print(f"Condition met: {caster(test_targets[2], test_values[2])}")
    print(f"Condition failed: {caster(test_targets[1], test_values[1])}")

    print("\nTesting spell sequence...")
    spells_list: list[Callable[[str, int], str]] = [fireball, heal]
    sequence = spell_sequence(spells_list)

    sequence_results = sequence(test_targets[3], test_values[0])
    print(f"Sequence results for {test_targets[3]}:")
    for result in sequence_results:
        print(f"  - {result}")

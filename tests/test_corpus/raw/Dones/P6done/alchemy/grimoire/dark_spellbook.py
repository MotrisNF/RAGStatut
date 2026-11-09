from alchemy.grimoire.dark_validator import validate_dark_ingredients


def dark_spell_allowed_ingredients() -> list[str]:
    return ["bats", "frogs", "arsenic", "eyeball"]


def dark_spell_record(spell_name: str, ingredients: str) -> str:
    string: str = validate_dark_ingredients(ingredients)
    if "VALID" in string:
        return "Spell recorded: " + spell_name + string
    else:
        return "Spell not recorded: " + spell_name + string

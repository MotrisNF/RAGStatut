def light_spell_allowed_ingredients() -> list[str]:
    return ["earth", "air", "fire", "water"]


def light_spell_record(spell_name: str, ingredients: str) -> str:
    from alchemy.grimoire.light_validator import validate_ingredients
    string: str = validate_ingredients(ingredients)
    if "VALID" in string:
        return "Spell recorded: " + spell_name + string
    else:
        return "Spell not recorded: " + spell_name + string

from alchemy.grimoire.light_spellbook import light_spell_allowed_ingredients


def validate_ingredients(ingredient: str) -> str:
    ingredients: list[str] = light_spell_allowed_ingredients()
    ingredient_input: list[str] = ingredient.lower().split()
    valid: bool = False
    for word in ingredient_input:
        if word.strip(",") in ingredients:
            valid = True
            break
    if valid:
        return " (" + ingredient + " - VALID)"
    else:
        return " (" + ingredient + " - INVALID)"

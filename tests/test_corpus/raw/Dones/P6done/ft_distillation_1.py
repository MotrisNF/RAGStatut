import alchemy

if __name__ == "__main__":
    print("=== Distillation 1 ===")
    print("Using: 'import alchemy' structure to access potions")
    print("Testing strength_potion: ", end="")
    print(alchemy.strength_potion())
    print("Testing heal alias: ", end="")
    print(alchemy.heal())

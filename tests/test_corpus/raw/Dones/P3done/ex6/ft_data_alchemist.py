#!/usr/bin/env python3

import random


def main() -> None:
    all_names: list[str] = [
        "Alice", "bob", "Charlie", "dylan", "Emma",
        "Gregory", "john", "kevin", "Liam"
        ]
    print(f"Initial list of players: {all_names}")
    capitalized: list[str] = [names.capitalize() for names in all_names]
    print(f"New list with all names capitalized: {capitalized}")
    only_cap: list[
        str] = [name for name in all_names if name == name.capitalize()]
    print(f"New list of capitalized names: {only_cap}\n")
    scr_dict: dict[
        str,
        int
        ] = {name: random.randint(1, 1000) for name in capitalized}
    print(f"Score dict: {scr_dict}")
    average: float = float(sum(scr_dict.values()) / len(scr_dict))
    print(f"Score average: {round(average, 2)}")
    max_score: dict[str, int] = {
        name: scr_dict[name] for name in scr_dict if scr_dict[name] > average
        }
    print(f"High scores: {max_score}")


if __name__ == "__main__":
    print("=== Game Data Alchemist ===\n")
    main()

#!usr/bin/env python3

import random

if __name__ == "__main__":
    users: list[str] = ["Alice", "Bob", "Charlie", "saperez-"]
    archievements: list[str] = [
        "Crafting Genius", "Strategist", "World Savior", "Speed Runner",
        "Survivor", "Master Explorer", "Treasure Hunter", "Unstoppable",
        "First Steps", "Collector Supreme", "Untouchable", "Sharp Mind",
        "Boss Slayer", "42 Student!"
    ]
    userset: list[set[str]] = []
    for user in users:
        rand_num = random.randint(1, len(archievements) - 1)
        new_set: set[str] = set(random.sample(archievements, rand_num))
        userset = userset + [new_set]
    i: int = 0
    while i < len(users):
        print(f"Player {users[i]}: {userset[i]}")
        i += 1
    common_data = set.intersection(*userset)
    print()
    print(f"\nAll distinct archievements: {set.union(*userset)}")
    print()
    print(f"\nCommon archievements: {common_data}")
    i = 0
    j: int = 0
    for u in userset:
        temp: set[str] = set()
        j = 0
        while j < len(userset):
            if j != i:
                temp = temp.union(userset[j])
            j += 1
        print(f"Only {users[i]} has: {u.difference(temp)}")
        i += 1
    print()
    i = 0
    archiev_set: set[str] = set()
    archiev_set = archiev_set.union(set(archievements))
    for u_set in userset:
        print(f"{users[i]} is missing: {archiev_set.difference(u_set)}")
        i += 1

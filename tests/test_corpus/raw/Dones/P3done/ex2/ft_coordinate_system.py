#!/usr/bin/env pyton3

import math


def get_cords() -> tuple[float, float, float]:
    while (1):
        valid: int = 1
        cords: tuple[float, float, float]
        entry: str = input("Enter new coordinates  \
as floats in format 'x,y,z': ")
        elements: list[str] = entry.split(",")
        if len(elements) != 3:
            print("Invalid syntax")
            continue

        for i in elements:
            try:
                val_float: float = float(i.strip())
                if val_float:
                    pass
            except ValueError as e:
                print(f"Error on parameter'{i.strip()}': {e}")
                valid = 0
                continue
        if valid:
            cords = (
                float(elements[0]),
                float(elements[1]),
                float(elements[2])
                )
            return cords
    return tuple[None, None, None]


if __name__ == "__main__":
    print("Get a first set of coordinates")
    cords1: tuple[float, float, float] = get_cords()
    print("Got a first tuple:", cords1)
    print(f"It includes: X={cords1[0]}, Y={cords1[1]}, Z={cords1[2]}")
    dist: float = math.sqrt(
        (0 - cords1[0])**2 + (
            0 - cords1[1])**2 + (
                0 - cords1[2])**2
        )
    print(f"Distance to center: {dist:.4f}")

    print("\nGet a second set of coordinates")
    cords2: tuple[float, float, float] = get_cords()
    dist = math.sqrt(
        (cords2[0] - cords1[0])**2 + (
            cords2[1] - cords1[1])**2 + (
                cords2[2] - cords1[2])**2
        )
    print(f"Distance between the 2 sets of coordinates: {dist:.4f}")

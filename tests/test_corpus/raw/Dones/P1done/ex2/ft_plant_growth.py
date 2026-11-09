#!/usr/bin/env python3

class Plant:
    def __init__(self, name: str, height: float, age_d: int, grow_spd: float):
        self.name: str = name.capitalize()
        self.height: float = height
        self.age_d: int = age_d
        self.grow_spd: float = grow_spd

    def show(self) -> None:
        print(f"{self.name}: ", end="")
        print(f"Height: {round(self.height, 1)}cm, ", end="")
        print(f"Age: {self.age_d} days old")

    def age(self) -> None:
        self.age_d += 1

    def grow(self) -> None:
        self.height += self.grow_spd


def main() -> None:
    rose = Plant("rose", 25.0, 30, 0.8)
    rose.show()
    temp_height: float = rose.height
    for i in range(1, 8):
        print(f"=== Day {i} ===")
        rose.grow()
        rose.age()
        rose.show()
    temp_height = rose.height - temp_height
    print(f"Growth this week: {round(temp_height, 1)}cm")


if __name__ == "__main__":
    print("=== Garden Plant Growth ===")
    main()

#!/usr/bin/env python3

class Plant:
    def __init__(
            self, name: str,
            height: float,
            age_d: int,
            grow_spd: float = 0
            ):
        self.name = name.capitalize()
        self.height = height
        self.age_d = age_d
        self.grow_spd = grow_spd
        print("Created: ", end="")
        self.show()

    def show(self) -> None:
        print(f"{self.name}: ", end="")
        print(f"Height: {round(self.height, 1)}cm, ", end="")
        print(f"Age: {self.age_d} days old")

    def age(self) -> None:
        self.age_d += 1

    def grow(self) -> None:
        self.height += self.grow_spd
        self.age()


def main() -> None:
    Plant("rose", 25.0, 30)
    Plant("oak", 200, 365)
    Plant("cactus", 5.0, 90)
    Plant("sunflower", 80.0, 45)
    Plant("fern", 15.0, 120)


if __name__ == "__main__":
    print("=== Plant Factory Output ===")
    main()

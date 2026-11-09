#!/usr/bin/env python3

class Plant:
    def __init__(self, name: str = "None", height: int = 0, age_days: int = 0):
        self.name: str = name.capitalize()
        self.height: int = height
        self.age_days: int = age_days

    def show(self) -> None:
        print(f"{self.name}: ", end="")
        print(f"Height: {self.height}cm, ", end="")
        print(f"Age: {self.age_days} days old")


def main() -> None:
    rose: Plant = Plant()
    rose.name = "Rose"
    rose.height = 25
    rose.age_days = 30
    sunflower: Plant = Plant()
    sunflower.name = "sunflower"
    sunflower.height = 80
    sunflower.age_days = 30
    cactus: Plant = Plant()
    cactus.name = "cactus"
    cactus.height = 10
    cactus.age_days = 120
    rose.show()
    sunflower.show()
    cactus.show()


if __name__ == "__main__":
    print("=== Welcome to My Garden ===")
    main()

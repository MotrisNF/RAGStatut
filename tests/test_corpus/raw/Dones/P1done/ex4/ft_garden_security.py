#!/usr/bin/env python3

class Plant:
    def __init__(
            self,
            name: str,
            height: float,
            age_d: int,
            grow_spd: float
            ):
        self._name = name.capitalize()
        self._height = 0.0
        self._age_d = 0
        self.set_age(age_d)
        self.set_height(height)
        self._grow_spd = grow_spd
        print("Plant created: ", end="")
        self.show()

    def show(self) -> None:
        print(f"{self._name}: ", end="")
        print(f"Height: {round(self._height, 1)}cm, ", end="")
        print(f"Age: {self._age_d} days old")

    def age(self) -> None:
        self._age_d += 1

    def grow(self) -> None:
        self._height += self._grow_spd
        self.age()

    def get_height(self) -> float:
        return self._height

    def set_height(self, new_height: float) -> None:
        if new_height >= 0:
            self._height = new_height
            print(f"Height updated: {self.get_height()}cm")
        else:
            print(f"{self._name}: Error, height can't be negative")
            print("Height update rejected")

    def get_age(self) -> int:
        return self._age_d

    def set_age(self, new_age: int) -> None:
        if new_age >= 0:
            self._age_d = new_age
            print(f"Age updated: {self.get_age()} days old")
        else:
            print(f"{self._name}: Error, age can't be negative")
            print("Age update rejected")


def main() -> None:
    rose: Plant = Plant("Rose", 25.5, -10, 0)
    print("")
    rose.set_height(26)
    rose.set_age(26)
    print("")
    rose.set_height(-1)
    rose.set_age(-1)
    print("")
    print("Current state: ", end="")
    rose.show()


if __name__ == "__main__":
    print("=== Garden Security System ===")
    main()

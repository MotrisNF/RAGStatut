class Plant:
    def __init__(self, name: str, height: float, age_d: int, grow_spd: float):
        self._name = name.capitalize()
        self._height = height
        self._age_d = age_d
        self._grow_spd = grow_spd
        self.show()

    def show(self) -> None:
        print(f"{self._name}: ", end="")
        print(f"Height: {round(self._height, 1)}cm, ", end="")
        print(f"Age: {self._age_d} days old")

    def age(self, days: int = 1) -> None:
        self._age_d += days

    def grow(self, days: int = 1) -> None:
        self._height += self._grow_spd * days
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


class Flower(Plant):
    def __init__(self, name: str, height: float, age_d: int,
                 grow_spd: float, color: str):
        self._color = color
        self._bloomed = False
        print("=== Flower")
        super().__init__(name, height, age_d, grow_spd)

    def show(self) -> None:
        super().show()
        print(f"Color: {self._color}")
        if self._bloomed:
            print("Rose is blooming beautifully!")
        else:
            print("Rose has not bloomed yet")

    def bloom(self) -> None:
        print("[asking the rose to bloom]")
        self._bloomed = True
        self.show()


class Tree(Plant):
    def __init__(self, name: str, height: float, age_d: int,
                 grow_spd: float, diameter: float):
        self.trunk_diameter = diameter
        print("=== Tree")
        super().__init__(name, height, age_d, grow_spd)

    def show(self) -> None:
        super().show()
        print(f"Trunk diameter: {self.trunk_diameter}cm")

    def produce_shade(self) -> None:
        print("[asking the oak to produce shade]")
        print(f"Tree {self._name} produce a shade of ", end="")
        print(f"{self._height}cm long and {self.trunk_diameter}cm wide.")


class Vegetable(Plant):
    def __init__(self, name: str, height: float, age_d: int,
                 grow_spd: float, nutrition: int, season: str):
        self.nutritional_value = nutrition
        self.harvest_season = season
        print("=== Vegetable")
        super().__init__(name, height, age_d, grow_spd)

    def show(self) -> None:
        super().show()
        print(f"Harvest season: {self.harvest_season}")
        print(f"Nutritional value: {self.nutritional_value}")

    def grow(self, days: int = 1) -> None:
        print(f"[make {self._name.lower()} grow and age for {days} days]")
        self._height = self._height + (self._grow_spd * days)
        self.nutritional_value += days
        super().set_age(super().get_age() + days)
        self.show()


def main() -> None:
    rose = Flower("rose", 25.0, 10, 2.1, "red")
    rose.bloom()
    print("")
    oak = Tree("oak", 200.0, 365, 1.0, 5.0)
    oak.produce_shade()
    print("")
    tomato = Vegetable("tomato", 5.0, 10, 2.1, 0, "April")
    tomato.grow(20)


if __name__ == "__main__":
    print("=== Garden Plant Types ===")
    main()

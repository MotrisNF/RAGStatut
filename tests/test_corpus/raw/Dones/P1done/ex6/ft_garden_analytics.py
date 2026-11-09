class Plant:

    class Statistics:
        def __init__(self) -> None:
            self._show: int = 0
            self._age: int = 0
            self._grow: int = 0

        def add_show(self) -> None:
            self._show += 1

        def add_age(self) -> None:
            self._age += 1

        def add_grow(self) -> None:
            self._grow += 1

        def show_statics(self) -> None:
            print(f"Stats: {self._grow} grow, {self._age} age, ", end="")
            print(f"{self._show} show")

    def __init__(self, name: str, height: float, age_d: int, grow_spd: float):
        self._name = name.capitalize()
        self._height = height
        self._age_d = age_d
        self._grow_spd = grow_spd
        self.stats = self.Statistics()

    def show(self) -> None:
        print(f"{self._name}: ", end="")
        print(f"Height: {round(self._height, 1)}cm, ", end="")
        print(f"Age: {self._age_d} days old")
        self.stats.add_show()

    def age(self) -> None:
        self._age_d += 1
        self.stats.add_age()

    def grow(self, days: int = 1) -> None:
        self._height += self._grow_spd * days
        self.stats.add_grow()

    def get_height(self) -> float:
        return self._height

    def set_height(self, new_height: float) -> None:
        if new_height >= 0:
            self._height = new_height
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

    @staticmethod
    def check_days(days: int) -> bool:
        if days > 365:
            print(f"Is {days} days more than a year? -> True")
            return True
        else:
            print(f"Is {days} days more than a year? -> False")
            return False

    @classmethod
    def create_anonymous(cls) -> "Plant":
        return cls("Unknown plant", 0.0, 0, 0)


class Flower(Plant):

    def __init__(self, name: str, height: float, age_d: int,
                 grow_spd: float, color: str):
        self._color = color
        self._bloomed = False
        super().__init__(name, height, age_d, grow_spd)

    def show(self) -> None:
        super().show()
        print(f"Color: {self._color}")
        if self._bloomed:
            print(f"{self._name} is blooming beautifully!")
        else:
            print(f"{self._name} has not bloomed yet")

    def bloom(self) -> None:
        self._bloomed = True


class Seed(Flower):

    def __init__(self, name: str, height: float, age_d: int,
                 grow_spd: float, color: str):
        self._seeds = 0
        super().__init__(name, height, age_d, grow_spd, color)

    def show(self) -> None:
        super().show()
        print(f"Seeds: {self._seeds}")

    def bloom(self) -> None:
        super().bloom()
        self._seeds = 42

    def grow(self, days: int = 1) -> None:
        self.set_height(self._height + self._grow_spd * days)
        super().grow(days)


class Tree(Plant):
    class Statistics(Plant.Statistics):
        def __init__(self) -> None:
            super().__init__()
            self._produce_shade = 0

        def add_shade(self) -> None:
            self._produce_shade += 1

        def show_statics(self) -> None:
            super().show_statics()
            print(f"{self._produce_shade} shade")

    def __init__(self, name: str, height: float, age_d: int,
                 grow_spd: float, diameter: float):
        self._diameter = diameter
        super().__init__(name, height, age_d, grow_spd)
        self.stats: Tree.Statistics = self.stats

    def show(self) -> None:
        super().show()
        print(f"Diameter: {self._diameter}cm")

    def produce_shade(self) -> None:
        print(f"Tree {self._name} now produce a shade of ", end="")
        print(f"{self._height}cm long and {self._diameter}cm wide.")
        self.stats.add_shade()


class Vegetable(Plant):
    def __init__(self, name: str, height: float, age_d: int,
                 grow_spd: float, nutrition: int, season: str):
        self._nutrition = nutrition
        self._season = season
        print("=== Vegetable")
        super().__init__(name, height, age_d, grow_spd)

    def show(self) -> None:
        super().show()
        print(f"Harvest season: {self._season}")
        print(f"Nutritional value: {self._nutrition}")

    def grow(self, days: int = 1) -> None:
        print(f"[make {self._name.lower()} grow and age for {days} days]")
        self._height = self._height + (self._grow_spd * days)
        self._nutrition += days
        super().set_age(super().get_age() + days)
        self.show()


def show_statics(plant: Plant) -> None:
    print(f"[statisticscs for {plant._name.lower()}]")
    plant.stats.show_statics()


def year() -> None:
    Plant.check_days(30)
    Plant.check_days(400)
    print("")


def flower() -> None:
    rose = Flower("rose", 15.0, 10, 5.5, "red")
    rose.show()
    show_statics(rose)
    print("[asking the rose to grow and bloom]")
    rose.grow()
    rose.bloom()
    rose.show()
    show_statics(rose)
    print("")


def tree() -> None:
    oak = Tree("oak", 200.0, 365, 5.5, 5.0)
    oak.show()
    show_statics(oak)
    print("[asking oak to produce shade]")
    oak.produce_shade()
    show_statics(oak)
    print("")


def seed() -> None:
    sun_seed = Seed("sunflower", 80.0, 45, 1.1, "yellow")
    sun_seed.show()
    print(f"[make {sun_seed._name.lower()} grow, age ando bloom]")
    sun_seed.grow()
    sun_seed.age()
    sun_seed.bloom()
    sun_seed.show()
    show_statics(sun_seed)
    print("")


def anonymous() -> None:
    anon: Plant = Plant.create_anonymous()
    anon.show()
    show_statics(anon)


def main() -> None:
    print("=== Check year_old")
    year()
    print("=== Flower")
    flower()
    print("=== Tree")
    tree()
    print("=== Seed")
    seed()
    print("=== Anonymous")
    anonymous()


if __name__ == "__main__":
    print("=== Garden statistics ===")
    main()

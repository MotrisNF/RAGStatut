class GardenError(Exception):
    def __init__(self, msg: str = "Unknown plant error") -> None:
        super().__init__(msg)


class WaterError(GardenError):
    def __init__(self) -> None:
        super().__init__("Not enough water in the tank!")


class PlantError(GardenError):
    def __init__(self, plant: str) -> None:
        super().__init__(f"The {plant} plant is wilting!")


def error_tester(plant_given: str = "Unknown") -> None:
    plant: str = plant_given
    print("\nTesting PlantError...")
    try:
        raise PlantError(plant)
    except PlantError as e:
        print(f"Caught PlantError: {e}")
    print("\nTesting WaterError...")
    try:
        raise WaterError
    except WaterError as e:
        print(f"Caught WaterError: {e}")
    print("\nTesting catching all garden errors...")
    try:
        raise PlantError(plant)
    except GardenError as e:
        print(f"Caught GardenError: {e}")
    try:
        raise WaterError
    except GardenError as e:
        print(f"Caught GardenError: {e}")
    print("\nTesting catching unknown garden errors...")
    try:
        raise GardenError
    except GardenError as e:
        print(f"{e}")


def main() -> None:
    error_tester("Tomato")
    print("\nAll custom error types work correctly!")


if __name__ == "__main__":
    print("=== Custom Garden Errors Demo ===")
    main()

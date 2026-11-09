class GardenError(Exception):
    def __init__(self, msg: str = "Unknown plant error") -> None:
        super().__init__(msg)


class WaterError(GardenError):
    def __init__(self, plant: str) -> None:
        super().__init__(f"Invalid plant name to water: '{plant}'")


def water_plant(plant_name: str = "Unknown") -> None:
    if plant_name == plant_name.capitalize():
        print(f"Waterin {plant_name}: [OK]")
    else:
        raise WaterError(plant_name)


def test_watering_system(*plants: str) -> None:
    print("Opening watering system")
    try:
        for plant in plants:
            water_plant(plant)
    except WaterError as e:
        print("... ending tests and returning to main")
        print(f"Caught PlantError: {e}")
    finally:
        print("Closing watering system")


if __name__ == "__main__":
    print("=== Garden Watering System ===")
    print("\nTesting valid plants...")
    test_watering_system("Tomato", "Lettuce", "Carrots")
    print("\nTesting invalid plants...")
    test_watering_system("Tomato", "lettuce", "potato")
    print("\nCleanup always happens, even with errors")

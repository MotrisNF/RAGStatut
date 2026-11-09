class Temperature_valid_input(Exception):
    def __init__(self, tmp: int):
        if (tmp > 40):
            super().__init__(f"{tmp}°C is too hot for plants (max 40°C)")
        if (tmp < 0):
            super().__init__(f"{tmp}°C is too cold for plants (min 0°C)")


def input_temperature(temp_str: str) -> int:
    result: int = int(temp_str)
    if result > 40 or result < 0:
        raise Temperature_valid_input(result)
    return result


def test_temperature() -> None:
    data: str
    data = "25"
    print(f"\nInput data is '{data}'")
    try:
        input_temperature(data)
        print(f"Temperature is now {data}°C")
    except Exception as e:
        print(f"Caught input_temperature error: {e}")
    data = "abc"
    print(f"\nInput data is '{data}'")
    try:
        input_temperature(data)
        print(f"Temperature is now {data}°C")
    except Exception as e:
        print(f"Caught input_temperature error: {e}")
    data = "100"
    print(f"\nInput data is '{data}'")
    try:
        input_temperature(data)
        print(f"Temperature is now {data}°C")
    except Exception as e:
        print(f"Caught input_temperature error: {e}")
    data = "-50"
    print(f"\nInput data is '{data}'")
    try:
        input_temperature(data)
        print(f"Temperature is now {data}°C")
    except Exception as e:
        print(f"Caught input_temperature error: {e}")
    print("\nAll tests completed - program didn't crash!")


if __name__ == "__main__":
    print("=== Garden Temperature Checker ===")
    test_temperature()

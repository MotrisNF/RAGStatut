def input_temperature(temp_str: str) -> int:
    result: int = int(temp_str)
    return result


def test_temperature() -> None:
    data: str
    data = "25"
    print(f"Input data is '{data}'")
    try:
        input_temperature(data)
        print(f"Temperature is now {data}°C\n")
    except Exception as e:
        print(f"Caught input_temperature error: {e}")
    data = "abc"
    print(f"Input data is '{data}'")
    try:
        input_temperature(data)
        print(f"Temperature is now {data}°C\n")
    except Exception as e:
        print(f"Caught input_temperature error: {e}")

    print("\nAll tests completed - program didn't crash!")


if __name__ == "__main__":
    print("=== Garden Temperature ===\n")
    test_temperature()

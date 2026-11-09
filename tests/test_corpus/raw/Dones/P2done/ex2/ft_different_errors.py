def garden_operation(operation: int) -> None:
    print(f"Testing operation {operation}...")
    try:
        if operation == 0:
            operation = int("abc")
        elif operation == 1:
            10 / 0
        elif operation == 2:
            open("/non/existent/file")
        elif operation == 3:
            operation = "hi!" + 3
        else:
            return
    except ValueError as e:
        print(f"Caught ValueError: {e}")
    except ZeroDivisionError as e:
        print(f"Caught ZeroDivisionError: {e}")
    except FileNotFoundError as e:
        print(f"Caught FileNotFoundError: {e}")
    except TypeError as e:
        print(f"Caught TypeError: {e}")


def test_error_types() -> None:
    print("=== Garden Error Types Demo ===")
    garden_operation(0)
    garden_operation(1)
    garden_operation(2)
    garden_operation(3)


if __name__ == "__main__":
    test_error_types()
    print("\nAll error types tested successfully!")

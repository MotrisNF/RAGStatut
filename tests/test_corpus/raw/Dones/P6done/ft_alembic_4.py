import alchemy

if __name__ == "__main__":
    print("=== Alembic 4 ===")
    print("Accessing the alchemy module using 'import alchemy'")
    print("Testing create_air: ", end="")
    print(alchemy.create_air())
    print("Now show that not all functions can be reached")
    try:
        alchemy.create_earth()  # type: ignore
    except Exception as e:
        print(f"The create_earth function is not imported: {e}")

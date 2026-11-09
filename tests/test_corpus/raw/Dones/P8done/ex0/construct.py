import os
import site
import sys


def is_in_virtual_env() -> bool:
    return sys.prefix != sys.base_prefix


def show_outside_matrix() -> None:
    print()
    print("MATRIX STATUS: You're still plugged in")
    print()
    print(f"Current Python: {sys.executable}")
    print("Virtual Environment: None detected")
    print()
    print("WARNING: You're in the global environment!")
    print("The machines can see everything you install.")
    print()
    print("To enter the construct, run:")
    print("python -m venv matrix_env")
    print("source matrix_env/bin/activate # On Unix")
    print(r"matrix_env\Scripts\activate  # On Windows")
    print()
    print("Then run this program again.")


def show_inside_construct() -> None:
    virtual_env = os.environ.get("VIRTUAL_ENV", sys.prefix)
    print()
    print("MATRIX STATUS: Welcome to the construct")
    print()
    print(f"Current Python: {sys.executable}")
    print(f"Virtual Environment: {os.path.basename(virtual_env)}")
    print(f"Environment Path: {sys.prefix}")
    print()
    print("SUCCESS: You're in an isolated environment!")
    print("Safe to install packages without affecting")
    print("the global system.")
    print()
    print("Package installation path:")
    print(site.getsitepackages()[0])


def main() -> None:
    if is_in_virtual_env():
        show_inside_construct()
    else:
        show_outside_matrix()


if __name__ == "__main__":
    main()

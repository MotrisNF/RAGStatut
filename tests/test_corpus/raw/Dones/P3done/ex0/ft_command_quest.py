#!/usr/bin/env python3
import sys


def main() -> None:
    print(f"Program name: {sys.argv[0]}")
    lenght: int = len(sys.argv)
    if lenght == 1:
        print("No arguments provided!")
    else:
        print(f"Arguments received: {lenght - 1}")
        i: int = 1
        for arg in sys.argv[1:]:
            print(f"Argument {i}: {arg}")
            i += 1
    print(f"Total arguments: {lenght}")


if __name__ == "__main__":
    print("=== Command Quest ===")
    main()

#!/usr/bin/env python3

import sys
from typing import IO


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: ft_ancient_text.py <file>")
        return
    print("=== Cyber Archives Recovery ===")
    fd: IO[str] | None = None
    try:
        print(f"Accessing file '{sys.argv[1]}'")
        fd = open(sys.argv[1], "r")
    except Exception as e:
        print(f"Error opening file '{sys.argv[1]}': {e}")
    if fd:
        print("---\n")
        i: int = 1
        for line in fd:
            print(f"[FRAGMENT {i:03d}] {line.strip()}")
            i += 1
        fd.close()
        print(f"\nFile '{sys.argv[1]}' closed.")
        print("\n---")


if __name__ == "__main__":
    main()

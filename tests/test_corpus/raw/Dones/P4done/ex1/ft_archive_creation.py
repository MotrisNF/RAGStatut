#!/usr/bin/env python3

import sys
from typing import IO


def main() -> None:
    list_fd: list[str] | None = open_file()
    if not list_fd:
        return
    i: int = 1
    print("\nTransform data:\n---\n")
    for t_line in list_fd:
        print(f"[FRAGMENT {i:03d}] {t_line.strip()}")
        i += 1
    print("\n---")
    name_file: str | None = input("Enter new file name (or empty): ")
    if name_file:
        try:
            fd: IO[str] = open(name_file, "wt")
            print(f"Saving data to {name_file}")
            for line in list_fd:
                fd.write(line)
            print(f"Data saved in file {name_file}.")
            fd.close()
        except Exception as e:
            sys.stderr.write(f"{e}\n")
    else:
        print("Not saving data.")


def open_file() -> list[str] | None:
    if len(sys.argv) != 2:
        print("Usage: ft_archive_creation.py <file>")
        return None
    print("=== Cyber Archives Recovery & Preservation ===")
    new_archive: list[str] = []
    fd: IO[str] | None = None
    try:
        print(f"Accessing file '{sys.argv[1]}'")
        fd = open(sys.argv[1], "r")
    except Exception as e:
        sys.stderr.write(f"Error opening file '{sys.argv[1]}': {e}\n")
    if fd:
        print("---\n")
        i: int = 1
        for line in fd:
            new_archive = new_archive + [line.strip() + "#\n"]
            print(f"[FRAGMENT {i:03d}] {line}", end="")
            i += 1
        fd.close()
        print(f"\nFile '{sys.argv[1]}' closed.")
        return new_archive
    return None


if __name__ == "__main__":
    main()

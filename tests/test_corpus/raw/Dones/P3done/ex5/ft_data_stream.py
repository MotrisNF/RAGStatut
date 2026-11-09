#!/usr/bin/env python3

import typing
import random


def gen_event() -> typing.Generator[tuple[str, str], None, None]:
    names: list[str] = ["Sarah", "Esther", "Marco", "Claudio"]
    actions: list[str] = ["run", "walk", "atack", "jump", "idle"]
    while True:
        yield (random.choice(names), random.choice(actions))


def consume_event(
        events: list[tuple[str, str]]
        ) -> typing.Generator[tuple[str, str], None, None]:
    while len(events) > 0:
        element: tuple[str, str] = random.choice(events)
        events.remove(element)
        yield element


def main() -> None:
    event_generator: typing.Generator[
        tuple[str, str],
        None,
        None] = gen_event()
    for i in range(0, 100):
        event: tuple[str, str] = next(event_generator)
        print(f"Event {i + 1}: Player {event[0]} did action {event[1]}")

    erasable: list[tuple[str, str]] = []
    for i in range(0, 10):
        erasable = erasable + [next(event_generator)]

    print(f"Built list of 10 events: {erasable}")
    remove_list: typing.Generator[
        tuple[str, str],
        None,
        None] = consume_event(erasable)
    for i in range(0, 10):
        erase_one: tuple[str, str] = next(remove_list)
        print(f"Got event from list: {erase_one}")
        print(f"Remains in list: {erasable}")


if __name__ == "__main__":
    main()

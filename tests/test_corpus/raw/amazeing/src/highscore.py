"""Persistence and display of the crazy-mode highscore leaderboard."""

from os import path
from src.terminal_io import cooked_input
from typing import Any


def highscore(level: int) -> None:
    """Ask for the player's name and record the reached level.

    Reads "highscore.txt" (created empty if missing), appends the new
    score, re-sorts it by level descending, and writes it back. If the
    player enters an empty name, the score is discarded instead.

    Args:
        level (int): The crazy-mode level reached, saved as the score.
    """
    scores: list[list[Any]] = []
    top_score: bool = False
    if not path.exists("highscore.txt"):
        with open("highscore.txt", 'w'):
            pass
    with open("highscore.txt", 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                scores.append(line.split(sep=": "))
    for score in scores:
        score[1] = int(score[1])
    new_score: str = ""
    new_score = cooked_input("Enter your name: ")
    if new_score == "":
        print("The score is not saved!")
        return
    scores.append([new_score, level])
    scores.sort(key=lambda x: x[1], reverse=True)
    with open("highscore.txt", 'w') as file:
        i: int = 0
        for score in scores:
            print(f"{score[0]}: {score[1]}", file=file)
            if score[0] == new_score and i < 10:
                top_score = True
            i += 1

    if top_score:
        print("You are in the highscore!!")
    else:
        print("Better luck the next time.")


def print_leaderboard() -> None:
    """Print the top 10 entries of "highscore.txt" as a numbered list."""
    i: int = 0
    with open("highscore.txt", 'r') as file:
        for score in file:
            temp: int = i + 1
            print(str(temp) + ":", score, end="")
            i += 1
            if i >= 10:
                break

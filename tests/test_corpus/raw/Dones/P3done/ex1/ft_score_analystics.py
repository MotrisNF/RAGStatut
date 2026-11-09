#!/usr/bin/env python3
import sys


def parse_args() -> list[float]:
    parsed_args: list[float] = []
    i: int = 0
    for arg in sys.argv[1:]:
        try:
            score: float = float(arg)
            parsed_args = parsed_args + [score]
            parsed_args[i] = score
            i += 1
        except Exception:
            print(f"Invalid parameter: {arg}")
    return parsed_args


def main() -> None:

    arg_list: list[float] = parse_args()
    argument_len: int = len(arg_list)

    if argument_len == 1 or not arg_list:
        print("No scores provided. ", end="")
        print("Usage: python3 ft_score_analytics.py", end="")
        print(" <score1> <score2> ..")
        return
    else:
        i: int = 0
        print("Score processed: [", end="")
        for arg in arg_list:
            print(f"{int(arg)}", end="")
            if (i < argument_len - 1):
                print(", ", end="")
                i += 1
            else:
                print("]")
        i = 0
        print(f"Total players: {argument_len}")
        print(f"Total score: {int(sum(arg_list))}")
        print(f"Average score: {sum(arg_list) / argument_len}")
        print(f"High score: {int(max(arg_list))}")
        print(f"Low score: {int(min(arg_list))}")
        print(f"Score range: {int(max(arg_list) - min(arg_list))}")


if __name__ == "__main__":
    main()

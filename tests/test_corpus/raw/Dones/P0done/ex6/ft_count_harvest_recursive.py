def ft_recursive(days: int, start: int) -> None:
    print(f"Day {start}")
    if start < days:
        ft_recursive(days, start + 1)
    else:
        print("Harvest time!")


def ft_count_harvest_recursive() -> None:
    days: int = int(input("Days until harvest: "))
    start: int = 1
    ft_recursive(days, start)

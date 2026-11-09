def ft_count_harvest_iterative() -> None:
    days = int(input("Days until harvest: "))
    for n in range(1, days + 1):
        print("Day", n)
    print("Harvest time!")

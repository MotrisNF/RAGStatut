#!usr/bin/env python3

import sys


def create_inventory(
        inventory_list: list[str]
        ) -> dict[str, int]:
    inventory_dict: dict[str, int] = {}

    for data in inventory_list[1:]:
        temp: list[str] = data.split(":")
        if len(temp) != 2:
            print(f"Error - invalid parameter {data}")
        else:
            try:
                key: str = temp[0]
                value: int = int(temp[1])
                if value <= 0:
                    raise Exception("value need to be greater than")
                if key not in inventory_dict:
                    inventory_dict[key] = value
                else:
                    print(f"Redundant item '{temp[0]}' - discarding")
            except Exception as e:
                print(f"Quantity error for '{temp[0]}': {e} '{temp[1]}'")

    return inventory_dict


def print_inventory(
        inventory_dict: dict[str, int]
        ) -> None:
    item_list: list[str] = list(inventory_dict.keys())
    print(f"Inventory list: {item_list}")
    value = sum(inventory_dict.values())
    quantity: int = len(inventory_dict)
    print(f"Total quantity of the {quantity} items: {value}")
    max_value: int = 0
    max_item: str = "None"
    min_value: int = 0
    min_item: str = "None"
    for item in inventory_dict:
        if max_value == 0:
            max_value = inventory_dict[item]
            min_value = inventory_dict[item]
            max_item = item
            min_item = item
        percentage: float = inventory_dict[item] / value * 100
        print(f"Item {item} represente {round(percentage, 1)}%")
        if inventory_dict[item] > max_value:
            max_value = inventory_dict[item]
            max_item = item
        if inventory_dict[item] < min_value:
            min_value = inventory_dict[item]
            min_item = item
    print(f"Item most abundat: {max_item} with quantity {max_value}")
    print(f"Item less abundat: {min_item} with quantity {min_value}")


def update_inventory(
        item: str = "None",
        value: int = 0,
        inventory_dict: dict[str, int] = {"None": 0}
        ) -> None:
    new_items: dict[str, int] = {item: value}
    inventory_dict.update(new_items)
    print(f"Updated inventory: {inventory_dict}")


def main() -> None:
    inventory_list: list[str] = sys.argv
    inventory_dict: dict[str, int] = create_inventory(inventory_list)
    if not inventory_dict:
        inventory_dict = {}
    print(f"Got inventory: {inventory_dict}")
    if inventory_dict:
        print_inventory(inventory_dict)
    update_inventory("Magic_item", 1, inventory_dict)


if __name__ == "__main__":
    main()

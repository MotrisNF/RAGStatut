from pydantic import BaseModel, Field, ValidationError
from datetime import datetime


class SpaceStation(BaseModel):
    station_id: str = Field(min_length=3, max_length=10)
    name: str = Field(min_length=1, max_length=50)
    crew_size: int = Field(ge=1, le=20)
    power_level: float = Field(ge=0.0, le=100.0)
    oxygen_level: float = Field(ge=0.0, le=100.0)
    last_maintenance: datetime
    is_operational: bool = Field(default=True)
    notes: str | None = Field(default=None, max_length=200)


def valid_imputs() -> None:
    print("Space Station Data Validation")
    print("========================================")
    try:
        station = SpaceStation(
            station_id="LGW125",
            name="Titan Mining Outpost",
            crew_size=6,
            power_level=76.4,
            oxygen_level=95.5,
            last_maintenance=datetime(2023, 7, 11),
            is_operational=True,
            notes=None,
        )
        print("Valid station created:")
        print(f"ID: {station.station_id}")
        print(f"Name: {station.name}")
        print(f"Crew: {station.crew_size}")
        print(f"Power: {station.power_level}%")
        print(f"Oxygen: {station.oxygen_level}%")
        print(f"Status: {station.is_operational}")
        print()
    except ValidationError as e:
        print(f"{e}:")
    print("\n========================================")


def invalid_inputs() -> None:
    print("Expected validation error\n")
    try:
        station = SpaceStation(
            station_id="LG",
            name="Titan Mining Outpost",
            crew_size=25,
            power_level=76.4,
            oxygen_level=95.5,
            last_maintenance=datetime(2023, 7, 11),
            is_operational=True,
            notes=None,
        )
        print("Valid station created:")
        print(f"ID: {station.station_id}")
        print(f"Name: {station.name}")
        print(f"Crew: {station.crew_size}")
        print(f"Power: {station.power_level}%")
        print(f"Oxygen: {station.oxygen_level}%")
        print(f"Status: {station.is_operational}")
        print()
    except ValidationError as e:
        print(f"{e}:")
        print()


def main() -> None:
    valid_imputs()
    invalid_inputs()


if __name__ == "__main__":
    main()

from pydantic import BaseModel, Field, ValidationError, model_validator
from typing_extensions import Self
from enum import Enum
from datetime import datetime


class Rank(Enum):
    CADET = "cadet"
    OFFICER = "officer"
    CAPTAIN = "captain"
    COMMANDER = "commander"
    LIEUTENANT = "lieutenant"


class CrewMember(BaseModel):
    member_id: str = Field(min_length=3, max_length=10)
    name: str = Field(min_length=2, max_length=50)
    rank: Rank
    age: int = Field(ge=18, le=80)
    specialization: str = Field(min_length=3, max_length=30)
    years_experience: int = Field(ge=0, le=50)
    is_active: bool = Field(default=True)


class SpaceMission(BaseModel):
    mission_id: str = Field(min_length=5, max_length=15)
    mission_name: str = Field(min_length=3, max_length=100)
    destination: str = Field(min_length=3, max_length=50)
    launch_date: datetime
    duration_days: int = Field(ge=1, le=3650)
    crew: list[CrewMember] = Field(min_length=1, max_length=12)
    mission_status: str = Field(default="planned")
    budget_millions: float = Field(ge=1.0, le=10000.0)

    @model_validator(mode='after')
    def mission_validator(self) -> Self:
        text: str = ""
        if self.mission_id[:1] != "M":
            raise ValueError("Mission ID must start with 'M")
        count_aptitude: int = 0
        for member in self.crew:
            if member.rank == Rank.COMMANDER or member.rank == Rank.CAPTAIN:
                count_aptitude += 1
        if count_aptitude == 0:
            text = "The mission need at least one Comander or Captain"
            raise ValueError(text)
        count_aptitude = 0
        if self.duration_days > 365:
            for member in self.crew:
                if member.years_experience > 5:
                    count_aptitude += 1
            if count_aptitude < len(self.crew) / 2:
                text = "Long missions (> 365 days) need 50% experienced crew"
                raise ValueError(text)
        count_aptitude = 0
        for member in self.crew:
            if not member.is_active:
                text = "All crew members must be active"
                raise ValueError(text)
        return self


def valid_inputs() -> None:
    print("Space Mission Crew Validation")
    print("========================================")
    try:
        mission = SpaceMission(
            mission_id="M2024_TITAN",
            mission_name="Solar Observatory Research Mission",
            destination="Solar Observatory",
            launch_date=datetime(2024, 3, 30),
            duration_days=120,
            crew=[
                CrewMember(
                    member_id="CM001",
                    name="Sarah Williams",
                    rank=Rank.CAPTAIN,
                    age=43,
                    specialization="Mission Command",
                    years_experience=19,
                    is_active=True,
                ),
                CrewMember(
                    member_id="CM002",
                    name="James Hernandez",
                    rank=Rank.OFFICER,
                    age=35,
                    specialization="Engineering",
                    years_experience=8,
                    is_active=True,
                ),
            ],
            mission_status="planned",
            budget_millions=250.0,
        )
        print("Valid mission created:")
        print(f"Mission: {mission.mission_name}")
        print(f"ID: {mission.mission_id}")
        print(f"Destination: {mission.destination}")
        print(f"Duration: {mission.duration_days} days")
        print(f"Budget: {mission.budget_millions}M")
        print(f"Crew size: {len(mission.crew)}")
        print("Crew members:")
        for member in mission.crew:
            name = member.name
            rank = member.rank.value
            specialitazion = member.specialization
            print(f"- {name} ({rank}) - {specialitazion}")
        print()
    except ValidationError as e:
        print(f"{e}:")
    print("\n========================================")


def invalid_inputs() -> None:
    print("Expected validation error:")
    print("========================================")
    try:
        mission = SpaceMission(
            mission_id="M2024_FAIL",
            mission_name="Failed Validation Mission",
            destination="Mars",
            launch_date=datetime(2024, 5, 1),
            duration_days=30,
            crew=[
                CrewMember(
                    member_id="CM003",
                    name="Alex Kim",
                    rank=Rank.CADET,
                    age=22,
                    specialization="Navigation",
                    years_experience=2,
                    is_active=True,
                ),
            ],
            mission_status="planned",
            budget_millions=50.0,
        )
        print("Valid mission created:")
        print(f"Mission: {mission.mission_name}")
        print(f"ID: {mission.mission_id}")
        print(f"Destination: {mission.destination}")
        print(f"Duration: {mission.duration_days} days")
        print(f"Budget: {mission.budget_millions}M")
        print(f"Crew size: {len(mission.crew)}")
        print("Crew members:")
        for member in mission.crew:
            name = member.name
            rank = member.rank.value
            specialitazion = member.specialization
            print(f"- {name} ({rank}) - {specialitazion}")
        print()
    except ValidationError as e:
        print(f"{e}:")
    print("\n========================================")


def main() -> None:
    valid_inputs()
    invalid_inputs()


if __name__ == "__main__":
    main()

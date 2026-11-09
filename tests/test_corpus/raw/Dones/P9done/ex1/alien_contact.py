from pydantic import BaseModel, Field, ValidationError, model_validator
from typing_extensions import Self
from enum import Enum
from datetime import datetime


class ContactType(Enum):
    RADIO = "radio"
    VISUAL = "visual"
    PHYSICAL = "physical"
    TELEPATHIC = "telepathic"


class AlienContact(BaseModel):
    contact_id: str = Field(min_length=5, max_length=15)
    timestamp: datetime
    location: str = Field(min_length=3, max_length=100)
    contact_type: ContactType
    signal_strength: float = Field(ge=0.0, le=10.0)
    duration_minutes: int = Field(ge=1, le=1440)
    witness_count: int = Field(ge=1, le=100)
    message_received: str | None = Field(default=None, max_length=500)
    is_verified: bool = Field(default=False)

    @model_validator(mode='after')
    def validator(self) -> Self:
        text: str = ""
        if self.contact_id[:2] != "AC":
            raise ValueError("ID must start with 'AC'")
        if self.contact_type == ContactType.PHYSICAL and not self.is_verified:
            raise ValueError("Physical contact must be verified")
        if self.contact_type == ContactType.TELEPATHIC:
            if self.witness_count < 3:
                text = "Telepathic contact requires at least 3 witnesses"
                raise ValueError(text)
        if self.signal_strength > 7.0 and self.message_received is None:
            text = "Strong signals should include received message"
            raise ValueError(text)
        return self


def valid_inputs() -> None:
    print("Alien Contact Log Validation")
    print("========================================")
    try:
        contact = AlienContact(
            contact_id="AC202401",
            timestamp=datetime(2024, 1, 20),
            location="Atacama Desert, Chile",
            contact_type=ContactType.VISUAL,
            signal_strength=9.6,
            duration_minutes=99,
            witness_count=11,
            message_received="Greetings from Zeta Reticuli",
            is_verified=False,
        )
        print("Valid contact report:")
        print(f"ID: {contact.contact_id}")
        print(f"Type: {contact.contact_type.value}")
        print(f"Location: {contact.location}")
        print(f"Signal: {contact.signal_strength}/10")
        print(f"Witnesses: {contact.witness_count}")
        if contact.message_received:
            print(f"Message: {contact.message_received}")
        else:
            print("No message")
        print()
    except ValidationError as e:
        print(f"{e}:")
    print("\n========================================")


def invalid_imputs() -> None:
    print("Expected validation error:")
    print("========================================")
    try:
        contact = AlienContact(
            contact_id="XY202401",
            timestamp=datetime(2024, 1, 20),
            location="Atacama Desert, Chile",
            contact_type=ContactType.VISUAL,
            signal_strength=9.6,
            duration_minutes=99,
            witness_count=11,
            message_received="Greetings from Zeta Reticuli",
            is_verified=False,
        )
        print("Valid contact report:")
        print(f"ID: {contact.contact_id}")
        print(f"Type: {contact.contact_type.value}")
        print(f"Location: {contact.location}")
        print(f"Signal: {contact.signal_strength}/10")
        print(f"Witnesses: {contact.witness_count}")
        if contact.message_received:
            print(f"Message: {contact.message_received}")
        else:
            print("No message")
        print()
    except ValidationError as e:
        print(f"{e}:")
    print("\n========================================")


def main() -> None:
    valid_inputs()
    invalid_imputs()


if __name__ == "__main__":
    main()

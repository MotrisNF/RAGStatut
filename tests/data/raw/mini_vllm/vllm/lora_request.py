"""Request-level LoRA information."""
from dataclasses import dataclass


@dataclass
class LoRARequest:
    """Which adapter a request must use.

    ``lora_name`` is a human readable name, ``lora_int_id`` a globally unique
    positive integer that identifies the adapter inside the engine, and
    ``lora_path`` the local folder where its weights are stored.
    """

    lora_name: str
    lora_int_id: int
    lora_path: str

    def __post_init__(self) -> None:
        """Ids start at 1: 0 is reserved for "no adapter"."""
        if self.lora_int_id < 1:
            raise ValueError(f"id must be > 0, got {self.lora_int_id}")

"""Sampling parameters for text generation."""


class SamplingParams:
    """How the next tokens are chosen during generation."""

    def __init__(self, temperature: float = 1.0, top_p: float = 1.0,
                 max_tokens: int = 16, seed: int | None = None) -> None:
        """Store the sampling options.

        A temperature of 0 means greedy sampling: the most likely token is
        always chosen.
        """
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.seed = seed

    def is_greedy(self) -> bool:
        """Whether sampling always picks the most likely token."""
        return self.temperature == 0.0

    def verify(self) -> None:
        """Reject impossible values."""
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}.")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be at least 1.")

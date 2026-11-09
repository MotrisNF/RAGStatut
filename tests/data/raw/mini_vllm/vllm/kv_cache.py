"""Paged key-value cache."""

DEFAULT_BLOCK_SIZE = 16


class KVCacheManager:
    """Allocates the key-value cache in fixed-size blocks (PagedAttention)."""

    def __init__(self, num_blocks: int,
                 block_size: int = DEFAULT_BLOCK_SIZE) -> None:
        """Create a pool of free blocks."""
        self.block_size = block_size
        self.free_blocks = list(range(num_blocks))

    def allocate(self, num_tokens: int) -> list[int]:
        """Reserve enough blocks for ``num_tokens`` tokens.

        Raises:
            MemoryError: Not enough free blocks; the request is preempted.
        """
        needed = -(-num_tokens // self.block_size)
        if needed > len(self.free_blocks):
            raise MemoryError("Not enough KV cache blocks.")
        blocks = self.free_blocks[:needed]
        self.free_blocks = self.free_blocks[needed:]
        return blocks

    def free(self, blocks: list[int]) -> None:
        """Return blocks to the pool."""
        self.free_blocks.extend(blocks)

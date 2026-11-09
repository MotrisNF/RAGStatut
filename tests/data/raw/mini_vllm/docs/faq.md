# Frequently Asked Questions

## Why does the server refuse to start on port 8000?

Another process is already using that port. Stop it or start the server on a
different port with `--port`.

## Can I use LoRA adapters on CPU?

Not yet. LoRA support currently requires a GPU backend.

## Why is the first request slower than the following ones?

The first request triggers the compilation of some kernels and fills the
caches. Later requests reuse that work.

## How do I make the output reproducible?

Set a fixed `seed` in the sampling parameters and use the same model,
hardware and engine version. Results can still differ slightly between GPU
types because of floating point rounding.

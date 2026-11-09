# Quantization

Quantization stores the weights with fewer bits to save memory and speed up
inference, at the cost of a small loss of accuracy.

## Supported methods

The engine supports AWQ, GPTQ and FP8 quantization. Choose the method with
the `--quantization` argument, for example `--quantization awq`.

## FP8 on older GPUs

FP8 needs hardware support. On GPUs without native FP8 the engine falls back
to a weight-only FP8 kernel, which saves memory but gives no speed-up.

# Performance Tuning

This guide collects the options that most affect throughput and latency.
Most of them trade memory for speed, so the right values depend on the GPU,
the model size and the expected traffic. Measure before and after every
change: a setting that helps a large model on a big GPU can hurt a small
model on a consumer card.

## GPU memory

At start-up the engine profiles the model and reserves a fraction of the
GPU memory for the weights, the activations and the key-value cache. The
fraction is controlled by `gpu_memory_utilization`, which defaults to 0.9.
Raising it leaves more room for the cache and therefore for concurrent
requests, but other processes on the same GPU may then fail to allocate
memory. If the engine reports that there is not enough space for the cache,
lower the maximum model length or the number of concurrent sequences before
touching this value. Lowering `gpu_memory_utilization` is the right fix when
several engines share one GPU.

## Batching

The scheduler groups requests into batches. The upper limit of sequences
processed in one step is `max_num_seqs`, 256 by default. Larger batches
improve throughput but increase the latency of each individual request.
Chunked prefill splits long prompts into pieces so that decoding requests
are not blocked behind a single huge prompt; it is enabled by default on
recent versions.

## CUDA graphs

CUDA graphs capture the GPU kernels of the decoding step once and replay
them, which removes most of the Python overhead. Capturing them takes some
seconds at start-up and some memory. Pass `enforce_eager=True` to disable
them, for example while debugging or when start-up time matters more than
speed.

## Serving many adapters

When many LoRA adapters are served, each one needs its own slot in the GPU.
Keep `max_loras` as low as the traffic allows: every extra slot is memory
that the key-value cache cannot use.

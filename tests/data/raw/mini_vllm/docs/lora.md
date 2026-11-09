# LoRA Adapters

This document shows how to use LoRA adapters with the engine on top of a
base model. Adapters are small sets of weights that change the behaviour of
the model without retraining it.

## Enabling LoRA

Start the engine with the `enable_lora=True` flag. The maximum rank of the
adapters is set with `max_lora_rank`, and `max_loras` controls how many
adapters can be active in the same batch.

## Using API Endpoints

To dynamically load a LoRA adapter, send a POST request to the
`/v1/load_lora_adapter` endpoint with the name and the path of the adapter.
To unload it, send a POST request to `/v1/unload_lora_adapter` with its name.
Dynamic loading must be allowed with the environment variable
`VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`.

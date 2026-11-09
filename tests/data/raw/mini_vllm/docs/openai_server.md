# OpenAI Compatible Server

The engine provides an HTTP server that implements the OpenAI Completions
and Chat Completions APIs, so existing OpenAI clients can talk to it.

## Starting the server

Run `vllm serve <model>` to start the server. By default it listens on port
8000; use `--port` to change it and `--host` to choose the interface.

## Authentication

Pass `--api-key` to require a key. Clients must then send it in the
`Authorization: Bearer <key>` header, or the request is rejected with a 401
error.

*This project has been created as part of the 42 curriculum by saperez-.*

# RAG against the machine

> A Retrieval-Augmented Generation system that answers questions about the vLLM codebase. It runs entirely on CPU, with a BM25 search engine written from scratch, a semantic re-ranker, a small local language model (Qwen3-0.6B), query and answer caches, incremental indexing and a local HTTP API.

| Retrieval (recall@5, moulinette) | Indexing | Retrieval speed | Generation |
|---|---|---|---|
| **docs 91.0 %** (target 80 %) · **code 82.8 %** (target 50 %) | ~13 s for 1,969 files (limit 5 min) · ~1.4 s after editing one file | 199 questions in ~5.4 s (limit 90 s for 200), ~0.7 s when cached | ~17–22 s per answer on CPU · 76 % correct or partial on the docs dataset |

On three other projects it was never tuned on, indexed together (Fly-in, A-Maze-ing and Dones, 220 handwritten questions), recall@5 is **71–93 %** depending on the dataset (**≈ 80 %** overall), and a correct source is in the top 10 for **≈ 88 %** of the questions (see [Other corpora](#other-corpora-fly-in-a-maze-ing-and-dones)).

## Table of contents

1. [Description](#description)
2. [Instructions](#instructions)
3. [Example usage](#example-usage)
4. [System architecture](#system-architecture)
5. [Chunking strategy](#chunking-strategy)
6. [Retrieval method](#retrieval-method)
7. [Answer generation](#answer-generation)
8. [Performance analysis](#performance-analysis)
9. [Design decisions](#design-decisions)
10. [Challenges faced](#challenges-faced)
11. [Known limitations](#known-limitations)
12. [Resources](#resources)

---

## Description

A language model only knows what it was trained on. When you ask a small model about a specific codebase, it either gives no useful answer or makes up a confident one. **Retrieval-Augmented Generation (RAG)** avoids retraining the model. When a question arrives, the system finds the relevant parts of the codebase, gives them to the model as context, and tells it to answer only from that context.

This project applies RAG to the [vLLM](https://github.com/vllm-project/vllm) repository, version 0.10.1: 1,969 Python, Markdown and text files, about 20 million characters. The pipeline has four stages:

1. **Indexing.** Every file is split into chunks of at most 2,000 characters, with one strategy for Python code and another for Markdown and text. The chunks are stored in a lexical index (BM25) and a vector index (embeddings).
2. **Retrieval.** BM25 matches the question against the index, and its best candidates are **re-ranked by meaning** with a lightweight static embedding model (`minishlab/potion-base-8M`). The result is a ranked list of source locations, each one a file path and a character range.
3. **Augmentation.** The best chunks are placed in the prompt, up to a fixed token budget.
4. **Generation.** **Qwen/Qwen3-0.6B** writes the answer from that context, or says that the context does not contain it.

The system is judged on two things:

- **Retrieval quality**, measured with recall@k against reference datasets. The targets are recall@5 ≥ 80 % on documentation questions and ≥ 50 % on code questions.
- **Answer quality**: answers must be coherent, on point and grounded in the retrieved sources.

Everything is exposed through a command-line interface built with Python Fire: `index`, `search`, `search_dataset`, `answer`, `answer_dataset` and `evaluate`. The `serve` command also exposes searching and answering over a local HTTP API (bonus 5).

All five bonuses are implemented. The system was also measured on a second, independent corpus with its own handwritten datasets, to check that it is not only fitted to vLLM.

### Project status

| Part | Status |
|---|---|
| Mandatory part | ✅ Complete: the six commands work end to end, and every target is met by both retrievers |
| Bonus 1: semantic embeddings | ✅ Vector index built with `potion-base-8M` next to the BM25 index |
| Bonus 2: hybrid retrieval | ✅ BM25 candidates re-ranked with the vector scores (default; `--retriever bm25` gives the lexical search alone) |
| Bonus 3: incremental indexing | ✅ `index` compares each file's size and modification time with `manifest.json` and only reads, tokenizes and embeds the new and modified files; the chunks of modified and deleted files leave both indexes. The result is identical to a full index. `--full` forces a full one |
| Bonus 4: caching | ✅ Cold start: the embedding model is saved with the index and run with numpy, so searching never imports PyTorch. Repeated queries: results are cached on disk, so a repeated search loads no index at all. Repeated answers: an answer already generated for the same question and sources is read from disk without importing PyTorch, in `answer`, `answer_dataset` and the API (`--cache=False` turns both caches off) |
| Bonus 5: local HTTP API | ✅ `serve` starts a FastAPI server on `127.0.0.1` with `/health`, `/search` and `/answer`, which return the same JSON as the CLI, plus `POST /index` to re-index without restarting. Each request can choose `bm25` or `hybrid`. The model is loaded on the first `/answer` that is not cached and kept for the next ones |
| Robustness | ✅ No traceback on degenerate input (empty or blank query, `k` of 0, negative or not a number, missing files, malformed JSON, a chunk size above 2,000, a missing corpus folder): each one ends with a clear message and exit code 1 |
| Quality gates | ✅ `flake8` and `mypy --strict` clean, docstrings on every module, class and function, about 1,370 pytest tests |

---

## Instructions

### Requirements

| Requirement | Details |
|---|---|
| Python | ≥ 3.10 |
| Package manager | [uv](https://docs.astral.sh/uv/). It is the only tool you need: `uv sync` installs everything. |
| Hardware | CPU only. PyTorch comes from the CPU wheel index configured in `pyproject.toml`, so no CUDA libraries are downloaded. |
| Memory for retrieval | ~0.5 GB (`index`, `search`, `search_dataset`, `evaluate`) |
| Memory for generation | The model type is chosen from the available memory before loading. It uses **float32** (fastest) when at least 3.5 GB is available and **bfloat16** (~2–3× slower on CPU) when at least 2.0 GB is available. With less, `answer` stops with a clear error before loading anything. |
| Network | Only the first time: `index` downloads the embedding model (~30 MB) and the first `answer` downloads Qwen3-0.6B (~1.2 GB) to the Hugging Face cache. After that, every run works offline. |

### Installation

```bash
uv sync            # or: make install
```

`uv sync --locked` installs exactly the versions in `uv.lock`, and fails if the lock file is out of date.

On computers with a small home directory (for example at 42), the virtual environment and the model cache can live elsewhere. `make` detects `/sgoinfre/students/$USER` and exports these variables automatically. You can also set them yourself:

```bash
export UV_PROJECT_ENVIRONMENT=/sgoinfre/students/$USER/rag-venv
export HF_HOME=/sgoinfre/students/$USER/hf
```

### Project layout

```
.
├── README.md
├── Makefile             # install, run, debug, lint, test and pipeline rules
├── pyproject.toml       # dependencies (CPU-only PyTorch index), mypy and pytest settings
├── uv.lock              # exact versions, for `uv sync`
├── .python-version      # Python version used by uv
├── .flake8              # flake8 settings (excludes data/ and the test corpora)
├── .gitignore
├── .vscode/             # editor settings (Pylance skips data/ and tests/data/)
├── src/                 # the Python package, run with: uv run python -m src <command>
│   ├── __main__.py      # entry point: Fire, error handling and exit codes
│   ├── commands.py      # RagCLI: the six commands and serve
│   ├── api.py           # RagApi, RetrieverLoader: the local HTTP API (bonus 5)
│   ├── answer_cache.py  # AnswerCache: answers saved between commands
│   ├── argv_guard.py    # command-line checks before Fire runs
│   ├── reader.py        # CorpusReader
│   ├── incremental_indexer.py  # IncrementalIndexer: what index re-indexes (bonus 3)
│   ├── chunker.py       # FixedSizeChunker, PyChunker, MdChunker
│   ├── tokenizer_bm.py  # Tokenizer (BM25 terms, query stopwords)
│   ├── base_retriever.py, bm25_index.py, vector_index.py,
│   │   static_encoder.py, hybrid_retriever.py, cached_retriever.py
│   ├── generator.py     # Generator (Qwen3-0.6B)
│   ├── evaluator.py     # recall@k, same rules as the moulinette
│   ├── models.py        # pydantic models
│   └── exceptions.py    # RagError hierarchy
├── tests/               # pytest suite: conftest.py and 19 test_*.py files
│   ├── data/            #   mini corpus and datasets for `make test_rag`
│   ├── answer_battery/  #   10 questions with expected answers (`make test_answer`)
│   └── test_corpus/     #   other corpora for `make test_corpus`, with the layout of data/
│       ├── raw/         #     fly-in/ (94 files), amazeing/ (25), Dones/ (83)
│       └── datasets/    #     5 handwritten datasets: flyin docs/code, amazeing docs/code, dones code
├── data/                # the subject's layout, not in the repository (provided at evaluation)
│   ├── raw/             #   the corpus: data/raw/vllm-0.10.1/...
│   ├── processed/       #   the index written by `index`
│   ├── datasets/        #   UnansweredQuestions/ and AnsweredQuestions/
│   └── output/          #   search_results/ and search_results_and_answer/, scoped by dataset
```

`data/` is not part of the repository (`.gitignore`): the corpus and the datasets are provided at evaluation, and the commands create the index and output folders they write to. Every path is a CLI option, and the defaults follow the subject's layout.

### Running

```bash
uv run python -m src <command> [options]     # or: uv run -m src ...
uv run -m src <command> --help               # options of a command
```

### Command reference

| Command | Options (default) | Output |
|---|---|---|
| `index` | `--max_chunk_size` (2000, at most 2000) · `--raw_dir` (`data/raw`) · `--index_dir` (`data/processed`) · `--full` (`False`) | The BM25 index (`index.json`), the vector index (`vectors.npy`, `owners.npy`), the embedding model (`embeddings.npy`, `tokenizer.json`) and the corpus stamps (`manifest.json`). Only the files changed since the last run are re-indexed; with no change, nothing is written. Removes the query and answer caches when the index changes |
| `search <query>` | `--k` (10) · `--index_dir` · `--retriever` (`hybrid` \| `bm25`) · `--cache` (`True`) · `--save_directory` (`data/output/search_results/SingleQuestions`) | `StudentSearchResults` JSON on stdout and in `<question_id>.json` |
| `search_dataset` | `--dataset_path` (`data/datasets/UnansweredQuestions/dataset_docs_public.json`) · `--k` (10) · `--save_directory` (`data/output/search_results/UnansweredQuestions`) · `--index_dir` · `--retriever` · `--cache` | `StudentSearchResults` JSON, with the same file name as the dataset |
| `answer <query>` | `--k` (10) · `--index_dir` · `--retriever` · `--dtype` (`auto` \| `float32` \| `bfloat16`) · `--cache` · `--save_directory` (`data/output/search_results_and_answer/SingleQuestions`) | `StudentSearchResultsAndAnswer` JSON on stdout and in `<question_id>.json` |
| `answer_dataset` | `--student_search_results_path` (`data/output/search_results/UnansweredQuestions/dataset_docs_public.json`, what `search_dataset` writes by default) · `--save_directory` (`data/output/search_results_and_answer/UnansweredQuestions`) · `--dtype` · `--index_dir` (`data/processed`, only for the answer cache) · `--cache` | `StudentSearchResultsAndAnswer` JSON, with the same file name as the input. The model is only loaded if a question with sources is not cached |
| `evaluate` | `--student_search_results_path` (the same default as `answer_dataset`) · `--dataset_path` (`data/datasets/AnsweredQuestions/dataset_docs_public.json`) · `--k` (10) · `--show_failed` | recall@1/3/5/10 report |
| `serve` | `--host` (`127.0.0.1`) · `--port` (8000) · `--index_dir` · `--retriever` (the default of each request) · `--dtype` · `--cache` · `--raw_dir` (`data/raw`, for `POST /index`) | A local HTTP API until Ctrl+C (see [HTTP API](#http-api-bonus-5)). Nothing is saved to `data/output` |

- `search` and `answer` print their JSON on stdout and also save it as `<save_directory>/<question_id>.json`. Progress bars and status messages, including where the file was saved, go to stderr, so you can still redirect or pipe the JSON (`> out.json`, `| jq`).
- Put boolean flags such as `--show_failed` after the paths, or write them as `--show_failed=True`. Otherwise Fire reads the next word as the flag's value.
- **The dataset commands run without options.** Their input files default to the public docs dataset, chained: `search_dataset` writes its results where `answer_dataset` and `evaluate` look for them, so `search_dataset`, then `evaluate` (or `answer_dataset`), works with no flags. A default file that does not exist is reported as an argument error with its path. Pass the options to use the code dataset or another file.
- `--dtype` and `--retriever` only accept the listed values. Any other value is rejected with a clear argument error.
- `index` is incremental: run it again after editing the corpus and only the new and modified files are processed. A different `--raw_dir` or `--max_chunk_size`, a missing or broken `manifest.json` or index file, or `--full` give a full index.
- `--cache` is on by default: searches already made with the same index and retriever are read from `<index_dir>/query_cache_<retriever>.json`, and answers already generated for the same question and sources from `<index_dir>/answer_cache.json`. A cached answer never imports PyTorch. Turn both off with `--cache=False`.

### Makefile

| Rule | What it does |
|---|---|
| `make install` | `uv sync` |
| `make run ARGS="..."` | `uv run -m src $(ARGS)` |
| `make debug ARGS="..."` | The same, under `pdb` |
| `make serve SERVE_ARGS="..."` | Starts the HTTP API (`serve`) until Ctrl+C |
| `make lint` | `flake8 .` and `mypy .` with the subject's flags |
| `make lint-strict` | `flake8 .` and `mypy . --strict` |
| `make test` | The pytest suite |
| `make test_rag` | index → search_dataset → evaluate on the mini corpus in `tests/data` (variables `TEST_DATASET`, `TEST_K`, `TEST_CHUNK`) |
| `make test_rag_base` | The same on the real data, as the moulinette runs it: k = 10, chunks of 2000, docs and code datasets |
| `make test_corpus` | The same as `test_rag_base` on three other projects in `tests/test_corpus/raw` (Fly-in, A-Maze-ing and Dones), indexed together once, then search_dataset → evaluate for five handwritten datasets, each recall reported apart (variables `CORPUS_K`, `CORPUS_CHUNK`, `CORPUS_DATASETS`; `ANSWER=1` also runs `answer_dataset`) |
| `make test_limits` | Checks the subject's time limits and prints PASS or FAIL: indexing ≤ 5 min (`test_index`) and 200 questions ≤ 90 s (`test_retrieval`) |
| `make test_limits_cached` | `test_limits` (indexing removes the query cache, so the searches are uncached), then `test_retrieval` again without indexing, with every question read from the cache |
| `make test_answer` | Runs `answer` on the 10-question battery in `tests/answer_battery` and prints the expected answer after each one |
| `make test_answer_dataset` | The same on a whole `AnsweredQuestions` dataset, loading the model once: `search_dataset`, then `answer_dataset`, then each question with its answer and its reference answer, with a log. Cached answers are not generated again (variables `ANSWER_DATASET`, `ANSWER_START`, `ANSWER_LIMIT`, `ANSWER_SEARCH_ARGS`, `ANSWER_ARGS`) |
| `FULL=0` | With `test_index`, `test_limits`, `test_rag`, `test_rag_base` or `test_corpus`: index incrementally instead of with `--full` (the default, `FULL=1`). To show the difference: `make test_index`, `touch` a corpus file, then `make test_index FULL=0` |
| `make clean` | Removes `__pycache__`, `.mypy_cache`, `.pytest_cache` and the outputs of `test_rag` and `test_corpus` |
| `make cleandata` | Removes the generated index and JSON outputs, and keeps the folder layout |

---

## Example usage

### Full pipeline, as the evaluation runs it

**1. Index the corpus once.**

```console
$ uv run python -m src index --max_chunk_size 2000
Read 1969 of 1969 files (20,147,166 characters) from data/raw
  .md: 178 | .py: 1763 | .txt: 28
12907 chunks (max 2000 characters), 12907 indexed now
Ingestion complete! Index saved under data/processed/
```

Run it again and only what changed is processed:

```console
$ uv run python -m src index
Index up to date: no file changed in data/raw (12907 chunks in data/processed/)

$ echo "" >> data/raw/vllm-0.10.1/README.md && uv run python -m src index
Read 1 of 1 files (11,364 characters) from data/raw
  .md: 1
Incremental: 0 new, 1 modified, 0 deleted
12907 chunks (max 2000 characters), 7 indexed now
Ingestion complete! Index saved under data/processed/
```

**2. Search a whole dataset.**

```console
$ uv run python -m src search_dataset \
    --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
    --k 10 \
    --save_directory data/output/search_results/UnansweredQuestions
File saved in 'data/output/search_results/UnansweredQuestions/dataset_docs_public.json'
```

**3. Score the results with the moulinette.**

```console
$ ./moulinette evaluate_student_search_results \
    data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    data/datasets/AnsweredQuestions/dataset_docs_public.json \
    --k 10 --max_context_length 2000
Student data is valid: True
Evaluation Results
========================================
Recall@1: 0.640 Recall@3: 0.870 Recall@5: 0.910 Recall@10: 0.920
```

**4. Generate the answers.**

```console
$ uv run python -m src answer_dataset \
    --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    --save_directory data/output/search_results_and_answer/UnansweredQuestions
Torch mode: float32
Generating answers: 100%|██████████| 100/100
File saved in 'data/output/search_results_and_answer/UnansweredQuestions/dataset_docs_public.json'
```

Each entry of the output keeps the `question_id`, the question and the retrieved sources from step 2, and adds the answer:

```json
{
  "question_id": "17526382-…",
  "question": "What HTTP endpoint is used to dynamically load a LoRA adapter in vLLM?",
  "retrieved_sources": [
    { "file_path": "data/raw/vllm-0.10.1/docs/features/lora.md", "first_character_index": …, "last_character_index": … }
  ],
  "answer": "The HTTP endpoint used to dynamically load a LoRA adapter in vLLM is `/v1/load_lora_adapter`."
}
```

### Single questions

```console
$ uv run -m src search "How do I load a LoRA adapter?" --k 5
$ uv run -m src search "How do I load a LoRA adapter?" --k 5 --retriever bm25
```

Each command prints a `StudentSearchResults` JSON with the 5 best sources (file path and character range), saves it under `data/output/search_results/SingleQuestions/`, and says where on stderr. Run the same search again and it is read from the query cache, without loading any index. `--cache=False` searches again:

```console
$ uv run -m src search "How do I load a LoRA adapter?" --k 5 --cache=False
```

```console
$ uv run -m src answer "What is the minimum version of bitsandbytes required for vLLM quantization?"
```

This prints a `StudentSearchResultsAndAnswer` JSON, also saved under `data/output/search_results_and_answer/SingleQuestions/`, whose answer is *"The minimum version of bitsandbytes required for vLLM quantization is 0.46.1."*

- A question about a flag that does not exist (`--turbo-mode`), or one unrelated to the codebase ("What is the capital of France?"), gets the fixed refusal: *"The context does not contain the answer."*
- A query that matches nothing (`"???"`, or only stopwords) gets *"No relevant sources were found in the corpus."* without loading the model.

### Own evaluation

`evaluate` computes the same recall as the moulinette, so you can iterate without it. `--show_failed` lists the questions whose sources were not found:

```console
$ uv run -m src evaluate \
    --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_code_public.json \
    --dataset_path data/datasets/AnsweredQuestions/dataset_code_public.json \
    --show_failed
```

### HTTP API (bonus 5)

```console
$ uv run -m src serve                       # or: make serve
http://127.0.0.1:8000/docs
```

With the cache on (the default), the index is loaded by the first search that is not cached, and then kept in memory; with `--cache=False`, at start-up. Open `http://127.0.0.1:8000/docs` for an interactive page that tries every endpoint from the browser, or call them from any other program, for example `curl` in another terminal:

```console
$ curl "http://127.0.0.1:8000/health"
{"status":"ok"}
$ curl -G "http://127.0.0.1:8000/search" --data-urlencode "query=How do I load a LoRA adapter?" -d k=5
$ curl -G "http://127.0.0.1:8000/answer" --data-urlencode "query=What is the minimum version of bitsandbytes required for vLLM quantization?"
$ curl -G "http://127.0.0.1:8000/search" --data-urlencode "query=How do I load a LoRA adapter?" -d retriever=bm25
$ curl -X POST "http://127.0.0.1:8000/index"
{"up_to_date":true,"full":false,"new":[],"modified":[],"deleted":[],"skipped":[],"chunk_count":12907,"fresh_count":0}
```

| Endpoint | Parameters | Response |
|---|---|---|
| `GET /health` | none | `{"status": "ok"}` |
| `GET /search` | `query` (required, not only spaces) · `k` (10, > 0) · `retriever` (`bm25` \| `hybrid`, default: `serve --retriever`) | `StudentSearchResults`, as the `search` command |
| `GET /answer` | `query` (required, not only spaces) · `k` (10, > 0) · `retriever` | `StudentSearchResultsAndAnswer`, as the `answer` command |
| `POST /index` | `full` (`false`) · `max_chunk_size` (2000, 1–2000) | `IndexSummary`: whether it was up to date or full, the new, modified, deleted and skipped files, and the chunk counts |

- **Same JSON as the CLI.** The endpoints return the same pydantic models, so the output can be fed to `evaluate` like the files the commands save. The API itself saves nothing.
- **The model is loaded once.** The first `/answer` that is not cached loads PyTorch and Qwen (tens of seconds). The next ones reuse the model and only pay for the generation. `/health` and `/search` never load it. `--dtype` is chosen when the server starts, because the model is loaded only once.
- **Answers are cached**, in the same `answer_cache.json` as the CLI: a question already answered with the same sources, by the API or by `answer` / `answer_dataset`, is returned at once, without the model. Several identical requests at the same time generate the answer once.
- **Re-index without restarting.** `POST /index` does what the `index` command does (only the changed files, or everything with `full=true`). If the index changed, the server loads it again and starts an empty answer cache. Searches wait while it runs. It is a `POST` because it changes the index on disk; a `GET` gives 405.
- **Each request chooses its retriever.** `retriever=bm25` or `hybrid`; without it, the one given to `serve`. The other one is loaded the first time it is asked for, and both share a single BM25 index in memory.
- **Errors are HTTP status codes, never a traceback.** Invalid parameters (an empty `query` or one of only spaces, `k` of 0 or not a number) give **422** with the reason. Every `RagError` (no index yet, a broken index, a model that does not fit in memory) gives **503** with the same message as the CLI, for example `{"detail": "No index found in 'data/processed'. Run 'index' first."}`. The server keeps running and the next request tries again, so running `index` in another terminal is enough, without restarting it.
- **Local by default.** The server listens on `127.0.0.1`, so only this machine can reach it. `--host 0.0.0.0` opens it to the network, without any authentication.

### Other corpora

```console
$ make test_corpus                    # index the three projects, then search + evaluate their 5 datasets
$ make test_corpus CORPUS_CHUNK=500   # the same with another chunk size
$ make test_corpus ANSWER=1           # also generate the answers with Qwen
```

### With `make`

The whole command goes inside `ARGS`. If the question has spaces, use single quotes outside so the inner double quotes reach the program:

```bash
make run ARGS='search "How do I load a LoRA adapter?" --k 5'
make test_rag TEST_DATASET=dataset_mini_hard.json TEST_K=10
make test_answer_dataset ANSWER_START=77 ANSWER_LIMIT=10
```

---

## System architecture

```
                 index
data/raw ──► CorpusReader ──► PyChunker (.py) ──┐   ┌─► BM25Index ──────────► index.json
                              MdChunker (.md,   ├───┤   (Tokenizer)
                                        .txt) ──┘   └─► VectorIndex ────────► vectors.npy, owners.npy
                              (same chunks, same order)  windows + StaticEncoder ► embeddings.npy, tokenizer.json

                 search / search_dataset / answer                (--retriever hybrid, the default)
question ──► BM25Index.search ──► top 30 candidates ──┐
        └──► VectorIndex.chunk_score ─────────────────┴─► HybridRetriever: re-rank ──► top-k sources ──► JSON
                                                                                          │
                                                          answer / answer_dataset         ▼
                                                              Generator (Qwen3-0.6B, loaded once)
                                                              context within the token budget → prompt
                                                              → greedy generation → answer → JSON
```

`index` goes through `IncrementalIndexer`. It compares the stamps of the corpus files (size and modification time, read with `stat` and without opening the files) with those saved in `manifest.json`. Only new and modified files are read and chunked. `_plan` builds the sorted list of chunks of the new index and a map `{old chunk id: new chunk id}` of the chunks that stay, and both indexes use that same map in `update`: kept chunks move their data to their new id, removed ones leave, and only the new ones are tokenized and embedded. With no usable manifest or index, the same path runs with no old chunks, which is a full index.

With `--retriever bm25`, the top-k sources come straight from `BM25Index.search` and the vector index is never loaded. Searching never reads the Hugging Face cache, because the embedding model is read from the index folder.

With `--cache` (the default), the chosen retriever is wrapped in a `CachedRetriever`. A question already searched is answered from `query_cache_<retriever>.json`, and the indexes are only loaded by the first question that is not cached, so a fully cached command never loads them.

| Component | File | Role |
|---|---|---|
| `RagCLI` | `src/commands.py` | The six Fire commands. Each one validates its arguments, calls the components and writes the output. `_load_retriever` is the only place where a retriever is chosen. |
| `CliCommand.validated` | `src/commands.py` | Decorator that runs pydantic `validate_call` on every command (Fire ignores type hints) and gives `--help` a clean signature. |
| `ArgvGuard` | `src/argv_guard.py` | Checks the command line **before** Fire runs it: unknown commands and options, private names, extra arguments, a misplaced `--help`. Without it, Fire would run the command first and complain afterwards. |
| `CorpusReader` | `src/reader.py` | Reads every `.py`, `.md` and `.txt` file under the corpus folder. Paths are stored relative to the project root, because the grader compares them verbatim. Folders outside the project are rejected. |
| `IncrementalIndexer` | `src/incremental_indexer.py` | Runs `index`: finds the new, modified and deleted files (`_diff`), plans the chunk ids of the new index (`_plan`), updates or builds both indexes, and writes `manifest.json` last. |
| `PyChunker`, `MdChunker` | `src/chunker.py` | The two chunking strategies, built on a shared `FixedSizeChunker`. |
| `Tokenizer` | `src/tokenizer_bm.py` | Turns text into terms, the same way for chunks and queries. Stopwords are removed from queries only. |
| `Retriever` | `src/base_retriever.py` | Abstract base with one method, `search(query, k)`. The commands only see a `Retriever`. |
| `BM25Index` | `src/bm25_index.py` | A `Retriever`. Builds the inverted index, ranks chunks, and saves and loads the index as JSON. |
| `VectorIndex` | `src/vector_index.py` | Splits every chunk into windows, embeds them and scores every chunk by its best window. Saves and loads two `.npy` files and the model. |
| `StaticEncoder` | `src/static_encoder.py` | The embedding model run with numpy: a tokenizer plus one vector per token. A text's vector is the normalised mean of its token vectors. |
| `HybridRetriever` | `src/hybrid_retriever.py` | A `Retriever` that re-ranks the BM25 candidates with the vector scores. |
| `CachedRetriever` | `src/cached_retriever.py` | A `Retriever` that wraps another one (decorator pattern), remembers its results by question and `k`, saves them to disk between commands, and only loads the wrapped retriever when a question is not cached. |
| `Generator` | `src/generator.py` | Loads Qwen3-0.6B once, fits the chunks into the token budget, builds the prompt and generates the answer. |
| `Evaluator` | `src/evaluator.py` | recall@k with the moulinette's rules: same file, character IoU ≥ 0.05, sources of at most 2,000 characters. |
| Models | `src/models.py` | The subject's pydantic models (`MinimalSource`, `StudentSearchResults`…) and internal ones (`Chunk`, `Document`, `SearchHit`, `IndexData`, `WindowChunk`, `QueryCache`, and for bonus 3 `FileStamp`, `IndexManifest`, `CorpusChanges`, `IndexPlan`, `IndexReport`). |
| Errors | `src/exceptions.py` | One hierarchy (`RagError`). Every expected failure ends with a clear message and an exit code, never a traceback. |

- `answer` and `answer_dataset` import `Generator` inside the command, so `index`, `search`, `search_dataset`, `evaluate` and `--help` never load PyTorch.
- `answer_dataset` reads the output of `search_dataset`, which only stores the location of each source. It rebuilds each chunk's text from its file and character range, reading every file once. Every source path is checked first: a path outside the project (`/etc/passwd`, `../..`) is rejected before PyTorch is loaded.

---

## Chunking strategy

Every file is split into chunks of at most `--max_chunk_size` characters. The default is 2,000, the moulinette's limit, and larger values are rejected. Offsets are exact: `text[first_character_index:last_character_index]` (end exclusive) gives back the chunk. Chunks never overlap, and only whitespace is left out. The corpus gives **12,907 chunks**.

### Common mechanism: `FixedSizeChunker`

1. The window `[start, start + size)` is cut at the best separator in a priority list, found with `rfind`.
2. A separator is only accepted **in the second half of the window**. This minimum fill avoids tiny chunks.
3. If no separator is found, the cut is hard.

The two strategies only change the separator list and `CUT_BEFORE`, the separators that start the next chunk instead of ending the current one. That way the size, offset and progress invariants are written and tested once, for both strategies.

### Python code: `PyChunker`

- **Separators, best first:** a top-level decorator, `class`, `def`, `async def`; then the same separators indented by four spaces (methods); then a blank line, a newline and a space.
- **Cut before definitions.** The cut goes before the keyword, so each chunk starts with a whole class, function or method when it fits. 51 % of its chunks start at a definition, against 27 % with plain fixed-size cuts.
- **Decorators stay with their definition.** A cut that lands on a definition is moved back over the `@decorator` lines just above it. Chunks that end with a decorator line went from 406 to 1. The cut never moves before the start of the chunk, so chunking always advances.

### Markdown and text: `MdChunker`

- **Separators, best first:** the end of a fenced code block (`\n```\n`), then a blank line, a newline and a space.
- **Code blocks stay whole.** In the vLLM docs almost every opening fence names a language (```` ```bash ````, ```` ```python ````): 820 such openings against 2 bare ones. So `\n```\n` matches the closing fences, and a command or snippet stays whole in the chunk that closes it.
- **Headings are not separators.** 79 of the 100 docs reference sources start at a heading, yet every heading-based variant measured about 4 points lower (see [Performance analysis](#markdown-chunking-variants)).
- `.txt` files use the same strategy, which is the subject's "Markdown / text" chunking.

---

## Retrieval method

Retrieval has two layers. **BM25**, implemented from scratch, finds the candidates. A **semantic re-ranker** then reorders them. `--retriever bm25` runs the first layer alone.

### Tokenizer

The text is lowercased and read in two passes:

1. Every run of letters or digits: `max_model_len` → `max`, `model`, `len`.
2. snake_case names kept whole, as an extra term: `max_model_len`. A question that quotes an identifier matches it exactly, and the parts still match a question that only names one of them.

The **file path** of each chunk is tokenized too, and its terms are added to the chunk's terms. 35 of the 99 code questions contain words (`keye`, `sampler`, `pynccl`) that only appear in the path of the right file.

### BM25 index and ranking

The index is an inverted index `term → {chunk id: term frequency}`, plus the length of each chunk in tokens. `index` builds it once, saves it as JSON, and pydantic (`IndexData`) validates it on load.

For a query, every distinct term adds to the score of every chunk that contains it:

```
score(chunk) = Σ  idf(t) · tf · (K1 + 1) / (tf + K1 · (1 − B + B · length / avg_length))
idf(t)       = ln((N − df + 0.5) / (df + 0.5) + 1)
```

| Parameter | Value | Effect |
|---|---|---|
| `idf` | | Rewards rare terms. Never negative. |
| `K1` | 1.9 | Saturation: each repetition of a term helps less than the previous one. |
| `B` | 0.5 | Length normalisation: lowers the score of long chunks. |

Only chunks that share at least one term with the query are scored. The top k are taken with `heapq.nsmallest` on `(−score, chunk id)`. The id breaks ties, so the ranking never depends on set iteration order and is identical between runs.

**Query stopwords.** Question words and common English words (`what`, `how`, `the`, `use`…) are removed from the **query only**. These words are rare in a code corpus, so BM25 would give them a high idf (`why` 5.54, `how` 3.78, against 2.45 for `lora`), and chunks that only share the wording of the question would rank high. The index keeps every term, so the list can change without rebuilding it.

### Semantic re-ranking (bonuses 1 and 2)

- **Model.** `minishlab/potion-base-8M` is a Model2Vec static embedding model of ~30 MB. It is only a tokenizer and a table with one 256-dimension vector per token, so `StaticEncoder` runs it with `tokenizers` and numpy: a text's vector is the mean of its token vectors, normalised. It gives the same vectors as `sentence-transformers` (largest difference 9e-8 over the whole corpus) without importing PyTorch.
- **Windows.** Each chunk is split into windows of whole lines of up to 400 characters, and the words of the chunk's file path are put in front of each window (`vllm model executor … bitsandbytes py\n<window>`). This gives 59,828 normalised vectors, saved as `vectors.npy`, plus `owners.npy` with the chunk id of each window.
- **Chunk score.** The question is embedded as written, with its stopwords. The dot product with every window is the window's cosine similarity, and each chunk keeps the score of its **best** window (`np.maximum.at`).
- **Re-ranking.** BM25 returns its top `CANDIDATES` (30), and each candidate gets a new score:

  ```
  combined = (1 − ALPHA) × BM25 score / best BM25 score + ALPHA × similarity      (ALPHA = 0.5)
  ```

  The candidates are then sorted again, and ties keep the BM25 order. Dividing by the best BM25 score puts both parts on the same 0–1 scale. The vectors only reorder chunks BM25 already found, so they can never bring in a chunk that shares no word with the question.
- **Consistency.** `index` builds both indexes from the same list of chunks, and every `SearchHit` carries its `chunk_id`. `HybridRetriever` checks that both indexes have the same number of chunks. If one was rebuilt without the other, it stops with *"Run 'index' again"*.

---

## Answer generation

### Context

The retrieved chunks are added best first. Each one is a block with a `[CHUNK i]` header, its file path and its text. Chunks are added until the budget of **2,200 tokens** is reached (`MAX_CONTEXT_TOKENS`, counted with Qwen's own tokenizer). The first chunk that does not fit stops the loop, so the best chunks are always kept.

### Prompt

The prompt is designed for a 0.6B model, which follows examples better than long lists of rules.

| Part | Content |
|---|---|
| System message | Answer only from the context and quote the exact names and values. When the answer is not there, give the fixed sentence *"The context does not contain the answer."* |
| Two few-shot turns | One invented question answered by quoting the context, and one unrelated question refused. The examples are invented on purpose, so the model learns the behaviour and not the content. |
| User message | The context first, then the **question last**, where a small model pays the most attention, then a one-line reminder: *"Answer the question using the context above."* |

### Decoding and post-processing

- **Greedy decoding** (`do_sample=False`). The same input always gives the same answer, so prompt variants can be compared fairly.
- **No thinking mode** (`enable_thinking=False`), and at most 150 new tokens (`MAX_NEW_TOKENS`).
- **Refusal clean-up.** Sometimes the model gives an answer and then adds the refusal sentence, which contradicts it. In that case the sentence is removed.

### Model loading

- **Loaded once per command.** `answer_dataset` loads the model once, not once per question.
- **Cache first.** The model is loaded with `local_files_only=True` and only downloaded when it is not cached. This avoids a request to the Hugging Face Hub on every run, which once hung a run for minutes. A failed download ends in a clean error.
- **Model type chosen from the available memory.** When Linux runs out of memory it kills the process (`Killed`, exit code 137), and there is no Python exception to catch. So the available memory is read from `/proc/meminfo` **before** loading:

| Available memory | Type | Why |
|---|---|---|
| ≥ 3.5 GB | float32 | Reads the prompt ~3× faster, because this CPU has no native bfloat16 and emulates it |
| ≥ 2.0 GB | bfloat16 | Half the memory |
| < 2.0 GB | none: clean error, exit 1 | The model would not fit |
| unknown (no `/proc/meminfo`) | bfloat16 | The conservative choice |

`--dtype float32` and `--dtype bfloat16` force a type for testing. If the forced type looks too large for the available memory, a warning is printed, but the model is still loaded: the user decides.

---

## Performance analysis

All recall figures were measured on the public datasets (100 docs questions, 99 code questions) and confirmed with the moulinette. `evaluate` gives the same numbers. With about 100 questions, **one question ≈ 1 %**, so differences of one or two points are noise.

### Current results

Moulinette, chunks of 2,000 characters, k = 10 (`make test_rag_base`):

| Retriever | Dataset | Recall@1 | Recall@3 | **Recall@5** | Recall@10 | Target (@5) |
|---|---|---|---|---|---|---|
| **hybrid** (default) | docs | 0.640 | 0.870 | **0.910** | 0.920 | ≥ 0.80 ✅ |
| | code | 0.545 | 0.778 | **0.828** | 0.859 | ≥ 0.50 ✅ |
| `bm25` | docs | 0.610 | 0.830 | **0.870** | 0.920 | ≥ 0.80 ✅ |
| | code | 0.515 | 0.717 | **0.758** | 0.818 | ≥ 0.50 ✅ |

Times measured with `make test_limits` on a 42 campus computer:

| Step | Time | Subject limit |
|---|---|---|
| `index`: read 1,969 files, chunk them, build and save both indexes and the model | **13.2 s**, including ~4 s to embed the 59,828 windows | 5 min |
| `search_dataset` × 2 (199 questions, each process loading both indexes and the model) | **5.4 s** | 90 s for 200 questions |
| One question, after loading | ~7 ms (hybrid) | |
| Load the BM25 index / the vector index and model | ~0.8 s / ~0.3 s | |
| `answer`: load Qwen3-0.6B | ~5.6 s | |
| `answer`: one question, model load included (float32, ~2,000-token prompt) | ~15–25 s | none |
| `answer_dataset`: 100 questions, model loaded once | ~17 s per question, ~28 min in total | none |

### Other corpora: Fly-in, A-Maze-ing and Dones

Every parameter above was tuned on the vLLM public datasets, so good numbers there could just mean the system is fitted to vLLM. To check it on code it has never seen, `make test_corpus` runs the same pipeline, unchanged, on three other projects, **indexed together** in one index, so each project is also noise for the questions about the others:

| Project | What it is | Files | Characters |
|---|---|---|---|
| `fly-in` | A drone-routing simulator | 94 (21 `.py`, 2 `.md`, 71 map `.txt`) | 208 K |
| `amazeing` | A maze generator and terminal / MiniLibX game, with a reusable `mazegen` package | 25 (19 `.py`, 3 `.md`, 3 `.txt`) | 167 K |
| `Dones` | Eleven series of Python exercises (classes, exceptions, files, ABCs, packages, pydantic, functional tools) | 83 (82 `.py`, 1 `.txt`) | 111 K |

The questions were written by hand for this test, in five datasets so each recall is reported apart, as with the public ones. Each source range was computed by a script from exact text anchors, checked to be unique in its file and to cover the code or text that answers the question (never counted by hand).

| Dataset | Questions | About | Median source |
|---|---|---|---|
| Fly-in docs | 39 | the two READMEs and the map files | 254 chars |
| Fly-in code | 53 | `src/` (parser, router, scheduler, drones, view...) | 425 chars |
| A-Maze-ing docs | 25 | the two READMEs, `config.txt`, `highscore.txt` | 601 chars |
| A-Maze-ing code | 38 | `mazegen/` (generator, solver) and `src/` (config, game modes, terminal, MiniLibX) | 760 chars |
| Dones code | 65 | the exercises, one question per idea (a default, a rule, an error, a decorator...) | 291 chars |

Default `hybrid` retriever, k = 10, chunks of 2,000 characters (408 chunks for the three projects; indexing 0.4 s, each `search_dataset` 0.3 s):

| Dataset | Recall@1 | Recall@3 | **Recall@5** | Recall@10 | Found within k = 10 |
|---|---|---|---|---|---|
| Fly-in docs | 0.538 | 0.692 | **0.769** | 0.872 | 34 / 39 |
| Fly-in code | 0.472 | 0.679 | **0.736** | 0.792 | 42 / 53 |
| A-Maze-ing docs | 0.240 | 0.560 | **0.760** | 0.840 | 21 / 25 |
| A-Maze-ing code | 0.395 | 0.658 | **0.711** | 0.842 | 32 / 38 |
| Dones code | 0.623 | 0.931 | **0.931** | 0.954 | 62 / 65 |
| **All 220 questions** | | | **≈ 0.80** | **≈ 0.88** | 191 / 220 |

- **The system generalises.** Without changing anything, recall@5 is 71–93 % and about 80 % overall, against 91 % / 83 % on vLLM. The drop of about ten points is an honest measure of how much the parameters (`K1`, `B`, `CANDIDATES`, `ALPHA`, the stopwords) are fitted to vLLM. Every code dataset is well above the 50 % the subject asks for vLLM; the docs datasets stay just below its 80 %.
- **Other projects in the index barely hurt.** Fly-in alone gave 0.744 (docs) and 0.755 (code) at recall@5; with A-Maze-ing and Dones in the same index it gives 0.769 and 0.736. The comparison is not exact (two Fly-in files are gone since), but the difference stays within ±2.5 points.
- **The right source is found more often than it is ranked first.** Recall@10 is far above recall@1 everywhere (A-Maze-ing docs: 0.24 → 0.84). The retriever reaches the right region but often not at the top. For answering this matters less: the generator receives all ten sources.
- **Each project fails for its own reason.** Dones is the easiest (0.931): small files, often a single chunk, and distinctive names (`memory_vault`, `spell_reducer`) that BM25 matches at once. A-Maze-ing docs has the lowest recall@1 because its two READMEs (the game's and the `mazegen` package's) describe the same things (perfect mazes, the "42" pattern, the solver, the seeds), so the other README often wins: an ambiguity of the corpus, not of the retriever. A-Maze-ing code has long docstrings that dilute the code, and constants repeated across files (`NO = 1, E = 2...` in the generator and in the drawer).
- **Read with care.** The datasets are small (with 25 questions, one question is 4 points), and they were written by someone who knows both the code and how the system searches, mixing questions that quote identifiers and questions that paraphrase. Questions written by someone else would be a harder test.

**Chunk size, measured earlier on Fly-in alone** (96 files at the time, 202 chunks at 2,000 characters):

| Chunk size | Chunks | Dataset | Recall@1 | Recall@3 | **Recall@5** | Recall@10 | Found within k = 10 |
|---|---|---|---|---|---|---|---|
| **2000** (default) | 202 | docs | 0.487 | 0.692 | **0.744** | **0.897** | 35 / 39 |
| | | code | 0.491 | 0.660 | **0.755** | **0.868** | 46 / 53 |
| 500 | 599 | docs | **0.538** | 0.692 | **0.769** | 0.821 | 32 / 39 |
| | | code | 0.491 | 0.660 | **0.792** | 0.849 | 45 / 53 |
| 300 | 957 | docs | 0.487 | 0.692 | **0.795** | 0.795 | 31 / 39 |
| | | code | 0.491 | 0.642 | **0.736** | 0.830 | 44 / 53 |

Indexing took 2.1 s (2,000), 0.8 s (500) and 1.1 s (300); each `search_dataset` 0.6 s.

- **Short sources make it harder.** The Fly-in sources are much shorter than vLLM's (median 254 / 425 characters against 878 in the vLLM code dataset), which makes the IoU 0.05 overlap harder to reach with a long chunk.
- **Chunk size confirms the trade-off seen on vLLM** (see *Chunk size* below). With 500-character chunks, the top results are more precise (recall@5 +2.5 docs, +3.7 code; recall@1 +5.1 docs), because a short answer is a bigger share of a short chunk. But recall@10 drops (−7.6 docs, −1.9 code): there are three times more chunks competing for the same 10 places, and some answers are split between two chunks.
- **At 300 characters, docs and code part ways.** Docs reach their best recall@5 (0.795), since the map comments and README bullets they ask about fit in a 300-character chunk. But their recall@10 is the worst (0.795, the same as @5): what is not in the top 5 is not found at all, because the 957 chunks crowd it out. Code gets worse at every k above 1 (@5 0.736, below even 2,000): most functions are longer than 300 characters, so the chunk that contains the words of the question is often not the one that contains the answer. Smaller is not better for code.
- **2,000 stays the default.** It gives the best recall@10 on both datasets here, it is the best on vLLM, which the moulinette scores, and it gives the generator the most context per chunk.

### Evolution of recall@5

| Change | docs | code |
|---|---|---|
| Baseline: fixed-size chunks, plain tokenizer, `K1 = 1.5`, `B = 0.75` | 0.850 | 0.596 |
| `PyChunker`, file path and snake_case terms, `K1 = 1.9`, `B = 0.5` | 0.830 | 0.727 |
| `MdChunker` with the code-fence separator | 0.860 | 0.727 |
| Dotted-name pass removed, `.txt` with `MdChunker`, decorators kept with their `def` | 0.860 | 0.717 |
| Query stopwords (**final BM25**) | 0.870 | 0.758 |
| Hybrid: BM25 top 30 re-ranked with `potion-base-8M` windows (α = 0.5) | **0.910** | **0.828** |

Each change was measured on its own against the previous configuration, and kept only when it helped. The main experiments follow.

### Chunk size

*Baseline configuration.*

| Chunk size | docs @5 | code @5 |
|---|---|---|
| **2000** | **0.850** | 0.596 |
| 1000 | 0.830 | **0.606** |
| 500 | 0.800 | 0.505 |

- Small chunks are more precise, but they split answers.
- Big chunks contain the answer more often, but can fall below IoU 0.05 when the answer is short: a 2,000-character chunk that contains a 40-character answer has an IoU of 0.02.

2,000 was kept: it is the best on docs, and it gives the generator the most context per chunk.

### Ablation of the indexed terms

*Chunks of 2,000, k = 10, recall@5. Each row changes one thing.*

| Configuration | docs @5 | code @5 | Effect of that change |
|---|---|---|---|
| With path, snake_case and dotted passes | 0.830 | 0.727 | |
| − file path | 0.870 | 0.636 | path: **code +9**, docs −4 |
| − snake_case pass | 0.820 | 0.657 | snake_case: **code +7** |
| − dotted pass | 0.830 | 0.717 | neutral → removed |
| `K1/B = 1.5/0.75` | 0.820 | 0.727 | neutral |

The path hurts docs because code chunks take top-5 slots from the documentation: many code paths contain words from docs questions (`vllm/lora/…`, `…/quantization/…`). The code-fence Markdown chunker recovered that loss.

### Markdown chunking variants

*Measured with only the file name in the indexed terms.*

| `.md` chunker | docs @5 |
|---|---|
| Plain fixed-size cuts | 0.860 |
| Heading separators (`#` > `##` > `###`) | 0.820 |
| One chunk per section | 0.820 |
| Headings + code-fence separator | 0.830 |
| **Code-fence separator only (implemented)** | **0.880** |

The questions lost with headings were mostly "right file, IoU < 0.05" cases: short sources whose chunk moved, sometimes to just below the threshold (IoU 0.046, 0.048).

### Query stopwords

| Query filter | docs @5 | code @5 |
|---|---|---|
| None | 0.86 | 0.717 |
| Question words only (17) | 0.86 | 0.747 |
| **Question + common words (~45, implemented)** | **0.87** | **0.758** |
| NLTK English list (~150) | 0.86 | 0.758 |

A longer list does not help. A different list per dataset was deliberately not tried, because the search does not know the type of the question.

### Discarded: chunk overlap

| Overlap | docs @5 | code @5 |
|---|---|---|
| 0 (implemented) | **0.86** | 0.717 |
| 200, all files | 0.83 | 0.737 |
| 400, all files | 0.80 | 0.747 |
| 200, `.py` only | 0.85 | 0.758 |
| 600, `.py` only | 0.85 | 0.727 |

On all files, two almost identical neighbouring chunks take two of the five slots. On `.py` files only, the gain does not grow with the overlap, so it is close to noise.

### Semantic embeddings: choosing the model

Five lightweight CPU models were measured on the real corpus: the time to embed it, and the recall with vectors alone (whole chunks).

| Model | Kind | Size | Whole corpus | docs @5 | code @5 |
|---|---|---|---|---|---|
| `all-MiniLM-L6-v2`, 256 tokens | transformer | 88 MB | ~400 s (~20 min on campus) | 0.510 | 0.374 |
| `all-MiniLM-L6-v2`, 128 tokens | transformer | 88 MB | 344 s | 0.460 | 0.333 |
| **`minishlab/potion-base-8M`** | static | **30 MB** | **~2 s** | 0.440 | 0.343 |
| `static-retrieval-mrl-en-v1` | static | 121 MB | 5.2 s | 0.350 | 0.293 |
| `minishlab/potion-retrieval-32M` | static | 125 MB | 2.1 s | 0.340 | 0.263 |

- MiniLM does not fit in the 5-minute indexing limit on a campus computer.
- `potion-base-8M` retrieves about as well as MiniLM cut to 128 tokens, and embeds the corpus ~150× faster, so the vector index can be built inside `index`.
- With whole chunks, **no model beat BM25, alone or fused with it** by reciprocal rank fusion. The tuning below fixed that.

### Semantic embeddings: tuning `potion-base-8M`

A static model represents a text by the **average** of its token vectors, so over a 2,000-character chunk the few words that answer a question are diluted.

| Vectors only | docs @5 | code @5 |
|---|---|---|
| Whole chunks | 0.440 | 0.343 |
| + windows of ~400 characters, best window per chunk | 0.700 | 0.323 |
| + file path in front of every window | **0.730** | **0.596** |
| … with the query stopwords removed | 0.720 | 0.576 |

Windows help most on the docs and the path helps most on the code. Removing the stopwords hurts the vectors, the opposite of BM25: for an embedding, question words carry part of the meaning.

**Combining with BM25.** Reciprocal rank fusion of both lists never improved the code questions: its best setting (vector weight 0.1) gave 0.900 / 0.758, and higher weights lowered both. **Re-ranking the BM25 candidates** improved both. Sweep of the number of candidates N and of α, recall@5 as `docs / code` (BM25 alone: 0.870 / 0.758):

| N \ α | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 |
|---|---|---|---|---|---|---|
| 10 | 0.920 / 0.788 | 0.920 / 0.798 | 0.920 / 0.798 | 0.920 / 0.778 | 0.910 / 0.788 | 0.910 / 0.778 |
| 20 | 0.910 / 0.788 | 0.910 / 0.808 | 0.910 / 0.818 | 0.910 / 0.798 | 0.910 / 0.808 | 0.910 / 0.778 |
| **30** | 0.910 / 0.798 | 0.910 / 0.818 | **0.910 / 0.828** | 0.910 / 0.808 | 0.910 / 0.818 | 0.910 / 0.778 |
| 50 | 0.910 / 0.798 | 0.910 / 0.818 | 0.910 / 0.828 | 0.910 / 0.808 | 0.910 / 0.818 | 0.910 / 0.778 |

- **+4 points on docs and +7 on code** at recall@5, for ~4 s more indexing and a few milliseconds per question.
- **Not a lucky cell.** Every configuration with 20–50 candidates and α between 0.3 and 0.8 beats BM25. The chosen one is the centre of that plateau, not its best cell.
- **Caveat.** The parameters were tuned on the same public datasets they are measured on. The plateau makes overfitting unlikely, but the private datasets may show a smaller gain. BM25 alone already meets both targets.

### Caching (bonus 4)

**Cold start.** `sentence-transformers` imports PyTorch, and that took ~6.5 s of a ~7.3 s single search. Because the model is just a lookup table and a mean, `StaticEncoder` runs it with numpy, and `index` saves it next to the vectors:

| | With `sentence-transformers` | With `StaticEncoder` |
|---|---|---|
| Load the embedding model | ~6.5 s | **~0.3 s** |
| `index` | 28.2 s | **13.2 s** |
| 199 questions (two processes) | 19.4 s | **5.4 s** |
| Recall@1/3/5/10 | | identical |

Removing the Python loop that averages the token vectors was measured and discarded: `np.add.reduceat` took 13–16 s against 1.6 s for the loop. The work is in the additions, not in the Python iterations.

**Repeated queries.** Measured on the development PC (faster than a campus computer), with the real index:

| | No cache (first run) | Cached |
|---|---|---|
| `search_dataset`, 100 docs questions, `hybrid` | 1.94 s | **0.32 s** |
| `search_dataset`, 100 docs questions, `bm25` | 0.94 s | **0.33 s** |
| `make test_retrieval`: 199 questions, two processes | 4.4 s | **0.7 s** |
| `search`, one question | 0.78 s | **0.33 s** |
| Output and recall@1/3/5/10 | | identical (`diff` of the JSON files) |

What is left in a cached run is starting Python and reading the cache file (~3.7 MB for the 100 docs questions). `make test_limits` still measures the cold case, because `index` removes the caches first. `make test_limits_cached` shows both cases one after the other.

**Repeated answers.** The search is milliseconds; the cost of an answer is loading PyTorch and Qwen (~6 s) and generating (~15 s). Generation is deterministic (greedy decoding), so a cached answer is exactly the one that would be generated. The cache saves the answer by a key made of the question and the path, range **and text** of each context chunk, in order, and checking it never imports PyTorch. Measured with `make test_answer`, the 10-question battery, which runs `answer` once per question (a new process each time), on a campus computer:

| `make test_answer` (10 questions) | No cache | Cached |
|---|---|---|
| One question | 14.2–22.2 s (~17 s): load PyTorch and Qwen, then generate | **0.3 s**: search, read the answer, no PyTorch |
| Whole battery | 172 s | **3 s** |
| Answers | | identical |

`make test_answer_dataset` runs `search_dataset` and then `answer_dataset`, so the model is loaded once for the whole dataset. On the 100 docs questions:

| `make test_answer_dataset` (100 docs questions) | No cache | Cached |
|---|---|---|
| `search_dataset` | 2.1 s | 0.3 s (query cache) |
| `answer_dataset` | 1,498 s (~25 min): Qwen loaded once, **~15 s per answer** | **0.3 s**: `100 of 100 answers read from the cache`, no PyTorch |
| Whole rule | ~25 min | **1 s** |

Before, the rule ran `answer` once per question, which loaded PyTorch and Qwen again for every question: ~20 s per question, ~35 min for the dataset.

Answers already generated by another command are reused too: `make test_answer_dataset` stopped after 8 questions and run again returned those 8 at once from the cache, and only loaded Qwen for the 9th. `answer_dataset` saves the cache every 10 generated answers and when it stops, also on Ctrl+C or an error, so an interrupted run keeps every answer it generated (a killed process, `kill -9`, keeps those up to the last save). A run stopped with Ctrl+C after 2 answers, then started again:

```console
$ make test_answer_dataset
...
0 of 100 answers read from the cache
Loading PyTorch...
Loading Qwen/Qwen3-0.6B...
Generating answers:   2%|▏         | 2/100 [00:24<19:52, 12.17s/question]^C
$ make test_answer_dataset
...
2 of 100 answers read from the cache
Loading PyTorch...
Loading Qwen/Qwen3-0.6B...
Generating answers:   0%|          | 0/98 [00:00<?, ?question/s]
```

The second run loads the model once and only generates the 98 answers that are missing.

### Incremental indexing (bonus 3)

Before the change, `index` spent ~8.1 s on campus. Reading and chunking took 5 % of it; tokenizing for BM25 took 30 % and the embeddings 50 %. So both indexes are updated, not only the vectors: rebuilding BM25 from every chunk would still tokenize the whole corpus.

Measured on the real corpus (1,969 files, 12,907 chunks), on one computer, the whole command:

| Run | Time | Chunks tokenized and embedded |
|---|---|---|
| Full (`--full`, or first run) | ~4.6 s | 12,907 |
| Nothing changed | **~0.8 s** | 0, and nothing is written |
| 1 file changed | **~1.35 s** | only its chunks (7 for `README.md`) |
| 50 files changed | ~1.35 s | |
| 500 files changed | ~2.0 s | |
| Index and recall@1/3/5/10 | | identical to a full index |

What is left with one changed file is starting Python, loading the old index and saving the new one, which grow with the corpus and not with the change. On the real corpus, after touching 140 files, the incremental index and a full one have the same chunks, postings, lengths, owners and vectors (compared with `==` and `np.array_equal`).

### Generation

Answers cannot be scored automatically, so they were judged by hand. Score: ✅ 2, 🟡 1, ⚪ refusal 0, ❌ −1.

**Battery of 10 questions** (`make test_answer`): 4 docs questions, 4 code questions, 1 off-topic question and 1 trap about a flag that does not exist. Every prompt change was run on the whole battery.

| Prompt version | Score | Confident wrong answers |
|---|---|---|
| Short prompt: "quote the exact names and values" | 7 | 4 |
| + off-topic rule | — | refused valid questions → dropped |
| + few-shot examples | 11 | 1 |
| + new reminder, budget 2,500 | 16 | 0 |
| **+ budget 2,200 (final)** | **17** | **0** |

- **Context budget.** Budgets of 1,500, 2,000, 2,200, 2,500, 2,800 and 3,500 tokens were tried. The answers only change when the *number of chunks* changes. More chunks means more distractors: with 3,500 the model copied values from the wrong chunk.
- **Sampling** refused more valid questions (4 → 6). **Thinking mode** was 3–7× slower (65–150 s per answer) and turned an honest refusal into a confident wrong answer. Both were discarded.

**Full docs dataset.** All 100 questions of `dataset_docs_public.json`, run with `answer_dataset` (BM25 retrieval; the hybrid retriever was added later). Each answer was compared with the reference answer:

| Result | Count | Correct source retrieved (92) | Correct source not retrieved (8) |
|---|---|---|---|
| ✅ correct | **54** | 54 | 0 |
| 🟡 partial or vague | 22 | 21 | 1 |
| ⚪ refusal | 11 | 7 | 4 (honest: the data was missing) |
| ❌ wrong | 13 | 10 | 3 |

- **76 %** of the answers are correct or partial, and **13 %** are wrong (score 117 / 200). An earlier run of the same prompt, with one `answer` call per question, scored 121; the gap is within the uncertainty of judging by hand.
- A correct source was retrieved for 92 of the 100 questions, so most of the loss is in generation. The main failures are over-refusal on "where can I find…" questions, values copied from a nearby but wrong chunk, and answers off the point. These are the "known reasoning limits" of a 0.6B model that the subject mentions.
- When the correct source was not retrieved, the model did not make up an answer in 5 of 8 cases.

### Tests

`make test` runs **about 1,370 pytest tests** (19 files) in under a minute: 1,376 with the vLLM corpus and the public datasets in `data/`, since the dataset checks are generated once per `.json` found. Without that data (a fresh clone), the tests that need it are skipped and name what is missing. It never loads Qwen or the embedding model, and never writes into the project's `data/output`: `tests/conftest.py` replaces the embedding model with a fake bag-of-words encoder in every test, and sends the default output folders to a temporary one.

- **Chunkers:** exact offsets, no gaps or overlaps, never above the maximum, always progressing. These are checked on random texts and on the whole real corpus. Separator priorities, whole code blocks and decorators are tested too.
- **BM25:** scores compared with a naive reference implementation of the formula, plus tie-breaking, query stopwords and the save/load round trip.
- **Vector index and hybrid retriever:** windows, path prefix, best window per chunk, save/load and broken files. The combination formula is checked on fixed scores: `ALPHA = 0` gives exactly the BM25 order, `ALPHA = 1` the similarity order, and only BM25 candidates come out.
- **`CachedRetriever`:** with a stand-in retriever that counts how often it is built and searched. Covers hits and misses, the wrapped retriever never built when everything is cached, another `k` as another entry, copies returned, the round trip to disk, another namespace, broken, empty or non-UTF-8 files, an unwritable folder, a failed final move that keeps the old file, no rewrite when nothing is new, and the namespace changing with the retriever, its parameters and the index files. Through the CLI: a repeated `search` loads no index, `--cache=False` loads every time, and `index` removes the caches.
- **`StaticEncoder`:** a tiny real model built inside the test (5-word tokenizer, table of known vectors). Covers the mean and normalisation computed by hand, unknown words, empty texts (zeros, never NaN), and loading from the cache, with a download, or with neither.
- **Incremental indexing:** the change detection (`_diff`) and the plan (`_plan`) on dictionaries and chunks built in memory, with new files that shift the ids, modified files with more or fewer chunks, deletions and everything at once. `BM25Index.update` and `VectorIndex.update` are compared with `build` (also on the real corpus for BM25), and spies check that only the new chunks are tokenized and embedded. `run` on real files: the first run is full, a second one writes nothing, new, modified and deleted files at once give the same index as a full run, only the changed files are read, and a missing or broken manifest, another chunk size, a missing index file or `--full` give a full index. A failed save leaves no manifest.
- **Evaluator:** the same numbers as the moulinette.
- **HTTP API:** with FastAPI's `TestClient`, a small BM25 index and a fake model. Covers `/health`, the JSON of `/search` and `/answer`, 422 for every invalid parameter (including a query of only spaces), 503 with the CLI's message for every `RagError` (a missing index, a model that does not fit), an index built while the server runs and used without a restart, no model load without sources or for `/search`, a single load with the server's `dtype` over several answers, 503 when PyTorch fails (and `/search` still answering), parallel searches that never overlap inside the retriever, and the query cache saved after each request. With the answer cache: a repeated question generated once, reused by a server started later, no model load on a hit (even without PyTorch), and four identical requests at once generating once while `/search` keeps answering. `POST /index`: up to date, a changed file found by the next search without a restart, `full`, the answer cache reset, 405 for `GET`, 422 and 503. `?retriever=`: the other retriever loaded on demand, one shared BM25 index, the same results as the CLI. PyTorch is never imported: the tests replace `src.generator` before the endpoint imports it.
- **`AnswerCache`:** the key changes with the question, the order, text and range of the chunks; the namespace changes with `generator.py`; round trip, broken, foreign or non-UTF-8 files as an empty cache, no rewrite without changes, an unwritable folder, a failed move that keeps the old file, and, in a fresh interpreter, that using it never imports `src.generator` or `torch`. Through the CLI with a fake generator: `answer` twice loads the model once, `--cache=False`, a changed `generator.py`, no sources, the answer kept when writing the output fails; `answer_dataset` with everything cached loads nothing, half cached generates only the missing questions in order, an edited source file regenerates the questions that used it, the cache is saved every few answers, and a run stopped with Ctrl+C keeps what it generated.
- **CLI:** every command, argument validation, error messages, unknown or private options, exit codes, `--help`, and source paths outside the project. The dataset commands run without options: from an empty folder, a missing default file is an argument error that names the option and the path, and `evaluate` with no option reads both default files.
- **Mini corpus** in `tests/data`, with easy and hard questions, also run end to end by `make test_rag`.

---

## Design decisions

- **BM25 written from scratch** instead of using a library. The inverted index, idf, saturation and tie-breaking are explicit and tested against the formula.
- **JSON index validated with pydantic**, not pickle. Loading a pickle can run code. The JSON index can be inspected, and a corrupted or foreign file gives a clear error instead of arbitrary behaviour.
- **Two chunkers on one cutting algorithm.** `PyChunker` and `MdChunker` only declare their separators, so the invariants are written and tested once.
- **One change at a time.** Every idea was measured alone: path terms, tokenizer passes, heading cuts, overlap, stopwords, embedding models, fusion methods, context budget and prompt wording. Heading cuts, overlap and list fusion sounded right, and the numbers discarded them.
- **Vectors re-rank BM25 instead of being fused with it.** Fusing the two lists let the vectors bring in chunks with none of the question's words, and lost to BM25 alone. Re-ranking the BM25 top 30 only reorders good candidates.
- **A static embedding model with windows** instead of a transformer like MiniLM. It fits the campus time limit with plenty of room, and windows with max-pooling recover what averaging a whole chunk loses.
- **No deep-learning library for the embeddings.** `potion-base-8M` is a lookup table, so numpy gives the same vectors without PyTorch. The model is saved inside the index folder, so a search depends only on that folder: no network and no Hugging Face cache.
- **One abstract `Retriever`.** `BM25Index`, `HybridRetriever` and `CachedRetriever` share `search(query, k)`. The commands pick one in a single place (`_load_retriever`) and never know which one they got. `VectorIndex` does not depend on BM25: `HybridRetriever` combines them.
- **The query cache is a decorator, not a change to the retrievers.** `CachedRetriever` is a `Retriever` that wraps any other one. It receives a function that builds the wrapped retriever instead of the retriever itself, so nothing is loaded until a question is not cached.
- **A cached result is only reused for the same index and configuration.** Each cache file carries a namespace: the retriever, its parameters (`K1`, `B`, `CANDIDATES`, `ALPHA`, the model) and the size and modification time of the index files. Rebuilding the index or changing a parameter gives a new namespace, and the old results are ignored. Only file metadata is read, so it is instant.
- **The cache never breaks a command.** A missing, broken or foreign cache file means an empty cache, and a cache that cannot be written only prints a warning. It is written to a temporary file and moved over the old one (`os.replace`), so an interrupted write never leaves a half-written file. JSON validated with pydantic, not pickle, as for the index.
- **The answer cache key is everything that goes into the prompt.** The question and the path, range and text of each context chunk, in order, hashed with sha256 (the text would make a key of ~20 KB). With the text in the key, an edited file can never return a stale answer, even with an old index. `index` also removes the answer cache whenever the index changes.
- **`index --full` is a reset of everything, caches included.** It removes the query and answer caches on purpose: it means starting from scratch, so everything in the index folder comes from that run. The caches never need it to stay correct: the query cache namespace holds the size and modification time of the index files, which a rewrite changes, and the answer cache key holds the text of the chunks. To keep the answers already generated, run `index` without `--full` (or `FULL=0` in the Makefile): if nothing changed, it writes nothing and keeps both caches.
- **The answer cache namespace is the text of `generator.py`.** The model, the prompt, the token budget and the decoding all live there, so changing any of them makes the old answers stale on their own. The file is read as text and hashed, never imported, so a cached answer never loads PyTorch. Its text, not its modification time: a `git checkout` that does not change it keeps the cache.
- **`dtype` is not in the answer cache key.** `--dtype auto`, the default, picks float32 or bfloat16 from the available memory, and both give practically the same answer. Keeping them apart would only split the cache.
- **Exact-match cache, not a semantic one.** Reusing the results of a *similar* question (embedding similarity above a threshold) was considered and discarded. With `potion-base-8M`, *"What is the minimum version of bitsandbytes?"* and *"… maximum version …"* have a similarity of 0.904, the same as a real paraphrase (*"How do I load a LoRA adapter?"* / *"… in vLLM?"*), and *load* / *unload* (0.847) share only 1 of their 5 BM25 results. Because the model averages word vectors, changing one word barely moves the vector, so no threshold separates paraphrases from opposite questions. A semantic cache would also make the output depend on earlier questions, and it would have to load the model on every command to save a ~7 ms search.
- **Incremental indexing on both indexes, with one shared plan.** A chunk id is the chunk's position, and BM25's postings and lengths and the vector owners all store it. Removing a chunk shifts every later id, which would give wrong results with no error. `IncrementalIndexer._plan` computes the map `{old id: new id}` once and gives the same one to `BM25Index.update` and `VectorIndex.update`, so the two indexes cannot drift apart.
- **The incremental index equals the full one.** Chunks are always sorted by file and position (`Chunk.order_key`), in the full index too, and a chunk is tokenized (`_chunk_tokens`) and embedded (`_embed`) by the same method in `build` and `update`. So the best test is simple: an updated index must be identical to a full index of the same corpus.
- **Changes found by `stat`, before reading.** Size and modification time in nanoseconds (`st_mtime_ns`), as for the query cache. Files are only opened if they are new or modified.
- **`manifest.json` is removed first and written last.** If `index` stops halfway, no manifest describes indexes that were only partly replaced, and the next run is a full index. A missing, broken or foreign manifest, or a missing index file, means a full index, never an error. A different `--raw_dir` or `--max_chunk_size` also gives a full one, because no saved chunk would be valid.
- **The lexical method is always available.** `--retriever bm25` runs the mandatory method alone and never loads the vector index.
- **Stopwords on the query only**, so the list can change without rebuilding the index.
- **Default paths, never hard-coded ones.** Every input and output path is a CLI option, as the subject requires. The dataset commands default to the public docs dataset and are chained (`search_dataset` writes where `answer_dataset` and `evaluate` read), so the whole pipeline runs without flags. pydantic does not validate defaults unless asked, so `CliCommand` sets `validate_default=True`: a default path arrives as a `Path`, and a missing default file is a clear argument error instead of a crash inside the command.
- **Other corpora to check generalisation.** Every parameter was tuned on the vLLM public datasets, which could hide overfitting. `make test_corpus` runs the same pipeline, unchanged, on three other projects indexed together, with five handwritten datasets whose source ranges were computed by a script from exact text anchors, never counted by hand.
- **Paths relative to the project root**, written exactly as in the corpus (`data/raw/vllm-0.10.1/...`), because the grader compares them verbatim.
- **FastAPI for the HTTP API**, not the standard library's `http.server`. The project already describes every input and output with pydantic models, and FastAPI validates the requests and serialises the responses with those same models, so the API returns exactly the CLI's JSON without building it by hand. Routing, validation (422) and the `/docs` page come with the framework.
- **The API keeps everything loaded between requests.** The index is loaded once (at start-up, or by the first search that is not cached when the cache is on) and the model on the first `/answer` that is not cached, so later requests skip the cold start that each CLI command pays.
- **Two locks (mutexes) in the API.** FastAPI runs each request in its own thread, so two requests can arrive at once. One lock lets a single search (with its cache write) run at a time, because two writers would corrupt the query cache. The other loads the model once and generates one answer at a time, because two generations at once would double the memory. They are never held together, so a `/search` does not wait for an answer being generated, and they cannot deadlock. A third lock guards the answer cache; it is only taken alone or inside one of the other two, never the other way round. A request that waited for the model checks the answer cache again before generating, so identical requests at the same time generate once.
- **Lazy import and single load of Qwen.** Only `answer` and `answer_dataset` pay for PyTorch, and only once per command.
- **float32 on CPU when it fits.** It reads the prompt ~3× faster than bfloat16. The type is chosen before loading, so a smaller machine gets a slower answer instead of a killed process.
- **A prompt for a small model.** Few positive instructions, examples instead of rules, the question after the context, a fixed refusal sentence that is easy to detect, and greedy decoding so answers are reproducible.
- **No traceback, ever.** Arguments are validated by pydantic before a command runs, and the command line is checked before Fire runs it. Every expected failure is a `RagError` with a message and an exit code: 1 for errors, 2 for usage, 130 for Ctrl+C.
- **Quality gates.** Object-oriented code, type hints everywhere, docstrings on every module, class and function, and `flake8` and `mypy --strict` clean.

---

## Challenges faced

| Challenge | Solution |
|---|---|
| **Fire is permissive.** It ignores type hints, runs a command before rejecting unknown options or a late `--help`, and exposes private methods by name. | `CliCommand.validated` validates every argument with pydantic, and `ArgvGuard` checks the command line against the public commands before Fire runs anything. |
| **Questions and code rarely share words.** | snake_case names are kept whole, the file path is added to the indexed terms, and question words are removed from the query. Code recall@5 went from 0.596 to 0.758 with BM25 alone. |
| **A fix that helps one dataset hurts the other.** The file path gave +9 on code and −4 on docs. | Found with a one-change-at-a-time ablation, and compensated by the code-fence Markdown chunker. |
| **Heading-based Markdown chunking was worse**, even though 79 % of the docs sources start at a heading. | Traced to short sources falling just below IoU 0.05. The code-fence separator gave the best docs recall instead. |
| **Decorators were separated from their function**, because the minimum-fill rule rejected the decorator. | The cut is moved back over the decorator lines, without ever going before the start of the chunk. |
| **Rankings changed between runs.** Equal scores came out in `set` iteration order. | Ranking on `(score, chunk id)`. |
| **The best CPU transformer was too slow**: MiniLM needed ~20 min on campus for the corpus. | A static model, `potion-base-8M`, which embeds the corpus in seconds. |
| **Embeddings lowered recall.** Every model, fused with BM25, scored below BM25 alone. | Short windows with the best window per chunk, the file path in each window, and re-ranking instead of fusion: +4 docs, +7 code. |
| **Cold start.** Importing `sentence-transformers` (PyTorch) took ~6.5 s per command. | The model runs with numpy and is saved with the index. It loads in 0.3 s, with identical vectors and recall. |
| **Slow generation**: 30–40 s per answer. | Profiling showed that bfloat16 is emulated on this CPU, so float32 is used when it fits. |
| **Running out of memory cannot be caught**: the kernel kills the process. | The available memory is checked before loading, which leads to float32, bfloat16 or a clean error. |
| **Evasive answers even with the right context.** | An *oracle test* (only the correct chunk in the context) separated retrieval failures from generation failures. A shorter prompt that asks for exact values fixed them. |
| **"Paris" answered from the model's own knowledge.** An explicit off-topic rule fixed it, but made the model refuse valid questions. | Replaced by a few-shot refused example, so off-topic questions are refused and valid ones are answered. |
| **The prompt leaked into the answers** ("The exact values are: …", the example's `(ServerConfig)`). | A neutral reminder, an example answer with nothing easy to copy at the end, and removal of a refusal sentence that follows a real answer. |
| **A run hung for minutes** on a request to the Hugging Face Hub, even though the model was already cached. | Models are loaded from the cache first, with `local_files_only=True`. |
| **Switching retrievers erased the other one's cache**: with a single cache file, a `bm25` search overwrote the 100 `hybrid` results. | One cache file per retriever (`query_cache_bm25.json`, `query_cache_hybrid.json`). |

---

## Known limitations

- **The re-ranker only reorders.** A chunk that BM25 does not place in its top 30 can never be returned, however close its meaning is.
- **Tuned on the public datasets.** The hybrid parameters come from a stable plateau, but the private datasets may show a smaller gain.
- **A 0.6B model.** It sometimes picks a value from a similar but wrong chunk, and it is sensitive to the exact mix of chunks in its context. The full-dataset answer evaluation was done with BM25 retrieval; with the hybrid retriever, a different set of chunks reaches the model.
- **Grounded is not always correct.** In one case the model quoted vLLM's own assertion message, which contradicts the condition next to it (`vllm/v1/core/kv_cache_coordinator.py`).
- **One answer at a time in the API.** Requests to `/answer` are served one after another, so a second user waits for the first answer. That is enough for a local server, and running them in parallel on a CPU would not be faster.
- **Incremental indexing trusts the stamps.** A change to the chunker or tokenizer code, or a file edited while keeping the same size and modification time, is not detected: `index --full` rebuilds everything. Loading and saving the indexes still grow with the corpus, so even one changed file costs about a second.
- **Lower recall on unseen code.** On the three other projects, recall@5 is 71–93 % (about 80 % overall) against 83–91 % on vLLM: the parameters were tuned on vLLM, and the docs datasets stay below the 80 % the subject asks for vLLM.
- **The query cache never forgets.** Entries are only dropped when the index or the configuration changes, so the cache file grows with the number of distinct questions.
- **No authentication in the API.** It listens on `127.0.0.1` by default; `--host 0.0.0.0` exposes it to the network as it is.
- **A forced `--dtype` can reuse an answer of the other type.** The answer cache ignores `dtype` (see the design decisions). With `--dtype auto` this never matters; forcing another type and wanting a fresh answer needs `--cache=False`.
- **Memory check in containers.** The check reads the memory of the whole machine (`/proc/meminfo`). Inside a container or cgroup with its own limit, it can overestimate what is available.

---

## Resources

### References

- P. Lewis et al., [*Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*](https://arxiv.org/abs/2005.11401), NeurIPS 2020. The original RAG paper.
- S. Robertson, H. Zaragoza, [*The Probabilistic Relevance Framework: BM25 and Beyond*](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf), 2009. The BM25 formula and its parameters.
- C. Manning, P. Raghavan, H. Schütze, [*Introduction to Information Retrieval*](https://nlp.stanford.edu/IR-book/), Cambridge University Press, 2008. Inverted indexes, tf-idf, evaluation with recall.
- G. Cormack, C. Clarke, S. Büttcher, [*Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods*](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf), SIGIR 2009. The fusion method compared against re-ranking.
- [Model2Vec](https://github.com/MinishLab/model2vec) and the [`potion-base-8M` model card](https://huggingface.co/minishlab/potion-base-8M). Static embeddings distilled from a sentence transformer.
- N. Liu et al., [*Lost in the Middle: How Language Models Use Long Contexts*](https://arxiv.org/abs/2307.03172), 2023. Why the position of information in the prompt matters.
- [Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B). The chat template, the `enable_thinking` switch and the recommended decoding settings.
- Hugging Face Transformers: [chat templates](https://huggingface.co/docs/transformers/chat_templating) and [text generation](https://huggingface.co/docs/transformers/main_classes/text_generation).
- [uv: using PyTorch with uv](https://docs.astral.sh/uv/guides/integration/pytorch/). CPU-only wheels.
- Documentation of [pydantic](https://docs.pydantic.dev/), [Python Fire](https://github.com/google/python-fire), [tqdm](https://tqdm.github.io/), [NumPy](https://numpy.org/doc/) and [Hugging Face Tokenizers](https://huggingface.co/docs/tokenizers).

### Use of AI

I used an AI assistant throughout the project, as a tutor, a reviewer and a pair programmer:

- **Learning.** Explanations of the subject, RAG, BM25, tokenization, embeddings, PyTorch and transformers, with exercises solved step by step (personal study notes, not part of the repository).
- **Review.** Reviews of my code (bugs found in the reader, the index, the evaluator, the CLI and the answer commands), of my design choices and of the Fire edge cases. It also checked the project against the subject's requirements.
- **Planning of the bonus work.** Step-by-step plans with class skeletons for the vector index, the hybrid retriever, the caching layer and the incremental indexing, which I then implemented and tested.
- **Code, on request:**
  - the CLI hardening (`CliCommand.validated`);
  - the decorator handling of `PyChunker`;
  - the coordination of the incremental index (`IncrementalIndexer.run` and `_save`) and the new `index` output;
  - the HTTP API fixes after my first version of `/answer`: the two locks, the `RagError` → 503 handler, the model kept between requests and the validation of blank queries;
  - the default dataset paths and the `validate_default` setting;
  - the datasets of the other corpora (Fly-in, A-Maze-ing, Dones), generated by a script from exact text anchors, and the `test_corpus` rule;
  - the query stopword list;
  - most of the test suite;
  - several Makefile rules.
- **Experiments.** Scripts for the ablations and measurements reported above (chunk sizes, path terms, Markdown variants, overlap, stopwords, embedding models, fusion and re-ranking sweeps, model types, prompt variants), and the analysis of their results.
- **Documentation.** Drafts of the docstrings and of this README.

I reviewed, ran and understood every part before keeping it, and the design decisions were mine.

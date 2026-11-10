ARGS ?=

# bash for the `time` keyword used by the rules that run the real program
# (test_rag, test_rag_base, test_answer): it prints how long each command
# took, in the format below.
SHELL := /bin/bash
export TIMEFORMAT := Time: %1Rs

# On 42 computers the home is too small for torch and the model, so the
# virtualenv, the uv cache and the Hugging Face cache go to sgoinfre (shared
# by every computer of the campus). Exported for every rule, so `uv run`
# also finds the venv there. Elsewhere (no sgoinfre) nothing changes.
# Variables already set in the environment win. Other folder: STORE=/path
STORE ?= $(if $(USER),$(wildcard /sgoinfre/students/$(USER)))
ifneq ($(STORE),)
export UV_PROJECT_ENVIRONMENT ?= $(STORE)/rag-venv
export UV_CACHE_DIR           ?= $(STORE)/uv-cache
export HF_HOME                ?= $(STORE)/hf
# Hardlinks do not work on NFS.
export UV_LINK_MODE           ?= copy
endif

all: install

install:
	@uv sync
	@echo "Everything is up to date now."
	@if [ -n "$(STORE)" ]; then echo "venv: $(UV_PROJECT_ENVIRONMENT)"; fi

run:
	@uv run -m src $(ARGS)

debug:
	uv run python -m pdb -m src $(ARGS)

# Local HTTP API (bonus 5) until Ctrl+C; try it at http://127.0.0.1:8000/docs
# Options for `serve`: SERVE_ARGS="--port 8080 --retriever bm25"
SERVE_ARGS ?=

serve:
	@uv run -m src serve $(SERVE_ARGS)

# MiConvenio: `make api` in one terminal and `make web` in another.
# A new question spends the free Gemini quota: one embedding and one answer.
# The API is the one that uses GEMINI_API_KEY. NO_KEY=1 starts it without
# the key, to try what someone without one sees.
LABOUR_DATA := data/labour
NO_KEY ?= 0
API_ENV := $(if $(filter 1,$(NO_KEY)),env -u GEMINI_API_KEY)

# Builds the search index of the labour corpus (needed once after cloning,
# and after adding a document). FULL=0 only re-indexes the changed files.
index:
	@uv run -m src index --raw_dir $(LABOUR_DATA)/raw \
		--index_dir $(LABOUR_DATA)/processed $(INDEX_FULL)

api:
	@$(API_ENV) uv run -m src serve --index_dir $(LABOUR_DATA)/processed \
		--raw_dir $(LABOUR_DATA)/raw --retriever gemini --generator gemini

web:
	@uv run streamlit run web/app.py

# Each rule keeps its own mypy cache: the cache is only valid for the
# flags that built it, so sharing one made every switch between lint and
# lint-strict re-check torch, transformers... from scratch (~50 s, not ~2 s).
lint:
	uv run -m flake8 . && uv run -m mypy --cache-dir .mypy_cache/lint \
	--warn-return-any \
	--warn-unused-ignores \
	--ignore-missing-imports \
	--disallow-untyped-defs \
	--check-untyped-defs \
	.

lint-strict:
	uv run -m flake8 . && uv run -m mypy --cache-dir .mypy_cache/strict --strict .

test:
	uv run pytest

# How `index` and test_rag build the index:
#   FULL=1 (default): --full, always the whole corpus; it also removes the
#                     query and answer caches.
#   FULL=0:           incremental, only the files changed since the last
#                     index; nothing is written if none changed.
FULL       ?= 1
INDEX_FULL := $(if $(filter 1,$(FULL)),--full)

# Full pipeline (index -> search_dataset -> evaluate) on the mini corpus in
# tests/data. Generated files go to tests/data/processed and
# tests/data/output (ignored by git, removed by `make clean`).
# Options: TEST_DATASET=dataset_mini_hard.json TEST_K=10 TEST_CHUNK=2000
TEST_DATA    := tests/data
TEST_INDEX   := $(TEST_DATA)/processed
TEST_OUTPUT  := $(TEST_DATA)/output/search_results
TEST_DATASET ?= dataset_mini.json
TEST_K       ?= 5
TEST_CHUNK   ?= 500

test_rag:
	time uv run -m src index --raw_dir $(TEST_DATA)/raw --index_dir $(TEST_INDEX) \
		--max_chunk_size $(TEST_CHUNK) $(INDEX_FULL)
	time uv run -m src search_dataset \
		--dataset_path $(TEST_DATA)/datasets/UnansweredQuestions/$(TEST_DATASET) \
		--k $(TEST_K) --save_directory $(TEST_OUTPUT) --index_dir $(TEST_INDEX)
	time uv run -m src evaluate \
		--student_search_results_path $(TEST_OUTPUT)/$(TEST_DATASET) \
		--dataset_path $(TEST_DATA)/datasets/AnsweredQuestions/$(TEST_DATASET) \
		--k $(TEST_K)

clean:
	@find . -path ./.venv -prune -o -type d -name '__pycache__' -exec rm -rf {} +
	@rm -rf .mypy_cache .pytest_cache $(TEST_INDEX) $(TEST_DATA)/output
	@echo "Cache cleaned"

.PHONY: all install run debug serve index api web lint lint-strict test test_rag clean

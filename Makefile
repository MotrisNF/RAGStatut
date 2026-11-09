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

# How test_rag, test_rag_base and test_index run `index` (bonus 3):
#   FULL=1 (default): --full, always the whole corpus. Measures the
#                     subject's limit and removes the query caches.
#   FULL=0:           incremental, only the files changed since the last
#                     index; nothing is written if none changed. To show
#                     the difference: make test_index, touch a corpus file,
#                     then make test_index FULL=0.
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

# Same pipeline on the real exercise data (data/), mimicking the moulinette:
# k=10 and max_chunk_size=2000. The corpus is indexed once and then
# search_dataset + evaluate run twice, for the docs and the code datasets.
BASE_DATA     := data
BASE_INDEX    := $(BASE_DATA)/processed
BASE_OUTPUT   := $(BASE_DATA)/output/search_results/UnansweredQuestions
BASE_DATASETS := dataset_docs_public.json dataset_code_public.json
BASE_K        := 10
BASE_CHUNK    := 2000

test_rag_base:
	time uv run -m src index --raw_dir $(BASE_DATA)/raw --index_dir $(BASE_INDEX) \
		--max_chunk_size $(BASE_CHUNK) $(INDEX_FULL)
	@for dataset in $(BASE_DATASETS); do \
		echo "===== $$dataset ====="; \
		time uv run -m src search_dataset \
			--dataset_path $(BASE_DATA)/datasets/UnansweredQuestions/$$dataset \
			--k $(BASE_K) --save_directory $(BASE_OUTPUT) \
			--index_dir $(BASE_INDEX) || exit 1; \
		time uv run -m src evaluate \
			--student_search_results_path $(BASE_OUTPUT)/$$dataset \
			--dataset_path $(BASE_DATA)/datasets/AnsweredQuestions/$$dataset \
			--k $(BASE_K) || exit 1; \
	done

# Same pipeline on a second, independent corpus: every project in
# tests/test_corpus/raw, indexed together, with handwritten datasets (exact
# source ranges), each reported apart: Fly-in (a drone-routing simulator)
# with dataset_flyin_docs.json (the READMEs and the map files) and
# dataset_flyin_code.json (src/), A-Maze-ing (a maze generator and game)
# with dataset_amazeing_docs.json (its two READMEs and config files) and
# dataset_amazeing_code.json (mazegen/ and src/), and Dones (Python
# exercises P0-P10) with dataset_dones_code.json. Checks that the system is
# not tuned to vLLM only; each project is also noise for the others'
# questions. Same layout as data/, so every path is an option. The corpus
# is indexed once, then search_dataset + evaluate run for each dataset.
#   CORPUS_DATASETS=... dataset file names (default: all five)
#   CORPUS_K=5          k for search and evaluate (default 10, as the
#                       moulinette)
#   CORPUS_CHUNK=500    max_chunk_size (default 2000)
#   ANSWER=1            also generate the answers with Qwen (answer_dataset:
#                       slow, ~20 s per question, and needs the model's
#                       memory); off by default
CORPUS_DATA     := tests/test_corpus
CORPUS_INDEX    := $(CORPUS_DATA)/processed
CORPUS_OUTPUT   := $(CORPUS_DATA)/output/search_results/UnansweredQuestions
CORPUS_ANSWERS  := $(CORPUS_DATA)/output/search_results_and_answer/UnansweredQuestions
CORPUS_DATASETS ?= dataset_flyin_docs.json dataset_flyin_code.json \
                   dataset_amazeing_docs.json dataset_amazeing_code.json \
                   dataset_dones_code.json
CORPUS_K        ?= 10
CORPUS_CHUNK    ?= 2000
ANSWER          ?= 0

test_corpus:
	time uv run -m src index --raw_dir $(CORPUS_DATA)/raw \
		--index_dir $(CORPUS_INDEX) --max_chunk_size $(CORPUS_CHUNK) $(INDEX_FULL)
	@for dataset in $(CORPUS_DATASETS); do \
		echo "===== $$dataset ====="; \
		time uv run -m src search_dataset \
			--dataset_path $(CORPUS_DATA)/datasets/UnansweredQuestions/$$dataset \
			--k $(CORPUS_K) --save_directory $(CORPUS_OUTPUT) \
			--index_dir $(CORPUS_INDEX) || exit 1; \
		time uv run -m src evaluate \
			--student_search_results_path $(CORPUS_OUTPUT)/$$dataset \
			--dataset_path $(CORPUS_DATA)/datasets/AnsweredQuestions/$$dataset \
			--k $(CORPUS_K) || exit 1; \
		if [ "$(ANSWER)" = 1 ]; then \
			time uv run -m src answer_dataset \
				--student_search_results_path $(CORPUS_OUTPUT)/$$dataset \
				--save_directory $(CORPUS_ANSWERS) \
				--index_dir $(CORPUS_INDEX) || exit 1; \
		fi; \
	done

# Subject time limits (chapter VII), checked on the real data with the
# moulinette settings. Each rule prints the time and PASS or FAIL.
#   test_index:     indexing the whole corpus in at most 5 minutes
#                   (FULL=0 for an incremental index instead, see FULL).
#   test_retrieval: at most 90 s for 200 questions (both public datasets,
#                   199 questions, each search_dataset loading the index).
#   test_limits:    both.
#   test_limits_cached: test_limits (index removes the query cache, so the
#                   searches are uncached; FULL=1 only), then test_retrieval
#                   again without indexing (every question read from the
#                   cache).
INDEX_LIMIT     := 300
PASS            := printf '\033[1;32mPASS\033[0m\n'
FAIL            := printf '\033[1;31mFAIL\033[0m\n'
RETRIEVAL_LIMIT := 90

test_index:
	@start=$$(date +%s%N); \
	uv run -m src index --raw_dir $(BASE_DATA)/raw --index_dir $(BASE_INDEX) \
		--max_chunk_size $(BASE_CHUNK) $(INDEX_FULL) || exit 1; \
	ms=$$(( ($$(date +%s%N) - start) / 1000000 )); \
	echo "===== Indexing ($(if $(INDEX_FULL),full,incremental)):" \
		"$$(LC_ALL=C awk "BEGIN {printf \"%.1f\", $$ms / 1000}") s" \
		"(limit $(INDEX_LIMIT) s)"; \
	if [ $$ms -le $$(( $(INDEX_LIMIT) * 1000 )) ]; then $(PASS); \
	else $(FAIL); exit 1; fi

test_retrieval:
	@questions=0; start=$$(date +%s%N); \
	for dataset in $(BASE_DATASETS); do \
		path=$(BASE_DATA)/datasets/UnansweredQuestions/$$dataset; \
		uv run -m src search_dataset --dataset_path $$path --k $(BASE_K) \
			--save_directory $(BASE_OUTPUT) --index_dir $(BASE_INDEX) \
			|| exit 1; \
		questions=$$(( questions + $$(grep -o '"question_id"' $$path | wc -l) )); \
	done; \
	ms=$$(( ($$(date +%s%N) - start) / 1000000 )); \
	echo "===== Retrieval: $$questions questions in" \
		"$$(LC_ALL=C awk "BEGIN {printf \"%.1f\", $$ms / 1000}") s (limit $(RETRIEVAL_LIMIT) s" \
		"for 200 questions)"; \
	if [ $$ms -le $$(( $(RETRIEVAL_LIMIT) * 1000 )) ]; then $(PASS); \
	else $(FAIL); exit 1; fi

test_limits: test_index test_retrieval

test_limits_cached:
	@echo "===== 1/2: index, then search without cache"
	@$(MAKE) --no-print-directory test_index test_retrieval
	@echo "===== 2/2: search again, no index (query cache)"
	@$(MAKE) --no-print-directory test_retrieval

# Answer battery: runs `answer` on every question of tests/answer_battery
# (one file per question: a QUESTION: line and an EXPECTED: line) and prints
# the expected answer after each one, to compare prompts by eye.
# Needs the index (`make test_rag_base` or `uv run -m src index`).
# Extra options for `answer`: ANSWER_ARGS="--k 5 --dtype bfloat16"
ANSWER_DIR  := tests/answer_battery
ANSWER_ARGS ?=

test_answer:
	@start=$$SECONDS; \
	for file in $(ANSWER_DIR)/*.txt; do \
		question=$$(sed -n 's/^QUESTION: //p' "$$file"); \
		expected=$$(sed -n 's/^EXPECTED: //p' "$$file"); \
		echo "===== $$(basename "$$file" .txt)"; \
		echo "Question: $$question"; \
		time uv run -m src answer "$$question" $(ANSWER_ARGS); \
		echo "Expected: $$expected"; \
		echo; \
	done; \
	echo "Battery total: $$((SECONDS - start)) s"

# Same as test_answer, but on a whole AnsweredQuestions dataset, loading
# the model only once: search_dataset, then answer_dataset (as the
# evaluation runs them), then each question with its answer and its
# reference answer. Answers already in the answer cache (from answer,
# answer_dataset or the API, same question and sources) are not
# generated again, and the model is not loaded if all are cached.
# ~15 s per new question. Options:
#   ANSWER_DATASET=.../dataset_code_public.json   (default: docs)
#   ANSWER_START=77       first question to run, 1-based (default 1)
#   ANSWER_LIMIT=10       only N questions from there (0 = all)
#   ANSWER_SEARCH_ARGS="--k 5 --retriever bm25"   options for search_dataset
#   ANSWER_ARGS="--dtype bfloat16 --cache=False"  options for answer_dataset
#   ANSWER_LOG=path.log   where the output is also saved (default below,
#                         one file per dataset, overwritten on each run)
# The questions, search results and answers of the run are kept in
# ANSWER_WORK.
ANSWER_DATASET     ?= $(BASE_DATA)/datasets/AnsweredQuestions/dataset_docs_public.json
ANSWER_START       ?= 1
ANSWER_LIMIT       ?= 0
ANSWER_SEARCH_ARGS ?=
ANSWER_NAME        := $(basename $(notdir $(ANSWER_DATASET)))
ANSWER_LOG         ?= $(BASE_DATA)/output/answer_logs/$(ANSWER_NAME).log
ANSWER_WORK        ?= $(BASE_DATA)/output/answer_logs/$(ANSWER_NAME)

test_answer_dataset:
	@mkdir -p $(ANSWER_WORK) $(dir $(ANSWER_LOG)); set -o pipefail; ( \
	start=$$SECONDS; \
	uv run python -c 'import json, sys; \
		qs = json.load(open(sys.argv[1]))["rag_questions"]; \
		first = int(sys.argv[3]) - 1; n = int(sys.argv[2]); \
		qs = qs[first:first + n] if n else qs[first:]; \
		json.dump({"rag_questions": qs}, open(sys.argv[4], "w")); \
		print(f"{len(qs)} questions from {sys.argv[1]}")' \
		$(ANSWER_DATASET) $(ANSWER_LIMIT) $(ANSWER_START) \
		$(ANSWER_WORK)/questions.json || exit 1; \
	time uv run -m src search_dataset \
		--dataset_path $(ANSWER_WORK)/questions.json \
		--save_directory $(ANSWER_WORK)/search $(ANSWER_SEARCH_ARGS) \
		< /dev/null || exit 1; \
	time uv run -m src answer_dataset \
		--student_search_results_path $(ANSWER_WORK)/search/questions.json \
		--save_directory $(ANSWER_WORK)/answers $(ANSWER_ARGS) \
		< /dev/null || exit 1; \
	uv run python -c 'import json, sys; \
		expected = {q["question_id"]: q.get("answer", "") for q in \
			json.load(open(sys.argv[1]))["rag_questions"]}; \
		results = json.load(open(sys.argv[2]))["search_results"]; \
		first = int(sys.argv[3]); last = first + len(results) - 1; \
		[print("===== [%d/%d]\nQuestion: %s\nAnswer: %s\nExpected: %s\n" \
			% (n, last, r["question"], r["answer"], \
			expected.get(r["question_id"], ""))) \
		for n, r in enumerate(results, start=first)]' \
		$(ANSWER_WORK)/questions.json \
		$(ANSWER_WORK)/answers/questions.json $(ANSWER_START) || exit 1; \
	echo "Dataset total: $$((SECONDS - start)) s" \
	) 2>&1 | tee $(ANSWER_LOG); \
	echo "Saved to $(ANSWER_LOG)"

clean:
	@find . -path ./.venv -prune -o -type d -name '__pycache__' -exec rm -rf {} +
	@rm -rf .mypy_cache .pytest_cache $(TEST_INDEX) $(TEST_DATA)/output \
		$(CORPUS_INDEX) $(CORPUS_DATA)/output
	@echo "Cache cleaned"

# Removes the index in data/processed and the json files generated in
# data/output, keeping the folder layout (.gitkeep files).
cleandata:
	@if [ -d $(BASE_INDEX) ]; then \
		find $(BASE_INDEX) -mindepth 1 ! -name '.gitkeep' -exec rm -rf {} +; fi
	@if [ -d $(BASE_DATA)/output ]; then \
		find $(BASE_DATA)/output -type f -name '*.json' -delete; fi
	@echo "Index and generated outputs removed"

.PHONY: all install run debug serve lint lint-strict test test_rag test_rag_base test_corpus test_index test_retrieval test_limits test_limits_cached test_answer test_answer_dataset clean cleandata

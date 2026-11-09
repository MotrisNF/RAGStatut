Mini vLLM corpus for quick tests.

It mimics the layout of the real repository: Markdown documentation under
docs/ and Python code under vllm/. Every question of the test datasets is
answered by exactly one passage of these files.

Two datasets use this corpus:
- dataset_mini.json: easy questions that share many words with the answer.
- dataset_mini_hard.json: paraphrases, identifiers, short answers inside long
  sections and passages with similar wording in other files. Each question
  has a "difficulty" field saying which case it tests.

# gastown-finetune - Implementation Plan

**Created:** 2026-02-22
**Status:** Draft
**Source Spec:** plans/gastown-finetune/02-spec/spec.md

---

## Overview

This plan describes how to build a distributed fine-tuning system that trains role-specific LLM agents native to Gas Town, using node0's decentralized training infrastructure. The system encompasses eight major subsystems: a data loading pipeline (the largest engineering effort), LoRA adapter support per pipeline stage, checkpoint gathering and export, autoregressive inference, a corpus collection pipeline, a model registry, an automated evaluation suite, and a RAG layer for dynamic context. Progressive rollout into Gas Town happens by registering a new "node0" provider in town settings, then switching role assignments from deacon outward.

The implementation approach follows the codebase's existing extension patterns throughout. New subsystems live under `src/node0/finetune/` as a top-level sub-package, each as an independent module with a clean interface. Modifications to existing node0 server code are surgical — additive fields on Pydantic models, conditional branches on new CLI flags, optional parameters on existing factory methods. The Gas Town integration layer (corpus hooks, formulas, settings, registry CLI) is built separately and touches Gas Town configuration files, not the node0 Python package directly.

The approach was chosen because node0's forward pass through 32 pipeline stages is already the inference path — it does not need replacement, only a generation loop on top. Similarly, node0's gradient averaging already works for distributed training — it needs data feeding from the outside, LoRA parameter filtering, and a checkpoint export mechanism. Building additions alongside existing code rather than rewriting core infrastructure minimizes risk and preserves the pretraining capability while adding fine-tuning.

---

## Architecture Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| New code location | `src/node0/finetune/` sub-package | Plan-context.md naming convention: new modules go under `finetune/` with `data/`, `lora/`, `checkpoint/`, `inference/`, `corpus/`, `registry/`, `eval/`, `rag/` sub-packages |
| LoRA application point | Inside each Expert `__init__` post base-model init, conditional on `model_args.lora_rank is not None` | Pipeline stage isolation constraint: LoRA must be applied per-stage, not globally; Expert classes (HeadExpert, BodyExpert, TailExpert) are the right boundary |
| Gradient filtering for LoRA | Caller-side: pass only LoRA params to GradientAverager | `GradientAverager.__init__` takes `parameters: Iterable[torch.nn.Parameter]`; no structural change needed, correct filtering at the `node0_server.py` call site |
| Auth bypass for self-hosted | New `--local-mode` flag in `run_server.py`; conditional branch before `authorize_with_pluralis()` | `PluralisAuthorizer` calls external `https://auth.pluralis.ai`; `Node0Server.create()` already accepts `authorizer` as kwarg — just skip the call |
| Training metrics schema | New `TrainingMetricsV1` Pydantic model in `security/validation.py` | Follows precedent of `WorkerMetricsV1`; `MonitorWorker.report()` extended to publish via DHT under new key |
| Corpus storage | Filesystem entries under `~/gt/.corpus/<role>/<id>.json` + Dolt metadata table | Matches existing pattern: Dolt for queryable metadata (`~/gt/.beads/` uses Dolt), filesystem for blobs |
| Model registry storage | Dolt table `model_registry` in HQ database + file artifacts under `~/gt/.models/` | Same split: Dolt for queryable metadata (eval scores, lineage), filesystem for weight files |
| DPO preference pairs | New Dolt table `dpo_preferences` | Structured relational data, links to existing bead records, queryable for training |
| Checkpoint export coordination | DHT-coordinated gather: each stage saves its shard; coordinator assembles | DHT is the coordination bus per codebase constraint; no direct connections between stages |
| Inference entry point | New `run_inference.py` alongside `run_server.py` | Mirrors `run_server.py` structure; reuses gRPC pipeline forwarding unchanged |
| Config format for fine-tuning | New YAML files under `src/node0/configs/finetune/` with `finetune_config` top-level key | Extends existing `class_name`/`init_args` pattern; `build_cls()` handles instantiation |
| NaN handling in fine-tuning | Replace `os.killpg` with `RetriableError` raise + bead-state update before re-raise | NaN grads more common in fine-tuning; existing `NonRetriableError`/`RetriableError` hierarchy in `utils/node_info.py` (corrected from plan-context.md); bead lifecycle must be updated to `failed` before exit |
| Test infrastructure | Establish `tests/unit/` and `tests/integration/` at repo root; use pytest | No existing test infrastructure; plan-context.md recommendation; highest-value targets are corpus pipeline and eval suite (pure Python, no distributed deps) |

---

## Shared Abstractions

These types and utilities are consumed across multiple phases. They must be built in Phase 1 so later phases can import them without circular dependencies.

**LoRAConfig**
- Location: `src/node0/finetune/lora/config.py`
- Purpose: Pydantic BaseModel for LoRA hyperparameters (`rank`, `alpha`, `dropout`, `target_modules`). Imported by `ModelArguments` extension and by the LoRA application logic in `layers.py`.
- Consumers: Phase 1 (ModelArguments), Phase 2 (LoRA adapter application), Phase 3 (YAML configs)

**FineTuneConfig**
- Location: `src/node0/finetune/config.py`
- Purpose: Top-level Pydantic BaseModel aggregating `mode` (sft|dpo), `corpus_path`, `checkpoint_dir`, `checkpoint_every_n_steps`, `eval_every_n_steps`, `role`, and a `lora: LoRAConfig | None` field. Parsed from YAML by `build_cls()`.
- Consumers: Phase 1 (entry point), Phase 2 (data loading, LoRA), Phase 3 (checkpoint), Phase 5 (inference)

**CorpusEntry**
- Location: `src/node0/finetune/corpus/schema.py`
- Purpose: Pydantic BaseModel for a single corpus record: `id`, `role`, `rig`, `task_type`, `outcome`, `created_at`, `messages: list[dict]`, `chosen: list[dict] | None`, `rejected: list[dict] | None`. Used by corpus collector, scrubber, formatter, and data loader.
- Consumers: Phase 2 (corpus pipeline), Phase 3 (data loader)

**ModelRegistryEntry**
- Location: `src/node0/finetune/registry/schema.py`
- Purpose: Pydantic BaseModel for a model registry row: `name`, `role`, `base_model`, `version`, `corpus_hash`, `eval_score`, `eval_scores_by_category`, `lora_adapters`, `deployment_status`, `deployed_to`, `created_at`, `training_config`, `artifact_path`. Maps 1:1 to the Dolt `model_registry` table.
- Consumers: Phase 4 (registry store and CLI), Phase 6 (eval suite populates `eval_score`)

**TrainingMetricsV1**
- Location: `src/node0/security/validation.py` (alongside existing `WorkerMetricsV1`)
- Purpose: Pydantic v1 BaseModel (use `from pydantic.v1` for Hivemind compatibility) for DHT-published training metrics: `peer_id`, `step`, `loss`, `role`, `corpus_hash`.
- Consumers: Phase 2 (MonitorWorker), Phase 3 (checkpoint events)

**EvalResult**
- Location: `src/node0/finetune/eval/schema.py`
- Purpose: Pydantic BaseModel for per-task eval output: `model_name`, `category`, `score`, `tasks_run`, `tasks_passed`, `timestamp`.
- Consumers: Phase 6 (eval runner and scorer), Phase 4 (registry stores composite score)

---

## Phased Delivery

### Phase 1: Foundation — Blockers, Config, and Auth Bypass

**Objective:** Eliminate all blockers that prevent any subsequent work. Fix the `vocab_size` correctness issue, add LoRA and fine-tune config fields to the Pydantic models, make authorization optional for self-hosted runs, and establish the test infrastructure. After Phase 1, a node0 server can start in local mode with a fine-tuning config without external auth.

**Prerequisites:** None (first phase)

#### Tasks

**1.1 Fix vocab_size and add LoRA fields to ModelArguments**
- **What:** Correct the `vocab_size` default in `LlamaArguments` from `50265` (OPT tokenizer) to `32000` (LLaMA 2) or `128256` (LLaMA 3). Add LoRA fields to base `ModelArguments` so all Expert constructors inherit them automatically. Increase `max_seq_len` default.
- **Files:**
  - Modify: `src/node0/models/arguments.py` — add fields `lora_rank: int | None = None`, `lora_alpha: float = 128.0`, `lora_dropout: float = 0.05`, `lora_target_modules: list[str] = ["wq", "wk", "wv", "wo"]`
  - Modify: `src/node0/models/llama/arguments.py` — change `vocab_size: int = 50265` to `vocab_size: int = 32000`; change `max_seq_len: int = 512` to `max_seq_len: int = 2048`
  - Create: `src/node0/finetune/__init__.py` — empty, establishes sub-package
  - Create: `src/node0/finetune/lora/__init__.py` — empty
  - Create: `src/node0/finetune/lora/config.py` — `LoRAConfig` Pydantic BaseModel
  - Create: `src/node0/finetune/config.py` — `FineTuneConfig` Pydantic BaseModel
- **Key details:** Use `from pydantic import BaseModel` (v2) for all new types. `LoRAConfig` fields: `rank: int = 64`, `alpha: float = 128.0`, `dropout: float = 0.05`, `target_modules: list[str] = ["wq", "wk", "wv", "wo"]`. `FineTuneConfig` fields: `mode: Literal["sft", "dpo"] = "sft"`, `corpus_path: str`, `checkpoint_dir: str`, `checkpoint_every_n_steps: int = 100`, `eval_every_n_steps: int = 500`, `role: Literal["mayor", "polecat", "witness", "deacon"]`, `lora: LoRAConfig | None = None`. Add copyright header (`# Copyright 2025 Pluralis Research`) to all new files. Use `from hivemind.utils.logging import get_logger; logger = get_logger(__name__)` in any module with logging.
- **Acceptance criteria:**
  - [ ] `LlamaArguments().vocab_size == 32000`
  - [ ] `ModelArguments` has `lora_rank`, `lora_alpha`, `lora_dropout`, `lora_target_modules` fields
  - [ ] `FineTuneConfig` and `LoRAConfig` importable from `node0.finetune.config` and `node0.finetune.lora.config`
  - [ ] `python -c "from node0.models.llama.arguments import LlamaArguments; a = LlamaArguments(); assert a.vocab_size == 32000"` passes
  - [ ] Tests: `tests/unit/test_model_arguments.py` — assert field defaults and that LoRA fields pass through Pydantic validation
- **Dependencies:** None

**1.2 Add TrainingMetricsV1 schema**
- **What:** Add `TrainingMetricsV1` to `security/validation.py` alongside existing `WorkerMetricsV1`. Update `make_validators()` to include a validator entry for the new schema.
- **Files:**
  - Modify: `src/node0/security/validation.py` — add `TrainingMetricsV1` class using `from pydantic.v1 import BaseModel, StrictFloat, StrictInt, StrictStr` (Hivemind compatibility); add `TrainingMetricSchema` wrapper class; update `make_validators()` to register it
- **Key details:** Fields for `TrainingMetricsV1`: `peer_id: str`, `step: StrictInt`, `loss: StrictFloat`, `role: StrictStr`, `corpus_hash: StrictStr`. DHT key pattern: `{experiment_prefix}_{stage}_training_metrics` (follows existing `worker_metrics` key pattern in `monitor.py`).

  **Important:** The actual `make_validators()` uses `SchemaValidator(MetricSchema, prefix=...)` where `MetricSchema` is a Pydantic v1 wrapper. Adding `TrainingMetricsV1` requires: (a) creating a `TrainingMetricSchema` wrapper class with field `training_metrics: dict[BytesWithPublicKeyType, TrainingMetricsV1]`; (b) instantiating `SchemaValidator(TrainingMetricSchema, prefix="training_metrics")`; (c) appending this to the validators list. Do not simply append `TrainingMetricsV1` directly — follow the existing `SchemaValidator(MetricSchema, ...)` pattern in `validation.py` exactly.
- **Acceptance criteria:**
  - [ ] `TrainingMetricsV1` importable from `node0.security.validation`
  - [ ] `TrainingMetricSchema` wrapper class exists with correct `training_metrics` field
  - [ ] `make_validators()` returns validator dict including `TrainingMetricSchema` entry
  - [ ] Tests: `tests/unit/test_validation.py` — construct valid and invalid `TrainingMetricsV1` instances
- **Dependencies:** None

**1.3 Add --local-mode flag and fine-tuning CLI args to run_server.py**
- **What:** Make `PluralisAuthorizer` optional. When `--local-mode` is passed, skip `authorize_with_pluralis()` and use a local stage assignment from CLI arg `--local-stage`. Add fine-tuning CLI args: `--finetune-config` (path to fine-tuning YAML), which is parsed separately before YAML config, then consumed with `args.pop()` before forwarding to server.
- **Files:**
  - Modify: `src/node0/run_server.py` — add argparse args `--local-mode` (boolean flag), `--local-stage` (string, e.g., `"head-0"`), `--finetune-config` (path); add conditional branch: if `local_mode`, skip `authorize_with_pluralis()`, construct minimal auth object or pass `authorizer=None` to `Node0Server.create()`; load `FineTuneConfig` from `--finetune-config` if provided; `args.pop("local_mode"); args.pop("local_stage"); args.pop("finetune_config")` before forwarding kwargs
- **Key details:** `Node0Server.create()` already accepts `authorizer` as kwarg and passes it to `Server` — passing `None` means no auth. When `--local-stage` is provided, set `model_args.stage` directly from it rather than from authorizer's `pipeline_stage` assignment. The `FineTuneConfig` object should be attached to the process context or passed through as a new optional parameter to `Node0Server.create()` (add `finetune_config: FineTuneConfig | None = None` to its signature).
- **Acceptance criteria:**
  - [ ] `python -m node0.run_server --local-mode --local-stage head-0 --config ...` starts without calling `https://auth.pluralis.ai`
  - [ ] `--finetune-config` arg is parsed, loaded as `FineTuneConfig`, and available to server creation
  - [ ] Original (non-local-mode) path is entirely unchanged — no regression
  - [ ] Tests: `tests/unit/test_run_server.py` — mock `authorize_with_pluralis` and assert it is not called when `--local-mode` is set
- **Dependencies:** Task 1.1 (FineTuneConfig must exist)

**1.4 Establish test infrastructure**
- **What:** Create `tests/` directory structure with pytest configuration, base fixtures, and initial unit tests.
- **Files:**
  - Create: `tests/__init__.py` — empty
  - Create: `tests/unit/__init__.py` — empty
  - Create: `tests/integration/__init__.py` — empty
  - Create: `tests/conftest.py` — shared pytest fixtures (minimal model args, mock DHT, mock queue)
  - Create: `pyproject.toml` modification — add `[tool.pytest.ini_options]` section with `testpaths = ["tests"]`, `python_files = "test_*.py"`, `python_classes = "Test*"`, `python_functions = "test_*"`
- **Key details:** Unit tests mock all network/DHT calls. Integration tests are gated by a `@pytest.mark.integration` marker and skipped in default `pytest` run (requires `pytest -m integration`). Fixtures in `conftest.py`: `minimal_llama_args()` returning `LlamaArguments(hidden_dim=64, n_heads=2, num_hidden_layers=2, n_kv_heads=2)`, `mock_dht()` returning a `MagicMock`.
- **Acceptance criteria:**
  - [ ] `pytest tests/unit/` runs without error from repo root
  - [ ] `pytest --co tests/` lists collected tests without import errors
- **Dependencies:** None (can run in parallel with 1.1-1.3)

**1.5 Add optional deps and new entry points to pyproject.toml**
- **What:** Add optional dependency group for fine-tuning. Register new console scripts.
- **Files:**
  - Modify: `pyproject.toml` — add `[project.optional-dependencies]` section with `finetune = ["transformers>=4.40", "peft>=0.10", "safetensors>=0.4", "chromadb>=0.5", "detect-secrets>=1.5", "tqdm"]`; add to `[project.scripts]`: `node0-inference = "node0.run_inference:main"`, `node0-finetune = "node0.run_finetune:main"`, `node0-corpus = "node0.run_corpus:main"`
- **Key details:** Scripts reference modules that will be created in later phases. Creating the entry points now allows `pip install -e ".[finetune]"` to succeed and `node0-inference --help` to fail with a clean import error rather than a missing entry point error.
- **Acceptance criteria:**
  - [ ] `pip install -e ".[finetune]"` succeeds (may fail if GPU deps not available — acceptable in dev)
  - [ ] `pyproject.toml` passes `pip check` for dependency consistency
- **Dependencies:** None

**1.6 Create run_finetune.py entry point**
- **What:** Create `src/node0/run_finetune.py` with a `main()` entry point that parses args and launches the fine-tuning pipeline. This is the module registered as the `node0-finetune` console script in `pyproject.toml` (Task 1.5). Without this file, the console script raises `ModuleNotFoundError` at runtime.
- **Files:**
  - Create: `src/node0/run_finetune.py` — `main()` function: argparse with `--config` (path to fine-tuning YAML), `--corpus-path` (overrides config value), `--checkpoint-dir` (overrides config value), `--role` (role filter: mayor/polecat/witness/deacon), `--local-mode` (bool flag, passed through to server startup), `--local-stage` (string, passed through). Loads `FineTuneConfig` from `--config` YAML. Validates corpus path exists. Prints startup banner. Delegates to fine-tuning pipeline coordinator (placeholder in Phase 1, wired to real implementation in Phase 3).
- **Key details:** Follow the same structure as `run_server.py` and `run_corpus.py`: logger initialization first (`logger = get_logger(__name__)`), then argparse, then config load, then component initialization. The `main()` must be importable and callable with no args (for `--help` to work). In Phase 1, after arg parsing and config load, the function can `raise NotImplementedError("Fine-tuning pipeline not yet implemented — wire in Phase 3")` or log and exit cleanly. The file must exist so `pip install -e ".[finetune]"` creates a working entry point stub. Add copyright header. Phase 3 will replace the stub body with real pipeline launch.
- **Acceptance criteria:**
  - [ ] `node0-finetune --help` runs without `ModuleNotFoundError` and prints usage
  - [ ] `node0-finetune --config /nonexistent.yaml` exits with a clean error message (not a traceback)
  - [ ] `from node0.run_finetune import main` succeeds in Python REPL
  - [ ] Tests: `tests/unit/test_run_finetune.py` — assert `main()` is importable; assert argparse exits on missing required args
- **Dependencies:** Task 1.1 (FineTuneConfig must exist), Task 1.5 (entry point registered in pyproject.toml)

**1.7 Update Dockerfile with CUDA training dependencies**
- **What:** Add CUDA-compatible training dependencies to the Dockerfile so that `node0-finetune`, `node0-inference`, and `node0-corpus` entry points are available inside the container with all required libraries.
- **Files:**
  - Modify: `Dockerfile` — add `RUN pip install "torch[cuda]" transformers>=4.40 peft>=0.10 safetensors>=0.4 chromadb>=0.5 detect-secrets>=1.5 tqdm` (or equivalent via requirements file); add `VOLUME ["/root/gt/.corpus", "/root/gt/.checkpoints", "/root/gt/.rag"]` declarations for persistent data directories; ensure `node0-finetune`, `node0-inference`, `node0-corpus` are on PATH after `pip install -e ".[finetune]"`
- **Key details:** PEFT library provides LoRA primitives (even though we implement LoRA directly, PEFT utilities may be used). CUDA training deps must be installed from CUDA-compatible PyTorch wheel index (`--index-url https://download.pytorch.org/whl/cu118` or appropriate CUDA version). The base image must have CUDA runtime available. Volume mounts ensure corpus and checkpoint data survive container restarts. Existing Dockerfile installs base node0 deps — this adds finetune extras on top.
- **Acceptance criteria:**
  - [ ] Docker build succeeds with new deps
  - [ ] `docker run ... node0-finetune --help` prints usage (entry point available in container)
  - [ ] Volume declarations present in `docker inspect` output
- **Dependencies:** Task 1.5 (optional deps must be defined in pyproject.toml before Dockerfile installs them)

**1.8 Update run.json with local mode and seed peer configuration**
- **What:** Add local mode and self-hosted fine-tuning configuration options to `run.json` so that operators can run fine-tuning without Pluralis DHT or external auth.
- **Files:**
  - Modify: `run.json` — add `"local_mode": false` field (boolean, enables `--local-mode` flag in server); add `"local_seed_peers": []` field (list of strings, peer addresses for self-hosted DHT without Pluralis discovery); add `"bypass_auth": false` field (explicit auth bypass flag); add an example template comment block showing a complete self-hosted fine-tuning run configuration
- **Key details:** `run.json` is the runtime configuration complement to the YAML model config. The new fields must be read by `run_server.py` (Task 1.3) when present, with CLI flags taking precedence over `run.json` values. When `local_mode: true` in `run.json`, server skips `authorize_with_pluralis()` matching the `--local-mode` CLI flag. `local_seed_peers` allows manual DHT peer configuration when Pluralis DHT is unavailable. These are opt-in fields — existing `run.json` files without them continue to work unchanged (default values: false/empty).
- **Acceptance criteria:**
  - [ ] `run.json` with `"local_mode": true` causes server to skip Pluralis auth (same behavior as `--local-mode` CLI flag)
  - [ ] `run.json` with `"local_seed_peers": ["1.2.3.4:8765"]` passes peers to DHT initialization
  - [ ] Existing `run.json` files without new fields continue to work unchanged
- **Dependencies:** Task 1.3 (local-mode flag must exist in run_server.py)

#### Phase 1 Exit Criteria
- [ ] `vocab_size` is `32000` in `LlamaArguments`
- [ ] LoRA fields present on `ModelArguments`
- [ ] `FineTuneConfig` and `LoRAConfig` importable
- [ ] `TrainingMetricsV1` importable from `security.validation`
- [ ] `--local-mode` flag works in `run_server.py` without calling external auth
- [ ] `pytest tests/unit/` passes (all unit tests green)
- [ ] `pyproject.toml` has `finetune` optional dep group and new script entry points
- [ ] `node0-finetune --help` runs without `ModuleNotFoundError`
- [ ] Dockerfile updated with CUDA training deps and volume declarations
- [ ] `run.json` has `local_mode`, `local_seed_peers`, and `bypass_auth` fields

---

### Phase 2: Data Pipeline and Corpus Collection

**Objective:** Build the two data-facing subsystems: (1) the internal data loading pipeline that feeds node0's training loop, and (2) the corpus collection pipeline that harvests Gas Town session transcripts. After Phase 2, the system can collect, scrub, and format training data, and feed batches into node0's gradient accumulation.

**Prerequisites:** Phase 1 — `FineTuneConfig`, `LoRAConfig`, `CorpusEntry` schema types must exist; `vocab_size` must be correct before tokenizer integration.

#### Tasks

**2.1 Corpus schema and store**
- **What:** Create `CorpusEntry` schema and filesystem-backed corpus store with Dolt metadata.
- **Files:**
  - Create: `src/node0/finetune/corpus/__init__.py` — empty
  - Create: `src/node0/finetune/corpus/schema.py` — `CorpusEntry` Pydantic BaseModel (fields as defined in Shared Abstractions)
  - Create: `src/node0/finetune/corpus/store.py` — `CorpusStore` class: `add(entry: CorpusEntry)` writes JSON to `~/gt/.corpus/<role>/<id>.json`; `list(role: str) -> list[CorpusEntry]` scans directory; `get(id: str) -> CorpusEntry`; `corpus_hash(role: str) -> str` returns SHA-256 of sorted entry IDs for reproducibility
- **Key details:** Use `pathlib.Path` throughout (not string paths — convention from plan-context.md). The `corpus_hash()` method hashes the sorted list of entry IDs (not content) for a stable, lightweight fingerprint used in `TrainingMetricsV1.corpus_hash`. No Dolt write in this task — Dolt metadata integration deferred to Phase 4 (registry). Store is the primary corpus access layer.
- **Acceptance criteria:**
  - [ ] `CorpusStore.add()` creates file at correct path
  - [ ] `CorpusStore.list(role="polecat")` returns entries from `~/gt/.corpus/polecat/`
  - [ ] `CorpusStore.corpus_hash()` is deterministic (same entries → same hash)
  - [ ] Tests: `tests/unit/test_corpus_store.py` — use `tmp_path` fixture for filesystem isolation
- **Dependencies:** None (within Phase 2; needs schema types from this task for 2.2-2.4)

**2.2 PII scrubber**
- **What:** Multi-layer automated PII scrubbing pipeline for corpus entries.
- **Files:**
  - Create: `src/node0/finetune/corpus/scrubber.py` — `CorpusScrubber` class with `scrub(text: str) -> tuple[str, list[str]]` returning (scrubbed text, list of match descriptions for audit); internally: (1) `detect-secrets` scan for JWTs, PEM blocks, API keys; (2) regex patterns for IP addresses, file paths with home dirs, email addresses; (3) confidence-based flagging: high-confidence matches are redacted with `[REDACTED:<type>]`, uncertain matches are quarantined (returned in a separate list for manual review)
- **Key details:** Import `detect_secrets` for layer 1. Regex patterns for layer 2: `r"(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36}"` (GitHub tokens), `r"[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}"` (IPs), `r"/home/[^/\s]+"` (home paths), `r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"` (email). Uncertain matches (detect-secrets below 0.9 confidence) go to a quarantine list rather than silent inclusion. All redactions logged via `logger.info(f"Redacted {len(matches)} items from entry {entry_id}")`.
- **Acceptance criteria:**
  - [ ] `scrub()` removes GitHub token patterns from text
  - [ ] `scrub()` redacts IP addresses
  - [ ] Uncertain matches returned in quarantine list, not silently included or silently redacted
  - [ ] Tests: `tests/unit/test_scrubber.py` — fixture strings with known PII patterns; assert redaction and quarantine behavior
- **Dependencies:** Task 2.1 (CorpusEntry schema)

**2.3 Corpus formatter**
- **What:** Convert raw session transcripts into `CorpusEntry` objects with correct role tagging and message formatting.
- **Files:**
  - Create: `src/node0/finetune/corpus/formatter.py` — `CorpusFormatter` class: `format_session(transcript: str, role: str, rig: str, bead_id: str | None, outcome: str, task_type: str) -> CorpusEntry`; parses tmux session transcript into `messages: list[dict]` with `{"role": "user|assistant|system", "content": "..."}`; generates `id` as `sha256(bead_id + timestamp)[:16]`; calls `CorpusScrubber.scrub()` on all content fields before constructing entry
- **Key details:** Transcript format from tmux: alternating user/assistant turns separated by role markers. The formatter must handle incomplete turns (session cut off mid-response) by truncating to the last complete turn. `task_type` is inferred from bead tags if `bead_id` is provided, otherwise passed explicitly. DPO entries (`chosen`/`rejected` fields) are left `None` by this formatter — DPO pairing happens in a separate step (Phase 7).
- **Acceptance criteria:**
  - [ ] `format_session()` produces valid `CorpusEntry` with scrubbed content
  - [ ] Incomplete turns are truncated, not included as partial messages
  - [ ] `id` field is deterministic given same inputs
  - [ ] Tests: `tests/unit/test_corpus_formatter.py` — fixture transcript strings covering complete and incomplete turn cases
- **Dependencies:** Task 2.2 (scrubber must exist before formatter calls it)

**2.4 Corpus collector (session hook)**
- **What:** A collector that fires on Gas Town session close, extracts the tmux transcript, formats, and stores the entry.
- **Files:**
  - Create: `src/node0/finetune/corpus/collector.py` — `CorpusCollector` class: `collect_from_session(session_name: str, bead_id: str | None, role: str, rig: str, outcome: str) -> CorpusEntry | None`; uses subprocess to `tmux capture-pane -p -t <session_name>` to extract transcript; calls `CorpusFormatter.format_session()`; calls `CorpusStore.add()`; returns `None` and logs warning if transcript is empty or below minimum length (configurable, default 100 tokens estimated)
  - Create: `src/node0/run_corpus.py` — CLI entry point: `main()` with subcommands: `collect` (trigger manual collection), `add` (add curated example from file), `list` (list corpus by role), `stats` (show counts per role/task_type). Uses argparse following same pattern as `run_server.py`.
- **Key details:** The collector is invoked from Gas Town's session lifecycle hook (a `gt hook session_shutdown` script — see Gas Town integration task 6.1). The `run_corpus.py` CLI is the `node0-corpus` entry point registered in `pyproject.toml`. The `collect` subcommand takes `--session`, `--bead-id`, `--role`, `--rig`, `--outcome` args. For wisp/non-bead sessions, `--bead-id` is optional.
- **Acceptance criteria:**
  - [ ] `node0-corpus collect --session my-session --role polecat --rig sfgt --outcome success` creates a corpus entry file
  - [ ] Empty transcripts produce a warning log and return `None` (no file written)
  - [ ] `node0-corpus list --role polecat` prints entry IDs and metadata
  - [ ] Tests: `tests/unit/test_collector.py` — mock subprocess call for tmux capture
- **Dependencies:** Tasks 2.1, 2.2, 2.3

**2.4b CorpusArtifactCollector — codebase and configuration sources**
- **What:** Collect Gas Town codebase artifacts (CLAUDE.md files, formula TOMLs, spec.md files, AGENTS.md) into corpus entries, providing convention-document training data that is separate from session transcripts.
- **Files:**
  - Create: `src/node0/finetune/corpus/artifact_collector.py` — `CorpusArtifactCollector` class: `collect_claude_md(rig_root: Path, role: str = "all") -> list[CorpusEntry]`: reads CLAUDE.md from rig root, wraps content as a corpus entry with `task_type="convention_doc"`, `role=role`; `collect_formulas(formulas_dir: Path) -> list[CorpusEntry]`: reads all `*.formula.toml` files from `~/gt/.beads/formulas/`, wraps each as corpus entry with `task_type="formula_definition"`; `collect_specs(plans_root: Path) -> list[CorpusEntry]`: reads `spec.md` files from `plans/*/02-spec/` paths, wraps as `task_type="spec_doc"`; `collect_agents_md(rig_root: Path) -> list[CorpusEntry]`: reads AGENTS.md if present; all methods call `CorpusScrubber.scrub()` on content and call `CorpusStore.add()` to persist
  - Modify: `src/node0/run_corpus.py` — add `artifacts` subcommand to trigger `CorpusArtifactCollector.collect_*()` for all artifact types from a given rig root
- **Key details:** Artifact entries use the same `CorpusEntry` schema as session transcripts but with `messages` formatted as a single-turn system document: `[{"role": "system", "content": "<artifact content>"}]`. The `bead_id` field is `None` for artifact entries (not session-linked). Generated `id` is `sha256(file_path + mtime)[:16]` for deduplication on re-collection. `outcome` field is set to `"collected"` for all artifact entries. This collector is run periodically (e.g., when CLAUDE.md changes) rather than per-session.
- **Acceptance criteria:**
  - [ ] `node0-corpus artifacts --rig-root /home/ubuntu/gt` creates corpus entries for CLAUDE.md and formula TOMLs
  - [ ] PII scrubber runs on all artifact content before storage
  - [ ] Re-running with unchanged files produces same entry IDs (idempotent)
  - [ ] Tests: `tests/unit/test_artifact_collector.py` — use `tmp_path` with fixture CLAUDE.md content; assert entry structure
- **Dependencies:** Tasks 2.1, 2.2

**2.4c OperationalDataCollector — bead lifecycle events and sling outcomes**
- **What:** Collect operational data from the Dolt beads database into corpus entries: bead lifecycle events (state transitions), sling outcomes (success/failure, duration, error type), and convoy logs.
- **Files:**
  - Create: `src/node0/finetune/corpus/operational_collector.py` — `OperationalDataCollector` class: `collect_bead_lifecycles(dolt_host: str, dolt_port: int, role: str | None = None) -> list[CorpusEntry]`: queries Dolt `beads` table for status transition history, formats each lifecycle as a corpus entry with `task_type="bead_lifecycle"`; `collect_sling_outcomes(dolt_host: str, dolt_port: int) -> list[CorpusEntry]`: queries for completed sling records, formats outcome summary as corpus entry with `task_type="sling_outcome"`; `collect_convoy_logs(logs_dir: Path) -> list[CorpusEntry]`: reads convoy log files, formats as corpus entries with `task_type="convoy_log"`; all connect to Dolt via MySQL protocol at `127.0.0.1:3307`
  - Modify: `src/node0/run_corpus.py` — add `operational` subcommand to trigger `OperationalDataCollector` collection
- **Key details:** Bead lifecycle corpus entries represent the sequence of status transitions as a conversation: each status change is a turn in `messages`, e.g., `[{"role": "system", "content": "Bead: <title>"}, {"role": "assistant", "content": "Status changed: open → in_progress at <ts>"}, ...]`. PII scrubber runs on all content. `outcome` is set to the final bead status (`done`, `failed`, etc.). `bead_id` is the source bead's ID. Dolt connection uses `mysql-connector-python` (same as model registry in Task 4.4). Sling outcome entries include duration, error type if failed, and role.
- **Acceptance criteria:**
  - [ ] `node0-corpus operational --collect beads` queries Dolt and creates corpus entries for bead lifecycle events
  - [ ] `CorpusEntry` objects produced are compatible with `CorpusStore.add()` and `CorpusDataset`
  - [ ] PII scrubber runs on all collected content
  - [ ] Tests: `tests/unit/test_operational_collector.py` — mock Dolt MySQL connection; assert entry structure and message formatting
- **Dependencies:** Tasks 2.1, 2.2

**2.4d Periodic PII scrubbing audit**
- **What:** Implement a periodic audit that re-scrubs the corpus store and flags entries where PII scrubbing may have missed content (based on updated scrubbing rules). Closes the fourth scrubbing layer from the spec.
- **Files:**
  - Modify: `src/node0/finetune/corpus/scrubber.py` — add `audit_entry(entry: CorpusEntry, strict: bool = False) -> list[str]`: re-applies all scrubbing rules to an existing entry's messages and returns a list of newly-detected match descriptions (empty list = clean); add `audit_store(store: CorpusStore, role: str | None = None) -> dict[str, list[str]]`: iterates all entries in store, calls `audit_entry()` on each, returns `{entry_id: [match_descriptions]}` for any entries with new detections
  - Modify: `src/node0/run_corpus.py` — add `audit` subcommand: takes `--role` (optional filter), calls `audit_store()`, prints summary (N entries audited, M flagged), writes flagged entry IDs and match descriptions to a report file at `~/gt/.corpus/audit-report-<date>.json`
- **Key details:** The audit uses updated scrubbing rules (may catch patterns that were missed when the entry was originally collected). Newly detected items are flagged for manual review — the audit does not automatically re-scrub or delete entries, it only reports. Log `logger.warning(f"Audit flagged {len(flagged)} entries with potential PII")`. The `audit` subcommand can be scheduled via cron (weekly) or triggered manually. This implements the spec's "periodic audit/sampling of scrubbed output" requirement (4th scrubbing layer).
- **Acceptance criteria:**
  - [ ] `node0-corpus audit --role polecat` scans all polecat entries and reports flagged ones
  - [ ] Audit does not modify existing entries (read-only)
  - [ ] Report file written to `~/gt/.corpus/audit-report-<date>.json`
  - [ ] Tests: `tests/unit/test_scrubber_audit.py` — inject entry with known PII; assert audit detects it
- **Dependencies:** Tasks 2.1, 2.2

**2.5 Tokenizer integration**
- **What:** Wrap HuggingFace tokenizer for use in data loading. Handle Gas Town-specific vocabulary considerations.
- **Files:**
  - Create: `src/node0/finetune/data/__init__.py` — empty
  - Create: `src/node0/finetune/data/tokenizer.py` — `load_tokenizer(model_name_or_path: str, max_length: int = 2048) -> PreTrainedTokenizerFast`; loads from HuggingFace `transformers`; sets `pad_token = eos_token` if not set (standard LLaMA pattern); validates that `tokenizer.vocab_size` matches `LlamaArguments.vocab_size` (raises `NonRetriableError` from `utils/node_info.py` if mismatch — this is the correctness gate that makes the vocab_size fix from Task 1.1 meaningful)
- **Key details:** Import `from transformers import AutoTokenizer`. `NonRetriableError` is in `src/node0/utils/node_info.py` (corrected import path). The vocab size validation error message should be explicit: `f"Tokenizer vocab_size {tokenizer.vocab_size} != model vocab_size {model_vocab_size}. Check LlamaArguments.vocab_size."`. No custom tokens in V1 — rely on base tokenizer (per spec open question Q4, deferred).
- **Acceptance criteria:**
  - [ ] `load_tokenizer("meta-llama/Llama-2-7b-hf")` returns tokenizer with `vocab_size == 32000`
  - [ ] Mismatch between tokenizer and model `vocab_size` raises `NonRetriableError` with descriptive message
  - [ ] Tests: `tests/unit/test_tokenizer.py` — mock HuggingFace download; test validation logic
- **Dependencies:** Task 1.1 (vocab_size fix is a prerequisite for correctness)

**2.6 Dataset and DataLoader**
- **What:** PyTorch Dataset and DataLoader wrapping CorpusStore entries for training consumption.
- **Files:**
  - Create: `src/node0/finetune/data/dataset.py` — `CorpusDataset(torch.utils.data.Dataset)`: takes `CorpusStore`, `role: str`, `tokenizer`, `max_length: int`; `__len__` returns corpus size; `__getitem__` tokenizes a `CorpusEntry.messages` list into a flat input_ids tensor with causal LM labels (labels = input_ids shifted by 1, padding positions set to -100); for DPO entries, produces dict with `chosen_ids` and `rejected_ids`
  - Create: `src/node0/finetune/data/loader.py` — `make_dataloader(dataset: CorpusDataset, batch_size: int, shuffle: bool = True) -> DataLoader`; uses `torch.utils.data.DataLoader` with `collate_fn` that pads sequences to max length in batch and creates attention masks
- **Key details:** Message formatting for causal LM: system prompt tokens + user turn tokens + assistant turn tokens concatenated. Labels for system+user tokens set to -100 (not trained on input, only on assistant response). This is standard SFT formatting. For DPO mode, `__getitem__` returns a dict with both chosen and rejected token sequences. `collate_fn` must handle variable-length sequences via right-padding to batch max length.
- **Acceptance criteria:**
  - [ ] `CorpusDataset.__getitem__` returns tensor of `input_ids` and `labels` for SFT mode
  - [ ] Labels for non-assistant tokens are -100
  - [ ] DataLoader yields correctly padded batches
  - [ ] Tests: `tests/unit/test_dataset.py` — use synthetic `CorpusEntry` fixtures; assert label masking
- **Dependencies:** Tasks 2.1, 2.5

**2.7 Loss functions**
- **What:** SFT cross-entropy loss and DPO loss implementations.
- **Files:**
  - Create: `src/node0/finetune/data/loss.py` — `sft_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor`: standard cross-entropy ignoring -100 labels; `dpo_loss(chosen_logits: torch.Tensor, rejected_logits: torch.Tensor, chosen_labels: torch.Tensor, rejected_labels: torch.Tensor, beta: float = 0.1) -> torch.Tensor`: DPO objective per Rafailov et al. 2023
- **Key details:** `sft_loss` uses `torch.nn.functional.cross_entropy(logits.view(-1, vocab_size), labels.view(-1), ignore_index=-100)`. `dpo_loss` computes log-probabilities of chosen and rejected sequences, then `loss = -F.logsigmoid(beta * (chosen_logprob - rejected_logprob)).mean()`. Note: in node0's pipeline-parallel setup, loss is computed only at the tail stage (`TailExpert.forward()` already computes cross-entropy). The SFT loss replaces the existing tail computation; DPO loss requires both chosen and rejected to flow through the full pipeline — this is a known architectural complexity (see Technical Risks).
- **Acceptance criteria:**
  - [ ] `sft_loss` is differentiable (`.backward()` succeeds)
  - [ ] `sft_loss` ignores positions where `labels == -100`
  - [ ] `dpo_loss` is differentiable and produces finite values for valid inputs
  - [ ] Tests: `tests/unit/test_loss.py` — synthetic logits; assert shapes and ignore-index behavior
- **Dependencies:** None (pure PyTorch, no prior-phase deps)

**2.8 Extend MonitorWorker for training metrics**
- **What:** Add training-specific metrics tracking (loss curve, eval scores) to the existing `MonitorWorker`.
- **Files:**
  - Modify: `src/node0/utils/monitor.py` — add regex patterns for fine-tuning log lines: `self.loss_pattern = re.compile(r"Training loss: ([0-9.]+)")`, `self.eval_pattern = re.compile(r"Eval score \[([a-z_]+)\]: ([0-9.]+)")`, `self.checkpoint_pattern = re.compile(r"Checkpoint saved: step ([0-9]+)")`; extend `monitor_callback()` to match these patterns; extend `report()` to publish `TrainingMetricsV1` to DHT under key `{experiment_prefix}_{stage}_training_metrics` when in fine-tuning mode (gated on `self.finetune_mode: bool` attribute added to `__init__`)
- **Key details:** `MonitorWorker.__init__` already receives `model_conf`; add `finetune_mode: bool = False` parameter. When `finetune_mode=True`, `report()` additionally publishes `TrainingMetricsV1` alongside existing `WorkerMetricsV1`. The `add_auth_info()` method signature is unchanged. Follow existing log-scraping pattern exactly: regex on log queue, not direct method calls.

  **TrainingMetricSchema wrapper class:** Adding `TrainingMetricsV1` to `make_validators()` requires more than appending to a list. The actual `make_validators()` function uses `SchemaValidator(MetricSchema, ...)` where `MetricSchema` is a Pydantic v1 wrapper class declaring the DHT key structure. The required steps are: (a) create a new `TrainingMetricSchema` wrapper class in `security/validation.py` with field `training_metrics: dict[BytesWithPublicKeyType, TrainingMetricsV1]`; (b) instantiate `SchemaValidator(TrainingMetricSchema, prefix="training_metrics")`; (c) append this validator to the list returned by `make_validators()`. Do not simply append `TrainingMetricsV1` directly — the method expects typed schema objects with defined field names and validation rules. Reference the existing validator schema pattern in `src/node0/security/validation.py` for the exact `SchemaValidator` and `MetricSchema` pattern to follow.
- **Acceptance criteria:**
  - [ ] `MonitorWorker(finetune_mode=True)` compiles new regex patterns without error
  - [ ] Log line `"Training loss: 2.34"` is matched and loss value extracted
  - [ ] `report()` publishes `TrainingMetricsV1` to DHT when `finetune_mode=True`
  - [ ] `make_validators()` returns a validator dict including `TrainingMetricSchema` entry
  - [ ] `TrainingMetricSchema` wrapper class exists in `security/validation.py` with correct field structure
  - [ ] Tests: `tests/unit/test_monitor.py` — inject synthetic log lines into monitor queue; assert metric extraction
- **Dependencies:** Task 1.2 (TrainingMetricsV1 schema)

#### Phase 2 Exit Criteria
- [ ] Corpus pipeline: collect → scrub → format → store works end-to-end
- [ ] `node0-corpus collect` CLI command works
- [ ] `node0-corpus artifacts` collects CLAUDE.md, formula TOMLs, spec files into corpus
- [ ] `node0-corpus operational` collects bead lifecycle events from Dolt into corpus
- [ ] `node0-corpus audit` runs PII re-scan and writes report for flagged entries
- [ ] `CorpusDataset` produces correctly masked SFT batches from corpus store
- [ ] `sft_loss` and `dpo_loss` are implemented and tested
- [ ] MonitorWorker emits training metrics in finetune mode
- [ ] All unit tests in `tests/unit/` pass

---

### Phase 3: LoRA Adapters and Training Integration

**Objective:** Wire the data pipeline into node0's training loop. Apply LoRA adapters per pipeline stage. Replace the aggressive NaN handler. After Phase 3, a distributed fine-tuning run can be launched end-to-end: data loads, gradients flow through LoRA parameters only, and the system converges.

**Prerequisites:** Phase 1 (LoRA config fields on ModelArguments), Phase 2 (DataLoader and loss functions must exist to feed the training loop).

#### Tasks

**3.1 LoRA adapter application per pipeline stage**
- **What:** Apply LoRA wrapping to the targeted linear layers inside each Expert class, conditional on `model_args.lora_rank is not None`.
- **Files:**
  - Create: `src/node0/finetune/lora/adapter.py` — `apply_lora(module: nn.Module, config: LoRAConfig) -> nn.Module`: replaces targeted linear layers by name with `LoRALinear` wrappers; `get_lora_parameters(module: nn.Module) -> list[nn.Parameter]`: returns only LoRA adapter parameters (A and B matrices), not base model params; `freeze_base_params(module: nn.Module) -> None`: sets `requires_grad=False` on all non-LoRA parameters
  - Create: `src/node0/finetune/lora/linear.py` — `LoRALinear(nn.Module)`: wraps an `nn.Linear`; forward = base_linear(x) + (lora_B @ lora_A)(x) * (alpha/rank); `lora_A` initialized with Kaiming uniform, `lora_B` initialized to zero (standard LoRA init ensuring zero delta at start)
  - Modify: `src/node0/models/llama/layers.py` — in `HeadExpert.__init__()`, `BodyExpert.__init__()`, `TailExpert.__init__()`: after base model construction, add `if self.args.lora_rank is not None: apply_lora(self, LoRAConfig(rank=self.args.lora_rank, alpha=self.args.lora_alpha, dropout=self.args.lora_dropout, target_modules=self.args.lora_target_modules))`
- **Key details:** `apply_lora()` walks `module.named_modules()`, identifies linears whose name matches any entry in `config.target_modules`, and replaces them using `setattr`. This is equivalent to PEFT's approach but implemented directly to avoid PEFT's full model assumption (incompatible with pipeline-parallel where each node only has one stage). After `apply_lora()`, call `freeze_base_params()` to set `requires_grad=False` on everything except LoRA A and B matrices. The Expert classes already receive `model_args` in their `__init__` (via `name_to_block` pattern) — no signature change needed.
- **Acceptance criteria:**
  - [ ] After `apply_lora()`, only LoRA A and B matrices have `requires_grad=True`
  - [ ] `LoRALinear.forward()` output matches `base_linear.forward()` when lora_B is zero (at init)
  - [ ] `get_lora_parameters()` returns only A and B matrices (not base weight)
  - [ ] Tests: `tests/unit/test_lora_adapter.py` — apply to a small `nn.Linear`; assert grad behavior; assert forward equivalence at init
- **Dependencies:** Task 1.1 (LoRA fields on ModelArguments)

**3.2 LoRA-aware gradient averager and optimizer parameter filtering**
- **What:** Ensure only LoRA parameters are averaged across peers and updated by the optimizer.
- **Files:**
  - Modify: `src/node0/server/node0_server.py` — at parameter collection site (lines 241-260): replace current param collection with `from node0.finetune.lora.adapter import get_lora_parameters; params = get_lora_parameters(expert) if finetune_config is not None else list(expert.parameters())`; pass filtered params to `GradientAverager` and optimizer; add `finetune_config: FineTuneConfig | None = None` parameter to `Node0Server.create()` classmethod signature
- **Key details:** `GradientAverager.__init__` takes `parameters: Iterable[torch.nn.Parameter]` — no structural change needed. Passing only LoRA parameters ensures: (1) only adapter gradients are averaged across peers, (2) base model weights are not updated. The `finetune_config` parameter flows in from `run_server.py` (Task 1.3). When `finetune_config is None`, behavior is identical to current pretraining (no regression).
- **Acceptance criteria:**
  - [ ] When `finetune_config` is provided, only LoRA parameters appear in optimizer param groups
  - [ ] When `finetune_config` is `None`, all parameters are used (pretraining mode unchanged)
  - [ ] Tests: `tests/unit/test_node0_server_params.py` — construct minimal server with mock expert; assert param filtering
- **Dependencies:** Tasks 1.1, 1.3, 3.1

**3.3 Data feeding integration via ModuleCollab**
- **What:** Feed training batches from DataLoader into the pipeline. Loss computation at TailExpert. Integration point is `ModuleCollab.on_backward()`.
- **Files:**
  - Create: `src/node0/finetune/data/feeder.py` — `DataFeeder(Thread)`: runs as background thread; holds a `DataLoader` iterator; on each call to `get_batch() -> dict | None` returns the next batch dict; `start()` / `stop()` following `threading.Event` pattern from `AutoStepOptimizer._should_stop`
  - Modify: `src/node0/server/module_collab.py` — add `data_feeder: DataFeeder | None = None` parameter to `ModuleCollab.__init__()`; in `on_backward()`: if `data_feeder is not None` and this is a tail stage, call `data_feeder.get_batch()` and inject batch into forward pass before gradient accumulation; log `"Training loss: {loss.item():.4f}"` so MonitorWorker regex picks it up
- **Key details:** The `on_backward()` hook is called by Hivemind's `ModuleBackend` after each backward pass through this stage. For head and body stages, `on_backward()` just calls the optimizer step (existing behavior). For the tail stage, the batch labels are needed to compute loss — the data feeder provides them. Pipeline stage identity is available from `model_args.stage`. Only the tail stage (`stage` contains "tail") calls `data_feeder.get_batch()` and computes `sft_loss` or `dpo_loss`. The loss `.backward()` is already triggered by Hivemind's pipeline backward — what changes is that the tail now uses labeled data to compute the right loss tensor before the backward.
- **Acceptance criteria:**
  - [ ] `DataFeeder` starts and stops cleanly via threading events
  - [ ] Tail stage logs `"Training loss: X.XXXX"` each step
  - [ ] Head/body stages are unaffected when `data_feeder=None`
  - [ ] Tests: `tests/unit/test_data_feeder.py` — synthetic DataLoader; assert batch delivery and thread cleanup
- **Dependencies:** Tasks 2.6, 2.7, 3.1

**3.4 Replace NaN handler with graceful recovery**
- **What:** Replace the `os.killpg` NaN gradient handler in `optim.py` with a recoverable error path that updates bead state before exiting.
- **Files:**
  - Modify: `src/node0/server/optim.py` — in `_check_and_accumulate_gradients()` at line 216: replace `os.killpg(os.getpgrp(), signal.SIGTERM)` with `raise RetriableError(f"NaN gradients detected at step {self.local_step}. Checkpoint resume required.")`; wrap caller in try/except that catches `RetriableError`, logs `logger.warning(f"NaN grads: {e}. Will attempt checkpoint resume.")`, and triggers checkpoint-resume flow (logs event for MonitorWorker to pick up)
- **Key details:** `RetriableError` is in `src/node0/utils/node_info.py` (corrected import path). The checkpoint-resume flow is: log `"Checkpoint resume triggered: step {step}"` (MonitorWorker picks this up), then return `False` from `_check_and_accumulate_gradients` to skip the current step. The pretraining behavior (kill on NaN) was appropriate for "fail fast" pretraining; fine-tuning needs retry because gradient instability is more common with LoRA on small corpora. Add `finetune_mode: bool = False` to `AutoStepOptimizer.__init__()` — only apply the retry behavior when `finetune_mode=True`, preserving pretraining behavior.

  **Bead state transitions:** State transitions (corpus-validating → ready → training → syncing → checkpointing → evaluating → done/failed/degraded) are written directly to the Dolt corpus database (MySQL protocol, port 3307), not via `bd` CLI calls. The training job's `bead_id` is passed in at launch; state transitions update a `training_jobs` table in Dolt. The full state sequence is: `corpus-validating` → `ready` → `training` → `syncing` → `checkpointing` → `evaluating` → `done`/`failed`/`degraded`.
- **Acceptance criteria:**
  - [ ] `finetune_mode=True`: NaN grads raise `RetriableError`, logged as warning, step skipped
  - [ ] `finetune_mode=False`: NaN grads still call `os.killpg` (pretraining unchanged)
  - [ ] Tests: `tests/unit/test_optim_nan.py` — inject NaN gradient tensor; assert exception vs kill behavior by mode
- **Dependencies:** Task 1.1 (finetune_mode concept established in Phase 1)

**3.5 Fine-tuning YAML configs**
- **What:** Create per-role fine-tuning YAML configuration files following the existing `llama_8B_C.yaml` format.
- **Files:**
  - Create: `src/node0/configs/finetune/lora_sft_deacon.yaml` — deacon role SFT config
  - Create: `src/node0/configs/finetune/lora_sft_witness.yaml` — witness role SFT config
  - Create: `src/node0/configs/finetune/lora_sft_polecat.yaml` — polecat role SFT config
  - Create: `src/node0/configs/finetune/dpo_polecat.yaml` — polecat DPO config
- **Key details:** Each YAML follows the existing structure: `model_config`, `optim_config`, `grad_avg_config`, `scheduler`, etc. Add top-level `finetune_config` section:
  ```yaml
  finetune_config:
    class_name: node0.finetune.config.FineTuneConfig
    init_args:
      mode: sft
      role: deacon
      corpus_path: ~/gt/.corpus/deacon/
      checkpoint_dir: ~/gt/.checkpoints/deacon/
      checkpoint_every_n_steps: 100
      eval_every_n_steps: 500
      lora:
        rank: 64
        alpha: 128.0
        dropout: 0.05
        target_modules: ["wq", "wk", "wv", "wo"]
  ```
  Correct `vocab_size: 32000` in `model_config.init_args`. LR for fine-tuning is lower: `lr: 0.00003` (vs pretraining `0.0003`). Add `num_warmup_steps: 100` (vs pretraining `4000`).
- **Acceptance criteria:**
  - [ ] Each YAML loads without error via `build_cls()`
  - [ ] `finetune_config` section parses to valid `FineTuneConfig` instance
  - [ ] `vocab_size: 32000` in all fine-tuning configs
- **Dependencies:** Tasks 1.1, 1.5

#### Phase 3 Exit Criteria
- [ ] LoRA adapters applied to Expert classes; only LoRA params have `requires_grad=True`
- [ ] `node0_server.py` passes only LoRA params to GradientAverager and optimizer in finetune mode
- [ ] DataFeeder thread delivers batches to tail stage for loss computation
- [ ] Training loss logged each step in format MonitorWorker can parse
- [ ] NaN handler is graceful in finetune mode
- [ ] Fine-tuning YAML configs exist and parse correctly
- [ ] All unit tests pass

---

### Phase 4: Checkpoint Export and Model Registry

**Objective:** Build the checkpoint gathering and export pipeline, and the model registry that tracks trained models. After Phase 4, a completed training run produces a versioned model artifact in the registry, queryable via `gt model list`.

**Prerequisites:** Phase 3 (training must run to produce checkpoints); Phase 1 (ModelRegistryEntry schema needs FineTuneConfig for training_config snapshot).

#### Tasks

**4.1 Checkpoint shard saving (per stage)**
- **What:** Each pipeline stage saves its local shard of the model (LoRA parameters + optionally full stage weights) to disk on checkpoint events.
- **Files:**
  - Create: `src/node0/finetune/checkpoint/__init__.py` — empty
  - Create: `src/node0/finetune/checkpoint/shard.py` — `save_shard(module: nn.Module, checkpoint_dir: Path, stage: str, step: int) -> Path`: extracts LoRA state dict (`get_lora_parameters()` names + tensors), saves as `{checkpoint_dir}/step_{step}/{stage}_lora.safetensors` using `safetensors.torch.save_file()`; `load_shard(checkpoint_dir: Path, stage: str, step: int) -> dict`: loads and returns state dict
- **Key details:** Use `safetensors.torch.save_file()` and `safetensors.torch.load_file()` — not `torch.save()` — for safe serialization (no pickle). Checkpoint directory structure: `{checkpoint_dir}/step_{step}/{stage}_lora.safetensors`. Trigger: `ModuleCollab.on_backward()` should check `step % finetune_config.checkpoint_every_n_steps == 0` and call `save_shard()`. Log `"Checkpoint saved: step {step} stage {stage}"` so MonitorWorker captures it.
- **Acceptance criteria:**
  - [ ] `save_shard()` creates `.safetensors` file at correct path
  - [ ] `load_shard()` returns dict matching saved state dict
  - [ ] Files use safetensors format (not pickle)
  - [ ] Tests: `tests/unit/test_checkpoint_shard.py` — save/load round-trip with synthetic LoRA module; use `tmp_path`
- **Dependencies:** Task 3.1 (LoRA adapter to extract LoRA state dict)

**4.2 Checkpoint gatherer (distributed assembly)**
- **What:** Coordinate gathering all 32 pipeline stage shards into a single assembled state dict via DHT.
- **Files:**
  - Create: `src/node0/finetune/checkpoint/gatherer.py` — `CheckpointGatherer`: `announce_shard(dht: DHT, experiment_prefix: str, stage: str, step: int, shard_path: str) -> None`: stores shard path in DHT under `{experiment_prefix}_checkpoint_{step}_{stage}`; `gather_all_shards(dht: DHT, experiment_prefix: str, step: int, expected_stages: list[str], timeout: float = 300.0) -> dict[str, Path] | None`: polls DHT until all expected stages have announced their shard paths or timeout; returns stage→path mapping or None on timeout
  - Create: `src/node0/finetune/checkpoint/assembler.py` — `assemble_checkpoint(stage_shards: dict[str, Path], output_path: Path) -> Path`: loads each shard, assembles into ordered state dict (head stages first, body stages in order, tail last), saves assembled model as single `merged_lora.safetensors`
- **Key details:** DHT is the coordination bus (constraint from plan-context.md). `announce_shard()` calls `dht.store(key, value, expiration_time=...)`. `gather_all_shards()` polls with `dht.get(key)` in a loop with sleep, using the `call_with_retries()` pattern from `utils/node_info.py` (corrected import path) for transient DHT failures. Expected stages list comes from `finetune_config` or is derived from `num_hidden_layers` (same as pretraining stage assignment). Timeout of 300 seconds handles straggler peers.
- **Acceptance criteria:**
  - [ ] `announce_shard()` stores to DHT without error (mocked DHT)
  - [ ] `gather_all_shards()` returns `None` after timeout if not all stages announce
  - [ ] `assemble_checkpoint()` produces single safetensors file from multiple shards
  - [ ] Tests: `tests/unit/test_checkpoint_gatherer.py` — mock DHT; test timeout and complete-gather cases
- **Dependencies:** Task 4.1

**4.3 GGUF export**
- **What:** Convert assembled safetensors checkpoint to GGUF format for local inference.
- **Files:**
  - Create: `src/node0/finetune/checkpoint/exporter.py` — `export_to_gguf(safetensors_path: Path, output_path: Path, quantization: str = "q4_k_m") -> Path`: shells out to `llama.cpp`'s `convert.py` and `quantize` binary; logs progress; returns output path
- **Key details:** GGUF export via `llama.cpp` (not a Python library dependency — shell out to the binary). This means `llama.cpp` must be available on the export node. The exporter wraps the subprocess call with `subprocess.run(..., check=True, capture_output=True)`. Failure raises `NonRetriableError(f"GGUF export failed: {result.stderr}")` imported from `src/node0/utils/node_info.py` (corrected import path). The safetensors format is the primary artifact; GGUF is for local inference compatibility.
- **Acceptance criteria:**
  - [ ] `export_to_gguf()` calls subprocess with correct arguments
  - [ ] Subprocess failure raises `NonRetriableError`
  - [ ] Tests: `tests/unit/test_exporter.py` — mock subprocess; assert argument construction
- **Dependencies:** Task 4.2

**4.4 Model registry store and Dolt schema**
- **What:** Create `ModelRegistryEntry` schema and Dolt-backed registry store.
- **Files:**
  - Create: `src/node0/finetune/registry/__init__.py` — empty
  - Create: `src/node0/finetune/registry/schema.py` — `ModelRegistryEntry` Pydantic BaseModel (fields as defined in Shared Abstractions)
  - Create: `src/node0/finetune/registry/store.py` — `ModelRegistry`: `register(entry: ModelRegistryEntry) -> None`: writes to Dolt `model_registry` table via MySQL connector (Dolt server runs at `127.0.0.1:3307` per `.beads/metadata.json`); `get(name: str) -> ModelRegistryEntry | None`; `list(role: str | None = None) -> list[ModelRegistryEntry]`; `update_eval_score(name: str, eval_score: float, by_category: dict) -> None`; `update_deployment_status(name: str, status: str, rigs: list[str]) -> None`
  - Create: `src/node0/finetune/registry/migrations/001_model_registry.sql` — DDL for `model_registry` table (schema from plan-context.md data layer section); DDL for `dpo_preferences` table
- **Key details:** Connect to Dolt at `127.0.0.1:3307` using `mysql-connector-python` (already available if Dolt server mode is running). Use standard MySQL connection pattern. The `corpus_hash` field links a registry entry to the corpus snapshot used for training (from `CorpusStore.corpus_hash()`). The `artifact_path` field stores the path to the assembled safetensors file.
- **Acceptance criteria:**
  - [ ] `ModelRegistry.register()` inserts row into Dolt (with mock DB connection in tests)
  - [ ] `ModelRegistry.list()` returns entries filtered by role
  - [ ] SQL migration file is valid DDL
  - [ ] Tests: `tests/unit/test_registry_store.py` — mock MySQL connection; assert SQL statements
- **Dependencies:** None (within Phase 4; independent of 4.1-4.3 except for full integration)

**4.5 Model registry CLI (`gt model`)**
- **What:** CLI commands `gt model list`, `gt model show`, `gt model adapters`, `gt model deploy`, `gt model rollback`, `gt model prune`.
- **Files:**
  - Create: `src/node0/finetune/registry/cli.py` — `main()` entry point with argparse subcommands matching the spec CLI interface; `list_models(role: str | None)` → tabular output matching spec format; `show_model(name: str)` → detailed output; `list_adapters()` → adapter table; `deploy_model(name: str, rigs: list[str])` → updates `deployment_status` and `deployed_to` in registry; `rollback_model(role: str, version: int)` → sets specified version to `active`, previous to `archived` in town settings; `prune_models(role: str | None, keep_last: int = 3)` → removes all but the N most recent versions per role
- **Key details:** Output format must match spec exactly (column headers, spacing). Use `tabulate` library or manual formatting (prefer manual to avoid new deps). `rollback_model()` modifies `/home/ubuntu/gt/settings/config.json` — reads current JSON, updates the relevant role-to-agent mapping to reference the rollback version, writes back. This is the only task that touches `settings/config.json` directly from node0 code.

  **`gt model prune` details:** `prune_models(role, keep_last=3)`: queries registry for all versions of each model name/role, sorts by `created_at` descending, deletes all but the `keep_last` most recent. Deletion means: (a) delete artifact files from `~/gt/.models/<name>/`; (b) delete Dolt row from `model_registry` table. Add `--dry-run` flag (print what would be deleted without deleting). Active models (status `"active"`) are never pruned regardless of `keep_last` — only `"archived"` and `"flagged"` versions are candidates. Add retention policy to CLI help text: `"Active models are always retained. Only archived/flagged versions are pruned."`.
- **Acceptance criteria:**
  - [ ] `gt model list` prints table with NAME, ROLE, BASE, VER, EVAL, STATUS columns
  - [ ] `gt model show mayor-v3` prints all fields from spec example
  - [ ] `gt model rollback --role deacon --version 1` updates town settings JSON
  - [ ] `gt model prune --role deacon --keep-last 2` deletes archived versions beyond the 2 most recent
  - [ ] `gt model prune --dry-run` prints what would be deleted without deleting
  - [ ] Active models are never pruned by `prune_models()`
  - [ ] Tests: `tests/unit/test_registry_cli.py` — mock registry store; assert output format and prune behavior
- **Dependencies:** Task 4.4

#### Phase 4 Exit Criteria
- [ ] `save_shard()` → `announce_shard()` → `gather_all_shards()` → `assemble_checkpoint()` → `export_to_gguf()` pipeline works end-to-end
- [ ] `ModelRegistry.register()` persists to Dolt
- [ ] `gt model list` and `gt model show` produce correct output
- [ ] SQL migration file creates valid Dolt tables
- [ ] All unit tests pass

---

### Phase 5: Autoregressive Inference and Provider Integration

**Objective:** Build the autoregressive generation loop on top of node0's existing gRPC pipeline forwarding, then wire it into Gas Town as the "node0" provider. After Phase 5, `gt sling` can route tasks to a node0-inference agent.

**Prerequisites:** Phase 3 (LoRA adapter loading must exist for inference to load adapter weights); Phase 4 (model registry must exist for inference to look up model by name).

#### Tasks

**5.1 KV-cache management**
- **What:** Per-stage KV-cache to avoid recomputing attention keys/values for previously generated tokens.
- **Files:**
  - Create: `src/node0/finetune/inference/__init__.py` — empty
  - Create: `src/node0/finetune/inference/kv_cache.py` — `KVCache`: stores `(key, value)` tensors per attention layer; `update(layer_idx: int, new_key: torch.Tensor, new_value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]`: appends new KV to cache and returns full (past + new) KV for attention; `reset() -> None`: clears cache for new sequence; `__len__()`: returns current sequence length
- **Key details:** Each pipeline stage only has the layers it owns, so each stage's `KVCache` only stores KV for its own layers. The cache is attached to the Expert (one cache per expert instance). During inference, the head stage feeds the full prompt on first pass, then feeds one token per pass, using the KV cache to avoid recomputation. Cache size limit: `max_seq_len` from `LlamaArguments` (2048 after Task 1.1 fix).
- **Acceptance criteria:**
  - [ ] `KVCache.update()` returns concatenated (past + new) KV tensors
  - [ ] `KVCache` length grows by 1 per token generated
  - [ ] `reset()` clears cache to zero length
  - [ ] Tests: `tests/unit/test_kv_cache.py` — synthetic tensors; assert shapes after update and reset
- **Dependencies:** None (pure tensor logic)

**5.2 Sampling logic**
- **What:** Token sampling with temperature, top-p, and top-k.
- **Files:**
  - Create: `src/node0/finetune/inference/sampling.py` — `sample_token(logits: torch.Tensor, temperature: float = 1.0, top_p: float = 0.9, top_k: int = 0) -> int`: applies temperature scaling, optional top-k filtering, optional nucleus (top-p) filtering, then samples from resulting distribution; returns token id as int
- **Key details:** Standard sampling implementation: `logits / temperature` → `top_k_filter(logits, k)` → `top_p_filter(logits, p)` → `torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)`. `top_k=0` means no top-k filter. `top_p=1.0` means no nucleus filter. Greedy decoding = `temperature→0` (set to 0.01 minimum to avoid division by zero). This is pure PyTorch arithmetic with no distributed dependencies.
- **Acceptance criteria:**
  - [ ] `sample_token` with `temperature=0.01` is near-greedy (highest logit token with high probability)
  - [ ] Output is always a valid int in `[0, vocab_size)`
  - [ ] Tests: `tests/unit/test_sampling.py` — synthetic logits; assert distribution properties
- **Dependencies:** None

**5.3 Autoregressive generation loop**
- **What:** The generation loop that decodes one token at a time using the pipeline's forward pass.
- **Files:**
  - Create: `src/node0/finetune/inference/generator.py` — `Generator`: holds reference to Hivemind's `Server` (reuses existing gRPC pipeline); `generate(prompt_ids: list[int], max_new_tokens: int = 512, temperature: float = 1.0, top_p: float = 0.9, stop_token_ids: list[int] | None = None) -> Generator[int, None, None]`: Python generator that yields token IDs as they are produced; internally: (1) feed full prompt through pipeline to get first logits, (2) sample next token, (3) feed single token + KV cache through pipeline, (4) repeat until EOS or max_new_tokens
- **Key details:** The gRPC pipeline forwarding (activations between stages) is the existing Hivemind MOE forward pass — `Generator` triggers it via the existing `Server` interface. The generation loop sends a single tensor (embedding of current token) as input to the head stage and receives logits from the tail stage via the existing gRPC activation passing. KV cache is managed on each stage locally (injected via the `KVCache` attached to each Expert). Streaming: yield each token ID immediately after sampling (don't buffer full response).
- **Acceptance criteria:**
  - [ ] `generator.generate(prompt_ids)` yields token IDs one at a time
  - [ ] Generation stops at EOS token or max_new_tokens
  - [ ] KV cache is reset between independent generation calls
  - [ ] Tests: `tests/unit/test_generator.py` — mock pipeline forward pass; assert token sequence properties
- **Dependencies:** Tasks 5.1, 5.2

**5.4 Prompt API server**
- **What:** stdin/stdout prompt API for tmux-compatible agent usage.
- **Files:**
  - Create: `src/node0/finetune/inference/prompt_api.py` — `PromptAPI`: reads prompt from stdin (line-buffered), tokenizes with loaded tokenizer, calls `Generator.generate()`, decodes and streams tokens to stdout, flushes after each token; handles SIGINT cleanly
  - Create: `src/node0/run_inference.py` — `main()`: argparse with `--model` (registry name), `--adapters` (comma-separated adapter names), `--temperature`, `--top-p`, `--max-tokens`; loads model from registry (artifact path from `ModelRegistry.get(model_name)`); loads LoRA adapter weights; starts `PromptAPI` loop
- **Key details:** `run_inference.py` is the `node0-inference` script registered in `pyproject.toml`. Town settings reference it as `"command": "node0-inference"`. Must work in tmux (unbuffered stdout — use `sys.stdout.flush()` after each write). The `--model` arg is the registry name (e.g., `"deacon-v1"`); `ModelRegistry.get()` returns the `artifact_path`. Load the LoRA adapter weights from registry artifact path, apply to base model, then run generation. Follow `run_server.py` startup pattern: logger first, then component initialization.
- **Acceptance criteria:**
  - [ ] `echo "hello" | node0-inference --model deacon-v1` produces output without hanging
  - [ ] Each token is written to stdout immediately (no buffering)
  - [ ] `--model unknown-model` exits with clear error message
  - [ ] Tests: `tests/unit/test_prompt_api.py` — mock generator; assert stdout output
- **Dependencies:** Tasks 4.4, 5.3

**5.5 RAG layer (ChromaDB integration)**
- **What:** Embed Gas Town dynamic state (beads, configs, recent sessions, formula definitions) in ChromaDB and retrieve relevant context at inference time.
- **Files:**
  - Create: `src/node0/finetune/rag/__init__.py` — empty
  - Create: `src/node0/finetune/rag/indexer.py` — `RAGIndexer`: `__init__(persist_dir: str)` initializes ChromaDB client with `chromadb.PersistentClient(path=persist_dir)`; `index_bead(bead_id: str, content: str) -> None`; `index_config(path: str, content: str) -> None`; `index_session(session_id: str, content: str) -> None`; `index_formula(formula_path: str, content: str) -> None`; each uses a separate ChromaDB collection (`beads`, `configs`, `sessions`, `formulas`)
  - Create: `src/node0/finetune/rag/retriever.py` — `RAGRetriever`: `retrieve(query: str, n_results: int = 5) -> str`: queries all four collections, merges top results by relevance score, formats as a context string prepended to the prompt
  - Modify: `src/node0/finetune/inference/prompt_api.py` — before calling `generator.generate()`, call `RAGRetriever.retrieve(prompt_text)` and prepend the retrieved context to the prompt (if RAG is enabled via `--rag-dir` flag)
  - Modify: `src/node0/run_corpus.py` — add `rag-index-formulas` subcommand: reads all `*.formula.toml` files from `~/gt/.beads/formulas/`, calls `RAGIndexer.index_formula()` for each; uses the formula `description` field and step descriptions as the primary text for embedding
- **Key details:** ChromaDB `PersistentClient` stores vectors on disk at `persist_dir` (default `~/gt/.rag/`). Embedding function: ChromaDB's default (sentence-transformers, no separate dep needed). Update triggers: a `gt hook bd_sync` script calls `node0-corpus rag-index --bead-id <id>` after each bead sync; formula indexing triggered by `node0-corpus rag-index-formulas`. The `--rag-dir` flag is optional in `run_inference.py` — if absent, RAG is disabled (fine-tuned model handles context without it).

  **Formula indexing:** For each `.formula.toml`, extract the `description` field from `[formula]` and the `description` field from each step. Concatenate as the embedding text: `"Formula: {name}\n{formula_description}\nSteps:\n{step_descriptions}"`. Store with metadata `{"path": formula_path, "name": formula_name}`. Index from `~/gt/.beads/formulas/` (the correct Gas Town formula path, not `~/.beads/formulas/`).
- **Acceptance criteria:**
  - [ ] `RAGIndexer.index_bead()` stores embedding in ChromaDB `beads` collection
  - [ ] `RAGIndexer.index_formula()` stores embedding in ChromaDB `formulas` collection
  - [ ] `RAGRetriever.retrieve("create a bead")` returns non-empty context string (may include formula definitions)
  - [ ] `node0-corpus rag-index-formulas` indexes all formula TOMLs from `~/gt/.beads/formulas/`
  - [ ] RAG context is prepended to prompt when `--rag-dir` is provided
  - [ ] Tests: `tests/unit/test_rag.py` — use ChromaDB in-memory client (ephemeral client for testing); test formula indexing and retrieval
- **Dependencies:** Task 5.4 (PromptAPI is the integration point)

**5.5b Modify gt prime for fine-tuned model context reduction**
- **What:** Modify `gt prime` in the Gas Town gastown rig to detect when a fine-tuned node0 model is active for a given role and inject a reduced, model-appropriate context (~2-3K tokens) instead of the full ~15K token prime. This directly addresses one of the four primary motivations for the project (prime bloat).
- **Files:**
  - Modify: `~/gt/gastown/<prime-command-file>` — the file that implements `gt prime` in the gastown rig (investigate exact path by running `gt help prime` or inspecting `~/gt/gastown/`); add a model-awareness check: (a) read the active model for the current role from town settings (`/home/ubuntu/gt/settings/config.json`); (b) query `ModelRegistry.get(model_name)` to check if `deployment_status == "active"` and model is a node0 fine-tuned model; (c) if fine-tuned model is active for this role, cap prime context to ~2-3K tokens by omitting large static sections (detailed command references, full CLAUDE.md) and keeping only the most dynamic/essential state (current beads, recent changes, active role); (d) include a small header `"# Reduced prime: fine-tuned model active for role {role}"` so the reduction is visible in transcripts
- **Key details:** The exact file to modify requires investigation — run `gt help prime` or inspect `~/gt/gastown/` directory to find the prime command implementation. The model-awareness check reads `config.json` role-to-agent mapping, then checks if the assigned agent is a node0 agent (has `"provider": "node0"` field). If node0 is active, use reduced prime template. If not active (standard Claude/pi/omp agent), use full prime unchanged. This is a conditional branch — no regression to non-node0 agents. Context reduction target: from ~15K tokens to ~2-3K tokens of RAG-injected dynamic state (RAG from Task 5.5 fills the remaining gap). Add acceptance criterion to validate final context size.
- **Acceptance criteria:**
  - [ ] When `gt prime` is run with a node0 role assignment active, context output is <= 3K tokens
  - [ ] When `gt prime` is run with a non-node0 role assignment, full context is unchanged
  - [ ] Prime output includes `"# Reduced prime: fine-tuned model active"` header when reduced
  - [ ] `gt prime` does not error if model registry is unavailable (falls back to full prime)
- **Dependencies:** Tasks 4.4 (ModelRegistry.get()), 5.6 (node0 agents registered in town settings)

**5.6 Register "node0" provider in town settings**
- **What:** Add node0 agent entries and initial role mappings to Gas Town settings.
- **Files:**
  - Modify: `/home/ubuntu/gt/settings/config.json` — add agent entries for `node0-deacon`, `node0-witness`, `node0-polecat` following the spec JSON structure; initially do NOT switch role assignments (deacon/witness/polecat still point to existing agents); progressive rollout happens in Phase 6
- **Key details:** The agent entries follow the existing agent definition format in `config.json` (same structure as claude, pi, omp agents). The `"provider": "node0"` field is new. The `"prompt_mode": "stdin"` field tells Gas Town to pipe prompts via stdin (matching `PromptAPI`'s stdin loop). Do NOT switch roles until Phase 6 eval gates are passed.
- **Acceptance criteria:**
  - [ ] `config.json` is valid JSON after modification
  - [ ] `node0-deacon` agent entry exists with correct fields
  - [ ] Existing role assignments are unchanged (no disruption to current Gas Town operation)
- **Dependencies:** None (config file change; independent of node0 Python code)

#### Phase 5 Exit Criteria
- [ ] `node0-inference --model <name>` accepts stdin and streams response to stdout
- [ ] Generation loop produces coherent token sequences (even without fine-tuning)
- [ ] `config.json` has node0 agent entries
- [ ] RAG indexer and retriever work end-to-end with ChromaDB
- [ ] `gt prime` detects active node0 model and caps context to <= 3K tokens for that role
- [ ] All unit tests pass

---

### Phase 6: Evaluation Suite and Progressive Rollout

**Objective:** Build the automated evaluation task suite that gates model deployment, then execute the progressive rollout: deacon → witness → polecat. After Phase 6, the first Gas Town-native agents are live.

**Prerequisites:** Phase 4 (registry must exist to store eval scores); Phase 5 (inference must work for the eval runner to query the model).

#### Tasks

**6.1 Gas Town corpus hooks**
- **What:** Wire corpus collection into Gas Town's session lifecycle via gt hooks.
- **Files:**
  - Create: a hook script at the Gas Town hook path for `session_shutdown` — calls `node0-corpus collect --session $GT_SESSION_NAME --bead-id $GT_BEAD_ID --role $GT_ROLE --rig $GT_RIG --outcome $GT_OUTCOME` (exact hook path and environment variable names need investigation per the Gas Town rig structure open question Q52)
  - Create: a hook script for `bd_sync` — calls `node0-corpus rag-index --bead-id $GT_BEAD_ID` to update ChromaDB after each bead sync
- **Key details:** The exact hook path needs investigation — run `gt help hooks` or inspect Gas Town config to find the hook registration mechanism. The hook script must be idempotent (calling twice for same session produces one corpus entry). Add `rag-index` subcommand to `run_corpus.py` CLI: takes `--bead-id`, fetches bead content from Dolt, indexes in ChromaDB.
- **Acceptance criteria:**
  - [ ] Closing a polecat bead creates a corpus entry file in `~/gt/.corpus/polecat/`
  - [ ] `bd sync` on a bead updates the ChromaDB `beads` collection
  - [ ] Hook script is idempotent
- **Dependencies:** Tasks 2.4, 5.5

**6.2 Eval task suite — bead management**
- **What:** Automated eval tasks for bead management category.
- **Files:**
  - Create: `src/node0/finetune/eval/__init__.py` — empty
  - Create: `src/node0/finetune/eval/schema.py` — `EvalResult` Pydantic BaseModel
  - Create: `src/node0/finetune/eval/tasks/__init__.py` — empty
  - Create: `src/node0/finetune/eval/tasks/bead_management.py` — `BeadManagementEval`: list of task prompts and expected outcomes; `run(model_name: str, inference_fn: Callable) -> list[EvalResult]`; tasks: (1) "Create a bead for task X" — validate output has correct fields (title, description, status); (2) "Close bead B with reason R" — validate status transition; (3) "Build an epic hierarchy for project P" — validate DAG structure (cycle-free, parallelism ratio > 0.5)
- **Key details:** Eval tasks are prompt strings sent to the inference function. The `inference_fn` takes a prompt string and returns a response string (wraps `PromptAPI` or mocked for tests). Scoring: binary per-field validation (field present and correctly formatted = 1, else 0). Category score = fraction of checks passed. Log `"Eval score [bead_management]: {score:.4f}"` so MonitorWorker picks it up.
- **Acceptance criteria:**
  - [ ] `BeadManagementEval.run()` returns list of `EvalResult` instances
  - [ ] Cycle detection works for epic hierarchy validation
  - [ ] Parallelism ratio computed correctly
  - [ ] Tests: `tests/unit/test_eval_bead_management.py` — mock inference_fn returning good/bad responses
- **Dependencies:** Task 5.4 (inference function interface)

**6.3 Eval task suite — git workflow, conventions, planning, code execution**
- **What:** Automated eval tasks for remaining categories.
- **Files:**
  - Create: `src/node0/finetune/eval/tasks/git_workflow.py` — tasks: branch-commit-push sequence (validate no --force), PR description generation (validate format), merge conflict resolution (validate clean merge); scoring: binary per-step
  - Create: `src/node0/finetune/eval/tasks/convention_adherence.py` — tasks: use `gt` commands not raw git, follow CLAUDE.md instructions, no `pkill`; scoring: binary per-violation-count (0 violations = 1.0)
  - Create: `src/node0/finetune/eval/tasks/planning.py` — tasks: generate bead hierarchy (cycle-free + parallelism ratio), estimate complexity, suggest dependency chains; scoring: cycle-free boolean + parallelism ratio
  - Create: `src/node0/finetune/eval/tasks/code_execution.py` — tasks: implement from bead description, fix bug, refactor; scoring: test pass rate + regression count
- **Key details:** All task modules follow the same interface: `class <Category>Eval: def run(model_name: str, inference_fn: Callable) -> list[EvalResult]`. Convention adherence checks response text for disallowed patterns (regex for `git push --force`, `pkill`, raw `git` commands where `gt` equivalents exist). Code execution tasks use a subprocess sandbox (run generated code in `subprocess.run(..., timeout=30)` and check return code).
- **Acceptance criteria:**
  - [ ] Each eval module's `run()` returns `list[EvalResult]`
  - [ ] All modules share the same interface (duck typing)
  - [ ] Tests: at least one test per module with mock inference_fn
- **Dependencies:** Task 6.2 (establishes pattern and EvalResult schema)

**6.4 Eval runner and scorer**
- **What:** Orchestrate all eval categories, compute composite score, update model registry.
- **Files:**
  - Create: `src/node0/finetune/eval/runner.py` — `EvalRunner`: `run_all(model_name: str, inference_fn: Callable) -> dict[str, float]`: runs all five category evals, returns category→score dict; `composite_score(scores: dict[str, float], weights: dict[str, float] | None = None) -> float`: weighted average; if `weights=None`, equal weighting
  - Create: `src/node0/finetune/eval/scorer.py` — `update_registry_with_scores(model_name: str, scores: dict, composite: float) -> None`: calls `ModelRegistry.update_eval_score()`
- **Key details:** `EvalRunner` logs `"Eval score [{category}]: {score:.4f}"` per category (MonitorWorker picks up). Weights are deferred to implementation per spec decision (empirically determined after first training run) — defaulting to equal weights is the V1 behavior. The eval runner is invoked post-checkpoint in the training workflow: `ModuleCollab.on_backward()` checks `step % finetune_config.eval_every_n_steps == 0` and triggers eval asynchronously (in a separate thread to not block gradient accumulation).
- **Acceptance criteria:**
  - [ ] `EvalRunner.run_all()` calls all five category evals
  - [ ] Composite score is weighted average of category scores
  - [ ] Registry is updated with scores after eval run
  - [ ] Tests: `tests/unit/test_eval_runner.py` — mock all category evals; assert composite computation
- **Dependencies:** Tasks 4.4, 6.2, 6.3

**6.5 Training formulas**
- **What:** Gas Town formula TOML files for training orchestration.
- **Files:**
  - Create: `~/gt/.beads/formulas/train-role-model.formula.toml` (i.e. `/home/ubuntu/gt/.beads/formulas/train-role-model.formula.toml`) — training pipeline formula (per spec structure: prepare → train → eval → export steps)
  - Create: `~/gt/.beads/formulas/corpus-collect.formula.toml` — corpus collection formula (batch export for wisps/non-bead sessions)
  - Create: `~/gt/.beads/formulas/eval-suite.formula.toml` — standalone eval suite formula (run eval against a named model)
- **Key details:** Formulas use `[[steps]]` blocks with `description` and `command` fields. By Phase 6, the concrete commands are known: `train-role-model` steps are: `node0-corpus validate`, `python -m node0.run_finetune --config ...`, `node0-corpus eval --model ...`, `node0-corpus export --model ...`. Variables: `role`, `base_model`, `corpus_path`, `lora_rank`. Follow existing formula TOML format from `~/gt/.beads/formulas/` (i.e. `/home/ubuntu/gt/.beads/formulas/`). Note: `~/.beads/` and `~/gt/.beads/` are distinct directories — formulas must go in the Gas Town rig path `/home/ubuntu/gt/.beads/formulas/`.
- **Acceptance criteria:**
  - [ ] Each TOML file is valid TOML (parseable)
  - [ ] Formula variables are declared with types
  - [ ] Steps have `description` and `command` fields
- **Dependencies:** Tasks 2.4, 5.4, 6.4 (commands must exist before formulas reference them)

**6.6 Progressive rollout: deacon and witness (Stage 1 and Stage 2)**
- **What:** Switch deacon and witness roles to use node0 agents in town settings, after eval gate passes. The eval gate is automatically enforced — `gt model deploy` refuses to advance if composite eval score is below threshold.
- **Files:**
  - Modify: `/home/ubuntu/gt/settings/config.json` — change `"deacon": "<current-agent>"` to `"deacon": "node0-deacon"` and `"witness": "<current-agent>"` to `"witness": "node0-witness"` ONLY after deacon/witness models pass eval threshold
  - Modify: `src/node0/finetune/registry/cli.py` — update `deploy_model()` to enforce eval gate: before modifying town settings, check `ModelRegistry.get(name).eval_score >= threshold`; if below threshold, raise error with message `"Model score {score:.1%} below deployment threshold {threshold:.1%}. Run eval suite again or override with --force-deploy."` The gate must be automatic — not just a manual check.
- **Key details:** This is the deployment gate. The eval runner (Task 6.4) must report scores above threshold before this change is made. `gt model deploy` now enforces this automatically. `--force-deploy` flag allows operator override with explicit acknowledgment. Rollback: `gt model rollback --role deacon --version 0` (restores original agent) via Task 4.5. The rollout controller must refuse to advance stages if composite eval score is below threshold, not just log a warning.
- **Acceptance criteria:**
  - [ ] `gt model deploy deacon-v1` fails with clear error if eval_score < threshold
  - [ ] `gt model deploy deacon-v1 --force-deploy` succeeds with warning even if below threshold
  - [ ] `config.json` has deacon/witness pointing to node0 agents after successful deployment
  - [ ] `gt sling` with deacon role routes to `node0-inference`
  - [ ] Rollback restores original role assignment
- **Dependencies:** Tasks 4.5, 5.6, 6.4

**6.7 Progressive rollout: polecat (Stage 3)**
- **What:** Switch polecat role to use node0 agents for specific task types, gated by eval suite scores. This is Stage 3 of the V1 progressive rollout (after deacon Stage 1 and witness Stage 2). Polecat is scoped to lower-risk task types initially: planning, bead creation, and architecture review. Code execution tasks remain on existing agents until confidence is established.
- **Files:**
  - Modify: `/home/ubuntu/gt/settings/config.json` — add `"node0-polecat"` agent entry; optionally switch polecat role to node0-polecat after eval gate passes; task-type routing (planning/bead-creation vs code-execution) may require conditional logic in town settings or a routing formula
  - Modify: `src/node0/finetune/registry/cli.py` — add polecat-specific eval gate check: polecat deployment requires `eval_score >= polecat_threshold` where polecat threshold may be higher than deacon/witness (more risk); add circuit breaker logic: if eval scores drop below threshold after deployment, `deploy_model()` sets `deployment_status = "degraded"` and reverts polecat role to previous agent
  - Create: a patrol agent modification (modify witness/deacon patrol agent scripts in gastown rig) — add fine-tuned model selection for polecat spawning: when a patrol agent spawns a polecat, check model registry for active polecat model; if available and eval score above threshold, use node0-polecat; circuit breaker falls back to base model if eval scores drop below threshold
- **Key details:** Polecat has higher blast radius than deacon/witness (it creates branches, writes code, manages dependencies). The circuit breaker is the key safety mechanism: if live eval scores (collected from polecat sessions post-deployment) regress below threshold, the circuit breaker automatically reverts polecat role to the previous agent without operator intervention. Eval scores are checked after each polecat session closes (using the session hook from Task 6.1). The specific polecat task types scoped for Stage 3: `planning`, `bead-create`, `architecture`. Code execution (`code-review`, `code-implement`) deferred until further confidence. Phase 4 `gt model rollback` provides the rollback mechanism.
- **Acceptance criteria:**
  - [ ] `gt model deploy polecat-v1` fails if polecat eval score < threshold
  - [ ] Circuit breaker reverts polecat role assignment when live eval scores drop below threshold
  - [ ] Fallback to base model works correctly (polecat sessions route to previous agent on circuit break)
  - [ ] Patrol agent modification selects node0-polecat when available and healthy
- **Dependencies:** Tasks 4.5, 5.6, 6.4, 6.6

**6.8 Corpus reversion detection**
- **What:** After eval completes (or during corpus audit), check if model eval scores have regressed versus the previous model version on key eval categories. If regression is detected, flag in model registry and hold progressive rollout.
- **Files:**
  - Create: `src/node0/finetune/eval/reversion.py` — `CorpusReversionDetector`: `check_regression(new_model: str, previous_model: str, registry: ModelRegistry) -> dict[str, float]`: compares `eval_scores_by_category` for new vs previous model, returns category → score delta dict (negative = regression); `has_significant_regression(deltas: dict[str, float], threshold: float = -0.05) -> bool`: returns True if any category regressed by more than threshold (default 5%); `flag_in_registry(model_name: str, regression_summary: dict, registry: ModelRegistry) -> None`: updates `deployment_status` to `"flagged"` and stores regression summary in registry; `hold_rollout(model_name: str) -> None`: ensures model cannot be deployed (sets status to `"holds_rollout"`)
  - Modify: `src/node0/finetune/eval/scorer.py` — after `update_registry_with_scores()`, call `CorpusReversionDetector.check_regression()` to compare against previous version; if regression detected, flag and hold rollout automatically
- **Key details:** "Corpus reversion" occurs when a new model performs worse than its predecessor on key eval categories — this may indicate corpus drift, CLAUDE.md contradictions, or bad training data. The regression check is automatic after every eval run: compare new model scores to the immediately previous version in the registry (`ModelRegistry.list(role=role)` sorted by version, take version N-1). If regression exceeds threshold on any category, set `deployment_status="flagged"` — `gt model deploy` will refuse to proceed (enforced by the gate in Task 6.6). The staleness tracker (Task 7.3) covers convention drift detection; this task covers score regression detection. Both complement each other.
- **Acceptance criteria:**
  - [ ] `check_regression()` returns negative deltas when new model scores lower than previous
  - [ ] `has_significant_regression()` returns True when any category drops > 5%
  - [ ] Model is flagged in registry and blocked from deployment on regression
  - [ ] Tests: `tests/unit/test_reversion_detector.py` — mock registry with two model versions; assert regression detection and flagging
- **Dependencies:** Tasks 4.4, 6.4

#### Phase 6 Exit Criteria
- [ ] Eval suite runs end-to-end against a model and produces category scores
- [ ] Composite score stored in model registry
- [ ] `gt model deploy` enforces eval gate automatically (refuses below threshold)
- [ ] Training formulas are valid and reference correct commands
- [ ] Corpus collection hook fires on session close
- [ ] Deacon and witness roles live on node0 agents (post eval gate) — Stage 1 and Stage 2
- [ ] Polecat rollout task implemented with circuit breaker — Stage 3
- [ ] Corpus reversion detector flags model regression automatically
- [ ] All unit tests pass

---

### Phase 7: DPO Preference Collection

**Objective:** Build the preference collection mechanism via Dolt branching and wire it into the retraining cycle. After Phase 7, the system accumulates human preference signal for DPO fine-tuning.

**Prerequisites:** Phase 5 (inference must work to run two models); Phase 4 (Dolt `dpo_preferences` table must exist).

#### Tasks

**7.1 DPO task routing (two-model sling)**
- **What:** Infrastructure to sling the same task to two model versions and produce two Dolt branches.
- **Files:**
  - Create: `src/node0/finetune/dpo/__init__.py` — empty
  - Create: `src/node0/finetune/dpo/router.py` — `DPORouter`: `create_comparison(task_bead_id: str, model_a: str, model_b: str) -> tuple[str, str]`: creates two Dolt branches from current `main` (e.g., `dpo-{bead_id}-a` and `dpo-{bead_id}-b`); returns branch names; `record_winner(task_bead_id: str, winner: str) -> None`: stores preference pair in `dpo_preferences` Dolt table
- **Key details:** Dolt branching via `dolt branch <name>` subprocess call (Dolt CLI available at `dolt`). Each model runs its task on its own branch. The operator evaluates the two outcomes and calls `record_winner()`. The `winner` argument is `"a"` or `"b"`. `record_winner()` maps to `chosen_branch`/`rejected_branch` in the `dpo_preferences` table.
- **Acceptance criteria:**
  - [ ] `create_comparison()` creates two Dolt branches (mock subprocess in tests)
  - [ ] `record_winner()` inserts row into `dpo_preferences` table
  - [ ] Tests: `tests/unit/test_dpo_router.py` — mock subprocess and DB; assert branch names and DB insert
- **Dependencies:** Task 4.4 (dpo_preferences table DDL)

**7.2 DPO corpus export**
- **What:** Convert accumulated `dpo_preferences` pairs into DPO training format (chosen/rejected corpus entries).
- **Files:**
  - Create: `src/node0/finetune/dpo/exporter.py` — `DPOExporter`: `export_pairs(role: str, output_path: Path) -> int`: queries `dpo_preferences` table for role, fetches session transcripts from chosen/rejected branches, formats as `CorpusEntry` objects with `chosen`/`rejected` fields populated, writes to `output_path`; returns count of exported pairs
- **Key details:** For each preference pair, the chosen branch's transcript becomes `CorpusEntry.chosen`, the rejected branch's transcript becomes `CorpusEntry.rejected`. These are loaded into `CorpusDataset` in DPO mode (Task 2.6's DPO handling). The exported pairs feed the `node0-finetune` run with `mode: dpo` in the config.
- **Acceptance criteria:**
  - [ ] `export_pairs()` produces `CorpusEntry` objects with both `chosen` and `rejected` populated
  - [ ] Exported count matches number of preference pairs in DB
  - [ ] Tests: `tests/unit/test_dpo_exporter.py` — mock DB; assert entry structure
- **Dependencies:** Tasks 7.1, 2.1

**7.3 Staleness detection**
- **What:** Track Gas Town convention changes (CLAUDE.md, formula TOMLs, key config files) and flag when the model corpus is stale. Extended beyond CLAUDE.md to cover formula and config changes per spec ("CLAUDE.md updates, new formulas, config changes").
- **Files:**
  - Create: `src/node0/finetune/corpus/staleness.py` — `ConventionStalenessTracker`: `check_drift(model_name: str) -> float`: compares CLAUDE.md hash at training time (stored in model registry `training_config`) with current CLAUDE.md hash; returns drift score (0.0 = no drift, 1.0 = completely different); `check_formula_drift(model_name: str) -> float`: computes hash of sorted formula TOML filenames + mtimes from `~/gt/.beads/formulas/`, compares to training-time formula hash stored in `training_config`; `check_config_drift(model_name: str) -> float`: computes hash of town settings config (`/home/ubuntu/gt/settings/config.json`) and compares to training-time config hash; `composite_staleness(model_name: str) -> float`: weighted average of all three drift signals (CLAUDE.md: 0.5, formulas: 0.3, config: 0.2); `should_retrain(model_name: str, threshold: float = 0.2) -> bool`: returns True if composite_staleness > threshold
- **Key details:** CLAUDE.md hash: `sha256(Path("/home/ubuntu/gt/CLAUDE.md").read_text())`. Formula directory hash: `sha256(",".join(sorted(str(p.stem) + str(p.stat().st_mtime) for p in Path("/home/ubuntu/gt/.beads/formulas/").glob("*.toml"))))`. Training-time hashes are stored in `ModelRegistryEntry.training_config` JSON (add `claude_md_hash`, `formula_dir_hash`, `config_hash` fields when registering a model). When `should_retrain()` returns True, log `logger.warning(f"Model {model_name} may be stale: composite_staleness={drift:.2f} (CLAUDE.md={cmd:.2f}, formulas={fd:.2f}, config={cd:.2f})")`. Trigger: run staleness check daily via a cron formula, or on every `gt prime` invocation.
- **Acceptance criteria:**
  - [ ] `check_drift()` returns 0.0 when CLAUDE.md unchanged
  - [ ] `check_drift()` returns >0.0 when CLAUDE.md has changed
  - [ ] `check_formula_drift()` returns >0.0 when a new formula TOML is added to `~/gt/.beads/formulas/`
  - [ ] `check_config_drift()` returns >0.0 when `config.json` changes
  - [ ] `composite_staleness()` is a weighted combination of all three signals
  - [ ] Tests: `tests/unit/test_staleness.py` — use `tmp_path` with fixture files; assert drift scores for each signal
- **Dependencies:** Task 4.4 (registry stores training_config)

#### Phase 7 Exit Criteria
- [ ] `create_comparison()` produces two Dolt branches for a task
- [ ] `record_winner()` stores preference pair in Dolt
- [ ] DPO corpus export produces valid CorpusEntry objects with chosen/rejected
- [ ] Staleness tracker detects drift in CLAUDE.md, formula TOMLs, and config files
- [ ] All unit tests pass

---

### Phase 8: Cross-Operator Corpus Sharing (V1+)

**Objective:** Implement opt-in cross-operator corpus contribution with an incentive model: operators who share anonymized corpus data receive extended model access quota. This is V1+ scope (post-launch of V1 core), not part of the V1 critical path. Phase 8 should only be started after Phase 7 is complete and V1 is stable.

**Prerequisites:** Phase 2 (corpus store and PII scrubber must exist); Phase 4 (model registry must be operational).

#### Tasks

**8.1 Opt-in corpus contribution flag**
- **What:** Add a per-rig flag to enable corpus sharing. When enabled, the rig's corpus is marked for contribution to the cross-operator pool after PII scrubbing and manual review.
- **Files:**
  - Modify: `/home/ubuntu/gt/settings/config.json` — add `"corpus_sharing": {"enabled": false, "rig_id": "<rig-identifier>", "anonymize": true}` field to rig settings; the `enabled` flag must be explicitly set to `true` by the operator (default is `false` — no sharing without opt-in)
  - Create: `src/node0/finetune/corpus/sharing.py` — `CorpusSharingExporter`: `export_for_sharing(store: CorpusStore, role: str, output_path: Path) -> int`: reads rig settings, checks `corpus_sharing.enabled`, applies additional anonymization layer (strips rig-specific identifiers, replaces rig_id with anonymized token), re-runs PII scrubber, writes anonymized corpus entries to `output_path` as JSONL; returns count of exported entries
  - Modify: `src/node0/run_corpus.py` — add `share` subcommand: `node0-corpus share --role polecat --output /tmp/shared-corpus.jsonl` triggers `CorpusSharingExporter`
- **Key details:** Anonymization layer: replace `rig` field value with `sha256(rig_id + secret_salt)[:8]` (consistent but unidentifiable); strip absolute file paths; remove operator-specific bead IDs. The `secret_salt` is a per-operator secret stored locally (not shared). The shared corpus does not include the salt — the anonymized rig token is one-way. Manual review gate: exported JSONL goes to a staging location and must be reviewed by the operator before transmission. No automatic upload — this is a human-in-the-loop step for V1.
- **Acceptance criteria:**
  - [ ] `node0-corpus share` fails with clear error if `corpus_sharing.enabled` is `false`
  - [ ] Exported JSONL has anonymized rig identifiers (not original rig name)
  - [ ] PII scrubber runs again on all exported content (double-scrub)
  - [ ] Tests: `tests/unit/test_corpus_sharing.py` — mock settings with sharing enabled/disabled; assert anonymization
- **Dependencies:** Tasks 2.1, 2.2

**8.2 Access grant mechanism**
- **What:** Simple access grant: operators who contribute corpus data (confirmed by manual review) receive extended model access quota. V1 implementation is manual review + flag in rig settings.
- **Files:**
  - Modify: `/home/ubuntu/gt/settings/config.json` — add `"model_access": {"tier": "standard", "corpus_contributor": false, "extended_quota": false}` fields; `corpus_contributor: true` is set manually after contribution is confirmed; `extended_quota: true` unlocks higher inference rate limits or priority access to newly trained models
  - Create: `src/node0/finetune/registry/access.py` — `AccessGrantManager`: `check_access(rig_settings: dict, requested_tier: str) -> bool`: reads `model_access` from settings, checks if rig has required tier; `grant_extended_access(config_path: Path) -> None`: sets `corpus_contributor: true` and `extended_quota: true` in settings (called manually by operator after contribution verified); `is_extended_quota(rig_settings: dict) -> bool`: returns `rig_settings["model_access"]["extended_quota"]`
- **Key details:** For V1, the "access grant" is entirely manual: after a corpus contribution is reviewed and accepted, the receiving operator sets `corpus_contributor: true` in the contributing operator's rig settings. No automated trust system, no API, no automatic verification. The incentive model is: contribute corpus → receive flag → extended model access (priority in training job scheduling or higher inference rate limit). Full automated trust and incentive system is deferred to V2 (federated Gas Town scope). The `grant_extended_access()` method is a utility for the manual operation.
- **Acceptance criteria:**
  - [ ] `AccessGrantManager.check_access()` returns False for standard tier requesting extended features
  - [ ] `grant_extended_access()` correctly updates `config.json` fields
  - [ ] `is_extended_quota()` returns True only when `extended_quota: true` in settings
  - [ ] Tests: `tests/unit/test_access_grant.py` — mock settings dicts; assert access checks
- **Dependencies:** Task 8.1

#### Phase 8 Exit Criteria
- [ ] `corpus_sharing.enabled` flag in rig settings controls opt-in sharing
- [ ] `node0-corpus share` exports anonymized corpus with double PII scrubbing
- [ ] Access grant mechanism reads extended_quota flag from settings
- [ ] All unit tests pass
- [ ] Manual review gate documented in operator runbook

---

## Cross-Cutting Concerns

### Error Handling

All new code uses the existing `NonRetriableError`/`RetriableError` hierarchy from `src/node0/utils/node_info.py`. Specific mappings:

- **Corpus scrubbing failures** (PII detection tool unavailable): `RetriableError` — retry loading the tool; if unavailable after retries, `NonRetriableError` (cannot proceed with unscrubbed data)
- **Tokenizer vocab_size mismatch**: `NonRetriableError` — shape mismatch is not recoverable without config fix
- **NaN gradients in fine-tune mode**: `RetriableError` — skip step and continue (replaces `os.killpg`)
- **DHT timeout during checkpoint gather**: `RetriableError` wrapped in `call_with_retries()` (10 retries, exponential backoff with `initial_delay=1.0`)
- **Dolt connection failure**: `RetriableError` via `call_with_retries()` — Dolt server may restart
- **GGUF export subprocess failure**: `NonRetriableError` — llama.cpp binary missing or model corrupt
- **Registry model not found**: `NonRetriableError` — inference cannot proceed without model

Background thread errors (DataFeeder, MonitorWorker) follow the existing pattern: `except Exception as e: logger.error(f"Error in thread: {e}")` — log and continue rather than crash, unless the error is fatal (e.g., corpus path doesn't exist).

All fatal paths that previously called `os.killpg` in fine-tuning mode must first log the event (so MonitorWorker can update bead state) before raising. The bead lifecycle update is via logging: MonitorWorker regex on `"Training job failed: step {step}"` triggers DHT update.

### Testing Strategy

**Phase 1:** Unit tests for config parsing, vocab_size fix, schema types, auth bypass.
**Phase 2:** Unit tests for scrubber (with known PII patterns), formatter, store (using `tmp_path`), dataset label masking, loss differentiability.
**Phase 3:** Unit tests for LoRA application (grad filtering), data feeder thread, NaN handler mode switching.
**Phase 4:** Unit tests for shard save/load round-trip, DHT-mocked gather timeout, registry SQL generation.
**Phase 5:** Unit tests for KV cache shapes, sampling distribution properties, prompt API stdout behavior, RAG indexer/retriever.
**Phase 6:** Unit tests for each eval task category with mock inference_fn; integration test for full eval runner.
**Phase 7:** Unit tests for DPO routing (mocked subprocess), corpus export structure, staleness detection.

All unit tests mock: DHT calls, subprocess calls (tmux, llama.cpp, dolt), Dolt MySQL connections, HuggingFace downloads. Integration tests (`@pytest.mark.integration`) require real Hivemind DHT and are not run in default `pytest` invocation.

Highest-value integration test (defer to after Phase 3): launch a 2-stage mini-model locally with `--local-mode`, feed 10 batches, assert loss decreases.

### Migration

No data migration needed for Phase 1-3 (all new tables and files). Phase 4 introduces new Dolt tables (`model_registry`, `dpo_preferences`) via the migration SQL in `src/node0/finetune/registry/migrations/001_model_registry.sql`. The migration must be run against the HQ Dolt database before Phase 4 registry operations. Run: `dolt sql -f src/node0/finetune/registry/migrations/001_model_registry.sql`. No backward compatibility concern — these are purely additive tables. Existing `issues.jsonl` and bead tables are unchanged.

---

## Technical Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| DPO loss with pipeline-parallel requires two full forward passes per batch (chosen + rejected), doubling throughput cost | H | M | Phase 3 implements SFT first; DPO is Phase 7. If throughput is unacceptable, fall back to SFT-only with DPO pair collection deferred to V2 |
| Checkpoint gather coordination requires all 32 stages to complete simultaneously; straggler peer causes timeout | M | H | `gather_all_shards()` has 300s timeout; on timeout, partial checkpoint is discarded and training continues. Periodically retry. Add DHT-announced "gather coordinator" to track which stages are present |
| ChromaDB embedding quality (default sentence-transformers) may be poor for Gas Town-specific queries (gt commands, bead IDs) | M | M | V1 accepts default embeddings; if retrieval quality is poor, switch to a fine-tuned embedding model in V2. Eval: test RAG retrieval with known queries in `tests/unit/test_rag.py` |
| Initial corpus volume (250-2500 per role) insufficient for meaningful fine-tuning | H | M | Spec documents this risk explicitly. Accept degraded first-run performance. Corpus grows with every session. Eval suite will quantify the gap objectively |
| vocab_size mismatch was silently wrong for a long time; other config values may be wrong too | L | H | Task 1.1 adds tokenizer validation that raises `NonRetriableError` on mismatch. Review all LlamaArguments defaults against reference LLaMA 2 / LLaMA 3 config before first fine-tuning run |
| `os.killpg` calls in `monitor.py` and `authorization.py` (in addition to `optim.py`) not addressed | M | M | Task 3.4 only fixes `optim.py`. Create follow-up bead to audit all `os.killpg` sites and add graceful bead-state update before exit |
| PluralisAuthorizer bypass (`--local-mode`) may leave security gaps if accidentally used in production | L | H | `--local-mode` logs a loud `logger.warning("Running in local mode: no external auth. NOT for production.")` at startup. Document in YAML config comments |
| Rig structure (Q52) unresolved — unclear where training runs vs where corpus is collected | M | M | Defer actual deployment topology to Phase 3 implementation. The code is topology-agnostic (DHT handles peer routing). Phase 6 rollout can be done on a single rig for V1 |
| llama.cpp binary not available on checkpoint export node | M | M | GGUF export is a post-processing step; failure does not block registry registration (safetensors is the primary format). Log warning and continue |

---

## Spec Coverage Matrix

| Spec Section | Plan Section | Phase |
|-------------|-------------|-------|
| Architecture: Corpus Pipeline | Phase 2: Tasks 2.1-2.4 (corpus schema, scrubber, formatter, collector) | 2 |
| Architecture: Training Infrastructure: What exists vs what we build | Phase 3: Tasks 3.1-3.5 (LoRA, data feeding, NaN handler, configs); Phase 1 (auth bypass) | 1, 3 |
| Architecture: Training Infrastructure: Data loading pipeline (EPIC) | Phase 2: Tasks 2.5-2.7 (tokenizer, dataset, dataloader, loss); Phase 3: Task 3.3 (integration) | 2, 3 |
| Architecture: Training Infrastructure: Checkpoint export | Phase 4: Tasks 4.1-4.3 (shard, gatherer, GGUF export) | 4 |
| Architecture: Training Infrastructure: LoRA adapter support | Phase 1: Task 1.1 (ModelArguments fields); Phase 3: Tasks 3.1-3.2 (adapter, param filtering) | 1, 3 |
| Architecture: Training Infrastructure: Autoregressive inference | Phase 5: Tasks 5.1-5.4 (KV cache, sampling, generator, prompt API) | 5 |
| Architecture: Training Infrastructure: DPO via Dolt branching | Phase 7: Tasks 7.1-7.2 (router, exporter) | 7 |
| Architecture: Training Infrastructure: Training job lifecycle (bead-mapped) | Phase 3: Task 3.4 (NaN handler, Dolt direct state writes, bead_id at launch); Phase 6: Task 6.1 (session hook); Phase 2: Task 2.8 (MonitorWorker) | 2, 3, 6 |
| Architecture: Training Infrastructure: Training job lifecycle (7 states) | Phase 3: Task 3.4 (state transitions via Dolt MySQL direct write, port 3307) | 3 |
| Architecture: Training Infrastructure: Multi-model experiment tracking | Phase 4: Task 4.4 (ModelRegistry with corpus_hash, training_config) | 4 |
| Architecture: Training Infrastructure: Corpus data validation and quality gates | Phase 2: Task 2.2 (PII scrubber multi-layer); Phase 2: Task 2.4d (periodic audit); Phase 2: Task 2.1 (CorpusStore.corpus_hash) | 2 |
| Architecture: Training Configuration via Formulas | Phase 6: Task 6.5 (training formulas TOML at ~/gt/.beads/formulas/) | 6 |
| Architecture: Training Infrastructure: vocab_size note | Phase 1: Task 1.1 (vocab_size fix, BLOCKER) | 1 |
| Architecture: Training Infrastructure: PluralisAuthorizer bypass | Phase 1: Task 1.3 (--local-mode flag); Phase 1: Task 1.8 (run.json local_mode field) | 1 |
| Architecture: Training Infrastructure: AutoStepOptimizer NaN handling | Phase 3: Task 3.4 (graceful recovery, Dolt state update) | 3 |
| Architecture: Training Infrastructure: MonitorWorker training metrics | Phase 2: Task 2.8 (training metrics extension, TrainingMetricSchema wrapper) | 2 |
| Model Registry & Deployment: Model Registry (storage design, CLI) | Phase 4: Tasks 4.4-4.5 (store, CLI including gt model prune) | 4 |
| Model Registry & Deployment: Model Registry UI (CLI-only in V1, UI deferred to V2) | Phase 4: Task 4.5 (CLI commands only); browsable UI deferred per plan review decision | 4 |
| Model Registry & Deployment: Versioned Rollback | Phase 4: Task 4.5 (`gt model rollback`); retention via `gt model prune --keep-last N` | 4 |
| Model Registry & Deployment: Provider Integration (node0 provider) | Phase 5: Task 5.6 (town settings); Phase 5: Task 5.4 (run_inference.py) | 5 |
| Model Registry & Deployment: Progressive Rollout — Stage 1 (deacon) | Phase 6: Task 6.6 (enforced eval gate) | 6 |
| Model Registry & Deployment: Progressive Rollout — Stage 2 (witness) | Phase 6: Task 6.6 (enforced eval gate) | 6 |
| Model Registry & Deployment: Progressive Rollout — Stage 3 (polecat) | Phase 6: Task 6.7 (polecat rollout with circuit breaker and patrol agent modification) | 6 |
| Data Pipeline & Corpus: Data Sources (session transcripts) | Phase 2: Tasks 2.3-2.4 (formatter, collector); Phase 6: Task 6.1 (session hook) | 2, 6 |
| Data Pipeline & Corpus: Data Sources (codebase artifacts: CLAUDE.md, formulas, specs) | Phase 2: Task 2.4b (CorpusArtifactCollector) | 2 |
| Data Pipeline & Corpus: Data Sources (operational data: bead lifecycles, sling outcomes) | Phase 2: Task 2.4c (OperationalDataCollector) | 2 |
| Data Pipeline & Corpus: Corpus Pipeline (COLLECT → SCRUB → FORMAT → PARTITION) | Phase 2: Tasks 2.1-2.4 | 2 |
| Data Pipeline & Corpus: Data Characteristics (volume, PII scrubbing, periodic audit) | Phase 2: Task 2.2 (multi-layer scrubber); Phase 2: Task 2.4d (audit subcommand) | 2 |
| Data Pipeline & Corpus: Extensibility Path (cross-rig sharing opt-in) | Phase 5: Task 5.6 (rig settings field); Phase 8: Tasks 8.1-8.2 (cross-operator sharing mechanism) | 5, 8 |
| Data Pipeline & Corpus: Corpus reversion detection (recency wins caveat) | Phase 6: Task 6.8 (CorpusReversionDetector, score regression check) | 6 |
| Evaluation & Quality: Automated Eval Task Suite | Phase 6: Tasks 6.2-6.4 (five categories, runner, scorer) | 6 |
| Evaluation & Quality: Eval scoring methodology (binary/numeric, category %, composite) | Phase 6: Task 6.4 (EvalRunner.composite_score) | 6 |
| Evaluation & Quality: Eval deployment gate (automated enforcement) | Phase 6: Task 6.6 (gt model deploy enforces threshold, not just logs warning) | 6 |
| Evaluation & Quality: DPO via Dolt Branching | Phase 7: Tasks 7.1-7.2 | 7 |
| Evaluation & Quality: Model Staleness Detection (CLAUDE.md + formulas + config) | Phase 7: Task 7.3 (ConventionStalenessTracker, three drift signals) | 7 |
| Inference Architecture: Generation loop, Prompt API, Sampling, Streaming, KV-cache | Phase 5: Tasks 5.1-5.4 | 5 |
| RAG Architecture: ChromaDB, indexing, retrieval at inference | Phase 5: Task 5.5 (beads, configs, sessions, formulas collections) | 5 |
| RAG Architecture: Formula definitions indexed in ChromaDB | Phase 5: Task 5.5 (formulas collection from ~/gt/.beads/formulas/) | 5 |
| Gas Town Integration: gt prime modification (model-aware context reduction) | Phase 5: Task 5.5b (detect fine-tuned model, cap prime to ~2-3K tokens) | 5 |
| Corpus Collection Mechanism: Bead-linked export (primary) | Phase 6: Task 6.1 (session_shutdown hook); Phase 2: Task 2.4 (collector) | 2, 6 |
| Corpus Collection Mechanism: Wisp/gap coverage (batch export, manual add) | Phase 2: Task 2.4 (`node0-corpus add` subcommand) | 2 |
| Corpus Collection Mechanism: Cross-rig sharing opt-in | Phase 5: Task 5.6 (rig config flag); Phase 8: Task 8.1 (sharing exporter with anonymization) | 5, 8 |
| Corpus Collection Mechanism: Recency wins conflict resolution | Phase 2: Task 2.1 (CorpusStore date-sorted list; newest entry for same bead_id wins) | 2 |
| Gas Town Integration: Component Integration Map (all rows) | Phases 2, 5, 6 (corpus hook, formulas, bead system, sling, monitor, registry, gt prime) | 2, 5, 6 |
| Architecture: Key Architectural Decisions: Training approach (distributed via node0) | Phase 3: Tasks 3.1-3.3 (LoRA in pipeline, data feeding) | 3 |
| Architecture: Key Architectural Decisions: Config management (formula templates) | Phase 6: Task 6.5 | 6 |
| Architecture: Key Architectural Decisions: Conflict resolution (recency wins) | Phase 2: Task 2.1 (CorpusStore.list sort order) | 2 |

---

## Appendix: Key File Paths

### New Files

| Path | Phase | Purpose |
|------|-------|---------|
| `src/node0/finetune/__init__.py` | 1 | Package root |
| `src/node0/finetune/config.py` | 1 | `FineTuneConfig` Pydantic model |
| `src/node0/finetune/lora/__init__.py` | 1 | LoRA sub-package |
| `src/node0/finetune/lora/config.py` | 1 | `LoRAConfig` Pydantic model |
| `src/node0/finetune/lora/adapter.py` | 3 | `apply_lora()`, `get_lora_parameters()`, `freeze_base_params()` |
| `src/node0/finetune/lora/linear.py` | 3 | `LoRALinear` nn.Module |
| `src/node0/finetune/corpus/__init__.py` | 2 | Corpus sub-package |
| `src/node0/finetune/corpus/schema.py` | 2 | `CorpusEntry` Pydantic model |
| `src/node0/finetune/corpus/store.py` | 2 | `CorpusStore` filesystem backend |
| `src/node0/finetune/corpus/scrubber.py` | 2 | `CorpusScrubber` multi-layer PII removal |
| `src/node0/finetune/corpus/formatter.py` | 2 | `CorpusFormatter` transcript → CorpusEntry |
| `src/node0/finetune/corpus/collector.py` | 2 | `CorpusCollector` session hook integration |
| `src/node0/finetune/corpus/staleness.py` | 7 | `ConventionStalenessTracker` |
| `src/node0/finetune/data/__init__.py` | 2 | Data sub-package |
| `src/node0/finetune/data/tokenizer.py` | 2 | `load_tokenizer()` with vocab validation |
| `src/node0/finetune/data/dataset.py` | 2 | `CorpusDataset` PyTorch Dataset |
| `src/node0/finetune/data/loader.py` | 2 | `make_dataloader()` with collate_fn |
| `src/node0/finetune/data/loss.py` | 2 | `sft_loss()`, `dpo_loss()` |
| `src/node0/finetune/data/feeder.py` | 3 | `DataFeeder` background thread |
| `src/node0/finetune/checkpoint/__init__.py` | 4 | Checkpoint sub-package |
| `src/node0/finetune/checkpoint/shard.py` | 4 | `save_shard()`, `load_shard()` |
| `src/node0/finetune/checkpoint/gatherer.py` | 4 | `CheckpointGatherer` DHT coordination |
| `src/node0/finetune/checkpoint/assembler.py` | 4 | `assemble_checkpoint()` |
| `src/node0/finetune/checkpoint/exporter.py` | 4 | `export_to_gguf()` llama.cpp wrapper |
| `src/node0/finetune/registry/__init__.py` | 4 | Registry sub-package |
| `src/node0/finetune/registry/schema.py` | 4 | `ModelRegistryEntry` Pydantic model |
| `src/node0/finetune/registry/store.py` | 4 | `ModelRegistry` Dolt backend |
| `src/node0/finetune/registry/cli.py` | 4 | `gt model` CLI commands |
| `src/node0/finetune/registry/migrations/001_model_registry.sql` | 4 | Dolt table DDL |
| `src/node0/finetune/inference/__init__.py` | 5 | Inference sub-package |
| `src/node0/finetune/inference/kv_cache.py` | 5 | `KVCache` per-stage cache |
| `src/node0/finetune/inference/sampling.py` | 5 | `sample_token()` with temperature/top-p/top-k |
| `src/node0/finetune/inference/generator.py` | 5 | `Generator` autoregressive loop |
| `src/node0/finetune/inference/prompt_api.py` | 5 | `PromptAPI` stdin/stdout interface |
| `src/node0/finetune/rag/__init__.py` | 5 | RAG sub-package |
| `src/node0/finetune/rag/indexer.py` | 5 | `RAGIndexer` ChromaDB indexing |
| `src/node0/finetune/rag/retriever.py` | 5 | `RAGRetriever` context retrieval |
| `src/node0/finetune/eval/__init__.py` | 6 | Eval sub-package |
| `src/node0/finetune/eval/schema.py` | 1 | `EvalResult` Pydantic model |
| `src/node0/finetune/eval/runner.py` | 6 | `EvalRunner` orchestrator |
| `src/node0/finetune/eval/scorer.py` | 6 | `update_registry_with_scores()` |
| `src/node0/finetune/eval/tasks/__init__.py` | 6 | Tasks sub-package |
| `src/node0/finetune/eval/tasks/bead_management.py` | 6 | Bead management eval tasks |
| `src/node0/finetune/eval/tasks/git_workflow.py` | 6 | Git workflow eval tasks |
| `src/node0/finetune/eval/tasks/convention_adherence.py` | 6 | Convention adherence eval tasks |
| `src/node0/finetune/eval/tasks/planning.py` | 6 | Planning eval tasks |
| `src/node0/finetune/eval/tasks/code_execution.py` | 6 | Code execution eval tasks |
| `src/node0/finetune/dpo/__init__.py` | 7 | DPO sub-package |
| `src/node0/finetune/dpo/router.py` | 7 | `DPORouter` Dolt branching + preference recording |
| `src/node0/finetune/dpo/exporter.py` | 7 | `DPOExporter` preference pairs → corpus entries |
| `src/node0/finetune/corpus/artifact_collector.py` | 2 | `CorpusArtifactCollector` CLAUDE.md, formulas, specs → corpus entries |
| `src/node0/finetune/corpus/operational_collector.py` | 2 | `OperationalDataCollector` bead lifecycle, sling outcomes → corpus entries |
| `src/node0/finetune/corpus/sharing.py` | 8 | `CorpusSharingExporter` anonymized cross-operator corpus export |
| `src/node0/finetune/eval/reversion.py` | 6 | `CorpusReversionDetector` score regression detection |
| `src/node0/finetune/registry/access.py` | 8 | `AccessGrantManager` corpus contributor access grants |
| `src/node0/run_inference.py` | 5 | `node0-inference` entry point |
| `src/node0/run_finetune.py` | 1 | `node0-finetune` entry point (stub in Phase 1, wired in Phase 3) |
| `src/node0/run_corpus.py` | 2 | `node0-corpus` entry point |
| `src/node0/configs/finetune/lora_sft_deacon.yaml` | 3 | Deacon SFT training config |
| `src/node0/configs/finetune/lora_sft_witness.yaml` | 3 | Witness SFT training config |
| `src/node0/configs/finetune/lora_sft_polecat.yaml` | 3 | Polecat SFT training config |
| `src/node0/configs/finetune/dpo_polecat.yaml` | 3 | Polecat DPO training config |
| `tests/__init__.py` | 1 | Test package root |
| `tests/unit/__init__.py` | 1 | Unit tests |
| `tests/integration/__init__.py` | 1 | Integration tests |
| `tests/conftest.py` | 1 | Shared pytest fixtures |
| `~/gt/.beads/formulas/train-role-model.formula.toml` | 6 | Training pipeline formula (`/home/ubuntu/gt/.beads/formulas/`) |
| `~/gt/.beads/formulas/corpus-collect.formula.toml` | 6 | Corpus collection formula |
| `~/gt/.beads/formulas/eval-suite.formula.toml` | 6 | Eval suite formula |

### Modified Files

| Path | Phase | Changes |
|------|-------|---------|
| `src/node0/models/arguments.py` | 1 | Add LoRA fields: `lora_rank`, `lora_alpha`, `lora_dropout`, `lora_target_modules` |
| `src/node0/models/llama/arguments.py` | 1 | Fix `vocab_size` 50265 → 32000; increase `max_seq_len` 512 → 2048 |
| `src/node0/models/llama/layers.py` | 3 | Apply LoRA in Expert `__init__` when `model_args.lora_rank is not None` |
| `src/node0/security/validation.py` | 1 | Add `TrainingMetricsV1` and `TrainingMetricSchema` wrapper; update `make_validators()` with SchemaValidator pattern |
| `src/node0/security/authorization.py` | 1 | No code change in Phase 1; auth bypass is in `run_server.py` |
| `src/node0/server/node0_server.py` | 3 | Add `finetune_config` param; filter LoRA params for optimizer/averager; parameterize `checkpoint_dir` |
| `src/node0/server/optim.py` | 3 | Add `finetune_mode` param; replace `os.killpg` NaN handler with `RetriableError` when in finetune mode; Dolt direct write for state transitions |
| `src/node0/server/module_collab.py` | 3 | Add `data_feeder` param; integrate DataFeeder in `on_backward()` for tail stage; trigger checkpoint saves |
| `src/node0/utils/monitor.py` | 2 | Add `finetune_mode` param; add training metric regex patterns; extend `report()` for `TrainingMetricsV1` |
| `src/node0/run_server.py` | 1 | Add `--local-mode`, `--local-stage`, `--finetune-config` args; conditional auth bypass |
| `src/node0/finetune/corpus/scrubber.py` | 2 | Add `audit_entry()` and `audit_store()` for periodic PII re-scan |
| `src/node0/finetune/registry/cli.py` | 4, 6 | Phase 4: all model CLI commands including `gt model prune`; Phase 6: enforce eval gate in `deploy_model()` |
| `src/node0/finetune/inference/prompt_api.py` | 5 | RAG context injection before generation |
| `pyproject.toml` | 1 | Add `finetune` optional deps; add `node0-inference`, `node0-finetune`, `node0-corpus` scripts; add pytest config |
| `Dockerfile` | 1 | Add CUDA training deps (torch, peft, transformers, chromadb, detect-secrets); add VOLUME declarations |
| `run.json` | 1 | Add `local_mode`, `local_seed_peers`, `bypass_auth` fields for self-hosted fine-tuning |
| `/home/ubuntu/gt/settings/config.json` | 5, 6, 8 | Phase 5: add node0 agent entries; Phase 6: switch deacon/witness/polecat role assignments; Phase 8: corpus_sharing and model_access fields |
| `~/gt/gastown/<prime-command-file>` | 5 | Add model-awareness check to reduce prime context for fine-tuned model roles |

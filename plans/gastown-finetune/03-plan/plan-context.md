# Codebase Analysis: gastown-finetune

**Generated:** 2026-02-22
**Source:** 3-agent parallel exploration
**Codebase root:** `/home/ubuntu/gt/node0/crew/neo/`

---

## Architecture Overview

### Project Structure

The node0 package is organized as a three-layer architecture with strict dependency boundaries:

```
/home/ubuntu/gt/node0/crew/neo/
├── src/node0/              # Main Python package (sole source tree)
│   ├── __init__.py
│   ├── run_server.py       # Single entry point — main() wires everything together
│   ├── configs/            # YAML run configs; currently one file: llama_8B_C.yaml
│   ├── models/             # Model definitions and argument schemas
│   │   ├── arguments.py    # Base Pydantic ModelArguments class
│   │   ├── lr_schedule.py  # LR scheduler registry
│   │   └── llama/
│   │       ├── arguments.py    # LlamaArguments (extends ModelArguments)
│   │       └── layers.py       # All LLaMA layer implementations
│   ├── security/           # Auth and integrity
│   │   ├── authorization.py        # PluralisAuthorizer — token + stage assignment
│   │   ├── validation.py           # WorkerMetricsV1, RunParameters Pydantic models
│   │   └── integrity_check.*.so    # Compiled binary for code integrity verification
│   ├── server/             # Distributed training core
│   │   ├── node0_server.py         # Node0Server.create() — wiring factory
│   │   ├── HM_averager.py          # DecentralizedAverager (butterfly allreduce)
│   │   ├── HM_gradient_averager.py # GradientAverager (accumulation + allreduce)
│   │   ├── HM_state_averager.py    # State averaging for model weights
│   │   ├── optim.py                # AutoStepOptimizer (timed step triggering)
│   │   ├── matchmaking.py          # Peer group formation for averaging rounds
│   │   ├── module_collab.py        # ModuleCollab (extends Hivemind ModuleBackend)
│   │   ├── power_sgd_averager.py   # PowerSGD gradient compression (Linux)
│   │   ├── power_sgd_averager_mac.py # PowerSGD gradient compression (macOS)
│   │   ├── state_averager_wrap.py  # TrainingStateAverager with index selection
│   │   └── ar_runner.py            # AllReduceRunner (butterfly allreduce protocol)
│   └── utils/              # Utilities
│       ├── __init__.py             # Re-exports core utilities
│       ├── common.py               # build_cls(), infer_expert_params(), load_ss_components()
│       ├── monitor.py              # BaseMonitor, MonitorWorker (log-scraping + DHT reporting)
│       ├── logging.py              # Node0Logger
│       ├── node_info.py            # Hardware/network profiling: GPU, RAM, speedtest
│       ├── dht_monitor.py          # DHT protocol logging patch
│       ├── dht_partition.py        # update_initial_peers() — DHT peer list management
│       ├── get_parameters.py       # get_parameter_store() — reads run params from DHT
│       ├── flops.py                # FLOPs estimation (fwd/bwd per token)
│       ├── mem_monitor.py          # MemoryTracker
│       ├── network_throughput.py   # NetworkMonitor
│       └── connection_test_server.py # TestServer for port reachability verification
├── Dockerfile              # NVIDIA CUDA 12.1.1 + conda + pip install .
├── pyproject.toml          # hatchling build, Python 3.11 strict, all deps declared
├── run.json                # Seed peer multiaddrs + run_config reference
└── generate_script.py      # Docker/source setup script generator
```

### Package Organization Principles

**Layer 1: Entry/Wiring** (`run_server.py`)
- Parses config, instantiates all components, calls `Node0Server.create()`. No business logic here — pure wiring.

**Layer 2: Domain Components** (`models/`, `security/`, `server/`)
- Independent subsystems. `server/` depends on `models/` and `security/` but not vice versa. `utils/` is depended on by all.
- Uses a plugin pattern: `ModelArguments` is the abstract base; `LlamaArguments` extends it. New architectures would add new subdirectories alongside `llama/`.
- YAML config specifies the full dotted class path (`node0.models.llama.arguments.LlamaArguments`), and `build_cls()` in `utils/common.py` instantiates it dynamically via `importlib`.

**Layer 3: Utilities** (`utils/`)
- No dependencies on domain components (except `security.validation` for `WorkerMetricsV1`).

### Module-Level Dependencies

```
run_server.py
  ├── node0.security.authorization   (PluralisAuthorizer, authorize_with_pluralis)
  ├── node0.security.validation      (make_validators)
  ├── node0.server.HM_gradient_averager  (GradientAverager)
  ├── node0.server.node0_server      (Node0Server)
  ├── node0.server.optim             (AutoStepOptimizer)
  └── node0.utils                    (MonitorWorker, Node0Logger, build_cls, ...)

node0_server.py
  ├── node0.models.arguments         (ModelArguments)
  ├── node0.models.lr_schedule       (schedule_name_to_scheduler)
  ├── node0.server.HM_gradient_averager
  ├── node0.server.module_collab     (ModuleCollab)
  ├── node0.utils.common             (load_ss_components)
  ├── node0.utils.dht_monitor
  ├── node0.utils.get_parameters     (get_parameter_store)
  └── node0.utils                    (MonitorWorker)
  -- (all via hivemind) -->
      hivemind.moe.server            (Server, ModuleBackend, name_to_block, name_to_input)
      hivemind.dht                   (DHT)
      hivemind.optim                 (Optimizer)

optim.py (AutoStepOptimizer)
  ├── node0.server.HM_gradient_averager  (GradientAverager)
  └── node0.server.state_averager_wrap   (TrainingStateAverager)
  -- extends --> hivemind.Optimizer

HM_gradient_averager.py (GradientAverager)
  └── node0.server.HM_averager  (DecentralizedAverager)

module_collab.py (ModuleCollab)
  -- extends --> hivemind.moe.server.ModuleBackend

security/authorization.py (PluralisAuthorizer)
  ├── node0.security.integrity_check  (compiled binary)
  ├── node0.utils.connection_test_server
  └── node0.utils.node_info
  -- extends --> hivemind.utils.auth.TokenAuthorizerBase

utils/monitor.py (MonitorWorker)
  ├── node0.security.validation  (WorkerMetricsV1)
  └── node0.utils.flops
```

### Core Dependencies

1. **hivemind** (pinned commit `4d5c414`) — Foundational library providing: DHT, P2P, gRPC pipeline, ModuleBackend (expert abstraction), DecentralizedAverager, Optimizer base, access token types, logging utilities.

2. **PyTorch 2.7.0** — Exactly this version (enforced at startup in `run_server.py`).

3. **Pydantic >= 2.0** — Used for all config/argument schemas. The YAML-to-Python config bridge runs through Pydantic.

---

## Integration Surface

### Files That Require Modification

#### `src/node0/models/arguments.py` — ModelArguments (Pydantic BaseModel)

**Current state:**
```python
class ModelArguments(BaseModel):
    hidden_dim: int
    n_heads: int
    num_hidden_layers: int
    n_layers: int = 0
    stage: int | None = None
    attn_proj: bool = False
    qk_norm: bool = False
    norm_reorder: bool = False
    trainable_rmsnorm: bool = True
    compression_rate: int | None = None
    use_compression: bool = False
    ss_component: str | None
```

**Changes needed:**
- Add LoRA fields: `lora_rank: int | None = None`, `lora_alpha: float | None = None`, `lora_dropout: float = 0.05`, `lora_target_modules: list[str] | None = None`
- These fields must be passed through to every expert constructor (`name_to_block[expert_cls](model_conf)`)

#### `src/node0/models/llama/arguments.py` — LlamaArguments

**Current state:**
```python
class LlamaArguments(ModelArguments):
    hidden_dim: int = 4096
    n_heads: int = 32
    n_kv_heads: int | None = None
    vocab_size: int = 50265        # CRITICAL ISSUE: OPT-2.7b tokenizer, not LLaMA
    multiple_of: int = 256
    ffn_dim_multiplier: float | None = None
    norm_eps: float = 1e-5
    rope_theta: float = 10000
    max_seq_len: int = 512         # May need increase for fine-tuning
    depth_init: bool = True
    constant_init: bool = False
    norm_type: str = "rmsnorm"
```

**Changes needed:**
- **BLOCKER:** `vocab_size` must be fixed. Currently `50265` (OPT tokenizer); must be changed to `32000` (LLaMA 2) or `128256` (LLaMA 3) depending on chosen base model. This is a correctness issue — tokenizer misalignment will cause shape mismatches during data loading.
- `max_seq_len` should be increased to accommodate Gas Town session lengths (likely > 512 tokens).

#### `src/node0/models/llama/layers.py` — Expert Classes

**Current structure:**
- `HeadExpert` — embedding layers + transformer blocks
- `BodyExpert` — middle transformer blocks only
- `TailExpert` — tail blocks + norm + output projection, computes cross-entropy loss

**Changes needed:**
- All three Expert classes need LoRA wrapping applied per-stage after model init
- LoRA wrapping must replace targeted linears (`wq`, `wk`, `wv`, `wo`, `w1`, `w2`, `w3`) with LoRA-augmented versions
- Cannot be applied globally — must be applied inside each expert's `__init__` or in a post-init hook
- `TailExpert.forward(hidden_states, labels)` already computes cross-entropy loss; loss computation method stays the same for fine-tuning

#### `src/node0/server/node0_server.py` — Node0Server.create()

**Current issues:**

1. Line 316: `checkpoint_dir=None` is hardcoded
   - Must become a configurable parameter
   - Checkpoint saving logic must be added
   - Requires gathering weights from all 32 pipeline stages

2. Lines 241-260: Optimizer parameter filtering currently selects params by name
   - For LoRA fine-tuning, must be replaced to select ONLY LoRA adapter parameters
   - Base model weights must remain frozen (`requires_grad=False`)

3. `PluralisAuthorizer` is required and hardcoded
   - Must be made optional for self-hosted fine-tuning
   - `Node0Server.create()` accepts `authorizer` as kwarg (passed to `Server`)
   - Issue is in `run_server.py` calling `authorize_with_pluralis()` unconditionally

**Method signatures affected:**
```python
@classmethod
def create(cls, ..., *, start: bool, **kwargs) -> Server
```

#### `src/node0/server/optim.py` — AutoStepOptimizer

**Current issues:**

1. Line 216: `_check_and_accumulate_gradients()` calls `os.killpg(os.getpgrp(), signal.SIGTERM)` on NaN grads
   - Too aggressive for fine-tuning where gradient instability is more common
   - Should trigger retry/checkpoint-resume, not hard kill

2. Auto-step timer logic assumes external training loop is pushing data
   - Fine-tuning needs data-driven step scheduling

3. `_maybe_schedule_gradient_averaging()` is hardcoded for global pretraining epoch
   - Fine-tuning with LoRA-only gradients must ensure only adapter params are averaged

**Method signatures affected:**
```python
def _check_and_accumulate_gradients(self, batch_size: int) -> bool
def _auto_step(self) -> None
def _make_gradient_averager(self, factory, **kwargs) -> GradientAverager
```

#### `src/node0/server/HM_gradient_averager.py` — GradientAverager

**Current method signature:**
```python
def __init__(self,
    parameters: Iterable[torch.nn.Parameter],
    *, dht: DHT, prefix: str,
    reuse_grad_buffers: bool = False,
    accumulate_grads_on: Optional[torch.device] = None,
    client_mode: bool = None, warn: bool = True,
    averaged_grads: Sequence[torch.Tensor] = (),
    **kwargs)
```

**Changes needed:**
- No structural change needed, but caller must pass filtered LoRA-only params for fine-tuning
- `has_nan_grads()` check iterates all passed parameters (correct if filtered properly)

#### `src/node0/utils/monitor.py` — MonitorWorker

**Current tracking:**
- Infrastructure metrics: FLOPs, allreduce success/failure, port reachability
- Reports via `WorkerMetricsV1` schema

**Changes needed:**
- Add training-specific metrics: loss curve, per-epoch eval score, corpus statistics, LoRA adapter convergence
- New regex patterns for fine-tuning log lines (loss values, eval scores, checkpoint events)
- Extend `report()` method to store new metrics (new DHT schema `TrainingMetricsV1` needed)

**Current patterns (in `__init__`):**
- Will need new patterns like: `self.loss_pattern = r".*Training loss: ([0-9.]+).*"`

#### `src/node0/security/validation.py` — RunParameters and Metrics Schemas

**Current schema (Pydantic v1 via hivemind compatibility):**
```python
class WorkerMetricsV1(BaseModel):
    peer_id: str
    num_flop: StrictFloat
    active_time: StrictFloat
```

**Changes needed:**
- Add new `TrainingMetricsV1` schema:
  ```python
  class TrainingMetricsV1(BaseModel):
      peer_id: str
      step: StrictInt
      loss: StrictFloat
      role: StrictStr
      corpus_hash: StrictStr
  ```
- Update `make_validators()` to include validator for new schema
- Add fine-tuning run parameters to DHT parameter store (corpus hash, LoRA rank, role target)

#### `src/node0/security/authorization.py` — PluralisAuthorizer

**Current behavior:**
- `join_experiment()` calls `https://auth.pluralis.ai/api/join` — external dependency
- Assigns `pipeline_stage: str` (e.g., `"head-0"`, `"body-3"`, `"tail-0"`)

**Changes needed:**
- Make auth server optional for self-hosted fine-tuning
- Add local stage assignment fallback when auth server is unavailable

#### `src/node0/run_server.py` — Entry Point

**Current flow:**
1. Calls `authorize_with_pluralis()` unconditionally
2. Parses YAML config via `build_cls()` pattern
3. Instantiates all components, calls `Node0Server.create()`

**Changes needed:**
1. Add `--local-mode` flag (or equivalent) that skips external auth and uses local key + stage assignment config
2. Add fine-tuning-specific CLI args: `--finetune-mode`, `--corpus-path`, `--lora-rank`, `--checkpoint-dir`, `--eval-every-n-steps`
3. Extend YAML schema to support fine-tuning configs (corpus_path, lora config, checkpoint_dir)
4. `args.pop()` pattern for consumed arguments before forwarding to server

#### `src/node0/configs/llama_8B_C.yaml` — YAML Config

**Current structure:**
```yaml
model_config:
  class_name: node0.models.llama.arguments.LlamaArguments
  init_args:
    hidden_dim: 4096
    ...

optim_config:
  class_name: torch.optim.AdamW
  init_args:
    lr: 0.0003

grad_avg_config:
  class_name: node0.server.power_sgd_averager.PowerSGDGradientAverager
  init_args:
    averager_rank: 64

scheduler: linear
num_warmup_steps: 4000
averaging_target_batch_size: 1024
```

**Changes needed:**
- New fine-tuning YAML configs (one per role: mayor, polecat, witness, deacon) with:
  - `corpus_path`, `lora_rank`, `finetune_mode: sft | dpo`, `checkpoint_dir`, `vocab_size: 32000`
  - Corrected `vocab_size` for LLaMA (NOT the existing `50265`)

#### `/home/ubuntu/gt/settings/config.json` — Gas Town Town Settings

**Required additions:**
```json
"agents": {
    "node0-deacon": {
        "command": "node0-inference",
        "args": ["--model", "deacon-v1", "--adapters", "bead-create"],
        "provider": "node0",
        "tmux": { "process_names": ["node0-inference", "python"] },
        "prompt_mode": "stdin"
    },
    "node0-witness": { ... },
    "node0-polecat": { ... }
},
"deacon": "node0-deacon",
"witness": "node0-witness"
```

### Interfaces & Types

#### Existing Types (Pydantic models)

**ModelArguments** (`src/node0/models/arguments.py`)
- Consumed by: `Node0Server.create()`, all Expert `__init__` methods, `MonitorWorker.monitor_callback()`, `build_cls()` in run_server.py
- Central config object — LoRA fields must be added here

**LlamaArguments** (extends `ModelArguments`)
- Specialization for LLaMA architecture

**GradientAverager** (`src/node0/server/HM_gradient_averager.py`)
```python
class GradientAverager(DecentralizedAverager):
    def __init__(self,
        parameters: Iterable[torch.nn.Parameter],
        *, dht: DHT, prefix: str,
        reuse_grad_buffers: bool = False,
        accumulate_grads_on: Optional[torch.device] = None,
        client_mode: bool = None, warn: bool = True,
        averaged_grads: Sequence[torch.Tensor] = (),
        **kwargs)
    def accumulate_grads_(self, batch_size: int)
    def has_nan_grads(self) -> bool
    def reset_accumulated_grads_(self)
```

**AutoStepOptimizer** (`src/node0/server/optim.py`, extends `hivemind.Optimizer`)
```python
class AutoStepOptimizer(Optimizer):
    def __init__(self, model, optimizer_lock, sparse_avg, auto_step_time, ...)
    def step(self, batch_size: Optional[int] = None) -> None
    def _check_and_accumulate_gradients(self, batch_size: int) -> bool
    def _make_gradient_averager(self, factory, **kwargs) -> GradientAverager
```

**ModuleCollab** (`src/node0/server/module_collab.py`, extends `hivemind.moe.server.ModuleBackend`)
```python
class ModuleCollab(ModuleBackend):
    def backward(self, *inputs: torch.Tensor) -> Tuple[torch.Tensor, ...]
    def on_backward(self, batch_size: int) -> None
        # calls self.optimizer.step(batch_size=batch_size)
        # calls self.optimizer.zero_grad()
```
This is where training steps happen. Integration point for data-driven loss.

**WorkerMetricsV1** (`src/node0/security/validation.py`, Pydantic v1 via `pydantic.v1`)
```python
class WorkerMetricsV1(BaseModel):
    peer_id: str
    num_flop: StrictFloat
    active_time: StrictFloat
```

**PluralisAuthorizer** (`src/node0/security/authorization.py`, extends `hivemind.utils.auth.TokenAuthorizerBase`)
- `join_experiment()` calls external auth server
- Assigns `pipeline_stage` — must be made optional for self-hosted

**MonitorWorker** (`src/node0/utils/monitor.py`, extends `BaseMonitor`, `Thread`)
```python
class MonitorWorker(BaseMonitor):
    def monitor_callback(self, model, model_conf, active_period_timeout)
    def report(self, current_time)  # stores WorkerMetricsV1 to DHT
    def add_auth_info(self, authorizer, peer_id, stage, local_public_key)
    def connect_dht(self, dht)
```

#### New Types Required

**LoRAConfig** (new Pydantic model for fine-tuning)
```python
class LoRAConfig(BaseModel):
    rank: int = 64
    alpha: float = 128.0
    dropout: float = 0.05
    target_modules: list[str] = ["wq", "wk", "wv", "wo"]
```

**FineTuneConfig** (new, for YAML config extension)
```python
class FineTuneConfig(BaseModel):
    mode: Literal["sft", "dpo"] = "sft"
    corpus_path: str
    checkpoint_dir: str
    checkpoint_every_n_steps: int = 100
    eval_every_n_steps: int = 500
    role: Literal["mayor", "polecat", "witness", "deacon"]
    lora: LoRAConfig | None = None
```

**CorpusEntry** (new, for data pipeline)
```python
class CorpusEntry(BaseModel):
    id: str               # bead_id or session_id
    role: str             # "polecat", "mayor", "witness", "deacon"
    rig: str
    task_type: str        # "bead-creation", "code-review", "git-workflow", "planning"
    outcome: str          # "success", "failure"
    created_at: datetime
    messages: list[dict]  # [{"role": "user|assistant|system", "content": "..."}]
    # For DPO pairs:
    chosen: list[dict] | None = None
    rejected: list[dict] | None = None
```

**ModelRegistryEntry** (new, Dolt table row)
```python
class ModelRegistryEntry(BaseModel):
    name: str             # "mayor-v3"
    role: str
    base_model: str       # "llama-8b"
    version: int
    corpus_hash: str
    eval_score: float | None
    eval_scores_by_category: dict | None
    lora_adapters: list[str]
    deployment_status: str  # "active", "archived", "candidate"
    deployed_to: list[str]  # rig names
    created_at: datetime
    training_config: dict   # hyperparams snapshot
```

**TrainingMetricsV1** (new, extends DHT schema)
```python
class TrainingMetricsV1(BaseModel):
    peer_id: str
    step: StrictInt
    loss: StrictFloat
    role: StrictStr
    corpus_hash: StrictStr
```

**EvalResult** (new)
```python
class EvalResult(BaseModel):
    model_name: str
    category: str         # "bead_management", "git_workflow", etc.
    score: float          # 0.0-1.0
    tasks_run: int
    tasks_passed: int
    timestamp: datetime
```

### Extension Points

1. **Expert Class Registry** (Hivemind `name_to_block` and `name_to_input` dicts)
   - In `layers.py`, experts register via `@register_expert_class(name, sample_input_fn)`
   - LoRA variants can register as new expert classes (e.g., `"lm_head_lora"`) OR LoRA wrapping applied inside existing expert `__init__` when `model_args.lora_rank is not None`
   - `custom_module_path` parameter provides clean extension point for custom experts

2. **Gradient Averager Factory Pattern**
   - `GradientAveragerFactory = Callable[..., TGradientAverager]`
   - LoRA-aware gradient averager can be registered as new class in YAML:
     ```yaml
     grad_avg_config:
       class_name: node0.server.lora_averager.LoRAGradientAverager
       init_args:
         averager_rank: 64
         lora_only: true
     ```

3. **Formula System** (`/home/ubuntu/gt/.beads/formulas/`)
   - Formulas are TOML files with `description`, `formula` (name), `version`, `[[steps]]` blocks, `[vars]`
   - New formulas needed: `train-role-model.formula.toml`, `corpus-collect.formula.toml`, `eval-suite.formula.toml`

4. **Town Settings Agent/Role Registration** (`/home/ubuntu/gt/settings/config.json`)
   - `"agents"` dict — each entry is an agent definition
   - Role keys at top level (`"deacon"`, `"witness"`, `"polecat"`, etc.) — point to an agent name
   - Progressive rollout happens by switching agent assignments

5. **Bead Status Extension**
   - Training job lifecycle: `corpus-validating` → `ready` → `training` → `syncing` → `checkpointing` → `evaluating` → `done`/`failed`/`degraded`
   - May use existing bead statuses (`open`, `in_progress`, `done`, `failed`) with tags/labels OR extend bead schema with custom status values
   - Dolt-backed bead system stores status in SQL table; investigation needed on support for custom statuses

### Data Layer

**Existing data models:**

1. **Bead/Issue Storage (Dolt)** — See `/home/ubuntu/gt/.beads/metadata.json`:
   ```json
   {
       "backend": "dolt",
       "database": "dolt",
       "dolt_database": "hq",
       "dolt_mode": "server",
       "dolt_server_host": "127.0.0.1",
       "dolt_server_port": 3307,
       "jsonl_export": "issues.jsonl"
   }
   ```
   - Dolt is primary bead store. Training jobs are first-class beads.
   - Dolt supports branching (mechanism for DPO preference collection)

2. **Node0 DHT Storage** — Used for:
   - Peer discovery and training coordination (Hivemind native)
   - Worker metrics: `{experiment_prefix}_{stage}_worker_metrics` key
   - Port reachability: `{experiment_prefix}_{peer_id}_worker_ports`
   - Run parameters: `RunParameters` schema

3. **Node0 YAML Config** — `/home/ubuntu/gt/node0/crew/neo/src/node0/configs/llama_8B_C.yaml`
   - Single config file currently. Fine-tuning requires additional per-role config files.

**New tables/collections required:**

1. **Model Registry (Dolt table in HQ database)**
   ```sql
   CREATE TABLE model_registry (
       name VARCHAR(255) PRIMARY KEY,
       role VARCHAR(50) NOT NULL,
       base_model VARCHAR(255) NOT NULL,
       version INT NOT NULL,
       corpus_hash VARCHAR(64),
       eval_score FLOAT,
       eval_scores_json TEXT,
       lora_adapters_json TEXT,
       deployment_status VARCHAR(50),
       deployed_to_json TEXT,
       created_at DATETIME NOT NULL,
       training_config_json TEXT,
       artifact_path VARCHAR(512)
   );
   ```

2. **Corpus Store (filesystem + metadata in Dolt)**
   - Corpus entries: file-based under `~/gt/.corpus/<role>/<entry_id>.json` OR embedded in Dolt table
   - Schema matches `CorpusEntry` type
   - Provenance fields (bead_id, rig, session_id) link to existing bead records

3. **ChromaDB Vector Index** (per rig or centralized)
   - Collections: `beads`, `configs`, `sessions`
   - Updated on every `bd sync` and config change

4. **DPO Preference Pairs (Dolt table)**
   ```sql
   CREATE TABLE dpo_preferences (
       id VARCHAR(64) PRIMARY KEY,
       task_bead_id VARCHAR(255),
       model_a VARCHAR(255),
       model_b VARCHAR(255),
       winner VARCHAR(10),
       chosen_branch VARCHAR(255),
       rejected_branch VARCHAR(255),
       created_at DATETIME NOT NULL,
       role VARCHAR(50)
   );
   ```

---

## Patterns & Conventions

### Precedent Features

The closest precedents for integration patterns are:

1. **MonitorWorker** (`src/node0/utils/monitor.py`)
   - Runs as `Thread` (daemon=True)
   - Receives events via `mp.Queue` attached to root logger
   - Reacts to lifecycle events by parsing log strings with regex patterns
   - Reports state externally (DHT store) on a timer
   - Injected into `Node0Server.create()` as optional `monitor=` parameter
   - Started explicitly: `monitor.start()` before server creation, then `monitor.connect_dht(dht)` after DHT is up

2. **AutoStepOptimizer** (`src/node0/server/optim.py`)
   - Extends `Hivemind.Optimizer` by adding background `threading.Thread` (`_monitor_thread`)
   - Uses `threading.Event` (`_should_stop`) for clean shutdown
   - Uses `mp.Lock` (`_step_lock`) for thread-safe shared state access
   - Calls `_start_monitor()` from `Node0Server.__init__()` after `super().__init__()`
   - New functionality contained in private methods

3. **ModuleCollab** (`src/node0/server/module_collab.py`)
   - Extends `hivemind.moe.server.ModuleBackend`
   - Overrides `backward()` to add `optimizer_lock` wrapping
   - `on_backward()` is the hook where training steps happen — integration point for data-feeding

**Pattern for new subsystems:**
- New class extending `Thread` or wrapping existing class
- Takes `queue` or callback for integration with existing event flow
- Optional parameter added to `Node0Server.create()` signature with `| None = None` default
- Startup sequence: create before server, connect after DHT is available
- Shutdown via `threading.Event` + `join(timeout=N)`

### File & Naming Conventions

**Files:**
- `snake_case.py`. Prefix `HM_` for direct Hivemind modifications
- New fine-tuning modules under `src/node0/finetune/` top-level package:
  - `data/` — tokenizer, dataset, dataloader, loss functions
  - `lora/` — LoRA wrapping, adapter application
  - `checkpoint/` — checkpoint gathering, export (safetensors/GGUF)
  - `inference/` — autoregressive generation, sampling, KV-cache
  - `corpus/` — corpus pipeline (scrubbing, formatting, partitioning)
  - `registry/` — model registry (metadata, versioning, CLI)
  - `eval/` — automated eval task suite

**Classes:**
- `PascalCase`. Suffix pattern for subclasses makes origin clear (e.g., `ModuleCollab`, `AutoStepOptimizer`, `Node0Server`)

**Functions:**
- `snake_case`. Factory functions named `<verb>_<noun>()` (e.g., `authorize_with_pluralis`, `get_node_info`, `make_validators`)

**Constants/registries:**
- Module-level dicts in `snake_case` (e.g., `schedule_name_to_scheduler`)

**Private methods:**
- Single underscore prefix (`_auto_step`, `_monitor_step`, `_resync_state`)

**Type aliases:**
- `CamelCase` at module level (e.g., `GradientAveragerFactory`, `GroupID`, `GatheredData`)

### Configuration Patterns

**YAML as primary config format:**
- Configs reference classes by dotted path string (`"node0.models.llama.arguments.LlamaArguments"`)
- `build_cls()` in `utils/common.py` does `importlib.import_module` + instantiate
- `partial_init=True` wraps in `functools.partial` for deferred instantiation (optimizer factories)
- New fine-tune configs should follow same `class_name` / `init_args` YAML structure

**Pydantic for all structured data:**
- All config/data transfer objects are `BaseModel` subclasses
- Type annotations use Python 3.10+ union syntax (`int | None`)
- Field defaults set in class body, not validators
- `model_validator(mode="after")` used for derived field computation
- New `LoRAArguments`, `CorpusConfig`, `DataLoaderConfig` should follow this pattern

**Argument merging in `run_server.py`:**
- CLI args override YAML config
- `args.pop("key")` removes keys consumed locally (not forwarded to server)
- Remaining kwargs passed to `Node0Server.create(**args)`

### Coding Patterns

**Logger instantiation (universal throughout):**
```python
from hivemind.utils.logging import get_logger
logger = get_logger(__name__)
```
Every module file declares this at module level. No exceptions. All logging via this logger — no `print()`.

**Log levels:**
- `logger.info()` — normal operational events
- `logger.error()` — errors, always followed by exit or raise
- `logger.warning()` — recoverable issues, retries
- `logger.debug()` — verbose detail
- `logger.log(self.status_loglevel, ...)` — configurable verbosity in loops

**Factory function pattern:**
```python
def authorize_with_pluralis(...) -> PluralisAuthorizer:
    """..."""
    authorizer = PluralisAuthorizer(...)
    try:
        authorizer.join_experiment(...)
        return authorizer
    except NotInAllowlistError as e:
        logger.error(f"Authorization failed: {e}. Exiting run.")
        exit(1)
```
Complex initialization logic lives in module-level factory functions, not `__init__`. Class itself is lean.

**@classmethod create() pattern:**
```python
@classmethod
def create(cls, ..., *, start: bool, **kwargs) -> Server:
    """..."""
    return cls(..., start=start)
```
- `@classmethod` named `create()` for complex multi-step initialization
- `start: bool` is keyword-only (after `*`) — forces explicit choice
- All optional parameters have defaults

**functools.partial for deferred factory construction:**
```python
grad_avg_factory = build_cls(grad_avg_config["class_name"], grad_avg_config["init_args"], partial_init=True)
```

**TypeVar for factory types:**
```python
TGradientAverager = TypeVar("TGradientAverager", bound="GradientAverager")
GradientAveragerFactory = Callable[..., TGradientAverager]
```

**Thread safety:**
- `mp.Lock()` for multiprocessing-safe locks (passed between processes)
- `threading.Lock()` for intra-process thread locks
- `threading.Event()` for stop signals (`_should_stop`)
- `mp.Queue()` for cross-process communication (logging)

**macOS fallbacks:**
```python
if platform.system().lower() == "darwin":
    grad_avg_config["class_name"] = "node0.server.power_sgd_averager_mac.PowerSGDGradientAverager"
```

**Exit patterns for fatal errors:**
```python
logger.error("Fatal error message. Exiting run.")
os.killpg(os.getpgrp(), signal.SIGTERM)  # if child processes need termination
# OR
exit(1)  # if in main process before children spawned
```

**Copyright header:**
```python
# Copyright 2025 Pluralis Research
#
# Licensed under the Apache License, Version 2.0 (the "License");
```

### Error Handling

**Custom exception hierarchy:**
```python
class NonRetriableError(Exception): pass
class RetriableError(Exception): pass

class NotInAllowlistError(NonRetriableError): pass
class BadRequestError(NonRetriableError): pass
class IntegrityError(NonRetriableError): pass
class ServerUnavailableError(RetriableError): pass
```

**call_with_retries pattern:**
```python
def call_with_retries(func: Callable, n_retries: int = 10, initial_delay: float = 1.0) -> Any:
    while True:
        try:
            return func()
        except NonRetriableError:
            raise
        except ServerUnavailableError as e:
            # special handling with server-specified delay
        except Exception as e:
            if i >= n_retries:
                raise
            delay = initial_delay * (2**i)  # exponential backoff
            time.sleep(delay)
```
Exponential backoff with `initial_delay * (2**i)`. Non-retriable errors propagate immediately.

**Error propagation in authorization:**
```python
try:
    ...
except requests.exceptions.HTTPError as e:
    if e.response.status_code in [401, 403, 429]:
        raise NotInAllowlistError(error_detail) from None
    if e.response.status_code in [400, 413, 418, 422, 424]:
        raise BadRequestError(e.response.json()["detail"]) from None
    raise e  # unexpected HTTP errors propagate as-is
```

**NaN/Inf gradient guard:**
```python
if self.grad_averager.has_nan_grads():
    self.tracker.report_local_progress(self.local_epoch, samples_accumulated=0)
    logger.error("Encountered incorrect value in grads, exiting run")
    os.killpg(os.getpgrp(), signal.SIGTERM)
```

**Background thread error handling:**
```python
def _auto_step(self) -> None:
    try:
        ...
    except Exception as e:
        logger.error(f"Error in auto step: {e}")
```

**Async error handling:**
```python
try:
    ...
except BaseException as e:
    self.finalize(exception=e)
    for task in pending_tasks:
        task.cancel()
    raise
finally:
    for task in pending_tasks:
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception as inner_exc:
            logger.debug(f"Task {task} failed with {inner_exc}", exc_info=True)
```

### Code Conventions to Observe

1. **Import ordering:** stdlib → third-party → local (from `node0.xxx` import). Local imports always absolute from package root.

2. **Type hints everywhere:** All function signatures have type hints including return types. `Any` for types that can't be easily typed (e.g., cross-module auth objects).

3. **f-strings for all logging:** `logger.info(f"Running with configuration: {json.dumps(args, indent=4)}")`. No %-style or `.format()`.

4. **No bare `except:` clauses:** Always `except Exception as e:` or specific types. Only `except BaseException` in async code for `CancelledError`.

5. **`args.pop()` for consumed config keys:** In `run_server.py`, keys consumed locally are removed before forwarding remaining kwargs to server.

6. **`Path` over string paths:** From `pathlib import Path`. File path manipulation uses `Path` objects.

7. **Pydantic v2 for new code, `pydantic.v1` for Hivemind compatibility:** `validation.py` uses `from pydantic.v1` because Hivemind requires v1 API. All new node0 code uses `from pydantic` (v2).

8. **Shared memory for cross-process tensors:** `torch.tensor(0.0).share_memory_()` pattern when tensors need to be read across processes.

9. **vocab_size fix is a blocker:** `LlamaArguments` has `vocab_size: int = 50265` (OPT tokenizer) instead of LLaMA's `32000`/`128256`. Must be corrected before data loading or tokenizer integration.

10. **No enforced linting config:** Codebase consistently formatted (4-space indentation, ~100 char line length, no trailing commas in function definitions).

### Testing Approach

**No existing test infrastructure.** No `tests/` directory, pytest config, or test files. `.gitignore` includes `pytest_cache/` and `coverage.xml` entries but are unused.

**Recommendation for new code:**
- Establish `tests/` at repo root with `tests/unit/` and `tests/integration/`
- Use `pytest` (natural for Python 3.11 + PyTorch + Hivemind)
- Unit tests should mock DHT, gRPC, network calls
- Integration tests would require actual Hivemind DHT (likely skipped in CI)
- Highest-value targets: corpus pipeline, data scrubbing, eval suite (pure Python with no distributed dependencies)

---

## Key Files Reference

### Critical Files for Fine-Tuning Implementation

| Feature | File(s) | Purpose |
|---------|---------|---------|
| Configuration schema | `src/node0/models/llama/arguments.py` | **BLOCKER:** Fix `vocab_size` (50265 → 32000/128256); add LoRA fields |
| Model wrapping | `src/node0/models/llama/layers.py` | Wrap HeadExpert, BodyExpert, TailExpert with LoRA per-stage |
| Training orchestration | `src/node0/server/node0_server.py` | Parameterize `checkpoint_dir`, fix param filtering for LoRA |
| Step scheduling | `src/node0/server/optim.py` | Replace `os.killpg` NaN handler; add fine-tuning step scheduling |
| Forward/backward hook | `src/node0/server/module_collab.py` | `on_backward()` integration point for data feeding |
| Entry point | `src/node0/run_server.py` | Add `--local-mode`, fine-tuning CLI args; skip auth conditionally |
| Monitoring | `src/node0/utils/monitor.py` | Add training metrics (loss, eval scores); new regex patterns |
| Metrics schemas | `src/node0/security/validation.py` | Add `TrainingMetricsV1` schema |
| Authorization | `src/node0/security/authorization.py` | Make `PluralisAuthorizer` optional |
| Gradient averager | `src/node0/server/HM_gradient_averager.py` | Works as-is if caller filters to LoRA params |

### New Modules to Build

| Subsystem | Files | Integration Hook | Pattern |
|-----------|-------|------------------|---------|
| Data loading | `finetune/data/loader.py`, `dataset.py`, `tokenizer.py`, `loss.py` | `module_collab.py:on_backward()` | `MonitorWorker` Thread pattern |
| LoRA adapters | `finetune/lora/adapter.py`, `config.py`, `optimizer_filter.py` | `node0_server.py:225` after `name_to_block[...]()` | `build_cls` + `ModelArguments` fields |
| Checkpoint export | `finetune/checkpoint/gatherer.py`, `assembler.py`, `exporter.py` | `node0_server.py:316` `checkpoint_dir` | `MonitorWorker` Thread pattern |
| Inference | `finetune/inference/generator.py`, `sampling.py`, `kv_cache.py`, `prompt_api.py`, `server.py` | New entry point, reuses gRPC pipeline | Mirrors `run_server.py` structure |
| Corpus pipeline | `finetune/corpus/collector.py`, `formatter.py`, `scrubber.py`, `store.py`, `validator.py` | Standalone + Gas Town session hooks | Factory function pattern |
| Model registry | `finetune/registry/store.py`, `cli.py`, `ui.py` | Dolt table + file storage | `NodeInfo` Pydantic model pattern |
| Eval suite | `finetune/eval/runner.py`, `tasks/`, `scorer.py` | Post-checkpoint gate | Standalone runner |
| PII scrubbing | `finetune/data/scrubber.py` | Within data pipeline | TruffleHog + detect-secrets integration |

### Build System Files

| File | Purpose | Changes Needed |
|------|---------|-----------------|
| `pyproject.toml` | Package metadata, deps, build config | Add optional deps: `finetune = [transformers, peft, safetensors, chromadb, detect-secrets]`; add `[project.scripts]`: `node0-inference`, `node0-finetune`, `node0-corpus` |
| `Dockerfile` | Container build | Add: transformers, chromadb, detect-secrets; volume mounts for corpus and checkpoints |
| `generate_script.py` | Docker/source setup script generator | Update to include fine-tuning binaries |
| `run.json` | Seed peer multiaddrs + run config ref | Support local seed peers + `bypass_auth: true`, `local_stage` config |
| `src/node0/configs/*.yaml` | Training configs | Create `finetune/lora_sft.yaml`, `finetune/dpo.yaml` per role |

---

## Constraints & Considerations

### Architectural Boundaries to Respect

1. **Pipeline stage isolation.** Each node runs exactly one stage. Any module that touches model parameters must be stage-aware. The LoRA adapter and checkpoint gatherer must know which stage they are operating on (from `model_args.stage`). Data loading must feed head nodes, not body nodes. Loss must be computed only on tail nodes.

2. **DHT as the coordination bus.** Parameters, run state, and worker metrics flow through DHT. New subsystems (corpus validation status, checkpoint gather coordination, eval results) should use DHT or compatible broadcast mechanism rather than direct connections.

3. **The training loop is external.** `GradientAverager` does not drive the training loop — the training loop drives it. The new data loader + loss computation in `src/node0/data/` must integrate at this point: run forward/backward on a batch, then call `grad_averager.accumulate_grads_(batch_size)`. `AutoStepOptimizer.step()` is called by `ModuleCollab.on_backward()` after each batch's backward pass.

4. **Authorization is a hard external dependency.** `PluralisAuthorizer` calls `https://auth.pluralis.ai` to receive a `pipeline_stage` assignment. This is incompatible with self-hosted fine-tuning. The authorization path must be conditional (`check_integrity=False` already exists; stage assignment also needs a bypass). New `--bypass_auth` or `--local_stage` flag needed in `run_server.py`.

5. **os.killpg as error handling.** Multiple places (`optim.py`, `monitor.py`, `authorization.py`) call `os.killpg(os.getpgrp(), signal.SIGTERM)` to terminate on error. This is fine for production pretraining (fail fast) but must be handled differently for fine-tuning where the bead lifecycle must be updated to `failed` state before exit. A pre-exit hook or graceful shutdown mechanism is needed.

### Known Issues to Fix Before Implementation

1. **vocab_size mismatch (BLOCKER):** `LlamaArguments.vocab_size = 50265` (OPT tokenizer) vs. LLaMA's `32000` (LLaMA 2) or `128256` (LLaMA 3). Tokenizer misalignment causes shape mismatches. Must be corrected before any data loading.

2. **No checkpoint saving:** `checkpoint_dir=None` hardcoded in `node0_server.py:316`. Checkpoint gathering from 32 pipeline stages is architecturally complex.

3. **No data loading:** Node0 has zero data loading. The `GradientAverager` expects `param.grad` to already be populated — doesn't know where data comes from.

4. **NaN handling too aggressive:** `os.killpg` on NaN grads in `optim.py:216` — appropriate for pretraining, not fine-tuning where gradient instability is more common.

5. **No test infrastructure:** No `tests/` directory or pytest config.

### Dependencies and Compatibility

- **Python 3.11 exactly** (`>=3.11,<3.12`). No flexibility.
- **hivemind pinned to commit `4d5c414`** via GitHub URL. Any Hivemind API changes require pinning new commit.
- **torch==2.7.0 exactly.** Enforced both in `pyproject.toml` and at runtime in `run_server.py`.
- **Pydantic >= 2.0** for new code; `pydantic.v1` for Hivemind compatibility only.
- **New dependencies for fine-tuning:** `transformers`, `peft` (or `unsloth`), `safetensors`, `chromadb`, `detect-secrets`.

### Integration with Gas Town

- Node0 package is the training/inference backend
- Gas Town provides: formulas (train-role-model, corpus-collect, eval-suite), bead lifecycle hooks, settings/provider registration
- Session corpus collection happens on bead close via Gas Town hook
- Progressive rollout by changing `settings/config.json` role-to-agent mappings
- DPO preference pairs collected via Dolt branching + manual winner selection

### Build and Deployment

**Build system:** `hatchling` (declared in `pyproject.toml`). No Makefile, npm, or other build systems.

```bash
pip install .     # installs node0 package
pip install -e .  # editable install for development
```

**Container build:** Dockerfile produces deployable image (NVIDIA CUDA 12.1.1 + conda + pip).
- For fine-tuning: add dependencies, volume mounts for corpus and checkpoints (weights not baked into image).

**Entry points:** Currently `src/node0/run_server.py`. New entry points needed:
- `node0-inference` — CLI for inference server
- `node0-finetune` — CLI to launch fine-tuning job
- `node0-corpus` — CLI for corpus management
- `gt model` — Gas Town CLI extension (registry management)

---

## Consolidated Integration Map

| Spec Requirement | Touch Point | File(s) | Action | Priority |
|---|---|---|---|---|
| vocab_size fix | Model config | `models/llama/arguments.py` | Change `50265` to LLaMA-appropriate value | **BLOCKER** |
| LoRA per pipeline stage | Expert init + config | `layers.py`, `arguments.py`, `llama/arguments.py` | Add LoRA fields; wrap linears in experts | HIGH |
| Data loading pipeline | Gradient computation | NEW: `finetune/data/` | Build tokenizer, dataset, dataloader, loss | HIGH |
| Checkpoint export | Server creation | `node0_server.py`, NEW: `finetune/checkpoint/` | Parameterize checkpoint_dir; add gather logic | HIGH |
| Autoregressive inference | Pipeline forwarding | NEW: `finetune/inference/`, `run_inference.py` | Generation loop, sampling, KV-cache | MEDIUM |
| Corpus collection + scrubbing | Training data input | NEW: `finetune/corpus/` | Transcript extraction, PII scrubbing, formatting | MEDIUM |
| Model registry | Artifact management | NEW: `finetune/registry/`, Dolt table | Metadata tracking, versioning, CLI | MEDIUM |
| Eval task suite | Quality gates | NEW: `finetune/eval/` | Automated eval runner, task suite | MEDIUM |
| Training metrics | Monitoring | `monitor.py`, `validation.py` | Add loss, eval scores; new schemas | MEDIUM |
| PluralisAuthorizer bypass | Auth flow | `authorization.py`, `run_server.py` | Add `--local-mode` flag; skip external auth | MEDIUM |
| NaN handling for fine-tune | Error recovery | `optim.py` | Replace kill with exception + retry | LOW |
| node0 provider in town settings | Agent registration | `/home/ubuntu/gt/settings/config.json` | Add agent entries + role mappings | MEDIUM |
| Training formulas | Orchestration | NEW TOML in `~/.beads/formulas/` | `train-role-model`, `corpus-collect`, `eval-suite` | MEDIUM |
| Progressive rollout | Deployment | `settings/config.json` | Role-to-agent assignment switching | LOW |
| New YAML configs | Configuration | `configs/finetune/*.yaml` | Per-role fine-tune configs with corpus_path, LoRA | MEDIUM |
| Build system updates | Packaging | `pyproject.toml`, `Dockerfile` | Add optional deps, new scripts, volumes | MEDIUM |
| DPO via Dolt branching | Preference collection | NEW: `dpo_preferences` table | Sling to two models, collect winner | LOW |
| RAG at inference time | Context retrieval | NEW: `finetune/rag/` | ChromaDB embed/retrieve beads, configs, sessions | LOW |

# Spec: Gas Town Fine-Tuning (gastown-finetune)

## Overview

Build a distributed fine-tuning system that trains role-specific, task-specialized LLM agents native to Gas Town, using node0's decentralized training infrastructure. Fine-tuned models progressively replace current Claude/pi/omp agents for Gas Town roles, starting with low-risk roles (deacon, boot, witness) and expanding to polecats, crew, and eventually mayor.

### Motivation

- **Prime bloat**: `gt prime` injects ~15K tokens of context every session. A fine-tuned model already knows conventions.
- **Convention errors**: Current agents frequently violate Gas Town conventions (wrong commands, bad dependency direction, skipping hooks) despite detailed instructions.
- **No baseline**: There's no measurable standard for "how good is a Gas Town agent?" — fine-tuning creates one.
- **Cost**: API costs for Claude/pi sessions. Self-hosted inference replaces per-token API pricing with fixed compute cost.

### What is a Gas Town-Native Agent?

A fully autonomous agent that knows Gas Town conventions, workflows, and tooling without being told. It can:
- Create, manage, and close beads with correct fields, dependencies, and lifecycle
- Use `gt` commands instead of raw shell equivalents
- Follow CLAUDE.md instructions and role-specific behaviors
- Execute the full sling lifecycle (branch, code, commit, PR)
- Plan work: generate bead hierarchies, estimate complexity, suggest dependency chains
- Operate within resource constraints without crashing the box

## Architecture

```
+------------------------------------------------------------------+
|                    Gas Town Orchestration                          |
|  +----------+  +----------+  +-------------------+               |
|  | Corpus    |  | Training |  | Model Registry    |               |
|  | Pipeline  |  | Formulas |  | (UI + CLI + API)  |               |
|  +-----+----+  +-----+----+  +---------+---------+               |
|        |              |                 |                          |
|        v              v                 v                          |
|  +----------------------------------------------+                |
|  |             node0 Training Layer              |                |
|  |  Hivemind DHT | Gradient Avg | Pipeline Par.  |                |
|  +----------------------------------------------+                |
|        |                              |                           |
|        v                              v                           |
|  +----------+                +----------------+                   |
|  | Role-     |                | Task-specific  |                   |
|  | specific  |                | LoRA adapters  |                   |
|  | models    |                | (stackable)    |                   |
|  +-----+----+                +-------+--------+                   |
|        |                              |                           |
|        v                              v                           |
|  +----------------------------------------------+                |
|  |        node0 Inference Pipeline               |                |
|  | gRPC pipeline fwd | Autoregressive loop      |                |
|  | Sampling | KV-cache | Prompt API              |                |
|  +---------------------+------------------------+                |
|                         |                                         |
|                    +----+----+                                    |
|                    | ChromaDB |   <-- RAG: beads, configs,        |
|                    | (RAG)    |       sessions, formulas           |
|                    +----+----+                                    |
|                         |                                         |
|                         v                                         |
|  +----------------------------------------------+                |
|  |      New Provider: "node0" in town settings   |                |
|  |  Replaces: deacon > witness > polecat > crew  |                |
|  +----+-----------------------------------------+                |
|       |                                                           |
|       +-----> Session logs -----> Corpus Pipeline (feedback loop) |
+------------------------------------------------------------------+
```

### Key Architectural Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Base model | Model-agnostic (V1 scoped to LLaMA family) | Try multiple long-term; V1 uses LLaMA since node0 only has LLaMA model defs. Adding new architectures requires new model layers, args classes, pipeline stage defs. |
| Model specialization | Role-specific | Separate models for mayor, polecat, witness, deacon — different training data and behavior |
| Task specialization | LoRA adapters | Stackable task-specific adapters (bead-creation, code-review, git-workflow, planning) switchable at inference |
| Training approach | Distributed via node0 | Node0 IS the distributed training infrastructure — Hivemind DHT, gradient averaging, pipeline parallelism |
| Training hardware | External GPU infrastructure | Not on the 10GB dev box. Proper GPU nodes for training |
| Inference | Decentralized via node0 pipeline forwarding | node0's forward pass through pipeline stages IS inference. Must add: autoregressive generation loop, prompt API, sampling logic. Not a separate system — extends existing gRPC pipeline. |
| RAG system | Embedded vector DB (ChromaDB) | Indexes beads, configs, recent sessions. Injected into context at inference time. |
| Corpus export | Bead-linked export (primary) | Session transcripts attached to bead on close. Supplemented by other methods for wisps/gaps. |
| Conflict resolution | Recency wins | Most recent version of any artifact takes precedence in corpus. Older entries auto-excluded. |
| Validation strategy | Progressive rollout IS the validation | No separate shadow/canary. Eval suite + progressive replacement (deacon → witness → polecat) is sufficient. |
| Cross-rig data | Opt-in per rig | Each rig explicitly enables corpus sharing. Default is rig-local only. |
| Safety | Standard Gas Town guardrails | GUPP violations, failing to submit MRs, standard gt conventions. Enforced via eval suite. |
| Eval thresholds | Deferred to implementation | Pick thresholds empirically after first training run. Can't know what's reasonable until real numbers exist. |
| Preference learning | DPO via Dolt branching | Same task slung to two models → two Dolt branches → user/rig picks winner → preference pairs collected at task level. No simultaneous forward passes needed. |
| Dynamic content | RAG complement | Fine-tune for behavior/conventions, RAG for evolving state (current beads, configs, recent context) |
| Data freshness | Retrain ASAP + RAG gap | Retrain when significant new data arrives. RAG covers the gap between training snapshots |
| Provider integration | New "node0" provider | Registered in town settings alongside claude, pi, omp |
| Config management | Gas Town formula templates | Training configs are formulas — composable, versionable, shareable |

## Data Pipeline & Corpus

### Data Sources

**Session transcripts:**
- Polecat sessions (both successful and failed — DPO signal)
- Mayor sessions
- Crew interactions
- Witness reports

**Codebase artifacts:**
- CLAUDE.md files across rigs
- Formula definitions (`.formula.toml`)
- Spec documents and plan files
- AGENTS.md and config files

**Operational data:**
- Bead lifecycles (create → in_progress → done/failed)
- Dependency graphs
- Sling outcomes (success/failure, duration, error type)
- Convoy logs
- `gt prime` output

### Corpus Pipeline

```
1. COLLECT              2. SCRUB                3. FORMAT               4. PARTITION (data)
Session logs       -->  Automated PII      -->  Role-tagged        -->  By role:
Bead history            removal:                instruction pairs:      - mayor corpus
Config files            - Credentials           - system prompt         - polecat corpus
Formula specs           - API tokens/keys       - user turn             - witness corpus
gt prime output         - IP addresses          - assistant turn        By task:
                                                DPO pairs:              - bead-creation
                                                - chosen (success)      - code-review
                                                - rejected (fail)       - git-workflow
                                                                        - planning
```

### Data Characteristics

- **Volume**: 1K-10K examples initially (note: split across 4+ roles = 250-2500 per role per model. Published research suggests 10K+ for a single task. Volume will grow over time as more sessions are captured — initial runs may underperform until corpus matures)
- **Formats**: Mixed — structured JSON, markdown, conversation logs
- **Scrubbing**: Multi-layer automated pipeline:
  1. Secret scanning tool (truffleHog or detect-secrets) for JWTs, PEM blocks, inline secrets
  2. Regex patterns for API keys, tokens, IP addresses, file paths
  3. Confidence-based flagging for uncertain matches (quarantine, don't silently include)
  4. Periodic audit/sampling of scrubbed output to catch novel credential formats
  - No manual review of every entry, but automated quality gates prevent silent credential leakage.
- **Data retention**: No post-training removal. No right-to-be-forgotten requirement.

### Extensibility Path

1. **V1**: Collect from our own Gas Town rigs
2. **V1+**: Extend so other operators can contribute data (incentive model: contribute data, get model access even without GPU)
3. **Future**: Federated Gas Town (out of V1 scope)

## Training Infrastructure

### node0: What Exists vs What We Build

**Exists in node0 (use as-is):**
- Hivemind DHT peer discovery
- gRPC pipeline forwarding (activations between stages)
- YAML config system

**Exists in node0 (needs modification for fine-tuning):**
- Decentralized gradient averaging — works for pretraining, needs adaptation for LoRA-only gradients
- PowerSGD compression — for gradient bandwidth reduction, not model compression
- Pipeline parallelism — each node runs 1 of 32 layers; LoRA adapters must be applied per-stage
- PluralisAuthorizer — requires auth.pluralis.ai and assigns pipeline stages; needs to be made optional or replaceable for self-hosted fine-tuning
- AutoStepOptimizer — auto-triggers steps on timer, kills process on NaN grads (`os.killpg`); needs fine-tuning-aware step scheduling and graceful error handling
- MonitorWorker — provides infrastructure metrics; needs training-specific metrics (loss, eval scores, corpus stats)

**Must be built (major subsystems):**

1. **Data loading pipeline (EPIC — largest engineering task)**
   - node0 has ZERO data loading. No DataLoader, no dataset abstraction, no tokenizer integration, no loss computation
   - The training loop is entirely external — GradientAverager expects gradients to already exist in `param.grad`
   - Sub-tasks: tokenizer integration, dataset format definition, DataLoader implementation, loss function (cross-entropy for SFT, DPO loss for preference), batch assembly, integration with pipeline-parallel architecture

2. **Checkpoint export and model assembly**
   - `checkpoint_dir=None` is hardcoded in `node0_server.py`. No save mechanism exists
   - Model exists only in distributed memory across 32 pipeline stages — no single node has the full model
   - Must: gather distributed weights from all pipeline stages, assemble into single state dict, serialize to safetensors + GGUF
   - Architecturally complex due to pipeline-parallel splitting

3. **LoRA adapter support (per pipeline stage)**
   - ModelArguments has no LoRA fields (rank, alpha, target_modules, dropout)
   - Must apply LoRA wrapping per pipeline stage, not globally
   - Optimizer must operate on adapter params only, not all params
   - Framework selection (PEFT, Unsloth) deferred to implementation

4. **Autoregressive inference (generation loop + prompt API)**
   - node0's forward pass through pipeline stages IS inference — gRPC moves activations between stages
   - Missing: autoregressive token generation loop, sampling/temperature/top-p, prompt-in/response-out API
   - This is a moderate addition to node0's existing pipeline forwarding, not a separate system

5. **DPO preference collection via Dolt branching**
   - Same task slung to two model versions → both produce Dolt branches → user/rig picks better outcome
   - Winner = "chosen", loser = "rejected" → preference pairs at task level
   - Periodically retrain with accumulated preferences
   - No simultaneous reference+policy forward passes needed (unlike standard ML DPO)

6. **Training job lifecycle mapped to beads**
   - Bead states: `corpus-validating` → `ready` → `training` → `syncing` → `checkpointing` → `evaluating` → `done` / `failed` / `degraded`
   - Must handle: partial failures, straggler peers, checkpoint resume, peer loss during training

7. **Multi-model experiment tracking**
8. **Corpus data validation and quality gates**
   - Quality scoring before corpus inclusion
   - Anomaly detection on new additions
   - Provenance tracking (which session/user/rig generated each example)
   - Model poisoning prevention — especially critical for V1+ multi-operator extension

**Note on vocab_size:** Current `LlamaArguments` has `vocab_size: int = 50265` (OPT-2.7b tokenizer), not LLaMA's standard 32000/128256. Must be clarified/fixed before fine-tuning.

### Training Job Lifecycle (Bead-Mapped)

```
bd create "Train mayor-v3"            --> job bead created (status: open)
bd update --status in_progress        --> node0 training starts
  ... distributed gradient averaging ...
  ... checkpoint at configurable interval ...
  ... eval gate runs against checkpoint ...
bd close                              --> model exported to registry (status: done)
```

Training jobs are first-class beads — tracked, queryable, with full dependency support. A training job can depend on a corpus-collection job completing first.

### Training Configuration via Formulas

Training runs are configured through Gas Town formulas — composable, versionable, shareable.

**Note**: The formula below is illustrative of the workflow structure. Concrete command mappings (what each step actually executes) will be defined during implementation once the node0 training CLI and data loading interfaces are built. The steps map to the training pipeline but the exact `command` fields are TBD.

```toml
[formula]
name = "train-role-model"
type = "composition"

[variables]
role = { type = "string", values = ["mayor", "polecat", "witness", "deacon"] }
base_model = { type = "string" }
corpus_path = { type = "path" }
lora_rank = { type = "int", default = 64 }

[steps.prepare]
description = "Partition corpus, validate format, generate node0 YAML config"
# command TBD — depends on corpus pipeline and node0 config generation tooling

[steps.train]
description = "Launch node0 distributed training job"
# command TBD — depends on node0 training CLI (data loader, LoRA config)

[steps.eval]
description = "Run automated task suite against checkpoint"
# command TBD — depends on eval framework and checkpoint export

[steps.export]
description = "Export to safetensors + GGUF, register in model registry"
# command TBD — depends on checkpoint gathering and model registry CLI
```

### Hardware

Training runs on proper GPU infrastructure, not the 10GB dev box. GPU-less operators can contribute corpus data without running training themselves (incentive model: data contribution grants model access).

## Model Registry & Deployment

### Model Registry (CLI in V1; browsable UI deferred to V2)

A registry showing all trained models, their lineage, eval scores, and deployment status across rigs. The V1 implementation is **CLI-only** — `gt model list`, `show`, `deploy`, `rollback`, `prune`. A browsable web or TUI registry UI is deferred to V2.

> **Note (added during plan review, 2026-02-22):** The original spec decision Q70 stated "Full UI in V1 — CLI + browsable UI". During plan review, it was determined that building a browsable UI in V1 would add significant scope with low immediate value. The V1 scope is revised: CLI-only registry interface, browsable UI deferred to V2. The CLI delivers the core value (queryable metadata, deployment control, rollback). See Decision Log entry Q70-revised below.

**Storage design (to be detailed in implementation):**
- Model artifacts (weights, adapters): file-based storage under `~/gt/.models/` or configurable path
- Metadata (lineage, eval scores, deployment status): Dolt table in HQ database
- Schema must track: model name, role, base model, version, training corpus hash, eval scores per category, LoRA adapters, deployment status per rig, creation timestamp
- The full browsable UI is a V2 product feature

**CLI interface:**

```
gt model list
NAME          ROLE      BASE       VER    EVAL    STATUS
mayor-v3      mayor     llama-8b   3      87.3%   active
mayor-v2      mayor     llama-8b   2      82.1%   archived
polecat-v1    polecat   qwen-7b    1      79.5%   active
witness-v1    witness   llama-8b   1      91.2%   active

gt model show mayor-v3
Name:        mayor-v3
Role:        mayor
Base:        llama-8b
LoRA:        rank-64
Trained:     2026-02-20
Corpus:      1,847 examples
Eval score:  87.3%
DPO pairs:   342
Adapters:    bead-create, planning
Deployed to: sfgastown, frankentui
Status:      active

gt model adapters
NAME          TASK           VER    COMPATIBLE WITH
bead-create   bead mgmt      1      mayor-v3, polecat-v1
code-review   code review    1      polecat-v1
planning      planning       1      mayor-v3
git-workflow  git ops        1      polecat-v1
```

### Versioned Rollback

- Every trained model gets a version number
- Town settings reference specific version
- Rollback = point town settings to previous version
- Old versions retained until explicitly pruned
- Full lineage tracking: which corpus, which base model, which hyperparams

### Provider Integration

New `node0` provider type registered in town settings:

```json
{
  "agents": {
    "node0-mayor": {
      "command": "node0-inference",
      "args": ["--model", "mayor-v3", "--adapters", "planning,bead-create"],
      "provider": "node0"
    },
    "node0-polecat": {
      "command": "node0-inference",
      "args": ["--model", "polecat-v1", "--adapters", "code-review,git-workflow"],
      "provider": "node0"
    }
  },
  "roles": {
    "deacon": "node0-deacon",
    "witness": "node0-witness"
  }
}
```

### Progressive Rollout

Each stage gated by automated eval suite before promotion:

1. **Deacon + boot agent** — lowest risk, simplest tasks
2. **Witness** — monitoring, read-heavy, low blast radius
3. **Polecat** (specific task types) — bead creation, simple code changes
4. **Crew** — general-purpose work
5. **Mayor** — highest autonomy, last to replace (V2+)

## Evaluation & Quality

### Automated Eval Task Suite

Gates every model before deployment. Must pass threshold to enter registry as deployable.

**Scoring methodology**: Each eval task produces a binary pass/fail or numeric score (0-100). Category scores are the percentage of tasks passed. Overall eval score = weighted average across categories (weights TBD after first training run). Thresholds are deferred to implementation — pick empirically after real numbers exist — but the methodology is fixed: automated, reproducible, per-category scoring with a composite gate.

| Task Category | Example Tasks | Pass Criteria | Scoring |
|---------------|---------------|---------------|---------|
| Bead management | Create issue from description; close with reason; build epic hierarchy | Correct fields, deps, status transitions | Binary per-field + structural validation |
| Git workflow | Branch, commit, push; generate PR description; resolve merge conflict | No --force, hooks pass, clean merge | Binary per-step, merge quality score |
| Convention adherence | Use gt commands; follow CLAUDE.md; respect resource constraints | No pkill, role-appropriate behavior | Binary per-convention violation count |
| Planning | Generate bead hierarchy; estimate complexity; suggest deps | Deps enable parallelism, no cycles | Cycle-free + parallelism ratio |
| Code execution | Implement from bead description; fix bug; refactor | Tests pass, no regressions | Test pass rate + regression count |

### DPO via Dolt Branching (Task-Level Preference Learning)

Instead of standard ML DPO (which requires simultaneous reference + policy model forward passes — extremely complex in pipeline-parallel), Gas Town uses a practical approach:

```
1. Same task slung to two model versions (or model A vs model B)
2. Both produce Dolt branches with their work
3. User/rig evaluates and picks the better outcome
4. Winner = "chosen", Loser = "rejected"
5. Preference pairs accumulated over time
6. Periodically retrain with collected preferences
```

**Advantages over standard DPO:**
- No reference model forward pass needed (no 2x pipeline overhead)
- Preferences collected at task level (what matters for Gas Town), not token level
- Natural integration with existing sling/polecat workflow
- Uses Dolt branching which already exists

**Failure modes captured for rejected signal:** timeout, wrong branch, broke tests, ignored conventions, memory crash, skipped hooks, wrong dependency direction.

### Model Staleness Detection

- Track Gas Town convention changes (CLAUDE.md updates, new formulas, config changes)
- Flag when training corpus diverges from current conventions
- Auto-trigger retrain pipeline when drift exceeds threshold
- RAG layer covers gap between training snapshots

### Eval Flow

```
New model trained --> eval suite runs --> score >= threshold?
                                           |-- yes --> register, available for deployment
                                           |-- no  --> flag for investigation, block deployment
```

## Gas Town Integration

### Component Integration Map

| Component | Integration | Action |
|-----------|-------------|--------|
| Town settings | New "node0" provider type, role-to-model mapping | MODIFY |
| Agent lifecycle | node0-inference binary as agent command, same tmux management | NEW |
| gt prime | Inject model-appropriate context (fine-tuned needs less priming) | MODIFY |
| Formulas | train-role-model, corpus-collect, eval-suite formulas | NEW |
| Bead system | Training jobs tracked as beads, corpus linked to source sessions | NEW |
| Sling | Sling to node0 agents, task-specific adapter selected by bead type | MODIFY |
| Monitor | Training progress in MonitorWorker, performance dashboards | MODIFY |
| Registry | gt model list/show/deploy/rollback, UI browser | NEW |
| Corpus collection | Hook into session lifecycle, auto-export on close, scrubbing | NEW |

### Day-to-Day Impact

**Before:**
- `gt sling` spawns claude polecat
- Claude needs full `gt prime` (~15K token context injection)
- $X per session (API costs)
- Session logs discarded on close

**After:**
- `gt sling` spawns node0-polecat (or claude, configurable per role)
- Fine-tuned model already knows conventions, minimal context needed
- RAG injects only dynamic state (current beads, recent changes)
- Self-hosted inference (compute cost only, no per-token API pricing)
- Session auto-exported to corpus on close, scrubbed and tagged by role/task

## Inference Architecture

Node0's forward pass through pipeline stages IS inference. The gRPC infrastructure that moves activations between stages during training is the same infrastructure needed for inference. What must be ADDED is the autoregressive generation loop.

**What exists (reusable for inference):**
- gRPC pipeline forwarding between stages (head → body → tail)
- Hivemind DHT for peer discovery
- Pipeline stage activation passing

**What must be built:**
- Autoregressive generation loop (decode one token → feed back → repeat)
- Prompt API (prompt-in/response-out endpoint for tmux agent usage)
- Sampling logic (temperature, top-p, top-k)
- Streaming response back to agent session
- KV-cache management across pipeline stages

```
Agent tmux session
  --> node0-inference client (stdin/stdout, tmux-compatible)
  --> Prompt tokenized, fed into head stage
  --> Activations flow through all 32 pipeline stages via gRPC
  --> Tail stage produces next-token logits
  --> Sampling selects token
  --> Token fed back into head stage (autoregressive loop)
  --> Response streamed back to agent session
```

This is a moderate extension of node0's existing pipeline forwarding — not a separate inference system. The `node0` provider in town settings connects to the DHT to route inference requests to available pipeline peers.

## RAG Architecture

Embedded vector DB (ChromaDB) complements the fine-tuned model with dynamic state:

```
Fine-tuned model knows:              RAG injects:
- Gas Town conventions                - Current bead state (open/blocked/ready)
- gt/bd command patterns              - Recent config changes
- Formula structure                   - Active rig assignments
- Sling lifecycle                     - Latest CLAUDE.md updates since training
- Bead dependency rules               - Session context from related beads
```

**Indexing**: ChromaDB indexes beads, configs, recent sessions, and formula definitions. Updated on every `bd sync` and config change.

**Retrieval at inference**: When the node0 agent receives a prompt, relevant context is retrieved from ChromaDB and prepended to the prompt — similar to how `gt prime` works today, but smaller and more targeted.

**Benefit**: The model's fine-tuned knowledge handles ~80% of context. RAG fills the remaining ~20% that changes between training runs. This dramatically reduces the context injection compared to current `gt prime` (~15K tokens → estimated ~2-3K tokens of RAG context).

## Corpus Collection Mechanism

**Primary: Bead-linked export**

When a polecat bead closes, its session transcript is automatically attached to the bead:

```
Polecat session completes
  --> gt hook (session_shutdown) fires
  --> Transcript extracted from tmux session
  --> Scrubbed (automated PII removal)
  --> Tagged with: role, rig, bead ID, task type, outcome (success/fail)
  --> Stored as corpus entry linked to the bead
  --> Available for next training run
```

**Supplementary: Wisp and gap coverage**

Wisps and non-bead sessions (mayor interactions, ad-hoc crew work) are captured via:
- Periodic batch export scanning tmux session logs
- Manual `gt corpus add` command for curated examples
- Formula-triggered export for spec/brainstorm sessions

**Cross-rig sharing**: Opt-in per rig. Each rig must explicitly enable corpus sharing via rig settings. Default is rig-local only.

**Conflict resolution**: When corpus contains contradictory entries (e.g., old vs new CLAUDE.md conventions), recency wins. Most recent version takes precedence; older conflicting entries are auto-excluded or down-weighted during training. **Caveat**: if a bad convention is introduced and later reverted, the revert must be explicitly captured as a new corpus entry — otherwise "recency wins" would preserve the bad convention. Corpus pipeline should flag reversions (detecting when a CLAUDE.md change undoes a recent change) for manual review.

## Open Questions (Carry to Implementation)

1. **Rig structure (Q52)**: Where does training infra vs corpus collection live? Dedicated node0 rig, embedded in each rig, or hybrid? Needs investigation.
2. **Exact node0 code changes**: Specific modifications needed in node0 for data loading, checkpoint export, LoRA support, and inference serving.
3. **LoRA framework**: PEFT, Unsloth, or other? Benchmarking decision deferred to implementation.
4. **Tokenizer**: Efficiency for Gas Town-specific vocabulary (gt commands, bead IDs, formula syntax). Custom tokens or rely on base tokenizer?
5. **ChromaDB integration**: Where does the vector DB live? Per-rig or centralized? How does it integrate with the inference pipeline?

## Out of Scope (V2+)

- Federated Gas Town (cross-operator model sharing network)
- Right-to-be-forgotten / data removal after training
- Mayor role replacement (last in progressive rollout)
- Gemini/GPT base model experiments (currently blocked by auth issues)
- Custom UI for training monitoring (use existing MonitorWorker + CLI)

## Decision Log

All decisions made during the brainstorm session on 2026-02-21:

| # | Question | Decision | Notes |
|---|----------|----------|-------|
| Q1 | Agent definition | Full autonomy — convention knowledge + workflow execution | |
| Q2 | Base model | Model-agnostic, try multiple | Infrastructure supports experimentation |
| Q3 | Motivation/baseline | All: prime bloat, convention errors, no baseline, cost | |
| Q5 | Data format | Mixed formats | JSON, markdown, conversation logs |
| Q6 | Data volume | 1K-10K examples | |
| Q7 | Success tasks | All categories + planning | Bead mgmt, git, conventions, code, planning |
| Q8 | Integration path | Progressive replacement | Deacon/witness first, then polecats/crew |
| Q9 | PII handling | Automated scrubbing | No manual review |
| Q11 | Training approach | Node0 distributed training | Node0 IS the infrastructure |
| Q13 | Freshness | Retrain ASAP + RAG complement | |
| Q16 | Role models | Role-specific | Separate models per Gas Town role |
| Q18 | Provider type | New "node0" provider in town settings | |
| Q27 | Eval approach | Automated task suite | Gates deployment |
| Q30 | Data policy | No removal after training | |
| Q33 | Preference learning | DPO in V1 | Success/failure polecat sessions |
| Q35 | Corpus scope | Our rigs first, extensible later | Federated out of scope |
| Q42 | GPU-less participation | Yes, with incentive model | Data contribution grants model access |
| Q44 | Rollback | Versioned model registry | |
| Q52 | Rig structure | Needs investigation | Deferred to implementation |
| Q57 | Config management | Gas Town formula templates | |
| Q64 | Task-specific tuning | Yes, LoRA adapters in V1 | Stackable on base role models |
| Q70 | Model registry | Full UI in V1 | CLI + browsable UI |
| Q75 | Planning assistance | Core V1 use case | Bead hierarchies, effort estimation, deps |

### Plan Review Additions (2026-02-22)

| # | Question | Decision | Notes |
|---|----------|----------|-------|
| Q70-revised | Model Registry UI scope | CLI-only in V1; browsable UI deferred to V2 | Original Q70 decision ("Full UI in V1") revised during plan review. CLI commands (`gt model list/show/deploy/rollback/prune`) deliver the core value. Browsable UI adds significant scope with low immediate value — deferred. |

### Spec Review Additions (2026-02-21)

| # | Question | Decision | Notes |
|---|----------|----------|-------|
| - | Eval thresholds | Deferred to implementation | Pick empirically after first training run |
| - | Inference architecture | Decentralized via Hivemind/node0 | Same infra for training AND inference |
| - | RAG system | Embedded vector DB (ChromaDB) | Indexes beads, configs, sessions |
| - | Corpus export mechanism | Bead-linked export (primary) | Supplemented by batch export for wisps/gaps |
| - | Contradictory corpus handling | Recency wins | Older conflicting entries auto-excluded |
| - | Production validation | Progressive rollout IS the validation | No separate shadow/canary needed |
| - | Cross-rig data sharing | Opt-in per rig | Default is rig-local only |
| - | Safety criteria | Standard Gas Town guardrails | GUPP violations, MR submission, gt conventions |

## Spec Review

**Reviewed:** 2026-02-21
**Gaps identified:** 9 (5 critical, 4 important)
**Gaps resolved:** 9

### Clarifications Added

| Topic | Clarification |
|-------|---------------|
| Eval thresholds | Deferred — pick empirically after first training run |
| Inference | Decentralized via Hivemind/node0, not a local binary |
| RAG | ChromaDB embedded vector DB, indexes beads/configs/sessions |
| Corpus export | Bead-linked on close (primary), batch export for wisps/gaps |
| Conflicts | Recency wins for contradictory corpus entries |
| Validation | Progressive rollout (deacon → witness → polecat) IS the validation |
| Data sharing | Opt-in per rig, default rig-local only |
| Safety | Standard Gas Town guardrails (GUPP, MRs, conventions) via eval |

### Deferred Items

| Item | Rationale | Revisit When |
|------|-----------|--------------|
| Eval score thresholds | Need real training data first | After first training run |
| Rig structure (Q52) | Multiple viable options | Implementation planning |
| LoRA framework selection | Benchmarking needed | Implementation |
| Tokenizer customization | Depends on base model | After base model selected |
| ChromaDB integration details | Depends on inference architecture | Implementation |

## Multi-Model Review

**Reviewed:** 2026-02-21 | **Models:** Opus 4.6, Kimi K2 (omp) | **Full report:** `spec-review.md`

19 issues identified (5 critical, 5 high, 6 medium, 3 low). All 19 addressed in this spec revision.

### Critical Issues Resolved

| # | Issue | Resolution |
|---|-------|------------|
| 1 | Hivemind doesn't do inference | Rewritten: node0's forward pass IS inference. Added "What must be built" list (generation loop, prompt API, sampling, KV-cache). |
| 2 | Data loading is an epic, not a bullet | Elevated to "#1 EPIC — largest engineering task" with 6 sub-tasks. |
| 3 | Pipeline-parallel breaks fine-tuning assumptions | Documented per-stage LoRA, tail-only loss, 32-stage checkpoint gathering throughout spec. |
| 4 | Checkpoint export architecturally complex | Detailed as must-build #2 with pipeline gathering + serialization requirements. |
| 5 | "Use as-is" components need modification | Split into 3 tiers: use as-is (3), needs modification (6 with details), must build (8). |

### High Issues Resolved

| # | Issue | Resolution |
|---|-------|------------|
| 6 | DPO requires new architecture | Redesigned as "DPO via Dolt branching" — task-level preference collection using existing infrastructure. |
| 7 | ChromaDB/RAG has no integration point | RAG Architecture section added with proper integration via inference pipeline. |
| 8 | PII scrubbing insufficient | Upgraded to multi-layer pipeline: secret scanning + regex + confidence flagging + periodic audit. |
| 9 | Model poisoning not addressed | Added "Corpus data validation and quality gates" as must-build #8. |
| 10 | Model registry is entire subsystem | Added storage design note and flagged as its own epic for implementation. |

### Medium/Low Issues Resolved

| # | Issue | Resolution |
|---|-------|------------|
| 11 | Model-agnostic contradicted | Scoped V1 to LLaMA family, documented non-trivial effort for new architectures. |
| 12 | Corpus volume per-role | Added note: 250-2500 per role initially, grows over time. |
| 13-14 | Eval methodology undefined | Added scoring methodology (binary/numeric per-task, category percentages, weighted composite). |
| 15 | Corpus partitioning confused | Clarified "PARTITION (data)" label — data by role/task, not pipeline stages. |
| 16 | Vocab size mismatch | Added note about OPT vs LLaMA vocab_size fix needed. |
| 17 | Formula syntax aspirational | Added note that command mappings are TBD pending implementation. |
| 18 | Bead lifecycle oversimplistic | Expanded to 7 states: corpus-validating → ready → training → syncing → checkpointing → evaluating → done/failed/degraded. |
| 19 | Recency regression risk | Added caveat about reverted conventions needing explicit capture + flagging. |

### Key Design Decisions from Review

1. **Inference = extended forward pass**: Both reviewers said Hivemind can't do inference. User correctly identified that the forward pass IS inference — just needs generation loop added. Moderate addition, not a rewrite.
2. **DPO via Dolt branching**: User proposed task-level preference collection through existing Dolt branching. Sidesteps the need for simultaneous reference+policy model forward passes entirely.
3. **Keep distributed pipeline-parallel**: User chose to keep the distributed architecture rather than gather to single node for fine-tuning.

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
+-----------------------------------------------------+
|                Gas Town Orchestration                 |
|  +----------+  +----------+  +-------------------+  |
|  | Corpus    |  | Training |  | Model Registry    |  |
|  | Pipeline  |  | Formulas |  | (UI + CLI + API)  |  |
|  +-----+----+  +-----+----+  +---------+---------+  |
|        |              |                 |            |
|        v              v                 v            |
|  +----------------------------------------------+   |
|  |             node0 Training Layer              |   |
|  |  Hivemind DHT | Gradient Avg | Pipeline Par.  |   |
|  +----------------------------------------------+   |
|        |                              |              |
|        v                              v              |
|  +----------+                +----------------+      |
|  | Role-     |                | Task-specific  |      |
|  | specific  |                | LoRA adapters  |      |
|  | models    |                | (stackable)    |      |
|  +----------+                +----------------+      |
|        |                                             |
|        v                                             |
|  +----------------------------------------------+   |
|  |      New Provider: "node0" in town settings   |   |
|  |  Replaces: deacon > witness > polecat > crew  |   |
|  +----------------------------------------------+   |
+-----------------------------------------------------+
```

### Key Architectural Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Base model | Model-agnostic | Try multiple (LLaMA, Qwen, etc.), compare, pick winners per role |
| Model specialization | Role-specific | Separate models for mayor, polecat, witness, deacon — different training data and behavior |
| Task specialization | LoRA adapters | Stackable task-specific adapters (bead-creation, code-review, git-workflow, planning) switchable at inference |
| Training approach | Distributed via node0 | Node0 IS the distributed training infrastructure — Hivemind DHT, gradient averaging, pipeline parallelism |
| Training hardware | External GPU infrastructure | Not on the 10GB dev box. Proper GPU nodes for training |
| Inference | Decentralized via Hivemind/node0 | Same infrastructure serves both training AND inference. Not a local binary. |
| RAG system | Embedded vector DB (ChromaDB) | Indexes beads, configs, recent sessions. Injected into context at inference time. |
| Corpus export | Bead-linked export (primary) | Session transcripts attached to bead on close. Supplemented by other methods for wisps/gaps. |
| Conflict resolution | Recency wins | Most recent version of any artifact takes precedence in corpus. Older entries auto-excluded. |
| Validation strategy | Progressive rollout IS the validation | No separate shadow/canary. Eval suite + progressive replacement (deacon → witness → polecat) is sufficient. |
| Cross-rig data | Opt-in per rig | Each rig explicitly enables corpus sharing. Default is rig-local only. |
| Safety | Standard Gas Town guardrails | GUPP violations, failing to submit MRs, standard gt conventions. Enforced via eval suite. |
| Eval thresholds | Deferred to implementation | Pick thresholds empirically after first training run. Can't know what's reasonable until real numbers exist. |
| Preference learning | DPO in V1 | Successful vs failed polecat sessions as preference pairs |
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
1. COLLECT              2. SCRUB                3. FORMAT               4. PARTITION
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

- **Volume**: 1K-10K examples initially
- **Formats**: Mixed — structured JSON, markdown, conversation logs
- **Scrubbing**: Fully automated pipeline (regex + pattern matching for credentials, API keys, tokens, IP addresses). No manual review step.
- **Data retention**: No post-training removal. No right-to-be-forgotten requirement.

### Extensibility Path

1. **V1**: Collect from our own Gas Town rigs
2. **V1+**: Extend so other operators can contribute data (incentive model: contribute data, get model access even without GPU)
3. **Future**: Federated Gas Town (out of V1 scope)

## Training Infrastructure

### node0: What Exists vs What We Build

**Exists in node0 (use as-is):**
- Hivemind DHT peer discovery
- Decentralized gradient averaging
- PowerSGD compression
- Pipeline parallelism
- PluralisAuthorizer
- YAML config system
- AutoStepOptimizer
- MonitorWorker metrics

**Must be built:**
- Data loading pipeline (node0 has NO data loading currently)
- Checkpoint export (safetensors + GGUF formats)
- LoRA adapter support
- Corpus partitioning per pipeline stage
- Training job lifecycle mapped to beads
- Pause/resume capability
- DPO training loop
- Multi-model experiment tracking

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

Training runs are configured through Gas Town formulas — composable, versionable, shareable:

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

[steps.train]
description = "Launch node0 distributed training job"

[steps.eval]
description = "Run automated task suite against checkpoint"

[steps.export]
description = "Export to safetensors + GGUF, register in model registry"
```

### Hardware

Training runs on proper GPU infrastructure, not the 10GB dev box. GPU-less operators can contribute corpus data without running training themselves (incentive model: data contribution grants model access).

## Model Registry & Deployment

### Model Registry (Full UI in V1)

A browsable registry showing all trained models, their lineage, eval scores, and deployment status across rigs.

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

| Task Category | Example Tasks | Pass Criteria |
|---------------|---------------|---------------|
| Bead management | Create issue from description; close with reason; build epic hierarchy | Correct fields, deps, status transitions |
| Git workflow | Branch, commit, push; generate PR description; resolve merge conflict | No --force, hooks pass, clean merge |
| Convention adherence | Use gt commands; follow CLAUDE.md; respect resource constraints | No pkill, role-appropriate behavior |
| Planning | Generate bead hierarchy; estimate complexity; suggest deps | Deps enable parallelism, no cycles |
| Code execution | Implement from bead description; fix bug; refactor | Tests pass, no regressions |

### DPO Training Signal

| Source | Signal |
|--------|--------|
| Successful polecat session | Chosen response (model should emulate) |
| Failed polecat session | Rejected response (model should avoid) |

Failure modes captured: timeout, wrong branch, broke tests, ignored conventions, memory crash, skipped hooks, wrong dependency direction.

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

Node0 serves both training AND inference via Hivemind's decentralized infrastructure. The same peer network used for gradient averaging during training serves model inference in production.

```
Agent tmux session
  --> node0-inference client (stdin/stdout, tmux-compatible)
  --> Hivemind DHT discovers inference peers
  --> Distributed inference across pipeline stages
  --> Response streamed back to agent session
```

This means inference is not a local binary running on the 10GB dev box — it's a distributed operation across GPU nodes in the Hivemind network. The `node0` provider in town settings connects to the DHT to route inference requests.

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

**Conflict resolution**: When corpus contains contradictory entries (e.g., old vs new CLAUDE.md conventions), recency wins. Most recent version takes precedence; older conflicting entries are auto-excluded or down-weighted during training.

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

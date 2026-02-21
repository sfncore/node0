# Codebase Context: gastown-finetune

## App Structure

**Repository Root:** `/home/ubuntu/gt/node0/mayor/rig/`

**Key Directories:**
- `src/node0/` — Main Python package
  - `configs/` — YAML configuration files (e.g., `llama_8B_C.yaml`)
  - `models/` — Model definitions
    - `llama/` — LLaMA model layers (`arguments.py`, `layers.py`)
    - `arguments.py` — Base `ModelArguments` Pydantic model
    - `lr_schedule.py` — Learning rate schedulers
  - `security/` — Authorization and validation
    - `authorization.py` — PluralisAuthorizer (HF token validation)
    - `validation.py` — Worker metrics validation
  - `server/` — Distributed training core
    - `node0_server.py` — Main Server class (extends Hivemind Server)
    - `HM_averager.py` — DecentralizedAverager (gradient/state averaging)
    - `HM_gradient_averager.py` — GradientAverager (accumulation + allreduce)
    - `HM_state_averager.py` — State averaging for model weights
    - `optim.py` — AutoStepOptimizer (ensures periodic step() calls)
    - `matchmaking.py` — Peer group formation
    - `power_sgd_averager.py` — PowerSGD compression for gradients
  - `utils/` — Utilities
    - `monitor.py` — Logging and metrics (BaseMonitor, MonitorWorker)
    - `node_info.py` — Hardware/network profiling (speedtest, GPU info, psutil)
    - `common.py` — Helpers (build_cls, infer_expert_params, load_ss_components)
- `generate_script.py` — Script generator (Docker or source setup)
- `run_server.py` — Main entry point
- `Dockerfile` — Docker image (NVIDIA CUDA 12.1, Conda, PyTorch 2.7)
- `pyproject.toml` — Project metadata and dependencies

## Node0's Purpose and Architecture

**High-Level Goal:** Decentralized collaborative pretraining (Protocol Learning)

- **Distributed Model Parallelism:** Each node holds a portion of the computation graph (pipeline stages: head, body, tail)
- **P2P Communication:** Uses Hivemind's DHT + P2P for peer discovery and gradient/state averaging
- **Commodity Hardware:** Supports 16GB+ GPUs; can join/leave dynamically
- **Asynchronous Averaging:** Nodes communicate over internet via communication-efficient protocols (PowerSGD)
- **Pipeline Parallelism:** Model split into stages; each node specializes in one stage

**Distributed Coordination:**
1. PluralisAuthorizer validates HuggingFace token + node integrity
2. DHT (Distributed Hash Table) for peer discovery (bootstrap with initial peers from `run.json`)
3. Matchmaking: nodes form averaging groups based on DHT keys
4. DecentralizedAverager aggregates gradients/states across groups via allreduce
5. Experts exchange activations/gradients between pipeline stages via gRPC

## Existing Features

**Training Loop:**
- Model split into pipeline stages (e.g., LLaMA 8B as single-layer expert per stage)
- Each expert runs forward/backward pass on local batches
- Gradients accumulated locally, then averaged with peers via DecentralizedAverager
- AutoStepOptimizer triggers optimizer.step() at fixed intervals
- State (model weights) periodically averaged across nodes

**Configuration System (YAML-based):**
- Model config: hidden_dim, n_heads, num_hidden_layers, compression rate, RoPE theta
- Optimizer: learning rate, weight decay
- Gradient averaging: PowerSGD rank, sparse averaging threshold
- Training schedule: warmup steps, total steps, batch sizes
- Network: matchmaking time, averaging timeout, request timeout

**Monitoring:** Prometheus metrics, file logging, network throughput profiling, GPU/memory monitoring

**Security:** RSA private key for P2P identity, HF token auth, integrity verification

**Compression:** PowerSGD (rank-64 default), subspace compression, configurable sparse averaging (5%)

## Tech Stack

- **PyTorch 2.7.0** — Deep learning
- **Hivemind** (specific commit pinned) — Decentralized averaging, DHT, P2P, MOE server
- **gRPC + protobuf** — Inter-node communication
- **Pydantic>=2.0.0** — Config validation
- **Python 3.11** (strict requirement)
- **CUDA 12.1.1 + cuDNN8** — Docker image
- **Docker** (preferred) or conda from source
- Apache 2.0 licensed

## Integration Points (Gas Town ↔ Node0)

1. **Data Injection:** Node0 does NOT currently include data loading — gradients are accumulated by external training loops. Integration opportunity: hook into GradientAverager with Gas Town's dataset/corpus.

2. **Configuration:** YAML configs define training params. Gas Town formulas could generate/modify these configs. Bead descriptions should specify training corpus path, fine-tuning task, target stage.

3. **Model Variants:** Currently single LLaMA 8B config. Extensible via `custom_module_path`. ModelArguments is Pydantic-based.

4. **Authorization:** HF token required (auth server: `https://auth.pluralis.ai`). Gas Town could extend MonitorWorker to log corpus usage, fine-tuning metrics.

5. **Checkpoints:** No explicit checkpoint saving (relies on Hivemind state averaging). Integration should add checkpoint export. Gas Town could manage checkpoint versioning.

6. **Networking:** Requires port forwarding (default 49200). Initial peers hardcoded in `run.json` (4 seed nodes).

## Key Files to Reference

| File | Purpose |
|------|---------|
| `src/node0/run_server.py` | Entry point: auth → node_info → DHT → server creation |
| `src/node0/configs/llama_8B_C.yaml` | Model, optimizer, scheduler, training hyperparameters |
| `src/node0/server/optim.py` | AutoStepOptimizer: training step orchestration |
| `src/node0/server/node0_server.py` | Node0Server.create() — distributed server instantiation |
| `src/node0/models/llama/layers.py` | RMSNorm, RoPE, attention — model architecture |
| `src/node0/security/authorization.py` | PluralisAuthorizer — token validation + peer registration |
| `src/node0/utils/common.py` | infer_expert_params(), load_ss_components() |
| `src/node0/utils/monitor.py` | MonitorWorker, logging pipeline |
| `pyproject.toml` | Dependencies and project metadata |
| `Dockerfile` | Container build (CUDA 12.1, Conda, PyTorch 2.7) |

## Critical Design Consideration

Node0 is a **pretraining infrastructure**, not a fine-tuning framework. Integration requires:
- Wrapping the training loop (data feeding, checkpoint export)
- Gas Town's role: corpus management, bead workflow, distributed orchestration
- Key challenge: distributed gradient aggregation is tightly coupled; corpus needs pre-partitioning or careful streaming
- Opportunity: extends decentralized training to downstream tasks (instruction-tuning, domain adaptation)

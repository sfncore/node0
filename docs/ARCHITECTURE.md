# Node0 Training Pipeline Architecture

Node0 is a decentralized collaborative training system for LLaMA models built on top of the [Hivemind](https://github.com/learning-at-home/hivemind) framework. Worker nodes join a DHT-based swarm, each running a subset of the model's layers (pipeline stage), and collaboratively train via gradient and state averaging.

## Table of Contents

1. [Training Loop Setup](#1-training-loop-setup)
2. [Pipeline Stages](#2-pipeline-stages)
3. [Optimizer Wrappers](#3-optimizer-wrappers)
4. [DHT Configuration](#4-dht-configuration)
5. [Config System](#5-config-system)
6. [Auth Integration](#6-auth-integration)
7. [Data Flow](#7-data-flow)

---

## 1. Training Loop Setup

### Entry Point

The server starts from `src/node0/run_server.py:114` (`main()`). The startup sequence is:

1. **Parse args** (`run_server.py:120`): CLI arguments merged with a YAML run config (`run_server.py:106-111`)
2. **Logging/monitoring** (`run_server.py:127-138`): `MonitorWorker` starts a background thread that parses log lines for FLOP counting, all-reduce failure detection, and metrics reporting to DHT
3. **Node info** (`run_server.py:151`): `get_node_info()` collects GPU, RAM, and network bandwidth
4. **Authorization** (`run_server.py:155-166`): `authorize_with_pluralis()` authenticates with the Pluralis auth server, which assigns a `pipeline_stage` (e.g., `head-0`, `body-3`, `tail-0`)
5. **Expert params** (`run_server.py:171-173`): `infer_expert_params()` (`utils/common.py:64-99`) maps the pipeline stage string to an expert UID pattern, expert class name, and stage index
6. **DHT peer routing** (`run_server.py:183-186`): `update_initial_peers()` (`utils/dht_partition.py:62-101`) adjusts bootstrap peer ports based on the assigned stage partition
7. **Build components** (`run_server.py:204-261`): Model args, optimizer, gradient averager factory, and compression settings are constructed using `build_cls()` (`utils/common.py:31-61`)
8. **Create server** (`run_server.py:279-294`): `Node0Server.create()` instantiates the full collaborative training server

### Server Creation

`Node0Server` (`server/node0_server.py:57`) extends Hivemind's `Server`. The `create()` classmethod (`node0_server.py:64-321`) orchestrates:

1. **DHT connection** (`node0_server.py:162`): Creates a DHT node with initial bootstrap peers
2. **UID generation** (`node0_server.py:181-187`): Generates expert UIDs from the pattern (e.g., `head.0.[32:1024]`) using Hivemind's `_generate_uids()`
3. **Parameter store lookup** (`node0_server.py:190-200`): `get_parameter_store()` (`utils/get_parameters.py:30-116`) fetches runtime training hyperparameters (batch size, scheduler, timeouts) from the DHT, published by a coordinator
4. **Expert initialization** (`node0_server.py:224-278`):
   - Creates the model block (Head/Body/Tail expert) via `name_to_block[expert_cls](model_conf)`
   - Configures parameter groups with weight decay exclusions
   - Optionally loads subspace compression components from remote URL
   - Wraps the model in `ModuleCollab` backend
5. **Collaborative optimizer** (`node0_server.py:281-305`): Creates `AutoStepOptimizer` with gradient and state averagers, then loads initial state from peers (`node0_server.py:306`)
6. **Server start** (`node0_server.py:310-321`): Returns `Node0Server` instance which inherits Hivemind's `Server.join()` event loop

### The Runtime Loop

Hivemind's `Server` class runs a `Runtime` that processes incoming forward/backward requests from pipeline peers. The `Runtime`:

- Accepts batched gRPC requests from other pipeline stages
- Routes forward calls through the expert module
- Routes backward calls through `ModuleCollab.backward()` (`server/module_collab.py:29-80`)
- After each backward, calls `on_backward()` (`module_collab.py:82-91`) which triggers `optimizer.step(batch_size=batch_size)`

There is **no local dataloader**. Batches arrive externally via Hivemind's pipeline protocol.

---

## 2. Pipeline Stages

### Three-Stage Pipeline

The model is split into three expert types, each registered via Hivemind's `@register_expert_class` decorator:

| Stage | Class | File:Line | Input | Output |
|-------|-------|-----------|-------|--------|
| **Head** | `HeadExpert` | `models/llama/layers.py:564` | Token IDs `(B, S)` | Hidden states `(B, S, H)` or compressed |
| **Body** | `BodyExpert` | `models/llama/layers.py:603` | Hidden states `(B, S, H)` | Hidden states `(B, S, H)` |
| **Tail** | `TailExpert` | `models/llama/layers.py:628` | Hidden states + labels | Cross-entropy loss |

### Expert Architecture

All three inherit from `BaseExpert` (`models/llama/layers.py:451`), which provides:

- **Transformer blocks**: `TransformerBlock` (`layers.py:341`) with `Attention` (`layers.py:172`) + `FeedForward` (`layers.py:297`)
- **RoPE embeddings**: Precomputed frequency tensors (`layers.py:82-102`)
- **Subspace compression** (optional): compress/decompress between stages using a learned `rcv` matrix and fixed token embeddings (`layers.py:502-534`)
- **Weight initialization**: Depth-dependent init (`layers.py:390-395`) with truncated normal

### Stage-Specific Behavior

**HeadExpert** (`layers.py:565-600`):
- Has a learnable `tok_embeddings` layer
- Forward: embeds tokens, runs transformer layers, optionally compresses output
- If compression enabled: appends token indices and compresses via `compress_output()`

**BodyExpert** (`layers.py:603-625`):
- No embedding layer; receives hidden states directly
- If compression enabled: decompresses input, runs layers, re-compresses output

**TailExpert** (`layers.py:628-675`):
- Has a final `norm` layer and `output` projection to vocab size
- Computes cross-entropy loss directly (`layers.py:673`)
- The last 2 layers skip compression (`layers.py:636`)
- Attention projection is disabled (`layers.py:631`)

### Expert UID Pattern

UIDs follow the format `{stage}{idx}.0.{uid}`, e.g., `head.0.42`, `body3.0.55`, `tail.0.100`. The stage name determines which DHT prefix to use for gradient/state averaging. UID allocation starts at index 32 with a max of 1024 (`utils/common.py:88`).

### Subspace Compression

When `use_compression=True` (`models/arguments.py:39`):
- A low-rank `rcv` matrix (`hidden_dim x compression_length`) compresses hidden states between stages
- Fixed (frozen) token embeddings provide a bias term for reconstruction
- Compression components are loaded from a remote URL (`utils/common.py:102-124`)
- After loading, `ss_regularize()` projects attention output and FFN weights into the subspace (`layers.py:544-553`)

---

## 3. Optimizer Wrappers

### AutoStepOptimizer

`AutoStepOptimizer` (`server/optim.py:36`) extends Hivemind's `Optimizer` with an auto-step mechanism:

- **Monitor thread** (`optim.py:141-153`): A daemon thread checks every 1s if `step()` hasn't been called within `auto_step_time` (default 3s). If idle too long, triggers `_auto_step()` (`optim.py:155-174`) to keep the collaborative optimizer progressing
- **Step logic** (`optim.py:176-190`): External calls from `ModuleCollab.on_backward()` acquire a step lock, resync state, then call `_step()`
- **Resync** (`optim.py:72-82`): Before each step, checks if the peer is behind the global epoch. If too far behind (`max_allowed_stale`), reloads state from peers. If slightly behind, catches up by advancing `local_epoch`
- **Gradient scheduling** (`optim.py:222-230`): When the next epoch is approaching, pre-schedules a gradient averaging round with estimated timing

### The `_step()` Method

`_step()` (`optim.py:232-280`) is the core training step:

1. Applies any delayed state averaging updates
2. Accumulates gradients via `_check_and_accumulate_gradients()` (`optim.py:211-220`)
3. Schedules gradient averaging when ready
4. When `ready_to_update_epoch` and `in_update`: acquires the optimizer lock and calls `_update_global_epoch()` (inherited from Hivemind `Optimizer`)
5. If compression is enabled, re-regularizes weights via `ss_regularize()`

### State Loading

`load_state_from_peers()` (`optim.py:282-353`):
- Waits until the current gradient accumulation round is nearly complete (<10% of target batch size)
- Downloads state (model params, optimizer state, scheduler) from the highest-priority peer
- Optionally waits for the next round to start before joining, avoiding mid-round disruptions

### Gradient Averagers

**Inheritance chain:**
```
DecentralizedAverager (server/HM_averager.py:65)
  -> GradientAverager (server/HM_gradient_averager.py:32)
    -> PowerSGDGradientAverager (server/power_sgd_averager.py:48)
```

**DecentralizedAverager** (`HM_averager.py:65`):
- Runs as a `mp.Process` with uvloop event loop
- Manages matchmaking, all-reduce groups, and state download/upload
- Uses `AllReduceRunner` (`server/ar_runner.py:52`) for butterfly all-reduce with tensor partitioning

**GradientAverager** (`HM_gradient_averager.py:32`):
- Manages three buffer sets: model `.grad` buffers, local accumulators, averaged gradients
- `accumulate_grads_()` (`HM_gradient_averager.py:145`): Adds gradients to accumulators with batch-size scaling
- `schedule_step()` / `step()`: Pre-schedule matchmaking, then trigger all-reduce

**PowerSGDGradientAverager** (`power_sgd_averager.py:48`):
- Implements PowerSGD low-rank compression (rank `r`) for gradient communication
- Approximates gradient matrices `(m,n)` as products `(m,r) x (r,n)`
- Two-phase all-reduce: Phase P averages the P matrices, Phase Q averages Q matrices + uncompressed tensors
- Maintains error feedback buffers (`_ms`) to preserve compression residuals across steps
- QR orthogonalization on P matrices between phases (`power_sgd_averager.py:216`)
- Mac variant (`power_sgd_averager_mac.py:161`) uses chunked matrix operations to avoid memory issues

### State Averager

**TrainingStateAverager** (`server/HM_state_averager.py:50`):
- Manages model parameters, optimizer statistics, and extra tensors (e.g., batchnorm stats)
- Supports offloaded optimizer (gradients copied to CPU optimizer)
- Delta rule averaging: computes `state += averaged - pre_averaged` to allow concurrent local updates
- Gradient clipping with stage-dependent thresholds: 0.1768 for non-tail, 0.8839 for tail (`HM_state_averager.py:150-153`)

**Node0's TrainingStateAverager wrapper** (`server/state_averager_wrap.py:77`):
- Adds sparse averaging via `PartitionedIndexSelector` (`state_averager_wrap.py:57`): random partitions of parameters, with only one partition averaged per round (controlled by `sparse_avg` config)
- Adds consecutive failure tracking with process termination on too many failures
- NaN detection in downloaded state with immediate termination

### Matchmaking

`Matchmaking` (`server/matchmaking.py:134`):
- Uses a simplified single fixed group (all peers in the same stage)
- `GroupKeyManager` (`matchmaking.py:43`) maintains a fixed DHT key `{prefix}.0b` for the group
- `look_for_group()` (`matchmaking.py:214-266`): Polls DHT for peers, waits up to `min_matchmaking_time`, forms a group when all peers in the `peer_table` have joined
- Background thread (`matchmaking.py:185-208`) maintains a peer table by polling DHT for expert declarations

---

## 4. DHT Configuration

### Bootstrap Nodes

Initial DHT peers are provided via `--initial_peers` CLI argument (multiaddrs). The `run.json` file specifies seed peers:

```json
{
  "seeds": ["/ip4/34.215.69.49/tcp/49200/p2p/Qm..."]
}
```

### DHT Partitioning

The DHT is partitioned across stages to reduce cross-stage communication overhead:

- `partition_array(num_stages, num_dht)` (`utils/dht_partition.py:16-46`): Distributes stages across `num_dht` DHT instances
- `update_initial_peers()` (`dht_partition.py:62-101`): Adjusts bootstrap peer TCP ports by adding an offset based on the stage-to-DHT mapping
- Example: With `num_stages=32` and `num_dht=4`, stages 0-7 use port+0, stages 8-15 use port+1, etc.

### DHT Usage

The DHT stores several types of data:

| Key Pattern | Purpose | File:Line |
|-------------|---------|-----------|
| `{stage}_paramstore` | Runtime training hyperparameters | `utils/get_parameters.py:46` |
| `{prefix}.0b` | Matchmaking group membership | `server/matchmaking.py:65` |
| `{prefix}_grad_averager` | Gradient averager state | `server/optim.py:113` |
| `{prefix}_state_averager` | Training state averager | `server/optim.py:93` |
| `{prefix}.all_averagers` | State download peer priority | `server/HM_averager.py:635` |
| `{exp}_{stage}_worker_metrics` | FLOP/time metrics | `utils/monitor.py:236` |
| `{exp}_{peer}_worker_ports` | Port reachability checks | `utils/monitor.py:265` |
| `{stage}.0.` | Expert declarations (peer table) | `server/matchmaking.py:195` |

### DHT Monitoring

`patch_dht_protocol_logging()` (`utils/dht_monitor.py:30-89`) monkey-patches Hivemind's `DHTProtocol` to count `rpc_store` and `rpc_find` calls, logging stats every 60 seconds.

---

## 5. Config System

### Two-Layer Configuration

Configuration is split into two files:

#### YAML Model Config (e.g., `configs/llama_8B_C.yaml`)

Defines model architecture, optimizer, gradient averager, and training hyperparameters:

```yaml
model_config:
  class_name: node0.models.llama.arguments.LlamaArguments
  init_args:
    hidden_dim: 4096
    n_heads: 32
    n_kv_heads: 8
    use_compression: True
    compression_rate: 100
    max_seq_len: 4096

optim_config:
  class_name: torch.optim.AdamW
  init_args:
    lr: 0.0003
    weight_decay: 0.1

grad_avg_config:
  class_name: node0.server.power_sgd_averager.PowerSGDGradientAverager
  init_args:
    averager_rank: 64
```

Classes are instantiated dynamically via `build_cls()` (`utils/common.py:31-61`) which does `importlib.import_module()` + `getattr()`.

#### run.json

Points to the YAML config and specifies the auth server + bootstrap peers:

```json
{
  "run_config": "llama_8B_C",
  "auth_server": "https://auth.pluralis.ai",
  "seeds": ["/ip4/.../tcp/49200/p2p/Qm..."]
}
```

The `--run_config` CLI argument loads the YAML, then CLI args are merged on top (`run_server.py:106-111`).

### Key Training Parameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `num_stages` | 32 | Total pipeline stages |
| `num_dht` | 5 | Number of DHT partitions (1 coordinator + 4 worker) |
| `averaging_target_batch_size` | 1024 | Samples before gradient averaging |
| `matchmaking_time` | 45s | Time window for peer matching |
| `averaging_timeout` | 120s | All-reduce timeout |
| `sparse_avg` | 0.05 | Fraction of parameters averaged per state round |
| `average_state_every` | 5 | Average state every N epochs |
| `max_allowed_stale` | 5 | Max epochs behind before forced resync |
| `averager_rank` | 64 | PowerSGD compression rank |

### Runtime Parameters from DHT

At startup, the server fetches these parameters from the DHT parameter store (`utils/get_parameters.py:30`):
- `averaging_target_batch_size`, `scheduler`, `num_warmup_steps`, `num_training_steps`
- `averaging_timeout`, `matchmaking_time`, `request_timeout`, `load_state_timeout`

These override the YAML defaults (`node0_server.py:190-200`), allowing a coordinator to update training parameters without restarting workers.

---

## 6. Auth Integration

### Pluralis Authorizer

`PluralisAuthorizer` (`security/authorization.py:48`) extends Hivemind's `TokenAuthorizerBase`:

1. **Key management** (`authorization.py:296-308`): Generates or loads an RSA private key from `private.key`. Derives a `PeerID` from the key
2. **Join experiment** (`authorization.py:102-207`):
   - Sends a PUT request to `{auth_server}/api/join` with:
     - Bearer token, peer ID, public key, device info, bandwidth, integrity hash
   - Receives: pipeline stage assignment, access token, auth server public key, monitor public key
   - On initial join: runs a port reachability test via `TestServer` (`utils/connection_test_server.py`)
3. **Token refresh** (`authorization.py:83-100`): `get_token()` is called by Hivemind when the token expires (checked via `does_token_need_refreshing()` with 1-minute latency buffer)
4. **Integrity check** (`authorization.py:117-123`): Optionally verifies file integrity via `verify_integrity()` (`security/integrity_check.py`)

### Validation

`make_validators()` (`security/validation.py:75-83`) creates four DHT record validators:

1. **MetricSchema** validator: Ensures worker metrics have signed subkeys
2. **PortSchema** validator: Ensures port check records are signed
3. **RunParametersSchema** validator: Validates parameter store entries
4. **RSASignatureValidator**: Cryptographic signature verification on all DHT records

These validators are passed to `Node0Server.create()` as `record_validators` (`run_server.py:288`), ensuring only authenticated peers can publish to the DHT.

### Error Handling

Authorization failures (`NotInAllowlistError`, `BadRequestError`, `IntegrityError`) result in immediate process termination via `os.killpg(os.getpgrp(), signal.SIGTERM)` (`authorization.py:90-100`). The authorizer re-authenticates periodically via the monitor thread (`utils/monitor.py:417-426`).

---

## 7. Data Flow

### No Local Dataloader

Node0 has **no dataloader**. It is purely a model server that processes batches sent by external clients through Hivemind's pipeline protocol.

### Pipeline Data Flow

```
Client/Trainer
    |
    v
[Head Expert] -- token IDs (B, S) --> hidden states (B, S, H)
    |
    | (optionally compressed via subspace)
    v
[Body Expert 0] -- hidden states --> hidden states
    |
    v
[Body Expert 1] -- hidden states --> hidden states
    |
    ...
    v
[Body Expert N] -- hidden states --> hidden states
    |
    | (decompressed)
    v
[Tail Expert] -- hidden states + labels --> loss
    |
    v
  Backpropagation (reverse order)
```

### Request Processing

1. **Hivemind Runtime** batches incoming gRPC requests (configurable `min_batch_size`/`max_batch_size`)
2. **Forward pass**: Runtime calls the expert's `forward()` method with batched inputs
3. **Backward pass**: `ModuleCollab.backward()` (`module_collab.py:29-80`):
   - Detaches inputs, re-enables gradients
   - Re-runs forward pass to compute outputs (stateless backward)
   - Calls `torch.autograd.backward()` with upstream `grad_outputs`
   - Acquires `optimizer_lock` to prevent race conditions with concurrent requests
4. **Optimizer step**: `on_backward()` (`module_collab.py:82-91`) calls `optimizer.step(batch_size)` which triggers:
   - Gradient accumulation
   - Collaborative gradient averaging (when target batch size reached)
   - Parameter update
   - LR scheduler step

### Gradient Synchronization

After sufficient batches accumulate across all peers in a stage:

1. `AutoStepOptimizer._step()` (`optim.py:232`) calls `_maybe_schedule_gradient_averaging()`
2. The gradient averager performs all-reduce (butterfly pattern or PowerSGD) with matched peers
3. Averaged gradients are applied via `_update_global_epoch()` (Hivemind base class)
4. State averaging occurs every `average_state_every` epochs, synchronizing model weights, optimizer state, and extra tensors

### Compression in Transit

Between pipeline stages, if `use_compression=True`:
- **Head output**: `compress_output()` (`layers.py:502-517`) projects `(B, S, H)` to `(B, S, H/compression_rate + 1)` via `rcv.T @ (x - fixed_embed)`, appending token indices
- **Body input**: `decompress_input()` (`layers.py:519-534`) reconstructs via `rcv @ compressed + fixed_embed`
- This reduces inter-stage communication by `compression_rate` (e.g., 100x for 8B model config)

---

## File Reference

| File | Purpose |
|------|---------|
| `src/node0/run_server.py` | Entry point, startup orchestration |
| `src/node0/server/node0_server.py` | Server class, expert/optimizer creation |
| `src/node0/server/module_collab.py` | Thread-safe backward pass with optimizer locking |
| `src/node0/server/optim.py` | `AutoStepOptimizer` - collaborative optimizer with auto-step |
| `src/node0/server/HM_averager.py` | `DecentralizedAverager` - base P2P averaging |
| `src/node0/server/HM_gradient_averager.py` | `GradientAverager` - gradient accumulation and averaging |
| `src/node0/server/HM_state_averager.py` | `TrainingStateAverager` - state synchronization |
| `src/node0/server/state_averager_wrap.py` | Sparse state averaging wrapper |
| `src/node0/server/power_sgd_averager.py` | PowerSGD gradient compression |
| `src/node0/server/power_sgd_averager_mac.py` | Mac-compatible PowerSGD with chunked matmul |
| `src/node0/server/ar_runner.py` | `AllReduceRunner` - butterfly all-reduce protocol |
| `src/node0/server/matchmaking.py` | Peer matchmaking with fixed group keys |
| `src/node0/models/arguments.py` | `ModelArguments` base config |
| `src/node0/models/llama/arguments.py` | `LlamaArguments` with LLaMA-specific defaults |
| `src/node0/models/llama/layers.py` | Transformer blocks, Head/Body/Tail experts |
| `src/node0/models/lr_schedule.py` | Cosine and linear LR schedulers |
| `src/node0/security/authorization.py` | `PluralisAuthorizer` - auth client |
| `src/node0/security/validation.py` | DHT record validators and schemas |
| `src/node0/utils/common.py` | `build_cls()`, `infer_expert_params()`, compression loading |
| `src/node0/utils/get_parameters.py` | DHT parameter store retrieval |
| `src/node0/utils/dht_partition.py` | DHT partitioning and peer port routing |
| `src/node0/utils/dht_monitor.py` | DHT RPC call monitoring |
| `src/node0/utils/monitor.py` | `MonitorWorker` - metrics, health checks, FLOP counting |
| `configs/llama_8B_C.yaml` | LLaMA 8B model configuration |
| `run.json` | Runtime config pointing to YAML + auth server + seeds |

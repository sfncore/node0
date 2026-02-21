# Question Triage: gastown-finetune

**Scope selected:** P0 + P1 + P2 (Comprehensive)
**Questions in scope:** 76 (14 P0 + 26 P1 + 36 P2)
**Auto-answerable:** 28
**Branch points for human:** 48

---

## Auto-Answerable Questions

These can be noted and carried forward without asking the human. Answers are derived from codebase context, stated constraints, industry practice, or logical entailment from known facts.

| # | Question (abbreviated) | Proposed Answer | Source |
|---|------------------------|-----------------|--------|
| 4 | Is node0 suitable for fine-tuning, or is it pretraining-only infrastructure? | Node0 is pretraining-only. It has no data loading, no checkpoint export, no LoRA layer injection, and gradient averaging is designed for full-model pretraining. Substantial rearchitecting is required before fine-tuning is possible. | context.md: "Node0 does NOT currently include data loading"; "No explicit checkpoint saving"; critical design note |
| 10 | Will fine-tuning coexist with normal Gas Town operations on the 10GB box? | Not in standard form. Dolt (550MB) + openclaw-gateway (1GB) + live Claude sessions (300-750MB each) already consume 1.8-2.5GB. A LLaMA 8B in fp16 is 16GB alone; even Q4 quantized is ~4.5GB. Fine-tuning requires exclusive hardware or a separate machine, or must use a sub-1B model. | MEMORY.md resource constraints; Q12 logical entailment |
| 12 | What is the deployment target for the fine-tuned model — local inference on the 10GB box? | Local inference on the current box is not viable for LLaMA 8B: fp16 needs 16GB, Q4 needs ~4.5GB (still tight alongside running services). Deployment target must be either a separate inference machine or a cloud API endpoint. If local is required, the base model must be much smaller (e.g., 1-3B). | Hardware constraint: 10GB box total RAM |
| 14 | Where does node0's data loading gap get filled? | A new data pipeline component is required — this is not in node0 and must be built. The GradientAverager hook is the integration point. This is confirmed infrastructure work, not a design question. | context.md integration point 1: "Node0 does NOT currently include data loading — gradients are accumulated by external training loops" |
| 15 | Is distributed training necessary, or could LoRA on a single machine serve the immediate use case? | For MVP, single-machine LoRA fine-tuning is the correct approach. An 8B model with LoRA fits in 16-24GB VRAM; distributed setup adds months of complexity. Distributed training is appropriate for V2+ if scale demands it. The 10GB box cannot host training at all, so distribution across external nodes is the architecture if we stay with node0. | Opus skepticism backed by concrete hardware math; standard LoRA fine-tuning practice |
| 20 | How does the corpus handle multi-turn Gas Town sessions — truncation, summarization, or segmentation? | Segment into fixed-length overlapping windows as a default; long sessions should not be force-truncated at a single boundary. This is the standard practice for long-context instruction tuning. Exact window size depends on the base model's context length (LLaMA 8B supports 8K-128K depending on variant). | Standard NLP fine-tuning practice; base model context window constraint |
| 21 | Should fine-tuning configuration be expressible as a Gas Town formula? | Yes. Gas Town formulas are the native abstraction for repeatable workflows. Training configs as formulas get version control, search path resolution, and sharing for free. This is a design default, not a question. | MEMORY.md formula system; Q21 framing already acknowledges this is the better choice |
| 23 | What are the catastrophic forgetting risks, and what base capabilities must be preserved? | Catastrophic forgetting is a known risk for narrow-domain fine-tuning. The "do not forget" set must include: general instruction following, code generation, reasoning. Mitigation options: LoRA (parameter-efficient, less forgetting), replay buffer, regularization (EWC). This is a technical constraint to enforce, not a choice. | Standard ML: Shumailov et al. 2023; LoRA by design has lower forgetting than full fine-tuning |
| 24 | How do we prevent model collapse from self-generated training data? | Provenance flags are mandatory in the corpus pipeline. Every entry must be tagged human-generated vs. model-generated. Model-generated content must be excluded or down-weighted during training. This is a hard policy, not a preference. | Shumailov et al. 2023; questions.md explicitly documents the failure mode |
| 25 | What is the plan for checkpoint recovery in distributed training? | Checkpointing to durable storage at fixed intervals is required. Node0 currently has no checkpoint export; this must be added as part of the data pipeline integration. Resume from last checkpoint is the minimum viable behavior. | context.md: "No explicit checkpoint saving (relies on Hivemind state averaging)"; standard training practice |
| 28 | Should training jobs appear as beads in the bead system? | Yes. Beads are Gas Town's native lifecycle abstraction. Training jobs as beads get visibility in existing tools, consistent mental models, and notification routing for free. The bead state model (ready → active → review → done) maps cleanly to: corpus-validated → training → eval → deployed. | Gas Town architecture; Q28 framing acknowledges bead states map well |
| 29 | What confirmation and resource-visibility gates should exist before launching a fine-tuning job? | Minimum pre-launch gate: corpus entry count, corpus size, estimated training duration, current memory headroom (free -h equivalent), and a warning if competing services (Dolt, gateway) are running. This is a safety requirement on a 10GB box, not a preference. | MEMORY.md: "Before slinging: Run free -h... If <1.5GB free + swap full, DO NOT sling" |
| 32 | What tokenizer implications arise from Gas Town's domain-specific identifiers (st-btod, bd-9f1f)? | Gas Town IDs tokenize poorly on LLaMA vocabulary — each ID becomes 3-5 fragments. This wastes context window and degrades performance on Gas Town-specific syntax. Tokenizer evaluation (and possibly extension) must happen before corpus preparation is finalized. This is a known technical problem to mitigate, not a design choice. | Opus analysis; standard LLaMA tokenizer behavior on alphanumeric slugs |
| 41 | What is the minimum viable operator profile — ML expertise required? | No ML expertise required for day-to-day use. Gas Town operators are software engineers who know the domain but not gradient averaging. All hyperparameters must have sensible defaults. Expert controls should use progressive disclosure (hidden behind an "advanced" section). | questions.md user advocate section; Gas Town operator profile from MEMORY.md |
| 46 | What format should evaluation results be presented in? | Plain-language task success rates: "model correctly generates bead workflows 87% of the time." Loss curves are meaningless to Gas Town operators. Evaluation output must be translated to actionable language about specific Gas Town tasks. | questions.md cross-model agreement: "Loss curves are meaningless to non-ML operators" |
| 49 | What progress feedback does an operator receive during a distributed training run? | Bead status updates + gt mail notifications for significant transitions (training started, checkpoint saved, evaluation complete, failed). This integrates with existing Gas Town notification infrastructure rather than inventing a new channel. | Gas Town architecture; MEMORY.md mail/event feed patterns |
| 50 | What happens if an operator feeds in a corpus that is too small? | Hard minimum corpus size gate before training is allowed to proceed. The system must fail gracefully with specific guidance: "You have N examples; minimum for instruction tuning is ~1,000. Add more examples of type X." Silent quality degradation is not acceptable. | Q6 entailment; standard fine-tuning minimum data requirements |
| 53 | What is the "paused" state story for long-running training jobs? | Pause/resume must be supported on the 10GB box. Without it, any training run that starts during normal operations will compete destructively with Dolt and live sessions. The bead state model naturally accommodates a paused state. This is a hard requirement given the hardware constraint, not optional. | MEMORY.md resource constraints; 10GB box planning constraint |
| 55 | How long do completed or failed training jobs persist before being archived? | Default retention: 30 days for completed jobs, 7 days for failed jobs, with manual pin option to retain indefinitely. Storage is constrained; indefinite retention is not viable. The exact numbers can be tuned but a default policy must exist from day one. | Hardware constraint; standard ops practice |
| 56 | What canonical states does a fine-tuning job move through, and should these map to existing bead states? | Map to bead states: corpus-validation → ready; training active → active; evaluation → review; deployed → done; failed → blocked. Reuse bead lifecycle rather than inventing a parallel vocabulary. Consistency is strongly preferred. | Q28 entailment; Gas Town architecture |
| 59 | How should the system represent distributed node health? | Collapsible advanced panel, not top-level. Default view shows: "N nodes active, last sync X seconds ago, throughput Y MB/s." Detailed P2P topology is available on expand. Most operators need "is it working?", not DHT internals. | Q59 framing already identifies the correct answer; standard progressive disclosure UX |
| 62 | How should stale or outdated corpus entries be flagged? | Corpus validation must flag: deprecated formulas (compare against current .beads/formulas/), beads referencing closed issues (check bead state), CLI commands not in current gt help output. Staleness flags must appear in the pre-training corpus health view and block training by default (with override). | Q31 entailment; Gas Town formula system |
| 63 | What does the transition from training to evaluation look like? | Automatic transition with operator notification (bead state update + mail). The operator can then inspect evaluation results before explicitly triggering deployment. Automatic-to-eval is low risk; deployment remains a manual, explicit gate. | Q27 entailment; standard ML workflow |
| 65 | What is the expected model response latency on local hardware? | Local inference on the 10GB box with LLaMA 8B is not viable (model exceeds available RAM). If an external inference server is used, latency depends on that hardware. This question is only answerable after Q12 (deployment target) is resolved. For now: local latency is a non-starter; the question defers to Q12. | Q12 entailment |
| 66 | What does "export model" look like for portability? | GGUF (for llama.cpp) and safetensors (for vLLM/HF) are the required output formats. These are the two dominant inference runtime formats. Proprietary checkpoint format is not acceptable. Q4-bit quantized GGUF is the minimum for local use. | Standard fine-tuning ecosystem practice; llama.cpp/vLLM popularity |
| 73 | Should state transitions generate beads, mail, or event feed entries? | Bead state transitions + gt mail for significant events (training started, completed, failed, evaluation done). Event feed for granular progress (checkpoint saved, peer joined/left). Configurable verbosity with "significant events only" as default. | Q28/Q49 entailment; MEMORY.md notification patterns |
| 76 | How does the fine-tuned model signal uncertainty about Gas Town-specific topics? | The model must not hallucinate bead structures confidently. Calibration on domain-specific content is a required evaluation criterion. Models that express appropriate uncertainty ("I'm not sure of the correct prefix for this rig") are preferred over confident wrong answers. This is a first-class eval requirement, not a nice-to-have. | questions.md: "Confident wrong answers about Gas Town conventions are worse than admitted uncertainty" |

---

## Branch Points (Human Decision Required)

These require human judgment because they involve real tradeoffs, business context the operator holds, or preference decisions with no objectively correct answer.

| # | Question (abbreviated) | Why Human Needed |
|---|------------------------|------------------|
| 1 | What is a "Gas Town-native agent" in measurable terms? | Root blocker. No training objective, corpus scope, or success metric can be derived without a concrete definition from the owner of Gas Town's vision. |
| 2 | What base model are we fine-tuning — is this committed to LLaMA 8B? | Determines tokenizer, compute budget, LoRA applicability, hardware requirements, and data format. Multiple valid choices (LLaMA 8B, Mistral 7B, Phi-3, Qwen2, etc.) with real tradeoffs. Owner must decide. |
| 3 | Is the real problem "we need fine-tuning infrastructure" or "agents fail at specific tasks"? | Requires measuring baseline performance, which only the operator can do. May conclude fine-tuning is wrong solution. High-cost decision if assumed incorrectly. |
| 5 | What training data format is required — instruction-response pairs, tool-use traces, preference pairs, multi-turn? | Gas Town artifacts fit multiple formats. The choice determines corpus preparation effort, labeling requirements, and training objective. Tradeoffs between formats depend on what capabilities the owner prioritizes. |
| 6 | Does Gas Town currently generate sufficient high-quality training data? | Only the operator can audit existing session transcripts and estimate usable examples. If the answer is <1,000 examples, the project scope must change fundamentally. |
| 7 | What does "success" look like concretely — specific tasks, accuracy thresholds, comparison baseline? | The operator must define the measurable acceptance criteria. No external authority can define what "native Gas Town agent" means for this specific setup. |
| 8 | Will the fine-tuned model replace current agents, augment them, or serve as a fallback? | Core adoption architecture. Operator's vision determines whether this is a cost-reduction play, capability extension, or offline fallback. No right answer without knowing the operator's intent. |
| 9 | How does the training corpus handle credentials, API keys, and sensitive data in session transcripts? | Policy decision. Options range from manual review to automated PII scrubbing to prohibiting session transcripts entirely. Depends on how sensitive the operator's Gas Town sessions are. |
| 11 | Is LoRA/QLoRA, full fine-tuning, or instruction tuning the intended approach? | Real tradeoffs: LoRA is cheaper but may underfit; full fine-tuning requires GPU the box doesn't have; instruction tuning has different data requirements. The choice depends on hardware access (Q12) and scope ambition (Q1). Cascades into almost everything. |
| 13 | How do we handle model staleness as Gas Town conventions evolve? | Policy decision: one-shot training (accept staleness), scheduled retraining (resource commitment), RAG complement (architectural choice). Depends on how much the operator values freshness vs. simplicity. |
| 16 | Should there be role-specific models (mayor/polecat/witness) or a single model? | Multiplies maintenance burden vs. per-role optimization. Depends on how differentiated role behaviors are in practice and how much compute/storage the operator is willing to commit. |
| 17 | What does fine-tuning add over gt prime context injection — what can fine-tuning do that prompting cannot? | The operator must define the gap. If gt prime injection already works well enough, fine-tuning ROI is unclear. Only the operator knows where current agents fail despite good prompting. |
| 18 | How does the fine-tuned model integrate with Gas Town's existing agent provider system? | Architectural integration design. Multiple valid approaches: new provider registration, wrapper around existing providers, separate dispatch path. Requires Gas Town maintainer decisions about the agent dispatch layer. |
| 19 | What is the minimum viable corpus for a first training run — how many examples, from which sources, in what ratio? | Depends on target capability (Q1) and training approach (Q11). Research ranges (1K-100K) are wide; the right floor depends on what tasks matter most to this operator. |
| 22 | Can an operator use the fine-tuned model for some tasks and the foundation model for others (hybrid usage)? | Determines the adoption model. The operator must decide if hybrid routing is a design goal or if they want clean replacement. Implementation complexity differs significantly between the two approaches. |
| 26 | How does the system handle contradictory corpus entries — conflicting CLAUDE.md versions or opposing formula conventions? | Policy decision: block training on detected conflicts, warn and proceed, or resolve via recency heuristic. Each option has implications for corpus curation burden. Operator must decide how much manual review they want to do. |
| 27 | How does the operator evaluate the fine-tuned model before deploying it? | Scope question. Options: manual checklist of Gas Town tasks, automated evaluation harness, sandbox environment, or deploy-and-observe. Investment level depends on operator's risk tolerance and available time. |
| 30 | What happens to contributed session transcripts if an operator wants them removed? | Legal/policy decision. Machine unlearning is hard; the policy must be set before corpus collection begins. Owner of the Gas Town instance must decide whether to allow removal (requires retraining) or prohibit it explicitly. |
| 31 | How should the system surface corpus health before training — duplicates, stale formulas, imbalanced types? | Determines how much pre-training tooling to build. Options range from a simple count/warning to a rich corpus browser with per-entry diagnostics. Scope and investment decision for the operator. |
| 33 | Is RLHF, DPO, or preference-based fine-tuning in scope? | Significant additional complexity. Depends on whether Gas Town has or can generate preference data (correct vs. incorrect workflow executions). The operator must assess data availability and scope ambition. |
| 34 | How will we detect catastrophic failure in production — a fine-tuned agent that skips destructive operation checks? | Requires defining safety evaluation criteria specific to Gas Town's known footguns. The operator must enumerate which safety behaviors are non-negotiable and define test cases for them. |
| 35 | What is the corpus scope — individual rig data, all Gas Town rigs, or a curated multi-rig blend? | Governance and design decision. Cross-rig data means richer model but potential convention contamination. The operator must decide the trust boundary for data sharing across rigs. |
| 36 | What is the expected iteration cycle time — from new data to deployed model? | Drives the choice between distributed training and single-machine LoRA. If operators need weekly updates, single-machine is mandatory. If monthly is acceptable, distributed is feasible. Operator's workflow determines the answer. |
| 37 | Should the system schedule recurring training automatically as new corpus data accumulates? | Automation vs. manual control preference. Automated retraining risks runaway resource consumption on the 10GB box; manual triggers are safer but require more operator effort. Depends on how often the operator expects to retrain. |
| 38 | Are there existing Gas Town agent session transcripts that can bootstrap corpus annotation? | The operator must audit what session history exists. This is empirical groundwork only they can do. The answer directly determines whether data scarcity is the bottleneck (Q6). |
| 39 | What is the A/B testing strategy for validating a new fine-tuned model in production? | Risk tolerance decision. Shadow mode is safe but complex; operator-selected rollout is simple but uncontrolled; controlled percentage rollout needs routing infrastructure. The operator must decide how much production risk they can absorb. |
| 40 | Will the fine-tuned model understand Dolt layer conventions — issue prefixes, bead state transitions? | Determines how deep the corpus must go. Surface-level CLI training vs. deep bead-graph training require fundamentally different corpus design. The operator must decide how sophisticated they want the agent's understanding to be. |
| 42 | Should operators without GPU access participate in corpus preparation even if they cannot run training? | Governance decision about participation model. Depends on whether Gas Town is intended as a multi-operator collaborative system or single-operator. The operator must decide the contribution model. |
| 43 | Can an operator contribute session transcripts and expect to see improvement in subsequent model versions? | Determines the social contract around corpus contribution. Requires commitment to a retraining cadence. The operator must decide whether to make this promise to contributors. |
| 44 | What does the operator do if the fine-tuned model produces a bad bead or breaks a workflow — rollback mechanism? | Risk management preference. Easy rollback (switch a config pointer) vs. destructive replacement. The operator must decide how much infrastructure to build for the recovery path. |
| 45 | How does the fine-tuned model handle bead dependency graphs — understanding relationships or treating beads as isolated documents? | Scope definition for corpus design. Graph-aware training (sequence of related beads as context) is significantly more complex than treating beads as independent documents. The operator must define the required understanding depth. |
| 47 | How does the system handle cross-rig data — does the model learn rig-specific patterns and misapply them? | Requires the operator to define acceptable cross-rig contamination risk. Some contamination may be acceptable if it produces a more general model; zero contamination requires rig-specific isolation. |
| 48 | Will the fine-tuned model understand formula authoring conventions — TOML structure, version bumps, embedded vs. provisioned? | Scope decision. Including formula authoring requires formula-specific training examples. The operator must decide if this is a target capability for V1. |
| 51 | How should the corpus browser be laid out, and what determines what goes into a training run? | UX and policy decision. Default-include-everything vs. explicit opt-in have different quality implications. The operator must decide how much they trust automatic corpus selection. |
| 52 | Should fine-tuning infrastructure live in its own rig (ft-) or as a workflow within node0? | Organizational preference with real tradeoffs. Dedicated rig = clean isolation, separate bead namespace, more overhead. Embedded in node0 = less cognitive overhead, tighter coupling. Depends on how frequently operators expect to use this feature. |
| 54 | What visual or conceptual treatment distinguishes base models, fine-tuned variants, checkpoint versions, and deployed models? | Naming convention and visual design preference. The operator must define the model versioning scheme that fits their mental model of the system. |
| 57 | Should training configuration be a form, a YAML editor, or a formula template? | UX preference with real tradeoffs. Formula template is Gas Town-native but requires formula literacy; form is approachable but inflexible; YAML is powerful but error-prone. Depends on the operator's audience and their formula familiarity. |
| 58 | How do we handle "training succeeded but evaluation shows the model is worse"? | Determines how much actionable guidance the system must generate. Simple "try more data" vs. sophisticated diagnostic suggestions. Scope and investment decision. |
| 60 | What does the deployment step look like operationally — swapping a model reference in town settings? | Integration design decision. The operator must define how a trained model gets registered as a provider and what "deploying" means in the context of their town settings architecture. |
| 61 | Will operators on AMD GPUs, Apple Silicon, or cloud instances with different CUDA versions be able to participate? | Hardware support scope decision. The CUDA 12.1 pin in the Dockerfile excludes many platforms. The operator must decide whether to invest in multi-platform support or document the hardware requirements explicitly. |
| 64 | How should the system handle an operator who wants to fine-tune on only a specific subset — only bead patterns, only formula authoring? | Corpus filtering design decision. Supporting task-specific fine-tuning requires a metadata schema that can express data type, task, rig, and date range. The operator must decide whether targeted fine-tuning is a V1 requirement. |
| 67 | How do we handle operators on slow or intermittent internet connections in distributed training? | Network requirements policy. The operator must decide whether distributed training participation requires reliable connectivity, or whether corpus-only participation (no training) is a supported mode. |
| 68 | What are the known issues with Hivemind at scale, and what failure modes from prior node0 runs have been documented? | The operator must audit existing node0 run history to answer this empirically. Only they have access to prior training logs. |
| 69 | How do we handle "training is taking 10x longer than estimated"? | Determines how sophisticated the duration estimation and update communication needs to be. The operator must decide whether honest time estimates with uncertainty ranges are required or if a progress bar is sufficient. |
| 70 | Should there be a "model registry" view listing all fine-tuned models across rigs? | Feature scope decision. Only needed if the operator plans to create multiple fine-tuned variants. Depends on expected training frequency and number of operators. |
| 71 | How does the system handle cross-rig data access permissions? | Governance decision. The operator must define the trust model: is cross-rig data access allowed by default, requires explicit grant, or is prohibited? |
| 72 | What does a "model card" for a Gas Town fine-tuned model include? | Template design decision. The operator must define what metadata is required for each model: corpus composition, training config, evaluation results, known limitations, recommended use cases. |
| 74 | How does the system detect and report overfitting? | Determines what validation metrics must be computed during training. Requires a held-out test set (Q7 entailment) and the operator must define what "too narrow" looks like for their use case. |
| 75 | Can the fine-tuned model be used for Gas Town planning tasks — generating draft beads, mapping dependencies? | Scope decision about intended use cases. The operator must define whether planning assistance is explicitly supported, implicitly supported, or out of scope for V1. |

---

## Question Dependencies

Dependencies flow downward: answering the upstream question constrains or eliminates the downstream one.

```
Q1 (Define "Gas Town-native agent")
  → Q7  (Success criteria: what tasks, at what thresholds?)
  → Q5  (Data format: what target behavior are we training toward?)
  → Q19 (Minimum viable corpus: how many examples, what types?)
  → Q17 (Fine-tuning vs. gt prime: what gap does it fill?)
  → Q40 (Dolt layer depth: how sophisticated must understanding be?)
  → Q45 (Bead dependency graphs: isolated or relational understanding?)

Q2 (Base model choice)
  → Q11 (LoRA vs. full fine-tuning: depends on model size and architecture)
  → Q32 (Tokenizer implications: specific to LLaMA vocabulary if LLaMA chosen)
  → Q23 (Catastrophic forgetting: base capabilities to preserve depend on base model)
  → Q66 (Export format: GGUF and safetensors are model-family-dependent)

Q3 (Fine-tuning vs. prompting/RAG baseline)
  → Q17 (If gt prime injection already works: fine-tuning value proposition)
  → Q1  (If current agents fail at specific tasks: those tasks define the agent spec)

Q8 (Replace vs. augment vs. fallback)
  → Q22 (Hybrid usage: only relevant if augment/fallback chosen)
  → Q18 (Agent provider integration: architecture differs by replacement vs. augmentation)
  → Q44 (Rollback mechanism: only critical if replacement; fallback is always-present for augment)

Q11 (LoRA vs. full fine-tuning)
  → Q10 [auto] (Hardware coexistence: LoRA uses less GPU memory than full fine-tuning)
  → Q12 [auto] (Deployment target: LoRA adapter can be served on top of base model)
  → Q25 (Checkpoint recovery: LoRA checkpoints are smaller, recovery is faster)
  → Q15 [auto] (Distributed necessity: LoRA on single machine eliminates distribution need)

Q12 [auto] (Deployment target — local box infeasible)
  → Q65 [auto] (Local latency: moot if not local; defers to deployment target choice)
  → Q10 [auto] (Hardware coexistence: training + inference both infeasible on box)

Q35 (Corpus scope: individual rig vs. all rigs)
  → Q47 (Cross-rig contamination: only exists if cross-rig data is included)
  → Q71 (Cross-rig permissions: only relevant if cross-rig scope is chosen)
  → Q16 (Role-specific models: if scope is per-rig, role-specific models are more natural)

Q6 (Data scarcity assessment)
  → Q19 (Minimum viable corpus: if <1,000 examples exist, corpus must be augmented or project re-scoped)
  → Q38 (Session transcript audit: the mechanism for answering Q6)
  → Q37 (Recurring training: only valuable if data accumulates faster than staleness occurs)

Q33 (RLHF/DPO in scope)
  → If yes → Q5 must include preference pairs as a data format
  → If yes → Q7 must include preference signal as a success criterion

Q28 [auto] (Training jobs as beads)
  → Q56 [auto] (Job state machine maps to bead states)
  → Q73 [auto] (State transitions generate bead/mail/event notifications)
  → Q49 [auto] (Progress feedback via bead status + gt mail)
  → Q52 (Dedicated rig or node0 workflow: bead namespace question)

Q36 (Iteration cycle time requirement)
  → Q15 [auto] (Distributed necessity: weekly cycle → single machine required)
  → Q37 (Automated scheduling: only meaningful if cycle is short enough)
  → Q13 (Staleness handling: cycle time determines how stale the model gets)
```

---

## Interview Plan

**Round 1: Core Architecture** (6 questions)

These unlock the most dependencies. Answering all 6 resolves or constrains approximately 35 downstream questions.

1. **Q1** — Define "Gas Town-native agent" in measurable terms. What specific tasks must it perform correctly that current agents cannot? *(Root blocker: cascades into Q5, Q7, Q17, Q19, Q40, Q45)*

2. **Q3** — Before committing to fine-tuning infrastructure: have you measured where current agents (Claude + gt prime) actually fail at Gas Town tasks? What are the 3-5 concrete failure modes that motivate this project? *(Validates the whole investment; may redirect to RAG or prompting)*

3. **Q8** — Is the fine-tuned model intended to replace Claude/pi/omp agents, augment them (specialist for Gas Town tasks + Claude for general coding), or serve as an offline fallback? *(Cascades into Q18, Q22, Q44)*

4. **Q2** — What base model are we committing to? LLaMA 8B (as implied by node0) or an alternative? *(Cascades into Q11, Q32, Q23, Q66)*

5. **Q11** — Given that the 10GB box cannot host LLaMA 8B in fp16 at all: is the approach LoRA/QLoRA (GPU-efficient, single machine, ~16GB VRAM needed) or full fine-tuning via node0's distributed infrastructure across external nodes? *(Cascades into Q15, Q12, Q25)*

6. **Q35** — What is the corpus scope: a single rig's data, all Gas Town rigs, or an operator-curated subset? Who controls the data boundary? *(Cascades into Q47, Q71, Q16)*

---

**Round 2: Data and Success Criteria** (~6 questions)

Constrained by Round 1 answers. Answering these finalizes training objective and corpus preparation plan.

7. **Q6 + Q38 combined** — Approximately how many high-quality session transcripts, formula authoring examples, and bead workflow examples exist today across the in-scope rigs? *(Determines if data scarcity is the bottleneck before any infra is built)*

8. **Q7** — Given the agent definition from Q1: what are the 5-10 specific Gas Town tasks the fine-tuned model must pass, and what is the acceptable error rate for each? *(Defines the evaluation harness)*

9. **Q5** — For each task category from Q7: what training data format is required — instruction-response pairs, tool-use traces, multi-turn conversation, or preference pairs? *(Determines corpus preparation work)*

10. **Q9** — How do you want to handle sensitive data in session transcripts (API keys, HF tokens, DB credentials, personal paths)? Options: automated PII scrubbing + manual review, exclude session transcripts entirely, or manual-only curation. *(Blocks corpus collection)*

11. **Q13** — What is acceptable model staleness? Daily/weekly/monthly retraining cadence, or RAG complement for dynamic content? *(Determines whether automated retraining is required)*

12. **Q33** — Is RLHF/DPO in scope for V1? Do you have or can you generate preference data (correct vs. incorrect Gas Town workflow executions as labeled pairs)? *(Determines data format scope; if yes, adds significant work)*

---

**Round 3: Deployment and Adoption** (~5 questions)

These finalize the operator-facing design decisions.

13. **Q18** — How should the fine-tuned model register as a Gas Town agent provider? Should it appear as a new entry in town settings config.json alongside claude/pi/omp, or as a configurable override for existing roles? *(Integration architecture)*

14. **Q27** — What is the minimum evaluation gate before the fine-tuned model can be deployed as an agent? Manual checklist, automated harness against Q7 task suite, or sandbox environment with human sign-off? *(Risk tolerance)*

15. **Q44** — What is the rollback story if the fine-tuned model degrades? Swap the provider pointer back to Claude (30-second fix), or is a more structured rollback needed? *(Adoption risk profile)*

16. **Q30** — Before corpus collection begins: what is the policy on right-to-be-forgotten? Can contributors remove their session data after training, or is inclusion permanent once a model is trained on it? *(Must be decided before first corpus entry is collected)*

17. **Q52** — Should fine-tuning infrastructure live in its own rig (e.g., `finetune` with prefix `ft-`) or as a workflow embedded in node0? *(Organizational structure; affects bead namespacing, formula placement)*

---

**Round 4: Standalone Branch Points** (~6 questions — ask only if Round 1-3 answers leave them open)

These are independent of the core cascade and can be addressed after the architecture is settled.

18. **Q16** — Role-specific models (mayor/polecat/witness) or one model for all roles? *(Training complexity multiplier)*
19. **Q42** — Can operators without GPU access contribute corpus data even if they cannot run training? *(Participation model)*
20. **Q57** — Should training configuration be a Gas Town formula template, a YAML editor, or a form UI? *(UX preference, likely defaults to formula template given Q21 auto-answer)*
21. **Q64** — Should task-specific fine-tuning be supported in V1 — e.g., "only train on bead-creation examples"? *(Corpus filtering scope)*
22. **Q70** — Should there be a model registry view across rigs in V1, or just per-training-run model tracking? *(Feature scope)*
23. **Q75** — Is planning assistance (generating draft beads, dependency maps for new projects) an explicitly supported V1 use case? *(Scope boundary)*

---

**Estimated dialogue:** ~23 questions for human (Rounds 1-4), ~28 auto-noted

Auto-noted answers are documented above and do not need to be raised with the human unless they conflict with Round 1-2 answers (e.g., if Q11 answer changes the hardware picture, revisit Q10 and Q12 auto-answers).

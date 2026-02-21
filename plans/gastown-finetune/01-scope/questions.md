# Scope Questions: gastown-finetune

## Models Used
- Opus 4.6 (Claude)
- Kimi K2 (via omp)
- GPT/Codex: unavailable (no auth)
- Gemini: unavailable (account disabled)

## Question Counts
- Raw questions from 6 analyses (3 perspectives × 2 models): 382
- After deduplication (merging similar): 97
- P0: 14 | P1: 26 | P2: 36 | P3: 21

---

## Cross-Model Comparison by Perspective

### User Advocate
**Both models flagged (high confidence):**
- Fine-tuned model replacing vs. augmenting current agents (Claude, pi, omp) — core expectation confusion
- Corpus quality: including session mistakes, credentials/secrets, and stale conventions
- Checkpoint/resume: job interruption produces unusable partial state
- Model staleness as Gas Town evolves (formulas, CLI, conventions change continuously)
- GPU hardware barrier excluding CPU-only operators
- ML jargon (gradient averaging, pipeline parallelism) excluding non-technical operators
- Privacy/data leakage in distributed network
- Rollback: reverting to previous model version when degradation detected
- Corpus assembly effort: manual vs. automated curation

**Only Opus flagged:**
- Role-aware behavior (mayor vs. polecat vs. witness acting differently)
- Hybrid usage: fine-tuned model for Gas Town tasks + foundation model for general coding
- Dolt-layer understanding (issue prefixes, bead state transitions, DB names)
- Operator emotional state: frustration-driven vs. proactive adoption
- Tokenizer/checkpoint format compatibility with inference runtimes (vLLM, llama.cpp)
- Multilingual corpus consideration
- Screen-reader / accessibility for terminal-based monitoring
- Cross-rig contamination (learning sfgastown patterns and applying in frankentui)

**Only Kimi flagged:**
- Right to be forgotten: removing contributed data from trained weights after the fact
- Bead dependency graph understanding (not just individual beads as documents)
- Using fine-tuned model for legacy bead migration (not just new content creation)
- Witness role: verifying bead completeness and acceptance criteria
- Generating synthetic session transcripts to deceive authorship tracking
- Whether fine-tuning is daily, weekly, or monthly activity
- Corpus scope: individual rig data vs. all Gas Town data — privacy and model personality

**Where models disagreed:**
- Opus treated "who curates corpus" as high anxiety (operators don't want to do it). Kimi treated it as a design decision (curation burden determines participation rates) — compatible but different emphasis.
- Opus focused on the operational danger of a rogue model (suggests unbounded parallelism on a 10GB box). Kimi focused on loss of control/agency as an adoption barrier.

### Product Designer
**Both models flagged (high confidence):**
- Job state visibility at a glance (preparing → training → evaluating → deployed → failed)
- Corpus health before committing GPU hours (size, composition breakdown, stale entries)
- Dry-run / validation mode before actual training starts
- Cancel semantics: what is preserved when user stops a running job
- Training-to-deployment gap: "model trained" ≠ "model usable as agent"
- Should training runs appear as beads (using existing bead lifecycle metaphors)?
- Evaluation results communication: not just "worse/better" — actionable guidance
- Comparison tooling between training runs/model versions
- Node health and distributed state: top-level or collapsed panel?
- Hyperparameter exposure: progressive disclosure (expert vs. novice path)
- Corpus composition visualization (formulas vs. workflows vs. transcripts breakdown)
- Queue behavior and resource contention communication on constrained hardware

**Only Opus flagged:**
- Per-entry corpus diagnostics (not bulk "corpus invalid")
- Traceability from model output back to specific training data entries
- Fine-tuning config expressible as a formula (Gas Town-native shareable recipes)
- "Superseded" / "archived" states for deployed models when newer version exists
- Backward state transitions (training → preparing if corpus needs change)
- Keyboard shortcuts / power-user paths for repeat runners
- Density level defaults (compact for experienced Gas Town operators)
- State transition notification via existing channels (beads, mail, event feed)

**Only Kimi flagged:**
- Scheduling recurring training (continuous learning automation)
- "10x longer than estimated" case: honest update vs. spinner
- Scope selector: "my rig only" vs. "all Gas Town data"
- Model card / training run card visual design
- "Waiting for nodes to join" state — purposeful vs. stuck appearance
- Checkpoint-saved milestones surfaced during training (invisible progress made visible)

**Where models disagreed:**
- Opus was more opinionated about form factors (formula templates > YAML editors > forms). Kimi raised the question without prescribing an answer — Kimi's framing is more appropriate for scoping.
- Opus proposed dedicated rig (`finetune` with prefix `ft-`). Kimi left this open as a question about navigation prominence. Both are valid angles.

### Domain Expert
**Both models flagged (high confidence):**
- "Gas Town agent" is undefined in measurable terms — no training objective derivable yet
- Base model is unspecified; node0 docs imply LLaMA 8B but brief doesn't commit
- Fine-tuning vs. pretraining: node0 is pretraining infrastructure; adapting it for fine-tuning requires rearchitecting
- Data format: instruction-response pairs vs. tool-use traces vs. preference pairs — not addressed
- Evaluation / holdout set is absent — no way to detect overfitting or measure improvement
- RAG as alternative or complement to fine-tuning: not evaluated
- Catastrophic forgetting: fine-tuning on narrow corpus can destroy base model capabilities
- Session transcripts include secrets/credentials — sanitization is mandatory, not mentioned
- Corpus staleness as conventions evolve: baked-in knowledge vs. dynamic retrieval
- LoRA/QLoRA vs. full fine-tuning distinction: different compute budgets, outcome profiles, and data requirements
- Success criteria undefined — what does "native Gas Town agent" mean measurably?
- Data scarcity vs. data organization: do we need more data or better curation of existing data?

**Only Opus flagged:**
- Curriculum design: general instruction following before Gas Town specifics, or mixed?
- Tokenizer efficiency: Gas Town IDs (st-btod, bd-9f1f) tokenize poorly on LLaMA vocabulary
- Byzantine fault tolerance for gradient poisoning in decentralized fine-tuning
- Model collapse from self-generated training data (polecat-generated code → merged → retrained on)
- Reproducibility in dynamic peer join/leave environment
- Prior art: Gorilla, ToolLLM/ToolBench, ShellGPT, StarCoder failure modes
- Role-specific models (mayor / polecat / witness) vs. single fine-tuned model
- Storage constraints: LLaMA 8B is 16GB+ in fp16 on a 10GB box
- RLHF/DPO alignment for "correct workflow execution" as preference signal
- Data pipeline gap: node0 explicitly does NOT include data loading

**Only Kimi flagged:**
- Agent lifecycle (spawn, operate, terminate) as a training target — stateful processes need sequence modeling
- Partial observability: agents need training on information-gathering before action-taking
- "Protocol Learning" mentioned in node0 docs — never operationally defined
- Rejected alternatives to Hivemind: understanding why alternatives were discarded reveals constraints
- Error handling patterns as high-value training data (canonical recovery sequences)
- Training-time vs. inference-time distinction for context injection (more context = less fine-tuning needed?)
- Online learning / learning from feedback after deployment
- A/B testing strategy for deployed agents in production
- Token efficiency as a success metric (output quality per inference cost)

**Where models disagreed:**
- Opus was skeptical that the distributed infrastructure is necessary at all for fine-tuning an 8B model (can be done on a single 24GB GPU). Kimi accepted the distributed framing and focused on how to make it work. Opus's skepticism is higher value for scoping.
- Opus raised model collapse from self-generated data explicitly (citing Shumailov et al., 2023). Kimi raised it as "recursive training on synthetic data" in the user section. Both, from different angles.

---

## P0 — Must Answer Before Proceeding

*Both models flagged, or single-model flag with critical-blocker impact. These block meaningful scope, corpus design, or infrastructure sizing decisions.*

**1. What is a "Gas Town-native agent" in measurable terms — what capabilities, autonomy boundaries, and interaction patterns distinguish it from a prompted frontier model?**
Without a crisp definition, no training objective can be derived, no corpus can be scoped, and no success metric can be defined. This is the root blocker for the entire project. *(Opus, Kimi K2)*

**2. What base model are we fine-tuning, and is this infrastructure model-agnostic or committed to LLaMA 8B?**
Node0's existing infrastructure implies LLaMA 8B, but the brief says "fine-tune an LLM." The base model determines tokenizer choices, compute budget, format requirements, and whether LoRA is even applicable. Everything downstream is conditional on this answer. *(Opus, Kimi K2)*

**3. Is the real problem "we need fine-tuning infrastructure" or "agents currently fail at specific Gas Town tasks" — and have we measured the baseline to confirm fine-tuning is the right solution rather than better prompting, RAG, or workflow constraints?**
We may be solving a data or scaffolding problem with an infrastructure solution. Without baseline measurements of current model performance on Gas Town tasks, we cannot justify the investment or know if fine-tuning will close the gap. *(Opus, Kimi K2)*

**4. Is node0 actually suitable for fine-tuning, or is it pretraining-only infrastructure that requires substantial rearchitecting?**
The context.md explicitly describes node0 as pretraining infrastructure. Fine-tuning has different dynamics (lower LR, fewer steps, different regularization, data parallelism vs. pipeline parallelism). The gap between "what node0 does" and "what fine-tuning needs" must be assessed before any scope is committed. *(Opus, Kimi K2)*

**5. What training data format is required — instruction-response pairs, tool-use traces, preference pairs (DPO), or multi-turn conversation?**
Formulas, session transcripts, and bead patterns are structurally different artifacts. Without specifying the target format, corpus preparation cannot begin. This is the first concrete deliverable of the corpus design work. *(Opus, Kimi K2)*

**6. Does Gas Town currently generate sufficient high-quality training data, or is this primarily a data scarcity problem that infrastructure cannot solve?**
If fewer than 1,000 high-quality Gas Town interaction traces exist, fine-tuning will not work regardless of the infrastructure. The corpus-first question must be answered before GPU hours are committed. *(Opus, Kimi K2)*

**7. What does "success" look like concretely — what specific tasks must the fine-tuned model perform correctly, at what accuracy threshold, and what is the comparison baseline?**
"Operate as a native Gas Town agent" is too vague to guide training, evaluation, or deployment decisions. We need a checklist: creates valid beads, runs formulas correctly, follows bead dependency conventions, respects resource constraints, uses correct issue prefixes, etc. *(Opus, Kimi K2)*

**8. Will the fine-tuned model replace the current Claude/pi/omp agents, augment them, or serve as a fallback — and how does this change the operator's daily workflow?**
Users will resist or abandon this feature if the replacement vs. augmentation question is unresolved. This also determines whether the training objective is behavioral specialization, capability extension, or cost reduction. *(Opus, Kimi K2)*

**9. How does the training corpus handle credentials, API keys, file paths, and other sensitive data that appear in session transcripts?**
Session transcripts may contain HF tokens, RSA key references, database credentials, or personal information. Data sanitization is a hard prerequisite for corpus collection; without a policy, collection cannot begin. *(Opus, Kimi K2)*

**10. Will the fine-tuning job coexist with normal Gas Town operations on the current 10GB box (Dolt at 550MB, openclaw-gateway at 1GB, claude sessions at 300-750MB each), or does it require exclusive hardware?**
This is a go/no-go for the current hardware target. If fine-tuning requires exclusive access, it is a non-starter for daily Gas Town operators on the current rig. Memory budget planning must happen before any design work. *(Opus, Kimi K2)*

**11. Is LoRA/QLoRA adapter training, full fine-tuning, or instruction tuning the intended approach, and does the choice align with node0's gradient-averaging architecture?**
These approaches have fundamentally different compute budgets, data requirements, and outcome profiles. Node0 uses Hivemind-based gradient averaging, which is designed for full-model training. Adapters may not integrate cleanly. The choice blocks all downstream infrastructure decisions. *(Opus, Kimi K2)*

**12. What is the deployment target for the fine-tuned model — local inference on the constrained box, external API serving, or a separate inference stack?**
A LLaMA 8B model is 16GB+ in fp16, exceeding the 10GB box entirely. This question determines whether the feature is even feasible on current hardware, and what quantization, format, and serving infrastructure decisions are required. *(Opus, Kimi K2)*

**13. How do we handle model staleness as Gas Town conventions evolve — is there a continuous retraining story, or does the model become a liability over time?**
Gas Town adds new formulas, rigs, CLI commands, and conventions continuously. A model trained today will diverge from current practice within weeks. Without a freshness strategy, the fine-tuned model will eventually give confidently wrong advice. *(Opus, Kimi K2)*

**14. Where does node0's data loading gap get filled — the context.md notes node0 does NOT include data loading (gradients come from external training loops). What is the plan for the training data pipeline?**
This is a concrete infrastructure gap identified in the existing codebase. Corpus preparation and training cannot be connected without solving this. It may require building a significant new component. *(Opus)*

---

## P1 — Should Answer

*High-impact questions raised with strong reasoning by one or both models. These shape corpus design, UX flows, and architectural choices.*

**15. Is the distributed training aspect actually necessary for the current scope, or does it add complexity without complication benefit — could LoRA fine-tuning on a single machine serve the immediate use case?**
An 8B model with LoRA can be fine-tuned on a single 24GB GPU in hours. The distributed infrastructure adds months of complexity. If distribution is aspirational rather than required, the MVP scope should be defined without it. *(Opus)*

**16. Should there be role-specific fine-tuned models (mayor / polecat / witness) or a single model for all Gas Town roles?**
Different roles have different responsibilities: a mayor orchestrates, a polecat executes narrow tasks, a witness validates. One model optimized for all roles may underperform role-specific ones. But role-specific models multiply maintenance burden. The answer determines corpus segmentation strategy. *(Opus, Kimi K2)*

**17. What is the relationship between fine-tuning and the existing `gt prime` context injection — if `gt prime` already injects Gas Town conventions into the model's context window, what does fine-tuning add beyond that?**
If context injection already solves the "model doesn't know Gas Town" problem adequately, the value proposition for fine-tuning is unclear. The answer should explicitly address what prompting cannot achieve that fine-tuning can. *(Opus, Kimi K2)*

**18. How does the fine-tuned model integrate with Gas Town's existing agent provider system — how does it register as a provider, how does it receive dispatch, and how does it interact with session management and polecat lifecycle?**
Fine-tuning produces a model, but Gas Town's dispatch layer currently maps roles to Claude, pi, and omp. A new provider needs hooks into all of these systems. This is a non-trivial integration task that must be designed before training completes. *(Opus)*

**19. What is the minimum viable corpus for a first training run — how many examples, from which sources, in what ratio?**
Research suggests instruction tuning needs 1K–100K high-quality examples. Tool-use fine-tuning may need structured function-call traces. Without a data budget estimate, the corpus preparation phase has no completion criterion. *(Opus, Kimi K2)*

**20. How does the corpus handle multi-turn Gas Town sessions — truncation, summarization, or segmentation — given that sessions can be thousands of turns and transformer context windows are finite?**
Each choice (truncate at N tokens, summarize midway, segment into chunks) introduces different training artifacts. The wrong choice can teach the model that context is always brief, or that sessions always end at turn 512. *(Opus)*

**21. Should fine-tuning configuration be expressible as a Gas Town formula, enabling reproducible and shareable training recipes?**
Formulas are Gas Town's native abstraction for repeatable workflows. Training configs that are not formulas will feel foreign to operators and will not benefit from the formula search path, versioning, or sharing mechanics already in place. *(Opus)*

**22. Can an operator use the fine-tuned model for some tasks and the foundation model for others (hybrid usage), or is it all-or-nothing agent replacement?**
Hybrid usage — fine-tuned model for Gas Town tasks, Claude for general coding — is the most likely real-world adoption pattern. If this is not supported, operators will not switch at all because they cannot accept degraded general capability. *(Opus)*

**23. What are the catastrophic forgetting risks for fine-tuning on Gas Town-specific data, and what base capabilities must be preserved during training?**
Aggressive fine-tuning on narrow domain data can destroy the base model's reasoning, instruction following, and general coding ability. A "do not forget" set must be defined, and a replay buffer or regularization strategy must be planned before training begins. *(Opus, Kimi K2)*

**24. How do we prevent model collapse from self-generated training data — if polecats generate code that gets merged and later included in the training corpus, we risk a feedback loop where model errors compound across generations?**
This is a documented failure mode (Shumailov et al., 2023). The corpus pipeline needs a provenance flag that distinguishes human-generated from model-generated content and excludes or down-weights synthetic data. *(Opus, Kimi K2)*

**25. What is the plan for checkpoint recovery — if a distributed training job crashes (OOM, node disconnect, network partition), can it resume from the last checkpoint, or does it restart from scratch?**
Long distributed training runs that cannot be resumed are unusable in practice. The answer determines both infrastructure requirements and the user experience of job management. *(Opus, Kimi K2)*

**26. How does the system handle contradictory corpus entries — old CLAUDE.md conventions conflicting with current ones, or two formulas that prescribe opposite approaches to the same task?**
The model learns confused behaviors from contradictory data. Conflict detection during corpus validation is essential, even if automatic resolution is not possible. *(Opus, Kimi K2)*

**27. How does the operator evaluate the fine-tuned model before deploying it as an agent — is there a sandbox, an evaluation harness, or must they deploy to test?**
Deploying an untested model risks breaking live workflows. A pre-deployment evaluation path (even a simple checklist of Gas Town tasks the model must pass) is required for operator confidence. *(Opus, Kimi K2)*

**28. Should training jobs appear as beads in the bead system, following standard bead lifecycle states (ready, active, review, done), for consistency with Gas Town's existing workflow metaphors?**
Using beads for training job tracking leverages existing mental models and tooling. It also makes training job status visible in the same tools operators already use daily. The alternative — a parallel state system — adds learning cost. *(Opus, Kimi K2)*

**29. What confirmation and resource-visibility gates should exist before launching a fine-tuning job — specifically, what information does the operator need to see before committing GPU hours?**
Accidental launches on a constrained box waste hours and can crash other services. A pre-launch summary showing corpus size, estimated time, estimated memory consumption, and current resource headroom is the minimum safe gate. *(Opus, Kimi K2)*

**30. What happens to an operator's contributed session transcripts if they later want them removed — is there a right-to-be-forgotten mechanism, or is the data baked into weights permanently?**
Data removal from trained weights is technically very hard (machine unlearning). Operators need to know this upfront, before contributing. A clear policy (no removal possible after training, or removal triggers a retraining run) must exist before corpus collection begins. *(Kimi K2)*

**31. How should the system surface corpus health before training — duplicates, stale formula versions, entries referencing deleted beads, imbalanced data types?**
Garbage in, garbage out. Pre-training corpus diagnostics prevent wasted GPU hours and give operators agency to improve their data before committing resources. This is a design-critical feature for the corpus preparation step. *(Opus, Kimi K2)*

**32. What tokenizer implications arise from Gas Town's domain-specific identifiers (bead IDs like st-btod, bd-9f1f; rig names; formula names; gt commands) on a LLaMA-family vocabulary?**
These identifiers may tokenize into 3–5 fragments each, wasting context window and degrading model performance on Gas Town-specific syntax. Tokenizer extension is non-trivial with pretrained models and must be evaluated early. *(Opus)*

**33. Is RLHF, DPO, or preference-based fine-tuning in scope — and is "correct Gas Town workflow execution" a usable preference signal?**
Agent behavior alignment benefits from preference data, not just instruction following. Constitutional AI and DPO approaches use human feedback signals. If Gas Town has objective correctness criteria (bead created correctly, formula ran without error), these can serve as automated preference signals. *(Opus, Kimi K2)*

**34. How will we detect and measure catastrophic failure in production — specifically, a fine-tuned agent that follows commands without checking for destructive operations (force push to main, rm -rf, nuke without verification)?**
The MEMORY.md documents known Gas Town footguns. A fine-tuned model that does not inherit safety intuitions from the base model is dangerous. Safety evaluation must be a first-class success criterion, not an afterthought. *(Opus)*

**35. What is the corpus scope for distributed training — individual rig data, all Gas Town rigs, or a curated multi-rig blend — and who controls the boundary?**
Rig-specific data produces a model with specific personality and blind spots. Cross-rig data produces a more general model but may introduce convention contamination (learning sfgastown patterns and applying them in frankentui). Scope has both technical and governance dimensions. *(Opus, Kimi K2)*

**36. What is the expected iteration cycle time — from new training data to deployed model — and is that fast enough for practical improvement?**
If a full retraining cycle takes weeks, the feedback loop is too slow for active Gas Town development. The target cycle time determines whether distributed training infrastructure is justified or whether faster single-machine approaches should be preferred. *(Opus, Kimi K2)*

**37. Should the system schedule recurring training automatically as new corpus data accumulates, and if so, what guardrails prevent runaway resource consumption?**
Continuous fine-tuning as Gas Town evolves is the correct long-term architecture, but automated retraining without resource limits and quality gates on a 10GB box would be catastrophic. *(Kimi K2)*

**38. Are there existing Gas Town agent implementations (session transcripts from Claude, pi, omp) that can be studied to understand implicit conventions not captured in documentation?**
Prior implementations reveal patterns that documentation omits. This is the fastest way to bootstrap corpus annotation guidelines and to identify gaps between documented and actual agent behavior. *(Kimi K2)*

**39. What is the A/B testing strategy for validating a new fine-tuned model in production — controlled rollout, shadow mode, or operator-selected?**
Production validation of agent behavior requires controlled comparison. Without a rollout strategy, operators either block all deployment (overly cautious) or deploy blindly (reckless). *(Kimi K2)*

**40. Will the fine-tuned model understand Dolt layer conventions — issue prefixes, database names, bead state transitions — or just surface-level CLI usage patterns?**
Beads and Dolt are the foundational data layer of Gas Town. Shallow understanding of bead state (ready, active, review, done) will lead to agents that talk about beads without being able to correctly navigate their lifecycle. *(Opus)*

---

## P2 — Good to Have

*Design-relevant questions worth considering during detailed design, but not blockers for proceeding to initial design decisions.*

**41. What is the minimum viable operator profile — does someone need ML expertise, or can a Gas Town operator with no ML background use this feature?**
The answer determines where the default abstraction level sits: expose gradient averaging and learning rate schedules, or hide everything behind sensible defaults. *(Opus, Kimi K2)*

**42. Should operators without GPU access be able to participate in corpus preparation even if they cannot run training?**
Separating corpus contribution from training execution widens participation. Corpus-only contributors may provide the highest quality data. *(Opus, Kimi K2)*

**43. Can an operator contribute their own session transcripts to the corpus and expect to see improvement in subsequent model versions (continuous contribution loop)?**
Users expect their contributions to matter. A one-shot training model where contributions go nowhere after the initial run feels dead. *(Kimi K2)*

**44. What does the operator do if the fine-tuned model produces a bad bead or breaks a workflow — is there a rollback mechanism, or do they just switch back to the foundation model?**
The answer determines how much trust an operator needs before trying the fine-tuned model. Easy rollback makes adoption low-risk; no rollback makes it a high-stakes commitment. *(Opus, Kimi K2)*

**45. How does the fine-tuned model handle bead dependency graphs — does it understand relationships between beads (blockers, epics, dependencies) or treat beads as isolated documents?**
Gas Town's value is in the graph of work. A model that creates beads without understanding dependency semantics will generate inconsistent or broken project structures. *(Kimi K2)*

**46. What format should evaluation results be presented in — loss curves, task success rates, or plain-language summaries like "model correctly generates bead workflows 87% of the time"?**
Loss curves are meaningless to non-ML operators. The evaluation interface must translate metrics into language that operators can act on. *(Opus, Kimi K2)*

**47. How does the system handle cross-rig data — if the corpus includes transcripts from sfgastown, frankencord, frankentui, and bv, does the model learn rig-specific patterns and misapply them?**
Cross-rig behavioral contamination is a real risk. The model may learn that bv uses JSONL-only reads and apply that pattern in sfgastown where Dolt is expected. *(Opus)*

**48. Will the fine-tuned model understand formula authoring conventions — TOML structure, version bumps, search paths, embedded vs. provisioned formulas?**
Formulas are a core Gas Town abstraction. A model that cannot help write or debug formulas misses a major use case for Gas Town operator assistance. *(Opus)*

**49. What progress feedback does an operator receive during a distributed training run — polling a dashboard, push notifications, bead status updates, or mail?**
The monitoring mechanism should integrate with Gas Town's existing notification patterns rather than inventing a new one. Progress indicators on long-running distributed jobs are essential for operator patience. *(Opus, Kimi K2)*

**50. What happens if an operator feeds in a corpus that is too small — does the system fail gracefully with guidance, or produce confidently wrong Gas Town syntax?**
Small-corpus training produces overfit models that appear to work but fail on any novel input. A minimum corpus size gate with actionable guidance is better than a silent quality degradation. *(Opus, Kimi K2)*

**51. How should the corpus browser be laid out, and what filtering / selection mechanism determines what goes into a training run?**
The selection mechanism determines corpus quality. A bad default selection (include everything) poisons every downstream model. Good defaults with explicit opt-out are safer than explicit opt-in that most operators will skip. *(Opus, Kimi K2)*

**52. Should fine-tuning infrastructure live in its own rig (e.g., `finetune` with prefix `ft-`) or as a workflow within node0?**
A dedicated rig provides clean isolation and its own bead namespace. Embedding it in node0 reduces cognitive overhead but muddies the boundaries between infrastructure and training workflow. *(Opus)*

**53. What is the "paused" state story for long-running training jobs — can a job be suspended to free memory for other Gas Town operations without losing progress?**
On a 10GB box, memory is a planning constraint. A pause/resume capability could make the feature coexist with normal operations. Without it, training and operations compete destructively. *(Opus, Kimi K2)*

**54. What visual or conceptual treatment distinguishes base models, fine-tuned variants, checkpoint versions, and deployed models?**
Without clear visual distinction, operators will confuse model versions, compare incompatible states, or deploy the wrong variant. Model versioning conventions need to be designed before the first model is deployed. *(Opus, Kimi K2)*

**55. How long do completed or failed training jobs persist before being archived or garbage-collected?**
Indefinite retention clutters the interface. Aggressive cleanup loses history needed for comparing runs. A retention policy must be designed, especially given storage constraints on the 10GB box. *(Opus)*

**56. What canonical states does a fine-tuning job move through, and should these map to existing bead states?**
A parallel state vocabulary that operators must learn separately adds cognitive overhead. If bead states can model the job lifecycle cleanly, consistency is strongly preferred. *(Opus, Kimi K2)*

**57. Should training configuration be a form, a YAML editor, or a formula template — and what are the tradeoffs for the Gas Town operator audience?**
Forms are approachable but inflexible. YAML editors are powerful but error-prone. Formula templates are Gas Town-native but require formula literacy. The choice must account for the actual operator audience, not a hypothetical one. *(Opus, Kimi K2)*

**58. How do we handle "training succeeded but evaluation shows the model is worse" — what guidance does the system provide?**
"Your model is worse" without explanation is demoralizing and useless. The system should suggest corpus improvements (more data of type X, remove conflicting entries) or config changes (reduce learning rate, add regularization). *(Opus, Kimi K2)*

**59. How does the system represent distributed node health — peer count, gradient sync status, network throughput — and should this be top-level or in a collapsible panel?**
Most operators care about "is it working?", not P2P topology. But operators debugging slowness need the details. A collapsible advanced panel balances both needs. *(Opus, Kimi K2)*

**60. What does the deployment step look like operationally — swapping a model reference in town settings, editing agent config, or something more involved?**
The gap between "model is trained" and "model is usable as a Gas Town agent" is where most fine-tuning projects die. Deployment should feel like flipping a switch, not editing five config files. *(Opus, Kimi K2)*

**61. Will operators using different hardware (AMD GPUs, Apple Silicon, cloud instances with different CUDA versions) be able to participate, given that the Dockerfile pins CUDA 12.1.1?**
The CUDA pin excludes a significant portion of potential contributors. Documenting hardware requirements explicitly before launch prevents wasted setup time. *(Opus)*

**62. How should stale or outdated corpus entries be flagged — formulas deprecated since last training, beads referencing closed issues, workflows that no longer match current CLI?**
Training on deprecated patterns teaches the model bad habits. Corpus freshness validation must be part of the pre-training review step. *(Opus, Kimi K2)*

**63. What does the transition from training to evaluation look like — is it automatic, or does the operator trigger it?**
Automatic transitions reduce friction but remove the operator's ability to inspect before evaluation begins. Operator-triggered transitions add a step but enable intervention. *(Opus, Kimi K2)*

**64. How should the system handle an operator who wants to fine-tune on only a specific subset — only beads patterns, only formula authoring, only bead-creation workflows?**
Task-specific fine-tuning is a reasonable and common request. Forcing the full corpus when a subset would suffice wastes resources and dilutes specialization. *(Opus)*

**65. What is the expected model response latency on local hardware, and is it acceptable for interactive agent use compared to current API-served frontier models?**
A locally-served fine-tuned model will have different latency characteristics than Claude Opus API calls. If the latency is 10x worse, operators will not use it interactively. *(Opus, Kimi K2)*

**66. What does "export model" look like for portability — what format, what quantization level, and what inference runtimes (vLLM, llama.cpp) is the output compatible with?**
Checkpoint format compatibility is a known pain point. Operators will expect standard formats (GGUF, safetensors) and clear export tooling, not a proprietary checkpoint that only works with node0. *(Opus, Kimi K2)*

**67. How do we handle operators on slow or intermittent internet connections — can they participate in distributed training, or does P2P coordination require reliable high-bandwidth connectivity?**
Node0 requires port forwarding and P2P communication. Operators behind restrictive firewalls or on cellular connections may be excluded from training participation but could still contribute to corpus preparation. *(Opus, Kimi K2)*

**68. What are the known issues with Hivemind at scale, and what failure modes from decentralized training have been documented in prior node0 runs?**
Infrastructure limitations constrain training corpus size and determine acceptable node count. Prior failure modes inform corpus design and infrastructure hardening priorities. *(Kimi K2)*

**69. How do we handle "training is taking 10x longer than estimated" — what honest communication reaches the operator without creating panic?**
Distributed training duration estimates are inherently unreliable. When reality diverges significantly from estimates, the system should provide updated estimates, not just show an indeterminate spinner. *(Kimi K2)*

**70. Should there be a "model registry" view listing all fine-tuned models across rigs — their base model, training date, corpus composition, and current deployment status?**
As operators create multiple fine-tuned variants across iterations, they need a single place to find, compare, and manage them. Without a registry, model management becomes a manual file-tracking exercise. *(Opus)*

**71. How does the system handle cross-rig data access permissions — if an operator wants to fine-tune on data from a rig they do not own, what boundaries apply?**
Cross-rig data access raises both practical concerns (prefix mismatches, convention differences) and trust concerns (data governance, content ownership). *(Opus)*

**72. What does a "model card" for a Gas Town fine-tuned model include — corpus composition, training config, evaluation results, known limitations, and recommended use cases?**
Model cards are the standard mechanism for communicating what a model is good at and where it fails. Without this, operators will deploy the model into situations it is unprepared for. *(Kimi K2)*

**73. Should state transitions in the training job lifecycle generate beads, mail notifications, or event feed entries — and at what verbosity level?**
Transitions are significant events that operators should not miss. But noisy notifications cause alert fatigue. Configurable notification verbosity with sensible defaults is required. *(Opus)*

**74. How does the system detect and report overfitting — specifically, when the model has learned to reproduce a specific rig's quirks but fails on standard Gas Town patterns?**
Overfitting is invisible without validation metrics. Operators need objective signals that their model is too narrow before they deploy it. *(Kimi K2)*

**75. Can the fine-tuned model be used for Gas Town planning tasks — generating draft beads for projects that do not yet exist, or mapping out dependencies for a new epic?**
Users naturally repurpose tools. Planning assistance is a high-value adjacent use case. Whether it is explicitly supported, implicitly supported, or actively discouraged should be defined. *(Kimi K2)*

**76. How does the fine-tuned model signal uncertainty about Gas Town-specific topics — does it hallucinate bead structures confidently, or indicate when it is operating outside its training distribution?**
Confident wrong answers about Gas Town conventions are worse than admitted uncertainty. Calibration on domain-specific content is a key evaluation criterion. *(Kimi K2)*

---

## P3 — Parking Lot

*Lower priority, or questions that are appropriately deferred until core scope is resolved. Includes technical questions that belong in implementation design, not scope definition.*

**77. Should training jobs appear in the main beads view or in a dedicated "fine-tuning" navigation section, and how much top-level navigation real estate does this feature deserve?**
Navigation placement is a detailed design decision that depends on how frequently the feature is used. Cannot be resolved until P0 scope questions are answered. *(Kimi K2)*

**78. Should fine-tuning runs be nameable, or should names be auto-generated from corpus hash and timestamp?**
Auto-names reduce friction; meaningful names help comparison. This is a micro-UX decision that depends on how often operators run fine-tuning. *(Kimi K2)*

**79. What keyboard shortcuts or quick actions should exist for power users who run fine-tuning frequently?**
Relevant only after the feature is validated with first-time users. Power user optimization is V2. *(Opus)*

**80. What density level should the training dashboard default to — compact for experienced Gas Town operators, or spacious for newcomers?**
Given Gas Town's power-user audience, compact is likely correct, but this is a detailed design decision. *(Opus)*

**81. How do we visualize distributed nodes — map, list, health-dot grid, or abstract representation?**
Geographic accuracy is probably irrelevant; health and contribution state matter. The right visualization depends on how many nodes are expected. This is a detail design question. *(Kimi K2)*

**82. Should corpus composition be visualized as a pie chart, a stacked bar, a list, or a tree?**
Visual encoding choice for a secondary informational screen. Depends on what "composition" dimensions are tracked. *(Kimi K2)*

**83. What retention period should completed and failed training jobs have before garbage collection?**
Retention policy depends on storage availability and operator workflow patterns. Cannot be determined until the storage budget for the feature is known. *(Opus)*

**84. Should there be a "fine-tuning dashboard" as a distinct screen, or should training job status be integrated into the existing beads and agent views?**
Aggregation vs. integration is a layout decision. Depends on whether fine-tuning is a frequent or occasional activity. *(Kimi K2)*

**85. What does "compare two training runs" look like visually — side-by-side, overlay, tabular diff?**
Comparison tooling design is V2 work. The first version needs a single-run view that works well. *(Opus, Kimi K2)*

**86. What is the PowerSGD compression configuration and does it interact differently with fine-tuning gradients than pretraining gradients?**
Technical infrastructure question. Relevant to implementation but not to scope definition. *(Kimi K2)*

**87. What is the relationship between node0's pipeline "experts" and Gas Town's "agents" — could the expert specialization model be used for role-specific routing?**
Interesting architectural concept but deferred until the fundamental question of single vs. role-specific models is resolved (Q16). *(Opus)*

**88. What multilingual Gas Town usage exists, and does the corpus need to cover non-English interactions?**
Depends on whether Gas Town has international users. Likely English-only for initial scope. *(Opus, Kimi K2)*

**89. Are there accessibility requirements for the training monitoring interface — screen reader compatibility, structured log output as an alternative to visual dashboards?**
Accessibility is important but this is not a core operator tool; it is an infrastructure management interface. Structured log output is the minimum viable accessibility target. *(Opus, Kimi K2)*

**90. Can an operator use this feature to generate training data for onboarding new Gas Town users, rather than just training automated agents?**
Interesting dual-use case. Out of scope for initial design but worth noting for V2 planning. *(Kimi K2)*

**91. What does "curriculum learning" look like for Gas Town fine-tuning — should the model learn general instruction following before Gas Town specifics, or mixed?**
Curriculum design is an implementation concern. It depends on base model capability and is better addressed during training experiment design than scope definition. *(Opus)*

**92. What replay buffer strategy prevents catastrophic forgetting during training?**
Implementation detail that follows from the base model choice and training approach. Belongs in technical design, not scope. *(Kimi K2)*

**93. How do we ensure reproducibility in dynamic peer join/leave training environments?**
Distributed systems reproducibility is inherently hard. This is an implementation constraint, not a scope question. *(Opus)*

**94. What is the Byzantine fault tolerance posture for gradient poisoning in decentralized fine-tuning?**
Security question for implementation phase. The relevant scope question (Q9 covers data sanitization as a prerequisite) is already captured at P0. *(Opus)*

**95. Can an operator schedule recurring fine-tuning runs on a cron-like basis?**
Scheduling automation is V2 functionality. First version needs manual-trigger training that works reliably. *(Kimi K2)*

**96. What does "retrain from checkpoint" look like vs. "start fresh" — are these presented as distinct flows or a single flow with a checkpoint option?**
UI detail that depends on how checkpointing is implemented. *(Kimi K2)*

**97. What is the "select corpus" interface for scoping a training run to a subset of available data — filter by type, date range, rig, bead tags?**
Interface design for corpus selection. Depends on corpus metadata schema, which is a P1 question. Cannot be designed until corpus format is specified. *(Kimi K2)*

---

## Cross-Model Agreement Summary

The following themes represent highest-confidence findings — both models converged independently:

- **"Gas Town agent" is undefined.** Neither model found a measurable definition of what the fine-tuned model should be able to do. This single gap cascades into every other scope, corpus, and infrastructure decision. Both models flagged it as the root blocker.

- **Data quality and corpus sanitization are the hardest unsolved problems.** Both models identified that raw session transcripts contain errors, mistakes, contradictions, credentials, and synthetic content. Neither the brief nor the existing infrastructure addresses how to clean, deduplicate, filter, or govern this data. Both estimated corpus preparation at 70-80% of total project effort.

- **The 10GB box is a hard constraint that the feature must respect, not ignore.** Both models independently calculated that a LLaMA 8B model in fp16 alone exceeds the box's total memory. Both flagged the coexistence problem with Dolt, openclaw-gateway, and live Claude sessions. Both required a concrete hardware and memory budget plan before any other design work.

- **Model staleness is a systemic risk.** Both models noted that Gas Town is under active development. A model trained today on current conventions will drift from practice within weeks. Neither the brief nor the infrastructure has a story for keeping the model current without manual retraining cycles.

- **The replacement vs. augmentation question must be answered first.** Both models flagged that user adoption depends entirely on clarity here. If operators expect a drop-in replacement for Claude Opus and get something weaker, they abandon the feature. If they understand it as a complement, expectations are calibrated correctly.

- **Fine-tuning's value proposition over prompting + RAG is not established.** Both models independently raised the question of whether `gt prime` context injection already solves the core problem. Neither the brief nor the existing infrastructure demonstrates that fine-tuning will outperform well-prompted frontier models on Gas Town tasks.

- **Evaluation infrastructure is absent.** Both models noted that without a held-out test set, objective metrics, and human evaluation protocol, there is no way to know if fine-tuning worked. Both flagged this as a blocker for making any training decisions.

---

## Cross-Model Divergence

**Where Opus was stronger:**
- Infrastructure skepticism: Opus explicitly questioned whether distributed fine-tuning is necessary at all for the immediate use case (single-machine LoRA may suffice). This is the most actionable cost-reduction insight in the analysis.
- Data pipeline gap: Opus specifically identified the node0 data loading gap as a concrete blocker from the codebase. Kimi treated data pipeline as a design question; Opus identified it as an implementation gap.
- Tokenizer efficiency: Opus raised Gas Town identifier tokenization as a concrete technical problem with a specific impact (3-5 tokens per bead ID). Kimi did not address this.
- Model collapse from self-referential training: Opus cited Shumailov et al. explicitly and traced the failure mode from polecat output → merge → corpus inclusion.
- Safety evaluation: Opus specifically connected Gas Town's known footguns (from MEMORY.md) to the safety evaluation requirement for the fine-tuned model.

**Where Kimi was stronger:**
- Operator emotional dynamics: Kimi gave more attention to operators feeling in control vs. displaced, and to the witness role as an underserved use case.
- Right-to-be-forgotten: Kimi raised this as a concrete legal/ethical requirement. Opus treated data inclusion as a one-way contribution. This is a genuine gap Opus missed.
- Online learning and feedback loops: Kimi pushed harder on the "model should improve from production feedback" pattern. Opus treated evaluation as a one-time gate.
- A/B testing in production: Kimi explicitly asked for a controlled rollout strategy. Opus stopped at evaluation before deployment.
- Scope clarity as a prerequisite: Kimi's "Critical Unknowns" section concisely surfaced the five blockers that gate all other decisions — a useful prioritization frame the Opus analysis spread across 62 domain expert questions.

**Genuine disagreements:**
- Opus was skeptical of distributed training necessity; Kimi accepted it and focused on making it work. Opus's skepticism is more valuable for scoping.
- Opus treated corpus contribution as primarily a burden operators want automated away. Kimi treated it as an engagement mechanism where operators want to see their contributions reflected — a more nuanced view for adoption design.

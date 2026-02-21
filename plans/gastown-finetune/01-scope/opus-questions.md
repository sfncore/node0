# Opus 4.6 Analysis: gastown-finetune

## User Advocate Perspective

**Feature:** Set up node0 distributed fine-tuning infrastructure and prepare a Gas Town training corpus (formulas, workflows, agent conventions, beads patterns, session transcripts) to fine-tune an LLM that can operate as a native Gas Town agent.

**Perspective:** Gas Town operators (mayors, polecats, witnesses) who will interact with and rely on this fine-tuned model daily.

---

### 1. User Expectations

**Q1: What does "native Gas Town agent" actually mean in practice — does it replace the current Claude/Pi/omp agents, augment them, or serve as a fallback?**
Users need to know whether this is a replacement or a supplement; if they expect a drop-in replacement for Claude Opus and get something weaker, they will be deeply frustrated.

**Q2: Will the fine-tuned model understand `gt` CLI commands, bead workflows, and formula conventions out of the box, or does it need explicit prompting each time?**
The whole point of fine-tuning is eliminating the need to constantly re-explain Gas Town conventions; if users still have to provide lengthy context, the feature has failed.

**Q3: How does the model's quality compare to the foundation models (Claude Opus 4.6, Gemini 3) that operators currently use — will it be noticeably worse at general coding tasks?**
Operators use agents for real code changes; if the fine-tuned model is Gas Town-fluent but poor at writing Go/Python/Rust, they will abandon it immediately.

**Q4: Can operators expect the model to correctly handle multi-rig workflows (sfgastown, frankentui, bv, pi_agent) or will it only understand a single rig's patterns?**
Gas Town operators work across rigs constantly; a model that only knows one rig's conventions is only partially useful.

**Q5: Will the fine-tuned model understand the Dolt database layer — issue prefixes, database names, bead state transitions — or just surface-level CLI usage?**
Beads and Dolt are foundational to Gas Town; shallow understanding will lead to broken workflows and operator distrust.

**Q6: What does "preparing a training corpus" mean for operators — do they need to manually curate data, or is it automated from existing session transcripts and formulas?**
If operators are expected to spend hours curating training data by hand, adoption will be extremely low; this needs to be mostly automated.

**Q7: Users who have worked with LoRA, QLoRA, or other fine-tuning approaches elsewhere will expect to specify rank, target modules, and quantization — will this system expose those controls or abstract them away?**
Experienced ML operators will be confused if they cannot tune hyperparameters; inexperienced operators will be overwhelmed if they must.

**Q8: How quickly can operators iterate — if the model gets something wrong about Gas Town conventions, how long does a retraining cycle take?**
If retraining takes days on distributed commodity hardware, the feedback loop is too slow for practical use; operators will revert to prompting foundation models.

**Q9: Will the model understand Gas Town's resource constraints (10GB box, memory budgets, staggered slings) and factor those into its recommendations?**
Operators currently rely on agents that understand their constrained environment; a model that suggests unbounded parallelism is dangerous.

**Q10: Do operators expect the model to handle Gas Town's security model — RSA keys, HF tokens, Pluralis auth — or is that out of scope?**
If the model suggests commands that bypass or break the auth layer, operators will lose trust fast.

**Q11: Will the fine-tuned model be aware of operator roles (mayor, polecat, witness, crew) and adjust its behavior accordingly, or does it treat all operators identically?**
Gas Town has a role hierarchy that affects what actions are appropriate; a model that ignores roles will suggest inappropriate actions to polecats or underserve mayors.

**Q12: Operators familiar with OpenAI's fine-tuning API or Hugging Face's Trainer will expect a similar level of polish — structured data format, validation, progress tracking — does this deliver that?**
Setting expectations correctly matters; if this is more like a research prototype than a production tool, operators need to know upfront.

**Q13: Will the model understand formula authoring conventions — TOML structure, version bumps, search paths, embedded vs. provisioned formulas?**
Formulas are a core Gas Town abstraction; if the model cannot help write or debug formulas, a major use case is missed.

### 2. User Journey

**Q14: What is the first thing an operator does to start using this feature — is there a `gt` command, a config change, or a manual multi-step process?**
If the onboarding path is unclear or involves too many manual steps, operators will bounce before they even try it.

**Q15: Where does "prepare a training corpus" fit in the operator's day — is it a one-time setup task, a periodic maintenance chore, or something that happens continuously in the background?**
Operators are typically rushed and task-focused; if corpus preparation requires dedicated attention, it needs to be clearly scoped.

**Q16: How does an operator know the fine-tuning is working — is there a dashboard, log output, or bead-based progress tracking?**
Distributed training is opaque by nature; without clear progress indicators, operators will feel lost and anxious about whether resources are being wasted.

**Q17: What happens when an operator starts a fine-tuning job and then needs to do other work — does it consume the box's resources and block normal Gas Town operations?**
On a 10GB box where memory is already a planning constraint, a fine-tuning job competing with polecats and Dolt could crash everything.

**Q18: After the model is trained, how does an operator deploy it — swap it into the agent config, run it locally, or point to an external endpoint?**
The gap between "model is trained" and "model is usable as a Gas Town agent" is where most fine-tuning projects die.

**Q19: What emotional state is the operator in when they reach for this feature — are they frustrated with current models not understanding Gas Town, or proactively investing in tooling?**
If this is frustration-driven, the feature needs to deliver fast visible improvement; if proactive, it can afford a longer ramp.

**Q20: What does the operator do if the fine-tuned model produces a bad bead or breaks a workflow — is there a rollback mechanism or do they just switch back to the foundation model?**
Operators need confidence that trying the fine-tuned model is low-risk; without easy rollback, they will not adopt it.

**Q21: Can an operator use the fine-tuned model for some tasks and the foundation model for others, or is it all-or-nothing?**
Hybrid usage (fine-tuned for Gas Town tasks, Claude for general coding) is the most likely real-world pattern and needs to be supported.

**Q22: How does the training corpus stay fresh — when new formulas are added, new rigs configured, or conventions change, does the corpus auto-update or does someone have to remember to retrain?**
Gas Town evolves constantly (new rigs, new formulas, new agent configs); a stale corpus produces a model that gives outdated advice.

**Q23: What happens before this feature exists — how are operators currently teaching models about Gas Town, and how does this improve on that workflow?**
If `gt prime` and CLAUDE.md already provide sufficient context injection, operators need to understand what fine-tuning adds beyond that.

**Q24: Is there a way for operators to validate the corpus before training — preview what the model will learn, catch errors, remove sensitive data?**
Operators will want to inspect and trust the training data, especially if it includes session transcripts that might contain credentials or personal information.

### 3. Edge Cases (User Behavior)

**Q25: What happens if an operator fine-tunes on a small, unrepresentative corpus — say, only formulas from one rig?**
The model might overfit to one rig's patterns and give confidently wrong answers for other rigs, which is worse than no fine-tuning at all.

**Q26: What if an operator tries to fine-tune a model that is too large for their hardware (e.g., LLaMA 8B on a 10GB box with 16GB GPU)?**
The system should fail gracefully with clear guidance rather than OOM-killing other Gas Town processes.

**Q27: What if the training corpus contains contradictory information — e.g., old CLAUDE.md conventions that conflict with current ones?**
The model will learn confused behaviors; there needs to be a way to deduplicate and resolve conflicts in the corpus.

**Q28: What if an operator includes session transcripts where the agent made mistakes — does the model learn the mistakes too?**
Naive inclusion of all transcripts, including failed sessions, will teach the model bad patterns; curation or quality filtering matters.

**Q29: What if an operator changes their mind mid-training and wants to stop — can they cancel cleanly, or does partial training leave artifacts?**
Distributed training that cannot be cleanly interrupted will leave orphan processes, wasted GPU hours, and confused node states.

**Q30: What if multiple operators on different nodes are fine-tuning different models simultaneously — does the system handle this or conflict?**
The node0 infrastructure uses shared DHT and peer discovery; concurrent fine-tuning jobs could interfere if not properly isolated.

**Q31: What if an operator feeds in a corpus that is mostly Gas Town conventions but includes some proprietary code or credentials by accident?**
The model will memorize and potentially regurgitate sensitive information; corpus sanitization is a safety requirement, not a nice-to-have.

**Q32: What happens if a node drops out mid-training (network issue, box crash, OOM kill) — does the training resume, restart, or silently produce a degraded model?**
Node0's design supports dynamic join/leave for pretraining, but fine-tuning convergence may be more sensitive to node churn.

**Q33: What if an operator tries to use the fine-tuned model as a polecat and it does not respect cgroup memory limits or resource constraints?**
A model that does not understand its execution environment will behave like a rogue process on a constrained box.

**Q34: What if an operator wants to fine-tune on a specific subset (only beads patterns, only formula authoring) rather than the full corpus?**
Task-specific fine-tuning is a common and reasonable request; forcing the full corpus when a subset would suffice wastes resources and dilutes specialization.

**Q35: What if the operator exports a checkpoint but the model format is incompatible with their inference runtime (vLLM, llama.cpp, etc.)?**
Checkpoint format compatibility is a known pain point in the fine-tuning ecosystem; operators will expect standard formats.

**Q36: What if an operator tries to resume training from a previous checkpoint with a different corpus?**
Curriculum learning and corpus evolution are real use cases; the system needs to handle (or clearly reject) checkpoint + new data combinations.

**Q37: What if the Gas Town conventions change significantly after fine-tuning — does the model's behavior drift from current practice?**
Model staleness is the fine-tuning version of documentation rot; there needs to be a story for keeping the model current.

**Q38: What if a node participating in distributed fine-tuning is malicious or corrupted — does it poison the model for everyone?**
Node0 has basic auth (Pluralis), but gradient poisoning in decentralized training is a known attack vector that operators should understand.

### 4. Accessibility & Inclusion

**Q39: Who is the minimum-viable operator for this feature — does someone need ML expertise, or can a Gas Town operator with no ML background use it?**
If this requires understanding of gradient averaging, PowerSGD compression, and learning rate schedules, most Gas Town operators are excluded.

**Q40: What about operators who only have CPU machines (no GPU) — can they contribute to corpus preparation even if they cannot run training?**
Corpus preparation and training are different tasks with different hardware requirements; separating them widens participation.

**Q41: Are we assuming all operators have fast, reliable internet for distributed training, or can operators on slow/unreliable connections participate?**
Node0 requires port forwarding and peer-to-peer communication; operators behind restrictive firewalls or on cellular connections are excluded.

**Q42: Is the documentation written for someone who understands both Gas Town AND distributed ML, or are there separate paths for each audience?**
An operator who knows Gas Town but not ML will need very different guidance than an ML engineer who does not know Gas Town.

**Q43: What about operators using different hardware — AMD GPUs, Apple Silicon, or cloud instances with different CUDA versions?**
The Dockerfile pins CUDA 12.1.1; operators with different hardware may be unable to participate without significant effort.

**Q44: Are we assuming operators have unlimited storage for training corpus, checkpoints, and model weights?**
On a 10GB box, a LLaMA 8B model alone is 16GB+ in fp16; storage constraints could make this feature impossible without external storage.

**Q45: What about operators who speak languages other than English — does the corpus include multilingual Gas Town usage, or is it English-only?**
If Gas Town is used by non-English speakers, an English-only corpus produces a model that does not serve them.

**Q46: Are we assuming operators understand what fine-tuning is and why they would want it, or does the feature need to educate them?**
Many competent Gas Town operators may not understand the difference between fine-tuning and prompting; the feature needs to justify itself.

**Q47: What about operators with accessibility needs — is the training monitoring interface (if any) screen-reader friendly, or is it all terminal-based visual output?**
Terminal-based progress bars and dashboards can be inaccessible; at minimum, structured log output should be available.

**Q48: Are we assuming operators can dedicate their machine exclusively to training, or must training coexist with normal Gas Town operations?**
Most operators have one machine running Gas Town; if training requires exclusive use, it is a non-starter for daily operators.

**Q49: What about operators who want to fine-tune for a domain Gas Town does not currently cover (e.g., a new rig type, a new language ecosystem)?**
The corpus preparation pipeline should be extensible, not hardcoded to current rigs and conventions.

**Q50: Do we assume all operators trust each other in the distributed training cluster, or do we need isolation between participants?**
Decentralized training with untrusted peers introduces model integrity risks that operators may not be aware of.

**Q51: What about operators who have used the system before and want to understand what changed between model versions?**
Model versioning and changelog (what corpus was used, what hyperparameters, what training duration) is essential for operators to trust updates.

**Q52: Are we assuming operators can evaluate model quality themselves, or do we need automated benchmarks against Gas Town tasks?**
Without objective evaluation (e.g., "correctly generates a bead workflow 87% of the time"), operators have no way to know if fine-tuning helped.

---

## Product Designer Perspective

This analysis examines the proposed fine-tuning system from a UX design standpoint: how users (Gas Town mayors, operators, and contributors) will interact with corpus preparation, training orchestration, and model deployment. The focus is on workflow clarity, information hierarchy, feedback loops, and the states a fine-tuning job moves through.

---

### 1. Information Architecture

**What information does the user need to see? What's the hierarchy? What can be hidden vs. must be visible?**

1. **What is the current state of my fine-tuning job at a glance (preparing, training, evaluating, deployed, failed)?** The job state is the single most important piece of information and must be visible without any clicks or scrolling.

2. **How large is my training corpus, and is it sufficient for the target task?** Users need to know whether their corpus meets minimum thresholds before they waste GPU hours on an underpowered dataset.

3. **What types of data are in my corpus (formulas, session transcripts, beads patterns, workflows)?** Corpus composition directly affects what the fine-tuned model will be good at; users need a breakdown by category to spot gaps.

4. **How much of my corpus has been validated vs. raw/unprocessed?** If half the corpus is still in raw form, training quality will suffer; this distinction must be surfaced prominently.

5. **Which Gas Town conventions and patterns are represented in the corpus, and which are missing?** A model fine-tuned without exposure to, say, bead dependency conventions will produce broken workflows; coverage gaps are critical information.

6. **What is the estimated training cost (GPU hours, wall-clock time, credits) for my current configuration?** Users need cost visibility before they commit to a run, not after it completes.

7. **Should corpus metadata (source rig, creation date, author, bead ID) be visible by default or only on drill-down?** Over-exposing provenance clutters the view, but hiding it entirely makes debugging impossible.

8. **What evaluation metrics should be shown, and which require explanation?** Loss curves mean nothing to non-ML users; the system must translate metrics into actionable language ("model is improving", "model is plateauing").

9. **How should the system surface the difference between a base model and a fine-tuned variant?** Users need to understand that fine-tuning produces a derivative, not a replacement, to set correct expectations.

10. **What historical context should be preserved — previous training runs, corpus versions, config snapshots?** Without run history, users cannot compare outcomes or understand what changed between iterations.

11. **Should node health (peer count, gradient sync status, network throughput) be top-level or tucked into an advanced panel?** Most users care about "is it working?" not the P2P topology, but operators debugging slowness need the details.

12. **How should stale or outdated corpus entries be flagged?** Formulas and workflows evolve; training on deprecated patterns teaches the model bad habits.

13. **What information should a contributor see when deciding whether to add their session transcript to the corpus?** Contributors need to understand what they are donating and how it will be used, or they will not participate.

14. **Should the system show which beads/formulas influenced a specific model behavior during evaluation?** Traceability from output back to training data builds trust and enables targeted corpus improvements.

### 2. Interaction Design

**How does the user trigger fine-tuning? What inputs are required vs. optional? What feedback indicates success/failure/progress?**

1. **How does a user initiate a fine-tuning run — a `gt` CLI command, a bead, a formula, or a dedicated UI screen?** The entry point determines discoverability; if it is buried, nobody will use it.

2. **What is the minimum viable set of inputs to start a run (corpus path, model base, target task)?** Every additional required field is friction; optional fields should have sensible defaults derived from the rig context.

3. **Should corpus assembly be a separate step from training launch, or a single unified flow?** Separating them gives control but adds complexity; combining them risks users launching training on an unprepared corpus.

4. **How does the user select which data goes into the corpus — manual curation, tag-based filtering, or "include everything"?** The selection mechanism determines corpus quality; a bad default here poisons every downstream model.

5. **What does "success" look like at each stage, and how is it communicated?** Users need distinct success signals for corpus validation ("corpus is clean"), training completion ("converged at step N"), and evaluation ("passes X of Y benchmarks").

6. **What does a training failure notification contain — just "failed" or actionable recovery steps?** A bare failure message forces the user to investigate manually; actionable context ("OOM at step 4200, reduce batch size") saves hours.

7. **How does the user cancel a running fine-tuning job, and what is preserved?** If cancellation discards all progress, users will be afraid to stop bad runs; if checkpoints are preserved, cancellation becomes a safe operation.

8. **Should there be a dry-run or preview mode that estimates cost and validates config without starting training?** Dry runs prevent expensive mistakes and build user confidence before committing resources.

9. **How does the user provide feedback on model quality after evaluation — thumbs up/down, structured rubric, free text?** Evaluation feedback is the signal that closes the improvement loop; the mechanism must be low-friction enough that people actually use it.

10. **What confirmation gates should exist before expensive operations (launching training, deploying a model)?** Accidental launches waste GPU hours; a confirmation step with cost estimate is essential but must not become annoying for experienced users.

11. **How does the system handle partial inputs — e.g., user specifies corpus but not model base?** Partial inputs should be accepted and filled with defaults rather than rejected, to lower the barrier to starting.

12. **Should fine-tuning configuration be expressible as a formula, enabling repeatable and shareable training recipes?** Formulas are Gas Town's native abstraction for repeatable workflows; training configs that are not formulas will feel foreign.

13. **How does the user compare two fine-tuned models side by side?** Without comparison tooling, users cannot tell if iteration N+1 is actually better than iteration N.

14. **What keyboard shortcuts or quick actions should exist for power users who run fine-tuning frequently?** The first run is exploratory; the tenth run should be fast and muscle-memory friendly.

### 3. User Flows

**Happy path, error path, edge cases.**

#### Happy Path

1. **What is step 1: does the user start from "I have data" or "I want a better agent"?** The starting mental model determines the entire flow direction — data-first vs. goal-first.

2. **How does the user go from "I want to fine-tune" to "corpus is assembled" — how many steps, how many decisions?** Every decision point is a potential abandonment point; the happy path must minimize required decisions.

3. **What happens between corpus assembly and training start — is there a review/validation gate?** Skipping validation risks garbage-in-garbage-out; too much validation slows the loop.

4. **How does the user monitor training progress — polling a dashboard, receiving notifications, or checking a bead status?** The monitoring mechanism should match Gas Town's existing notification patterns (beads, mail) rather than inventing a new one.

5. **What happens when training completes — does the model auto-deploy, require explicit promotion, or sit in a staging area?** Auto-deployment is dangerous for production agents; manual promotion adds a step but prevents bad models from going live.

6. **How does the user test the fine-tuned model before deploying it as an agent?** Without a sandbox or evaluation harness, users must deploy to test, which risks breaking live workflows.

7. **What does the deployment step look like — swapping a model reference in town settings, or something more involved?** Deployment should feel like "flip a switch," not "edit five config files."

#### Error Path

8. **What happens when the corpus fails validation — does the user see which entries failed and why?** Bulk "corpus invalid" messages are useless; per-entry diagnostics enable targeted fixes.

9. **How does the user recover from a mid-training crash (OOM, node disconnect, network partition)?** Recovery must be possible without restarting from scratch, or users will avoid large training runs.

10. **What happens when evaluation shows the model is worse than the base — what guidance does the system provide?** "Your model is worse" without explanation is demoralizing; the system should suggest corpus improvements or config changes.

11. **How does the system handle a user trying to fine-tune with zero corpus data?** This is a guaranteed early error; the message should guide them to corpus assembly, not just reject the request.

12. **What happens when the user's GPU is insufficient for the selected model configuration?** Hardware mismatch must be caught before training starts, not after 20 minutes of downloading model weights.

#### Edge Cases

13. **What happens when the corpus is enormous (100K+ entries) — does the UI degrade, does processing time explode?** Large corpora need pagination, sampling previews, and async processing to remain usable.

14. **What happens when the corpus contains contradictory examples (e.g., two formulas that prescribe opposite approaches)?** Contradictions confuse training; the system should detect and surface them, even if it cannot resolve them automatically.

15. **How does the system handle corpus entries that reference deleted or archived beads?** Dangling references produce incoherent training examples; they should be flagged during validation.

16. **What happens when multiple users try to fine-tune against the same corpus simultaneously?** Concurrent access needs either locking, copy-on-write semantics, or explicit conflict resolution.

17. **What if the user wants to fine-tune on data from a rig they don't own — are there permission boundaries?** Cross-rig data access raises both practical (prefix mismatches) and trust (data governance) concerns.

### 4. Visual & Layout

**Where does this live in Gas Town? Own rig or fits existing workflow? What patterns should it follow?**

1. **Should fine-tuning be its own rig (e.g., `finetune` with prefix `ft-`) or a workflow within an existing rig?** A dedicated rig provides clean isolation and its own bead namespace; embedding it in an existing rig reduces cognitive overhead but muddies boundaries.

2. **Does the corpus preparation workflow belong in the same view as training orchestration, or are they separate screens?** Combining them creates a long flow; separating them risks users losing context about which corpus goes with which training run.

3. **Where does the fine-tuning dashboard live relative to existing Gas Town screens (beads list, agent tree, event feed)?** It must be reachable within 1-2 navigation steps from the main Gas Town interface or it will be forgotten.

4. **Should training runs appear as beads in the bead system, following standard bead lifecycle (ready, active, done)?** Using beads maintains consistency with every other Gas Town workflow; inventing a new entity type adds learning cost.

5. **What existing Gas Town visual patterns (bead cards, status badges, progress indicators) should this system reuse?** Reusing patterns reduces design and implementation cost and leverages user familiarity.

6. **How should the corpus browser be laid out — list view, card view, tree view grouped by type?** The layout must support both quick scanning ("do I have enough data?") and deep inspection ("what exactly is in entry #4027?").

7. **Should training configuration be a form, a YAML editor, or a formula template?** Forms are approachable but inflexible; YAML editors are powerful but error-prone; formula templates balance both but require formula literacy.

8. **Where do evaluation results live — inline with the training run, a separate evaluation screen, or attached to the model bead?** Results must be findable both from "I'm looking at this run" and "I'm looking at this model" perspectives.

9. **How should the system visually distinguish between base models, fine-tuned variants, and deployed models?** Without clear visual distinction, users will confuse model versions and deploy the wrong one.

10. **Should there be a dedicated "model registry" view that lists all fine-tuned models across rigs?** As users create multiple fine-tuned variants, they need a single place to find, compare, and manage them.

11. **How does the fine-tuning workflow interact with the existing command palette and `gt` CLI?** CLI and TUI must be first-class citizens; a workflow that only works in one modality alienates half the user base.

12. **What density level is appropriate — compact for experienced users or spacious for newcomers?** Given Gas Town's power-user audience, default to dense with an option to expand, not the reverse.

13. **Should distributed node status (peer map, gradient sync visualization) be a separate "infrastructure" panel or integrated into the training view?** Infrastructure details support debugging but overwhelm during normal operation; a collapsible panel balances both needs.

### 5. States & Transitions

**What states can a fine-tuning job be in? How does the user move between states?**

1. **What are the canonical states — and should they map to bead states (ready, active, review, done) for consistency?** Mapping to bead states leverages existing mental models; diverging creates a parallel state vocabulary users must learn.

2. **Is "corpus preparation" a state of the fine-tuning job or a separate pre-job activity?** If corpus prep is inside the job lifecycle, the job can own its data; if outside, multiple jobs can share a corpus but provenance tracking is harder.

3. **What visual treatment distinguishes each state — color, icon, label, badge?** States must be distinguishable at a glance; relying solely on text labels is too slow for scanning a list of jobs.

4. **Can a job move backward (e.g., from "training" back to "preparing" if the user realizes the corpus needs changes)?** Backward transitions enable iteration but complicate state tracking; disallowing them forces users to create new jobs.

5. **What happens in the transition from "training" to "evaluating" — is it automatic or user-triggered?** Automatic transitions reduce friction but remove the user's ability to pause and inspect before evaluation begins.

6. **What does the "deployed" state mean concretely — the model is referenced in town settings as an agent's backing model?** "Deployed" must have a precise, observable meaning, or users will not trust the state label.

7. **Can a deployed model be "undeployed" or rolled back to a previous version?** Rollback is essential for production safety; without it, a bad deployment is a crisis instead of a recoverable mistake.

8. **Should there be a "paused" state for long-running training jobs that the user wants to suspend without canceling?** On a resource-constrained box (10GB RAM), pausing frees memory without losing progress — critical for Gas Town's hardware reality.

9. **What state does a job enter when it fails — "failed" as a terminal state, or "error" as a recoverable state?** If failure is terminal, users must recreate the job; if recoverable, they can fix the issue and resume.

10. **How long do completed/failed jobs persist in the system before being archived or garbage-collected?** Indefinite retention clutters the interface; aggressive cleanup loses history; the policy must balance both.

11. **Should state transitions generate beads, mail notifications, or event feed entries?** Transitions are significant events; if they are silent, users miss them; if they are noisy, users mute them.

12. **What state is a fine-tuning job in while the corpus is being validated but before training resources are allocated?** This "limbo" state — committed but not yet running — needs a name and a visual treatment, or users will think the system is stuck.

13. **Can multiple fine-tuning jobs be active simultaneously, and if so, how does the UI handle resource contention?** On a 10GB box, two simultaneous training runs will crash the system; the UI must communicate resource limits and queue behavior.

14. **What state does a model enter after deployment when a newer fine-tuned version is created — "superseded," "active," "archived"?** Model versioning is inevitable; the state model must account for the lifecycle of deployed models, not just training jobs.

15. **How does the user see the full lifecycle history of a job — a timeline, a log, a state-change audit trail?** History enables debugging ("why did this fail at step 3?") and learning ("what configuration worked last time?").

---

## Domain Expert Perspective

**Perspective: LLM Fine-Tuning, Distributed Training, and AI Agent Orchestration**

---

### 1. Domain Concepts

#### Terminology Assumed but Not Defined

1. **What exactly is a "Gas Town-native agent"?** Without a precise definition of what behaviors, conventions, and competencies constitute "native" operation, we cannot define a training objective or measure success.

2. **What does "fine-tune" mean in this context -- full fine-tuning, LoRA/QLoRA adapter training, or instruction tuning?** These are fundamentally different approaches with different data requirements, compute budgets, and outcome profiles; the brief conflates them.

3. **What is the target base model?** The brief says "fine-tune an LLM" but node0 currently only supports LLaMA 8B. Are we adapting that specific model, or is this infrastructure meant to be model-agnostic? The choice determines everything downstream.

4. **What is a "training corpus" in the Gas Town context?** Formulas, workflows, beads patterns, and session transcripts are structurally different artifacts. Are these all treated as text documents, or do we need structured input formats (instruction-response pairs, multi-turn conversations, tool-use traces)?

5. **What does "distributed" mean here -- data parallelism, pipeline parallelism, or expert parallelism?** Node0 uses Hivemind-based pipeline parallelism with decentralized gradient averaging. Fine-tuning typically uses data parallelism. These are not the same thing and the infrastructure may need substantial rearchitecting.

6. **What is the distinction between "pretraining" and "fine-tuning" in the node0 architecture?** The context.md explicitly says node0 is pretraining infrastructure. Fine-tuning requires different training dynamics (lower LR, fewer steps, different regularization). How much of node0's existing code path actually applies?

7. **What does "operate as a native Gas Town agent" mean in terms of tool use?** Modern agent fine-tuning requires tool-calling training data (function signatures, invocation patterns, result handling). Is this in scope or do we assume the base model already has tool-use capability?

8. **What is "convention adherence" in measurable terms?** The brief mentions Gas Town conventions (beads patterns, formula usage, workflow steps) but these are not formalized as a grammar or schema that could serve as an evaluation rubric.

9. **What is the relationship between node0's "experts" (pipeline stages) and Gas Town's "agents" (mayor, polecat, refinery)?** The word "expert" means completely different things in these two contexts and conflating them will cause confusion.

10. **What does "prepare a corpus" actually entail -- collection, cleaning, formatting, deduplication, or all of the above?** Corpus preparation is typically 70-80% of a fine-tuning project's effort. The brief treats it as a single bullet point.

11. **What is the intended inference infrastructure?** Fine-tuning produces a model, but that model needs to be served. Is the plan to serve the fine-tuned model through node0's distributed infrastructure, or through a separate inference stack? This affects model format, quantization choices, and export requirements.

#### Missing Concepts

12. **Data format specification is entirely absent.** Instruction tuning needs instruction-response pairs. RLHF needs preference pairs. DPO needs chosen/rejected pairs. Tool-use training needs function call traces. Which format(s) do we need?

13. **There is no mention of evaluation or validation sets.** Without held-out data, we cannot detect overfitting, measure improvement, or compare approaches.

14. **Tokenizer alignment is not discussed.** If we use a base LLaMA model, its tokenizer may not efficiently encode Gas Town-specific tokens (bead IDs, formula names, gt commands). Do we need tokenizer extension?

15. **The concept of "curriculum" is missing.** Should the model learn general instruction following first, then Gas Town specifics? Or do we mix everything? Curriculum design significantly affects fine-tuning outcomes.

### 2. Prior Art

16. **How does this compare to Gorilla (Berkeley's API-calling LLM)?** Gorilla fine-tuned LLaMA on API documentation and achieved strong tool-use performance. Their approach to formatting API calls as training data is directly relevant -- have we studied it?

17. **What can we learn from ToolLLM/ToolBench (Qin et al., 2023)?** They created a large-scale tool-use dataset and fine-tuned models to use 16K+ APIs. Their DFSDT (Depth-First Search Decision Tree) approach to multi-step tool use is the closest prior art to what a Gas Town agent needs.

18. **Has anyone fine-tuned models on CLI/terminal interaction traces before?** Projects like ShellGPT, Open Interpreter, and Aider generate terminal interaction data. Their failure modes (hallucinating commands, wrong flags, unsafe operations) are the exact failure modes we need to prevent.

19. **What happened with the "train your own Copilot" wave (StarCoder, CodeLlama)?** Code-specific fine-tuning showed that domain-specific pretraining data matters more than fine-tuning data volume. If Gas Town conventions are too niche, fine-tuning alone may not work -- we may need continued pretraining first.

20. **How do existing agent frameworks (LangChain, CrewAI, AutoGen) handle agent behavior without fine-tuning?** They use prompt engineering and few-shot examples. We need to articulate why fine-tuning is necessary instead of (or in addition to) better prompting. If we cannot, we are solving the wrong problem.

21. **What are the known failure modes of LoRA/QLoRA for behavioral fine-tuning?** Research shows LoRA is excellent for style transfer and knowledge injection but struggles with teaching genuinely new capabilities (tool use patterns, multi-step reasoning). Full fine-tuning or continued pretraining may be needed for deep behavioral changes.

22. **What does the RLHF/DPO landscape look like for agent behavior?** Constitutional AI (Anthropic), DPO (Rafailov et al.), and ORPO are all approaches to aligning model behavior with preferences. For a Gas Town agent, "correct workflow execution" is a preference signal -- are we planning to use it?

23. **What about retrieval-augmented generation (RAG) as an alternative or complement?** Fine-tuning bakes knowledge into weights. RAG retrieves it at inference time. For rapidly evolving conventions (Gas Town is under active development), RAG may be more appropriate for factual knowledge while fine-tuning handles behavioral patterns.

24. **How did Pluralis (node0's original authors) handle data loading for pretraining?** The context.md notes that node0 does NOT include data loading -- gradients come from external training loops. This is a massive gap. What was Pluralis's solution, and can we reuse or adapt it?

25. **What can we learn from Hivemind's own fine-tuning examples (Training Transformers Together)?** The "Training Transformers Together" project used Hivemind for collaborative pretraining of a GPT-like model. Their data pipeline design and failure modes are directly relevant.

26. **What do practitioners expect from a distributed fine-tuning setup?** Standards like DeepSpeed ZeRO, FSDP, and Megatron-LM have set expectations around checkpoint management, mixed precision, gradient accumulation, and data loading. Node0's Hivemind-based approach is unconventional -- how do we bridge the gap in tooling and documentation?

### 3. Problem Depth

27. **Is the real problem "we need a fine-tuned model" or "we need better agent performance"?** Fine-tuning is one approach to improving agent performance. Prompt engineering, RAG, better tool definitions, and workflow constraints are others. Have we validated that the base model's failures are actually addressable through fine-tuning rather than better scaffolding?

28. **Is the distributed training aspect actually necessary, or is it aspirational?** An 8B parameter model can be LoRA fine-tuned on a single 24GB GPU in hours. The distributed infrastructure adds enormous complexity. What is the actual compute constraint that requires distribution? Is it future-proofing for larger models, or a requirement for the current scope?

29. **Are we solving a data problem disguised as a training problem?** If the bottleneck is lack of high-quality Gas Town interaction traces, no amount of infrastructure will help. Do we have enough data? What constitutes "enough"?

30. **Is a single fine-tuned model the right architecture, or do we need role-specific models?** A mayor agent has different responsibilities than a polecat or refinery agent. One model fine-tuned for all roles may underperform role-specific models. But role-specific models multiply the maintenance burden.

31. **What related problems will users expect us to solve that are NOT in this brief?** Specifically: (a) model versioning and rollback, (b) A/B testing between model versions, (c) continuous fine-tuning as new session data accumulates, (d) monitoring model drift in production, (e) safety guardrails for the fine-tuned model.

32. **What are we explicitly NOT solving?** The brief does not mention: inference serving, model quantization for deployment, safety/alignment, multi-modal capabilities, or real-time learning. Are these out of scope by design or by omission?

33. **How does this interact with the existing agent provider system?** Gas Town already maps roles to agents (Claude Opus 4.6, pi, etc.). A fine-tuned model would be a new provider. How does it integrate with the existing dispatch, session management, and polecat lifecycle?

34. **Is the 10GB box (mentioned in MEMORY.md) the target training environment?** If so, fine-tuning even an 8B model with QLoRA requires careful memory management. The box already runs Dolt (550MB), openclaw-gateway (1GB), and claude sessions (300-750MB each). Where does training fit?

35. **What is the deployment target?** Is the fine-tuned model intended to run locally on the same resource-constrained box, or served externally via API? This determines quantization requirements, model size constraints, and the entire inference story.

36. **Are session transcripts the right training signal, or do we need curated demonstrations?** Raw transcripts include false starts, errors, retries, and off-topic tangents. Practitioners know that training on messy data produces messy outputs. Do we need expert curation, or is the volume large enough to overcome noise?

### 4. Edge Cases (Domain)

37. **What happens when Gas Town conventions change after fine-tuning?** The codebase is under active development. Formulas get added, workflows evolve, command syntax changes. A model trained on today's conventions will drift from tomorrow's reality. How do we handle this without retraining from scratch?

38. **What about cross-rig behavioral contamination?** If the corpus includes transcripts from multiple rigs (sfgastown, frankencord, frankentui, bv, pi_agent), the model may learn rig-specific patterns and apply them in the wrong context. How do we prevent this?

39. **What if the training data contains secrets, tokens, or credentials?** Session transcripts may include HF tokens, API keys, file paths with usernames, or other sensitive data. The MEMORY.md mentions .env and credentials concerns. Data sanitization is critical but not mentioned.

40. **How do we handle the "needle in a haystack" problem for rare but critical patterns?** Some Gas Town operations (town shutdown, migration, convoy cleanup) are rare but important. If they appear in <1% of training data, the model may not learn them. Do we oversample rare events?

41. **What about model collapse from self-generated training data?** If polecats generate code that gets reviewed and merged, and those sessions become training data, we risk a feedback loop where model errors compound across generations. This is a well-documented phenomenon (Shumailov et al., 2023).

42. **What if the base model already performs well on Gas Town tasks with good prompting?** We may invest months in fine-tuning only to discover marginal improvement over a well-prompted Claude or GPT. Do we have baseline measurements to justify the investment?

43. **How do we handle multi-turn context in training data?** Gas Town sessions are long (thousands of turns). Transformer context windows are finite. Do we truncate, summarize, or segment? Each choice introduces artifacts.

44. **What about adversarial or malformed inputs during inference?** A fine-tuned model deployed as a Gas Town agent will encounter unexpected situations. Fine-tuning can reduce robustness to out-of-distribution inputs (the "alignment tax"). How do we maintain the base model's general capabilities?

45. **What happens when distributed training nodes disagree on gradient direction?** Node0 uses decentralized averaging, which means Byzantine nodes can poison the model. For pretraining on public data this is manageable, but for fine-tuning on curated data the stakes are higher. Is Byzantine fault tolerance in scope?

46. **What about catastrophic forgetting of the base model's general capabilities?** Aggressive fine-tuning on domain-specific data can destroy the model's ability to do basic reasoning, follow instructions, or write coherent text. This is especially dangerous with full fine-tuning on small datasets.

47. **How do we handle the tokenizer efficiency problem?** Gas Town uses domain-specific identifiers (bead IDs like "st-btod", "bd-9f1f"; rig names; formula names). These may tokenize into 3-5 tokens each, wasting context window and degrading performance. Tokenizer extension is non-trivial with pretrained models.

48. **What if training data quality varies dramatically across sources?** Mayor sessions (Claude Opus) may be high quality. Polecat sessions may include failed attempts. Pi agent sessions have different formatting. Mixing heterogeneous quality data without weighting can drag down the model.

49. **What about reproducibility?** Distributed training with dynamic peer join/leave is inherently non-deterministic. How do we ensure reproducible training runs for comparison and debugging?

### 5. Success Criteria

50. **How would we know the fine-tuned model is better than prompting a frontier model (Claude, GPT)?** If the fine-tuned 8B model underperforms a well-prompted Claude Opus with RAG, the entire project is a net negative. We need a clear comparison framework.

51. **What specific tasks should the model be able to perform?** Vague goals like "operate as a native Gas Town agent" need decomposition: Can it create beads? Run formulas? Interpret `gt` CLI output? Review PRs? Dispatch polecats? Each is independently measurable.

52. **What is the acceptable error rate?** A model that gets Gas Town workflows right 90% of the time but silently corrupts data 10% of the time is worse than useless. What is the threshold for each task category?

53. **What does "convention adherence" look like concretely?** Can we define a checklist: uses `gt` commands instead of raw shell equivalents, follows bead dependency conventions, respects resource constraints, uses correct issue prefixes, etc.?

54. **What is the latency budget?** A fine-tuned model running on local hardware will have different latency than API-served frontier models. What is the acceptable response time for interactive agent use?

55. **What is the cost target?** Fine-tuning and serving a model has ongoing costs (compute, storage, maintenance). At what cost differential vs. API calls does self-hosting become worthwhile?

56. **How do we measure "Gas Town nativeness" vs. "general capability"?** We need a balanced evaluation that tests both domain-specific tasks AND general reasoning/coding ability. A model that aces Gas Town conventions but cannot debug Python is useless.

57. **What is the minimum viable corpus size?** Research suggests instruction tuning needs 1K-100K high-quality examples depending on the task. Tool-use fine-tuning may need more. Do we have a data budget estimate?

58. **How do we evaluate multi-step workflow correctness?** A model may get individual commands right but fail at sequencing (e.g., creating a bead, then slinging it, then monitoring the polecat). End-to-end workflow evaluation is harder than per-step evaluation.

59. **What is the iteration cycle time target?** If a training run takes 2 weeks, iteration is too slow for practical development. What is the target time from "new training data" to "deployed model"?

60. **Do we have human evaluators, and what is their rubric?** Automated metrics (perplexity, BLEU, task success rate) miss nuances. Human evaluation of agent behavior requires evaluators who understand Gas Town conventions deeply. Who are they, and how do we calibrate them?

61. **What is the minimum hardware configuration for inference?** If the fine-tuned model requires a 24GB GPU to serve, it cannot run on the current 10GB box alongside other services. This constrains model size and quantization choices.

62. **How do we measure safety/harmlessness after fine-tuning?** Fine-tuning can degrade safety training. A Gas Town agent that follows commands without checking for destructive operations (force push to main, rm -rf, nuke without verification) is dangerous. The MEMORY.md already documents these footguns.

---

## Cross-Perspective Themes (Opus)

### 1. **Data Quality & Curation (User + Domain)**
The three perspectives consistently highlight that raw session transcripts and corpora require significant curation to be useful. Users worry about including mistakes and edge cases; domain experts note that training on messy data produces messy models. There is no clear story for automated data cleaning, deduplication, conflict resolution, or quality filtering. This is 70-80% of the effort in real fine-tuning projects but is treated as a single bullet point in the brief.

### 2. **Resource Constraints & Hardware Reality (User + Designer + Domain)**
All three perspectives emphasize that the 10GB box is memory-constrained and already running critical services (Dolt, gateway, Claude sessions). Fine-tuning infrastructure must coexist with normal Gas Town operations without crashing the box. Users need clear resource budgets; designers need UI patterns that communicate queue behavior and resource limits; domain experts need honest acknowledgment that distributed training on commodity hardware is non-trivial. Without solving this, the feature will be unreliable and dangerous.

### 3. **Baseline & Success Metrics (Domain + Product + User)**
None of the three perspectives define what "success" looks like. Is the goal to match Claude Opus? Beat it? Just avoid being catastrophically worse? Without baseline measurements (how well do current models + good prompting perform on Gas Town tasks?) and clear success criteria, it is impossible to know if fine-tuning is the right solution or if simpler alternatives (RAG, better prompting, workflow constraints) would be better. Users need to understand the value proposition; designers need measurable states to present; domain experts need a rubric to guide technical choices.

### 4. **Model Staleness & Corpus Freshness (User + Domain)**
Gas Town is under active development. Formulas, workflows, CLI commands, and conventions evolve. Both users and domain experts flag that a model trained on today's conventions will drift from tomorrow's reality. There is no story for continuous retraining, corpus updates, or model versioning that keeps the model current without manual intervention. This is a systemic problem: either the model becomes a liability that gives outdated advice, or the operational burden of keeping it fresh becomes unbearable.

### 5. **Integration with Existing Gas Town Abstractions (Product + User + Domain)**
The designer and user perspectives both note that the feature must integrate with Gas Town's existing patterns (beads, formulas, CLI, agent dispatch) rather than creating parallel systems. The domain expert questions whether node0's pretraining infrastructure is actually suitable for fine-tuning. There is a tension between making the feature feel native to Gas Town (which users expect) and whether the underlying technical approach is sound. Without resolving this, users will feel the feature is bolted on and the infrastructure may not support the workload.


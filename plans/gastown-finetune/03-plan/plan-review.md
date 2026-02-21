# Plan Review: gastown-finetune

**Generated:** 2026-02-22
**Reviewers:** Forward (spec→plan), Reverse (plan→spec), Context (plan→codebase)

---

## Summary

| Category | P0 | P1 | P2 | Total |
|----------|----|----|----|----|
| Coverage Gaps | 0 | 5 | 4 | 9 |
| Scope Creep | 0 | 0 | 0 | 0 |
| Codebase Misalignment | 3 | 5 | 2 | 10 |
| Consistency Issues | 0 | 0 | 1 | 1 |
| **Total** | **3** | **10** | **7** | **20** |

---

## P0 Findings (Must Fix)

### 1. Error Class Import Path Wrong [CORROBORATED]
- **Category:** Codebase Misalignment
- **Found by:** Context review (pattern compliance section)
- **What:** Plan imports `NonRetriableError`, `RetriableError`, and `call_with_retries` from `src/node0/utils/logging.py`, but these classes are actually in `src/node0/utils/node_info.py`. Any new code following the plan will have `ImportError` at runtime.
- **Evidence:** Context verification of actual codebase shows error hierarchy defined at `node_info.py`, not `logging.py`. Multiple tasks reference these imports: Tasks 2.5 (tokenizer validation), 3.4 (NaN handler), 4.2 (checkpoint gatherer), 4.3 (GGUF export).
- **Action:** Update plan - correct import path in all tasks
- **Recommendation:** Change all imports from `from node0.utils.logging import NonRetriableError, RetriableError, call_with_retries` to `from node0.utils.node_info import NonRetriableError, RetriableError, call_with_retries`. Update Tasks 2.5, 3.4, 4.2, and 4.3 to reference this corrected import.

### 2. Formula File Path Wrong [CORROBORATED]
- **Category:** Codebase Misalignment
- **Found by:** Context review (integration surface accuracy)
- **What:** Plan Task 6.5 targets `~/.beads/formulas/` for formula TOML files, but the actual Gas Town formula directory is `~/gt/.beads/formulas/` (i.e., `/home/ubuntu/gt/.beads/formulas/`). Files written to `~/.beads/formulas/` will not be picked up by Gas Town.
- **Evidence:** Context confirms `/home/ubuntu/gt/.beads/formulas/` exists with existing formulas. The `~/.beads/` path is a different SQLite-backed storage with no `formulas/` subdirectory. Git status shows deletion of repo-local `.beads/metadata.json`, consistent with formulas living in GT-level beads, not repo-level.
- **Action:** Update plan
- **Recommendation:** Correct Task 6.5 to create formulas at `~/gt/.beads/formulas/train-role-model.formula.toml`, `~/gt/.beads/formulas/corpus-collect.formula.toml`, and `~/gt/.beads/formulas/eval-suite.formula.toml`. Path expansion should resolve `~` to `/home/ubuntu/gt/.beads/formulas/`.

### 3. Missing `run_finetune.py` Entry Point Module
- **Category:** Codebase Misalignment
- **Found by:** Context review (missed integration points, integration surface accuracy)
- **What:** Plan Task 1.5 registers `node0-finetune = "node0.run_finetune:main"` as a console script entry point in `pyproject.toml`, but no phase creates `src/node0/run_finetune.py`. This is a missing file that breaks the entry point.
- **Evidence:** Context lists `node0-finetune` as a required entry point alongside `node0-inference` and `node0-corpus`. Phase 5 creates `run_inference.py` and Phase 2 creates `run_corpus.py` but no task creates the corresponding `run_finetune.py`. The entry point reference would raise `ModuleNotFoundError` at runtime.
- **Action:** Update plan
- **Recommendation:** Add a new task in Phase 1 or Phase 3 to create `src/node0/run_finetune.py` as the main CLI entry point for launching fine-tuning jobs. Should follow the pattern of `run_server.py` and `run_inference.py` with `def main()` accepting command-line arguments for training config, corpus path, checkpoint directory, etc.

---

## P1 Findings (Should Fix)

### 1. Data Sources Incomplete — Codebase Artifacts and Operational Data Not Collected [CORROBORATED]
- **Category:** Coverage Gap
- **Found by:** Forward (review-forward.tmp, "Data Pipeline & Corpus: Data Sources")
- **What:** Spec lists three data source categories: (1) session transcripts, (2) codebase artifacts (CLAUDE.md, formulas, specs, AGENTS.md, config files), (3) operational data (bead lifecycles, dependency graphs, sling outcomes, convoy logs, gt prime output). Plan covers (1) well but does not address (2) codebase artifacts or (3) operational data as training corpus sources.
- **Evidence:** Forward review: "The plan covers (1) well. Category (2) — codebase artifacts — is not addressed by any task. Category (3) — operational data — is mentioned only in the RAG indexer (Task 5.5) but never collected into training corpus format." Context confirms corpus collection only addresses session transcripts via Task 2.4 (session_shutdown hook).
- **Action:** Update plan
- **Recommendation:** Add tasks to Phase 2 for: (a) ingesting CLAUDE.md files from each rig as corpus entries (corpus_type="convention_doc", role="all"); (b) extracting formula definitions as corpus entries; (c) collecting bead lifecycle events (create, status transitions, completion) as operational corpus entries. These should be collected periodically via background jobs or triggered by corresponding Gas Town events.

### 2. V1+ Extensibility — Cross-Operator Incentive Model Has No Implementation
- **Category:** Coverage Gap
- **Found by:** Forward (review-forward.tmp, "Data Pipeline & Corpus: Extensibility Path")
- **What:** Spec defines V1+ (after V1 launch) as including cross-operator data contribution with incentive model: "contribute data, get model access even without GPU." This design decision (Q42) is listed as V1+ in the spec but has no implementation tasks in the plan.
- **Evidence:** Forward review: "The plan addresses opt-in sharing at the rig config level (Task 5.6) but does not define any mechanism for cross-operator data contribution, access grants, or the identity/trust model needed for V1+ corpus intake." Task 5.6 only handles opt-in rig config sharing, not the cross-operator mechanism.
- **Action:** Accept as-is
- **Recommendation:** This is correctly scoped as V1+ (post-launch), not V1. The plan should explicitly note in Technical Risks that the Q42 incentive model is deferred to Phase V1+ and that only intra-rig opt-in (Task 5.6) is included in V1. No plan change needed, but acceptance should be explicit in the project.

### 3. Training Job Bead Lifecycle — Custom States Not Implemented as Bead Metadata [CORROBORATED]
- **Category:** Codebase Misalignment
- **Found by:** Forward (review-forward.tmp, "Training Infrastructure: Training Job Lifecycle (Bead-Mapped)"), Context (missed integration points, #6)
- **What:** Spec defines 7 explicit bead states for training jobs: `corpus-validating` → `ready` → `training` → `syncing` → `checkpointing` → `evaluating` → `done`/`failed`/`degraded`. Plan's MonitorWorker (Task 2.8) logs these events but does not implement them as actual bead status transitions via `bd update --status`. The lifecycle diagram in spec uses `bd create` / `bd update --status` / `bd close` as the interface, not log-scraping.
- **Evidence:** Forward review: "The plan does not have a task that implements these custom bead states as actual bead status values. The MonitorWorker log-scraping approach handles `failed` but there is no task that creates the intermediate states as real bead metadata updates." Context: "Plan's MonitorWorker extension (Task 2.8) logs these events but no task investigates or implements the bead schema extension."
- **Action:** Update plan
- **Recommendation:** Add a task in Phase 3 to implement direct bead state updates: (a) investigate if `bd update --status` needs to accept custom status values beyond the hardcoded set; (b) modify training job launch to create a bead with `status="corpus-validating"`; (c) wire bead updates through training phases via direct `bd update` calls (not log-scraping). This is a design decision to make: either accept log-based indirect state management or implement direct bead status transitions. If log-based is acceptable, spec should be updated to reflect this tradeoff.

### 4. Model Registry UI Dropped Silently [CORROBORATED]
- **Category:** Coverage Gap
- **Found by:** Forward (review-forward.tmp, "Model Registry & Deployment: Model Registry"), Reverse (review-reverse.tmp notes this is Spec-backed, not checked)
- **What:** Spec decision Q70 states "Full UI in V1 — CLI + browsable UI" for model registry. Plan implements CLI commands (`gt model list`, `show`, etc.) but notes "model registry UI is a product feature scoped as its own epic" without including it in any phase. V1 ships without the browsable UI.
- **Evidence:** Forward review: "The spec says 'Full UI in V1' (Q70, explicit decision). The plan correctly notes the UI is its own epic but does not include it in any phase or task. This means V1 ships without the browsable UI that is a committed spec decision."
- **Action:** Update plan or update spec
- **Recommendation:** Either (a) add a Phase 6+ task for building a minimal browsable UI (e.g., HTML template showing model versions, scores, deployment status), or (b) explicitly update spec to defer "Full UI" to V1.5/V2 and mark Q70 as revised. This is a scope decision that needs to be conscious.

### 5. gt prime Modification Not Tasked — Prime Context Bloat Not Reduced [CORROBORATED]
- **Category:** Coverage Gap
- **Found by:** Forward (review-forward.tmp, "Gas Town Integration: Component Integration Map"), Context (missed integration points, partial note)
- **What:** Spec explicitly names "gt prime — MODIFY" in the Component Integration Map with goal "inject model-appropriate context (fine-tuned needs less priming)." This is a stated design to reduce prime bloat (~15K → ~2-3K tokens RAG context), a key motivation. Plan has no task to modify `gt prime` behavior. Without it, fine-tuned agents receive full 15K token prime context, negating one of the four primary motivations.
- **Evidence:** Forward review: "The spec says `gt prime` should be MODIFIED to 'inject model-appropriate context (fine-tuned needs less priming).' The plan has no task for modifying `gt prime` behavior. This is a named integration point in the spec." Performance requirement gap: "No task measures this reduction. No acceptance criterion in any phase validates that the fine-tuned + RAG combination actually reduces context injection size."
- **Action:** Update plan
- **Recommendation:** Add a task in Phase 5 (after inference is ready) to modify `gt prime` to detect when serving a fine-tuned model and reduce context injection. Detection mechanism: check `model_args.lora_rank` or query model registry for model type. Injection reduction: cap context to ~2-3K tokens (RAG fills the gap). Add acceptance criterion to validate final context size is within target range.

### 6. Polecat Progressive Rollout Stage Not Included
- **Category:** Coverage Gap
- **Found by:** Forward (review-forward.tmp, "Model Registry & Deployment: Progressive Rollout")
- **What:** Spec progressive rollout has five stages: deacon, boot, witness, polecat, crew (mayor is V2+). Plan Task 6.6 covers stages 1–2 (deacon/witness) and explicitly defers polecat. However, the spec lists polecat as V1, not V2+. Spec text: "polecat is stage 3 of V1 progressive rollout" (for specific task types before crew).
- **Evidence:** Forward review: "The spec puts mayor in V2+, but polecat is listed as stage 3 of V1 progressive rollout. The plan stops at witness without a task or phase for polecat rollout." Spec "Progressive Rollout" section lists polecat as stage 3 deployment target in V1.
- **Action:** Update plan
- **Recommendation:** Add a task in Phase 6 for polecat rollout (stage 3): select specific task types in polecat role (e.g., planning, code review, architecture) that are lower risk than crew-wide deployment. Gate rollout on eval suite scores like deacon/witness stages. Alternatively, if polecat rollout is deemed too risky for V1, update spec to move it to V2 scope.

### 7. Dockerfile Updates Not Assigned
- **Category:** Codebase Misalignment
- **Found by:** Context review (missed integration points, #3)
- **What:** Context identifies Dockerfile as requiring updates: add dependencies (`transformers`, `chromadb`, `detect-secrets`), add volume mounts for corpus and checkpoints. Plan Task 1.5 updates `pyproject.toml` but does not include a task to update Dockerfile.
- **Evidence:** Context: "The context's 'Build System Files' section identifies the Dockerfile as needing updates (add `transformers`, `chromadb`, `detect-secrets`; add volume mounts for corpus and checkpoints). The plan's Task 1.5 covers `pyproject.toml` but does not include a task to update the Dockerfile."
- **Action:** Update plan
- **Recommendation:** Add a task in Phase 1 (alongside Task 1.5) to update Dockerfile: (a) add `transformers`, `chromadb`, `detect-secrets` to base image dependencies or pip install in RUN layer; (b) add `VOLUME ["/root/gt/.corpus", "/root/gt/.checkpoints"]` declarations; (c) ensure all new entry points (`node0-finetune`, `node0-inference`, `node0-corpus`) are on PATH.

### 8. `run.json` Deployment Configuration Not Addressed
- **Category:** Codebase Misalignment
- **Found by:** Context review (missed integration points, #4)
- **What:** Context identifies `run.json` as requiring updates to support local seed peers and bypass-auth config for self-hosted fine-tuning runs. Plan covers `run_server.py` CLI flags (`--local-mode`) but never addresses `run.json`.
- **Evidence:** Context: "The context's 'Build System Files' section identifies `run.json` as needing updates to support local seed peers and `bypass_auth: true` / `local_stage` config for self-hosted runs. The plan covers the `run_server.py` CLI flags but never addresses `run.json`."
- **Action:** Update plan
- **Recommendation:** Add a task in Phase 1 to update `run.json` with new configuration options: (a) `"bypass_auth": true` field for local mode; (b) `"local_seed_peers": []` for peer discovery without Pluralis DHT; (c) example self-hosted run template. Ensure this is documented in Technical Risks as "Local DHT seed peer configuration requires manual `run.json` editing" with acceptance criteria for ease of setup.

### 9. `make_validators()` Integration Requires Schema Wrapper Class
- **Category:** Codebase Misalignment
- **Found by:** Context review (integration surface accuracy)
- **What:** Plan Task 1.2 says "update `make_validators()` to register it" for `TrainingMetricsV1`, but actual `make_validators()` uses `SchemaValidator(MetricSchema, ...)` where `MetricSchema` is a Pydantic v1 wrapper class. Adding `TrainingMetricsV1` requires: (a) creating a `TrainingMetricSchema` wrapper with `training_metrics: dict[BytesWithPublicKeyType, TrainingMetricsV1]`, (b) creating a `SchemaValidator(TrainingMetricSchema, ...)`, (c) appending to validators list. Plan's task description does not capture this.
- **Evidence:** Context: "The actual function constructs `SchemaValidator(MetricSchema, prefix=...)` objects, where `MetricSchema` is a Pydantic v1 wrapper that declares the DHT key structure with type-constrained dicts. The plan says 'update `make_validators()` to register it' but adding `TrainingMetricsV1` requires: (a) creating a `TrainingMetricSchema` wrapper class...(c) appending to the validators list. The plan's task description does not capture this level of detail."
- **Action:** Update plan
- **Recommendation:** Expand Task 1.2 description to explicitly state: Create `TrainingMetricSchema` wrapper class in `security/validation.py` with field `training_metrics: dict[BytesWithPublicKeyType, TrainingMetricsV1]`. Then create `SchemaValidator(TrainingMetricSchema, prefix="training_metrics")` and append to the list returned by `make_validators()`. Add this detail to Task 1.2 acceptance criteria.

---

## P2 Findings (Consider)

### 1. Polecat Progressive Rollout Missing from Phase Coverage
- **Category:** Coverage Gap
- **Found by:** Forward (review-forward.tmp, coverage matrix verification)
- **What:** Forward coverage matrix notes "Progressive Rollout: polecat (stage 3)" is not in the spec coverage matrix. Matrix entry only mentions "deacon/witness rollout". Polecat stage is missing.
- **Evidence:** Forward review: "Missing from matrix: gt prime modification; model registry UI; polecat progressive rollout stage; reversion detection; V1+ incentive model details; codebase artifact ingestion as training data."
- **Action:** Update plan
- **Recommendation:** Add polecat stage to spec coverage matrix with reference to planned Phase 6 task (once P1 finding #6 is addressed). Matrix should show all five stages with risk levels.

### 2. Periodic Scrubbing Audit Mechanism Has No Operational Task
- **Category:** Coverage Gap
- **Found by:** Forward (review-forward.tmp, "Data Pipeline & Corpus: Data Characteristics")
- **What:** Spec defines four scrubbing layers including "periodic audit/sampling of scrubbed output". Plan covers this with test fixtures (known PII patterns) but does not include an explicit `node0-corpus audit` CLI subcommand or scheduled audit mechanism for operational audits.
- **Evidence:** Forward review: "The plan covers this by including test fixtures with known PII patterns, but does not explicitly include a `node0-corpus audit` CLI subcommand or scheduled audit mechanism. Tests catch known patterns but don't constitute a periodic operational audit."
- **Action:** Update plan
- **Recommendation:** Add a task in Phase 2 (Task 2.2 extension) to implement `node0-corpus audit` subcommand: sample recent corpus entries, re-scan with scrubber rules, report false negatives/positives. Automate via cron job (e.g., weekly) with results logged. This closes the fourth scrubbing layer.

### 3. Model Prune Command Missing — Version Accumulation
- **Category:** Coverage Gap
- **Found by:** Forward (review-forward.tmp, "Model Registry & Deployment: Versioned Rollback")
- **What:** Spec states "Old versions retained until explicitly pruned" but plan has no `gt model prune` command to delete old model artifacts. Model versions accumulate indefinitely.
- **Evidence:** Forward review: "The plan has no `gt model prune` command. Minor — pruning is an operational detail, but retention policy is mentioned as a requirement."
- **Action:** Update plan
- **Recommendation:** Add a task in Phase 4 (Task 4.5 extension) to implement `gt model prune <model_name> --keep=N` command: removes all but the N most recent versions, deletes associated artifacts from `~/.models/`, updates Dolt metadata. Include dry-run option. Document retention policy in model registry CLI help.

### 4. Automated Eval Deployment Gate Not Enforced
- **Category:** Coverage Gap
- **Found by:** Forward (review-forward.tmp, "Evaluation & Quality: Automated Eval Task Suite")
- **What:** Spec calls for automated deployment gate: "score >= threshold? → yes → register, deployable; no → flag for investigation, block deployment." Plan implements scoring but the gate in Task 6.6 is manual: "run `gt model show` and verify eval_score is above operator-determined threshold". No automated enforcement prevents deployment below threshold.
- **Evidence:** Forward review: "The plan implements the scoring but the gate check in Task 6.6 is manual ('run `gt model show` and verify eval_score is above operator-determined threshold'). There is no automated enforcement that prevents deployment when scores are below threshold."
- **Action:** Update plan
- **Recommendation:** Add implementation detail to Task 6.6: `gt model deploy` should be modified to check eval score >= threshold before modifying town settings. If score < threshold, raise error with message "Model score N below threshold T. Run eval suite again or override with --force-deploy." This makes the gate automatic in the normal flow.

### 5. Staleness Tracker Incomplete — Formula and Config Changes Not Tracked
- **Category:** Coverage Gap
- **Found by:** Forward (review-forward.tmp, "Evaluation & Quality: Model Staleness Detection"), Reverse (review-reverse.tmp, architecture decision review, row "Staleness detection scope")
- **What:** Spec says staleness covers "Gas Town convention changes (CLAUDE.md updates, new formulas, config changes)." Plan's staleness tracker (Task 7.3) only tracks CLAUDE.md hash drift. Formula and config changes are not tracked as staleness signals.
- **Evidence:** Forward review: "The plan's staleness tracker only tracks CLAUDE.md hash drift. New formulas and config changes are not tracked as staleness signals. Minor." Reverse review flags this as "narrower than spec; plan does not explain why formulas and config changes are omitted."
- **Action:** Update plan
- **Recommendation:** Expand Task 7.3 (`ConventionStalenessTracker`) to also track: (a) formula directory hash (`hashlib.sha256(sorted(Path(formulas_dir).glob('*.toml')))`) for new formula detection; (b) town settings config hash for config changes. If any hash changes exceed threshold, trigger retrain pipeline. Add acceptance criterion documenting hash stability over 7-day no-change baseline.

### 6. RAG Indexer Missing Formula Definitions Collection
- **Category:** Coverage Gap
- **Found by:** Forward (review-forward.tmp, "Inference Architecture: RAG Architecture")
- **What:** Spec says RAG indexes formula definitions in addition to beads, configs, and sessions. Plan's `RAGIndexer` (Task 5.5) has collections for `beads`, `configs`, `sessions` but no `formulas` collection. Formula definitions are indexed during corpus ingestion but not available at inference-time retrieval.
- **Evidence:** Forward review: "The spec says RAG indexes formula definitions in addition to beads, configs, and sessions. The plan's RAGIndexer has collections for `beads`, `configs`, `sessions` but no `formulas` collection. Minor omission."
- **Action:** Update plan
- **Recommendation:** Add a `formulas` collection to `RAGIndexer` (Task 5.5): index all `.toml` formula files from `~/.beads/formulas/`, extract task descriptions and parameter definitions, store as retrievable documents. During inference, include formula definitions in RAG context when available. Document as optional enhancement if it requires significant additional work.

### 7. Corpus Reversion Detection Not Implemented
- **Category:** Coverage Gap
- **Found by:** Forward (review-forward.tmp, "Corpus Collection Mechanism")
- **What:** Spec includes specific caveat: "if a bad convention is introduced and later reverted, the revert must be explicitly captured as a new corpus entry — otherwise 'recency wins' preserves the bad convention. Corpus pipeline should flag reversions (detecting when a CLAUDE.md change undoes a recent change) for manual review." Plan has no task for reversion detection.
- **Evidence:** Forward review: "The spec includes a specific caveat about reversion risk... The plan has no task for reversion detection. The staleness tracker (Task 7.3) detects drift but does not flag reversions specifically."
- **Action:** Update plan
- **Recommendation:** Add a task in Phase 2 (Task 2.2 extension) to implement reversion detection: hash recent CLAUDE.md entries, detect when new entry reverts recent changes (content matches version N-k, k > 0), flag for manual review with message "CLAUDE.md reversion detected: current hash matches version from X days ago. Please confirm intentional reversion and add explicit corpus entry if so." Surface via `node0-corpus audit` subcommand.

---

## Coverage Summary

**Forward (Spec→Plan):**
- Fully covered: 13 sections
- Partially covered: 4 sections
- Not covered: 1 section

**Reverse (Plan→Spec):**
- Spec-backed: 27 tasks
- Spec-implied: 5 tasks
- Infrastructure: 2 tasks
- Scope creep: 0 tasks
- Gold-plating: 1 task (EvalResult Phase 1 vs Phase 6, but acceptable)

**Context Alignment:**
- Aligned: 12 architecture decisions
- Contradicts: 3 decisions
- Unverifiable: 0 decisions

---

## Corroborated Issues Summary

Issues found by multiple reviewers (elevated priority):

1. **Error class imports wrong** — Forward context (pattern compliance), Reverse (not directly addressed), Context (identified location mismatch)
2. **Formula file path wrong** — Forward (implicit in data collection), Context (explicit path mismatch)
3. **Training job bead lifecycle underdefined** — Forward (explicit spec-plan gap), Context (design investigation deferred)
4. **gt prime modification missing** — Forward (explicit spec-plan gap), Forward performance requirements gap
5. **Model registry UI silently dropped** — Forward (explicit spec decision not included)
6. **Data sources incomplete** — Forward (spec coverage gap), Context (corpus collection only covers transcripts)

---

## Acceptance Criteria for Plan Update

Before plan is considered approved:

1. All P0 contradictions must be resolved:
   - [ ] Error class import paths corrected (Tests 2.5, 3.4, 4.2, 4.3)
   - [ ] Formula file paths corrected to `~/gt/.beads/formulas/` (Task 6.5)
   - [ ] `run_finetune.py` entry point module created (new task in Phase 1 or 3)

2. All corroborated P1 gaps must be addressed:
   - [ ] Data sources task added for codebase artifacts and operational data (Phase 2)
   - [ ] Training job bead lifecycle implementation tasked (Phase 3)
   - [ ] Model registry UI scope decision made and documented (Phase 6 or spec revision)
   - [ ] gt prime modification tasked (Phase 5)
   - [ ] Polecat rollout stage tasked (Phase 6)

3. Documentation updates:
   - [ ] Spec coverage matrix corrected to include all covered/partially-covered/uncovered sections
   - [ ] Technical Risks section explicitly documents deferred V1+ work (cross-operator incentive model)
   - [ ] Component Integration Map updated to show gt prime modification status

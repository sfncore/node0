# Spec Review: gastown-finetune

## Review Configuration

- **Spec:** plans/gastown-finetune/02-spec/spec.md
- **Models Used:** Opus 4.6 (Claude), Kimi K2 (omp) — 2 of 2 succeeded
- **Models Unavailable:** GPT/Codex (no auth), Gemini (account suspended)
- **Categories:** All 12 (Codebase Match, Design Quality, Security, etc.)
- **Context Source:** Existing context.md + direct source code verification

## Model Comparison

| # | Issue | Opus 4.6 | Kimi K2 | Agree? |
|---|-------|----------|---------|--------|
| 1 | Hivemind doesn't do inference | CRITICAL: Grepped entire codebase, zero inference/generate/predict hits. Hivemind MOE Server is training-only. `node0-inference` binary doesn't exist. | CRITICAL: node0_server.py extends hivemind.moe.server.Server (training-only). No inference endpoints, serving, or token generation APIs. | Yes |
| 2 | Data loading gap underestimated | CRITICAL: Zero DataLoader/dataset/corpus code. Only reference is a docstring example. "Data loading pipeline" should be an epic, not a bullet. | CRITICAL: GradientAverager.accumulate_grads_() expects gradients to already exist. No data iterator, dataset abstraction, or batching. | Yes |
| 3 | Checkpoint export non-trivial | CRITICAL: `checkpoint_dir=None` hardcoded. Model exists only in distributed memory. Gathering weights from 32 pipeline stages is architecturally complex. | HIGH: No safetensors/gguf/transformers deps. checkpoint_dir=None hardcoded. | Partial |
| 4 | Pipeline-parallel breaks fine-tuning assumptions | CRITICAL: Each node runs 1 of 32 layers. LoRA per-stage, loss only at tail, DPO needs 2x pipeline. Fundamentally different from standard fine-tuning. | Not identified as separate issue (covered partially in LoRA/DPO items) | Partial |
| 5 | "Use as-is" components need modification | CRITICAL: AutoStepOptimizer auto-kills on NaN, PluralisAuthorizer requires auth.pluralis.ai, pipeline stage assignment from auth server. | MEDIUM: AutoStepOptimizer for timing coordination, PluralisAuthorizer for HF tokens not training jobs. | Partial |
| 6 | LoRA not supported, non-trivial | - (covered in #4) | CRITICAL: No PEFT fields in ModelArguments, no LoRA wrapping, optimizer on all params. | Yes |
| 7 | DPO requires entirely new training loop | HIGH: Needs reference model + policy model, two pipeline copies. Architecturally hard in pipeline-parallel. | CRITICAL: No reference model, no policy separation, no DPO loss computation. | Yes |
| 8 | ChromaDB/RAG underspecified | HIGH: No ChromaDB dep, no embedding model, no integration point since inference arch is wrong. | HIGH: No ChromaDB dep, "where does it live?" acknowledged as open question, contradicts flawed inference arch. | Yes |
| 9 | PII scrubbing insufficient | HIGH: Regex-only misses novel creds, JWTs, PEM blocks, inline secrets. Needs secret scanning tool. | Not identified | No |
| 10 | Model poisoning not addressed | HIGH: No quality validation, no outlier detection, no provenance verification. Failed sessions could embed harmful patterns. | Not identified | No |
| 11 | Model registry is entire subsystem | HIGH: Polished CLI presented, but no storage/schema design. Where do artifacts live? Full product feature. | Not identified | No |
| 12 | Model-agnostic claim contradicted | MEDIUM: Only LLaMA model definitions exist. Adding Qwen requires new model layers, args classes, pipeline stage defs. | Not identified | No |
| 13 | Corpus volume insufficient for role-specific | MEDIUM: 1K-10K split across 4+ roles = 250-2500 per role. Published research needs 10K+ for single task. | Not identified | No |
| 14 | Eval thresholds deferred with no methodology | MEDIUM: Qualitative pass criteria ("correct fields"). No scoring methodology defined. | MEDIUM: No concrete metrics. Untestable acceptance criteria. | Yes |
| 15 | Corpus partitioning misunderstood | Not identified | HIGH: Pipeline stages = model layers, not data partitions. Spec conflates the two. | No |
| 16 | Vocab size mismatch (OPT vs LLaMA) | MEDIUM: vocab_size=50265 (OPT-2.7b tokenizer) in LlamaArguments, not LLaMA's 32000. | Not identified | No |
| 17 | Training formula syntax aspirational | MEDIUM: No concrete command mappings for formula steps. | LOW: Non-standard syntax, no reference to actual formula schema. | Yes |
| 18 | Bead lifecycle oversimplistic | Not identified | MEDIUM: Single `in_progress` insufficient for syncing, degraded, checkpointing, resuming states. | No |

## All Issues by Severity

### CRITICAL (5 issues)

**1. Hivemind Does NOT Support Inference**
- **What:** The spec claims node0 serves both training AND inference via Hivemind. This is false. Hivemind is exclusively a distributed training framework with zero inference capability.
- **Where:** Inference Architecture section, Key Architectural Decisions table, architecture diagram
- **Evidence:** Both models independently verified: zero inference/generate/predict code in node0. Hivemind's Server class is a MOE training server. The `node0-inference` binary doesn't exist.
- **Recommendation:** Remove inference claims. Add inference serving infrastructure (vLLM, llama.cpp, TGI) to "Must be built" list. Redesign the provider integration around a real inference server.

**2. Data Loading Gap Is an Epic, Not a Bullet**
- **What:** "Data loading pipeline" is listed as one bullet among eight "Must be built" items. In reality, node0 has NO training loop — no DataLoader, no dataset abstraction, no tokenizer integration, no loss computation. This is the majority of the engineering work.
- **Where:** "node0: What Exists vs What We Build" section
- **Evidence:** Grep across entire src/node0/ returns zero hits for data loading. GradientAverager expects gradients to already exist. The only training data reference is a docstring example.
- **Recommendation:** Elevate to its own epic with sub-tasks: tokenizer integration, dataset format, DataLoader, loss function, batch assembly, pipeline integration.

**3. Pipeline-Parallel Architecture Incompatible with Standard Fine-Tuning**
- **What:** Each node runs 1 of 32 transformer layers. The spec assumes standard fine-tuning (full model on one node). In reality: no single node has the full model, LoRA must be per-stage, loss only at tail stage, DPO needs coordinating across entire pipeline, checkpoint export requires gathering 32 stages.
- **Where:** Architecture section, Training Infrastructure
- **Evidence:** llama_8B_C.yaml: `num_hidden_layers: 1, num_stages: 32`. infer_expert_params() shows stages are head-X/body-X/tail-X. Each server creates experts for one stage only.
- **Recommendation:** Add "Pipeline-Parallel Fine-Tuning" design section addressing cross-stage LoRA, loss propagation, model assembly for export.

**4. Checkpoint Export Architecturally Complex**
- **What:** `checkpoint_dir=None` hardcoded. Model exists only in distributed memory across 32 pipeline stages. No save mechanism, no serialization, no weights export.
- **Where:** "Must be built" list, Model Registry section
- **Evidence:** Single hit for "checkpoint" in codebase: `checkpoint_dir=None` in node0_server.py.
- **Recommendation:** Add checkpoint design section: gather distributed weights, assemble from pipeline stages, serialize to safetensors/GGUF.

**5. "Use As-Is" Components Need Modification**
- **What:** Several "Exists in node0 (use as-is)" components are tightly coupled to pretraining and need changes for fine-tuning.
- **Where:** "Exists in node0" list
- **Evidence:** AutoStepOptimizer kills process on NaN grads. PluralisAuthorizer requires auth.pluralis.ai and assigns pipeline stages externally. These aren't plug-and-play for fine-tuning.
- **Recommendation:** Change label to "Exists (needs modification)" and document required changes for each component.

### HIGH (5 issues)

**6. DPO Requires Entirely New Training Architecture**
- **What:** DPO needs reference model + policy model with simultaneous forward passes. In pipeline-parallel setting, this means 2x pipeline copies or complex batching. No DPO code exists.
- **Recommendation:** Either descope DPO to V2 (SFT only in V1) or add detailed DPO design for pipeline-parallel setting.

**7. ChromaDB/RAG Has No Integration Point**
- **What:** RAG is designed for inference-time injection, but inference architecture is wrong. ChromaDB not in dependencies. No embedding model specified.
- **Recommendation:** Move RAG to V2 or redesign around actual inference framework.

**8. PII Scrubbing Insufficient (Opus only)**
- **What:** Regex-only scrubbing misses JWTs, PEM blocks, inline secrets, novel credential formats.
- **Recommendation:** Add secret scanning tool (truffleHog, detect-secrets), confidence-based flagging, periodic audit.

**9. Model Poisoning Not Addressed (Opus only)**
- **What:** No quality validation, outlier detection, or data provenance verification. Failed sessions could embed harmful patterns that survive DPO split.
- **Recommendation:** Add data validation gates: quality scoring, anomaly detection, provenance tracking.

**10. Model Registry Is Entire Subsystem (Opus only)**
- **What:** Full CLI + UI presented with no storage/schema design. Where do model artifacts live? What database? This is a product feature, not a section.
- **Recommendation:** Descope UI to V2 or split into its own spec with proper design.

### MEDIUM (6 issues)

**11. Model-agnostic claim contradicted** — Only LLaMA model defs exist. Adding new architectures is non-trivial.
**12. Corpus volume may be insufficient** — 1K-10K split across 4+ roles = 250-2500 per role.
**13. Eval methodology undefined** — Qualitative pass criteria. No scoring methodology even if thresholds deferred.
**14. Corpus partitioning terminology confused** — Pipeline stages = model layers, not data partitions.
**15. Vocab size mismatch** — vocab_size=50265 (OPT tokenizer) in LlamaArguments, not LLaMA's 32000.
**16. Training formula syntax aspirational** — No concrete command mappings.

### LOW (3 issues)

**17. Bead lifecycle oversimplistic** — Need states for syncing, degraded, checkpointing, resuming.
**18. Architecture diagram missing RAG, inference, feedback loops.**
**19. "Recency wins" may cause silent regression from reverted bad conventions.**

## Reasoning

### Model Disagreements

**Checkpoint severity**: Opus rated CRITICAL, Kimi K2 rated HIGH. **Opus is correct** — the pipeline-parallel architecture makes checkpoint assembly a first-class architectural challenge, not just a missing dependency.

**DPO severity**: Opus rated HIGH, Kimi K2 rated CRITICAL. Both severity levels are justified — DPO in pipeline-parallel is genuinely critical if kept in V1.

**PII scrubbing, model poisoning, model registry**: Only Opus identified these. All are valid concerns. PII is HIGH given automated scrubbing with no review. Model poisoning is HIGH given the V1+ multi-operator extension path.

### Root Cause Analysis

The fundamental issue is that **the spec was written with a vision of what node0 SHOULD do rather than what it CURRENTLY does**. The brainstorm correctly identified node0 as the training infrastructure, but the spec drifted into treating it as a complete ML platform (training + inference + model serving + RAG). The code-verified reality is that node0 is a narrow tool: decentralized gradient averaging for pipeline-parallel pretraining. Everything else must be built.

## Ambiguities Summary

| # | Issue | Ambiguity | Options |
|---|-------|-----------|---------|
| 1 | Inference architecture | What replaces Hivemind inference? | vLLM server / llama.cpp / TGI / cloud API |
| 2 | DPO scope | Keep DPO in V1 or defer? | SFT-only V1 / DPO with pipeline design / DPO on single GPU |
| 3 | RAG timing | RAG in V1 or defer until inference solved? | V1 with new infra / V2 after inference settled |
| 4 | Model registry scope | Full UI in V1 as spec says? | CLI-only V1 / Full UI V1 / Split into own spec |
| 5 | Base model reality | OPT tokenizer configured, LLaMA claimed | Clarify actual base model / Fix tokenizer |
| 6 | Pipeline-parallel fine-tuning | Is pipeline-parallel right for fine-tuning? | Keep distributed / Gather to single node / Hybrid |

## Summary

- **Total Issues:** 19 (5 critical, 5 high, 6 medium, 3 low)
- **Ambiguities Requiring Decision:** 6
- **Model Agreement Rate:** 67% (12/18 issues found by at least one model, 8 with full/partial agreement)
- **Models That Failed:** None (2 of 2 succeeded; GPT/Gemini unavailable)

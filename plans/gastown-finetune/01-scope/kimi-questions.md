# Kimi K2 Analysis: gastown-finetune

## User Advocate Perspective

### What would users assume this does?

1. **Q: Will this let me train a model that understands Gas Town's specific workflows without me having to explain them every time?**
   - *Why it matters:* Users expect fine-tuning to capture institutional knowledge—bead patterns, formula structures, session conventions—so the model "just knows" how Gas Town works.

2. **Q: Can I contribute my own session transcripts and have them improve the model, or is this a one-shot training run?**
   - *Why it matters:* Users expect continuous learning; if their contributions don't feed back into the model, they'll feel their work is wasted.

3. **Q: Will the fine-tuned model replace the current Claude integration, or work alongside it?**
   - *Why it matters:* Users need clarity on whether they're gaining a new tool or losing an existing one—uncertainty creates resistance.

4. **Q: Does "distributed" mean I can run this on my laptop, or do I need a data center?**
   - *Why it matters:* Node0 supports 16GB+ GPUs, but users may assume "distributed fine-tuning" requires enterprise infrastructure they'll never have.

5. **Q: Will the model be able to generate valid formulas and bead issues that actually work in the system?**
   - *Why it matters:* Users expect functional output, not plausible-looking garbage that fails validation when imported.

6. **Q: Can I ask the fine-tuned model to explain why a particular bead was created or what a formula does?**
   - *Why it matters:* Interpretability is crucial for trust; users won't rely on a model that can't explain its reasoning in Gas Town terms.

7. **Q: Will this understand the relationships between beads—dependencies, blockers, epics—or just treat them as isolated documents?**
   - *Why it matters:* Gas Town's value is in the graph of work; a model that doesn't understand dependencies misses the point.

8. **Q: If I fine-tune on my rig's data, will the model leak my private bead content to other nodes in the distributed network?**
   - *Why it matters:* Privacy expectations are absolute; users won't participate if they can't control data visibility.

9. **Q: Can I use this to generate session transcripts for training new polecats, or is it only for automated agents?**
   - *Why it matters:* Users may want dual-purpose output—both automation and human education—but the brief focuses only on agents.

10. **Q: Will the model know when it doesn't know something about Gas Town, or will it hallucinate bead structures?**
    - *Why it matters:* Confidence in wrong answers is worse than no answer; users need the model to signal uncertainty.

### What similar features exist elsewhere?

11. **Q: How is this different from just using Claude with a RAG system over my beads?**
    - *Why it matters:* Users need to understand the value proposition—why fine-tuning beats context-window approaches they've heard of.

12. **Q: Will this be like GitHub Copilot where it suggests completions, or more like ChatGPT where I have a conversation?**
    - *Why it matters:* Interaction model shapes daily workflow; users need to know if this is assistive or autonomous.

13. **Q: Can I export the fine-tuned model and run it locally like Llama or Mistral, or am I locked into a service?**
    - *Why it matters:* Users have been burned by vendor lock-in; local execution is often a hard requirement.

### What would disappoint them?

14. **Q: If the model generates a bead that looks perfect but fails validation, do I get helpful error messages or just "invalid"?**
    - *Why it matters:* Opaque failures waste time; users expect the model to help debug its own mistakes.

15. **Q: Will I need to learn YAML configuration and distributed systems concepts just to use this?**
    - *Why it matters:* The brief mentions YAML configs and Hivemind parameters—users may fear this is only for infrastructure engineers.

16. **Q: If training takes days and fails, do I lose everything or can I resume?**
    - *Why it matters:* The context mentions no explicit checkpoint saving—users will be furious if long jobs die without recovery.

17. **Q: Will the model be updated regularly with new Gas Town patterns, or will it become outdated as the system evolves?**
    - *Why it matters:* Stale training data produces worse results over time; users need a freshness guarantee.

### What are they trying to accomplish?

18. **Q: As a mayor, do I want an agent that can delegate work to polecats, or one that replaces polecats for certain tasks?**
    - *Why it matters:* The brief blurs automation vs. augmentation; mayors need clarity on whether this empowers or displaces their team.

19. **Q: As a polecat, will this help me understand complex bead dependencies faster, or will it make me feel replaceable?**
    - *Why it matters:* Emotional response determines adoption; polecats won't engage with a tool that threatens their role.

20. **Q: As a witness, can I use the fine-tuned model to verify bead completeness and catch missing acceptance criteria?**
    - *Why it matters:* Witnesses have specific validation responsibilities; the brief doesn't address how this serves their role.

21. **Q: Do I want the model to generate initial bead drafts that I refine, or complete ready-to-use beads?**
    - *Why it matters:* Degree of automation affects trust; users may want collaborative drafting, not black-box generation.

22. **Q: Can I use this to migrate old bead issues to new formats, or is it only for creating new content?**
    - *Why it matters:* Legacy data migration is a common need; the brief focuses only on new corpus generation.

### Emotional state?

23. **Q: Will using this make me feel more in control of Gas Town, or like I'm ceding authority to an opaque system?**
    - *Why it matters:* Agency is core to the operator role; automation that feels like loss of control will be rejected.

24. **Q: If the model suggests a bead structure that contradicts my intuition, will the system help me understand why or just override me?**
    - *Why it matters:* Users need to feel heard; being overruled by AI breeds resentment and disuse.

25. **Q: Will I feel proud contributing my transcripts to the corpus, or worried about exposure of sensitive session content?**
    - *Why it matters:* Emotional safety determines contribution rates; fear of exposure kills the corpus-building phase.

### What happens before?

26. **Q: What do I need to collect before I can start—just beads, or do I need session transcripts, formula specs, and workflow documentation too?**
    - *Why it matters:* Users need a clear inventory of what's required; unclear prerequisites cause procrastination.

27. **Q: Do I need to clean or annotate my existing beads before they're useful for training, or can I dump them as-is?**
    - *Why it matters:* Data preparation burden determines participation; if it's too hard, users won't contribute.

28. **Q: If I'm starting a new rig with no history, can I still benefit from this or do I need months of bead data first?**
    - *Why it matters:* New users need a viable path; the corpus requirement may exclude early-stage rigs.

### What happens after?

29. **Q: Once the model is fine-tuned, how do I actually use it—do I type in a chat, or does it integrate into my existing bead commands?**
    - *Why it matters:* The brief describes infrastructure setup but not the user interface; users need to know how to interact with the result.

30. **Q: If the fine-tuned model generates a bead, does it automatically get created in my rig or do I review and approve first?**
    - *Why it matters:* Users need control over their rig; automatic creation without review would be alarming.

31. **Q: Can I give feedback on the model's output to improve it over time, or is the training done and fixed?**
    - *Why it matters:* Users expect iterative improvement; a static model feels like a dead end.

32. **Q: Will other rigs benefit from my training data, or is each fine-tuned model isolated to my rig?**
    - *Why it matters:* Users may want to contribute to a collective improvement or keep their model private—both need support.

### What weird things might they try?

33. **Q: What happens if I feed the model its own generated beads as training data—will it spiral into incoherence?**
    - *Why it matters:* Users experiment; recursive training on synthetic data is a known failure mode that needs handling.

34. **Q: Can I train on beads from multiple unrelated rigs with different conventions, or will that confuse the model?**
    - *Why it matters:* Users may want to bootstrap with public bead data; mixing conventions without guidance produces garbage.

35. **Q: What if I include beads with mistakes or abandoned work—will the model learn to reproduce errors?**
    - *Why it matters:* Real data is messy; users need guidance on data curation or automatic filtering.

36. **Q: Can I use this to generate fake session transcripts that make it look like I did work I didn't do?**
    - *Why it matters:* Bad actors exist; the system needs to distinguish authentic from synthetic transcripts.

37. **Q: What happens if two users simultaneously fine-tune on conflicting bead conventions—whose model wins?**
    - *Why it matters:* Distributed systems have convergence issues; users need to know if their local model can diverge.

38. **Q: Can I ask the model to generate beads for projects that don't exist yet, essentially using it for planning?**
    - *Why it matters:* Users repurpose tools; the brief focuses on existing corpus but planning is a natural extension.

### What if they use it wrong?

39. **Q: If I train on too few beads, will the model fail gracefully or produce confidently wrong Gas Town syntax?**
    - *Why it matters:* Small data regimes are common early on; users need guardrails, not garbage output.

40. **Q: What if I accidentally include API keys or secrets in my session transcripts—will they be extractable from the model?**
    - *Why it matters:* Data leakage from training is a real risk; users need sanitization guidance or automatic detection.

41. **Q: If I stop training halfway through, do I get a partially working model or nothing at all?**
    - *Why it matters:* Interrupted jobs are common; users need to know if partial results are usable.

42. **Q: Can I overfit the model to my specific rig's quirks so it fails on standard Gas Town patterns?**
    - *Why it matters:* Overfitting is a real risk; users need validation metrics to detect when their model is too narrow.

### What if they change their mind?

43. **Q: If I start training and realize my corpus is bad, can I stop and restart with different data without losing compute credits?**
    - *Why it matters:* Sunk cost fallacy is real; users need escape hatches to correct mistakes.

44. **Q: Can I revert to a previous version of the model if the new one starts generating worse beads?**
    - *Why it matters:* Model degradation happens; users need versioning like they have with beads themselves.

45. **Q: If I decide I don't want my data in the corpus anymore, can it be removed or is it baked in permanently?**
    - *Why it matters:* Right to be forgotten is a legal and ethical requirement; training data removal is technically hard but expected.

### Who might struggle?

46. **Q: Will users without GPU access be able to participate in fine-tuning, or is this only for people with hardware?**
    - *Why it matters:* Node0 requires 16GB+ GPUs; this excludes many potential contributors and centralizes power.

47. **Q: Do I need to understand machine learning concepts like "gradient averaging" and "pipeline parallelism" to use this?**
    - *Why it matters:* The context is full of ML jargon; users may self-exclude if they feel underqualified.

48. **Q: Will users who primarily work in languages other than English be able to contribute transcripts and benefit from the model?**
    - *Why it matters:* Gas Town may have international users; English-only training creates exclusion.

49. **Q: Can users with visual impairments interact with this through screen readers, or is it terminal-based with ASCII diagrams?**
    - *Why it matters:* Accessibility is often an afterthought in developer tools; explicit consideration is needed.

50. **Q: Will users on slow or intermittent internet connections be able to participate in distributed training?**
    - *Why it matters:* Node0 requires P2P communication; users in rural or constrained environments may be excluded.

### What assumptions are we making?

51. **Q: Are we assuming all Gas Town users are software developers comfortable with Docker, YAML, and CLI tools?**
    - *Why it matters:* The setup involves Docker, YAML configs, and Python—this excludes non-technical users who might benefit.

52. **Q: Are we assuming users want automation, or might some prefer the current manual bead workflow as a deliberate practice?**
    - *Why it matters:* Some users may value the mindfulness of manual work; forced automation alienates them.

53. **Q: Are we assuming that more training data is always better, regardless of quality or relevance?**
    - *Why it matters:* Data volume vs. quality is a known ML tradeoff; the brief emphasizes corpus size without addressing curation.

54. **Q: Are we assuming users trust the distributed network with their training data, or have we verified this?**
    - *Why it matters:* Trust is assumed but not earned; users may not want their proprietary workflows on a P2P network.

55. **Q: Are we assuming that a single fine-tuned model can serve all Gas Town roles—mayor, polecat, witness—equally well?**
    - *Why it matters:* Different roles have different needs; a one-size-fits-all model may serve none well.

---

## Product Designer Perspective

### Information Architecture

*What info does the user need? Hierarchy? Hidden vs visible?*

1. **Where does the user discover fine-tuning is possible?**
   *Why it matters:* If buried in settings, no one uses it; if too prominent, it distracts from core workflows.

2. **Should corpus statistics be visible before starting a job?**
   *Why it matters:* Users need confidence their data is sufficient before committing GPU hours.

3. **How do we surface "training corpus health" — duplicates, stale formulas, gaps?**
   *Why it matters:* Garbage in, garbage out; users should know data quality before training begins.

4. **What metadata about a training run is essential vs. nice-to-have?**
   *Why it matters:* Cognitive load — too much detail overwhelms, too little leaves users guessing.

5. **Should historical training runs be searchable or just browsable?**
   *Why it matters:* Power users need to compare experiments; casual users need simplicity.

6. **How do we represent distributed node status in a single view?**
   *Why it matters:* Users need to understand "the system" not just "their node" — but without complexity.

7. **Where do checkpoint versions live in the mental model?**
   *Why it matters:* Checkpoints are outputs, but users may think of them as "saved models" — alignment matters.

8. **Should corpus composition (formulas vs. workflows vs. transcripts) be visualized?**
   *Why it matters:* Users need to understand what they're actually training on.

9. **How do we handle "partial" or "in-progress" corpus states?**
   *Why it matters:* Users may start training before corpus is "complete" — we need clear signaling.

10. **What happens to training job history when a rig is archived?**
    *Why it matters:* Long-term discoverability vs. data retention policies.

11. **Should we show estimated time-to-completion, and if so, how?**
    *Why it matters:* Distributed training is unpredictable — false precision erodes trust.

12. **How do we represent "global" vs. "local" training progress?**
    *Why it matters:* Users care about "when will MY model be ready" not just aggregate metrics.

### Interaction Design

*How trigger fine-tuning? Required vs optional inputs? Success/failure feedback?*

1. **Is fine-tuning a button, a command, or a workflow wizard?**
   *Why it matters:* Triggers signal importance and complexity; a button feels lightweight, a wizard feels serious.

2. **What is the ONE required input to start fine-tuning?**
   *Why it matters:* Every required field is friction; we need to know the true minimum.

3. **Should users name their training runs, or auto-generate names?**
   *Why it matters:* Naming is cognitive work; auto-names reduce friction but hurt discoverability.

4. **How do we handle "corpus not ready" when user tries to start?**
   *Why it matters:* Block with error, or allow queueing for later? Each sends a different message.

5. **What does "pause" mean in distributed training?**
   *Why it matters:* Users expect pause/resume, but distributed state makes this complex — we need honest UX.

6. **Should users configure hyperparameters, or trust defaults?**
   *Why it matters:* Exposing knobs empowers experts but intimidates novices; progressive disclosure is key.

7. **How do we show "your job is queued behind 3 others"?**
   *Why it matters:* Waiting without context feels broken; transparency builds patience.

8. **What happens when a node drops mid-training?**
   *Why it matters:* Users need to know if their job is at risk, without alarmist messaging.

9. **Should training be cancellable, and what are the consequences?**
   *Why it matters:* Partial checkpoints may be useless — users need to understand sunk cost.

10. **How do we handle "training succeeded but evaluation failed"?**
    *Why it matters:* Partial success states are confusing; we need clear outcome communication.

11. **What feedback does a user get when their model is "deployed"?**
    *Why it matters:* "Deployed" is abstract — users need to know HOW to use their new model.

12. **Should we allow "dry run" or "validation only" before real training?**
    *Why it matters:* Catches configuration errors early, but adds steps to the workflow.

### User Flows

*Happy path? Error path? Edge cases (empty corpus, stale data)?*

1. **What does the happy path LOOK like from start to deployed model?**
   *Why it matters:* The ideal journey defines all other flows; we need to visualize it clearly.

2. **What happens when a user tries to fine-tune with an empty corpus?**
   *Why it matters:* Empty states need to guide toward action, not just say "nothing here."

3. **How do we handle "corpus has formulas but no workflows"?**
   *Why it matters:* Partial data may produce lopsided models; users need warnings, not just errors.

4. **What if the corpus hasn't been updated in 30 days?**
   *Why it matters:* Stale data may no longer represent current practices — freshness matters.

5. **What does "retrain from checkpoint" look like vs. "start fresh"?**
   *Why it matters:* Forking vs. continuing are different mental models; UI should reflect the difference.

6. **How does a user recover from a failed training job?**
   *Why it matters:* Failure is inevitable; the path forward determines whether users persist or abandon.

7. **What if multiple users trigger training on the same corpus simultaneously?**
   *Why it matters:* Coordination or collision? The UX should make the system's behavior predictable.

8. **How do we handle "training succeeded but model is worse"?**
   *Why it matters:* Not all successful jobs produce good outcomes; users need comparison tools.

9. **What does "export model" look like for use outside Gas Town?**
   *Why it matters:* Portability is a feature; the export flow should feel like a gift, not an escape hatch.

10. **How do we handle "I want to train on MY rig's data only" vs. "all Gas Town data"?**
    *Why it matters:* Scope selection affects model personality; users need clear boundaries.

11. **What happens when a user wants to schedule recurring training?**
    *Why it matters:* Continuous learning is powerful but complex; automation needs guardrails.

12. **How do we handle "training is taking 10x longer than estimated"?**
    *Why it matters:* Estimates are guesses; when reality diverges, users need honest updates, not spinners.

### Visual & Layout

*Where does this live in Gas Town? Own rig or existing workflow?*

1. **Does fine-tuning deserve its own top-level navigation item?**
   *Why it matters:* Navigation real estate signals importance; misplaced items are never found.

2. **Should training jobs appear in the main beads view, or a separate panel?**
   *Why it matters:* Beads are the core metaphor; training jobs may not fit the bead mental model.

3. **Where do corpus management and training job management live relative to each other?**
   *Why it matters:* They're related but distinct; proximity implies relationship, separation implies independence.

4. **What does a "training in progress" indicator look like in the main UI?**
   *Why it matters:* Background jobs need ambient awareness without distraction.

5. **Should model variants be displayed like beads, or like a separate library?**
   *Why it matters:* Consistency with existing patterns reduces learning curve, but may force wrong abstractions.

6. **How do we visualize distributed nodes — map, list, or abstract representation?**
   *Why it matters:* Geographic accuracy may not matter; health and contribution do.

7. **What does the "model card" or "training run card" look like?**
   *Why it matters:* Summary views need to communicate status, progress, and actionability at a glance.

8. **Should corpus composition be a pie chart, a list, or something else?**
   *Why it matters:* Visual encoding affects understanding; wrong charts mislead.

9. **Where does "deployed model" live once training is complete?**
   *Why it matters:* Outputs need homes; undiscoverable models are wasted effort.

10. **How do we handle "compare two training runs" visually?**
    *Why it matters:* Side-by-side, overlay, or tabular? Each supports different comparison tasks.

11. **What does the "select corpus" interface look like?**
    *Why it matters:* Corpus selection may involve filters, tags, date ranges — complexity needs containment.

12. **Should there be a "fine-tuning dashboard" or is that overkill?**
    *Why it matters:* Dashboards aggregate; but aggregation can obscure the specific status users care about.

### States & Transitions

*Job states (preparing, training, evaluating, deployed)? How to move between them?*

1. **What does "preparing" look like and how long should it feel?**
   *Why it matters:* Preparation may involve data validation, node discovery — users need progress indicators.

2. **How do we represent "waiting for nodes to join"?**
   *Why it matters:* Distributed systems have coordination delays; waiting should feel purposeful, not stuck.

3. **What visual distinction exists between "training" and "averaging" phases?**
   *Why it matters:* Node0 alternates between local compute and distributed sync; users may want visibility.

4. **How do we show "evaluation in progress" vs. "training in progress"?**
   *Why it matters:* Different phases, different stakes; clear distinction manages expectations.

5. **What does "ready for deployment" look like?**
   *Why it matters:* Completion should feel like achievement; unclear endings feel anticlimactic.

6. **How do we handle "deployment failed but training succeeded"?**
   *Why it matters:* Partial success needs clear next steps, not just error red.

7. **What does "archived" or "retired" model state look like?**
   *Why it matters:* Old models accumulate; we need graceful degradation, not clutter.

8. **How do we represent "checkpoint saved" moments during training?**
   *Why it matters:* Checkpoints are invisible milestones; surfacing them builds confidence.

9. **What does "rollback to previous checkpoint" look like?**
   *Why it matters:* Recovery is a feature; the path back should be as clear as the path forward.

10. **How do we show "this model is now the default for new sessions"?**
    *Why it matters:* Default changes affect behavior; users need to know what they're getting.

11. **What does "training job expired" or "timed out" look like?**
    *Why it matters:* Distributed jobs may stall; expiration needs clear communication and recovery path.

12. **How do we represent "model is being used in production" vs. "experimental"?**
    *Why it matters:* Production models have higher stakes; visual distinction prevents accidents.

### Key Tensions to Resolve

| Tension | Description |
|---------|-------------|
| **Simplicity vs. Control** | Users want one-click training but also hyperparameter tuning |
| **Transparency vs. Overwhelm** | Distributed systems are complex; how much do users need to see? |
| **Immediate vs. Long-term** | Training takes hours/days; how to maintain engagement? |
| **Individual vs. Collective** | My rig's data vs. all Gas Town data — scope clarity matters |
| **Automation vs. Oversight** | How much can we automate before users feel loss of control? |

### Open Questions for Stakeholders

1. Who is the primary user? Core Gas Town contributors, or external rig operators?
2. Is fine-tuning a daily activity or a monthly activity?
3. What is the cost of a failed training run? (Time, money, trust)
4. How many training jobs do we expect to be active simultaneously?
5. What does "success" look like for the first version? (Shipped model, or infrastructure only?)

---

## Domain Expert Perspective

### Domain Concepts

#### Terminology & Undefined Concepts

1. **What exactly is a "Gas Town agent"?**
   - Why it matters: The entire corpus is being built to train agents, but we lack a crisp definition of agent capabilities, autonomy boundaries, and interaction patterns.

2. **What distinguishes "formulas" from "workflows" in Gas Town?**
   - Why it matters: These are core abstractions; conflating them will contaminate the training corpus with inconsistent patterns.

3. **What is a "bead" vs "beads pattern"?**
   - Why it matters: The corpus includes "beads patterns" but the ontology is unclear—are beads tasks, issues, state containers, or all three?

4. **What is "Protocol Learning" in practice?**
   - Why it matters: Node0's documentation mentions this as the high-level goal, but it's never defined operationally.

5. **What constitutes a "session transcript" for training purposes?**
   - Why it matters: Raw logs vs. curated dialogues vs. structured traces have different training dynamics and data quality requirements.

6. **What is the relationship between "rig" and "node0"?**
   - Why it matters: The project uses both terms; unclear mapping will lead to confused corpus organization.

7. **What does "native Gas Town agent" mean technically?**
   - Why it matters: Native could mean "runs in Gas Town infra" or "thinks in Gas Town abstractions"—these require different training objectives.

8. **What is the "Dolt-native beads" data model?**
   - Why it matters: The corpus will be stored in Dolt; we need to understand the schema to structure training data correctly.

9. **What is the "Hivemind" abstraction layer responsible for?**
   - Why it matters: Node0 builds on Hivemind; unclear boundaries mean we can't reason about failure modes or optimization opportunities.

10. **What is "PowerSGD" and why is it the default compression?**
    - Why it matters: Compression affects gradient quality; we need to know if this is a constraint or an optimization for our training setup.

#### Missing Concepts

11. **Where is the definition of "agent convention" in Gas Town?**
    - Why it matters: The brief mentions agent conventions as corpus content, but no canonical specification exists.

12. **What is the expected agent lifecycle (spawn, operate, terminate)?**
    - Why it matters: Training agents requires modeling stateful, long-running processes; we need to know the lifecycle to generate appropriate training sequences.

13. **What is the "Gas Town mental model" an agent should internalize?**
    - Why it matters: Fine-tuning for domain expertise requires understanding the target cognitive frame.

14. **How do agents handle partial observability in Gas Town?**
    - Why it matters: Distributed systems have incomplete state; agents need training on information gathering vs. action taking.

15. **What is the canonical error handling pattern for Gas Town agents?**
    - Why it matters: Error recovery sequences are high-value training data; we need to know the expected patterns.

### Prior Art

#### Existing Methods & Expectations

16. **Why LoRA/QLoRA instead of full fine-tuning?**
    - Why it matters: Node0 does distributed full-model training; the brief implies fine-tuning. We need to reconcile these approaches.

17. **Is instruction tuning the right paradigm, or do we need RLHF/DPO?**
    - Why it matters: Agent behavior requires preference alignment, not just instruction following.

18. **What is the expected base model for fine-tuning?**
    - Why it matters: The brief doesn't specify; different base models have different optimal fine-tuning strategies.

19. **Are there existing Gas Town agent implementations to study?**
    - Why it matters: Prior implementations reveal implicit conventions not captured in documentation.

20. **What distributed training frameworks were evaluated before Hivemind?**
    - Why it matters: Understanding rejected alternatives reveals constraints and requirements.

21. **What is the expected inference cost target for a Gas Town agent?**
    - Why it matters: Model size and architecture decisions flow from deployment constraints.

22. **Are there existing evaluations of LLM agents in similar orchestration domains?**
    - Why it matters: Benchmarks define success; we need comparable baselines.

23. **What data formats has Node0 historically used for training?**
    - Why it matters: Data pipeline compatibility depends on understanding existing conventions.

24. **What is the state of the art for "agent-native" fine-tuning?**
    - Why it matters: Toolformer, Gorilla, and similar work may inform our approach.

25. **What failure modes are documented for decentralized training?**
    - Why it matters: The corpus should include recovery patterns, which requires knowing common failures.

#### Past Failures & Lessons

26. **Has anyone attempted to train a Gas Town agent before? What failed?**
    - Why it matters: Repeated failures indicate fundamental blockers vs. implementation issues.

27. **What caused previous distributed training runs to collapse?**
    - Why it matters: Data quality issues, gradient explosion, or consensus failures inform corpus design.

28. **Are there known issues with Hivemind at scale?**
    - Why it matters: Infrastructure limitations constrain training corpus size and distribution.

29. **What is the track record of formula-based agent training?**
    - Why it matters: If formulas are a novel contribution, we lack validation data.

30. **Have session transcripts been collected before? Were they useful?**
    - Why it matters: Historical data quality informs curation priorities.

### Problem Depth

#### Root vs. Symptom

31. **Is the real problem "we need fine-tuning infrastructure" or "agents don't understand Gas Town"?**
    - Why it matters: Infrastructure is a means; the symptom is poor agent performance. We might solve the symptom without the infrastructure.

32. **Is the problem data scarcity or data organization?**
    - Why it matters: If Gas Town already generates sufficient traces, the problem is curation, not collection.

33. **Is the problem agent capability or agent reliability?**
    - Why it matters: Capability gaps need different training data than consistency gaps.

34. **Is the problem training data or inference-time context?**
    - Why it matters: More context at inference may reduce fine-tuning requirements.

35. **Is the problem model knowledge or model alignment?**
    - Why it matters: Knowledge gaps need continued pretraining; alignment gaps need preference tuning.

#### Related Problems Users Expect Solved

36. **Will users expect agents to explain their reasoning?**
    - Why it matters: Chain-of-thought requirements change training data structure.

37. **Will users expect agents to handle novel tools/workflows?**
    - Why it matters: Generalization requirements affect model architecture and training scope.

38. **Will users expect agents to collaborate with each other?**
    - Why it matters: Multi-agent scenarios require different training distributions.

39. **Will users expect agents to learn from feedback?**
    - Why it matters: Online learning requirements change the training/inference boundary.

40. **Will users expect agents to operate offline?**
    - Why it matters: Deployment constraints affect model size and quantization requirements.

#### Explicitly NOT Solving

41. **Are we training a general-purpose agent or a Gas Town specialist?**
    - Why it matters: Scope determines data diversity requirements.

42. **Are we solving the cold-start problem for new Gas Town users?**
    - Why it matters: Onboarding flows may need different training data than operational flows.

43. **Are we addressing multi-modal capabilities (vision, audio)?**
    - Why it matters: Modality scope affects model selection and data pipeline.

44. **Are we training for code generation or system orchestration?**
    - Why it matters: Different competencies require different corpus emphasis.

45. **Are we solving for latency or correctness?**
    - Why it matters: Optimization targets affect model architecture and serving infrastructure.

### Edge Cases

#### Unusual Valid Scenarios

46. **How should agents handle circular dependencies in beads?**
    - Why it matters: Real workflows have cycles; training data should include resolution patterns.

47. **How should agents behave when the DHT is partitioned?**
    - Why it matters: Network partitions are inevitable; agents need graceful degradation training.

48. **How should agents handle concurrent modifications to shared state?**
    - Why it matters: Distributed systems have race conditions; agents need conflict resolution training.

49. **How should agents respond to ambiguous user requests?**
    - Why it matters: Clarification dialogues are valuable training data.

50. **How should agents handle long-running operations that exceed context windows?**
    - Why it matters: State management across context boundaries requires specific training.

#### Data Quality & Contamination

51. **How do we prevent training on test/validation workflows?**
    - Why it matters: Data leakage invalidates evaluation.

52. **How do we handle PII in session transcripts?**
    - Why it matters: Privacy violations are non-negotiable; we need a policy before collection.

53. **How do we detect and exclude adversarial examples?**
    - Why it matters: Poisoning attacks on training data are a real threat.

54. **How do we ensure formula diversity without formula duplication?**
    - Why it matters: Near-duplicate training examples waste compute and bias the model.

55. **How do we handle version drift in Gas Town conventions?**
    - Why it matters: Evolving conventions create inconsistent training signals.

#### Model Collapse & Catastrophic Forgetting

56. **What base capabilities must be preserved during fine-tuning?**
    - Why it matters: We need to define the "do not forget" set for evaluation.

57. **How will we detect catastrophic forgetting during training?**
    - Why it matters: Early detection enables intervention before model degradation.

58. **What is the replay buffer strategy for maintaining base capabilities?**
    - Why it matters: Mixed training requires careful data mixing ratios.

59. **How do we prevent the model from overfitting to common formulas?**
    - Why it matters: Popular patterns can dominate the loss landscape.

60. **What is the rollback strategy if fine-tuning degrades performance?**
    - Why it matters: Production safety requires recovery options.

### Success Criteria

#### What Good Looks Like

61. **What is the minimum viable agent capability?**
    - Why it matters: Success thresholds determine when to stop training.

62. **What tasks must a Gas Town agent perform correctly 100% of the time?**
    - Why it matters: Critical path reliability requirements inform evaluation design.

63. **What is the acceptable error rate for non-critical tasks?**
    - Why it matters: Resource allocation depends on precision requirements.

64. **How do we measure "understanding" of Gas Town abstractions?**
    - Why it matters: Behavioral metrics may not capture conceptual alignment.

65. **What is the human baseline for Gas Town task completion?**
    - Why it matters: Agent performance is relative to human capability.

#### Metrics That Matter

66. **What is the target perplexity on Gas Town domain text?**
    - Why it matters: Perplexity correlates with domain understanding.

67. **What is the target accuracy on formula execution?**
    - Why it matters: Formula correctness is a core competency.

68. **What is the acceptable latency for agent responses?**
    - Why it matters: Deployment constraints affect model selection.

69. **How do we measure agent "helpfulness" vs. "harmlessness"?**
    - Why it matters: Tradeoffs between capability and safety need quantification.

70. **What is the target token efficiency (output quality per inference cost)?**
    - Why it matters: Operational costs scale with token usage.

#### Validation Strategy

71. **What is the holdout evaluation set composition?**
    - Why it matters: Evaluation validity depends on representative test data.

72. **How do we prevent overfitting to the evaluation set?**
    - Why it matters: Repeated evaluation creates indirect training signals.

73. **What is the human evaluation protocol?**
    - Why it matters: Automated metrics are insufficient for agent evaluation.

74. **How do we measure generalization to unseen workflows?**
    - Why it matters: Real-world utility depends on compositional generalization.

75. **What is the A/B testing strategy for deployed agents?**
    - Why it matters: Production validation requires controlled rollout.

### Critical Unknowns

The following questions block meaningful progress until answered:

1. **What is a Gas Town agent?** (Concept #1)
2. **Is this a data scarcity or data organization problem?** (Problem #32)
3. **What base model are we fine-tuning?** (Prior Art #18)
4. **What is the minimum viable capability?** (Success #61)
5. **What are we explicitly NOT solving?** (Scope #41-45)

Without answers to these five, the training corpus cannot be properly scoped, the infrastructure cannot be sized, and success cannot be defined.

---

## Cross-Perspective Themes (Kimi K2)

### Theme 1: Clarity of Purpose & Scope
**Appears in:** User Advocate (Q18-22, Q31, Q32, Q43-45), Product Designer (Stakeholder Q1, Q5), Domain Expert (Problem #31-45, Critical Unknown #1, #5)

Users, designers, and domain experts all need clear answers about what this system is actually for: automation vs. augmentation? New capability or replacement? Individual or collective? One-shot or continuous learning? The lack of scope clarity cascades into every other decision and undermines trust.

### Theme 2: Data Quality & Governance
**Appears in:** User Advocate (Q27, Q34-35, Q40, Q53), Product Designer (IA #3, Q2-3), Domain Expert (Prior Art #26-30, Edge Cases #51-55)

All perspectives grapple with how to ensure the training corpus is high-quality, representative, and safe. Users worry about including mistakes and secrets; designers need to surface data health; domain experts need policies for PII, duplicates, and version drift. This is foundational to the entire system.

### Theme 3: Accessibility & Barriers to Entry
**Appears in:** User Advocate (Q15, Q46-54), Product Designer (IA #1, Interaction #2, Layout #1), Domain Expert (Scope #42)

Whether it's GPU hardware, ML terminology, or UI placement—multiple perspectives identify concerns that non-technical or resource-constrained users may be excluded. This threatens both adoption and the diversity of the training corpus.

### Theme 4: Output Validation & Confidence
**Appears in:** User Advocate (Q5, Q6, Q10, Q14, Q39-40, Q45), Product Designer (Interaction #1-2, #10, User Flows #8-9), Domain Expert (Success #61-75)

Users need to trust that generated outputs are correct or at least come with confidence signals. Designers must decide how to surface validation results. Domain experts must define what "correct" even means. Without this, the entire system lacks credibility.

### Theme 5: Iteration & Feedback Loops
**Appears in:** User Advocate (Q2, Q21, Q31, Q44), Product Designer (Interaction #6, User Flows #5-6, States #9), Domain Expert (Success #39, Validation #72-75)

All perspectives want the system to be dynamic—users want to improve models over time, designers want UI that supports progression, domain experts need mechanisms for online learning and A/B testing. A static, one-shot system will feel dead on arrival.

### Theme 6: Explainability & User Agency
**Appears in:** User Advocate (Q6, Q23-24, Q33), Product Designer (IA #4-12, States #1-12), Domain Expert (Success #36, #64)

Users fear being overruled by an opaque system; designers need clear state transitions and progress signals; domain experts need metrics that capture understanding, not just behavior. Automation without agency kills adoption.

---

*Document consolidated from three Kimi K2 analysis files: kimi-user-advocate.tmp, kimi-product-designer.tmp, kimi-domain-expert.tmp*

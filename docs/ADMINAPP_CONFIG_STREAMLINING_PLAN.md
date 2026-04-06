# AdminApp config streamlining plan

## Goals

1. Make common workflows easy with sane defaults and a minimal UI.
2. Expose advanced knobs only when they are relevant to the selected task.
3. Ensure prompt precompute and prompt inference are fully resumable with immutable run settings.
4. Prevent any attempt to download Hugging Face assets on VA networks.

## Current behavior (baseline)

- The large-corpus precompute and inference tabs accept free-form JSON override blobs (`Config overrides`, `LLM overrides`) and include an advanced settings dialog with many backend sections.
- Precompute and inference already attempt resume by reusing `job_manifest.json` and backfilling missing overrides from prior manifests.
- Prompt precompute determines whether a retrieval index is needed from phenotype level + `single_doc_context`.
- Inference still wires `scjitter` through family labeler construction even though inference requests use `json_jitter=False`.
- Embedding and reranker models are constructed by passing names/paths directly to `SentenceTransformer` and `CrossEncoder`; without strict guards this can trigger HF download behavior when names are unresolved locally.

## Proposed product/UX changes

### 1) Split config into **Profiles + Advanced Overrides**

Adopt a two-layer model:

- **Profile (required, user-friendly):**
  - `workflow`: `active_learning`, `prompt_precompute`, `prompt_inference`, `round_inference_experiments`.
  - `backend`: `azure` or `local`.
  - `labeling_mode`: `family` or `single_prompt`.
  - `context_mode` for single-doc only (`rag` or `full`).
  - Optional "speed/quality" preset (`fast`, `balanced`, `high_recall`) mapped to vetted defaults.
- **Advanced overrides (optional):**
  - Hidden by default behind "Show advanced settings".
  - Validated against an allowlist per workflow.

This keeps common users out of JSON while preserving power-user flexibility.

### 2) Introduce workflow-scoped setting visibility

Define a **visibility matrix** so sections/fields appear only when meaningful:

- `prompt_precompute`: show retrieval/index/prompt context knobs; hide jitter and selection-bucket settings.
- `prompt_inference`: show LLM execution + output controls; hide selection/jitter/diversity/disagreement.
- `active_learning`: show full selection + uncertainty + jitter controls.

Implementation detail: add metadata to advanced-settings fields, e.g. `applies_to={...}`, and filter at render time.

### 3) Replace free-form JSON entry in main flow with structured controls

For precompute/inference tabs:

- Keep current text JSON editors as a fallback in an "Expert JSON" expander.
- Add structured controls for the top 6–10 options users actually need.
- Display a compact "effective config" summary and diff from defaults before launch.

### 4) Make run settings immutable once a job starts

Strengthen reproducibility:

- On first run, persist `resolved_config.json` (fully merged defaults + profile + overrides + env-derived settings) in the job directory.
- On resume, load `resolved_config.json` as authoritative.
- If UI selections differ from persisted config, show a warning and require explicit "clone as new job" to change settings.

### 5) Codify resume semantics

- Continue batch-level status in manifest.
- Add `config_hash` and `code_version` to manifest.
- Resume only when `config_hash` matches; otherwise require a new job id (or clone).

### 6) Enforce strict offline/local model policy for HF-dependent components

- Require embed/reranker to be **existing local directories** before job launch.
- Set offline env guards for worker processes (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`).
- In model construction, detect unresolved/non-local identifiers and fail fast with actionable error text instead of attempting remote download.
- Add a startup health check button: "Validate model paths and offline readiness".

## Suggested implementation phases

### Phase 1 (low-risk, immediate UX win)

1. Add workflow-aware field visibility in `AIAdvancedConfigDialog`.
2. Hide `scjitter` for precompute/inference workflows.
3. Add explicit "Expert JSON overrides" collapsible area.
4. Persist and display a generated `resolved_config.json` per job.

### Phase 2 (reliability and resume hardening)

1. Add `config_hash`, `code_version`, and immutable-config checks.
2. Add "Resume with existing config" vs "Clone with edits" UX.
3. Improve warnings for mismatch between prompt job and inference job settings.

### Phase 3 (network-hardening and policy)

1. Enforce local-path-only for embedding/reranker everywhere.
2. Set global offline HF env vars in long-running job workers.
3. Add tests that assert no network fallback path is attempted.

## Acceptance criteria

- New users can run precompute/inference without writing JSON.
- Jitter controls are absent from precompute/inference UX but present in active-learning UX.
- Resuming a stopped job uses exactly the same effective config automatically.
- Attempting to run with non-local HF identifiers fails immediately with a clear message.
- Existing power users can still apply advanced overrides (behind an explicit advanced/expert affordance).

## Regression test additions

1. UI tests for workflow visibility matrix (which sections/fields appear per workflow).
2. Job tests verifying `resolved_config.json` is created and reused on resume.
3. Tests verifying config mismatch blocks resume and suggests clone.
4. Embedding init tests that reject non-local model identifiers when offline policy is active.
5. End-to-end prompt precompute + prompt inference interruption/resume test with config hash validation.

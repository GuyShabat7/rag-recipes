# Skills & Plugins Reference

Recommended Claude Code skills, plugins, and MCP servers for this RAG recipe recommendation project.

---

## Tier 1 — Core (Must Have)

### `pyright-lsp`
Python static type checker via LSP. Catches type mismatches across the data pipeline at edit time — critical because `score_recipe()`, `retrieve()`, and the FAISS search all pass `pd.DataFrame`, `np.ndarray`, and dict types across module boundaries. A silent type error here fails mid-request, not at import.

**Install:** `pip install pyright`
**Activate:** `/plugin install pyright-lsp@claude-plugins-official`

---

### `feature-dev`
Structured 7-phase feature development: Discovery → Codebase Exploration → Clarifying Questions → Architecture Design → Implementation → Review → Wrap-up. Use for building remaining features: `is_diabetic_friendly`/`is_gluten_free` flags, notebooks 01–05, ingredient scoring edge cases.

**Activate:** `/plugin install feature-dev@claude-plugins-official`
**Trigger:** `/feature-dev <description>`

---

### `pr-review-toolkit`
6 specialized parallel review agents. The three most critical for this project:
- **silent-failure-hunter** — The retriever silently relaxes dietary constraints when no candidates survive the filter, and the generator silently falls back. These need to be logged and surfaced.
- **pr-test-analyzer** — Only 4 test files exist; this identifies uncovered paths (hybrid scoring formula, FAISS index builder, LLM call layer).
- **type-design-analyzer** — Reviews Pydantic models in `src/api/models.py` and scoring return types.

**Activate:** `/plugin install pr-review-toolkit@claude-plugins-official`
**Trigger:** Ask for review and the right agent triggers automatically.

---

### `security-guidance`
API security review. Two surfaces to harden before demo:
1. `POST /recommend` — free-text `query` and unbounded `ingredients` list, no length/character validation
2. `GET /explain/{recipe_id}` — direct `.loc[]` lookup with user-controlled integer

**Activate:** `/plugin install security-guidance@claude-plugins-official`

---

### `claude-code-setup`
Scans the repo and recommends tailored automations: hooks, MCP servers, slash commands. Will likely recommend: auto-`pytest` on save, `ruff`/`black` format hook, and `context7` for live library docs.

**Activate:** `/plugin install claude-code-setup@claude-plugins-official`
**Trigger:** "recommend automations for this project"

---

## Tier 2 — High Value Additions

### `context7` (MCP Server — External)
Pulls live, version-pinned documentation into Claude's context during development. This project pins 8+ libraries: `sentence-transformers==3.0.1`, `faiss-cpu==1.8.0`, `openai==1.30.0`, `ranx==0.3.16`, `pydantic==2.7.0`. Prevents hallucinated API signatures.

**Install:** `/plugin install context7@claude-plugins-official`

---

### `serena` (MCP Server — External)
Semantic code analysis via LSP — understands actual code structure, not text. The call chain `routes.py → RAGPipeline → RecipeRetriever → RecipeEncoder → FAISS` spans 5 files. Serena enables "find all callers of `score_recipe()`" and "what does `retrieve()` return" with semantic precision, not grep.

**Install:** `/plugin install serena@claude-plugins-official`

---

### `hookify`
Create protective hooks from plain English — no JSON editing. Three concrete hooks this project needs:
- Block `rm -rf data/processed/` — losing the FAISS index costs 20+ minutes of re-embedding
- Warn before editing `configs/config.yaml` thresholds — they cascade to all 9 dietary flags and the evaluation ground truth
- Auto-run `pytest tests/` after edits to `src/scoring/` or `src/embeddings/`

**Activate:** `/plugin install hookify@claude-plugins-official`
**Trigger:** `/hookify <plain English description of rule>`

---

### `ralph-loop`
Iterative autonomous agent loop via stop-hook. Use for long-running pipeline tasks:
- Full data pipeline: `loader → cleaner → feature_engineering → encoder → indexer`
- Evaluation run: BM25 + embedding + full pipeline across 200 queries (~30 min)

Fire once, Claude iterates through blockers without intervention.

**Activate:** `/plugin install ralph-loop@claude-plugins-official`
**Trigger:** `/ralph-loop "<task>" --completion-promise "DONE"`

---

## Tier 3 — Portfolio Polish

### `playground` (skill)
Builds interactive HTML explorers. Use to create a live demo artifact: drag sliders for the hybrid scoring weights (`semantic: 0.6, nutritional: 0.3, popularity: 0.1`), see which recipes surface, understand the architecture visually. Differentiates the portfolio demo.

**Activate:** `/plugin install playground@claude-plugins-official`
**Trigger:** `/playground for tuning RAG hybrid scoring weights`

---

### `commit-commands`
Auto-generates commit messages from diffs matching the repo's style. Git history is part of the portfolio artifact — interviewers read `git log`.

**Activate:** `/plugin install commit-commands@claude-plugins-official`
**Trigger:** `/commit`

---

### `claude-md-management` (skill: `claude-md-improver`)
Audits CLAUDE.md against current codebase state — checks if referenced paths still exist, flags missing commands, detects new modules not yet documented. Run periodically as notebooks 01–05 get completed and new `data/splits/` files appear.

**Activate:** `/plugin install claude-md-management@claude-plugins-official`
**Trigger:** "audit my CLAUDE.md"

---

### `session-report`
Generates an HTML dashboard of token consumption, cache hit rates, and expensive prompts. Use after heavy evaluation runs (200 queries × 3 systems) to catch runaway token costs.

**Activate:** `/plugin install session-report@claude-plugins-official`
**Trigger:** "generate session report"

---

## Priority Stack Summary

```
MUST HAVE         pyright-lsp, feature-dev, pr-review-toolkit
HIGH VALUE        context7, serena, hookify, security-guidance
PORTFOLIO POLISH  claude-md-management, playground, commit-commands
OPERATIONAL       ralph-loop, session-report, claude-code-setup
```

# RAG Pipeline — Claude Code Guide

> This file is auto-loaded by Claude Code at session start. It combines two frameworks:
> - \*\*Karpathy Wiki\*\* — persistent, compounding knowledge base pattern (by Andrej Karpathy / toolboxmd)
> - \*\*Superpowers\*\* — structured software development methodology (by Jesse Vincent / obra)

\---

## Project Overview

A Retrieval-Augmented Generation (RAG) pipeline built from first principles. The pipeline connects a document store to an LLM, enabling grounded, citation-backed answers drawn from a curated knowledge base rather than model weights alone.

**Architecture at a glance:**

```
Sources (PDFs, URLs, text)
    │
    ▼
\[Ingestion] → chunk → embed → store
    │
    ▼
Vector Store  ←──── \[Retriever] ←── User Query
                          │
                          ▼
                    \[Generator / LLM]
                          │
                          ▼
                    Grounded Answer + Citations
```

**Tech stack:**

* Embeddings: sentence-transformers or OpenAI `text-embedding-3-small`
* Vector store: FAISS (local, zero-dependency) or Chroma (persistent)
* LLM: Claude via Anthropic SDK (with prompt caching)
* Orchestration: plain Python — no LangChain, no heavy frameworks

\---

## Development Commands

```bash
# Install deps
pip install -r requirements.txt

# Ingest documents
python -m rag.ingest --source docs/

# Query the pipeline
python -m rag.query "What is the attention mechanism?"

# Run test suite (TDD discipline — tests before implementation)
pytest tests/ -v

# Lint
ruff check . \&\& mypy rag/
```

\---

## Repository Layout

```
RAG\_pipeline/
├── CLAUDE.md              # This file
├── README.md
├── requirements.txt
├── rag/
│   ├── \_\_init\_\_.py
│   ├── ingest.py          # Document chunking + embedding
│   ├── retriever.py       # Vector similarity search
│   ├── generator.py       # LLM call with retrieved context
│   ├── pipeline.py        # End-to-end orchestration
│   └── config.py          # Settings (chunk size, top-k, model IDs)
├── tests/
│   ├── test\_ingest.py
│   ├── test\_retriever.py
│   └── test\_generator.py
├── docs/                  # Source documents to ingest
└── wiki/                  # Karpathy wiki (auto-managed, see below)
    ├── schema.md
    ├── index.md
    └── .wiki-pending/     # Pre-ingest captures (never edit manually)
```

\---

## Karpathy Coding Principles

Behavioral guidelines to reduce common LLM coding mistakes, derived from [Andrej Karpathy's observations](https://x.com/karpathy/status/2015883857489522876) on LLM coding pitfalls.

> \*\*Tradeoff:\*\* These guidelines bias toward caution over speed. For trivial tasks, use judgment.

### 1 · Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:

* State your assumptions explicitly. If uncertain, ask.
* If multiple interpretations exist, present them — don't pick silently.
* If a simpler approach exists, say so. Push back when warranted.
* If something is unclear, stop. Name what's confusing. Ask.

### 2 · Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

* No features beyond what was asked.
* No abstractions for single-use code.
* No "flexibility" or "configurability" that wasn't requested.
* No error handling for impossible scenarios.
* If you write 200 lines and it could be 50, rewrite it.

Ask yourself: *"Would a senior engineer say this is overcomplicated?"* If yes, simplify.

### 3 · Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:

* Don't "improve" adjacent code, comments, or formatting.
* Don't refactor things that aren't broken.
* Match existing style, even if you'd do it differently.
* If you notice unrelated dead code, mention it — don't delete it.

When your changes create orphans:

* Remove imports/variables/functions that **your** changes made unused.
* Don't remove pre-existing dead code unless asked.

**The test:** Every changed line should trace directly to the user's request.

### 4 · Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:

* `"Add validation"` → `"Write tests for invalid inputs, then make them pass"`
* `"Fix the bug"` → `"Write a test that reproduces it, then make it pass"`
* `"Refactor X"` → `"Ensure tests pass before and after"`

For multi-step tasks, state a brief plan:

```
1. \[Step] → verify: \[check]
2. \[Step] → verify: \[check]
3. \[Step] → verify: \[check]
```

Strong success criteria let you loop independently. Weak criteria (*"make it work"*) require constant clarification.

\---

## Superpowers Development Methodology

This project follows the **Superpowers** structured methodology. Before writing any code, follow this sequence:

### 1 · Brainstorming (before coding)

When requirements are ambiguous, run a Socratic brainstorming session:

* Ask clarifying questions until the design is concrete
* Surface assumptions, edge cases, and constraints
* Present the design in digestible chunks; get approval before moving on
* Do NOT start coding until the design is agreed

### 2 · Writing a Plan

Break the approved design into **2–5 minute tasks** with exact specifications:

* Each task has a clear input, output, and acceptance criterion
* Tasks are ordered by dependency
* TDD tasks are listed first (write failing tests) before implementation tasks
* Architectural decisions are noted with rationale

### 3 · Test-Driven Development (RED → GREEN → REFACTOR)

```
RED:     Write a failing test that captures the requirement
GREEN:   Write the minimum code to make the test pass
REFACTOR: Clean up — no new functionality, tests still pass
```

**Iron rules:**

* Tests must fail before implementation exists
* Never skip the refactor phase
* Never add functionality not required by a failing test

### 4 · Subagent-Driven Development

For tasks that can run in parallel, dispatch fresh subagents per task:

* Each subagent gets a self-contained prompt with full context
* Main agent reviews each subagent's output before integration
* Two-stage review: correctness first, then style/quality

### 5 · Systematic Debugging

When something is broken, follow the four-phase method:

1. **Reproduce**: get a reliable failing case (test or REPL session)
2. **Locate**: bisect to the exact line/call that misbehaves
3. **Root cause**: understand *why* it fails (don't guess, read the stack)
4. **Fix \& verify**: fix the root cause, re-run the reproduction case, confirm green

**Never** paper over a symptom. Fix the root cause.

### 6 · Code Review

Before merging any branch:

* Review against the plan specification, not against vague intuition
* Check: does the implementation match what was planned?
* Check: are all tests passing and covering the new code paths?
* Check: are new dependencies justified?

### 7 · Verification Before Completion

Before marking any task done:

* Run the full test suite
* Run the linter and type checker
* Do a manual smoke test of the end-to-end pipeline
* Confirm the feature works as described in the plan

\---

## Karpathy Wiki — Persistent Knowledge Base

This project uses the **karpathy-wiki** plugin. Knowledge discovered during development is captured to `wiki/` so future sessions start informed rather than blank.

### Iron Laws

```
NO WIKI WRITE IN THE FOREGROUND
NO PAGE EDIT WITHOUT READING THE PAGE FIRST
NO SKIPPING A CAPTURE BECAUSE "IT DOESN'T LOOK WIKI-SHAPED"
NO ANSWERING ANY USER QUESTION WITHOUT ORIENTING FIRST
```

### When to Capture (triggers fire automatically)

* A research subagent completes with durable findings
* New factual information appears (web search, doc fetch, external fact)
* A confusion is resolved — something neither agent nor user knew before
* A gotcha, quirk, or non-obvious behavior is discovered
* A pattern is validated (approaches compared, one chosen, with reasons)
* An architectural decision is made with explicit rationale
* User pastes a URL or document for study
* User says "add to wiki" / "remember this" / "wiki it"

### When NOT to Capture

* Routine file edits and syntax lookups
* Time-sensitive data that must be fetched fresh
* One-off debugging with trivial root causes

### Announce Line

When writing a capture, running ingest, or answering from the wiki, prefix with:

> \*\*Using the karpathy-wiki skill to \[capture this / ingest pending captures / answer from wiki].\*\*

Then do all wiki work silently. No narration of mechanics.

### Orient Before Answering

For any **question** (not a code-edit or file-op request), orient on the wiki first:

1. Read `wiki/schema.md` and the relevant `wiki/<category>/\_index.md`
2. Count candidate pages
3. Read inline (≤3 pages), spawn a read subagent (4–10 pages), or web-search if zero hits
4. Cite the page(s) used in the answer

Resist-table — do NOT skip orientation for these rationalizations:

|Rationalization|Reality|
|-|-|
|"This is general knowledge"|Pre-classifying questions as wiki-irrelevant is the forbidden rationalization. Orient.|
|"The question is trivial"|Orientation is once per session; subsequent checks are near-zero cost.|
|"Pure syntax question"|You cannot know without checking. If `\_index.md` has zero matches, do a web search + capture the gap.|

\---

## RAG-Specific Knowledge Base (Seed)

The following entries are pre-seeded to orient every new session:

### Chunking Strategy

**Decision (updated):** Use recursive character splitting with `chunk\_size=1000`, `chunk\_overlap=150`.

* Rationale: 1000 characters balances context density with retrieval precision for mixed document types. 150-character overlap prevents context loss at chunk boundaries.
* Revisit if: retrieval recall drops below 0.7 on the eval set, or if source documents are highly structured (tables, code — consider semantic/section-based chunking instead).

### Embedding Model

**Decision:** `text-embedding-3-small` (OpenAI) for cloud; `all-MiniLM-L6-v2` (sentence-transformers) for local/offline.

* Rationale: `text-embedding-3-small` has excellent quality/cost ratio. `all-MiniLM-L6-v2` is the best sub-100MB model for offline use.
* Switch to `text-embedding-3-large` only if eval NDCG@10 < 0.6 on the target corpus.

### Prompt Caching

Claude's prompt caching is enabled for the system prompt and retrieved context blocks. Cache breakpoints are placed at:

1. End of system prompt (static)
2. End of retrieved context (changes per query, but within a conversation stays stable)

This reduces cost by \~60% on multi-turn sessions. See `rag/generator.py` for the `cache\_control` block placement.

### Top-K Retrieval

**Default:** `top\_k=5`. Return the 5 most similar chunks.

* Too few (k<3): misses relevant context, hurts faithfulness
* Too many (k>8): floods the prompt, hurts coherence and increases latency
* Tune per corpus: run `python -m rag.eval --metric precision\_at\_k` and adjust

### Eval Metrics

|Metric|Tool|Target|
|-|-|-|
|Retrieval Precision@5|`rag.eval`|≥ 0.75|
|Answer Faithfulness|LLM-as-judge|≥ 0.80|
|Answer Relevance|LLM-as-judge|≥ 0.80|
|Latency (p95)|`rag.eval`|≤ 3 s|

\---

## Git Workflow

* Default branch: `main`
* Feature branches: `feature/<short-description>`
* Commit messages: imperative mood, ≤72 chars, reference the plan task if applicable
* Never force-push to `main`
* One logical change per commit; squash noise before merging

\---

## Skill Reference

The following skills are available via `/skill-name` in Claude Code. Use them:

|Skill|When to use|
|-|-|
|`/brainstorming`|Requirements are ambiguous or underspecified|
|`/writing-plans`|Ready to break a design into implementation tasks|
|`/test-driven-development`|Implementing any new feature|
|`/systematic-debugging`|Something is broken and root cause is unclear|
|`/subagent-driven-development`|Tasks can run in parallel|
|`/requesting-code-review`|Branch is ready for review|
|`/verification-before-completion`|About to mark a task done|
|`/finishing-a-development-branch`|Branch is reviewed and needs to be merged|

\---

## Sources

* [Karpathy LLM Wiki pattern](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f)
* [karpathy-wiki Claude Code plugin](https://github.com/toolboxmd/karpathy-wiki)
* [Karpathy coding principles skill](https://github.com/multica-ai/andrej-karpathy-skills)
* [Superpowers development methodology](https://github.com/obra/superpowers)
* [Karpathy-inspired CLAUDE.md (TechTimes)](https://www.techtimes.com/articles/316798/20260518/karpathy-inspired-claudemd-passes-220000-combined-github-stars-four-rules-that-stop-ai-breaking.htm)


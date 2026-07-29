# Metroid

This repository contains `metroid`, a Python library to simulate streaks and trails in astronomical images that are caused by orbital objects such as satellites and space debris. This document contains critical information about working with this codebase. Follow these guidelines precisely.

## Convention Hierarchy

When sources conflict, follow this precedence (higher overrides lower):

| Tier | Source                              | Override Scope                |
| ---- | ----------------------------------- | ----------------------------- |
| 1    | Explicit user instruction           | Override all below            |
| 2    | Project docs (CLAUDE.md, README.md) | Override conventions/defaults |
| 3    | Universal best practices            | Confirm if uncertain          |

**Conflict resolution**: Lower tier numbers win. Subdirectory docs override root docs for that subtree.

## Knowledge Strategy

**CLAUDE.md** = navigation index (WHAT is here, WHEN to read)
**README.md** = invisible knowledge (WHY it's structured this way)

## Core Workflow

All tasks should be performed within the scope of the Git Workflow. The generalized pattern is

1. Parse task; retrieve additional context only for determining the scope of the task.
2. Stage within git workflow
2. Plan
3. Implement
4. Evaluate
5. Document
6. Finalize within git workflow

### Git Workflow

- Always use feature branches; do not commit directly to `main`
  - Name branches descriptively: `fix/auth-timeout`, `documentation/code_examples`, `feature/api-pagination`
  - Keep one logical change per branch to simplify review and rollback
- Create pull requests for all changes
  - Open a draft PR early for visibility; convert to ready when complete
  - Ensure tests pass locally before marking ready for review
- Link issues
  - Before starting, reference an existing issue or create one
  - Use commit/PR messages like `Fixes #123` for auto-linking and closure
- Commit practices
  - Make atomic commits (one logical change per commit)
  - Prefer conventional commit style: `type(scope): short description`
    - Examples: `feature(eval): group OBS logs per test`, `fix(cli): handle missing API key`
  - Squash only when merging to `main`; keep granular history on the feature branch
- Practical workflow
  1. Create or reference an issue
  2. `git checkout -b feature/issue-123-description`
  3. Commit in small, logical increments
  4. Open a draft PR early
  5. Convert to ready PR when functionally complete and tests pass
  6. Never merge automatically, always prompt first

## Tools

## Python Tools

- use `black` to fix PEP 8 (this is the authoritative source for formatting)
- use `mypy` for static type checking
- use `pytest` for unit tests

## Repository Layout:

- Production code in `src/metroid`
- Tests in `tests` mirror the package structure
- Subpackages in subdirectories of `src/metroid`
- Public utilities in `src/metroid/utils`

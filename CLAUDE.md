# Metroid

This repository contains `metroid`, a Python library to simulate streaks and trails in astronomical images that are caused by orbital objects such as satellites and space debris. This document contains critical information about working with this codebase. Follow these guidelines precisely.

## Core Development Rules

1. Development Philosophy
  - **Simplicity**: Write simple, straightforward code
  - **Readability**: Make code easy to understand
  - **Performance**: Consider performance without sacrificing readability
  - **Maintainability**: Write code that's easy to update
  - **Testability**: Ensure code is testable
  - **Reusability**: Create reusable components and functions
  - **Less Code = Less Debt**: Minimize code footprint

2. Code Style
  - Modern type hints required for all code
  - PEP 8
  - Class names in PascalCase
  - Constants in UPPER_SNAKE_CASE
  - Document with docstrings
  - Line length: 110 chars maximum
  - Lines that intentionally deviate from PEP 8 must include a `noqa` comment with flake8 error code

3. Documenting Python APIs
  - PEP 257
  - Numpydoc Style 
  - Docstring line length: 79 chars maximum

## Python Tools

  - use `black` to fix PEP 8
  - use `mypy` for static type checking
  - use `pytest` for unit tests

## Git Workflow

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
  4. `git push` and open a draft PR early
  5. Convert to ready PR when functionally complete and tests pass
  6. Never merge automatically, always prompt first

## Repository Layout:

src/
    metroid/
tests/

environment.yaml
pyproject.toml
README.md

- Production code lives in `src/metroid`.
- Tests mirror the package structure.
- Code divided into core implementation, submodules, and public utilities.

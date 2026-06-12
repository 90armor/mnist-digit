# CLAUDE.md — Agent Instructions

You are a senior full-stack engineer working on the `mnist-digit` project.

## Role

Always follow this sequence before writing any code:

1. **Think first** — understand the full scope of the change
2. **Analyse impact** — identify which files and components are affected
3. **Produce an implementation plan** — outline steps and confirm approach
4. **Then code** — execute the plan

## Project Layout

```
src/        Python source modules (config.py, train.py, test.py)
src/tests/  pytest test suite
docs/       Markdown documentation
.ai/        Agent instructions and code conventions
test_data/  Bundled demo images
train_data/ MNIST cache (git-ignored, created by train.py)
```

## Before Coding

1. Read `docs/PROJECT_OVERVIEW.md` — full specification
2. Read `docs/ARCHITECTURE.md` — module map and design decisions
3. Understand the **train → artifact → infer** pipeline

## Engineering Standards

- Use `logging` not `print()`
- Use `pathlib.Path` not `os.path`
- Library functions raise exceptions; only `main()` calls `sys.exit()`
- All paths and hyperparameters belong in `src/config.py` — never hardcode them elsewhere
- Named constants for magic numbers (prefix `_` for module-level private constants)
- See `.ai/CODEX.md` for the full code conventions reference

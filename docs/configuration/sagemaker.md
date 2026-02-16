# SageMaker usage (CLI + notebooks)

Swarmee River runs as a standard Python CLI (`swarmee`) and can also be used inside notebooks via the `%%swarmee` cell magic.

This guide focuses on **SageMaker interactive environments** (terminal + Jupyter) and other “minimal” images (no Node/Bun).

## Install

```bash
pip install swarmee-river
```

For notebook magic:

```bash
pip install "swarmee-river[jupyter]"
```

## Recommended environment variables

Redirect runtime state writes (artifacts, logs, sessions, project map) to a writable location:

```bash
export SWARMEE_STATE_DIR="/tmp/swarmee"
```

Pick a provider (example: OpenAI):

```bash
export SWARMEE_MODEL_PROVIDER="openai"
export OPENAI_API_KEY="..."
```

Non-interactive automation defaults (optional):

```bash
export SWARMEE_AUTO_APPROVE="true"
```

## CLI (terminal)

Interactive:

```bash
swarmee
```

One-shot:

```bash
swarmee "Summarize this repo and list next steps."
```

Plan/approve in one-shot mode:

```bash
swarmee --yes "Implement X and run tests."
```

## Notebooks

Load the extension:

```python
%load_ext swarmee_river.jupyter
```

Generate a plan (no execution):

```python
%%swarmee --plan
Add a CLI flag to control SWARMEE_STATE_DIR.
```

Plan + execute (auto-approve tools for this one invocation):

```python
%%swarmee --yes
Refactor this notebook code for clarity.
```

Notes:
- The notebook extension runs **non-interactively by default** (tool consent fails closed) unless you pass `--yes`.
- Use `%%swarmee --no-context` (or `SWARMEE_NOTEBOOK_NO_CONTEXT=true`) for quick one-offs without notebook context injection.

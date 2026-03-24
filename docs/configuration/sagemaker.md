# SageMaker usage (CLI + notebooks)

Swarmee River runs as a standard Python CLI (`swarmee`) and can also be used inside notebooks via the `%%swarmee` cell magic.

This guide focuses on **SageMaker interactive environments** (terminal + Jupyter) and other “minimal” images (no Node/Bun).

## Install

```bash
pip install swarmee-river
```

Optional (recommended only if you need upstream Strands Tools integrations like Slack, media generation, etc.):

```bash
pip install "swarmee-river[strands_tools]"
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

Diagnostics are enabled by default. Optional tuning:

```bash
export SWARMEE_DIAG_LEVEL="baseline"       # or verbose
export SWARMEE_DIAG_REDACT="true"
export SWARMEE_DIAG_RETENTION_DAYS="7"
export SWARMEE_DIAG_MAX_BYTES="52428800"
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
- For Bedrock in SageMaker, role-based credentials should resolve through the default AWS credential chain without setting `AWS_PROFILE`.

## Clipboard and tmux

Many SageMaker terminal environments run inside `tmux`, which means copy requests can succeed only inside the tmux buffer unless OSC 52 passthrough is enabled end to end.

Recommended tmux settings:

```tmux
set -g set-clipboard on
set -g allow-passthrough on
```

Also make sure the terminal connected to SageMaker allows OSC 52 clipboard access. If it does not, Swarmee will now report terminal-copy requests as best effort instead of claiming the clipboard was definitely updated.

Practical notes:
- Native clipboard tools such as `pbcopy`, `wl-copy`, or `xclip` are uncommon in SageMaker shells, so terminal clipboard integration is the normal path.
- If terminal clipboard integration is unavailable, Swarmee falls back to saving the copied text as an artifact so the content is still recoverable.

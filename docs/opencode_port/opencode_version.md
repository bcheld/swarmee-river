# OpenCode version pinned for parity

This port targets a pinned OpenCode version so “core parity” has a clear source of truth.

## Pinned upstream

- Repo: `anomalyco/opencode`
- Release: `v1.1.65` (Feb 13, 2026)
- Commit: `34ebe814ddd130a787455dda089facb23538ca20`
- Package manager/runtime: Bun (`packageManager: bun@1.3.9`)

## Repro (upstream)

```bash
git clone https://github.com/anomalyco/opencode.git
cd opencode
git checkout 34ebe814ddd130a787455dda089facb23538ca20

bun install

# Core CLI dev loop
bun --cwd packages/opencode dev

# Typecheck (monorepo)
bun turbo typecheck
```

## Notes

- OpenCode is a monorepo; the CLI package lives under `packages/opencode`.
- If we bump the pinned version, update this file first, then re-derive the capability matrix.


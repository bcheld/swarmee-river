# SOPs

Put `*.sop.md` files in this directory to make them available at runtime.

- File naming: `my-workflow.sop.md` becomes SOP name `my-workflow`
- SOPs can be listed/loaded via the `sop` tool
- You can also point at additional SOP directories with `SWARMEE_SOP_PATHS` (use your OS path separator)

Local SOPs override built-in SOPs with the same name.

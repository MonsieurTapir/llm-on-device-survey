"""The `survey` subcommand implementations, one module each.

`cli.py` owns argument parsing and dispatch; each module here owns one
subcommand's logic. `fetch` lives in `survey.fetch` (it's the only command with no
backend/spawn machinery), so the CLI dispatches straight to it.
"""

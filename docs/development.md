# Development

This project uses `uv` for dependency management and task execution.

## Runtime

```bash
uv sync --no-dev
uv run --no-dev python main.py
```

## Development

```bash
uv sync
uv run pytest
uv run ruff check .
uv run ruff format .
uv run ty check
```

## Notes

- The `dev` dependency group contains the test and lint tooling.
- The project is intentionally marked as a virtual project, so uv does not build or install it as a package.
- The application is run from source with `uv run --no-dev python main.py`.

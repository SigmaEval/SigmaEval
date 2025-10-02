# Project Structure

## Package Organization
```
sigmaeval/
├── __init__.py           # Package initialization and public API
├── core/                 # Core evaluation functionality
├── evaluators/           # Evaluator implementations
├── utils/                # Utility functions
└── exceptions.py         # Custom exceptions
```

## File Naming
- Use lowercase with underscores for module names (e.g., `evaluation_engine.py`)
- Keep module names short and descriptive
- One main class per file is preferred

## Public API
- Define `__all__` in `__init__.py` to control public exports
- Keep the public API minimal and stable
- Internal utilities should start with underscore (`_internal_helper`)

## Version Management
- Update version in `sigmaeval/__init__.py`
- Follow semantic versioning (MAJOR.MINOR.PATCH)
- Update CHANGELOG.md for each release

## Dependencies
- Keep dependencies minimal
- Pin major versions in `pyproject.toml`
- Use optional dependencies for non-core features


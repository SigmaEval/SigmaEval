# Testing Standards

## Test Organization
- All tests live in the `tests/` directory
- Test files must be named `test_*.py`
- Test functions must start with `test_`
- Mirror the package structure in tests (e.g., `sigmaeval/foo.py` → `tests/test_foo.py`)

## Test Requirements
- Write tests for all public APIs
- Aim for high code coverage (>80%)
- Use pytest fixtures for shared setup
- Use descriptive test names that explain what is being tested

## Example Test Structure
```python
"""Tests for the evaluation module."""

import pytest
from sigmaeval import evaluate


def test_evaluate_returns_float():
    """Test that evaluate returns a float value."""
    result = evaluate({"score": 0.5}, "accuracy")
    assert isinstance(result, float)


def test_evaluate_with_invalid_metric():
    """Test that evaluate raises error for invalid metric."""
    with pytest.raises(ValueError):
        evaluate({"score": 0.5}, "invalid_metric")
```

## Running Tests
```bash
pytest                    # Run all tests
pytest tests/test_foo.py  # Run specific test file
pytest -v                 # Verbose output
pytest --cov=sigmaeval    # With coverage report
```


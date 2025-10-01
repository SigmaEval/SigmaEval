# Python Style Guide

## Code Formatting
- Use Black for code formatting (line length: 100)
- Use Ruff for linting
- Follow PEP 8 conventions
- Use type hints for function parameters and return values

## Imports
- Standard library imports first
- Third-party imports second
- Local imports last
- Use absolute imports from `sigmaeval`

## Documentation
- All public functions, classes, and modules must have docstrings
- Use Google-style docstrings format
- Include examples in docstrings where helpful

## Example
```python
"""Module for evaluation tasks."""

from typing import Optional


def evaluate(data: dict, metric: str) -> float:
    """Evaluate data using the specified metric.
    
    Args:
        data: The data to evaluate.
        metric: The metric name to use.
        
    Returns:
        The evaluation score.
        
    Example:
        >>> evaluate({"score": 0.5}, "accuracy")
        0.5
    """
    pass
```


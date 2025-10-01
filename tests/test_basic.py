"""Basic tests for SigmaEval."""

import sigmaeval


def test_version():
    """Test that version is defined."""
    assert hasattr(sigmaeval, "__version__")
    assert isinstance(sigmaeval.__version__, str)


def test_import():
    """Test that the package can be imported."""
    assert sigmaeval is not None


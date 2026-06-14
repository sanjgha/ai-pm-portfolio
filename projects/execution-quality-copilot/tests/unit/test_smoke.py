"""Smoke test confirming the test suite and import path work."""


def test_package_imports():
    """The src package should import cleanly."""
    import src  # noqa: F401


def test_truth():
    """Sanity check so a fresh checkout has at least one passing test."""
    assert True

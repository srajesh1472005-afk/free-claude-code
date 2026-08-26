"""Tests for Streamlit application structure."""

from free_claude_code.streamlit import app


def test_streamlit_module_import() -> None:
    assert app is not None

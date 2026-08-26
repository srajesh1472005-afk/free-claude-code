"""Root entrypoint wrapper for Streamlit Community Cloud and local execution."""

import sys
from pathlib import Path

# Ensure src directory is in Python path for package imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from free_claude_code.streamlit.app import *  # noqa: F401, F403

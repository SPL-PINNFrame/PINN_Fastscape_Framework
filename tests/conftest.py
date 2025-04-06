import sys
import os
import pytest

# Add project root to the Python path to allow absolute imports from src
# Assumes conftest.py is in the tests/ directory, one level below the project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    print(f"\nInserting project root into sys.path: {project_root}\n")
    sys.path.insert(0, project_root)

# You can define shared fixtures here later, e.g., for dummy data or models
@pytest.fixture(scope="session")
def project_root_dir():
    """Provides the absolute path to the project root directory."""
    return project_root
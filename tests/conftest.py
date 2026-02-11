"""
Shared test configuration and fixtures for AegisLang test suite.
"""

import os

# Disable API authentication for all tests.
# The server checks AEGISLANG_DISABLE_AUTH at import time via get_valid_api_keys(),
# so this must be set before any test imports the app.
os.environ["AEGISLANG_DISABLE_AUTH"] = "true"

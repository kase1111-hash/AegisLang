"""
Shared test configuration and fixtures for AegisLang test suite.
"""

import os

# Disable API authentication for all tests.
# The server checks AEGISLANG_DISABLE_AUTH at import time via get_valid_api_keys(),
# so this must be set before any test imports the app.
os.environ["AEGISLANG_DISABLE_AUTH"] = "true"

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    """Reset the global rate limiter between tests to prevent cross-test 429s."""
    from aegislang.api.server import _rate_limiter
    _rate_limiter._minute_counts.clear()
    _rate_limiter._hour_counts.clear()

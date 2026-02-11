"""Performance test configuration.

These tests require optional dependencies (locust) and are designed
to run as standalone scripts, not via pytest collection.
"""

collect_ignore = ["locustfile.py"]

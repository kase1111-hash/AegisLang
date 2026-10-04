"""
Test for compliance rule: FATF_RECOMMENDATION_10_8D5616_FATF_RECOMMENDATION_10_S009_C000_CL001

Source: Countries may permit financial institutions to rely on third parties t...
Generated: 2026-10-04T20:17:07.948802+00:00
Confidence: 0.5
"""

import pytest
from datetime import datetime, timedelta


class TestFATF_RECOMMENDATION_10_8D5616_FATF_RECOMMENDATION_10_S009_C000_CL001:
    """Test cases for permission rule: FATF_RECOMMENDATION_10_8D5616_FATF_RECOMMENDATION_10_S009_C000_CL001"""

    @pytest.fixture
    def compliant_countries(self):
        """Create a compliant Countries fixture."""
        return {
            "id": "test_countries_001",


            "cdd_measures": "valid_value",

            "created_at": datetime.utcnow(),
        }

    @pytest.fixture
    def non_compliant_countries(self):
        """Create a non-compliant Countries fixture."""
        return {
            "id": "test_countries_002",


            "cdd_measures": None,

            "created_at": datetime.utcnow(),
        }








# Metadata for test discovery
CLAUSE_ID = "FATF_RECOMMENDATION_10_8D5616_FATF_RECOMMENDATION_10_S009_C000_CL001"
CLAUSE_TYPE = "permission"
SOURCE_TEXT = """Countries may permit financial institutions to rely on third parties to perform CDD measures."""
CONFIDENCE = 0.5
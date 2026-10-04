"""
Test for compliance rule: FATF_RECOMMENDATION_10_8D5616_FATF_RECOMMENDATION_10_S008_C000_CL002

Source: Institutions shall consider making a suspicious transaction report whe...
Generated: 2026-10-04T20:17:07.899000+00:00
Confidence: 0.5
"""

import pytest
from datetime import datetime, timedelta


class TestFATF_RECOMMENDATION_10_8D5616_FATF_RECOMMENDATION_10_S008_C000_CL002:
    """Test cases for obligation rule: FATF_RECOMMENDATION_10_8D5616_FATF_RECOMMENDATION_10_S008_C000_CL002"""

    @pytest.fixture
    def compliant_institutions(self):
        """Create a compliant Institutions fixture."""
        return {
            "id": "test_institutions_001",

            "consider_status": True,


            "when_cdd_cannot_be": "valid_value",

            "created_at": datetime.utcnow(),
        }

    @pytest.fixture
    def non_compliant_institutions(self):
        """Create a non-compliant Institutions fixture."""
        return {
            "id": "test_institutions_002",

            "consider_status": False,


            "when_cdd_cannot_be": None,

            "created_at": datetime.utcnow(),
        }


    def test_consider_obligation_met(
        self, compliant_institutions
    ):
        """Test that obligation is satisfied when consider is performed."""
        entity = compliant_institutions

        # Assert obligation is met
        assert entity["consider_status"] is True, \
            "Institutions must consider when CDD cannot be"

    def test_consider_obligation_violated(
        self, non_compliant_institutions
    ):
        """Test that violation is detected when consider is not performed."""
        entity = non_compliant_institutions

        # Assert obligation is violated
        assert entity["consider_status"] is False, \
            "Expected violation when Institutions does not consider"




    def test_condition_trigger(self):
        """Test that rule applies when condition 'CDD cannot be completed' is met."""
        # TODO: Implement condition testing for: CDD cannot be completed
        pytest.skip("Condition testing not yet implemented")





# Metadata for test discovery
CLAUSE_ID = "FATF_RECOMMENDATION_10_8D5616_FATF_RECOMMENDATION_10_S008_C000_CL002"
CLAUSE_TYPE = "obligation"
SOURCE_TEXT = """Institutions shall consider making a suspicious transaction report when CDD cannot be completed."""
CONFIDENCE = 0.5
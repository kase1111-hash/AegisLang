"""
Test for compliance rule: FFIEC_CIP_MANUAL_CA1E58_FFIEC_CIP_MANUAL_S008_C000_CL003

Source: Institutions may satisfy this requirement through posting a notice in ...
Generated: 2026-10-04T20:17:08.847602+00:00
Confidence: 0.5
"""

import pytest
from datetime import datetime, timedelta


class TestFFIEC_CIP_MANUAL_CA1E58_FFIEC_CIP_MANUAL_S008_C000_CL003:
    """Test cases for permission rule: FFIEC_CIP_MANUAL_CA1E58_FFIEC_CIP_MANUAL_S008_C000_CL003"""

    @pytest.fixture
    def compliant_institutions(self):
        """Create a compliant Institutions fixture."""
        return {
            "id": "test_institutions_001",


            "created_at": datetime.utcnow(),
        }

    @pytest.fixture
    def non_compliant_institutions(self):
        """Create a non-compliant Institutions fixture."""
        return {
            "id": "test_institutions_002",


            "created_at": datetime.utcnow(),
        }








# Metadata for test discovery
CLAUSE_ID = "FFIEC_CIP_MANUAL_CA1E58_FFIEC_CIP_MANUAL_S008_C000_CL003"
CLAUSE_TYPE = "permission"
SOURCE_TEXT = """Institutions may satisfy this requirement through posting a notice in the lobby or on the website."""
CONFIDENCE = 0.5
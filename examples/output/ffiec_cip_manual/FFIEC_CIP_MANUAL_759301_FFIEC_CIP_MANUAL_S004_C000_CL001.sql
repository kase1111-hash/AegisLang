-- Source: Each bank must implement a written Customer Identification Program app...
-- Clause ID: FFIEC_CIP_MANUAL_759301_FFIEC_CIP_MANUAL_S004_C000_CL001
-- Generated: 2026-02-11T18:36:04.291873+00:00
-- Confidence: 0.5


-- Obligation: Each bank must implement
ALTER TABLE compliance_table
ADD CONSTRAINT chk_ffiec_cip_manual_759301_ffiec_cip_manual_s004_c000_cl001
CHECK (
    implement_status = TRUE
);

-- Trigger for enforcement
CREATE OR REPLACE FUNCTION enforce_ffiec_cip_manual_759301_ffiec_cip_manual_s004_c000_cl001()
RETURNS TRIGGER AS $$
BEGIN
    IF NOT (implement_status = TRUE) THEN
        RAISE EXCEPTION 'Compliance violation: FFIEC_CIP_MANUAL_759301_FFIEC_CIP_MANUAL_S004_C000_CL001 - Each bank must implement';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_ffiec_cip_manual_759301_ffiec_cip_manual_s004_c000_cl001
BEFORE INSERT OR UPDATE ON compliance_table
FOR EACH ROW
EXECUTE FUNCTION enforce_ffiec_cip_manual_759301_ffiec_cip_manual_s004_c000_cl001();



COMMENT ON CONSTRAINT chk_ffiec_cip_manual_759301_ffiec_cip_manual_s004_c000_cl001 ON compliance_table
IS 'AegisLang: Each bank must implement a written Customer Identification Program appropriate for its size and type of business.';
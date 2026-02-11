-- Source: Banks must provide adequate notice to customers that the institution i...
-- Clause ID: FFIEC_CIP_MANUAL_759301_FFIEC_CIP_MANUAL_S008_C000_CL001
-- Generated: 2026-02-11T18:36:04.567664+00:00
-- Confidence: 0.5


-- Obligation: Banks must provide
ALTER TABLE compliance_table
ADD CONSTRAINT chk_ffiec_cip_manual_759301_ffiec_cip_manual_s008_c000_cl001
CHECK (
    provide_status = TRUE
);

-- Trigger for enforcement
CREATE OR REPLACE FUNCTION enforce_ffiec_cip_manual_759301_ffiec_cip_manual_s008_c000_cl001()
RETURNS TRIGGER AS $$
BEGIN
    IF NOT (provide_status = TRUE) THEN
        RAISE EXCEPTION 'Compliance violation: FFIEC_CIP_MANUAL_759301_FFIEC_CIP_MANUAL_S008_C000_CL001 - Banks must provide';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_ffiec_cip_manual_759301_ffiec_cip_manual_s008_c000_cl001
BEFORE INSERT OR UPDATE ON compliance_table
FOR EACH ROW
EXECUTE FUNCTION enforce_ffiec_cip_manual_759301_ffiec_cip_manual_s008_c000_cl001();



COMMENT ON CONSTRAINT chk_ffiec_cip_manual_759301_ffiec_cip_manual_s008_c000_cl001 ON compliance_table
IS 'AegisLang: Banks must provide adequate notice to customers that the institution is requesting information to verify their identity.';
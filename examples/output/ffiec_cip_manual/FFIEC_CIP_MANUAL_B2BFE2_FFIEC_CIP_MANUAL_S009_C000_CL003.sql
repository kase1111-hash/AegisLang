-- Source: Banks must enter into a written agreement with the other institution s...
-- Clause ID: FFIEC_CIP_MANUAL_B2BFE2_FFIEC_CIP_MANUAL_S009_C000_CL003
-- Generated: 2026-02-11T18:21:29.476092+00:00
-- Confidence: 0.5


-- Obligation: Banks must enter
ALTER TABLE compliance_table
ADD CONSTRAINT chk_ffiec_cip_manual_b2bfe2_ffiec_cip_manual_s009_c000_cl003
CHECK (
    enter_status = TRUE
);

-- Trigger for enforcement
CREATE OR REPLACE FUNCTION enforce_ffiec_cip_manual_b2bfe2_ffiec_cip_manual_s009_c000_cl003()
RETURNS TRIGGER AS $$
BEGIN
    IF NOT (enter_status = TRUE) THEN
        RAISE EXCEPTION 'Compliance violation: FFIEC_CIP_MANUAL_B2BFE2_FFIEC_CIP_MANUAL_S009_C000_CL003 - Banks must enter';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_ffiec_cip_manual_b2bfe2_ffiec_cip_manual_s009_c000_cl003
BEFORE INSERT OR UPDATE ON compliance_table
FOR EACH ROW
EXECUTE FUNCTION enforce_ffiec_cip_manual_b2bfe2_ffiec_cip_manual_s009_c000_cl003();



COMMENT ON CONSTRAINT chk_ffiec_cip_manual_b2bfe2_ffiec_cip_manual_s009_c000_cl003 ON compliance_table
IS 'AegisLang: Banks must enter into a written agreement with the other institution specifying the delegated CIP procedures.';
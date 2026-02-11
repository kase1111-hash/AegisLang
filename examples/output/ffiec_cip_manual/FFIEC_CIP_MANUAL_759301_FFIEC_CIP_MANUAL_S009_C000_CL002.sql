-- Source: The relied-upon institution must be subject to a rule implementing the...
-- Clause ID: FFIEC_CIP_MANUAL_759301_FFIEC_CIP_MANUAL_S009_C000_CL002
-- Generated: 2026-02-11T18:36:04.664822+00:00
-- Confidence: 0.5


-- Obligation: unspecified entity must be
ALTER TABLE compliance_table
ADD CONSTRAINT chk_ffiec_cip_manual_759301_ffiec_cip_manual_s009_c000_cl002
CHECK (
    be_status = TRUE
);

-- Trigger for enforcement
CREATE OR REPLACE FUNCTION enforce_ffiec_cip_manual_759301_ffiec_cip_manual_s009_c000_cl002()
RETURNS TRIGGER AS $$
BEGIN
    IF NOT (be_status = TRUE) THEN
        RAISE EXCEPTION 'Compliance violation: FFIEC_CIP_MANUAL_759301_FFIEC_CIP_MANUAL_S009_C000_CL002 - unspecified entity must be';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_ffiec_cip_manual_759301_ffiec_cip_manual_s009_c000_cl002
BEFORE INSERT OR UPDATE ON compliance_table
FOR EACH ROW
EXECUTE FUNCTION enforce_ffiec_cip_manual_759301_ffiec_cip_manual_s009_c000_cl002();



COMMENT ON CONSTRAINT chk_ffiec_cip_manual_759301_ffiec_cip_manual_s009_c000_cl002 ON compliance_table
IS 'AegisLang: The relied-upon institution must be subject to a rule implementing the BSA/AML compliance program requirements.';
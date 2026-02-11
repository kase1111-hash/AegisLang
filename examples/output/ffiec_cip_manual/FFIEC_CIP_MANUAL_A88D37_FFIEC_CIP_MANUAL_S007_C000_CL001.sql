-- Source: Banks must determine whether a customer appears on any list of known o...
-- Clause ID: FFIEC_CIP_MANUAL_A88D37_FFIEC_CIP_MANUAL_S007_C000_CL001
-- Generated: 2026-02-11T18:20:55.765475+00:00
-- Confidence: 0.5


-- Obligation: Banks must determine
ALTER TABLE compliance_table
ADD CONSTRAINT chk_ffiec_cip_manual_a88d37_ffiec_cip_manual_s007_c000_cl001
CHECK (
    determine_status = TRUE
);

-- Trigger for enforcement
CREATE OR REPLACE FUNCTION enforce_ffiec_cip_manual_a88d37_ffiec_cip_manual_s007_c000_cl001()
RETURNS TRIGGER AS $$
BEGIN
    IF NOT (determine_status = TRUE) THEN
        RAISE EXCEPTION 'Compliance violation: FFIEC_CIP_MANUAL_A88D37_FFIEC_CIP_MANUAL_S007_C000_CL001 - Banks must determine';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_ffiec_cip_manual_a88d37_ffiec_cip_manual_s007_c000_cl001
BEFORE INSERT OR UPDATE ON compliance_table
FOR EACH ROW
EXECUTE FUNCTION enforce_ffiec_cip_manual_a88d37_ffiec_cip_manual_s007_c000_cl001();



COMMENT ON CONSTRAINT chk_ffiec_cip_manual_a88d37_ffiec_cip_manual_s007_c000_cl001 ON compliance_table
IS 'AegisLang: Banks must determine whether a customer appears on any list of known or suspected terrorists or terrorist organizations.';
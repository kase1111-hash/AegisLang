-- Source: Institutions must retain records of the methods used and results of an...
-- Clause ID: FFIEC_CIP_MANUAL_CA1E58_FFIEC_CIP_MANUAL_S006_C000_CL003
-- Generated: 2026-10-04T20:17:08.682910+00:00
-- Confidence: 0.5


-- Obligation: Institutions must retain
ALTER TABLE compliance_table
ADD CONSTRAINT chk_ffiec_cip_manual_ca1e58_ffiec_cip_manual_s006_c000_cl003
CHECK (
    retain_status = TRUE
);

-- Trigger for enforcement
CREATE OR REPLACE FUNCTION enforce_ffiec_cip_manual_ca1e58_ffiec_cip_manual_s006_c000_cl003()
RETURNS TRIGGER AS $$
BEGIN
    IF NOT (retain_status = TRUE) THEN
        RAISE EXCEPTION 'Compliance violation: FFIEC_CIP_MANUAL_CA1E58_FFIEC_CIP_MANUAL_S006_C000_CL003 - Institutions must retain';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_ffiec_cip_manual_ca1e58_ffiec_cip_manual_s006_c000_cl003
BEFORE INSERT OR UPDATE ON compliance_table
FOR EACH ROW
EXECUTE FUNCTION enforce_ffiec_cip_manual_ca1e58_ffiec_cip_manual_s006_c000_cl003();



COMMENT ON CONSTRAINT chk_ffiec_cip_manual_ca1e58_ffiec_cip_manual_s006_c000_cl003 ON compliance_table
IS 'AegisLang: Institutions must retain records of the methods used and results of any non-documentary verification measures.';

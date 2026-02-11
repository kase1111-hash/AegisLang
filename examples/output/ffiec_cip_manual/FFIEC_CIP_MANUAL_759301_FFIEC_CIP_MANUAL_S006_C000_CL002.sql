-- Source: Records shall include a description of any document relied upon for id...
-- Clause ID: FFIEC_CIP_MANUAL_759301_FFIEC_CIP_MANUAL_S006_C000_CL002
-- Generated: 2026-02-11T18:36:04.448253+00:00
-- Confidence: 0.5


-- Obligation: Records must include
ALTER TABLE compliance_table
ADD CONSTRAINT chk_ffiec_cip_manual_759301_ffiec_cip_manual_s006_c000_cl002
CHECK (
    include_status = TRUE
);

-- Trigger for enforcement
CREATE OR REPLACE FUNCTION enforce_ffiec_cip_manual_759301_ffiec_cip_manual_s006_c000_cl002()
RETURNS TRIGGER AS $$
BEGIN
    IF NOT (include_status = TRUE) THEN
        RAISE EXCEPTION 'Compliance violation: FFIEC_CIP_MANUAL_759301_FFIEC_CIP_MANUAL_S006_C000_CL002 - Records must include';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_ffiec_cip_manual_759301_ffiec_cip_manual_s006_c000_cl002
BEFORE INSERT OR UPDATE ON compliance_table
FOR EACH ROW
EXECUTE FUNCTION enforce_ffiec_cip_manual_759301_ffiec_cip_manual_s006_c000_cl002();



COMMENT ON CONSTRAINT chk_ffiec_cip_manual_759301_ffiec_cip_manual_s006_c000_cl002 ON compliance_table
IS 'AegisLang: Records shall include a description of any document relied upon for identity verification, including the type of document, identification number, place of issuance, and date of issuance or expiration.';
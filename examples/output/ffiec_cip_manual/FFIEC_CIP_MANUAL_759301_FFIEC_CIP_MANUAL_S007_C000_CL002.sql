-- Source: Institutions shall check customer names against the Office of Foreign ...
-- Clause ID: FFIEC_CIP_MANUAL_759301_FFIEC_CIP_MANUAL_S007_C000_CL002
-- Generated: 2026-02-11T18:36:04.520705+00:00
-- Confidence: 0.5


-- Obligation: Institutions must check
ALTER TABLE compliance_table
ADD CONSTRAINT chk_ffiec_cip_manual_759301_ffiec_cip_manual_s007_c000_cl002
CHECK (
    check_status = TRUE
);

-- Trigger for enforcement
CREATE OR REPLACE FUNCTION enforce_ffiec_cip_manual_759301_ffiec_cip_manual_s007_c000_cl002()
RETURNS TRIGGER AS $$
BEGIN
    IF NOT (check_status = TRUE) THEN
        RAISE EXCEPTION 'Compliance violation: FFIEC_CIP_MANUAL_759301_FFIEC_CIP_MANUAL_S007_C000_CL002 - Institutions must check';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_ffiec_cip_manual_759301_ffiec_cip_manual_s007_c000_cl002
BEFORE INSERT OR UPDATE ON compliance_table
FOR EACH ROW
EXECUTE FUNCTION enforce_ffiec_cip_manual_759301_ffiec_cip_manual_s007_c000_cl002();



COMMENT ON CONSTRAINT chk_ffiec_cip_manual_759301_ffiec_cip_manual_s007_c000_cl002 ON compliance_table
IS 'AegisLang: Institutions shall check customer names against the Office of Foreign Assets Control (OFAC) Specially Designated Nationals list.';
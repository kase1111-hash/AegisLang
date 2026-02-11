-- Source: The board of directors must approve the CIP.
-- Clause ID: FFIEC_CIP_MANUAL_A88D37_FFIEC_CIP_MANUAL_S004_C000_CL003
-- Generated: 2026-02-11T18:20:55.614925+00:00
-- Confidence: 0.5


-- Obligation: The board of directors must approve
ALTER TABLE compliance_table
ADD CONSTRAINT chk_ffiec_cip_manual_a88d37_ffiec_cip_manual_s004_c000_cl003
CHECK (
    approve_status = TRUE
);

-- Trigger for enforcement
CREATE OR REPLACE FUNCTION enforce_ffiec_cip_manual_a88d37_ffiec_cip_manual_s004_c000_cl003()
RETURNS TRIGGER AS $$
BEGIN
    IF NOT (approve_status = TRUE) THEN
        RAISE EXCEPTION 'Compliance violation: FFIEC_CIP_MANUAL_A88D37_FFIEC_CIP_MANUAL_S004_C000_CL003 - The board of directors must approve';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_ffiec_cip_manual_a88d37_ffiec_cip_manual_s004_c000_cl003
BEFORE INSERT OR UPDATE ON compliance_table
FOR EACH ROW
EXECUTE FUNCTION enforce_ffiec_cip_manual_a88d37_ffiec_cip_manual_s004_c000_cl003();



COMMENT ON CONSTRAINT chk_ffiec_cip_manual_a88d37_ffiec_cip_manual_s004_c000_cl003 ON compliance_table
IS 'AegisLang: The board of directors must approve the CIP.';
-- Source: The CIP must be incorporated into the bank's BSA/AML compliance progra...
-- Clause ID: FFIEC_CIP_MANUAL_A88D37_FFIEC_CIP_MANUAL_S004_C000_CL002
-- Generated: 2026-02-11T18:20:55.589097+00:00
-- Confidence: 0.5


-- Obligation: The CIP must be
ALTER TABLE compliance_table
ADD CONSTRAINT chk_ffiec_cip_manual_a88d37_ffiec_cip_manual_s004_c000_cl002
CHECK (
    be_status = TRUE
);

-- Trigger for enforcement
CREATE OR REPLACE FUNCTION enforce_ffiec_cip_manual_a88d37_ffiec_cip_manual_s004_c000_cl002()
RETURNS TRIGGER AS $$
BEGIN
    IF NOT (be_status = TRUE) THEN
        RAISE EXCEPTION 'Compliance violation: FFIEC_CIP_MANUAL_A88D37_FFIEC_CIP_MANUAL_S004_C000_CL002 - The CIP must be';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_ffiec_cip_manual_a88d37_ffiec_cip_manual_s004_c000_cl002
BEFORE INSERT OR UPDATE ON compliance_table
FOR EACH ROW
EXECUTE FUNCTION enforce_ffiec_cip_manual_a88d37_ffiec_cip_manual_s004_c000_cl002();



COMMENT ON CONSTRAINT chk_ffiec_cip_manual_a88d37_ffiec_cip_manual_s004_c000_cl002 ON compliance_table
IS 'AegisLang: The CIP must be incorporated into the bank''s BSA/AML compliance program.';
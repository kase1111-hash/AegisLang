-- Source: Institutions shall retain beneficial ownership information for five ye...
-- Clause ID: FINCEN_CDD_RULE_BBA7C7_FINCEN_CDD_RULE_S009_C000_CL002
-- Generated: 2026-02-11T18:20:56.357955+00:00
-- Confidence: 0.5


-- Obligation: Institutions must retain
ALTER TABLE compliance_table
ADD CONSTRAINT chk_fincen_cdd_rule_bba7c7_fincen_cdd_rule_s009_c000_cl002
CHECK (
    retain_status = TRUE
);

-- Trigger for enforcement
CREATE OR REPLACE FUNCTION enforce_fincen_cdd_rule_bba7c7_fincen_cdd_rule_s009_c000_cl002()
RETURNS TRIGGER AS $$
BEGIN
    IF NOT (retain_status = TRUE) THEN
        RAISE EXCEPTION 'Compliance violation: FINCEN_CDD_RULE_BBA7C7_FINCEN_CDD_RULE_S009_C000_CL002 - Institutions must retain';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_fincen_cdd_rule_bba7c7_fincen_cdd_rule_s009_c000_cl002
BEFORE INSERT OR UPDATE ON compliance_table
FOR EACH ROW
EXECUTE FUNCTION enforce_fincen_cdd_rule_bba7c7_fincen_cdd_rule_s009_c000_cl002();



COMMENT ON CONSTRAINT chk_fincen_cdd_rule_bba7c7_fincen_cdd_rule_s009_c000_cl002 ON compliance_table
IS 'AegisLang: Institutions shall retain beneficial ownership information for five years after the date the account is closed.';
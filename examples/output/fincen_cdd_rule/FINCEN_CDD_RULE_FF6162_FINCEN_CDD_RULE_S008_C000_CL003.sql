-- Source: Staff must file Suspicious Activity Reports for transactions that are ...
-- Clause ID: FINCEN_CDD_RULE_FF6162_FINCEN_CDD_RULE_S008_C000_CL003
-- Generated: 2026-02-11T18:36:05.024828+00:00
-- Confidence: 0.9


-- Obligation: Staff must file
ALTER TABLE audit_log
ADD CONSTRAINT chk_fincen_cdd_rule_ff6162_fincen_cdd_rule_s008_c000_cl003
CHECK (
    file_status = TRUE
);

-- Trigger for enforcement
CREATE OR REPLACE FUNCTION enforce_fincen_cdd_rule_ff6162_fincen_cdd_rule_s008_c000_cl003()
RETURNS TRIGGER AS $$
BEGIN
    IF NOT (file_status = TRUE) THEN
        RAISE EXCEPTION 'Compliance violation: FINCEN_CDD_RULE_FF6162_FINCEN_CDD_RULE_S008_C000_CL003 - Staff must file';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_fincen_cdd_rule_ff6162_fincen_cdd_rule_s008_c000_cl003
BEFORE INSERT OR UPDATE ON audit_log
FOR EACH ROW
EXECUTE FUNCTION enforce_fincen_cdd_rule_ff6162_fincen_cdd_rule_s008_c000_cl003();



COMMENT ON CONSTRAINT chk_fincen_cdd_rule_ff6162_fincen_cdd_rule_s008_c000_cl003 ON audit_log
IS 'AegisLang: Staff must file Suspicious Activity Reports for transactions that are suspicious or have no lawful purpose.';
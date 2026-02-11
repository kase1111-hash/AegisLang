-- Source: Banks must update customer risk profiles on a periodic basis.
-- Clause ID: FINCEN_CDD_RULE_FF6162_FINCEN_CDD_RULE_S007_C000_CL003
-- Generated: 2026-02-11T18:36:04.951986+00:00
-- Confidence: 0.5


-- Obligation: Banks must update
ALTER TABLE compliance_table
ADD CONSTRAINT chk_fincen_cdd_rule_ff6162_fincen_cdd_rule_s007_c000_cl003
CHECK (
    update_status = TRUE
);

-- Trigger for enforcement
CREATE OR REPLACE FUNCTION enforce_fincen_cdd_rule_ff6162_fincen_cdd_rule_s007_c000_cl003()
RETURNS TRIGGER AS $$
BEGIN
    IF NOT (update_status = TRUE) THEN
        RAISE EXCEPTION 'Compliance violation: FINCEN_CDD_RULE_FF6162_FINCEN_CDD_RULE_S007_C000_CL003 - Banks must update';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_fincen_cdd_rule_ff6162_fincen_cdd_rule_s007_c000_cl003
BEFORE INSERT OR UPDATE ON compliance_table
FOR EACH ROW
EXECUTE FUNCTION enforce_fincen_cdd_rule_ff6162_fincen_cdd_rule_s007_c000_cl003();



COMMENT ON CONSTRAINT chk_fincen_cdd_rule_ff6162_fincen_cdd_rule_s007_c000_cl003 ON compliance_table
IS 'AegisLang: Banks must update customer risk profiles on a periodic basis.';
-- Source: Financial institutions must maintain records of customer identificatio...
-- Clause ID: FINCEN_CDD_RULE_FF6162_FINCEN_CDD_RULE_S009_C000_CL001
-- Generated: 2026-02-11T18:36:05.049066+00:00
-- Confidence: 0.5


-- Obligation: Financial institutions must maintain
ALTER TABLE compliance_table
ADD CONSTRAINT chk_fincen_cdd_rule_ff6162_fincen_cdd_rule_s009_c000_cl001
CHECK (
    maintain_status = TRUE
);

-- Trigger for enforcement
CREATE OR REPLACE FUNCTION enforce_fincen_cdd_rule_ff6162_fincen_cdd_rule_s009_c000_cl001()
RETURNS TRIGGER AS $$
BEGIN
    IF NOT (maintain_status = TRUE) THEN
        RAISE EXCEPTION 'Compliance violation: FINCEN_CDD_RULE_FF6162_FINCEN_CDD_RULE_S009_C000_CL001 - Financial institutions must maintain';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_fincen_cdd_rule_ff6162_fincen_cdd_rule_s009_c000_cl001
BEFORE INSERT OR UPDATE ON compliance_table
FOR EACH ROW
EXECUTE FUNCTION enforce_fincen_cdd_rule_ff6162_fincen_cdd_rule_s009_c000_cl001();



COMMENT ON CONSTRAINT chk_fincen_cdd_rule_ff6162_fincen_cdd_rule_s009_c000_cl001 ON compliance_table
IS 'AegisLang: Financial institutions must maintain records of customer identification information for five years after account closure.';
-- Source: Institutions must conduct ongoing monitoring to maintain and update cu...
-- Clause ID: FINCEN_CDD_RULE_FF6162_FINCEN_CDD_RULE_S008_C000_CL001
-- Generated: 2026-02-11T18:36:04.976287+00:00
-- Confidence: 0.5


-- Obligation: Institutions must conduct
ALTER TABLE compliance_table
ADD CONSTRAINT chk_fincen_cdd_rule_ff6162_fincen_cdd_rule_s008_c000_cl001
CHECK (
    conduct_status = TRUE
);

-- Trigger for enforcement
CREATE OR REPLACE FUNCTION enforce_fincen_cdd_rule_ff6162_fincen_cdd_rule_s008_c000_cl001()
RETURNS TRIGGER AS $$
BEGIN
    IF NOT (conduct_status = TRUE) THEN
        RAISE EXCEPTION 'Compliance violation: FINCEN_CDD_RULE_FF6162_FINCEN_CDD_RULE_S008_C000_CL001 - Institutions must conduct';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_fincen_cdd_rule_ff6162_fincen_cdd_rule_s008_c000_cl001
BEFORE INSERT OR UPDATE ON compliance_table
FOR EACH ROW
EXECUTE FUNCTION enforce_fincen_cdd_rule_ff6162_fincen_cdd_rule_s008_c000_cl001();



COMMENT ON CONSTRAINT chk_fincen_cdd_rule_ff6162_fincen_cdd_rule_s008_c000_cl001 ON compliance_table
IS 'AegisLang: Institutions must conduct ongoing monitoring to maintain and update customer information.';
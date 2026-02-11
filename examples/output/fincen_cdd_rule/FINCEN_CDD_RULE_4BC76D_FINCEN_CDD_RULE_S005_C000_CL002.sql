-- Source: Banks shall collect the following information at account opening: name...
-- Clause ID: FINCEN_CDD_RULE_4BC76D_FINCEN_CDD_RULE_S005_C000_CL002
-- Generated: 2026-02-11T18:21:29.606150+00:00
-- Confidence: 0.5


-- Obligation: Banks must collect
ALTER TABLE compliance_table
ADD CONSTRAINT chk_fincen_cdd_rule_4bc76d_fincen_cdd_rule_s005_c000_cl002
CHECK (
    collect_status = TRUE
);

-- Trigger for enforcement
CREATE OR REPLACE FUNCTION enforce_fincen_cdd_rule_4bc76d_fincen_cdd_rule_s005_c000_cl002()
RETURNS TRIGGER AS $$
BEGIN
    IF NOT (collect_status = TRUE) THEN
        RAISE EXCEPTION 'Compliance violation: FINCEN_CDD_RULE_4BC76D_FINCEN_CDD_RULE_S005_C000_CL002 - Banks must collect';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_fincen_cdd_rule_4bc76d_fincen_cdd_rule_s005_c000_cl002
BEFORE INSERT OR UPDATE ON compliance_table
FOR EACH ROW
EXECUTE FUNCTION enforce_fincen_cdd_rule_4bc76d_fincen_cdd_rule_s005_c000_cl002();



COMMENT ON CONSTRAINT chk_fincen_cdd_rule_4bc76d_fincen_cdd_rule_s005_c000_cl002 ON compliance_table
IS 'AegisLang: Banks shall collect the following information at account opening: name, date of birth, address, and identification number.';
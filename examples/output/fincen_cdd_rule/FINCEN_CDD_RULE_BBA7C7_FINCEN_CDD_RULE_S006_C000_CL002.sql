-- Source: Institutions shall identify one individual with significant responsibi...
-- Clause ID: FINCEN_CDD_RULE_BBA7C7_FINCEN_CDD_RULE_S006_C000_CL002
-- Generated: 2026-02-11T18:20:56.156319+00:00
-- Confidence: 0.5


-- Obligation: Institutions must identify
ALTER TABLE compliance_table
ADD CONSTRAINT chk_fincen_cdd_rule_bba7c7_fincen_cdd_rule_s006_c000_cl002
CHECK (
    identify_status = TRUE
);

-- Trigger for enforcement
CREATE OR REPLACE FUNCTION enforce_fincen_cdd_rule_bba7c7_fincen_cdd_rule_s006_c000_cl002()
RETURNS TRIGGER AS $$
BEGIN
    IF NOT (identify_status = TRUE) THEN
        RAISE EXCEPTION 'Compliance violation: FINCEN_CDD_RULE_BBA7C7_FINCEN_CDD_RULE_S006_C000_CL002 - Institutions must identify';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_fincen_cdd_rule_bba7c7_fincen_cdd_rule_s006_c000_cl002
BEFORE INSERT OR UPDATE ON compliance_table
FOR EACH ROW
EXECUTE FUNCTION enforce_fincen_cdd_rule_bba7c7_fincen_cdd_rule_s006_c000_cl002();



COMMENT ON CONSTRAINT chk_fincen_cdd_rule_bba7c7_fincen_cdd_rule_s006_c000_cl002 ON compliance_table
IS 'AegisLang: Institutions shall identify one individual with significant responsibility for managing the legal entity.';
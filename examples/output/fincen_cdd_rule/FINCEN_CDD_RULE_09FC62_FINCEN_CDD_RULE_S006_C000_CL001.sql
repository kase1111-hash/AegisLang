-- Source: Covered financial institutions must identify and verify the identity o...
-- Clause ID: FINCEN_CDD_RULE_09FC62_FINCEN_CDD_RULE_S006_C000_CL001
-- Generated: 2026-10-04T20:17:09.556660+00:00
-- Confidence: 0.95


-- Obligation: Covered financial institutions must identify
ALTER TABLE customers
ADD CONSTRAINT chk_fincen_cdd_rule_09fc62_fincen_cdd_rule_s006_c000_cl001
CHECK (
    identify_status = TRUE
);

-- Trigger for enforcement
CREATE OR REPLACE FUNCTION enforce_fincen_cdd_rule_09fc62_fincen_cdd_rule_s006_c000_cl001()
RETURNS TRIGGER AS $$
BEGIN
    IF NOT (identify_status = TRUE) THEN
        RAISE EXCEPTION 'Compliance violation: FINCEN_CDD_RULE_09FC62_FINCEN_CDD_RULE_S006_C000_CL001 - Covered financial institutions must identify';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_fincen_cdd_rule_09fc62_fincen_cdd_rule_s006_c000_cl001
BEFORE INSERT OR UPDATE ON customers
FOR EACH ROW
EXECUTE FUNCTION enforce_fincen_cdd_rule_09fc62_fincen_cdd_rule_s006_c000_cl001();



COMMENT ON CONSTRAINT chk_fincen_cdd_rule_09fc62_fincen_cdd_rule_s006_c000_cl001 ON customers
IS 'AegisLang: Covered financial institutions must identify and verify the identity of beneficial owners of legal entity customers.';

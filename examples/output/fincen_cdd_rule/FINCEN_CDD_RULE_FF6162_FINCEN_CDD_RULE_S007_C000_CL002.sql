-- Source: Institutions shall assign risk ratings to customers using criteria tha...
-- Clause ID: FINCEN_CDD_RULE_FF6162_FINCEN_CDD_RULE_S007_C000_CL002
-- Generated: 2026-02-11T18:36:04.926253+00:00
-- Confidence: 0.95


-- Obligation: Institutions must assign
ALTER TABLE customers
ADD CONSTRAINT chk_fincen_cdd_rule_ff6162_fincen_cdd_rule_s007_c000_cl002
CHECK (
    assign_status = TRUE
);

-- Trigger for enforcement
CREATE OR REPLACE FUNCTION enforce_fincen_cdd_rule_ff6162_fincen_cdd_rule_s007_c000_cl002()
RETURNS TRIGGER AS $$
BEGIN
    IF NOT (assign_status = TRUE) THEN
        RAISE EXCEPTION 'Compliance violation: FINCEN_CDD_RULE_FF6162_FINCEN_CDD_RULE_S007_C000_CL002 - Institutions must assign';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_fincen_cdd_rule_ff6162_fincen_cdd_rule_s007_c000_cl002
BEFORE INSERT OR UPDATE ON customers
FOR EACH ROW
EXECUTE FUNCTION enforce_fincen_cdd_rule_ff6162_fincen_cdd_rule_s007_c000_cl002();



COMMENT ON CONSTRAINT chk_fincen_cdd_rule_ff6162_fincen_cdd_rule_s007_c000_cl002 ON customers
IS 'AegisLang: Institutions shall assign risk ratings to customers using criteria that include the nature of the customer''s business, geographic location, and transaction patterns.';
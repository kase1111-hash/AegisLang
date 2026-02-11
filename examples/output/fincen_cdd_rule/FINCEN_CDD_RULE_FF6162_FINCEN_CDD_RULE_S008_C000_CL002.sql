-- Source: Banks shall monitor transactions to identify suspicious activity consi...
-- Clause ID: FINCEN_CDD_RULE_FF6162_FINCEN_CDD_RULE_S008_C000_CL002
-- Generated: 2026-02-11T18:36:05.000522+00:00
-- Confidence: 0.85


-- Obligation: Banks must monitor
ALTER TABLE transactions
ADD CONSTRAINT chk_fincen_cdd_rule_ff6162_fincen_cdd_rule_s008_c000_cl002
CHECK (
    monitor_status = TRUE
);

-- Trigger for enforcement
CREATE OR REPLACE FUNCTION enforce_fincen_cdd_rule_ff6162_fincen_cdd_rule_s008_c000_cl002()
RETURNS TRIGGER AS $$
BEGIN
    IF NOT (monitor_status = TRUE) THEN
        RAISE EXCEPTION 'Compliance violation: FINCEN_CDD_RULE_FF6162_FINCEN_CDD_RULE_S008_C000_CL002 - Banks must monitor';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_fincen_cdd_rule_ff6162_fincen_cdd_rule_s008_c000_cl002
BEFORE INSERT OR UPDATE ON transactions
FOR EACH ROW
EXECUTE FUNCTION enforce_fincen_cdd_rule_ff6162_fincen_cdd_rule_s008_c000_cl002();



COMMENT ON CONSTRAINT chk_fincen_cdd_rule_ff6162_fincen_cdd_rule_s008_c000_cl002 ON transactions
IS 'AegisLang: Banks shall monitor transactions to identify suspicious activity consistent with the customer''s risk profile.';
-- Source: Banks must not destroy records that are subject to ongoing investigati...
-- Clause ID: FINCEN_CDD_RULE_09FC62_FINCEN_CDD_RULE_S009_C000_CL003
-- Generated: 2026-10-04T20:17:09.804325+00:00
-- Confidence: 0.5


-- Prohibition: Banks must not destroy
ALTER TABLE compliance_table
ADD CONSTRAINT chk_fincen_cdd_rule_09fc62_fincen_cdd_rule_s009_c000_cl003_prohibit
CHECK (
    NOT (NOT destroy_status)
);



COMMENT ON CONSTRAINT chk_fincen_cdd_rule_09fc62_fincen_cdd_rule_s009_c000_cl003_prohibit ON compliance_table
IS 'AegisLang: Banks must not destroy records that are subject to ongoing investigations or legal proceedings.';

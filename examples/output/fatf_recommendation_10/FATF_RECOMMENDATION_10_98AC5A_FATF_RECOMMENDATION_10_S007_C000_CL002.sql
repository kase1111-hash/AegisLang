-- Source: Simplified measures shall not apply when there is a suspicion of money...
-- Clause ID: FATF_RECOMMENDATION_10_98AC5A_FATF_RECOMMENDATION_10_S007_C000_CL002
-- Generated: 2026-02-11T18:21:28.798345+00:00
-- Confidence: 0.5


-- Prohibition: Simplified measures must not apply
ALTER TABLE compliance_table
ADD CONSTRAINT chk_fatf_recommendation_10_98ac5a_fatf_recommendation_10_s007_c000_cl002_prohibit
CHECK (
    NOT (NOT apply_status)
);



COMMENT ON CONSTRAINT chk_fatf_recommendation_10_98ac5a_fatf_recommendation_10_s007_c000_cl002_prohibit ON compliance_table
IS 'AegisLang: Simplified measures shall not apply when there is a suspicion of money laundering or terrorist financing.';
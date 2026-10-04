-- Source: Financial institutions should not open the account or commence the bus...
-- Clause ID: FATF_RECOMMENDATION_10_8D5616_FATF_RECOMMENDATION_10_S008_C000_CL001
-- Generated: 2026-10-04T20:17:07.866677+00:00
-- Confidence: 0.5


-- Prohibition: Financial institutions must not open
ALTER TABLE compliance_table
ADD CONSTRAINT chk_fatf_recommendation_10_8d5616_fatf_recommendation_10_s008_c000_cl001_prohibit
CHECK (
    NOT (NOT open_status)
);



COMMENT ON CONSTRAINT chk_fatf_recommendation_10_8d5616_fatf_recommendation_10_s008_c000_cl001_prohibit ON compliance_table
IS 'AegisLang: Financial institutions should not open the account or commence the business relationship when they are unable to satisfactorily complete CDD measures.';

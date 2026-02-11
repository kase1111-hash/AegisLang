-- Source: Institutions must satisfy themselves that the third party is regulated...
-- Clause ID: FATF_RECOMMENDATION_10_2F3583_FATF_RECOMMENDATION_10_S009_C000_CL002
-- Generated: 2026-02-11T18:36:04.181932+00:00
-- Confidence: 0.5


-- Obligation: Institutions must satisfy
ALTER TABLE compliance_table
ADD CONSTRAINT chk_fatf_recommendation_10_2f3583_fatf_recommendation_10_s009_c000_cl002
CHECK (
    satisfy_status = TRUE
);

-- Trigger for enforcement
CREATE OR REPLACE FUNCTION enforce_fatf_recommendation_10_2f3583_fatf_recommendation_10_s009_c000_cl002()
RETURNS TRIGGER AS $$
BEGIN
    IF NOT (satisfy_status = TRUE) THEN
        RAISE EXCEPTION 'Compliance violation: FATF_RECOMMENDATION_10_2F3583_FATF_RECOMMENDATION_10_S009_C000_CL002 - Institutions must satisfy';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_fatf_recommendation_10_2f3583_fatf_recommendation_10_s009_c000_cl002
BEFORE INSERT OR UPDATE ON compliance_table
FOR EACH ROW
EXECUTE FUNCTION enforce_fatf_recommendation_10_2f3583_fatf_recommendation_10_s009_c000_cl002();



COMMENT ON CONSTRAINT chk_fatf_recommendation_10_2f3583_fatf_recommendation_10_s009_c000_cl002 ON compliance_table
IS 'AegisLang: Institutions must satisfy themselves that the third party is regulated and supervised for AML/CFT requirements.';
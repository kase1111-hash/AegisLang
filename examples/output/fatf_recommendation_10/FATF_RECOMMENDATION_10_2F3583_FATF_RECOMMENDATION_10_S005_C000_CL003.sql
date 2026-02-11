-- Source: Financial institutions shall understand and obtain information on the ...
-- Clause ID: FATF_RECOMMENDATION_10_2F3583_FATF_RECOMMENDATION_10_S005_C000_CL003
-- Generated: 2026-02-11T18:36:03.909685+00:00
-- Confidence: 0.5


-- Obligation: Financial institutions must understand
ALTER TABLE compliance_table
ADD CONSTRAINT chk_fatf_recommendation_10_2f3583_fatf_recommendation_10_s005_c000_cl003
CHECK (
    understand_status = TRUE
);

-- Trigger for enforcement
CREATE OR REPLACE FUNCTION enforce_fatf_recommendation_10_2f3583_fatf_recommendation_10_s005_c000_cl003()
RETURNS TRIGGER AS $$
BEGIN
    IF NOT (understand_status = TRUE) THEN
        RAISE EXCEPTION 'Compliance violation: FATF_RECOMMENDATION_10_2F3583_FATF_RECOMMENDATION_10_S005_C000_CL003 - Financial institutions must understand';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_fatf_recommendation_10_2f3583_fatf_recommendation_10_s005_c000_cl003
BEFORE INSERT OR UPDATE ON compliance_table
FOR EACH ROW
EXECUTE FUNCTION enforce_fatf_recommendation_10_2f3583_fatf_recommendation_10_s005_c000_cl003();



COMMENT ON CONSTRAINT chk_fatf_recommendation_10_2f3583_fatf_recommendation_10_s005_c000_cl003 ON compliance_table
IS 'AegisLang: Financial institutions shall understand and obtain information on the purpose and intended nature of the business relationship.';
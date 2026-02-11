-- Source: CDD is required when there is a suspicion of money laundering or terro...
-- Clause ID: FATF_RECOMMENDATION_10_2F3583_FATF_RECOMMENDATION_10_S004_C000_CL003
-- Generated: 2026-02-11T18:36:03.811263+00:00
-- Confidence: 0.5


-- Obligation: unspecified entity must comply
ALTER TABLE compliance_table
ADD CONSTRAINT chk_fatf_recommendation_10_2f3583_fatf_recommendation_10_s004_c000_cl003
CHECK (
    comply_status = TRUE
);

-- Trigger for enforcement
CREATE OR REPLACE FUNCTION enforce_fatf_recommendation_10_2f3583_fatf_recommendation_10_s004_c000_cl003()
RETURNS TRIGGER AS $$
BEGIN
    IF NOT (comply_status = TRUE) THEN
        RAISE EXCEPTION 'Compliance violation: FATF_RECOMMENDATION_10_2F3583_FATF_RECOMMENDATION_10_S004_C000_CL003 - unspecified entity must comply';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_fatf_recommendation_10_2f3583_fatf_recommendation_10_s004_c000_cl003
BEFORE INSERT OR UPDATE ON compliance_table
FOR EACH ROW
EXECUTE FUNCTION enforce_fatf_recommendation_10_2f3583_fatf_recommendation_10_s004_c000_cl003();



COMMENT ON CONSTRAINT chk_fatf_recommendation_10_2f3583_fatf_recommendation_10_s004_c000_cl003 ON compliance_table
IS 'AegisLang: CDD is required when there is a suspicion of money laundering or terrorist financing regardless of any threshold.';
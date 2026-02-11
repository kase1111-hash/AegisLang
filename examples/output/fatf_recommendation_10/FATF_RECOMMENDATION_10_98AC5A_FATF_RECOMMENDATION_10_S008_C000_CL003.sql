-- Source: Existing customers whose identity cannot be verified must be subject t...
-- Clause ID: FATF_RECOMMENDATION_10_98AC5A_FATF_RECOMMENDATION_10_S008_C000_CL003
-- Generated: 2026-02-11T18:21:28.902149+00:00
-- Confidence: 0.5


-- Obligation: Existing customers whose identity cannot be verified must be
ALTER TABLE compliance_table
ADD CONSTRAINT chk_fatf_recommendation_10_98ac5a_fatf_recommendation_10_s008_c000_cl003
CHECK (
    be_status = TRUE
);

-- Trigger for enforcement
CREATE OR REPLACE FUNCTION enforce_fatf_recommendation_10_98ac5a_fatf_recommendation_10_s008_c000_cl003()
RETURNS TRIGGER AS $$
BEGIN
    IF NOT (be_status = TRUE) THEN
        RAISE EXCEPTION 'Compliance violation: FATF_RECOMMENDATION_10_98AC5A_FATF_RECOMMENDATION_10_S008_C000_CL003 - Existing customers whose identity cannot be verified must be';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_fatf_recommendation_10_98ac5a_fatf_recommendation_10_s008_c000_cl003
BEFORE INSERT OR UPDATE ON compliance_table
FOR EACH ROW
EXECUTE FUNCTION enforce_fatf_recommendation_10_98ac5a_fatf_recommendation_10_s008_c000_cl003();



COMMENT ON CONSTRAINT chk_fatf_recommendation_10_98ac5a_fatf_recommendation_10_s008_c000_cl003 ON compliance_table
IS 'AegisLang: Existing customers whose identity cannot be verified must be subject to enhanced monitoring or account termination.';
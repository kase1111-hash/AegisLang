-- Source: Politically exposed persons must be subject to enhanced due diligence ...
-- Clause ID: FATF_RECOMMENDATION_10_98AC5A_FATF_RECOMMENDATION_10_S006_C000_CL002
-- Generated: 2026-02-11T18:21:28.750098+00:00
-- Confidence: 0.5


-- Obligation: Politically exposed persons must be
ALTER TABLE compliance_table
ADD CONSTRAINT chk_fatf_recommendation_10_98ac5a_fatf_recommendation_10_s006_c000_cl002
CHECK (
    be_status = TRUE
);

-- Trigger for enforcement
CREATE OR REPLACE FUNCTION enforce_fatf_recommendation_10_98ac5a_fatf_recommendation_10_s006_c000_cl002()
RETURNS TRIGGER AS $$
BEGIN
    IF NOT (be_status = TRUE) THEN
        RAISE EXCEPTION 'Compliance violation: FATF_RECOMMENDATION_10_98AC5A_FATF_RECOMMENDATION_10_S006_C000_CL002 - Politically exposed persons must be';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_fatf_recommendation_10_98ac5a_fatf_recommendation_10_s006_c000_cl002
BEFORE INSERT OR UPDATE ON compliance_table
FOR EACH ROW
EXECUTE FUNCTION enforce_fatf_recommendation_10_98ac5a_fatf_recommendation_10_s006_c000_cl002();



COMMENT ON CONSTRAINT chk_fatf_recommendation_10_98ac5a_fatf_recommendation_10_s006_c000_cl002 ON compliance_table
IS 'AegisLang: Politically exposed persons must be subject to enhanced due diligence measures including senior management approval for establishing the business relationship.';
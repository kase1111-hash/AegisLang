-- Source: Institutions must identify the beneficial owner and take reasonable me...
-- Clause ID: FATF_RECOMMENDATION_10_2F3583_FATF_RECOMMENDATION_10_S005_C000_CL002
-- Generated: 2026-02-11T18:36:03.884491+00:00
-- Confidence: 0.95


-- Obligation: Institutions must identify
ALTER TABLE customers
ADD CONSTRAINT chk_fatf_recommendation_10_2f3583_fatf_recommendation_10_s005_c000_cl002
CHECK (
    identify_status = TRUE
);

-- Trigger for enforcement
CREATE OR REPLACE FUNCTION enforce_fatf_recommendation_10_2f3583_fatf_recommendation_10_s005_c000_cl002()
RETURNS TRIGGER AS $$
BEGIN
    IF NOT (identify_status = TRUE) THEN
        RAISE EXCEPTION 'Compliance violation: FATF_RECOMMENDATION_10_2F3583_FATF_RECOMMENDATION_10_S005_C000_CL002 - Institutions must identify';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_fatf_recommendation_10_2f3583_fatf_recommendation_10_s005_c000_cl002
BEFORE INSERT OR UPDATE ON customers
FOR EACH ROW
EXECUTE FUNCTION enforce_fatf_recommendation_10_2f3583_fatf_recommendation_10_s005_c000_cl002();



COMMENT ON CONSTRAINT chk_fatf_recommendation_10_2f3583_fatf_recommendation_10_s005_c000_cl002 ON customers
IS 'AegisLang: Institutions must identify the beneficial owner and take reasonable measures to verify the identity of the beneficial owner.';
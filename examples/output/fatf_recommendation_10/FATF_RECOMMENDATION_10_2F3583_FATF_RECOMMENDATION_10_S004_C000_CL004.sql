-- Source: Institutions shall perform CDD when there are doubts about the veracit...
-- Clause ID: FATF_RECOMMENDATION_10_2F3583_FATF_RECOMMENDATION_10_S004_C000_CL004
-- Generated: 2026-02-11T18:36:03.835629+00:00
-- Confidence: 0.5


-- Obligation: Institutions must perform
ALTER TABLE compliance_table
ADD CONSTRAINT chk_fatf_recommendation_10_2f3583_fatf_recommendation_10_s004_c000_cl004
CHECK (
    perform_status = TRUE
);

-- Trigger for enforcement
CREATE OR REPLACE FUNCTION enforce_fatf_recommendation_10_2f3583_fatf_recommendation_10_s004_c000_cl004()
RETURNS TRIGGER AS $$
BEGIN
    IF NOT (perform_status = TRUE) THEN
        RAISE EXCEPTION 'Compliance violation: FATF_RECOMMENDATION_10_2F3583_FATF_RECOMMENDATION_10_S004_C000_CL004 - Institutions must perform';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_fatf_recommendation_10_2f3583_fatf_recommendation_10_s004_c000_cl004
BEFORE INSERT OR UPDATE ON compliance_table
FOR EACH ROW
EXECUTE FUNCTION enforce_fatf_recommendation_10_2f3583_fatf_recommendation_10_s004_c000_cl004();



COMMENT ON CONSTRAINT chk_fatf_recommendation_10_2f3583_fatf_recommendation_10_s004_c000_cl004 ON compliance_table
IS 'AegisLang: Institutions shall perform CDD when there are doubts about the veracity of previously obtained customer identification data.';
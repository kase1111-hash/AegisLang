-- Source: Institutions shall perform CDD when there are doubts about the veracit...
-- Clause ID: FATF_RECOMMENDATION_10_98AC5A_FATF_RECOMMENDATION_10_S004_C000_CL004
-- Generated: 2026-02-11T18:21:28.599110+00:00
-- Confidence: 0.5


-- Obligation: Institutions must perform
ALTER TABLE compliance_table
ADD CONSTRAINT chk_fatf_recommendation_10_98ac5a_fatf_recommendation_10_s004_c000_cl004
CHECK (
    perform_status = TRUE
);

-- Trigger for enforcement
CREATE OR REPLACE FUNCTION enforce_fatf_recommendation_10_98ac5a_fatf_recommendation_10_s004_c000_cl004()
RETURNS TRIGGER AS $$
BEGIN
    IF NOT (perform_status = TRUE) THEN
        RAISE EXCEPTION 'Compliance violation: FATF_RECOMMENDATION_10_98AC5A_FATF_RECOMMENDATION_10_S004_C000_CL004 - Institutions must perform';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_fatf_recommendation_10_98ac5a_fatf_recommendation_10_s004_c000_cl004
BEFORE INSERT OR UPDATE ON compliance_table
FOR EACH ROW
EXECUTE FUNCTION enforce_fatf_recommendation_10_98ac5a_fatf_recommendation_10_s004_c000_cl004();



COMMENT ON CONSTRAINT chk_fatf_recommendation_10_98ac5a_fatf_recommendation_10_s004_c000_cl004 ON compliance_table
IS 'AegisLang: Institutions shall perform CDD when there are doubts about the veracity of previously obtained customer identification data.';
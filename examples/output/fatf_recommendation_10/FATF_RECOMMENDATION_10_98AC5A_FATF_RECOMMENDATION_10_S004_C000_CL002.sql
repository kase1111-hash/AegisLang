-- Source: Institutions must perform CDD when carrying out occasional transaction...
-- Clause ID: FATF_RECOMMENDATION_10_98AC5A_FATF_RECOMMENDATION_10_S004_C000_CL002
-- Generated: 2026-02-11T18:21:28.549057+00:00
-- Confidence: 0.5


-- Obligation: Institutions must perform
ALTER TABLE compliance_table
ADD CONSTRAINT chk_fatf_recommendation_10_98ac5a_fatf_recommendation_10_s004_c000_cl002
CHECK (
    perform_status = TRUE
);

-- Trigger for enforcement
CREATE OR REPLACE FUNCTION enforce_fatf_recommendation_10_98ac5a_fatf_recommendation_10_s004_c000_cl002()
RETURNS TRIGGER AS $$
BEGIN
    IF NOT (perform_status = TRUE) THEN
        RAISE EXCEPTION 'Compliance violation: FATF_RECOMMENDATION_10_98AC5A_FATF_RECOMMENDATION_10_S004_C000_CL002 - Institutions must perform';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_fatf_recommendation_10_98ac5a_fatf_recommendation_10_s004_c000_cl002
BEFORE INSERT OR UPDATE ON compliance_table
FOR EACH ROW
EXECUTE FUNCTION enforce_fatf_recommendation_10_98ac5a_fatf_recommendation_10_s004_c000_cl002();



COMMENT ON CONSTRAINT chk_fatf_recommendation_10_98ac5a_fatf_recommendation_10_s004_c000_cl002 ON compliance_table
IS 'AegisLang: Institutions must perform CDD when carrying out occasional transactions above the applicable designated threshold of USD 15,000.';
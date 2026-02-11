-- Source: Institutions must document the risk assessment that justifies the appl...
-- Clause ID: FATF_RECOMMENDATION_10_2A2FA7_FATF_RECOMMENDATION_10_S007_C000_CL003
-- Generated: 2026-02-11T18:20:55.330049+00:00
-- Confidence: 0.5


-- Obligation: Institutions must document
ALTER TABLE compliance_table
ADD CONSTRAINT chk_fatf_recommendation_10_2a2fa7_fatf_recommendation_10_s007_c000_cl003
CHECK (
    document_status = TRUE
);

-- Trigger for enforcement
CREATE OR REPLACE FUNCTION enforce_fatf_recommendation_10_2a2fa7_fatf_recommendation_10_s007_c000_cl003()
RETURNS TRIGGER AS $$
BEGIN
    IF NOT (document_status = TRUE) THEN
        RAISE EXCEPTION 'Compliance violation: FATF_RECOMMENDATION_10_2A2FA7_FATF_RECOMMENDATION_10_S007_C000_CL003 - Institutions must document';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_fatf_recommendation_10_2a2fa7_fatf_recommendation_10_s007_c000_cl003
BEFORE INSERT OR UPDATE ON compliance_table
FOR EACH ROW
EXECUTE FUNCTION enforce_fatf_recommendation_10_2a2fa7_fatf_recommendation_10_s007_c000_cl003();



COMMENT ON CONSTRAINT chk_fatf_recommendation_10_2a2fa7_fatf_recommendation_10_s007_c000_cl003 ON compliance_table
IS 'AegisLang: Institutions must document the risk assessment that justifies the application of simplified due diligence.';
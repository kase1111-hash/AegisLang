-- Source: Financial institutions should be required to undertake CDD measures wh...
-- Clause ID: FATF_RECOMMENDATION_10_2A2FA7_FATF_RECOMMENDATION_10_S004_C000_CL001
-- Generated: 2026-02-11T18:20:55.020738+00:00
-- Confidence: 0.5


-- Obligation: Financial institutions must be
ALTER TABLE compliance_table
ADD CONSTRAINT chk_fatf_recommendation_10_2a2fa7_fatf_recommendation_10_s004_c000_cl001
CHECK (
    be_status = TRUE
);

-- Trigger for enforcement
CREATE OR REPLACE FUNCTION enforce_fatf_recommendation_10_2a2fa7_fatf_recommendation_10_s004_c000_cl001()
RETURNS TRIGGER AS $$
BEGIN
    IF NOT (be_status = TRUE) THEN
        RAISE EXCEPTION 'Compliance violation: FATF_RECOMMENDATION_10_2A2FA7_FATF_RECOMMENDATION_10_S004_C000_CL001 - Financial institutions must be';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_fatf_recommendation_10_2a2fa7_fatf_recommendation_10_s004_c000_cl001
BEFORE INSERT OR UPDATE ON compliance_table
FOR EACH ROW
EXECUTE FUNCTION enforce_fatf_recommendation_10_2a2fa7_fatf_recommendation_10_s004_c000_cl001();



COMMENT ON CONSTRAINT chk_fatf_recommendation_10_2a2fa7_fatf_recommendation_10_s004_c000_cl001 ON compliance_table
IS 'AegisLang: Financial institutions should be required to undertake CDD measures when establishing business relationships.';
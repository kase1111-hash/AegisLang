-- Source: Institutions must conduct ongoing due diligence on the business relati...
-- Clause ID: FATF_RECOMMENDATION_10_2A2FA7_FATF_RECOMMENDATION_10_S005_C000_CL004
-- Generated: 2026-02-11T18:20:55.206158+00:00
-- Confidence: 0.5


-- Obligation: Institutions must conduct
ALTER TABLE compliance_table
ADD CONSTRAINT chk_fatf_recommendation_10_2a2fa7_fatf_recommendation_10_s005_c000_cl004
CHECK (
    conduct_status = TRUE
);

-- Trigger for enforcement
CREATE OR REPLACE FUNCTION enforce_fatf_recommendation_10_2a2fa7_fatf_recommendation_10_s005_c000_cl004()
RETURNS TRIGGER AS $$
BEGIN
    IF NOT (conduct_status = TRUE) THEN
        RAISE EXCEPTION 'Compliance violation: FATF_RECOMMENDATION_10_2A2FA7_FATF_RECOMMENDATION_10_S005_C000_CL004 - Institutions must conduct';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_fatf_recommendation_10_2a2fa7_fatf_recommendation_10_s005_c000_cl004
BEFORE INSERT OR UPDATE ON compliance_table
FOR EACH ROW
EXECUTE FUNCTION enforce_fatf_recommendation_10_2a2fa7_fatf_recommendation_10_s005_c000_cl004();



COMMENT ON CONSTRAINT chk_fatf_recommendation_10_2a2fa7_fatf_recommendation_10_s005_c000_cl004 ON compliance_table
IS 'AegisLang: Institutions must conduct ongoing due diligence on the business relationship and scrutinize transactions to ensure consistency with the institution''s knowledge of the customer.';
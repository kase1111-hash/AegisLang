-- Source: Banks must verify the identity of each customer using documentary or n...
-- Clause ID: FFIEC_CIP_MANUAL_759301_FFIEC_CIP_MANUAL_S005_C000_CL001
-- Generated: 2026-02-11T18:36:04.365515+00:00
-- Confidence: 0.95


-- Obligation: Banks must verify
ALTER TABLE customers
ADD CONSTRAINT chk_ffiec_cip_manual_759301_ffiec_cip_manual_s005_c000_cl001
CHECK (
    verify_status = TRUE
);

-- Trigger for enforcement
CREATE OR REPLACE FUNCTION enforce_ffiec_cip_manual_759301_ffiec_cip_manual_s005_c000_cl001()
RETURNS TRIGGER AS $$
BEGIN
    IF NOT (verify_status = TRUE) THEN
        RAISE EXCEPTION 'Compliance violation: FFIEC_CIP_MANUAL_759301_FFIEC_CIP_MANUAL_S005_C000_CL001 - Banks must verify';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_ffiec_cip_manual_759301_ffiec_cip_manual_s005_c000_cl001
BEFORE INSERT OR UPDATE ON customers
FOR EACH ROW
EXECUTE FUNCTION enforce_ffiec_cip_manual_759301_ffiec_cip_manual_s005_c000_cl001();



COMMENT ON CONSTRAINT chk_ffiec_cip_manual_759301_ffiec_cip_manual_s005_c000_cl001 ON customers
IS 'AegisLang: Banks must verify the identity of each customer using documentary or non-documentary methods.';
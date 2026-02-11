-- Source: Staff must not open accounts for individuals or entities identified on...
-- Clause ID: FFIEC_CIP_MANUAL_759301_FFIEC_CIP_MANUAL_S007_C000_CL003
-- Generated: 2026-02-11T18:36:04.543441+00:00
-- Confidence: 0.95


-- Prohibition: Staff must not open
ALTER TABLE audit_log
ADD CONSTRAINT chk_ffiec_cip_manual_759301_ffiec_cip_manual_s007_c000_cl003_prohibit
CHECK (
    NOT (NOT open_status)
);



COMMENT ON CONSTRAINT chk_ffiec_cip_manual_759301_ffiec_cip_manual_s007_c000_cl003_prohibit ON audit_log
IS 'AegisLang: Staff must not open accounts for individuals or entities identified on government sanctions lists.';
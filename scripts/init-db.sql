-- =============================================================================
-- AegisLang Database Initialization
-- =============================================================================

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- -----------------------------------------------------------------------------
-- Schema Registry Tables
-- -----------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS schemas (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    schema_id VARCHAR(255) UNIQUE NOT NULL,
    schema_type VARCHAR(50) NOT NULL,
    version VARCHAR(50) NOT NULL DEFAULT '1.0.0',
    tables_json JSONB NOT NULL DEFAULT '[]',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_schemas_schema_id ON schemas(schema_id);

-- -----------------------------------------------------------------------------
-- Documents Table
-- -----------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    doc_id VARCHAR(255) UNIQUE NOT NULL,
    source_file VARCHAR(1024),
    document_type VARCHAR(50),
    metadata_json JSONB NOT NULL DEFAULT '{}',
    content_hash VARCHAR(64),
    status VARCHAR(50) DEFAULT 'ingested',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_documents_doc_id ON documents(doc_id);
CREATE INDEX idx_documents_status ON documents(status);

-- -----------------------------------------------------------------------------
-- Clauses Table
-- -----------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS clauses (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    clause_id VARCHAR(255) UNIQUE NOT NULL,
    doc_id VARCHAR(255) NOT NULL REFERENCES documents(doc_id),
    clause_type VARCHAR(50) NOT NULL,
    source_text TEXT,
    parsed_json JSONB NOT NULL DEFAULT '{}',
    confidence DECIMAL(3,2),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_clauses_clause_id ON clauses(clause_id);
CREATE INDEX idx_clauses_doc_id ON clauses(doc_id);
CREATE INDEX idx_clauses_type ON clauses(clause_type);

-- -----------------------------------------------------------------------------
-- Artifacts Table
-- -----------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS artifacts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    artifact_id VARCHAR(255) UNIQUE NOT NULL,
    clause_id VARCHAR(255) NOT NULL REFERENCES clauses(clause_id),
    format VARCHAR(50) NOT NULL,
    content TEXT NOT NULL,
    file_path VARCHAR(1024),
    syntax_valid BOOLEAN DEFAULT TRUE,
    template_used VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_artifacts_artifact_id ON artifacts(artifact_id);
CREATE INDEX idx_artifacts_clause_id ON artifacts(clause_id);
CREATE INDEX idx_artifacts_format ON artifacts(format);

-- -----------------------------------------------------------------------------
-- Validation Results Table
-- -----------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS validation_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    trace_id VARCHAR(255) UNIQUE NOT NULL,
    clause_id VARCHAR(255) NOT NULL REFERENCES clauses(clause_id),
    artifact_id VARCHAR(255) NOT NULL REFERENCES artifacts(artifact_id),
    validation_status VARCHAR(50) NOT NULL,
    confidence_score DECIMAL(4,3),
    checks_json JSONB NOT NULL DEFAULT '[]',
    review_flags JSONB NOT NULL DEFAULT '[]',
    validated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_validation_trace_id ON validation_results(trace_id);
CREATE INDEX idx_validation_status ON validation_results(validation_status);

-- -----------------------------------------------------------------------------
-- Compliance Audit Log
-- -----------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS compliance_audit_log (
    id BIGSERIAL PRIMARY KEY,
    clause_id VARCHAR(255) NOT NULL,
    violation_type VARCHAR(100) NOT NULL,
    table_name VARCHAR(255),
    record_id VARCHAR(255),
    actor_id VARCHAR(255),
    violation_message TEXT,
    severity VARCHAR(50) DEFAULT 'medium',
    resolved BOOLEAN DEFAULT FALSE,
    resolved_at TIMESTAMP WITH TIME ZONE,
    resolved_by VARCHAR(255),
    resolution_notes TEXT,
    occurred_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    CONSTRAINT chk_severity CHECK (severity IN ('low', 'medium', 'high', 'critical'))
);

CREATE INDEX idx_audit_clause_id ON compliance_audit_log(clause_id);
CREATE INDEX idx_audit_occurred_at ON compliance_audit_log(occurred_at);
CREATE INDEX idx_audit_severity ON compliance_audit_log(severity);
CREATE INDEX idx_audit_resolved ON compliance_audit_log(resolved);

-- -----------------------------------------------------------------------------
-- Jobs Table
-- -----------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS jobs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_id VARCHAR(255) UNIQUE NOT NULL,
    job_type VARCHAR(50) NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    result_json JSONB,
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX idx_jobs_job_id ON jobs(job_id);
CREATE INDEX idx_jobs_status ON jobs(status);

-- -----------------------------------------------------------------------------
-- Views
-- -----------------------------------------------------------------------------

CREATE OR REPLACE VIEW document_summary AS
SELECT
    d.doc_id,
    d.document_type,
    d.status,
    COUNT(DISTINCT c.id) as clause_count,
    COUNT(DISTINCT a.id) as artifact_count,
    AVG(c.confidence) as avg_confidence,
    d.created_at
FROM documents d
LEFT JOIN clauses c ON d.doc_id = c.doc_id
LEFT JOIN artifacts a ON c.clause_id = a.clause_id
GROUP BY d.doc_id, d.document_type, d.status, d.created_at;

CREATE OR REPLACE VIEW validation_summary AS
SELECT
    v.validation_status,
    COUNT(*) as count,
    AVG(v.confidence_score) as avg_confidence
FROM validation_results v
GROUP BY v.validation_status;

-- -----------------------------------------------------------------------------
-- Initial Data
-- -----------------------------------------------------------------------------

-- Insert default schema
INSERT INTO schemas (schema_id, schema_type, version, tables_json)
VALUES (
    'default_kyc_schema',
    'sql',
    '1.0.0',
    '[
        {
            "table_name": "customer",
            "fields": [
                {"field_name": "customer_id", "field_type": "UUID"},
                {"field_name": "identity_verified", "field_type": "BOOLEAN"}
            ]
        }
    ]'::jsonb
)
ON CONFLICT (schema_id) DO NOTHING;

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO aegislang;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO aegislang;

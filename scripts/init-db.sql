-- AegisLang Database Initialization Script
-- This script runs automatically when the PostgreSQL container starts

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- =============================================================================
-- Documents Table
-- =============================================================================
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    doc_id VARCHAR(100) UNIQUE NOT NULL,
    title VARCHAR(500),
    source_type VARCHAR(50) NOT NULL,
    file_path VARCHAR(1000),
    content_hash VARCHAR(64),
    metadata JSONB DEFAULT '{}',
    ingested_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_documents_doc_id ON documents(doc_id);
CREATE INDEX idx_documents_source_type ON documents(source_type);
CREATE INDEX idx_documents_ingested_at ON documents(ingested_at);

-- =============================================================================
-- Clauses Table
-- =============================================================================
CREATE TABLE IF NOT EXISTS clauses (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    clause_id VARCHAR(100) UNIQUE NOT NULL,
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    clause_type VARCHAR(50) NOT NULL,
    source_text TEXT NOT NULL,
    parsed_data JSONB NOT NULL,
    confidence DECIMAL(3,2) DEFAULT 0.00,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_clauses_clause_id ON clauses(clause_id);
CREATE INDEX idx_clauses_document_id ON clauses(document_id);
CREATE INDEX idx_clauses_type ON clauses(clause_type);
CREATE INDEX idx_clauses_confidence ON clauses(confidence);

-- =============================================================================
-- Entity Mappings Table
-- =============================================================================
CREATE TABLE IF NOT EXISTS entity_mappings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    clause_id UUID REFERENCES clauses(id) ON DELETE CASCADE,
    source_entity VARCHAR(500) NOT NULL,
    source_role VARCHAR(50) NOT NULL,
    target_path VARCHAR(500),
    target_type VARCHAR(100),
    confidence DECIMAL(3,2) DEFAULT 0.00,
    mapping_method VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_entity_mappings_clause_id ON entity_mappings(clause_id);
CREATE INDEX idx_entity_mappings_source_entity ON entity_mappings(source_entity);

-- =============================================================================
-- Compiled Artifacts Table
-- =============================================================================
CREATE TABLE IF NOT EXISTS artifacts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    artifact_id VARCHAR(200) UNIQUE NOT NULL,
    clause_id UUID REFERENCES clauses(id) ON DELETE CASCADE,
    format VARCHAR(50) NOT NULL,
    content TEXT NOT NULL,
    file_path VARCHAR(500),
    syntax_valid BOOLEAN DEFAULT TRUE,
    template_used VARCHAR(100),
    warnings JSONB DEFAULT '[]',
    compiled_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_artifacts_artifact_id ON artifacts(artifact_id);
CREATE INDEX idx_artifacts_clause_id ON artifacts(clause_id);
CREATE INDEX idx_artifacts_format ON artifacts(format);

-- =============================================================================
-- Validation Traces Table
-- =============================================================================
CREATE TABLE IF NOT EXISTS validation_traces (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    trace_id VARCHAR(100) UNIQUE NOT NULL,
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    validation_status VARCHAR(50) NOT NULL,
    total_clauses INTEGER DEFAULT 0,
    valid_clauses INTEGER DEFAULT 0,
    warnings_count INTEGER DEFAULT 0,
    errors_count INTEGER DEFAULT 0,
    trace_data JSONB NOT NULL,
    validated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_validation_traces_trace_id ON validation_traces(trace_id);
CREATE INDEX idx_validation_traces_document_id ON validation_traces(document_id);
CREATE INDEX idx_validation_traces_status ON validation_traces(validation_status);

-- =============================================================================
-- Audit Log Table
-- =============================================================================
CREATE TABLE IF NOT EXISTS audit_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_type VARCHAR(100) NOT NULL,
    entity_type VARCHAR(50) NOT NULL,
    entity_id VARCHAR(100),
    actor VARCHAR(100),
    action VARCHAR(100) NOT NULL,
    details JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_audit_log_event_type ON audit_log(event_type);
CREATE INDEX idx_audit_log_entity ON audit_log(entity_type, entity_id);
CREATE INDEX idx_audit_log_created_at ON audit_log(created_at);

-- =============================================================================
-- Schema Definitions Table (for L3 mapping)
-- =============================================================================
CREATE TABLE IF NOT EXISTS schema_definitions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    schema_name VARCHAR(200) UNIQUE NOT NULL,
    schema_type VARCHAR(50) NOT NULL,
    definition JSONB NOT NULL,
    embeddings_stored BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_schema_definitions_name ON schema_definitions(schema_name);
CREATE INDEX idx_schema_definitions_type ON schema_definitions(schema_type);

-- =============================================================================
-- Functions
-- =============================================================================

-- Update timestamp trigger
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply update trigger to tables
CREATE TRIGGER update_documents_updated_at
    BEFORE UPDATE ON documents
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_clauses_updated_at
    BEFORE UPDATE ON clauses
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_schema_definitions_updated_at
    BEFORE UPDATE ON schema_definitions
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- =============================================================================
-- Initial Data
-- =============================================================================

-- Insert default schema definitions for common compliance domains
INSERT INTO schema_definitions (schema_name, schema_type, definition) VALUES
('gdpr.data_subject', 'entity', '{"description": "Individual whose personal data is processed", "attributes": ["name", "email", "consent_status"]}'),
('gdpr.data_controller', 'entity', '{"description": "Entity determining purposes of data processing", "attributes": ["organization_name", "dpo_contact"]}'),
('gdpr.personal_data', 'object', '{"description": "Any information relating to an identified person", "categories": ["basic", "sensitive", "biometric"]}'),
('hipaa.covered_entity', 'entity', '{"description": "Healthcare provider, plan, or clearinghouse", "attributes": ["entity_type", "npi_number"]}'),
('hipaa.phi', 'object', '{"description": "Protected Health Information", "identifiers": ["name", "ssn", "medical_record"]}'),
('sox.public_company', 'entity', '{"description": "Company subject to SOX requirements", "attributes": ["ticker_symbol", "sec_filing_status"]}'),
('pci.cardholder_data', 'object', '{"description": "Credit card and related information", "elements": ["pan", "cvv", "expiry"]}')
ON CONFLICT (schema_name) DO NOTHING;

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO aegis;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO aegis;
